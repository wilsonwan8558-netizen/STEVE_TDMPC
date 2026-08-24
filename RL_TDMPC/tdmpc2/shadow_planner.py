"""Side-effect-free primary-Safety-risk shadow MPPI planning.

This module deliberately sits outside :mod:`tdmpc2.agent`.  The production
TD-MPC2 planner never imports it and therefore has no Safety-Head inference
overhead.  A shadow run reproduces the production planner from an immutable
pre-call snapshot and an explicit schedule containing every exogenous PyTorch
random draw used by latent MPPI.  It may score candidates with

``task_value - safety_weight * primary_tracking_trajectory_risk``

but it never writes ``agent.previous_mean`` and never selects the action sent
to the environment.

Only the configured primary tracking-risk channel is allowed into the shadow
score.  All channels are decoded together to preserve the Safety-Head
interface, but non-primary channels are never read by the scoring path.  The
historical ``translation`` public function names are retained for compatibility.
"""

from __future__ import annotations

import hashlib
import math
import operator
import time
from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from safety_schema import validate_safety_schema

from .networks import two_hot_inv


SUPPORTED_AGGREGATIONS = ("max", "discounted_sum")


@dataclass(frozen=True)
class PlannerSnapshot:
    """Immutable inputs and warm-start state for one planning call."""

    latent_single: torch.Tensor
    previous_mean: torch.Tensor
    first_step: bool


@dataclass(frozen=True)
class PlannerIterationNoise:
    """Random draws consumed by one production MPPI iteration."""

    proposal: torch.Tensor
    terminal_policy: torch.Tensor
    q_indices: torch.Tensor


@dataclass(frozen=True)
class PlannerNoiseSchedule:
    """All randomness needed to replay one evaluation-mode planning call."""

    policy: Tuple[torch.Tensor, ...]
    iterations: Tuple[PlannerIterationNoise, ...]
    diagnostic_terminal_policy: torch.Tensor
    diagnostic_q_indices: torch.Tensor
    fingerprint: str
    core_end_torch_state: torch.Tensor


@dataclass
class ShadowPlanTrace:
    """Numerical trace for one independent task/Safety MPPI run."""

    safety_weight: float
    aggregation: str
    aggregation_discount: float
    planner_cap: float
    action: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    final_actions: torch.Tensor
    final_task_values: torch.Tensor
    final_uncapped_risks: torch.Tensor
    final_capped_risks: torch.Tensor
    final_scores: torch.Tensor
    final_elite_indices: torch.Tensor
    final_elite_weights: torch.Tensor
    selected_task_value: torch.Tensor
    selected_uncapped_risk: torch.Tensor
    selected_capped_risk: torch.Tensor
    selected_step_uncapped_risk: torch.Tensor
    selected_step_capped_risk: torch.Tensor
    iteration_means: Tuple[torch.Tensor, ...]
    iteration_stds: Tuple[torch.Tensor, ...]
    iteration_elites: Tuple[torch.Tensor, ...]
    planner_seconds: float
    safety_inference_seconds: float
    core_planner_seconds: float
    candidate_safety_inference_seconds: float
    selected_diagnostic_seconds: float
    noise_fingerprint: str
    input_fingerprint: str


def _clone_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().clone()


def _synchronize(device: torch.device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))


def _strict_safety_index(value: Any, *, source: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{source} must be an integer, not bool")
    try:
        parsed = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{source} must be an integer") from exc
    return int(parsed)


def _resolve_primary_safety_channel(agent: Any) -> Tuple[int, str, int]:
    """Resolve one strictly validated primary channel without fixed indices."""

    model = getattr(agent, "model", None)
    if model is None:
        raise TypeError("Shadow planner agent must expose model")
    if not hasattr(model, "safety_cost_names") or not hasattr(
        model, "safety_dim"
    ):
        raise ValueError(
            "Shadow planner model must expose safety_cost_names and safety_dim"
        )
    names = validate_safety_schema(
        model.safety_cost_names,
        model.safety_dim,
        source="model.safety_cost_names",
    )
    safety_dim = _strict_safety_index(
        model.safety_dim,
        source="model.safety_dim",
    )
    if not hasattr(agent, "safety_cost_names") or not hasattr(
        agent, "safety_dim"
    ):
        raise ValueError(
            "Shadow planner agent must expose safety_cost_names and safety_dim"
        )
    agent_names = validate_safety_schema(
        agent.safety_cost_names,
        agent.safety_dim,
        source="agent.safety_cost_names",
    )
    agent_dim = _strict_safety_index(
        agent.safety_dim,
        source="agent.safety_dim",
    )
    if agent_names != names or agent_dim != safety_dim:
        raise ValueError(
            "Agent and model Safety schemas must exactly match: "
            f"{agent_names!r}/{agent_dim} != {names!r}/{safety_dim}"
        )

    channel_values = []
    index_values = []
    for owner_name, owner in (("agent", agent), ("model", model)):
        if hasattr(owner, "safety_primary_risk_channel"):
            channel = getattr(owner, "safety_primary_risk_channel")
            if not isinstance(channel, str):
                raise TypeError(
                    f"{owner_name}.safety_primary_risk_channel must be a string"
                )
            if not channel.strip():
                raise ValueError(
                    f"{owner_name}.safety_primary_risk_channel must be non-empty"
                )
            channel_values.append((owner_name, channel))
        if hasattr(owner, "safety_primary_risk_index"):
            index = _strict_safety_index(
                getattr(owner, "safety_primary_risk_index"),
                source=f"{owner_name}.safety_primary_risk_index",
            )
            if not 0 <= index < safety_dim:
                raise ValueError(
                    f"{owner_name}.safety_primary_risk_index {index} is outside "
                    f"[0, {safety_dim})"
                )
            index_values.append((owner_name, index))

    distinct_channels = {value for _, value in channel_values}
    if len(distinct_channels) > 1:
        raise ValueError(
            "Agent and model disagree on safety_primary_risk_channel: "
            f"{channel_values!r}"
        )
    distinct_indices = {value for _, value in index_values}
    if len(distinct_indices) > 1:
        raise ValueError(
            "Agent and model disagree on safety_primary_risk_index: "
            f"{index_values!r}"
        )

    if distinct_channels:
        primary_name = next(iter(distinct_channels))
        if primary_name not in names:
            raise ValueError(
                "safety_primary_risk_channel must name one configured Safety "
                f"channel; got {primary_name!r} for {names!r}"
            )
        name_index = names.index(primary_name)
    else:
        primary_name = None
        name_index = None

    if distinct_indices:
        primary_index = next(iter(distinct_indices))
        indexed_name = names[primary_index]
        if primary_name is not None and primary_index != name_index:
            raise ValueError(
                "safety_primary_risk_channel and safety_primary_risk_index "
                f"disagree: {primary_name!r} is index {name_index}, not "
                f"{primary_index}"
            )
        primary_name = indexed_name
    elif primary_name is not None:
        primary_index = int(name_index)
    else:
        raise ValueError(
            "Shadow planning requires safety_primary_risk_channel or "
            "safety_primary_risk_index"
        )

    return primary_index, primary_name, safety_dim


def _validate_agent_for_shadow(agent: Any) -> None:
    if agent.model.training:
        raise RuntimeError(
            "Shadow planning requires model.eval(); stochastic dropout would "
            "invalidate the explicit-noise replay"
        )
    if any(
        isinstance(module, torch.nn.Dropout) and module.training
        for module in agent.model.modules()
    ):
        raise RuntimeError(
            "Shadow planning requires every Dropout module to be in eval mode"
        )
    _resolve_primary_safety_channel(agent)
    if int(agent.config["num_samples"]) <= 0:
        raise ValueError("num_samples must be positive")
    if int(agent.config["iterations"]) <= 0:
        raise ValueError("iterations must be positive")
    num_elites = int(agent.config["num_elites"])
    if not 0 < num_elites <= int(agent.config["num_samples"]):
        raise ValueError("num_elites must be in [1, num_samples]")


@torch.no_grad()
def capture_planner_snapshot(
    agent: Any,
    observation: Any,
    *,
    first_step: bool,
) -> PlannerSnapshot:
    """Encode an observation and clone the planner state without mutation."""

    _validate_agent_for_shadow(agent)
    if isinstance(observation, torch.Tensor):
        tensor = observation.to(device=agent.device, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
    else:
        array = np.asarray(observation, dtype=np.float32)
        tensor = torch.as_tensor(array, device=agent.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
    if tuple(tensor.shape) != (1, int(agent.observation_dim)):
        raise ValueError(
            "Planner observation must have shape "
            f"(1, {agent.observation_dim}), got {tuple(tensor.shape)}"
        )
    if not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError("Planner observation contains NaN or infinity")
    latent = agent.model.encode(tensor)
    if tuple(latent.shape) != (1, int(agent.model.latent_dim)):
        raise RuntimeError(f"Unexpected encoded latent shape {tuple(latent.shape)}")
    return PlannerSnapshot(
        latent_single=_clone_tensor(latent),
        previous_mean=_clone_tensor(agent.previous_mean),
        first_step=bool(first_step),
    )


def capture_device_torch_rng_state(device: torch.device) -> torch.Tensor:
    """Clone the RNG state used by tensor draws on ``device``."""

    device = torch.device(device)
    if device.type == "cuda":
        return torch.cuda.get_rng_state(device).clone()
    if device.type != "cpu":
        raise ValueError(f"Unsupported planner device {device}")
    return torch.get_rng_state().clone()


def _set_device_torch_rng_state(
    device: torch.device,
    state: torch.Tensor,
) -> None:
    if device.type == "cuda":
        torch.cuda.set_rng_state(state, device)
    else:
        torch.set_rng_state(state)


def _hash_schedule(
    policy: Sequence[torch.Tensor],
    iterations: Sequence[PlannerIterationNoise],
    diagnostic_terminal_policy: torch.Tensor,
    diagnostic_q_indices: torch.Tensor,
) -> str:
    digest = hashlib.sha256()
    tensors = list(policy)
    for iteration in iterations:
        tensors.extend(
            (iteration.proposal, iteration.terminal_policy, iteration.q_indices)
        )
    tensors.extend((diagnostic_terminal_policy, diagnostic_q_indices))
    for tensor in tensors:
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def _snapshot_fingerprint(agent: Any, snapshot: PlannerSnapshot) -> str:
    digest = hashlib.sha256()
    for tensor in (snapshot.latent_single, snapshot.previous_mean):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.numpy().tobytes())
    digest.update(str(bool(snapshot.first_step)).encode("ascii"))
    planner_spec = (
        int(agent.horizon),
        int(agent.action_dim),
        int(agent.config["num_samples"]),
        int(agent.config["num_pi_trajs"]),
        int(agent.config["num_elites"]),
        int(agent.config["iterations"]),
        float(agent.config["temperature"]),
        float(agent.config["min_std"]),
        float(agent.config["max_std"]),
        float(agent.discount),
        bool(agent.config.get("episodic", True)),
    )
    digest.update(repr(planner_spec).encode("ascii"))
    return digest.hexdigest()


def _generate_noise_schedule(agent: Any) -> PlannerNoiseSchedule:
    """Generate a schedule using the currently installed device RNG state."""

    device = torch.device(agent.device)
    dtype = agent.previous_mean.dtype
    horizon = int(agent.horizon)
    samples = int(agent.config["num_samples"])
    policy_count = min(int(agent.config["num_pi_trajs"]), samples)
    action_dim = int(agent.action_dim)
    q_count = min(2, int(agent.model.num_q))

    policy = tuple(
        torch.randn(
            policy_count,
            action_dim,
            device=device,
            dtype=dtype,
        )
        for _ in range(horizon)
    )
    iterations = []
    for _ in range(int(agent.config["iterations"])):
        proposal = torch.randn(
            horizon,
            samples - policy_count,
            action_dim,
            device=device,
            dtype=dtype,
        )
        terminal_policy = torch.randn(
            samples,
            action_dim,
            device=device,
            dtype=dtype,
        )
        q_indices = torch.randperm(
            int(agent.model.num_q),
            device=device,
        )[:q_count]
        iterations.append(
            PlannerIterationNoise(
                proposal=proposal,
                terminal_policy=terminal_policy,
                q_indices=q_indices,
            )
        )

    # This is exactly the RNG point at which production evaluation-mode
    # _plan() returns.  Extra noise below is diagnostic-only and is generated
    # after that state has been captured.
    core_end_state = capture_device_torch_rng_state(device)
    policy_tuple = tuple(_clone_tensor(value) for value in policy)
    iteration_tuple = tuple(
        PlannerIterationNoise(
            proposal=_clone_tensor(value.proposal),
            terminal_policy=_clone_tensor(value.terminal_policy),
            q_indices=_clone_tensor(value.q_indices),
        )
        for value in iterations
    )
    core_tensors = list(policy_tuple)
    for value in iteration_tuple:
        core_tensors.extend(
            (value.proposal, value.terminal_policy, value.q_indices)
        )
    core_digest = hashlib.sha256()
    for value in core_tensors:
        contiguous = value.detach().cpu().contiguous()
        core_digest.update(contiguous.numpy().tobytes())
    diagnostic_seed = int.from_bytes(
        core_digest.digest()[:8],
        byteorder="little",
        signed=False,
    ) % (2**63 - 1)
    diagnostic_generator = torch.Generator(device=device)
    diagnostic_generator.manual_seed(diagnostic_seed)
    diagnostic_terminal_policy = torch.randn(
        1,
        action_dim,
        device=device,
        dtype=dtype,
        generator=diagnostic_generator,
    )
    diagnostic_q_indices = torch.randperm(
        int(agent.model.num_q),
        device=device,
        generator=diagnostic_generator,
    )[:q_count]
    diagnostic_terminal_policy = _clone_tensor(diagnostic_terminal_policy)
    diagnostic_q_indices = _clone_tensor(diagnostic_q_indices)
    return PlannerNoiseSchedule(
        policy=policy_tuple,
        iterations=iteration_tuple,
        diagnostic_terminal_policy=diagnostic_terminal_policy,
        diagnostic_q_indices=diagnostic_q_indices,
        fingerprint=_hash_schedule(
            policy_tuple,
            iteration_tuple,
            diagnostic_terminal_policy,
            diagnostic_q_indices,
        ),
        core_end_torch_state=core_end_state,
    )


def make_planner_noise_schedule(
    agent: Any,
    *,
    initial_torch_state: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,
) -> PlannerNoiseSchedule:
    """Create explicit production-order noise without advancing global RNG.

    Exactly one of ``initial_torch_state`` and ``seed`` may be supplied.  With
    an initial state, the core portion reproduces a production ``_plan`` call
    starting from that state.  With a seed, a private seeded state is used.
    """

    _validate_agent_for_shadow(agent)
    if initial_torch_state is not None and seed is not None:
        raise ValueError("Pass initial_torch_state or seed, not both")
    device = torch.device(agent.device)
    outside_state = capture_device_torch_rng_state(device)
    try:
        if initial_torch_state is not None:
            _set_device_torch_rng_state(device, initial_torch_state.clone())
        elif seed is not None:
            if device.type == "cuda":
                generator = torch.Generator(device=device)
                generator.manual_seed(int(seed))
                _set_device_torch_rng_state(device, generator.get_state())
            else:
                generator = torch.Generator()
                generator.manual_seed(int(seed))
                _set_device_torch_rng_state(device, generator.get_state())
        return _generate_noise_schedule(agent)
    finally:
        _set_device_torch_rng_state(device, outside_state)


def _policy_with_noise(
    model: torch.nn.Module,
    latent: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    """Reproduce ``WorldModel.pi`` while injecting its Gaussian draw."""

    mean, raw_log_std = model.policy(latent).chunk(2, dim=-1)
    if tuple(noise.shape) != tuple(mean.shape):
        raise ValueError(
            f"Policy noise shape {tuple(noise.shape)} != {tuple(mean.shape)}"
        )
    log_std = model.log_std_min + 0.5 * model.log_std_range * (
        torch.tanh(raw_log_std) + 1.0
    )
    return torch.tanh(mean + noise * log_std.exp())


def _q_with_indices(
    model: torch.nn.Module,
    latent: torch.Tensor,
    action: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Reproduce ``WorldModel.q(..., reduction='avg')`` without RNG."""

    logits = model.q_values(latent, action)
    expected = min(2, int(model.num_q))
    if tuple(indices.shape) != (expected,):
        raise ValueError(f"Q index shape must be {(expected,)}, got {tuple(indices.shape)}")
    if bool(torch.any(indices < 0)) or bool(torch.any(indices >= int(model.num_q))):
        raise ValueError("Q index schedule is out of range")
    values = two_hot_inv(
        logits[indices],
        model.num_bins,
        model.value_min,
        model.value_max,
    )
    return values.mean(dim=0)


@torch.no_grad()
def estimate_task_value_explicit(
    agent: Any,
    latent: torch.Tensor,
    actions: torch.Tensor,
    terminal_policy_noise: torch.Tensor,
    q_indices: torch.Tensor,
) -> torch.Tensor:
    """Production-equivalent latent return with no random draws."""

    batch = int(latent.shape[0])
    expected_actions = (
        int(agent.horizon),
        batch,
        int(agent.action_dim),
    )
    if tuple(actions.shape) != expected_actions:
        raise ValueError(
            f"Actions must have shape {expected_actions}, got {tuple(actions.shape)}"
        )
    returns = torch.zeros(
        batch,
        1,
        device=latent.device,
        dtype=latent.dtype,
    )
    discount = 1.0
    termination = torch.zeros_like(returns)
    rollout_latent = latent
    for step in range(int(agent.horizon)):
        reward_logits = agent.model.reward(rollout_latent, actions[step])
        reward = two_hot_inv(
            reward_logits,
            agent.model.num_bins,
            agent.model.value_min,
            agent.model.value_max,
        )
        rollout_latent = agent.model.next(rollout_latent, actions[step])
        returns.add_(discount * (1.0 - termination) * reward)
        discount *= float(agent.discount)
        if bool(agent.config.get("episodic", True)):
            termination = torch.clamp(
                termination
                + (agent.model.termination(rollout_latent) > 0.5).float(),
                max=1.0,
            )
    final_action = _policy_with_noise(
        agent.model,
        rollout_latent,
        terminal_policy_noise,
    )
    final_q = _q_with_indices(
        agent.model,
        rollout_latent,
        final_action,
        q_indices,
    )
    values = returns + discount * (1.0 - termination) * final_q
    # This is the exact sanitization used by production _plan.
    return torch.nan_to_num(values, nan=0.0)


def _validate_risk_options(
    aggregation: str,
    aggregation_discount: float,
    planner_cap: float,
) -> Tuple[str, float, float]:
    aggregation = str(aggregation)
    if aggregation not in SUPPORTED_AGGREGATIONS:
        raise ValueError(
            f"aggregation must be one of {SUPPORTED_AGGREGATIONS}, got {aggregation!r}"
        )
    discount = float(aggregation_discount)
    if not math.isfinite(discount) or not 0.0 <= discount <= 1.0:
        raise ValueError("aggregation_discount must be finite and in [0, 1]")
    cap = float(planner_cap)
    if not math.isfinite(cap) or cap <= 0.0:
        raise ValueError("planner_cap must be finite and positive")
    return aggregation, discount, cap


def _aggregate_risk(
    per_step: torch.Tensor,
    *,
    aggregation: str,
    discount: float,
) -> torch.Tensor:
    if aggregation == "max":
        return per_step.max(dim=0).values
    weights = per_step.new_tensor(
        [discount ** step for step in range(per_step.shape[0])]
    ).unsqueeze(1)
    return (weights * per_step).sum(dim=0)


@torch.no_grad()
def estimate_translation_risk(
    agent: Any,
    latent: torch.Tensor,
    actions: torch.Tensor,
    *,
    aggregation: str,
    aggregation_discount: float,
    planner_cap: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Decode and aggregate the configured primary tracking-risk channel.

    The historical public name is retained for downstream compatibility.
    """

    aggregation, discount, cap = _validate_risk_options(
        aggregation,
        aggregation_discount,
        planner_cap,
    )
    primary_index, primary_name, safety_dim = _resolve_primary_safety_channel(
        agent
    )
    batch = int(latent.shape[0])
    expected_actions = (
        int(agent.horizon),
        batch,
        int(agent.action_dim),
    )
    if tuple(actions.shape) != expected_actions:
        raise ValueError(
            f"Actions must have shape {expected_actions}, got {tuple(actions.shape)}"
        )
    values = []
    rollout_latent = latent
    _synchronize(latent.device)
    started = time.perf_counter()
    for step in range(int(agent.horizon)):
        transformed = agent.model.safety_transformed(
            rollout_latent,
            actions[step],
        )
        expected = (batch, safety_dim)
        if tuple(transformed.shape) != expected:
            raise RuntimeError(
                f"Safety transformed output must be {expected}, got "
                f"{tuple(transformed.shape)}"
            )
        primary_transformed = transformed[:, primary_index]
        if not bool(torch.isfinite(primary_transformed).all()):
            raise FloatingPointError(
                "Primary Safety transformed prediction contains NaN or infinity "
                f"for channel {primary_name!r} (index {primary_index})"
            )
        decoded = agent.model.decode_safety_transformed(transformed)
        if tuple(decoded.shape) != expected:
            raise RuntimeError(
                f"Safety decoded output must be {expected}, got {tuple(decoded.shape)}"
            )
        primary_decoded = decoded[:, primary_index]
        if not bool(torch.isfinite(primary_decoded).all()):
            raise FloatingPointError(
                "Primary Safety decoded prediction is non-finite for channel "
                f"{primary_name!r} (index {primary_index})"
            )
        if bool(torch.any(primary_decoded < 0)):
            raise ValueError(
                "Primary Safety decoded prediction must be nonnegative for "
                f"channel {primary_name!r} (index {primary_index})"
            )
        # Scoring reads only the configured primary channel. Other columns are
        # intentionally neither indexed nor reduced anywhere in this path.
        values.append(primary_decoded)
        rollout_latent = agent.model.next(rollout_latent, actions[step])
    _synchronize(latent.device)
    elapsed = time.perf_counter() - started
    per_step_uncapped = torch.stack(values, dim=0)
    per_step_capped = per_step_uncapped.clamp(max=cap)
    uncapped = _aggregate_risk(
        per_step_uncapped,
        aggregation=aggregation,
        discount=discount,
    )
    capped = _aggregate_risk(
        per_step_capped,
        aggregation=aggregation,
        discount=discount,
    )
    for result, name in (
        (per_step_uncapped, "per-step uncapped risk"),
        (per_step_capped, "per-step capped risk"),
        (uncapped, "uncapped trajectory risk"),
        (capped, "capped trajectory risk"),
    ):
        if not bool(torch.isfinite(result).all()) or bool(torch.any(result < 0)):
            raise FloatingPointError(f"{name} is invalid")
    return per_step_uncapped, per_step_capped, uncapped, capped, elapsed


def _validate_schedule(agent: Any, schedule: PlannerNoiseSchedule) -> None:
    horizon = int(agent.horizon)
    samples = int(agent.config["num_samples"])
    policy_count = min(int(agent.config["num_pi_trajs"]), samples)
    action_dim = int(agent.action_dim)
    iterations = int(agent.config["iterations"])
    q_count = min(2, int(agent.model.num_q))
    if len(schedule.policy) != horizon:
        raise ValueError("Policy-noise schedule has wrong horizon")
    for value in schedule.policy:
        if tuple(value.shape) != (policy_count, action_dim):
            raise ValueError("Policy-noise schedule has wrong shape")
    if len(schedule.iterations) != iterations:
        raise ValueError("Iteration-noise schedule has wrong length")
    for value in schedule.iterations:
        if tuple(value.proposal.shape) != (
            horizon,
            samples - policy_count,
            action_dim,
        ):
            raise ValueError("Proposal-noise schedule has wrong shape")
        if tuple(value.terminal_policy.shape) != (samples, action_dim):
            raise ValueError("Terminal-policy schedule has wrong shape")
        if tuple(value.q_indices.shape) != (q_count,):
            raise ValueError("Q-index schedule has wrong shape")
    if tuple(schedule.diagnostic_terminal_policy.shape) != (1, action_dim):
        raise ValueError("Diagnostic terminal-policy noise has wrong shape")
    if tuple(schedule.diagnostic_q_indices.shape) != (q_count,):
        raise ValueError("Diagnostic Q-index schedule has wrong shape")
    actual_fingerprint = _hash_schedule(
        schedule.policy,
        schedule.iterations,
        schedule.diagnostic_terminal_policy,
        schedule.diagnostic_q_indices,
    )
    if actual_fingerprint != schedule.fingerprint:
        raise ValueError(
            "Planner-noise schedule tensors were modified after construction"
        )


@torch.no_grad()
def run_translation_shadow_plan(
    agent: Any,
    snapshot: PlannerSnapshot,
    schedule: PlannerNoiseSchedule,
    *,
    safety_weight: float,
    aggregation: str,
    aggregation_discount: float = 1.0,
    planner_cap: float = 1.0,
) -> ShadowPlanTrace:
    """Run one independent shadow MPPI optimization without side effects."""

    _validate_agent_for_shadow(agent)
    _validate_schedule(agent, schedule)
    aggregation, aggregation_discount, planner_cap = _validate_risk_options(
        aggregation,
        aggregation_discount,
        planner_cap,
    )
    safety_weight = float(safety_weight)
    if not math.isfinite(safety_weight) or safety_weight < 0.0:
        raise ValueError("safety_weight must be finite and nonnegative")
    expected_previous = (int(agent.horizon), int(agent.action_dim))
    if tuple(snapshot.previous_mean.shape) != expected_previous:
        raise ValueError(
            f"Snapshot previous_mean must be {expected_previous}, got "
            f"{tuple(snapshot.previous_mean.shape)}"
        )

    _synchronize(agent.device)
    started = time.perf_counter()
    latent_single = snapshot.latent_single
    samples = int(agent.config["num_samples"])
    policy_count = min(int(agent.config["num_pi_trajs"]), samples)
    elites = int(agent.config["num_elites"])
    horizon = int(agent.horizon)
    action_dim = int(agent.action_dim)

    policy_actions: Optional[torch.Tensor] = None
    if policy_count > 0:
        policy_actions = torch.empty(
            horizon,
            policy_count,
            action_dim,
            device=agent.device,
            dtype=latent_single.dtype,
        )
        policy_latent = latent_single.repeat(policy_count, 1)
        for step in range(horizon):
            policy_actions[step].copy_(
                _policy_with_noise(
                    agent.model,
                    policy_latent,
                    schedule.policy[step],
                )
            )
            policy_latent = agent.model.next(
                policy_latent,
                policy_actions[step],
            )

    latent = latent_single.repeat(samples, 1)
    mean = torch.zeros(
        horizon,
        action_dim,
        device=agent.device,
        dtype=latent_single.dtype,
    )
    std = torch.full_like(mean, float(agent.config["max_std"]))
    if not snapshot.first_step:
        mean[:-1].copy_(snapshot.previous_mean[1:])
    actions = torch.empty(
        horizon,
        samples,
        action_dim,
        device=agent.device,
        dtype=latent_single.dtype,
    )
    if policy_actions is not None:
        actions[:, :policy_count].copy_(policy_actions)

    iteration_means = []
    iteration_stds = []
    iteration_elites = []
    safety_seconds = 0.0
    final_task = final_uncapped = final_capped = final_scores = None
    final_elite_indices = final_elite_weights = None
    for iteration_noise in schedule.iterations:
        random_actions = (
            mean.unsqueeze(1)
            + std.unsqueeze(1) * iteration_noise.proposal
        )
        actions[:, policy_count:].copy_(random_actions.clamp(-1.0, 1.0))
        task_values = estimate_task_value_explicit(
            agent,
            latent,
            actions,
            iteration_noise.terminal_policy,
            iteration_noise.q_indices,
        )
        (
            _,
            _,
            uncapped_risk,
            capped_risk,
            risk_seconds,
        ) = estimate_translation_risk(
            agent,
            latent,
            actions,
            aggregation=aggregation,
            aggregation_discount=aggregation_discount,
            planner_cap=planner_cap,
        )
        safety_seconds += risk_seconds
        if safety_weight == 0.0:
            # Preserve the exact baseline tensor path for identity testing.
            planner_scores = task_values
        else:
            planner_scores = task_values - safety_weight * capped_risk.unsqueeze(-1)
        if not bool(torch.isfinite(planner_scores).all()):
            raise FloatingPointError("Planner scores contain NaN or infinity")
        elite_indices = torch.topk(
            planner_scores.squeeze(-1),
            elites,
        ).indices
        elite_values = planner_scores[elite_indices]
        elite_actions = actions[:, elite_indices]
        maximum = elite_values.max(dim=0).values
        elite_weights = torch.exp(
            float(agent.config["temperature"]) * (elite_values - maximum)
        )
        elite_weights = elite_weights / (
            elite_weights.sum(dim=0, keepdim=True) + 1e-9
        )
        broadcast_weights = elite_weights.unsqueeze(0)
        mean = (broadcast_weights * elite_actions).sum(dim=1) / (
            broadcast_weights.sum(dim=1) + 1e-9
        )
        variance = (
            broadcast_weights
            * (elite_actions - mean.unsqueeze(1)).pow(2)
        ).sum(dim=1) / (broadcast_weights.sum(dim=1) + 1e-9)
        std = variance.sqrt().clamp(
            float(agent.config["min_std"]),
            float(agent.config["max_std"]),
        )
        iteration_means.append(_clone_tensor(mean))
        iteration_stds.append(_clone_tensor(std))
        iteration_elites.append(_clone_tensor(elite_indices))
        final_task = task_values
        final_uncapped = uncapped_risk
        final_capped = capped_risk
        final_scores = planner_scores
        final_elite_indices = elite_indices
        final_elite_weights = elite_weights

    assert final_task is not None
    assert final_uncapped is not None and final_capped is not None
    assert final_scores is not None
    assert final_elite_indices is not None and final_elite_weights is not None
    _synchronize(agent.device)
    core_elapsed = time.perf_counter() - started
    candidate_safety_seconds = safety_seconds
    _synchronize(agent.device)
    selected_started = time.perf_counter()
    selected_actions = mean.unsqueeze(1)
    selected_task = estimate_task_value_explicit(
        agent,
        latent_single,
        selected_actions,
        schedule.diagnostic_terminal_policy,
        schedule.diagnostic_q_indices,
    )
    (
        selected_step_uncapped,
        selected_step_capped,
        selected_uncapped,
        selected_capped,
        selected_risk_seconds,
    ) = estimate_translation_risk(
        agent,
        latent_single,
        selected_actions,
        aggregation=aggregation,
        aggregation_discount=aggregation_discount,
        planner_cap=planner_cap,
    )
    safety_seconds += selected_risk_seconds
    _synchronize(agent.device)
    selected_diagnostic_seconds = time.perf_counter() - selected_started
    elapsed = time.perf_counter() - started
    return ShadowPlanTrace(
        safety_weight=safety_weight,
        aggregation=aggregation,
        aggregation_discount=aggregation_discount,
        planner_cap=planner_cap,
        action=_clone_tensor(mean[0].clamp(-1.0, 1.0)),
        mean=_clone_tensor(mean),
        std=_clone_tensor(std),
        final_actions=_clone_tensor(actions),
        final_task_values=_clone_tensor(final_task.squeeze(-1)),
        final_uncapped_risks=_clone_tensor(final_uncapped),
        final_capped_risks=_clone_tensor(final_capped),
        final_scores=_clone_tensor(final_scores.squeeze(-1)),
        final_elite_indices=_clone_tensor(final_elite_indices),
        final_elite_weights=_clone_tensor(final_elite_weights.squeeze(-1)),
        selected_task_value=_clone_tensor(selected_task.squeeze()),
        selected_uncapped_risk=_clone_tensor(selected_uncapped.squeeze()),
        selected_capped_risk=_clone_tensor(selected_capped.squeeze()),
        selected_step_uncapped_risk=_clone_tensor(
            selected_step_uncapped.squeeze(-1)
        ),
        selected_step_capped_risk=_clone_tensor(
            selected_step_capped.squeeze(-1)
        ),
        iteration_means=tuple(iteration_means),
        iteration_stds=tuple(iteration_stds),
        iteration_elites=tuple(iteration_elites),
        planner_seconds=float(elapsed),
        safety_inference_seconds=float(safety_seconds),
        core_planner_seconds=float(core_elapsed),
        candidate_safety_inference_seconds=float(candidate_safety_seconds),
        selected_diagnostic_seconds=float(selected_diagnostic_seconds),
        noise_fingerprint=schedule.fingerprint,
        input_fingerprint=_snapshot_fingerprint(agent, snapshot),
    )


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks with deterministic tie handling, without SciPy."""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(left: torch.Tensor, right: torch.Tensor) -> Optional[float]:
    left_array = left.detach().cpu().numpy().astype(np.float64, copy=False)
    right_array = right.detach().cpu().numpy().astype(np.float64, copy=False)
    left_rank = _rankdata(left_array)
    right_rank = _rankdata(right_array)
    if np.std(left_rank) == 0.0 or np.std(right_rank) == 0.0:
        return None
    value = float(np.corrcoef(left_rank, right_rank)[0, 1])
    return value if math.isfinite(value) else None


def _descending_rank(
    candidates: torch.Tensor,
    selected: torch.Tensor,
) -> int:
    values = candidates.detach().cpu().numpy().astype(np.float64, copy=False)
    target = float(selected.detach().cpu().item())
    return 1 + int(np.count_nonzero(values > target))


def compare_shadow_trace(
    baseline: ShadowPlanTrace,
    shadow: ShadowPlanTrace,
) -> Dict[str, Any]:
    """Return per-call action, score, risk, and ranking diagnostics."""

    if baseline.noise_fingerprint != shadow.noise_fingerprint:
        raise ValueError("Baseline and shadow traces did not use the same noise")
    if baseline.input_fingerprint != shadow.input_fingerprint:
        raise ValueError("Baseline and shadow traces did not use the same input")
    if baseline.safety_weight != 0.0:
        raise ValueError("The comparison baseline must have safety_weight=0")
    if (
        baseline.aggregation != shadow.aggregation
        or baseline.aggregation_discount != shadow.aggregation_discount
        or baseline.planner_cap != shadow.planner_cap
    ):
        raise ValueError("Baseline and shadow traces use different risk semantics")
    baseline_action = baseline.action.detach().cpu().numpy()
    shadow_action = shadow.action.detach().cpu().numpy()
    delta = shadow_action - baseline_action
    baseline_noise_slot_elites = set(
        int(value) for value in baseline.final_elite_indices.cpu().tolist()
    )
    shadow_noise_slot_elites = set(
        int(value) for value in shadow.final_elite_indices.cpu().tolist()
    )
    slot_union = baseline_noise_slot_elites | shadow_noise_slot_elites
    slot_intersection = baseline_noise_slot_elites & shadow_noise_slot_elites
    elite_count = int(shadow.final_elite_indices.numel())
    task_elites = set(
        int(value)
        for value in torch.topk(
            shadow.final_task_values,
            elite_count,
        ).indices.cpu().tolist()
    )
    penalized_elites = set(
        int(value) for value in shadow.final_elite_indices.cpu().tolist()
    )
    candidate_intersection = task_elites & penalized_elites
    candidate_union = task_elites | penalized_elites
    task_top = int(torch.argmax(shadow.final_task_values).item())
    shadow_top = int(torch.argmax(shadow.final_scores).item())
    task_sacrifice = float(
        baseline.selected_task_value.item()
        - shadow.selected_task_value.item()
    )
    capped_reduction = float(
        baseline.selected_capped_risk.item()
        - shadow.selected_capped_risk.item()
    )
    uncapped_reduction = float(
        baseline.selected_uncapped_risk.item()
        - shadow.selected_uncapped_risk.item()
    )
    return {
        "action_baseline": baseline_action.astype(float).tolist(),
        "action_shadow": shadow_action.astype(float).tolist(),
        "action_delta": delta.astype(float).tolist(),
        "translation_action_delta": float(delta[0]),
        "rotation_action_delta": float(delta[1]),
        "action_l2": float(np.linalg.norm(delta)),
        "exact_action_changed": bool(np.any(delta != 0.0)),
        "translation_delta_direction": int(np.sign(float(delta[0]))),
        "baseline_selected_task_value": float(
            baseline.selected_task_value.item()
        ),
        "shadow_selected_task_value": float(shadow.selected_task_value.item()),
        "task_value_sacrifice": task_sacrifice,
        "baseline_selected_uncapped_risk": float(
            baseline.selected_uncapped_risk.item()
        ),
        "shadow_selected_uncapped_risk": float(
            shadow.selected_uncapped_risk.item()
        ),
        "uncapped_risk_reduction": uncapped_reduction,
        "baseline_selected_capped_risk": float(
            baseline.selected_capped_risk.item()
        ),
        "shadow_selected_capped_risk": float(
            shadow.selected_capped_risk.item()
        ),
        "capped_risk_reduction": capped_reduction,
        "candidate_task_score_min": float(shadow.final_task_values.min().item()),
        "candidate_task_score_max": float(shadow.final_task_values.max().item()),
        "candidate_task_score_mean": float(
            shadow.final_task_values.mean().item()
        ),
        "candidate_penalized_score_min": float(shadow.final_scores.min().item()),
        "candidate_penalized_score_max": float(shadow.final_scores.max().item()),
        "candidate_penalized_score_mean": float(
            shadow.final_scores.mean().item()
        ),
        "candidate_uncapped_risk_min": float(
            shadow.final_uncapped_risks.min().item()
        ),
        "candidate_uncapped_risk_max": float(
            shadow.final_uncapped_risks.max().item()
        ),
        "candidate_uncapped_risk_mean": float(
            shadow.final_uncapped_risks.mean().item()
        ),
        "candidate_capped_risk_min": float(
            shadow.final_capped_risks.min().item()
        ),
        "candidate_capped_risk_max": float(
            shadow.final_capped_risks.max().item()
        ),
        "candidate_capped_risk_mean": float(
            shadow.final_capped_risks.mean().item()
        ),
        "task_penalized_spearman": _spearman(
            shadow.final_task_values,
            shadow.final_scores,
        ),
        "top1_agreement": bool(task_top == shadow_top),
        "task_top1_index": task_top,
        "penalized_top1_index": shadow_top,
        "elite_overlap_count": len(candidate_intersection),
        "elite_overlap_fraction": float(
            len(candidate_intersection) / max(1, elite_count)
        ),
        "elite_jaccard": float(
            len(candidate_intersection) / max(1, len(candidate_union))
        ),
        "cross_run_noise_slot_elite_overlap_count": len(slot_intersection),
        "cross_run_noise_slot_elite_overlap_fraction": float(
            len(slot_intersection) / max(1, len(baseline_noise_slot_elites))
        ),
        "cross_run_noise_slot_elite_jaccard": float(
            len(slot_intersection) / max(1, len(slot_union))
        ),
        "baseline_sequence_risk_rank": _descending_rank(
            shadow.final_capped_risks,
            baseline.selected_capped_risk,
        ),
        "shadow_sequence_task_rank": _descending_rank(
            shadow.final_task_values,
            shadow.selected_task_value,
        ),
        "planner_seconds": float(shadow.planner_seconds),
        "safety_inference_seconds": float(shadow.safety_inference_seconds),
        "noise_fingerprint": shadow.noise_fingerprint,
    }


def run_translation_shadow_sweep(
    agent: Any,
    snapshot: PlannerSnapshot,
    schedule: PlannerNoiseSchedule,
    *,
    safety_weights: Sequence[float],
    aggregations: Sequence[str] = SUPPORTED_AGGREGATIONS,
    aggregation_discount: float = 1.0,
    planner_cap: float = 1.0,
) -> Dict[str, Dict[float, ShadowPlanTrace]]:
    """Run independent same-noise MPPI instances for a weight/aggregation grid."""

    weights = tuple(float(value) for value in safety_weights)
    if not weights:
        raise ValueError("At least one safety weight is required")
    if any(not math.isfinite(value) or value < 0.0 for value in weights):
        raise ValueError("Safety weights must be finite and nonnegative")
    if len(set(weights)) != len(weights):
        raise ValueError("Safety weights must be unique")
    output: Dict[str, Dict[float, ShadowPlanTrace]] = {}
    canonical_baseline: Optional[ShadowPlanTrace] = None
    for aggregation in aggregations:
        traces: Dict[float, ShadowPlanTrace] = {}
        for weight in weights:
            if weight == 0.0 and canonical_baseline is not None:
                candidate_started = time.perf_counter()
                (
                    _,
                    _,
                    candidate_uncapped,
                    candidate_capped,
                    candidate_safety_seconds,
                ) = estimate_translation_risk(
                    agent,
                    snapshot.latent_single.repeat(
                        int(agent.config["num_samples"]),
                        1,
                    ),
                    canonical_baseline.final_actions,
                    aggregation=aggregation,
                    aggregation_discount=aggregation_discount,
                    planner_cap=planner_cap,
                )
                candidate_elapsed = time.perf_counter() - candidate_started
                selected_started = time.perf_counter()
                (
                    selected_step_uncapped,
                    selected_step_capped,
                    selected_uncapped,
                    selected_capped,
                    selected_safety_seconds,
                ) = estimate_translation_risk(
                    agent,
                    snapshot.latent_single,
                    canonical_baseline.mean.unsqueeze(1),
                    aggregation=aggregation,
                    aggregation_discount=aggregation_discount,
                    planner_cap=planner_cap,
                )
                selected_elapsed = time.perf_counter() - selected_started
                traces[weight] = replace(
                    canonical_baseline,
                    aggregation=str(aggregation),
                    aggregation_discount=float(aggregation_discount),
                    planner_cap=float(planner_cap),
                    final_uncapped_risks=_clone_tensor(candidate_uncapped),
                    final_capped_risks=_clone_tensor(candidate_capped),
                    selected_uncapped_risk=_clone_tensor(
                        selected_uncapped.squeeze()
                    ),
                    selected_capped_risk=_clone_tensor(
                        selected_capped.squeeze()
                    ),
                    selected_step_uncapped_risk=_clone_tensor(
                        selected_step_uncapped.squeeze(-1)
                    ),
                    selected_step_capped_risk=_clone_tensor(
                        selected_step_capped.squeeze(-1)
                    ),
                    planner_seconds=float(
                        canonical_baseline.core_planner_seconds
                        + candidate_elapsed
                        + selected_elapsed
                    ),
                    safety_inference_seconds=float(
                        candidate_safety_seconds + selected_safety_seconds
                    ),
                    candidate_safety_inference_seconds=float(
                        candidate_safety_seconds
                    ),
                    selected_diagnostic_seconds=float(selected_elapsed),
                )
            else:
                traces[weight] = run_translation_shadow_plan(
                    agent,
                    snapshot,
                    schedule,
                    safety_weight=weight,
                    aggregation=aggregation,
                    aggregation_discount=aggregation_discount,
                    planner_cap=planner_cap,
                )
                if weight == 0.0 and canonical_baseline is None:
                    canonical_baseline = traces[weight]
        output[str(aggregation)] = traces
    return output
