"""Focused, simulation-free checks for Translation-only shadow MPPI.

The checks in this file deliberately do not construct SOFA.  They exercise the
read-only planner against a small real TD-MPC2 agent and against a
hand-computable multi-iteration MPPI fixture.
"""

from __future__ import annotations

import copy
import inspect
import json
import math
import random
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import tdmpc2.agent as agent_module
import tdmpc2.shadow_planner as shadow_planner
from evaluate_safety_rollout import build_episode_windows, strict_json_dumps
from tdmpc2.agent import TDMPC2Agent
from tdmpc2.common import load_config
from tdmpc2.replay_buffer import EpisodeReplayBuffer, REPLAY_SAFETY_COST_NAMES
from tdmpc2.shadow_planner import (
    PlannerIterationNoise,
    PlannerNoiseSchedule,
    PlannerSnapshot,
    ShadowPlanTrace,
    capture_device_torch_rng_state,
    capture_planner_snapshot,
    compare_shadow_trace,
    estimate_translation_risk,
    make_planner_noise_schedule,
    run_translation_shadow_plan,
    run_translation_shadow_sweep,
)
from train import DEFAULT_CONFIG, build_agent_config


def _assert_nested_equal(left, right) -> None:
    """Compare RNG, optimizer, and model states without tolerance."""

    assert type(left) is type(right)
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(left_item, right_item)
    elif isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    else:
        assert left == right


def _assert_raises(function, exception_type) -> None:
    try:
        function()
    except exception_type:
        return
    if isinstance(exception_type, tuple):
        expected = "/".join(item.__name__ for item in exception_type)
    else:
        expected = exception_type.__name__
    raise AssertionError(f"Expected {expected}, but no exception was raised")


def _small_agent(
    *,
    horizon: int = 3,
    iterations: int = 2,
    num_samples: int = 8,
    num_elites: int = 3,
    num_pi_trajs: int = 2,
) -> TDMPC2Agent:
    full_config = load_config(DEFAULT_CONFIG)
    full_config["model"].update(
        {
            "latent_dim": 16,
            "enc_dim": 16,
            "mlp_dim": 16,
            "num_enc_layers": 2,
            "simnorm_dim": 4,
            "num_q": 3,
            "dropout": 0.0,
            "num_bins": 11,
            "vmin": -5.0,
            "vmax": 5.0,
        }
    )
    full_config["training"].update(
        {
            "horizon": int(horizon),
            "lr": 3e-4,
            "enc_lr_scale": 0.3,
            "rho": 0.5,
            "consistency_coef": 20.0,
            "reward_coef": 0.1,
            "value_coef": 0.1,
            "termination_coef": 1.0,
            "entropy_coef": 1e-4,
            "grad_clip_norm": 20.0,
            "tau": 0.01,
            "discount_denom": 5.0,
            "discount_min": 0.95,
            "discount_max": 0.995,
        }
    )
    full_config["planning"].update(
        {
            "horizon": int(horizon),
            "mpc": True,
            "num_samples": int(num_samples),
            "num_elites": int(num_elites),
            "num_pi_trajs": int(num_pi_trajs),
            "iterations": int(iterations),
            "max_std": 1.0,
            "min_std": 0.05,
            "temperature": 0.5,
        }
    )
    full_config["safety"].update(
        {
            "loss_coef": 1.0,
            "curvature_loss_coef": 1.0,
            "translation_error_loss_coef": 1.0,
            "curvature_scale_mm_inv": 0.1,
            "translation_error_scale": 1.0,
        }
    )
    return TDMPC2Agent(
        14,
        2,
        build_agent_config(full_config),
        episode_length=20,
        device=torch.device("cpu"),
    )


def _trace_tensor_fields(trace: ShadowPlanTrace):
    names = (
        "action",
        "mean",
        "std",
        "final_actions",
        "final_task_values",
        "final_uncapped_risks",
        "final_capped_risks",
        "final_scores",
        "final_elite_indices",
        "final_elite_weights",
        "selected_task_value",
        "selected_uncapped_risk",
        "selected_capped_risk",
        "selected_step_uncapped_risk",
        "selected_step_capped_risk",
    )
    return {name: getattr(trace, name) for name in names}


def _assert_trace_equal(left: ShadowPlanTrace, right: ShadowPlanTrace) -> None:
    """Compare deterministic trace content; measured timings may differ."""

    assert left.safety_weight == right.safety_weight
    assert left.aggregation == right.aggregation
    assert left.aggregation_discount == right.aggregation_discount
    assert left.planner_cap == right.planner_cap
    assert left.noise_fingerprint == right.noise_fingerprint
    left_fields = _trace_tensor_fields(left)
    right_fields = _trace_tensor_fields(right)
    assert left_fields.keys() == right_fields.keys()
    for name in left_fields:
        assert torch.equal(left_fields[name], right_fields[name]), name
    for left_values, right_values in (
        (left.iteration_means, right.iteration_means),
        (left.iteration_stds, right.iteration_stds),
        (left.iteration_elites, right.iteration_elites),
    ):
        assert len(left_values) == len(right_values)
        for left_value, right_value in zip(left_values, right_values):
            assert torch.equal(left_value, right_value)


def _schedule_tensors(schedule: PlannerNoiseSchedule):
    values = list(schedule.policy)
    for iteration in schedule.iterations:
        values.extend(
            (
                iteration.proposal,
                iteration.terminal_policy,
                iteration.q_indices,
            )
        )
    values.extend(
        (
            schedule.diagnostic_terminal_policy,
            schedule.diagnostic_q_indices,
            schedule.core_end_torch_state,
        )
    )
    return tuple(values)


class _FakeEnvironment:
    """Expose every RNG family used by the current stEVE wrapper."""

    def __init__(self) -> None:
        self.np_random = np.random.default_rng(101)
        self._env = SimpleNamespace(_np_random=np.random.default_rng(102))
        self._simulation = SimpleNamespace(_rng=np.random.default_rng(103))
        self.action_space = SimpleNamespace(np_random=np.random.default_rng(104))
        self.actions = []

    def step(self, action):
        self.actions.append(np.asarray(action, dtype=np.float32).copy())


def _environment_rng_state(environment: _FakeEnvironment):
    generators = (
        environment.np_random,
        environment._env._np_random,
        environment._simulation._rng,
        environment.action_space.np_random,
    )
    return tuple(
        copy.deepcopy(generator.bit_generator.state) for generator in generators
    )


def _test_read_only_identity_and_complete_isolation() -> None:
    # The normal agent path must neither import nor call shadow planning.
    assert "shadow_planner" not in inspect.getsource(agent_module)

    torch.manual_seed(4701)
    np.random.seed(4701)
    random.seed(4701)
    agent = _small_agent(iterations=3)
    observation = np.linspace(-1.0, 1.0, 14, dtype=np.float32)
    with torch.no_grad():
        agent.previous_mean.copy_(
            torch.tensor(
                [[0.30, -0.20], [0.10, 0.40], [-0.25, 0.15]],
                dtype=torch.float32,
            )
        )

    environment = _FakeEnvironment()
    replay = EpisodeReplayBuffer(
        32,
        14,
        2,
        3,
        2,
        safety_cost_names=REPLAY_SAFETY_COST_NAMES,
        seed=4702,
    )

    previous_before = agent.previous_mean.detach().clone()
    model_before = copy.deepcopy(agent.model.state_dict())
    model_flags_before = {
        name: module.training for name, module in agent.model.named_modules()
    }
    model_optimizer_before = copy.deepcopy(agent.model_optimizer.state_dict())
    policy_optimizer_before = copy.deepcopy(agent.policy_optimizer.state_dict())
    torch_before = torch.get_rng_state().clone()
    numpy_before = copy.deepcopy(np.random.get_state())
    python_before = random.getstate()
    environment_before = _environment_rng_state(environment)
    replay_before = copy.deepcopy(replay.state_dict()["rng_state"])

    snapshot = capture_planner_snapshot(agent, observation, first_step=False)
    schedule = make_planner_noise_schedule(
        agent,
        initial_torch_state=torch_before,
    )
    schedule_before = tuple(value.clone() for value in _schedule_tensors(schedule))
    sweep = run_translation_shadow_sweep(
        agent,
        snapshot,
        schedule,
        safety_weights=(0.0, 0.5),
        aggregations=("max", "discounted_sum"),
        aggregation_discount=0.75,
        planner_cap=1.0,
    )
    baseline = sweep["max"][0.0]

    assert torch.equal(agent.previous_mean, previous_before)
    _assert_nested_equal(agent.model.state_dict(), model_before)
    assert {
        name: module.training for name, module in agent.model.named_modules()
    } == model_flags_before
    _assert_nested_equal(agent.model_optimizer.state_dict(), model_optimizer_before)
    _assert_nested_equal(agent.policy_optimizer.state_dict(), policy_optimizer_before)
    assert torch.equal(torch.get_rng_state(), torch_before)
    _assert_nested_equal(np.random.get_state(), numpy_before)
    assert random.getstate() == python_before
    _assert_nested_equal(_environment_rng_state(environment), environment_before)
    _assert_nested_equal(replay.state_dict()["rng_state"], replay_before)
    for current, expected in zip(_schedule_tensors(schedule), schedule_before):
        assert torch.equal(current, expected)
    assert not environment.actions

    # A fixed value query is exactly unchanged by the shadow calculation.
    latent = agent.model.encode(agent._tensor_observation(observation))
    actions = torch.zeros(agent.horizon, 1, agent.action_dim)
    value_rng = torch.get_rng_state().clone()
    value_before = agent._estimate_value(latent, actions)
    torch.set_rng_state(value_rng)
    run_translation_shadow_plan(
        agent,
        snapshot,
        schedule,
        safety_weight=0.2,
        aggregation="max",
    )
    assert torch.equal(torch.get_rng_state(), value_rng)
    value_after = agent._estimate_value(latent, actions)
    assert torch.equal(value_before, value_after)

    # The explicit schedule reproduces production _plan bit-for-bit.  Only
    # production planning is then allowed to update previous_mean/RNG.
    agent.previous_mean.copy_(previous_before)
    torch.set_rng_state(torch_before)
    production_action = agent._plan(
        agent._tensor_observation(observation),
        first_step=False,
        eval_mode=True,
    )
    assert torch.equal(production_action, baseline.action)
    assert torch.equal(agent.previous_mean, baseline.mean)
    assert torch.equal(torch.get_rng_state(), schedule.core_end_torch_state)

    agent.previous_mean.copy_(previous_before)
    torch.set_rng_state(torch_before)
    production_numpy_action = agent.act(
        observation,
        first_step=False,
        eval_mode=True,
    )
    np.testing.assert_array_equal(
        production_numpy_action,
        baseline.action.cpu().numpy().astype(np.float32),
    )
    environment.step(production_numpy_action)
    assert len(environment.actions) == 1
    np.testing.assert_array_equal(environment.actions[0], production_numpy_action)

    # Ordinary act/_plan continues to have no Safety-Head dependency.
    agent.previous_mean.copy_(previous_before)
    torch.set_rng_state(torch_before)
    with patch.object(
        agent.model,
        "safety_transformed",
        side_effect=AssertionError("Production planning called Safety"),
    ) as safety_spy:
        agent.act(observation, first_step=False, eval_mode=True)
    assert safety_spy.call_count == 0


def _risk_side_effect(step_values):
    values = iter(float(value) for value in step_values)

    def predict(latent, action):
        del action
        value = next(values)
        curvature = latent.new_full((latent.shape[0],), 99.0)
        translation = latent.new_full((latent.shape[0],), value)
        return torch.stack((curvature, translation), dim=-1)

    return predict


def _test_risk_aggregation_cap_and_validation() -> None:
    agent = _small_agent(horizon=3, iterations=1)
    latent = torch.zeros(1, agent.model.latent_dim)
    actions = torch.zeros(3, 1, 2)

    with patch.object(
        agent.model,
        "safety_transformed",
        side_effect=_risk_side_effect((0.2, 1.5, 0.4)),
    ), patch.object(
        agent.model,
        "decode_safety_transformed",
        side_effect=lambda value: value,
    ):
        per_step, bounded, uncapped, capped, elapsed = estimate_translation_risk(
            agent,
            latent,
            actions,
            aggregation="max",
            aggregation_discount=0.5,
            planner_cap=1.0,
        )
    torch.testing.assert_close(per_step[:, 0], torch.tensor([0.2, 1.5, 0.4]))
    torch.testing.assert_close(bounded[:, 0], torch.tensor([0.2, 1.0, 0.4]))
    torch.testing.assert_close(uncapped, torch.tensor([1.5]))
    torch.testing.assert_close(capped, torch.tensor([1.0]))
    assert math.isfinite(elapsed) and elapsed >= 0.0

    with patch.object(
        agent.model,
        "safety_transformed",
        side_effect=_risk_side_effect((0.2, 1.5, 0.4)),
    ), patch.object(
        agent.model,
        "decode_safety_transformed",
        side_effect=lambda value: value,
    ):
        per_step_sum, bounded_sum, uncapped_sum, capped_sum, _ = (
            estimate_translation_risk(
                agent,
                latent,
                actions,
                aggregation="discounted_sum",
                aggregation_discount=0.5,
                planner_cap=1.0,
            )
        )
    assert torch.equal(per_step_sum, per_step)
    assert torch.equal(bounded_sum, bounded)
    torch.testing.assert_close(uncapped_sum, torch.tensor([1.05]))
    torch.testing.assert_close(capped_sum, torch.tensor([0.8]))

    for kwargs in (
        {"aggregation": "thresholded"},
        {"aggregation": "max", "aggregation_discount": -0.1},
        {"aggregation": "max", "planner_cap": 0.0},
    ):
        options = {
            "aggregation": "max",
            "aggregation_discount": 1.0,
            "planner_cap": 1.0,
        }
        options.update(kwargs)
        _assert_raises(
            lambda options=options: estimate_translation_risk(
                agent,
                latent,
                actions,
                **options,
            ),
            ValueError,
        )

    with patch.object(
        agent.model,
        "safety_transformed",
        return_value=torch.tensor([[0.0, float("nan")]]),
    ):
        _assert_raises(
            lambda: estimate_translation_risk(
                agent,
                latent,
                actions,
                aggregation="max",
                aggregation_discount=1.0,
                planner_cap=1.0,
            ),
            FloatingPointError,
        )
    with patch.object(
        agent.model,
        "safety_transformed",
        return_value=torch.tensor([[0.0, -0.1]]),
    ), patch.object(
        agent.model,
        "decode_safety_transformed",
        side_effect=lambda value: value,
    ):
        _assert_raises(
            lambda: estimate_translation_risk(
                agent,
                latent,
                actions,
                aggregation="max",
                aggregation_discount=1.0,
                planner_cap=1.0,
            ),
            ValueError,
        )


def _test_same_noise_lambda_zero_and_channel_isolation() -> None:
    torch.manual_seed(4710)
    agent = _small_agent(iterations=2)
    observation = np.linspace(0.75, -0.75, 14, dtype=np.float32)
    snapshot = capture_planner_snapshot(agent, observation, first_step=True)
    schedule = make_planner_noise_schedule(agent, seed=4711)
    schedule_clone = tuple(value.clone() for value in _schedule_tensors(schedule))

    first = run_translation_shadow_plan(
        agent,
        snapshot,
        schedule,
        safety_weight=0.4,
        aggregation="max",
    )
    second = run_translation_shadow_plan(
        agent,
        snapshot,
        schedule,
        safety_weight=0.4,
        aggregation="max",
    )
    _assert_trace_equal(first, second)
    for current, expected in zip(_schedule_tensors(schedule), schedule_clone):
        assert torch.equal(current, expected)

    zero_max = run_translation_shadow_plan(
        agent,
        snapshot,
        schedule,
        safety_weight=0.0,
        aggregation="max",
        aggregation_discount=0.25,
    )
    zero_sum = run_translation_shadow_plan(
        agent,
        snapshot,
        schedule,
        safety_weight=0.0,
        aggregation="discounted_sum",
        aggregation_discount=0.25,
    )
    for name in (
        "action",
        "mean",
        "std",
        "final_actions",
        "final_task_values",
        "final_scores",
        "final_elite_indices",
        "final_elite_weights",
        "selected_task_value",
    ):
        assert torch.equal(getattr(zero_max, name), getattr(zero_sum, name)), name
    assert zero_max.noise_fingerprint == zero_sum.noise_fingerprint

    def curvature_output(value):
        def forward(features):
            return features.new_full((features.shape[0], 1), float(value))

        return forward

    with patch.object(
        agent.model.safety_curvature_head,
        "forward",
        side_effect=curvature_output(0.0),
    ) as curvature_zero_spy:
        curvature_zero = run_translation_shadow_plan(
            agent,
            snapshot,
            schedule,
            safety_weight=0.5,
            aggregation="max",
        )
    with patch.object(
        agent.model.safety_curvature_head,
        "forward",
        side_effect=curvature_output(10.0),
    ) as curvature_large_spy:
        curvature_large = run_translation_shadow_plan(
            agent,
            snapshot,
            schedule,
            safety_weight=0.5,
            aggregation="max",
        )
    assert curvature_zero_spy.call_count > 0
    assert curvature_large_spy.call_count == curvature_zero_spy.call_count
    _assert_trace_equal(curvature_zero, curvature_large)

    def translation_output(value):
        def forward(features):
            return features.new_full((features.shape[0], 1), float(value))

        return forward

    with patch.object(
        agent.model.safety_translation_error_head,
        "forward",
        side_effect=translation_output(0.0),
    ):
        translation_zero = run_translation_shadow_plan(
            agent,
            snapshot,
            schedule,
            safety_weight=0.5,
            aggregation="max",
        )
    with patch.object(
        agent.model.safety_translation_error_head,
        "forward",
        side_effect=translation_output(math.log(2.0)),
    ):
        translation_positive = run_translation_shadow_plan(
            agent,
            snapshot,
            schedule,
            safety_weight=0.5,
            aggregation="max",
        )
    assert torch.equal(
        translation_positive.final_task_values,
        translation_zero.final_task_values,
    )
    torch.testing.assert_close(
        translation_positive.final_capped_risks,
        torch.ones_like(translation_positive.final_capped_risks),
    )
    torch.testing.assert_close(
        translation_positive.final_scores,
        translation_positive.final_task_values - 0.5,
    )
    assert not torch.equal(
        translation_positive.final_scores,
        translation_zero.final_scores,
    )


def _synthetic_task_value(agent, latent, actions, terminal_noise, q_indices):
    del agent, latent, terminal_noise, q_indices
    value = -torch.square(actions[..., 0] - 0.75).sum(dim=0)
    value -= 0.1 * torch.square(actions[..., 1]).sum(dim=0)
    return value.unsqueeze(-1)


def _synthetic_risk(
    agent,
    latent,
    actions,
    *,
    aggregation,
    aggregation_discount,
    planner_cap,
):
    del agent, latent
    batch = actions.shape[1]
    rollout_latent = actions.new_zeros(batch)
    values = []
    for step in range(actions.shape[0]):
        values.append(
            torch.relu(actions[step, :, 0] + 0.5 * rollout_latent)
        )
        rollout_latent = rollout_latent + actions[step, :, 0]
    uncapped_steps = torch.stack(values)
    capped_steps = uncapped_steps.clamp(max=float(planner_cap))
    if aggregation == "max":
        uncapped = uncapped_steps.max(dim=0).values
        capped = capped_steps.max(dim=0).values
    else:
        weights = actions.new_tensor(
            [
                float(aggregation_discount) ** step
                for step in range(actions.shape[0])
            ]
        ).unsqueeze(1)
        uncapped = (weights * uncapped_steps).sum(dim=0)
        capped = (weights * capped_steps).sum(dim=0)
    return uncapped_steps, capped_steps, uncapped, capped, 0.0


def _synthetic_noise_schedule(agent: TDMPC2Agent) -> PlannerNoiseSchedule:
    proposal_values = []
    translations = (
        (
            (-0.9, -0.2, 0.4, 0.9),
            (-0.7, 0.1, 0.6, 1.0),
        ),
        (
            (-1.0, -0.3, 0.2, 0.8),
            (-0.8, -0.1, 0.5, 0.9),
        ),
    )
    rotations = (
        (
            (0.2, -0.3, 0.1, -0.2),
            (-0.1, 0.2, -0.2, 0.3),
        ),
        (
            (-0.2, 0.3, -0.1, 0.2),
            (0.2, -0.3, 0.1, -0.2),
        ),
    )
    for translation, rotation in zip(translations, rotations):
        proposal = torch.empty(2, 4, 2)
        proposal[..., 0] = torch.tensor(translation)
        proposal[..., 1] = torch.tensor(rotation)
        proposal_values.append(
            PlannerIterationNoise(
                proposal=proposal,
                terminal_policy=torch.zeros(4, 2),
                q_indices=torch.tensor([0, 1]),
            )
        )
    policy = (torch.empty(0, 2), torch.empty(0, 2))
    iteration_tuple = tuple(proposal_values)
    diagnostic_terminal_policy = torch.zeros(1, 2)
    diagnostic_q_indices = torch.tensor([0, 1])
    fingerprint = shadow_planner._hash_schedule(
        policy,
        iteration_tuple,
        diagnostic_terminal_policy,
        diagnostic_q_indices,
    )
    return PlannerNoiseSchedule(
        policy=policy,
        iterations=iteration_tuple,
        diagnostic_terminal_policy=diagnostic_terminal_policy,
        diagnostic_q_indices=diagnostic_q_indices,
        fingerprint=fingerprint,
        core_end_torch_state=capture_device_torch_rng_state(agent.device),
    )


def _reference_synthetic_mppi(schedule, safety_weight):
    mean = torch.zeros(2, 2)
    std = torch.ones(2, 2)
    means = []
    stds = []
    elites = []
    final_actions = None
    final_task = None
    final_risk = None
    for iteration in schedule.iterations:
        actions = (
            mean.unsqueeze(1) + std.unsqueeze(1) * iteration.proposal
        ).clamp(-1.0, 1.0)
        task = _synthetic_task_value(
            None, None, actions, None, None
        ).squeeze(-1)
        risk = _synthetic_risk(
            None,
            None,
            actions,
            aggregation="max",
            aggregation_discount=1.0,
            planner_cap=1.0,
        )[3]
        score = task - float(safety_weight) * risk
        elite_indices = torch.topk(score, 2).indices
        elite_values = score[elite_indices].unsqueeze(-1)
        elite_actions = actions[:, elite_indices]
        maximum = elite_values.max(dim=0).values
        weights = torch.exp(elite_values - maximum)
        weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-9)
        broadcast = weights.unsqueeze(0)
        mean = (broadcast * elite_actions).sum(dim=1) / (
            broadcast.sum(dim=1) + 1e-9
        )
        variance = (
            broadcast * (elite_actions - mean.unsqueeze(1)).pow(2)
        ).sum(dim=1) / (broadcast.sum(dim=1) + 1e-9)
        std = variance.sqrt().clamp(0.05, 1.0)
        means.append(mean.clone())
        stds.append(std.clone())
        elites.append(elite_indices.clone())
        final_actions = actions
        final_task = task
        final_risk = risk
    return {
        "mean": mean,
        "std": std,
        "means": tuple(means),
        "stds": tuple(stds),
        "elites": tuple(elites),
        "final_actions": final_actions,
        "final_task": final_task,
        "final_risk": final_risk,
    }


def _test_multi_iteration_mppi_fidelity() -> None:
    agent = _small_agent(
        horizon=2,
        iterations=2,
        num_samples=4,
        num_elites=2,
        num_pi_trajs=0,
    )
    agent.config["temperature"] = 1.0
    snapshot = PlannerSnapshot(
        latent_single=torch.zeros(1, agent.model.latent_dim),
        previous_mean=torch.zeros(2, 2),
        first_step=True,
    )
    schedule = _synthetic_noise_schedule(agent)
    with patch.object(
        shadow_planner,
        "estimate_task_value_explicit",
        side_effect=_synthetic_task_value,
    ), patch.object(
        shadow_planner,
        "estimate_translation_risk",
        side_effect=_synthetic_risk,
    ):
        baseline = run_translation_shadow_plan(
            agent,
            snapshot,
            schedule,
            safety_weight=0.0,
            aggregation="max",
        )
        shadow = run_translation_shadow_plan(
            agent,
            snapshot,
            schedule,
            safety_weight=2.0,
            aggregation="max",
        )

    baseline_reference = _reference_synthetic_mppi(schedule, 0.0)
    shadow_reference = _reference_synthetic_mppi(schedule, 2.0)
    for trace, reference in (
        (baseline, baseline_reference),
        (shadow, shadow_reference),
    ):
        torch.testing.assert_close(trace.mean, reference["mean"], atol=1e-7, rtol=0)
        torch.testing.assert_close(trace.std, reference["std"], atol=1e-7, rtol=0)
        for actual, expected in zip(trace.iteration_means, reference["means"]):
            torch.testing.assert_close(actual, expected, atol=1e-7, rtol=0)
        for actual, expected in zip(trace.iteration_stds, reference["stds"]):
            torch.testing.assert_close(actual, expected, atol=1e-7, rtol=0)
        for actual, expected in zip(trace.iteration_elites, reference["elites"]):
            assert torch.equal(actual, expected)
    assert not torch.equal(
        baseline.iteration_means[0],
        shadow.iteration_means[0],
    )
    assert baseline.noise_fingerprint == shadow.noise_fingerprint

    # Deliberately wrong approximation: re-rank only the final baseline
    # population.  It must not equal independently iterated shadow MPPI.
    actions = baseline.final_actions
    task = _synthetic_task_value(None, None, actions, None, None).squeeze(-1)
    risk = _synthetic_risk(
        None,
        None,
        actions,
        aggregation="max",
        aggregation_discount=1.0,
        planner_cap=1.0,
    )[3]
    wrong_score = task - 2.0 * risk
    wrong_elites = torch.topk(wrong_score, 2).indices
    wrong_values = wrong_score[wrong_elites].unsqueeze(-1)
    wrong_weights = torch.exp(wrong_values - wrong_values.max())
    wrong_weights = wrong_weights / (
        wrong_weights.sum(dim=0, keepdim=True) + 1e-9
    )
    wrong_broadcast = wrong_weights.unsqueeze(0)
    wrong_mean = (
        wrong_broadcast * actions[:, wrong_elites]
    ).sum(dim=1) / (wrong_broadcast.sum(dim=1) + 1e-9)
    assert torch.linalg.vector_norm(wrong_mean - shadow.mean).item() > 0.1


def _action_divergence(
    baseline,
    shadow,
    *,
    translation_tolerance=1e-3,
    rotation_tolerance=1e-3,
):
    baseline = np.asarray(baseline, dtype=np.float64)
    shadow = np.asarray(shadow, dtype=np.float64)
    delta = shadow - baseline
    sign_change = bool(
        abs(baseline[0]) > translation_tolerance
        and abs(shadow[0]) > translation_tolerance
        and baseline[0] * shadow[0] < 0.0
    )
    norm = float(np.linalg.norm(delta))
    return {
        "diverged": bool(
            abs(delta[0]) > translation_tolerance
            or abs(delta[1]) > rotation_tolerance
        ),
        "sign_change": sign_change,
        "l2": norm,
        "l2_gt_001": bool(norm > 0.01),
        "l2_gt_005": bool(norm > 0.05),
        "l2_gt_010": bool(norm > 0.10),
    }


def _complete_future_labels(episode_ids, translation_cost, horizon):
    identifiers = np.asarray(episode_ids)
    costs = np.asarray(translation_cost, dtype=np.float64)
    labels = np.zeros(identifiers.size, dtype=bool)
    valid = np.zeros(identifiers.size, dtype=bool)
    for window in build_episode_windows(identifiers, horizon):
        start = int(window[0])
        labels[start] = bool(np.any(costs[window] > 0.0))
        valid[start] = True
    return labels, valid


def _analysis_group_masks(modes, labels, valid):
    modes = np.asarray(modes)
    return {
        "future_blockage": valid & labels,
        "no_block": valid & ~labels,
        "lower_boundary": modes == "lower_boundary",
        "tree_end": modes == "tree_end",
        "ordinary_policy": modes == "policy",
    }


def _test_divergence_and_group_label_semantics() -> None:
    assert not _action_divergence([0.0, 0.0], [0.001, -0.001])["diverged"]
    assert _action_divergence([0.0, 0.0], [0.00101, 0.0])["diverged"]
    assert _action_divergence([0.2, 0.0], [-0.1, 0.0])["sign_change"]
    assert _action_divergence([-0.2, 0.0], [0.1, 0.0])["sign_change"]
    assert not _action_divergence([0.2, 0.0], [0.0, 0.0])["sign_change"]
    assert not _action_divergence([0.0, 0.0], [-0.1, 0.0])["sign_change"]
    assert not _action_divergence([1e-4, 0.0], [-1e-4, 0.0])[
        "sign_change"
    ]
    practical = _action_divergence([0.0, 0.0], [0.11, 0.0])
    assert practical["l2_gt_001"]
    assert practical["l2_gt_005"]
    assert practical["l2_gt_010"]

    episode_ids = np.asarray([0, 0, 0, 0, 1, 1, 1])
    cost = np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    labels, valid = _complete_future_labels(episode_ids, cost, 3)
    np.testing.assert_array_equal(valid, [True, True, False, False, True, False, False])
    np.testing.assert_array_equal(labels[valid], [True, True, True])
    # Index 3 must not see the positive transition in the next episode.
    assert not valid[3]

    modes = np.asarray(
        [
            "policy",
            "lower_boundary",
            "lower_boundary",
            "random",
            "tree_end",
            "tree_end",
            "policy",
        ]
    )
    groups = _analysis_group_masks(modes, labels, valid)
    np.testing.assert_array_equal(groups["future_blockage"], valid)
    assert not np.any(groups["no_block"])
    np.testing.assert_array_equal(
        groups["lower_boundary"],
        [False, True, True, False, False, False, False],
    )
    np.testing.assert_array_equal(
        groups["tree_end"],
        [False, False, False, False, True, True, False],
    )
    np.testing.assert_array_equal(
        groups["ordinary_policy"],
        [True, False, False, False, False, False, True],
    )


def _assert_all_defined_numbers_finite(value) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float, np.generic)):
        assert math.isfinite(float(value))
        return
    if isinstance(value, dict):
        for item in value.values():
            _assert_all_defined_numbers_finite(item)
        return
    if isinstance(value, (list, tuple, np.ndarray)):
        for item in value:
            _assert_all_defined_numbers_finite(item)
        return
    raise AssertionError(f"Unexpected report type {type(value).__name__}")


def _test_comparison_and_strict_json() -> None:
    agent = _small_agent(iterations=1)
    observation = np.zeros(14, dtype=np.float32)
    snapshot = capture_planner_snapshot(agent, observation, first_step=True)
    schedule = make_planner_noise_schedule(agent, seed=4720)
    baseline = run_translation_shadow_plan(
        agent,
        snapshot,
        schedule,
        safety_weight=0.0,
        aggregation="max",
    )
    shadow = run_translation_shadow_plan(
        agent,
        snapshot,
        schedule,
        safety_weight=0.2,
        aggregation="max",
    )
    comparison = compare_shadow_trace(baseline, shadow)
    payload = {
        "comparison": comparison,
        "unavailable_relative_reduction": None,
        "empty_group": {"count": 0, "mean": None},
        "numpy": np.asarray([1.0, 2.0], dtype=np.float32),
    }
    _assert_all_defined_numbers_finite(payload)
    text = strict_json_dumps(payload)
    decoded = json.loads(text)
    assert decoded["unavailable_relative_reduction"] is None
    assert decoded["empty_group"]["mean"] is None
    assert "NaN" not in text
    assert "Infinity" not in text
    for invalid in (float("nan"), float("inf"), float("-inf")):
        _assert_raises(
            lambda invalid=invalid: strict_json_dumps({"invalid": invalid}),
            (ValueError, TypeError),
        )


def main() -> None:
    _test_read_only_identity_and_complete_isolation()
    _test_risk_aggregation_cap_and_validation()
    _test_same_noise_lambda_zero_and_channel_isolation()
    _test_multi_iteration_mppi_fidelity()
    _test_divergence_and_group_label_semantics()
    _test_comparison_and_strict_json()
    print(
        "PASS: baseline bitwise identity, planner/model/RNG/replay isolation, "
        "same-noise deterministic replay, Translation risk max/sum/cap, "
        "lambda=0 identity, Curvature isolation, Translation scoring, "
        "multi-iteration MPPI fidelity, divergence/group labels, and strict JSON."
    )


if __name__ == "__main__":
    main()
