"""Fast dependency/shape test for the TD-MPC2 model, update, and planner."""

from __future__ import annotations

import copy
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

import evaluate_safety_head as safety_head_evaluation
from smoke_test_safety_aux import main as run_safety_aux_smoke_tests
from envs.safety import SAFETY_COST_NAMES as ENV_SAFETY_COST_NAMES
from eve.intervention import TRANSLATION_BLOCK_REASON_NAMES
from evaluate import build_agent_config as build_evaluation_agent_config
from train import (
    CHECKPOINT_FORMAT_VERSION,
    DEFAULT_CONFIG,
    LossAccumulator,
    SAFETY_MODEL_SCHEMA_VERSION,
    build_agent_config,
    save_checkpoint,
    validate_collected_safety_cost,
    validate_checkpoint_schema,
)
from tdmpc2.agent import TDMPC2Agent
from tdmpc2.common import (
    apply_cli_overrides,
    build_safety_aux_config,
    curvature_boundaries_from_diagnostics,
    load_config,
    load_torch_checkpoint,
    scalar_metrics,
)
from tdmpc2.replay_buffer import (
    LEGACY_THREE_CHANNEL_SAFETY_COST_NAMES,
    REPLAY_SAFETY_COST_NAMES,
    SAFETY_COST_SCHEMA_VERSION,
    EpisodeReplayBuffer,
)
from tdmpc2.safety_diagnostics import safety_batch_diagnostics
from tdmpc2.safety_aux_supervision import (
    SAFETY_AUXILIARY_CHECKPOINT_KEY,
    SafetyAuxiliarySupervisor,
)


class FixedReplay:
    """Return one deterministic, clone-on-read replay batch."""

    def __init__(self, batch, *, poison_safety: bool = False) -> None:
        self.batch = tuple(tensor.detach().clone() for tensor in batch)
        self.poison_safety = bool(poison_safety)

    def sample(self, device: torch.device):
        tensors = [tensor.detach().clone().to(device) for tensor in self.batch]
        if self.poison_safety:
            tensors[-1].fill_(float("nan"))
        return tuple(tensors)

    def sample_diagnostics(self, device: torch.device, *, seed: int):
        del seed
        return self.sample(device)


def _module_has_finite_nonzero_gradient(module: torch.nn.Module) -> bool:
    gradients = [
        parameter.grad
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    return bool(gradients) and all(
        torch.isfinite(gradient).all() for gradient in gradients
    ) and any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients)


def _assert_parameters_unchanged(
    before: dict[str, torch.Tensor],
    module: torch.nn.Module,
) -> None:
    after = dict(module.named_parameters())
    assert before.keys() == after.keys()
    for name, expected in before.items():
        torch.testing.assert_close(after[name], expected, rtol=0.0, atol=0.0)


def _assert_nested_state_equal(left, right) -> None:
    """Compare optimizer/RNG-style nested state without numeric tolerance."""

    assert type(left) is type(right)
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_state_equal(left_item, right_item)
    elif isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    else:
        assert left == right


def main() -> None:
    run_safety_aux_smoke_tests()
    assert scalar_metrics({"unavailable": float("nan")})["unavailable"] is None

    accumulator = LossAccumulator()
    accumulator.add(
        {
            "ordinary_loss": 1.0,
            "safety_translation_error_positive_count": 2.0,
            "safety_translation_error_positive_mae": 0.5,
        }
    )
    accumulator.add(
        {
            "ordinary_loss": 3.0,
            "safety_translation_error_positive_count": 1.0,
            "safety_translation_error_positive_mae": 2.0,
        }
    )
    accumulator.add(
        {
            "ordinary_loss": 5.0,
            "safety_translation_error_positive_count": 0.0,
            "safety_translation_error_positive_mae": float("nan"),
        }
    )
    accumulated = accumulator.pop_means()
    assert accumulated["ordinary_loss"] == 3.0
    assert accumulated["safety_translation_error_positive_count"] == 3.0
    np.testing.assert_allclose(
        accumulated["safety_translation_error_positive_mae"],
        1.0,
    )

    assert tuple(ENV_SAFETY_COST_NAMES) == REPLAY_SAFETY_COST_NAMES
    safety_cost_names = REPLAY_SAFETY_COST_NAMES
    valid_transition_cost = np.asarray([0.25, 0.5], dtype=np.float32)
    validated_transition_cost = validate_collected_safety_cost(
        valid_transition_cost,
        safety_cost_names=safety_cost_names,
    )
    np.testing.assert_array_equal(
        validated_transition_cost,
        valid_transition_cost,
    )
    assert validated_transition_cost is not valid_transition_cost
    for invalid_cost, expected_error, message in (
        (
            np.zeros(3, dtype=np.float32),
            ValueError,
            "unsupported legacy three-channel shape",
        ),
        (
            np.asarray([np.nan, 0.0], dtype=np.float32),
            FloatingPointError,
            "NaN or infinity",
        ),
        (
            np.asarray([-1.0, 0.0], dtype=np.float32),
            ValueError,
            "nonnegative",
        ),
        (
            np.zeros(2, dtype=np.float64),
            TypeError,
            "float32",
        ),
        (
            [0.0, 0.0],
            TypeError,
            "numpy.ndarray",
        ),
    ):
        try:
            validate_collected_safety_cost(
                invalid_cost,
                safety_cost_names=safety_cost_names,
            )
        except expected_error as exc:
            assert message in str(exc)
        else:
            raise AssertionError(
                "Training collection accepted invalid safety_cost data"
            )
    full_config = load_config(DEFAULT_CONFIG)
    full_config["model"].update(
        {
            "latent_dim": 16,
            "enc_dim": 16,
            "mlp_dim": 16,
            "num_enc_layers": 2,
            "simnorm_dim": 4,
            "num_q": 2,
            "dropout": 0.2,
            "num_bins": 11,
            "vmin": -5.0,
            "vmax": 5.0,
        }
    )
    full_config["training"].update(
        {
            "horizon": 3,
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
            "horizon": 3,
            "mpc": True,
            "num_samples": 8,
            "num_elites": 2,
            "num_pi_trajs": 2,
            "iterations": 1,
            "max_std": 2.0,
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
    overridden_config = copy.deepcopy(full_config)
    apply_cli_overrides(overridden_config, safety_loss_coef=0.1)
    assert overridden_config["safety"]["loss_coef"] == 0.1
    assert build_agent_config(overridden_config)["safety_loss_coef"] == 0.1
    config = build_agent_config(full_config)
    assert config == build_evaluation_agent_config(full_config)
    assert tuple(config["safety_cost_names"]) == tuple(
        ENV_SAFETY_COST_NAMES
    )
    assert config["safety_dim"] == 2
    for scale_key, invalid_scale in (
        ("curvature_scale_mm_inv", 1.0e-50),
        ("translation_error_scale", 1.0e40),
    ):
        invalid_scale_config = copy.deepcopy(full_config)
        invalid_scale_config["safety"][scale_key] = invalid_scale
        try:
            build_agent_config(invalid_scale_config)
        except ValueError as exc:
            assert "float32 replay data" in str(exc)
        else:
            raise AssertionError(
                f"Configuration accepted unrepresentable safety scale {invalid_scale}"
            )
    device = torch.device("cpu")
    agent = TDMPC2Agent(14, 2, config, episode_length=200, device=device)
    assert not any(
        isinstance(module, torch.nn.Dropout)
        for module in agent.model.safety_trunk.modules()
    )
    assert any(
        isinstance(module, torch.nn.Dropout)
        for module in agent.model.q_ensemble.modules()
    )
    try:
        EpisodeReplayBuffer(
            100,
            14,
            2,
            3,
            4,
            safety_cost_names=LEGACY_THREE_CHANNEL_SAFETY_COST_NAMES,
        )
    except ValueError as exc:
        assert "unsupported legacy three-channel" in str(exc)
    else:
        raise AssertionError("Replay accepted the legacy three-channel schema")
    try:
        EpisodeReplayBuffer(
            100,
            14,
            2,
            3,
            4,
            safety_cost_names=tuple(reversed(safety_cost_names)),
        )
    except ValueError as exc:
        assert "in this exact order" in str(exc)
    else:
        raise AssertionError("Replay accepted reordered two-channel names")
    replay = EpisodeReplayBuffer(
        100,
        14,
        2,
        3,
        4,
        safety_cost_names=safety_cost_names,
        seed=1,
    )
    rng = np.random.default_rng(1)
    observations = rng.uniform(-1, 1, size=(9, 14)).astype(np.float32)
    actions = rng.uniform(-1, 1, size=(8, 2)).astype(np.float32)
    rewards = (np.arange(8, dtype=np.float32) / 10.0).astype(np.float32)
    terminated = np.zeros(8, dtype=bool)
    terminated[-1] = True
    safety_cost = np.stack(
        (rewards, rewards + 1.0), axis=-1
    ).astype(np.float32)

    try:
        replay.add_episode(observations, actions, rewards, terminated)
    except ValueError as exc:
        assert "safety_cost is required" in str(exc)
    else:
        raise AssertionError("Replay accepted an episode without safety_cost")

    for invalid_cost, expected_error in (
        (np.zeros((8, 1), dtype=np.float32), ValueError),
        (
            np.full(
                (8, len(safety_cost_names)),
                np.nan,
                dtype=np.float32,
            ),
            FloatingPointError,
        ),
        (
            -np.ones(
                (8, len(safety_cost_names)),
                dtype=np.float32,
            ),
            ValueError,
        ),
        (
            np.zeros(
                (8, len(safety_cost_names)),
                dtype=np.float64,
            ),
            TypeError,
        ),
    ):
        try:
            replay.add_episode(
                observations,
                actions,
                rewards,
                terminated,
                safety_cost=invalid_cost,
            )
        except expected_error:
            pass
        else:
            raise AssertionError("Replay accepted invalid safety_cost data")

    try:
        replay.add_episode(
            observations,
            actions,
            rewards,
            terminated,
            safety_cost=np.zeros((8, 3), dtype=np.float32),
        )
    except ValueError as exc:
        assert "unsupported legacy three-channel" in str(exc)
        assert "expected (8, 2)" in str(exc)
    else:
        raise AssertionError("Replay accepted legacy three-channel transition data")

    replay.add_episode(
        observations,
        actions,
        rewards,
        terminated,
        safety_cost=safety_cost,
    )
    batch = replay.sample(device)
    assert len(batch) == 5
    (
        sampled_observations,
        sampled_actions,
        sampled_rewards,
        sampled_terminated,
        sampled_safety_cost,
    ) = batch
    assert sampled_observations.shape == (4, 4, 14)
    assert sampled_actions.shape == (3, 4, 2)
    assert sampled_rewards.shape == (3, 4, 1)
    assert sampled_terminated.shape == (3, 4, 1)
    assert sampled_safety_cost.shape == (3, 4, 2)
    assert sampled_safety_cost.dtype == torch.float32
    assert torch.all(torch.isfinite(sampled_safety_cost))
    assert torch.all(sampled_safety_cost >= 0.0)
    torch.testing.assert_close(
        sampled_safety_cost[..., 0], sampled_rewards[..., 0]
    )
    torch.testing.assert_close(
        sampled_safety_cost[..., 1], sampled_rewards[..., 0] + 1.0
    )

    # A validation sample uses an independent RNG and must leave the next
    # training sample exactly unchanged.
    replay_before_diagnostics = copy.deepcopy(replay.state_dict())
    diagnostic_batch = replay.sample_diagnostics(device, seed=777)
    assert len(diagnostic_batch) == 5
    assert (
        replay.state_dict()["rng_state"]
        == replay_before_diagnostics["rng_state"]
    )
    replay_without_diagnostics = EpisodeReplayBuffer(
        100,
        14,
        2,
        3,
        4,
        safety_cost_names=safety_cost_names,
        seed=123,
    )
    replay_without_diagnostics.load_state_dict(replay_before_diagnostics)
    for actual, expected in zip(
        replay.sample(device),
        replay_without_diagnostics.sample(device),
    ):
        torch.testing.assert_close(actual, expected)

    replay_state = replay.state_dict()
    assert replay_state["safety_cost_schema_version"] == SAFETY_COST_SCHEMA_VERSION
    assert replay_state["safety_cost_names"] == safety_cost_names
    assert replay_state["safety_cost_dim"] == 2
    stored_cost = replay_state["episodes"][0]["safety_cost"]
    assert stored_cost.shape == (8, 2)
    assert stored_cost.dtype == np.float32
    restored_replay = EpisodeReplayBuffer(
        100,
        14,
        2,
        3,
        4,
        safety_cost_names=safety_cost_names,
        seed=999,
    )
    restored_replay.load_state_dict(replay_state)
    for original_tensor, restored_tensor in zip(
        replay.sample(device), restored_replay.sample(device)
    ):
        torch.testing.assert_close(original_tensor, restored_tensor)

    legacy_state = copy.deepcopy(replay_state)
    legacy_state.pop("safety_cost_names")
    try:
        restored_replay.load_state_dict(legacy_state)
    except ValueError as exc:
        assert "predates the required safety_cost schema" in str(exc)
    else:
        raise AssertionError("Replay accepted legacy data without safety_cost")

    versionless_state = copy.deepcopy(replay_state)
    versionless_state.pop("safety_cost_schema_version")
    try:
        restored_replay.load_state_dict(versionless_state)
    except ValueError as exc:
        assert "predates safety-cost schema version" in str(exc)
    else:
        raise AssertionError("Replay accepted state without a schema version")

    reordered_state = copy.deepcopy(replay_state)
    reordered_state["safety_cost_names"] = tuple(
        reversed(safety_cost_names)
    )
    try:
        restored_replay.load_state_dict(reordered_state)
    except ValueError as exc:
        assert "in this exact order" in str(exc)
    else:
        raise AssertionError("Replay accepted reordered safety-cost channels")

    legacy_three_channel_state = copy.deepcopy(replay_state)
    legacy_three_channel_state["safety_cost_names"] = (
        LEGACY_THREE_CHANNEL_SAFETY_COST_NAMES
    )
    legacy_three_channel_state["safety_cost_dim"] = 3
    for episode in legacy_three_channel_state["episodes"]:
        episode["safety_cost"] = np.pad(
            episode["safety_cost"],
            ((0, 0), (0, 1)),
        ).astype(np.float32)
    try:
        restored_replay.load_state_dict(legacy_three_channel_state)
    except ValueError as exc:
        assert "unsupported legacy three-channel" in str(exc)
    else:
        raise AssertionError("Replay restored legacy three-channel transition data")

    # The model exposes the exact canonical ordering through two explicit
    # scalar branches and supports both batch and time-major latent tensors.
    assert agent.model.safety_cost_names == tuple(ENV_SAFETY_COST_NAMES)
    for latent_shape in ((5, 16), (3, 5, 16)):
        latent_input = torch.randn(*latent_shape)
        action_input = torch.randn(*latent_shape[:-1], 2)
        transformed_prediction = agent.model.safety_transformed(
            latent_input, action_input
        )
        decoded_prediction = agent.model.safety(latent_input, action_input)
        expected_shape = (*latent_shape[:-1], 2)
        assert transformed_prediction.shape == expected_shape
        assert decoded_prediction.shape == expected_shape
        assert transformed_prediction.dtype == latent_input.dtype
        assert decoded_prediction.dtype == latent_input.dtype
        assert torch.isfinite(transformed_prediction).all()
        assert torch.isfinite(decoded_prediction).all()
        assert torch.all(decoded_prediction >= 0.0)

    branch_state = {
        "curvature": copy.deepcopy(
            agent.model.safety_curvature_head.state_dict()
        ),
        "translation": copy.deepcopy(
            agent.model.safety_translation_error_head.state_dict()
        ),
    }
    with torch.no_grad():
        agent.model.safety_curvature_head.weight.zero_()
        agent.model.safety_curvature_head.bias.fill_(1.0)
        agent.model.safety_translation_error_head.weight.zero_()
        agent.model.safety_translation_error_head.bias.fill_(2.0)
    ordered_prediction = agent.model.safety_transformed(
        torch.zeros(2, 16), torch.zeros(2, 2)
    )
    torch.testing.assert_close(
        ordered_prediction,
        torch.tensor([[1.0, 2.0], [1.0, 2.0]]),
    )
    agent.model.safety_curvature_head.load_state_dict(
        branch_state["curvature"]
    )
    agent.model.safety_translation_error_head.load_state_dict(
        branch_state["translation"]
    )

    original_cost = torch.tensor(
        [[0.0, 0.0], [0.1, 1.0], [0.35, 4.5]],
        dtype=torch.float32,
    )
    transformed_target = agent.model.transform_safety_targets(original_cost)
    expected_target = torch.log1p(
        original_cost / torch.tensor([0.1, 1.0])
    )
    assert torch.isfinite(transformed_target).all()
    torch.testing.assert_close(transformed_target, expected_target)
    torch.testing.assert_close(
        agent.model.decode_safety_transformed(transformed_target),
        original_cost,
    )
    # The translation-error channel is intentionally not clipped at one.
    assert transformed_target[-1, 1] > np.log(2.0)

    # Batch diagnostics preserve channel order, exact target>0 rare-event
    # semantics, and the documented low/medium/high curvature partitions.
    diagnostic_target = torch.tensor(
        [
            [0.01, 0.0],
            [0.04, 0.0],
            [0.05, 0.2],
            [0.09, 0.0],
            [0.10, 0.5],
            [0.20, 0.0],
        ],
        dtype=torch.float32,
    )
    diagnostic_prediction = torch.tensor(
        [
            [0.02, 0.1],
            [0.05, 0.2],
            [0.07, 0.3],
            [0.10, 0.4],
            [0.13, 0.7],
            [0.24, 0.6],
        ],
        dtype=torch.float32,
    )
    diagnostic_metrics = safety_batch_diagnostics(
        agent.model.transform_safety_targets(diagnostic_prediction),
        diagnostic_target,
        agent.model,
        curvature_boundaries=(0.05, 0.10),
        safety_cost_names=safety_cost_names,
    )
    assert all(
        torch.isfinite(value).all()
        for value in diagnostic_metrics.values()
    )
    assert int(
        diagnostic_metrics["safety_translation_error_positive_count"]
    ) == 2
    np.testing.assert_allclose(
        float(
            diagnostic_metrics[
                "safety_translation_error_positive_fraction"
            ]
        ),
        2.0 / 6.0,
    )
    np.testing.assert_allclose(
        float(
            diagnostic_metrics["safety_translation_error_zero_fraction"]
        ),
        4.0 / 6.0,
    )
    np.testing.assert_allclose(
        float(
            diagnostic_metrics[
                "safety_translation_error_positive_mae"
            ]
        ),
        0.15,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        float(
            diagnostic_metrics[
                "safety_translation_error_zero_mae"
            ]
        ),
        0.325,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        float(
            diagnostic_metrics[
                "safety_translation_error_positive_pred_mean"
            ]
        ),
        0.5,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        float(
            diagnostic_metrics[
                "safety_translation_error_positive_target_mean"
            ]
        ),
        0.35,
        rtol=1e-6,
    )
    for curvature_group in ("low", "medium", "high"):
        assert int(
            diagnostic_metrics[
                f"safety_curvature_{curvature_group}_count"
            ]
        ) == 2

    no_positive_target = diagnostic_target.clone()
    no_positive_target[:, 1].zero_()
    no_positive_metrics = safety_batch_diagnostics(
        agent.model.transform_safety_targets(diagnostic_prediction),
        no_positive_target,
        agent.model,
        safety_cost_names=safety_cost_names,
    )
    assert int(
        no_positive_metrics["safety_translation_error_positive_count"]
    ) == 0
    assert torch.isnan(
        no_positive_metrics["safety_translation_error_positive_mae"]
    )
    assert torch.isnan(
        no_positive_metrics[
            "safety_translation_error_positive_pred_mean"
        ]
    )

    for dtype, tiny_value in (
        (torch.float16, 1.0e-5),
        (torch.float32, 1.0e-9),
        (torch.float64, 1.0e-20),
    ):
        tiny_cost = torch.full((1, 2), tiny_value, dtype=dtype)
        tiny_scale = torch.tensor([0.1, 1.0], dtype=dtype)
        tiny_transformed = agent.model.transform_safety_targets(tiny_cost)
        tiny_reference = torch.log1p(tiny_cost / tiny_scale)
        assert torch.all(tiny_transformed > 0.0)
        torch.testing.assert_close(tiny_transformed, tiny_reference)
    extreme_decoded = agent.model.decode_safety_transformed(
        torch.tensor(
            [[-100.0, 1.0e6], [float("nan"), float("inf")]],
            dtype=torch.float32,
        )
    )
    assert torch.isfinite(extreme_decoded).all()
    assert torch.all(extreme_decoded >= 0.0)
    for dtype in (torch.float16, torch.float32, torch.float64):
        dtype_max = torch.finfo(dtype).max
        extreme_target = torch.full((2, 2), dtype_max, dtype=dtype)
        extreme_transformed_target = agent.model.transform_safety_targets(
            extreme_target
        )
        assert torch.isfinite(extreme_transformed_target).all()
        dtype_decoded = agent.model.decode_safety_transformed(
            torch.tensor(
                [[-1.0, dtype_max], [float("nan"), float("inf")]],
                dtype=dtype,
            )
        )
        assert torch.isfinite(dtype_decoded).all()
        assert torch.all(dtype_decoded >= 0.0)

    safety_parameter_ids = {
        id(parameter) for parameter in agent.model.safety_head_parameters()
    }
    assert agent.model_optimizer.param_groups[-1]["name"] == "safety"
    assert {
        id(parameter)
        for parameter in agent.model_optimizer.param_groups[-1]["params"]
    } == safety_parameter_ids
    assert safety_parameter_ids.isdisjoint(
        id(parameter) for parameter in agent.model.policy.parameters()
    )

    horizon, batch_size = 3, 4
    fixed_observations = torch.linspace(
        -1.0,
        1.0,
        (horizon + 1) * batch_size * 14,
        dtype=torch.float32,
    ).reshape(horizon + 1, batch_size, 14)
    fixed_actions = torch.linspace(
        -0.8,
        0.8,
        horizon * batch_size * 2,
        dtype=torch.float32,
    ).reshape(horizon, batch_size, 2)
    fixed_rewards = torch.linspace(
        -0.2,
        0.9,
        horizon * batch_size,
        dtype=torch.float32,
    ).reshape(horizon, batch_size, 1)
    fixed_terminated = torch.zeros(horizon, batch_size, 1)
    fixed_terminated[-1, -1, 0] = 1.0
    temporal_id = torch.arange(
        horizon * batch_size, dtype=torch.float32
    ).reshape(horizon, batch_size)
    fixed_safety_cost = torch.stack(
        (
            0.01 + temporal_id * 0.013,
            0.2 + temporal_id * 0.37,
        ),
        dim=-1,
    )
    fixed_batch = (
        fixed_observations,
        fixed_actions,
        fixed_rewards,
        fixed_terminated,
        fixed_safety_cost,
    )

    # Commit 4.6C: independent offline transitions supervise only the Safety
    # Head. Their loss uses the same target transform and channel coefficients
    # as main replay Safety training, but has no temporal rho weighting.
    auxiliary_batch = {
        "observation": (
            fixed_observations[0].detach().cpu().numpy().astype(
                np.float32,
                copy=True,
            )
        ),
        "action": (
            fixed_actions[0].detach().cpu().numpy().astype(
                np.float32,
                copy=True,
            )
        ),
        "safety_cost": (
            fixed_safety_cost[0].detach().cpu().numpy().astype(
                np.float32,
                copy=True,
            )
        ),
    }
    tiny_clip_config = copy.deepcopy(config)
    tiny_clip_config["grad_clip_norm"] = 1.0e-5
    tiny_clip_config["gradient_interval"] = 1
    tiny_clip_config["validation_interval"] = 1000
    torch.manual_seed(4603)
    auxiliary_baseline_agent = TDMPC2Agent(
        14,
        2,
        tiny_clip_config,
        episode_length=200,
        device=device,
    )
    torch.manual_seed(4603)
    auxiliary_agent = TDMPC2Agent(
        14,
        2,
        tiny_clip_config,
        episode_length=200,
        device=device,
    )

    direct_auxiliary_info = (
        auxiliary_agent._compute_auxiliary_safety_loss(auxiliary_batch)
    )
    with torch.no_grad():
        auxiliary_observation_tensor = torch.as_tensor(
            auxiliary_batch["observation"],
            device=device,
        )
        auxiliary_action_tensor = torch.as_tensor(
            auxiliary_batch["action"],
            device=device,
        )
        auxiliary_cost_tensor = torch.as_tensor(
            auxiliary_batch["safety_cost"],
            device=device,
        )
        detached_auxiliary_latent = auxiliary_agent.model.encode(
            auxiliary_observation_tensor
        ).detach()
        assert not detached_auxiliary_latent.requires_grad
        expected_auxiliary_prediction = (
            auxiliary_agent.model.safety_transformed(
                detached_auxiliary_latent,
                auxiliary_action_tensor,
            )
        )
        expected_auxiliary_target = (
            auxiliary_agent.model.transform_safety_targets(
                auxiliary_cost_tensor
            )
        )
        expected_auxiliary_channel_losses = F.smooth_l1_loss(
            expected_auxiliary_prediction,
            expected_auxiliary_target,
            reduction="none",
        ).mean(dim=0)
        expected_auxiliary_loss = (
            auxiliary_agent.safety_curvature_loss_coef
            * expected_auxiliary_channel_losses[0]
            + auxiliary_agent.safety_translation_error_loss_coef
            * expected_auxiliary_channel_losses[1]
        )
    torch.testing.assert_close(
        direct_auxiliary_info["safety_aux_curvature_loss"],
        expected_auxiliary_channel_losses[0],
    )
    torch.testing.assert_close(
        direct_auxiliary_info["safety_aux_translation_error_loss"],
        expected_auxiliary_channel_losses[1],
    )
    torch.testing.assert_close(
        direct_auxiliary_info["safety_aux_loss"],
        expected_auxiliary_loss,
    )
    assert float(direct_auxiliary_info["safety_aux_batch_size"]) == batch_size

    invalid_auxiliary_batch = copy.deepcopy(auxiliary_batch)
    invalid_auxiliary_batch["observation"] = invalid_auxiliary_batch[
        "observation"
    ][:, :-1]
    try:
        auxiliary_agent._compute_auxiliary_safety_loss(
            invalid_auxiliary_batch
        )
    except ValueError as exc:
        assert "(B, 14)" in str(exc)
    else:
        raise AssertionError(
            "Auxiliary Safety loss accepted a wrong observation shape"
        )

    def cloned_main_replay() -> EpisodeReplayBuffer:
        clone = EpisodeReplayBuffer(
            100,
            14,
            2,
            3,
            4,
            safety_cost_names=safety_cost_names,
            seed=999,
        )
        clone.load_state_dict(copy.deepcopy(replay.state_dict()))
        return clone

    baseline_main_replay = cloned_main_replay()
    auxiliary_main_replay = cloned_main_replay()
    safety_parameters_before_auxiliary = {
        name: parameter.detach().clone()
        for name, parameter in auxiliary_agent.model.named_parameters()
        if name.startswith("safety_")
    }
    with patch.object(
        auxiliary_baseline_agent,
        "_compute_auxiliary_safety_loss",
        side_effect=AssertionError(
            "Unscheduled update evaluated auxiliary Safety data"
        ),
    ) as unscheduled_auxiliary_spy:
        torch.manual_seed(4604)
        auxiliary_baseline_metrics = auxiliary_baseline_agent.update(
            baseline_main_replay
        )
    assert unscheduled_auxiliary_spy.call_count == 0
    for key in (
        "safety_aux_loss",
        "safety_aux_curvature_loss",
        "safety_aux_translation_error_loss",
        "safety_aux_batch_size",
        "aux_grad_norm_safety_head",
    ):
        assert np.isnan(auxiliary_baseline_metrics[key])

    torch.manual_seed(4604)
    auxiliary_metrics = auxiliary_agent.update(
        auxiliary_main_replay,
        safety_aux_batch=auxiliary_batch,
        safety_aux_loss_coef=1.0,
    )
    np.testing.assert_allclose(
        auxiliary_metrics["total_loss"],
        auxiliary_baseline_metrics["total_loss"]
        + auxiliary_metrics["safety_aux_loss"],
        rtol=1.0e-6,
    )
    for key in (
        "consistency_loss",
        "reward_loss",
        "value_loss",
        "termination_loss",
        "policy_loss",
        "policy_entropy",
    ):
        np.testing.assert_allclose(
            auxiliary_metrics[key],
            auxiliary_baseline_metrics[key],
            rtol=0.0,
            atol=0.0,
        )
    assert auxiliary_metrics["safety_aux_batch_size"] == batch_size
    np.testing.assert_allclose(
        auxiliary_metrics["safety_aux_curvature_loss"],
        float(expected_auxiliary_channel_losses[0]),
        rtol=1.0e-6,
    )
    np.testing.assert_allclose(
        auxiliary_metrics["safety_aux_translation_error_loss"],
        float(expected_auxiliary_channel_losses[1]),
        rtol=1.0e-6,
    )
    for module_name in (
        "safety_head",
        "safety_trunk",
        "safety_curvature_branch",
        "safety_translation_error_branch",
    ):
        assert np.isfinite(
            auxiliary_metrics[f"aux_grad_norm_{module_name}"]
        )
        assert auxiliary_metrics[f"aux_grad_norm_{module_name}"] > 0.0
        assert np.isfinite(
            auxiliary_metrics[f"main_grad_norm_{module_name}"]
        )
        assert np.isfinite(
            auxiliary_metrics[f"combined_grad_norm_{module_name}"]
        )
    for forbidden_module in (
        "encoder",
        "dynamics",
        "reward_head",
        "termination_head",
        "q_ensemble",
        "policy",
    ):
        assert auxiliary_metrics[
            f"aux_grad_norm_{forbidden_module}"
        ] == 0.0
        np.testing.assert_allclose(
            auxiliary_metrics[
                f"combined_grad_norm_{forbidden_module}"
            ],
            auxiliary_metrics[f"main_grad_norm_{forbidden_module}"],
            rtol=0.0,
            atol=0.0,
        )
    assert (
        auxiliary_metrics["model_grad_norm"]
        > tiny_clip_config["grad_clip_norm"]
    )
    assert (
        auxiliary_metrics["aux_grad_norm_safety_head"]
        > auxiliary_metrics["aux_grad_norm_safety_head_clipped"]
    )
    assert (
        auxiliary_metrics["aux_grad_norm_safety_head_clipped"]
        <= tiny_clip_config["grad_clip_norm"] + 1.0e-7
    )

    auxiliary_parameters = dict(auxiliary_agent.model.named_parameters())
    for name, baseline_parameter in (
        auxiliary_baseline_agent.model.named_parameters()
    ):
        if not name.startswith("safety_"):
            torch.testing.assert_close(
                auxiliary_parameters[name],
                baseline_parameter,
                rtol=0.0,
                atol=0.0,
            )
    for module_prefix in (
        "safety_trunk",
        "safety_curvature_head",
        "safety_translation_error_head",
    ):
        assert any(
            not torch.equal(
                safety_parameters_before_auxiliary[name],
                parameter,
            )
            for name, parameter in auxiliary_parameters.items()
            if name.startswith(module_prefix)
        )

    _assert_nested_state_equal(
        baseline_main_replay.state_dict()["rng_state"],
        auxiliary_main_replay.state_dict()["rng_state"],
    )
    for baseline_sample, auxiliary_sample in zip(
        baseline_main_replay.sample(device),
        auxiliary_main_replay.sample(device),
    ):
        torch.testing.assert_close(
            baseline_sample,
            auxiliary_sample,
            rtol=0.0,
            atol=0.0,
        )
    torch.manual_seed(4605)
    baseline_planning_action = auxiliary_baseline_agent.act(
        observations[0],
        first_step=True,
        eval_mode=True,
    )
    torch.manual_seed(4605)
    auxiliary_planning_action = auxiliary_agent.act(
        observations[0],
        first_step=True,
        eval_mode=True,
    )
    np.testing.assert_array_equal(
        baseline_planning_action,
        auxiliary_planning_action,
    )

    # Disable all baseline model coefficients so every encoder/dynamics/head
    # gradient below is attributable to the Safety auxiliary objective.
    safety_only_config = copy.deepcopy(config)
    for coefficient in (
        "consistency_coef",
        "reward_coef",
        "value_coef",
        "termination_coef",
    ):
        safety_only_config[coefficient] = 0.0
    safety_only_config["safety_loss_coef"] = 1.5
    safety_only_config["safety_curvature_loss_coef"] = 2.0
    safety_only_config["safety_translation_error_loss_coef"] = 3.0
    safety_only_config["validation_interval"] = 1
    safety_only_config["gradient_interval"] = 1
    safety_only_config["grad_clip_norm"] = 1.0e6
    torch.manual_seed(11)
    safety_agent = TDMPC2Agent(
        14,
        2,
        safety_only_config,
        episode_length=200,
        device=device,
    )
    with torch.no_grad():
        expected_rollout = [safety_agent.model.encode(fixed_observations[0])]
        for step in range(horizon):
            expected_rollout.append(
                safety_agent.model.next(
                    expected_rollout[-1],
                    fixed_actions[step],
                )
            )
        expected_rollout_tensor = torch.stack(expected_rollout)
        prediction_before_update = safety_agent.model.safety_transformed(
            expected_rollout_tensor[:-1],
            fixed_actions,
        )
        target_transformed = safety_agent.model.transform_safety_targets(
            fixed_safety_cost
        )
        weights = torch.pow(
            torch.tensor(float(config["rho"])),
            torch.arange(horizon),
        )
        expected_channel_losses = (
            F.smooth_l1_loss(
                prediction_before_update,
                target_transformed,
                reduction="none",
            ).mean(dim=1)
            * weights.unsqueeze(-1)
        ).sum(dim=0) / horizon
        expected_combined_safety_loss = (
            safety_only_config["safety_curvature_loss_coef"]
            * expected_channel_losses[0]
            + safety_only_config["safety_translation_error_loss_coef"]
            * expected_channel_losses[1]
        )
        expected_decoded_means = (
            safety_agent.model.decode_safety_transformed(
                prediction_before_update
            ).mean(dim=(0, 1))
        )

    validation_cost = fixed_safety_cost.clone()
    validation_zero_mask = (
        torch.arange(horizon * batch_size).reshape(horizon, batch_size) % 2
        == 0
    )
    validation_cost[..., 1][validation_zero_mask] = 0.0
    validation_batch = (
        fixed_observations,
        fixed_actions,
        fixed_rewards,
        fixed_terminated,
        validation_cost,
    )
    validation_parameters_before = {
        name: parameter.detach().clone()
        for name, parameter in safety_agent.model.named_parameters()
    }
    validation_model_optimizer_before = copy.deepcopy(
        safety_agent.model_optimizer.state_dict()
    )
    validation_policy_optimizer_before = copy.deepcopy(
        safety_agent.policy_optimizer.state_dict()
    )
    validation_rng_before = torch.get_rng_state().clone()
    safety_agent.model.train(True)
    direct_validation_metrics = safety_agent.safety_validation_metrics(
        validation_batch
    )
    repeated_validation_metrics = safety_agent.safety_validation_metrics(
        validation_batch
    )
    assert safety_agent.model.training
    assert direct_validation_metrics.keys() == repeated_validation_metrics.keys()
    for name in direct_validation_metrics:
        torch.testing.assert_close(
            direct_validation_metrics[name],
            repeated_validation_metrics[name],
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )
    _assert_parameters_unchanged(
        validation_parameters_before,
        safety_agent.model,
    )
    _assert_nested_state_equal(
        validation_model_optimizer_before,
        safety_agent.model_optimizer.state_dict(),
    )
    _assert_nested_state_equal(
        validation_policy_optimizer_before,
        safety_agent.policy_optimizer.state_dict(),
    )
    torch.testing.assert_close(
        validation_rng_before,
        torch.get_rng_state(),
        rtol=0.0,
        atol=0.0,
    )
    assert all(
        parameter.grad is None
        for parameter in safety_agent.model.parameters()
    )
    for key in (
        "val_safety_loss",
        "val_safety_curvature_loss",
        "val_safety_translation_error_loss",
        "val_safety_curvature_mae",
        "val_safety_translation_error_mae",
        "val_safety_translation_error_positive_mae",
        "val_safety_translation_error_zero_mae",
    ):
        assert np.isfinite(float(direct_validation_metrics[key]))
    assert (
        int(
            direct_validation_metrics[
                "val_safety_translation_error_positive_count"
            ]
        )
        == horizon * batch_size // 2
    )
    np.testing.assert_allclose(
        float(direct_validation_metrics["val_safety_target_curvature_mean"]),
        float(validation_cost[..., 0].mean()),
        rtol=1.0e-7,
    )
    np.testing.assert_allclose(
        float(
            direct_validation_metrics[
                "val_safety_target_translation_error_mean"
            ]
        ),
        float(validation_cost[..., 1].mean()),
        rtol=1.0e-7,
    )
    np.testing.assert_allclose(
        float(direct_validation_metrics["val_safety_target_curvature_max"]),
        float(validation_cost[..., 0].max()),
        rtol=1.0e-7,
    )
    np.testing.assert_allclose(
        float(
            direct_validation_metrics[
                "val_safety_target_translation_error_max"
            ]
        ),
        float(validation_cost[..., 1].max()),
        rtol=1.0e-7,
    )
    safety_agent.model.train(False)

    captured_alignment: dict[str, torch.Tensor] = {}
    original_compute_safety_loss = safety_agent._compute_safety_loss

    def capture_safety_alignment(
        rollout_latent: torch.Tensor,
        action_tensor: torch.Tensor,
        cost_tensor: torch.Tensor,
        weight_tensor: torch.Tensor,
    ):
        captured_alignment["latent"] = rollout_latent.detach().clone()
        captured_alignment["actions"] = action_tensor.detach().clone()
        captured_alignment["cost"] = cost_tensor.detach().clone()
        return original_compute_safety_loss(
            rollout_latent,
            action_tensor,
            cost_tensor,
            weight_tensor,
        )

    safety_agent._compute_safety_loss = capture_safety_alignment
    safety_parameters_before = {
        name: parameter.detach().clone()
        for name, parameter in safety_agent.model.named_parameters()
        if name.startswith("safety_")
    }
    torch.manual_seed(29)
    metrics = safety_agent.update(FixedReplay(fixed_batch))
    assert not any(np.isinf(value) for value in metrics.values())
    for original_metric in (
        "consistency_loss",
        "reward_loss",
        "value_loss",
        "termination_loss",
        "policy_loss",
        "total_loss",
    ):
        assert np.isfinite(metrics[original_metric])
    torch.testing.assert_close(
        captured_alignment["latent"],
        expected_rollout_tensor[:-1],
    )
    torch.testing.assert_close(captured_alignment["actions"], fixed_actions)
    torch.testing.assert_close(captured_alignment["cost"], fixed_safety_cost)
    assert not torch.allclose(
        captured_alignment["latent"],
        expected_rollout_tensor[1:],
    )
    np.testing.assert_allclose(
        metrics["safety_curvature_loss"],
        float(expected_channel_losses[0]),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        metrics["safety_translation_error_loss"],
        float(expected_channel_losses[1]),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        metrics["safety_loss"],
        float(expected_combined_safety_loss),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        metrics["total_loss"],
        safety_only_config["safety_loss_coef"]
        * metrics["safety_loss"],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        metrics["safety_target_curvature_mean"],
        float(fixed_safety_cost[..., 0].mean()),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        metrics["safety_target_translation_error_mean"],
        float(fixed_safety_cost[..., 1].mean()),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        metrics["safety_pred_curvature_mean"],
        float(expected_decoded_means[0]),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        metrics["safety_pred_translation_error_mean"],
        float(expected_decoded_means[1]),
        rtol=1e-6,
    )
    required_safety_metrics = {
        "safety_loss",
        "safety_curvature_loss",
        "safety_translation_error_loss",
        "safety_pred_curvature_mean",
        "safety_target_curvature_mean",
        "safety_pred_curvature_max",
        "safety_target_curvature_max",
        "safety_pred_translation_error_mean",
        "safety_target_translation_error_mean",
        "safety_pred_translation_error_max",
        "safety_target_translation_error_max",
        "safety_curvature_mae_transformed",
        "safety_translation_error_mae_transformed",
        "safety_curvature_mae",
        "safety_translation_error_mae",
        "safety_translation_error_positive_count",
        "safety_translation_error_positive_fraction",
        "safety_translation_error_zero_fraction",
        "safety_translation_error_positive_mae",
        "safety_translation_error_zero_mae",
        "safety_translation_error_positive_pred_mean",
        "safety_translation_error_positive_target_mean",
    }
    assert required_safety_metrics.issubset(metrics)
    required_validation_metrics = {
        f"val_{name}" for name in required_safety_metrics
    }
    assert required_validation_metrics.issubset(metrics)
    for key in (
        "val_safety_loss",
        "val_safety_curvature_loss",
        "val_safety_translation_error_loss",
        "val_safety_curvature_mae",
        "val_safety_translation_error_mae",
    ):
        assert np.isfinite(metrics[key])
    required_gradient_metrics = {
        "safety_grad_norm_trunk",
        "safety_grad_norm_curvature_branch",
        "safety_grad_norm_translation_error_branch",
        "safety_grad_norm_encoder",
        "safety_grad_norm_dynamics",
    }
    assert required_gradient_metrics.issubset(metrics)
    assert all(
        np.isfinite(metrics[key]) and metrics[key] > 0.0
        for key in required_gradient_metrics
    )
    for shared_module in ("encoder", "dynamics"):
        total_key = f"total_grad_norm_{shared_module}"
        safety_key = f"safety_grad_norm_{shared_module}"
        assert np.isfinite(metrics[total_key])
        assert metrics[total_key] > 0.0
        np.testing.assert_allclose(
            metrics[total_key],
            metrics[safety_key],
            rtol=1e-5,
            atol=1e-7,
        )
    assert _module_has_finite_nonzero_gradient(
        safety_agent.model.safety_trunk
    )
    assert _module_has_finite_nonzero_gradient(
        safety_agent.model.safety_curvature_head
    )
    assert _module_has_finite_nonzero_gradient(
        safety_agent.model.safety_translation_error_head
    )
    assert _module_has_finite_nonzero_gradient(safety_agent.model.encoder)
    assert _module_has_finite_nonzero_gradient(safety_agent.model.dynamics)
    assert any(
        not torch.equal(
            safety_parameters_before[name],
            parameter.detach(),
        )
        for name, parameter in safety_agent.model.named_parameters()
        if name in safety_parameters_before
    )
    original_curvature_coef = safety_agent.safety_curvature_loss_coef
    safety_agent.safety_curvature_loss_coef = 0.0
    single_channel_info = safety_agent._compute_safety_loss(
        captured_alignment["latent"],
        fixed_actions,
        fixed_safety_cost,
        weights,
    )
    assert single_channel_info["safety_curvature_loss"] > 0.0
    assert single_channel_info["safety_translation_error_loss"] > 0.0
    torch.testing.assert_close(
        single_channel_info["safety_loss"],
        safety_agent.safety_translation_error_loss_coef
        * single_channel_info["safety_translation_error_loss"],
    )
    safety_agent.safety_curvature_loss_coef = original_curvature_coef
    maximum_finite_cost = torch.full_like(
        fixed_safety_cost,
        torch.finfo(fixed_safety_cost.dtype).max,
    )
    extreme_safety_info = safety_agent._compute_safety_loss(
        captured_alignment["latent"],
        fixed_actions,
        maximum_finite_cost,
        weights,
    )
    assert not any(
        torch.isinf(value).any() for value in extreme_safety_info.values()
    )
    for key in (
        "safety_loss",
        "safety_curvature_loss",
        "safety_translation_error_loss",
        "safety_curvature_mae",
        "safety_translation_error_mae",
    ):
        assert torch.isfinite(extreme_safety_info[key])

    # Planning must stay isolated even while Safety prediction is enabled.
    with ExitStack() as enabled_planning_stack:
        enabled_safety_forward_spy = enabled_planning_stack.enter_context(
            patch.object(
                safety_agent.model,
                "safety_transformed",
                side_effect=AssertionError(
                    "enabled planning called transformed Safety prediction"
                ),
            )
        )
        enabled_safety_decode_spy = enabled_planning_stack.enter_context(
            patch.object(
                safety_agent.model,
                "decode_safety_transformed",
                side_effect=AssertionError(
                    "enabled planning decoded Safety prediction"
                ),
            )
        )
        enabled_safety_spy = enabled_planning_stack.enter_context(
            patch.object(
                safety_agent.model,
                "safety",
                side_effect=AssertionError(
                    "enabled planning called Safety prediction"
                ),
            )
        )
        torch.manual_seed(37)
        safety_agent.act(observations[0], first_step=True, eval_mode=True)
        active_observation_tensor = safety_agent._tensor_observation(
            observations[0]
        )
        torch.manual_seed(38)
        safety_agent._plan(
            active_observation_tensor,
            first_step=True,
            eval_mode=True,
        )
        active_planning_latent = safety_agent.model.encode(
            active_observation_tensor
        ).repeat(2, 1)
        safety_agent._estimate_value(
            active_planning_latent,
            torch.zeros(horizon, 2, 2),
        )
        assert enabled_safety_forward_spy.call_count == 0
        assert enabled_safety_decode_spy.call_count == 0
        assert enabled_safety_spy.call_count == 0

    # With the global coefficient at zero, neither invalid targets nor arbitrary
    # Safety weights can affect baseline losses, shared updates, or planning.
    zero_config = copy.deepcopy(config)
    zero_config["safety_loss_coef"] = 0.0
    zero_config["validation_interval"] = 1
    zero_config["gradient_interval"] = 1
    torch.manual_seed(41)
    zero_agent_a = TDMPC2Agent(
        14, 2, zero_config, episode_length=200, device=device
    )
    torch.manual_seed(41)
    zero_agent_b = TDMPC2Agent(
        14, 2, zero_config, episode_length=200, device=device
    )
    # Use the exact five original world-model parameter groups as a local
    # pre-Safety optimizer reference.
    baseline_learning_rate = float(zero_config["lr"])
    zero_agent_b.model_optimizer = torch.optim.Adam(
        [
            {
                "params": zero_agent_b.model.encoder.parameters(),
                "lr": baseline_learning_rate
                * float(zero_config["enc_lr_scale"]),
            },
            {"params": zero_agent_b.model.dynamics.parameters()},
            {"params": zero_agent_b.model.reward_head.parameters()},
            {"params": zero_agent_b.model.termination_head.parameters()},
            {"params": zero_agent_b.model.q_ensemble.parameters()},
        ],
        lr=baseline_learning_rate,
    )
    with torch.no_grad():
        for parameter in zero_agent_b.model.safety_head_parameters():
            parameter.add_(3.0)
    zero_safety_before_a = {
        name: parameter.detach().clone()
        for name, parameter in zero_agent_a.model.named_parameters()
        if name.startswith("safety_")
    }
    zero_safety_before_b = {
        name: parameter.detach().clone()
        for name, parameter in zero_agent_b.model.named_parameters()
        if name.startswith("safety_")
    }
    with ExitStack() as safety_patch_stack:
        safety_forward_spy = safety_patch_stack.enter_context(
            patch.object(
            zero_agent_a.model,
            "safety_transformed",
            side_effect=AssertionError("disabled update called Safety Head"),
            )
        )
        target_transform_spy = safety_patch_stack.enter_context(
            patch.object(
            zero_agent_a.model,
            "transform_safety_targets",
            side_effect=AssertionError("disabled update transformed targets"),
            )
        )
        safety_decode_spy = safety_patch_stack.enter_context(
            patch.object(
            zero_agent_a.model,
            "decode_safety_transformed",
            side_effect=AssertionError("disabled update decoded predictions"),
            )
        )
        decoded_safety_spy = safety_patch_stack.enter_context(
            patch.object(
            zero_agent_a.model,
            "safety",
            side_effect=AssertionError("planning called Safety prediction"),
            )
        )
        torch.manual_seed(53)
        zero_metrics_a = zero_agent_a.update(FixedReplay(fixed_batch))
        torch.manual_seed(53)
        zero_metrics_b = zero_agent_b.update(
            FixedReplay(fixed_batch, poison_safety=True)
        )
        assert safety_forward_spy.call_count == 0
        assert target_transform_spy.call_count == 0
        assert safety_decode_spy.call_count == 0
        for key in (
            "safety_loss",
            "safety_curvature_loss",
            "safety_translation_error_loss",
        ):
            assert zero_metrics_a[key] == 0.0
            assert zero_metrics_b[key] == 0.0
        for key in required_safety_metrics - {
            "safety_loss",
            "safety_curvature_loss",
            "safety_translation_error_loss",
        }:
            assert np.isnan(zero_metrics_a[key])
            assert np.isnan(zero_metrics_b[key])
        assert not any(
            key.startswith("val_") for key in zero_metrics_a
        )
        expected_zero_total_loss = (
            float(zero_config["consistency_coef"])
            * zero_metrics_a["consistency_loss"]
            + float(zero_config["reward_coef"])
            * zero_metrics_a["reward_loss"]
            + float(zero_config["value_coef"])
            * zero_metrics_a["value_loss"]
            + float(zero_config["termination_coef"])
            * zero_metrics_a["termination_loss"]
        )
        np.testing.assert_allclose(
            zero_metrics_a["total_loss"],
            expected_zero_total_loss,
            rtol=1e-6,
        )
        for key in zero_metrics_a:
            if np.isnan(zero_metrics_a[key]):
                assert np.isnan(zero_metrics_b[key])
            else:
                np.testing.assert_allclose(
                    zero_metrics_a[key],
                    zero_metrics_b[key],
                    rtol=0.0,
                    atol=0.0,
                )

        shared_parameters_a = dict(zero_agent_a.model.named_parameters())
        shared_parameters_b = dict(zero_agent_b.model.named_parameters())
        for name in shared_parameters_a:
            if not name.startswith("safety_"):
                torch.testing.assert_close(
                    shared_parameters_a[name],
                    shared_parameters_b[name],
                    rtol=0.0,
                    atol=0.0,
                )
        _assert_parameters_unchanged(
            zero_safety_before_a,
            torch.nn.ModuleDict(
                {
                    "safety_trunk": zero_agent_a.model.safety_trunk,
                    "safety_curvature_head": (
                        zero_agent_a.model.safety_curvature_head
                    ),
                    "safety_translation_error_head": (
                        zero_agent_a.model.safety_translation_error_head
                    ),
                }
            ),
        )
        _assert_parameters_unchanged(
            zero_safety_before_b,
            torch.nn.ModuleDict(
                {
                    "safety_trunk": zero_agent_b.model.safety_trunk,
                    "safety_curvature_head": (
                        zero_agent_b.model.safety_curvature_head
                    ),
                    "safety_translation_error_head": (
                        zero_agent_b.model.safety_translation_error_head
                    ),
                }
            ),
        )
        assert all(
            parameter.grad is None
            for parameter in zero_agent_a.model.safety_head_parameters()
        )
        assert all(
            parameter.grad is None
            for parameter in zero_agent_b.model.safety_head_parameters()
        )

        torch.manual_seed(67)
        action_a = zero_agent_a.act(
            observations[0], first_step=True, eval_mode=True
        )
        torch.manual_seed(67)
        action_b = zero_agent_b.act(
            observations[0], first_step=True, eval_mode=True
        )
        np.testing.assert_array_equal(action_a, action_b)
        assert action_a.shape == (2,)
        assert action_a.dtype == np.float32
        assert np.all(np.isfinite(action_a))
        assert np.all(np.abs(action_a) <= 1.0)

        observation_tensor = zero_agent_a._tensor_observation(observations[0])
        torch.manual_seed(71)
        zero_agent_a._plan(
            observation_tensor,
            first_step=True,
            eval_mode=True,
        )
        planning_latent = zero_agent_a.model.encode(
            observation_tensor
        ).repeat(2, 1)
        zero_agent_a._estimate_value(
            planning_latent,
            torch.zeros(horizon, 2, 2),
        )
        assert safety_forward_spy.call_count == 0
        assert target_transform_spy.call_count == 0
        assert safety_decode_spy.call_count == 0
        assert decoded_safety_spy.call_count == 0

    clone = TDMPC2Agent(
        14, 2, safety_only_config, episode_length=200, device=device
    )
    clone.load_state_dict(safety_agent.state_dict(), load_optimizers=True)
    assert clone.update_count == safety_agent.update_count
    assert len(clone.model_optimizer.param_groups) == 6
    assert clone.model_optimizer.param_groups[-1]["name"] == "safety"
    for restored, expected in zip(
        clone.model.safety_head_parameters(),
        safety_agent.model.safety_head_parameters(),
    ):
        torch.testing.assert_close(restored, expected)

    old_agent_state = copy.deepcopy(safety_agent.state_dict())
    old_agent_state.pop("safety_model_schema_version")
    try:
        clone.load_state_dict(old_agent_state, load_optimizers=False)
    except ValueError as exc:
        assert "predates" in str(exc)
        assert "Safety Head" in str(exc)
    else:
        raise AssertionError("Agent accepted a checkpoint without Safety Head schema")

    old_optimizer_state = copy.deepcopy(safety_agent.state_dict())
    old_optimizer_state["model_optimizer"]["param_groups"].pop()
    old_optimizer_agent = TDMPC2Agent(
        14, 2, safety_only_config, episode_length=200, device=device
    )
    try:
        old_optimizer_agent.load_state_dict(
            old_optimizer_state,
            load_optimizers=True,
        )
    except ValueError as exc:
        assert "parameter group" in str(exc)
    else:
        raise AssertionError("Agent silently loaded an old five-group optimizer")

    with TemporaryDirectory(prefix="steve-tdmpc2-checkpoint-", dir="/tmp") as temp_dir:
        checkpoint_path = Path(temp_dir) / "schema_v3.pt"
        checkpoint_config = copy.deepcopy(full_config)
        checkpoint_config["safety"].update(
            {
                "loss_coef": safety_only_config["safety_loss_coef"],
                "curvature_loss_coef": (
                    safety_only_config["safety_curvature_loss_coef"]
                ),
                "translation_error_loss_coef": (
                    safety_only_config[
                        "safety_translation_error_loss_coef"
                    ]
                ),
            }
        )
        for key in (
            "consistency_coef",
            "reward_coef",
            "value_coef",
            "termination_coef",
            "grad_clip_norm",
        ):
            checkpoint_config["training"][key] = safety_only_config[key]
        for key in (
            "validation_interval",
            "gradient_interval",
            "curvature_low_max_mm_inv",
            "curvature_medium_max_mm_inv",
            "curvature_high_max_mm_inv",
        ):
            checkpoint_config["diagnostics"][key] = safety_only_config[key]
        checkpoint_agent_config = build_agent_config(checkpoint_config)
        for key, value in safety_only_config.items():
            assert checkpoint_agent_config[key] == value
        save_checkpoint(
            path=checkpoint_path,
            config=checkpoint_config,
            agent=safety_agent,
            replay=replay,
            total_env_steps=8,
            episode_index=1,
            success_count=0,
            include_replay=True,
        )
        checkpoint = load_torch_checkpoint(
            checkpoint_path,
            map_location=device,
        )
        assert SAFETY_AUXILIARY_CHECKPOINT_KEY not in checkpoint
        validate_checkpoint_schema(
            checkpoint,
            config=checkpoint_config,
            source="Smoke-test checkpoint",
        )
        disabled_aux_variant = copy.deepcopy(checkpoint_config)
        disabled_aux_variant["safety_aux"].update(
            {
                "loss_coef": 4.0,
                "batch_size": 17,
                "update_interval": 3,
                "sampling_mode": "uniform",
                "translation_fraction": 0.75,
                "curvature_fraction": 0.0,
                "sample_with_replacement": False,
            }
        )
        disabled_aux_variant["diagnostics"][
            "curvature_high_max_mm_inv"
        ] = 0.30
        validate_checkpoint_schema(
            checkpoint,
            config=disabled_aux_variant,
            source="Disabled Safety auxiliary config variant",
        )
        forbidden_disabled_aux_state = copy.deepcopy(checkpoint)
        forbidden_disabled_aux_state[SAFETY_AUXILIARY_CHECKPOINT_KEY] = None
        try:
            validate_checkpoint_schema(
                forbidden_disabled_aux_state,
                source="Disabled checkpoint with auxiliary state key",
            )
        except ValueError as exc:
            assert "safety_aux.enabled=false" in str(exc)
        else:
            raise AssertionError(
                "Disabled checkpoint accepted a safety_auxiliary state key"
            )
        legacy_collection_checkpoint = copy.deepcopy(checkpoint)
        legacy_collection_checkpoint["config"]["safety_aux"] = {
            "enabled": True,
            "capacity": 100000,
            "translation_fraction": 0.5,
            "curvature_fraction": 0.5,
        }
        legacy_collection_checkpoint["safety_aux_replay"] = {
            "schema_version": 2,
        }
        try:
            validate_checkpoint_schema(
                legacy_collection_checkpoint,
                source="Legacy enabled collection-only checkpoint",
            )
        except ValueError as exc:
            assert "legacy four-field" in str(exc)
            assert "cannot be enabled" in str(exc)
        else:
            raise AssertionError(
                "Checkpoint validation accepted the legacy enabled collection "
                "schema"
            )
        legacy_collection_key_checkpoint = copy.deepcopy(checkpoint)
        legacy_collection_key_checkpoint["safety_aux_replay"] = {
            "schema_version": 2,
        }
        try:
            validate_checkpoint_schema(
                legacy_collection_key_checkpoint,
                source="Legacy collection-only state checkpoint",
            )
        except ValueError as exc:
            assert "collection-only safety_aux_replay" in str(exc)
        else:
            raise AssertionError(
                "Checkpoint validation accepted a legacy collection-only state"
            )
        commit4_checkpoint = copy.deepcopy(checkpoint)
        commit4_checkpoint["config"].pop("diagnostics")
        commit4_checkpoint["config"].pop("safety_aux")
        validate_checkpoint_schema(
            commit4_checkpoint,
            source="Commit-4 format-v3 checkpoint",
        )
        legacy_two_boundary_checkpoint = copy.deepcopy(checkpoint)
        legacy_two_boundary_checkpoint["config"]["diagnostics"].pop(
            "curvature_high_max_mm_inv"
        )
        legacy_two_boundary_checkpoint["config"]["diagnostics"][
            "curvature_medium_max_mm_inv"
        ] = 0.30
        validate_checkpoint_schema(
            legacy_two_boundary_checkpoint,
            source="Legacy two-boundary format-v3 checkpoint",
        )
        assert checkpoint["format_version"] == CHECKPOINT_FORMAT_VERSION
        assert (
            checkpoint["safety_model_schema_version"]
            == SAFETY_MODEL_SCHEMA_VERSION
        )
        assert (
            checkpoint["safety_cost_schema_version"]
            == SAFETY_COST_SCHEMA_VERSION
        )
        assert tuple(checkpoint["safety_cost_names"]) == safety_cost_names
        assert checkpoint["safety_dim"] == 2
        assert checkpoint["safety_curvature_scale_mm_inv"] == 0.1
        assert checkpoint["safety_translation_error_scale"] == 1.0
        assert checkpoint["safety_config"] == checkpoint_config["safety"]
        checkpoint_replay = EpisodeReplayBuffer(
            100,
            14,
            2,
            3,
            4,
            safety_cost_names=safety_cost_names,
            seed=1234,
        )
        checkpoint_replay.load_state_dict(checkpoint["replay"])
        assert len(checkpoint_replay) == len(replay)
        assert checkpoint_replay.state_dict()["episodes"][0][
            "safety_cost"
        ].shape == (8, 2)
        checkpoint_agent = TDMPC2Agent(
            14,
            2,
            checkpoint_agent_config,
            episode_length=200,
            device=device,
        )
        checkpoint_agent.load_state_dict(
            checkpoint["agent"],
            load_optimizers=True,
        )
        assert checkpoint_agent.update_count == safety_agent.update_count
        assert checkpoint_agent.model_optimizer.param_groups[-1]["name"] == "safety"

        auxiliary_dataset_path = Path(
            "/tmp/steve_commit46b_safety_aux.pt"
        ).resolve()
        assert auxiliary_dataset_path.is_file()
        auxiliary_checkpoint_config = copy.deepcopy(checkpoint_config)
        auxiliary_checkpoint_config["safety_aux"].update(
            {
                "enabled": True,
                "dataset_path": str(auxiliary_dataset_path),
                "loss_coef": 1.0,
                "batch_size": 64,
                "update_interval": 1,
                "sampling_mode": "mixed",
                "translation_fraction": 0.5,
                "curvature_fraction": 0.5,
                "sample_with_replacement": True,
            }
        )
        auxiliary_config = build_safety_aux_config(
            auxiliary_checkpoint_config
        )
        auxiliary_boundaries = curvature_boundaries_from_diagnostics(
            auxiliary_checkpoint_config
        )
        auxiliary_supervisor = SafetyAuxiliarySupervisor(
            auxiliary_config,
            observation_dim=14,
            action_dim=2,
            safety_cost_names=safety_cost_names,
            curvature_boundaries_mm_inv=auxiliary_boundaries,
            seed=77,
        )
        for normal_update_count in range(1, safety_agent.update_count + 1):
            scheduled_batch, scheduled_metadata = (
                auxiliary_supervisor.sample_for_update(normal_update_count)
            )
            assert scheduled_batch is not None
            assert scheduled_metadata is not None
        assert (
            auxiliary_supervisor.last_normal_update_count
            == safety_agent.update_count
        )
        auxiliary_checkpoint_path = (
            Path(temp_dir) / "schema_v3_with_safety_aux.pt"
        )
        save_checkpoint(
            path=auxiliary_checkpoint_path,
            config=auxiliary_checkpoint_config,
            agent=safety_agent,
            replay=replay,
            total_env_steps=8,
            episode_index=1,
            success_count=0,
            include_replay=True,
            safety_auxiliary=auxiliary_supervisor,
        )
        auxiliary_checkpoint = load_torch_checkpoint(
            auxiliary_checkpoint_path,
            map_location=device,
        )
        validate_checkpoint_schema(
            auxiliary_checkpoint,
            config=auxiliary_checkpoint_config,
            source="Safety auxiliary smoke-test checkpoint",
        )
        auxiliary_state = auxiliary_checkpoint[
            SAFETY_AUXILIARY_CHECKPOINT_KEY
        ]
        assert auxiliary_state["enabled"] is True
        assert auxiliary_state["dataset"]["fingerprint"] == (
            auxiliary_supervisor.fingerprint
        )
        assert auxiliary_state["dataset"]["split"]["validation_indices"].shape == (
            219,
        )
        assert "observation" not in auxiliary_state["sampler"]

        expected_next_auxiliary = auxiliary_supervisor.sample_for_update(
            safety_agent.update_count + 1
        )
        resumed_auxiliary_supervisor = SafetyAuxiliarySupervisor(
            auxiliary_config,
            observation_dim=14,
            action_dim=2,
            safety_cost_names=safety_cost_names,
            curvature_boundaries_mm_inv=auxiliary_boundaries,
            seed=9999,
        )
        resumed_auxiliary_supervisor.load_state_dict(
            auxiliary_state,
            expected_normal_update_count=safety_agent.update_count,
        )
        actual_next_auxiliary = (
            resumed_auxiliary_supervisor.sample_for_update(
                safety_agent.update_count + 1
            )
        )
        _assert_nested_state_equal(
            expected_next_auxiliary,
            actual_next_auxiliary,
        )

        missing_auxiliary_state = copy.deepcopy(auxiliary_checkpoint)
        missing_auxiliary_state.pop(SAFETY_AUXILIARY_CHECKPOINT_KEY)
        try:
            validate_checkpoint_schema(
                missing_auxiliary_state,
                source="Missing Safety auxiliary supervisor checkpoint",
            )
        except ValueError as exc:
            assert "missing" in str(exc)
            assert SAFETY_AUXILIARY_CHECKPOINT_KEY in str(exc)
        else:
            raise AssertionError(
                "Checkpoint validation accepted enabled safety_aux without state"
            )

        fingerprint_mismatch = copy.deepcopy(auxiliary_checkpoint)
        fingerprint_mismatch[SAFETY_AUXILIARY_CHECKPOINT_KEY]["dataset"][
            "fingerprint"
        ] = "0" * 64
        try:
            validate_checkpoint_schema(
                fingerprint_mismatch,
                config=auxiliary_checkpoint_config,
                source="Auxiliary fingerprint mismatch checkpoint",
            )
        except ValueError as exc:
            assert "fingerprint" in str(exc)
        else:
            raise AssertionError(
                "Checkpoint validation accepted an auxiliary fingerprint mismatch"
            )

        split_mismatch = copy.deepcopy(auxiliary_checkpoint)
        split_train_indices = split_mismatch[
            SAFETY_AUXILIARY_CHECKPOINT_KEY
        ]["dataset"]["split"]["train_indices"]
        split_train_indices[[0, 1]] = split_train_indices[[1, 0]]
        try:
            validate_checkpoint_schema(
                split_mismatch,
                config=auxiliary_checkpoint_config,
                source="Auxiliary fixed-split mismatch checkpoint",
            )
        except ValueError as exc:
            assert "fixed split" in str(exc)
        else:
            raise AssertionError(
                "Checkpoint validation accepted an auxiliary split mismatch"
            )

        missing_dataset_checkpoint = copy.deepcopy(auxiliary_checkpoint)
        missing_dataset_path = (
            Path(temp_dir) / "missing_safety_aux_dataset.pt"
        )
        missing_dataset_checkpoint["config"]["safety_aux"][
            "dataset_path"
        ] = str(missing_dataset_path)
        try:
            validate_checkpoint_schema(
                missing_dataset_checkpoint,
                source="Missing auxiliary dataset checkpoint",
            )
        except FileNotFoundError as exc:
            assert "does not exist" in str(exc)
            assert str(missing_dataset_path) in str(exc)
        else:
            raise AssertionError(
                "Checkpoint validation accepted a missing auxiliary dataset"
            )

        counter_mismatch = copy.deepcopy(auxiliary_checkpoint)
        counter_mismatch[SAFETY_AUXILIARY_CHECKPOINT_KEY][
            "auxiliary_update_count"
        ] += 1
        try:
            validate_checkpoint_schema(
                counter_mismatch,
                config=auxiliary_checkpoint_config,
                source="Auxiliary state-counter mismatch checkpoint",
            )
        except ValueError as exc:
            assert "counter is inconsistent" in str(exc)
        else:
            raise AssertionError(
                "Checkpoint validation accepted an auxiliary counter mismatch"
            )

        try:
            validate_checkpoint_schema(
                auxiliary_checkpoint,
                config=checkpoint_config,
                source="Enabled checkpoint with disabled requested config",
            )
        except ValueError as exc:
            assert "safety_aux.enabled" in str(exc)
            assert "requested config" in str(exc)
        else:
            raise AssertionError(
                "Enabled auxiliary checkpoint accepted a disabled config"
            )
        try:
            validate_checkpoint_schema(
                checkpoint,
                config=auxiliary_checkpoint_config,
                source="Disabled checkpoint with enabled requested config",
            )
        except ValueError as exc:
            assert "safety_aux.enabled" in str(exc)
            assert "requested config" in str(exc)
        else:
            raise AssertionError(
                "Disabled auxiliary checkpoint accepted an enabled config"
            )

        try:
            save_checkpoint(
                path=Path(temp_dir) / "missing_auxiliary_state_on_save.pt",
                config=auxiliary_checkpoint_config,
                agent=safety_agent,
                replay=replay,
                total_env_steps=8,
                episode_index=1,
                success_count=0,
                include_replay=False,
            )
        except ValueError as exc:
            assert "supervisor disagree" in str(exc)
        else:
            raise AssertionError(
                "Enabled auxiliary checkpoint save accepted no supervisor"
            )
        try:
            save_checkpoint(
                path=Path(temp_dir) / "unexpected_auxiliary_state_on_save.pt",
                config=checkpoint_config,
                agent=safety_agent,
                replay=replay,
                total_env_steps=8,
                episode_index=1,
                success_count=0,
                include_replay=False,
                safety_auxiliary=resumed_auxiliary_supervisor,
            )
        except ValueError as exc:
            assert "supervisor disagree" in str(exc)
        else:
            raise AssertionError(
                "Disabled auxiliary checkpoint save accepted a supervisor"
            )

        offline_arguments = safety_head_evaluation.parse_args(
            [
                "--checkpoint",
                str(auxiliary_checkpoint_path),
                "--safety-dataset",
                str(auxiliary_dataset_path),
                "--device",
                "cpu",
                "--safety-batch-size",
                "64",
            ]
        )
        with patch.object(
            safety_head_evaluation,
            "_make_evaluation_environment",
            side_effect=AssertionError(
                "Offline dataset-only evaluation constructed stEVE"
            ),
        ), patch("builtins.print"):
            first_offline_report = safety_head_evaluation.run(
                offline_arguments
            )
            second_offline_report = safety_head_evaluation.run(
                offline_arguments
            )
        assert first_offline_report["online_environment_constructed"] is False
        assert first_offline_report["evaluation_modes"] == [
            "fixed-offline-validation"
        ]
        first_fixed_validation = first_offline_report["fixed_validation"]
        second_fixed_validation = second_offline_report["fixed_validation"]
        assert first_fixed_validation["dataset"]["total_size"] == 1109
        assert first_fixed_validation["dataset"]["train_size"] == 890
        assert first_fixed_validation["dataset"]["validation_size"] == 219
        assert first_fixed_validation["evaluation"]["sample_count"] == 219
        assert first_fixed_validation["metrics"]["sample_count"] == 219
        assert (
            first_fixed_validation["evaluation"]["inference_batch_count"]
            == 4
        )
        offline_metrics = first_fixed_validation["metrics"]
        reason_groups = offline_metrics["per_translation_reason"]
        curvature_groups = offline_metrics["per_curvature_stratum"]
        assert tuple(reason_groups) == tuple(TRANSLATION_BLOCK_REASON_NAMES)
        assert tuple(curvature_groups) == (
            "low",
            "medium",
            "high",
            "extreme",
        )
        assert sum(group["count"] for group in reason_groups.values()) == 219
        assert sum(group["count"] for group in curvature_groups.values()) == 219
        for empty_reason in ("device_length_limit", "other"):
            empty_group = reason_groups[empty_reason]
            assert empty_group["count"] == 0
            assert all(
                value is None
                for channel in empty_group["channels"].values()
                for value in channel.values()
            )
        _assert_nested_state_equal(
            first_fixed_validation["metrics"],
            second_fixed_validation["metrics"],
        )
        _assert_nested_state_equal(
            first_fixed_validation["checkpoint_dataset_identity"],
            second_fixed_validation["checkpoint_dataset_identity"],
        )

        class FakeEvaluationEnv:
            def __init__(self) -> None:
                self.observation_space = type(
                    "ObservationSpace",
                    (),
                    {"shape": (14,)},
                )()
                self.action_space = type(
                    "ActionSpace",
                    (),
                    {"shape": (2,)},
                )()
                self.closed = False

            def close(self) -> None:
                self.closed = True

        fake_evaluation_env = FakeEvaluationEnv()
        with patch.object(
            safety_head_evaluation,
            "make_steve_env",
            return_value=fake_evaluation_env,
        ):
            (
                loaded_evaluation_checkpoint,
                loaded_evaluation_config,
                loaded_evaluation_env,
                loaded_evaluation_agent,
                loaded_evaluation_device,
            ) = safety_head_evaluation.load_evaluation_components(
                checkpoint_path,
                requested_device="cpu",
            )
        assert (
            loaded_evaluation_checkpoint["format_version"]
            == CHECKPOINT_FORMAT_VERSION
        )
        assert loaded_evaluation_config == checkpoint_config
        assert loaded_evaluation_env is fake_evaluation_env
        assert loaded_evaluation_device == device
        assert (
            loaded_evaluation_agent.safety_cost_names
            == safety_cost_names
        )

        evaluation_observation = np.linspace(
            -1.0,
            1.0,
            14,
            dtype=np.float32,
        )
        evaluation_action = np.asarray([0.25, -0.5], dtype=np.float32)
        evaluation_action_before = evaluation_action.copy()
        previous_mean_before = (
            loaded_evaluation_agent.previous_mean.detach().clone()
        )
        evaluation_rng_before = torch.get_rng_state().clone()
        forced_transformed_prediction = torch.tensor(
            [[-0.75, 0.5]],
            dtype=torch.float32,
        )
        expected_evaluation_prediction = (
            loaded_evaluation_agent.model.decode_safety_transformed(
                forced_transformed_prediction
            )
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
        )
        with ExitStack() as evaluation_stack:
            evaluation_plan_spy = evaluation_stack.enter_context(
                patch.object(
                    loaded_evaluation_agent,
                    "_plan",
                    side_effect=AssertionError(
                        "Safety diagnostic prediction called MPC planning"
                    ),
                )
            )
            evaluation_stack.enter_context(
                patch.object(
                    loaded_evaluation_agent.model,
                    "safety_transformed",
                    return_value=forced_transformed_prediction,
                )
            )
            (
                evaluation_prediction,
                evaluation_prediction_transformed,
                evaluation_inference_ms,
            ) = safety_head_evaluation.predict_safety_before_step(
                loaded_evaluation_agent,
                evaluation_observation,
                evaluation_action,
            )
        assert evaluation_plan_spy.call_count == 0
        np.testing.assert_array_equal(
            evaluation_action,
            evaluation_action_before,
        )
        np.testing.assert_allclose(
            evaluation_prediction,
            expected_evaluation_prediction,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            evaluation_prediction_transformed,
            forced_transformed_prediction.squeeze(0).numpy(),
            rtol=0.0,
            atol=0.0,
        )
        assert np.isfinite(evaluation_prediction).all()
        assert np.isfinite(evaluation_prediction_transformed).all()
        assert np.isfinite(evaluation_inference_ms)
        assert evaluation_inference_ms >= 0.0
        torch.testing.assert_close(
            loaded_evaluation_agent.previous_mean,
            previous_mean_before,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            torch.get_rng_state(),
            evaluation_rng_before,
            rtol=0.0,
            atol=0.0,
        )
        loaded_evaluation_env.close()
        assert fake_evaluation_env.closed

        old_checkpoint = copy.deepcopy(checkpoint)
        old_checkpoint["format_version"] = 2
        old_checkpoint.pop("safety_model_schema_version")
        try:
            validate_checkpoint_schema(
                old_checkpoint,
                source="Old smoke-test checkpoint",
            )
        except ValueError as exc:
            assert "predate" in str(exc)
            assert "safety-model" in str(exc)
        else:
            raise AssertionError("Training accepted a format-v2 checkpoint")

        legacy_checkpoint = copy.deepcopy(checkpoint)
        legacy_checkpoint["safety_cost_names"] = (
            LEGACY_THREE_CHANNEL_SAFETY_COST_NAMES
        )
        try:
            validate_checkpoint_schema(
                legacy_checkpoint,
                source="Legacy smoke-test checkpoint",
            )
        except ValueError as exc:
            assert "unsupported legacy three-channel" in str(exc)
        else:
            raise AssertionError("Training accepted a legacy safety schema")

        reordered_checkpoint = copy.deepcopy(checkpoint)
        reordered_checkpoint["safety_cost_names"] = tuple(
            reversed(safety_cost_names)
        )
        try:
            validate_checkpoint_schema(
                reordered_checkpoint,
                source="Reordered smoke-test checkpoint",
            )
        except ValueError as exc:
            assert "exact order" in str(exc)
        else:
            raise AssertionError("Training accepted reordered Safety channels")

        scale_mismatch_checkpoint = copy.deepcopy(checkpoint)
        scale_mismatch_checkpoint["safety_curvature_scale_mm_inv"] = 0.2
        try:
            validate_checkpoint_schema(
                scale_mismatch_checkpoint,
                source="Scale-mismatch smoke-test checkpoint",
            )
        except ValueError as exc:
            assert "does not match embedded config" in str(exc)
        else:
            raise AssertionError("Training accepted a mismatched Safety scale")

    print(
        "PASS: two-channel Safety Head shapes/transforms, aligned supervised "
        "losses/gradients, coefficient-zero and planning isolation, and strict "
        "format-v3 checkpoint restore."
    )


if __name__ == "__main__":
    main()
