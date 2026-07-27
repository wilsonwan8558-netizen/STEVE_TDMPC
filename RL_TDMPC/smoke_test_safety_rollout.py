"""Focused, simulation-free checks for multi-step Safety rollout evaluation.

The evaluator is deliberately post-hoc: these checks use synthetic trajectories
and a tiny TD-MPC2 agent, and never construct or advance a SOFA environment.
"""

from __future__ import annotations

import copy
import inspect
import json
import random
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

import tdmpc2.agent as agent_module
from evaluate_safety_rollout import (
    aggregate_future_event_targets_and_scores,
    build_episode_windows,
    compute_latent_drift,
    evaluate_early_warnings,
    open_loop_rollout,
    strict_json_dumps,
    teacher_forced_rollout,
    write_strict_json,
)
from tdmpc2.agent import TDMPC2Agent
from tdmpc2.common import load_config
from train import DEFAULT_CONFIG, build_agent_config


class SyntheticWorldModel(torch.nn.Module):
    """Small deterministic model that exposes rollout ordering mistakes."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.25))
        self.encode_rows = 0
        self.calls = []

    def reset_trace(self) -> None:
        self.encode_rows = 0
        self.calls.clear()

    def encode(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        self.encode_rows += int(observation.shape[0])
        self.calls.append(("encode", int(observation.shape[0])))
        # Keep a real model parameter in the graph without changing the
        # deterministic synthetic latent values.
        return observation[..., :2] + self.anchor * 0.0

    def safety_transformed(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        self.calls.append(("safety", int(latent.shape[0])))
        return latent + 10.0 * action

    def decode_safety_transformed(
        self,
        transformed: torch.Tensor,
    ) -> torch.Tensor:
        self.calls.append(("decode", int(transformed.shape[0])))
        return transformed

    def next(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        self.calls.append(("next", int(latent.shape[0])))
        return latent + action


def _as_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _get_alias(mapping, *names):
    for name in names:
        if name in mapping:
            return mapping[name]
    raise AssertionError(
        f"Expected one of fields {names}, got {sorted(mapping.keys())}"
    )


def _assert_nested_state_equal(left, right) -> None:
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


def _assert_raises(callable_object, exception_type) -> None:
    try:
        callable_object()
    except exception_type:
        return
    if isinstance(exception_type, tuple):
        expected_name = "/".join(item.__name__ for item in exception_type)
    else:
        expected_name = exception_type.__name__
    raise AssertionError(f"Expected {expected_name}, but no exception was raised")


def _test_episode_windows_without_padding() -> None:
    episode_ids = np.asarray([7, 7, 7, 9, 9, 9, 9, 12], dtype=np.int64)
    windows = build_episode_windows(episode_ids, 3)
    expected = (
        np.asarray([0, 1, 2]),
        np.asarray([3, 4, 5]),
        np.asarray([4, 5, 6]),
    )
    assert len(windows) == len(expected)
    for actual, target in zip(windows, expected):
        actual = np.asarray(actual)
        np.testing.assert_array_equal(actual, target)
        assert actual.shape == (3,)
        assert np.unique(episode_ids[actual]).size == 1

    single_step_windows = build_episode_windows(episode_ids, 1)
    assert len(single_step_windows) == episode_ids.size
    for index, window in enumerate(single_step_windows):
        np.testing.assert_array_equal(window, np.asarray([index]))

    assert build_episode_windows(episode_ids, 9) == []
    _assert_raises(lambda: build_episode_windows(episode_ids, 0), ValueError)


def _test_temporal_alignment_and_open_loop_behavior() -> None:
    model = SyntheticWorldModel()
    device = torch.device("cpu")
    observations = np.asarray(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
        dtype=np.float32,
    )
    requested_actions = np.asarray(
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        dtype=np.float32,
    )
    applied_actions = np.asarray(
        [[-0.1, -0.2], [-0.3, -0.4], [-0.5, -0.6]],
        dtype=np.float32,
    )

    teacher = teacher_forced_rollout(
        model,
        observations,
        requested_actions,
        device,
    )
    teacher_prediction = _as_numpy(teacher["prediction"])
    expected_teacher = observations + 10.0 * requested_actions
    applied_action_prediction = observations + 10.0 * applied_actions
    np.testing.assert_allclose(
        teacher_prediction,
        expected_teacher,
        rtol=0.0,
        atol=1e-6,
    )
    assert not np.allclose(teacher_prediction, applied_action_prediction)
    # Use the exact prediction as a synthetic safety_cost_t. Same-step
    # alignment gives zero error; a one-transition target shift cannot.
    safety_cost_t = expected_teacher.copy()
    assert np.mean(np.abs(teacher_prediction - safety_cost_t)) == 0.0
    assert (
        np.mean(np.abs(teacher_prediction[:-1] - safety_cost_t[1:])) > 0.0
    )
    assert model.encode_rows == observations.shape[0]
    np.testing.assert_allclose(
        _as_numpy(teacher["latent"]),
        observations,
        rtol=0.0,
        atol=1e-6,
    )

    model.reset_trace()
    opened = open_loop_rollout(
        model,
        observations[0],
        requested_actions,
        device,
    )
    open_prediction = _as_numpy(opened["prediction"])
    open_latent = _as_numpy(opened["latent"])
    expected_latent = np.asarray(
        [
            [1.0, 10.0],
            [1.1, 10.2],
            [1.4, 10.6],
            [1.9, 11.2],
        ],
        dtype=np.float32,
    )
    expected_open_prediction = expected_latent[:-1] + (
        10.0 * requested_actions
    )
    np.testing.assert_allclose(
        open_latent,
        expected_latent,
        rtol=0.0,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        open_prediction,
        expected_open_prediction,
        rtol=0.0,
        atol=2e-6,
    )
    assert open_latent.shape[0] == requested_actions.shape[0] + 1
    assert model.encode_rows == 1
    calls = [name for name, _ in model.calls if name != "decode"]
    assert calls == [
        "encode",
        "safety",
        "next",
        "safety",
        "next",
        "safety",
        "next",
    ]
    assert np.isfinite(open_prediction).all()
    assert np.isfinite(open_latent).all()


def _test_future_event_aggregation() -> None:
    result = aggregate_future_event_targets_and_scores(
        np.asarray([0.0, 0.7, 0.0], dtype=np.float32),
        np.asarray([0.1, 0.4, 0.2], dtype=np.float32),
        0.5,
    )
    true_event = _get_alias(
        result,
        "true_event",
        "true_block_within_h",
        "true_block_within_H",
    )
    risk_max = _get_alias(
        result,
        "risk_max",
        "predicted_block_risk_max",
        "predicted_block_risk_max_H",
    )
    risk_sum = _get_alias(
        result,
        "risk_sum",
        "predicted_block_risk_sum",
        "predicted_block_risk_sum_H",
    )
    assert bool(np.asarray(true_event).item())
    np.testing.assert_allclose(float(np.asarray(risk_max).item()), 0.4)
    np.testing.assert_allclose(float(np.asarray(risk_sum).item()), 0.35)

    no_event = aggregate_future_event_targets_and_scores(
        np.zeros(2, dtype=np.float32),
        np.asarray([0.2, 0.1], dtype=np.float32),
        1.0,
    )
    assert not bool(
        np.asarray(
            _get_alias(
                no_event,
                "true_event",
                "true_block_within_h",
                "true_block_within_H",
            )
        ).item()
    )
    np.testing.assert_allclose(
        float(
            np.asarray(
                _get_alias(
                    no_event,
                    "risk_max",
                    "predicted_block_risk_max",
                    "predicted_block_risk_max_H",
                )
            ).item()
        ),
        0.2,
    )
    _assert_raises(
        lambda: aggregate_future_event_targets_and_scores(
            np.zeros(2, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            1.0,
        ),
        ValueError,
    )


def _test_early_warning_logic() -> None:
    episode_ids = np.asarray([10, 10, 10, 10, 20, 20, 20, 20])
    translation_targets = np.asarray(
        # Index 7 remains positive in the same episode. It belongs to the
        # blockage run that starts at index 6 and must not become a second
        # event whose "warning" is the already-blocked transition at index 6.
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.5],
        dtype=np.float32,
    )
    actual_step_durations = np.asarray(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        dtype=np.float32,
    )
    risks = {
        0: np.asarray([0.01, 0.30, 0.40, 0.80]),
        1: np.asarray([0.01, 0.10, 0.70]),
        2: np.asarray([0.10, 0.60]),
        3: np.asarray([0.90]),
        4: np.asarray([0.01, 0.05, 0.10]),
        5: np.asarray([0.05, 0.10]),
        6: np.asarray([0.90]),
        7: np.asarray([0.90]),
    }
    predictor_calls = []

    def predictor(start_index: int, horizon: int) -> np.ndarray:
        predictor_calls.append((int(start_index), int(horizon)))
        prediction = risks[int(start_index)][: int(horizon)]
        if prediction.shape != (int(horizon),):
            raise AssertionError("Early-warning evaluator requested padding")
        return prediction

    result = evaluate_early_warnings(
        episode_ids,
        translation_targets,
        predictor,
        (1, 3),
        0.2,
        actual_step_durations,
    )
    events = _get_alias(result, "events", "event_records")
    assert len(events) == 2
    event_by_episode = {
        int(_get_alias(event, "episode_id")): event for event in events
    }

    warned = event_by_episode[10]
    assert int(_get_alias(warned, "event_index", "index")) == 3
    assert bool(_get_alias(warned, "immediate_detection"))
    warned_lookbacks = _get_alias(warned, "by_lookback", "lookbacks")
    one_step = warned_lookbacks.get("1", warned_lookbacks.get(1))
    three_step = warned_lookbacks.get("3", warned_lookbacks.get(3))
    assert one_step is not None and three_step is not None
    assert not bool(_get_alias(one_step, "missed"))
    assert int(_get_alias(one_step, "lead_steps")) == 1
    assert int(
        _get_alias(
            one_step,
            "earliest_warning_index",
            "warning_index",
        )
    ) == 2
    assert not bool(_get_alias(three_step, "missed"))
    assert int(_get_alias(three_step, "lead_steps")) == 3
    assert int(
        _get_alias(
            three_step,
            "earliest_warning_index",
            "warning_index",
        )
    ) == 0
    lead_seconds = _get_alias(
        three_step,
        "lead_seconds",
        "warning_lead_seconds",
    )
    np.testing.assert_allclose(float(lead_seconds), 0.6, atol=1e-6)

    immediate_only = event_by_episode[20]
    assert int(_get_alias(immediate_only, "event_index", "index")) == 6
    assert bool(_get_alias(immediate_only, "immediate_detection"))
    missed_lookbacks = _get_alias(
        immediate_only,
        "by_lookback",
        "lookbacks",
    )
    for key in ("1", "3"):
        record = missed_lookbacks.get(key, missed_lookbacks.get(int(key)))
        assert record is not None
        assert bool(_get_alias(record, "missed"))
        assert _get_alias(
            record,
            "earliest_warning_index",
            "warning_index",
        ) is None
        assert int(_get_alias(record, "lead_steps")) == 0

    # Every requested rollout stays within its real episode and terminates at
    # the event. A blocked-transition-only detection is queried separately and
    # is never counted as early warning.
    for start_index, horizon in predictor_calls:
        stop_index = start_index + horizon - 1
        assert episode_ids[start_index] == episode_ids[stop_index]


def _test_latent_drift() -> None:
    encoded = np.asarray(
        [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        dtype=np.float32,
    )
    perfect = compute_latent_drift(encoded.copy(), encoded)
    perfect_mse = _as_numpy(_get_alias(perfect, "mse"))
    perfect_l1 = _as_numpy(_get_alias(perfect, "l1", "l1_error"))
    perfect_cosine = _as_numpy(
        _get_alias(perfect, "cosine_similarity", "cosine")
    )
    assert perfect_mse.shape == (3,)
    assert perfect_l1.shape == (3,)
    assert perfect_cosine.shape == (3,)
    np.testing.assert_allclose(perfect_mse, 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(perfect_l1, 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(perfect_cosine, 1.0, atol=1e-6)

    perturbed = encoded.copy()
    perturbed[1, 1] = 1.0
    drift = compute_latent_drift(perturbed, encoded)
    mse = _as_numpy(_get_alias(drift, "mse"))
    l1 = _as_numpy(_get_alias(drift, "l1", "l1_error"))
    cosine = _as_numpy(_get_alias(drift, "cosine_similarity", "cosine"))
    np.testing.assert_allclose(mse[[0, 2]], 0.0, atol=0.0)
    np.testing.assert_allclose(l1[[0, 2]], 0.0, atol=0.0)
    np.testing.assert_allclose(cosine[[0, 2]], 1.0, atol=1e-6)
    assert mse[1] > 0.0
    assert l1[1] > 0.0
    assert cosine[1] < 1.0
    _assert_raises(
        lambda: compute_latent_drift(perturbed[:-1], encoded),
        ValueError,
    )


def _test_strict_json() -> None:
    valid = {
        "defined": 1.25,
        "unavailable": None,
        "nested": [True, 2],
    }
    text = strict_json_dumps(valid)
    decoded = json.loads(text)
    assert decoded == valid
    assert "NaN" not in text
    assert "Infinity" not in text

    for invalid in (float("nan"), float("inf"), float("-inf")):
        _assert_raises(
            lambda invalid=invalid: strict_json_dumps({"invalid": invalid}),
            (ValueError, TypeError),
        )

    with TemporaryDirectory(prefix="steve-rollout-smoke-") as directory:
        path = Path(directory) / "strict.json"
        write_strict_json(path, valid)
        written = path.read_text(encoding="utf-8")
        assert json.loads(written) == valid
        assert "NaN" not in written
        assert "Infinity" not in written
        _assert_raises(
            lambda: write_strict_json(path, {"invalid": np.inf}),
            (ValueError, TypeError),
        )


def _small_agent() -> TDMPC2Agent:
    full_config = load_config(DEFAULT_CONFIG)
    full_config["model"].update(
        {
            "latent_dim": 16,
            "enc_dim": 16,
            "mlp_dim": 16,
            "num_enc_layers": 2,
            "simnorm_dim": 4,
            "num_q": 2,
            "dropout": 0.0,
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
    config = build_agent_config(full_config)
    return TDMPC2Agent(
        14,
        2,
        config,
        episode_length=20,
        device=torch.device("cpu"),
    )


def _test_planning_isolation() -> None:
    assert "evaluate_safety_rollout" not in inspect.getsource(agent_module)
    plan_source = inspect.getsource(TDMPC2Agent._plan)
    assert "open_loop_rollout" not in plan_source
    assert "teacher_forced_rollout" not in plan_source
    # Active Safety-MPC is now an explicit opt-in branch. The default config
    # used below remains disabled, while the planner source may reference only
    # the dedicated Translation-risk helper (never post-hoc rollout utilities
    # or a Curvature output).
    assert "self.safety_mpc_active" in plan_source
    assert "_estimate_translation_trajectory_risk" in plan_source
    assert "safety_curvature" not in plan_source

    torch.manual_seed(311)
    np.random.seed(311)
    random.seed(311)
    agent = _small_agent()
    observation = np.linspace(-1.0, 1.0, 14, dtype=np.float32)
    rollout_observations = np.stack(
        (observation, observation * 0.5, observation * 0.25),
        axis=0,
    )
    rollout_actions = np.asarray(
        [[0.2, -0.3], [0.1, 0.4], [-0.2, 0.0]],
        dtype=np.float32,
    )

    previous_mean_start = agent.previous_mean.detach().clone()
    torch_rng_start = torch.get_rng_state().clone()
    numpy_rng_start = copy.deepcopy(np.random.get_state())
    python_rng_start = random.getstate()
    action_without_rollout = agent.act(
        observation,
        first_step=True,
        eval_mode=True,
    )

    agent.previous_mean.copy_(previous_mean_start)
    torch.set_rng_state(torch_rng_start)
    np.random.set_state(numpy_rng_start)
    random.setstate(python_rng_start)
    model_state_before = copy.deepcopy(agent.model.state_dict())
    training_flags_before = {
        name: module.training for name, module in agent.model.named_modules()
    }

    teacher_forced_rollout(
        agent.model,
        rollout_observations,
        rollout_actions,
        torch.device("cpu"),
    )
    open_loop_rollout(
        agent.model,
        rollout_observations[0],
        rollout_actions,
        torch.device("cpu"),
    )

    torch.testing.assert_close(
        torch.get_rng_state(),
        torch_rng_start,
        rtol=0.0,
        atol=0.0,
    )
    _assert_nested_state_equal(np.random.get_state(), numpy_rng_start)
    assert random.getstate() == python_rng_start
    torch.testing.assert_close(
        agent.previous_mean,
        previous_mean_start,
        rtol=0.0,
        atol=0.0,
    )
    _assert_nested_state_equal(agent.model.state_dict(), model_state_before)
    assert {
        name: module.training for name, module in agent.model.named_modules()
    } == training_flags_before

    value_latent = agent.model.encode(
        torch.as_tensor(observation).unsqueeze(0)
    )
    value_actions = torch.zeros(
        agent.horizon,
        1,
        agent.action_dim,
        dtype=torch.float32,
    )
    value_before = agent._estimate_value(value_latent, value_actions)
    open_loop_rollout(
        agent.model,
        rollout_observations[0],
        rollout_actions,
        torch.device("cpu"),
    )
    value_after = agent._estimate_value(value_latent, value_actions)
    torch.testing.assert_close(value_after, value_before, rtol=0.0, atol=0.0)

    # The post-hoc evaluator itself must not be reachable from act()/_plan().
    agent.previous_mean.copy_(previous_mean_start)
    torch.set_rng_state(torch_rng_start)
    np.random.set_state(numpy_rng_start)
    random.setstate(python_rng_start)
    with patch(
        "evaluate_safety_rollout.open_loop_rollout",
        side_effect=AssertionError(
            "Safety rollout evaluation was called during planning"
        ),
    ) as rollout_spy:
        action_after_rollout = agent.act(
            observation,
            first_step=True,
            eval_mode=True,
        )
    assert rollout_spy.call_count == 0
    np.testing.assert_array_equal(action_after_rollout, action_without_rollout)


def main() -> None:
    _test_episode_windows_without_padding()
    _test_temporal_alignment_and_open_loop_behavior()
    _test_future_event_aggregation()
    _test_early_warning_logic()
    _test_latent_drift()
    _test_strict_json()
    _test_planning_isolation()
    print(
        "PASS: same-step Safety rollout alignment, initial-only open-loop "
        "encoding, unpadded episode windows, future-event aggregation, "
        "early-warning semantics, latent drift, strict JSON, and planning "
        "isolation."
    )


if __name__ == "__main__":
    main()
