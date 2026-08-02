"""Focused, simulation-free checks for Integral Lagrangian Safety-MPC."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

from evaluate import (
    EVALUATION_REPORT_SCHEMA_VERSION,
    apply_evaluation_safety_mpc_config,
    compute_episode_tree_end_blockage_fraction,
    resolve_evaluation_integral_lagrangian_config,
    resolve_evaluation_safety_mpc_config,
    state_sha256,
    validate_evaluation_planner_compatibility,
)
from smoke_test_safety_mpc_active import (
    _agent,
    _full_config,
    _test_intervention_mask_unchanged,
)
from tdmpc2.common import (
    atomic_json_save,
    build_integral_lagrangian_config,
)
from tdmpc2.integral_lagrangian import (
    DEFAULT_INTEGRAL_LAGRANGIAN_CONFIG,
    IntegralLagrangianController,
)
from train import resolved_safety_mpc_config


def _assert_raises(function, exception_type) -> None:
    try:
        function()
    except exception_type:
        return
    if isinstance(exception_type, tuple):
        name = "/".join(item.__name__ for item in exception_type)
    else:
        name = exception_type.__name__
    raise AssertionError(f"Expected {name}, but no exception was raised")


def _config(**updates) -> dict:
    config = copy.deepcopy(DEFAULT_INTEGRAL_LAGRANGIAN_CONFIG)
    config["enabled"] = True
    config.update(updates)
    return config


def _assert_close(actual: float, expected: float) -> None:
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-12)


def _assert_finite_tree(value) -> None:
    if isinstance(value, dict):
        for child in value.values():
            _assert_finite_tree(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_finite_tree(child)
    elif isinstance(value, float):
        assert math.isfinite(value)


def _test_exact_update_warmup_and_window() -> None:
    _assert_close(compute_episode_tree_end_blockage_fraction(2, 8), 0.25)
    _assert_close(compute_episode_tree_end_blockage_fraction(0, 0), 0.0)
    for counts, expected_error in (
        ((True, 1), TypeError),
        ((1, 1.0), TypeError),
        ((-1, 1), ValueError),
        ((2, 1), ValueError),
    ):
        _assert_raises(
            lambda counts=counts: compute_episode_tree_end_blockage_fraction(
                *counts
            ),
            expected_error,
        )
    controller = IntegralLagrangianController(
        _config(
            initial_alpha=0.1,
            integral_gain=0.2,
            cost_limit=0.1,
            rolling_window_episodes=1,
            warmup_episodes=1,
        )
    )
    first = controller.observe_episode_cost(0.5)
    assert first["warmup_active"]
    assert not first["dual_update_applied"]
    _assert_close(first["alpha_before_update"], 0.1)
    _assert_close(first["alpha_after_update"], 0.1)
    second = controller.observe_episode_cost(0.5)
    assert not second["warmup_active"]
    assert second["dual_update_applied"]
    _assert_close(second["dual_error"], 0.4)
    _assert_close(second["alpha_after_update"], 0.18)
    _assert_close(second["alpha_update_amount"], 0.08)

    windowed = IntegralLagrangianController(
        _config(
            integral_gain=0.0,
            cost_limit=0.0,
            rolling_window_episodes=2,
            warmup_episodes=1,
        )
    )
    records = [
        windowed.observe_episode_cost(cost) for cost in (0.1, 0.2, 0.5)
    ]
    _assert_close(records[0]["rolling_mean_cost"], 0.1)
    _assert_close(records[1]["rolling_mean_cost"], 0.15)
    _assert_close(records[2]["rolling_mean_cost"], 0.35)
    assert windowed.state_dict()["recent_episode_costs"] == [0.2, 0.5]


def _test_clipping_and_integral_direction() -> None:
    lower = IntegralLagrangianController(
        _config(
            initial_alpha=0.05,
            integral_gain=1.0,
            cost_limit=0.5,
            alpha_max=0.5,
            rolling_window_episodes=1,
            warmup_episodes=1,
        )
    )
    lower.observe_episode_cost(0.0)
    lower_record = lower.observe_episode_cost(0.0)
    _assert_close(lower.current_alpha, 0.0)
    assert lower_record["lower_clipping_active"]
    assert not lower_record["upper_clipping_active"]
    assert lower.summary()["lower_clip_count"] == 1

    upper = IntegralLagrangianController(
        _config(
            initial_alpha=0.4,
            integral_gain=1.0,
            cost_limit=0.0,
            alpha_max=0.5,
            rolling_window_episodes=1,
            warmup_episodes=1,
        )
    )
    upper.observe_episode_cost(1.0)
    upper_record = upper.observe_episode_cost(1.0)
    _assert_close(upper.current_alpha, 0.5)
    assert upper_record["upper_clipping_active"]
    assert not upper_record["lower_clipping_active"]
    assert upper.summary()["upper_clip_count"] == 1

    increasing = IntegralLagrangianController(
        _config(
            initial_alpha=0.1,
            integral_gain=0.1,
            cost_limit=0.1,
            alpha_max=1.0,
            rolling_window_episodes=1,
            warmup_episodes=1,
        )
    )
    for _ in range(4):
        increasing.observe_episode_cost(0.4)
    trajectory = increasing.summary()["alpha_trajectory"]
    assert trajectory[2] < trajectory[3] < trajectory[4]

    decreasing = IntegralLagrangianController(
        _config(
            initial_alpha=0.8,
            integral_gain=0.1,
            cost_limit=0.5,
            alpha_max=1.0,
            rolling_window_episodes=1,
            warmup_episodes=1,
        )
    )
    for _ in range(4):
        decreasing.observe_episode_cost(0.1)
    trajectory = decreasing.summary()["alpha_trajectory"]
    assert trajectory[2] > trajectory[3] > trajectory[4]


def _test_zero_gain_equal_limit_and_reset() -> None:
    zero_gain = IntegralLagrangianController(
        _config(
            initial_alpha=0.23,
            integral_gain=0.0,
            cost_limit=0.01,
            rolling_window_episodes=1,
            warmup_episodes=1,
        )
    )
    for cost in (1.0, 1.0, 0.0):
        zero_gain.observe_episode_cost(cost)
    assert zero_gain.summary()["alpha_trajectory"] == [0.23] * 4

    equal = IntegralLagrangianController(
        _config(
            initial_alpha=0.2,
            integral_gain=0.9,
            cost_limit=0.3,
            rolling_window_episodes=1,
            warmup_episodes=1,
        )
    )
    equal.observe_episode_cost(0.3)
    equal_record = equal.observe_episode_cost(0.3)
    assert equal_record["dual_update_applied"]
    _assert_close(equal_record["alpha_update_amount"], 0.0)

    original_config = copy.deepcopy(equal.config)
    external_view = equal.config
    external_view["initial_alpha"] = float("nan")
    assert equal.config == original_config
    equal.reset()
    _assert_close(equal.current_alpha, 0.2)
    assert equal.config == original_config
    assert equal.state_dict()["observed_cost_trajectory"] == []
    assert equal.summary()["alpha_trajectory"] == [0.2]
    equal.reset()
    assert equal.summary()["alpha_trajectory"] == [0.2]
    first_replay = [
        equal.observe_episode_cost(cost) for cost in (0.3, 0.4, 0.2)
    ]
    first_trajectory = equal.summary()["alpha_trajectory"]
    equal.reset()
    second_replay = [
        equal.observe_episode_cost(cost) for cost in (0.3, 0.4, 0.2)
    ]
    assert second_replay == first_replay
    assert equal.summary()["alpha_trajectory"] == first_trajectory


def _test_six_episode_warmup_boundary_and_json() -> None:
    controller = IntegralLagrangianController(_config())
    costs = (0.0, 0.015, 0.0, 0.0, 0.0, 1.0)
    records = [controller.observe_episode_cost(cost) for cost in costs]
    assert all(record["warmup_active"] for record in records[:5])
    assert all(not record["dual_update_applied"] for record in records[:5])
    assert not records[5]["warmup_active"]
    assert records[5]["dual_update_applied"]
    summary = controller.summary()
    assert summary["dual_update_count"] == 1
    assert len(summary["alpha_trajectory"]) == 7
    assert len(summary["observed_cost_trajectory"]) == 6
    assert len(summary["rolling_cost_trajectory"]) == 6
    assert summary["observed_cost_trajectory"] == list(costs)
    _assert_close(summary["final_alpha"], 0.1193)
    assert summary["episode_alpha_trajectory"] == [0.1] * 6
    assert summary["alpha_statistics_basis"] == (
        "controller_state_trajectory"
    )
    _assert_close(summary["alpha_mean"], (0.6 + 0.1193) / 7.0)
    _assert_close(summary["minimum_alpha"], 0.1)
    _assert_close(summary["maximum_alpha"], 0.1193)
    _assert_close(summary["episode_applied_minimum_alpha"], 0.1)
    _assert_close(summary["episode_applied_maximum_alpha"], 0.1)
    _assert_close(summary["episode_applied_alpha_mean"], 0.1)
    assert controller.state_dict()["schema_version"] == 1
    _assert_finite_tree(records)
    _assert_finite_tree(summary)

    report = {
        "schema_version": EVALUATION_REPORT_SCHEMA_VERSION,
        "episodes": [
            {"episode_index": index + 1, "integral_lagrangian": record}
            for index, record in enumerate(records)
        ],
        "summary": {"integral_lagrangian": summary},
    }
    with TemporaryDirectory(prefix="steve-integral-json-") as directory:
        output = Path(directory) / "report.json"
        atomic_json_save(report, output)
        with output.open("r", encoding="utf-8") as stream:
            restored = json.load(stream)
        assert restored == report
        raw = output.read_text(encoding="utf-8")
        assert "NaN" not in raw and "Infinity" not in raw


def _test_config_and_runtime_validation() -> None:
    assert build_integral_lagrangian_config({}) == (
        DEFAULT_INTEGRAL_LAGRANGIAN_CONFIG
    )
    valid = build_integral_lagrangian_config(
        {"integral_lagrangian": _config()}
    )
    assert valid == _config()

    for key in DEFAULT_INTEGRAL_LAGRANGIAN_CONFIG:
        missing = _config()
        missing.pop(key)
        _assert_raises(
            lambda missing=missing: build_integral_lagrangian_config(
                {"integral_lagrangian": missing}
            ),
            KeyError,
        )
    extra = _config(extra=1)
    _assert_raises(
        lambda: build_integral_lagrangian_config(
            {"integral_lagrangian": extra}
        ),
        ValueError,
    )
    for section in (None, [], "enabled"):
        _assert_raises(
            lambda section=section: build_integral_lagrangian_config(
                {"integral_lagrangian": section}
            ),
            TypeError,
        )
    for key in (
        "initial_alpha",
        "integral_gain",
        "cost_limit",
    ):
        for value in (-0.1, float("nan"), float("inf")):
            invalid = _config(**{key: value})
            _assert_raises(
                lambda invalid=invalid: build_integral_lagrangian_config(
                    {"integral_lagrangian": invalid}
                ),
                ValueError,
            )
    for value in (0.0, -0.1, float("nan"), float("inf")):
        invalid = _config(alpha_max=value)
        _assert_raises(
            lambda invalid=invalid: build_integral_lagrangian_config(
                {"integral_lagrangian": invalid}
            ),
            ValueError,
        )
    for key in ("rolling_window_episodes", "warmup_episodes"):
        for value in (0, -1):
            invalid = _config(**{key: value})
            _assert_raises(
                lambda invalid=invalid: build_integral_lagrangian_config(
                    {"integral_lagrangian": invalid}
                ),
                ValueError,
            )
        for value in (True, 1.0, "1"):
            invalid = _config(**{key: value})
            _assert_raises(
                lambda invalid=invalid: build_integral_lagrangian_config(
                    {"integral_lagrangian": invalid}
                ),
                TypeError,
            )
    for key in (
        "initial_alpha",
        "integral_gain",
        "cost_limit",
        "alpha_max",
    ):
        invalid = _config(**{key: True})
        _assert_raises(
            lambda invalid=invalid: build_integral_lagrangian_config(
                {"integral_lagrangian": invalid}
            ),
            TypeError,
        )
    _assert_raises(
        lambda: build_integral_lagrangian_config(
            {"integral_lagrangian": _config(initial_alpha=0.6)}
        ),
        ValueError,
    )
    _assert_raises(
        lambda: IntegralLagrangianController(
            DEFAULT_INTEGRAL_LAGRANGIAN_CONFIG
        ),
        ValueError,
    )

    controller = IntegralLagrangianController(_config())
    pristine = controller.state_dict()
    for invalid_cost, expected_error in (
        (True, TypeError),
        ("0.1", TypeError),
        (-0.1, ValueError),
        (1.1, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
    ):
        _assert_raises(
            lambda invalid_cost=invalid_cost: (
                controller.observe_episode_cost(invalid_cost)
            ),
            expected_error,
        )
        assert controller.state_dict() == pristine

    overflowing = IntegralLagrangianController(
        _config(
            integral_gain=1.0e308,
            cost_limit=1.0e308,
            rolling_window_episodes=1,
            warmup_episodes=1,
        )
    )
    overflowing.observe_episode_cost(0.0)
    before_overflow = overflowing.state_dict()
    _assert_raises(
        lambda: overflowing.observe_episode_cost(0.0),
        FloatingPointError,
    )
    assert overflowing.state_dict() == before_overflow


def _test_evaluation_override_and_planner_isolation() -> None:
    checkpoint_config = _full_config(enabled=True, alpha=0.0)
    checkpoint_config["integral_lagrangian"] = copy.deepcopy(
        DEFAULT_INTEGRAL_LAGRANGIAN_CONFIG
    )
    original_config = copy.deepcopy(checkpoint_config)
    checkpoint_safety = resolved_safety_mpc_config(checkpoint_config)

    inherited = resolve_evaluation_integral_lagrangian_config(
        checkpoint_config,
        override=None,
        initial_alpha=None,
        integral_gain=None,
        cost_limit=None,
        alpha_max=None,
        rolling_window_episodes=None,
        warmup_episodes=None,
    )
    assert inherited == DEFAULT_INTEGRAL_LAGRANGIAN_CONFIG
    explicit = resolve_evaluation_integral_lagrangian_config(
        checkpoint_config,
        override="enabled",
        initial_alpha=0.1,
        integral_gain=0.2,
        cost_limit=0.03,
        alpha_max=0.4,
        rolling_window_episodes=3,
        warmup_episodes=2,
    )
    assert explicit == {
        "enabled": True,
        "initial_alpha": 0.1,
        "integral_gain": 0.2,
        "cost_limit": 0.03,
        "alpha_max": 0.4,
        "rolling_window_episodes": 3,
        "warmup_episodes": 2,
    }
    assert checkpoint_config == original_config
    _assert_raises(
        lambda: resolve_evaluation_integral_lagrangian_config(
            checkpoint_config,
            override=None,
            initial_alpha=0.1,
            integral_gain=None,
            cost_limit=None,
            alpha_max=None,
            rolling_window_episodes=None,
            warmup_episodes=None,
        ),
        ValueError,
    )
    _assert_raises(
        lambda: validate_evaluation_planner_compatibility(
            {**checkpoint_safety, "enabled": False}, explicit
        ),
        ValueError,
    )
    validate_evaluation_planner_compatibility(
        {**checkpoint_safety, "enabled": True}, explicit
    )
    _assert_raises(
        lambda: validate_evaluation_planner_compatibility(
            {**checkpoint_safety, "enabled": True},
            explicit,
            mpc_enabled=False,
        ),
        ValueError,
    )
    _assert_raises(
        lambda: validate_evaluation_planner_compatibility(
            {**checkpoint_safety, "enabled": True, "alpha": 0.1},
            inherited,
            mpc_enabled=False,
        ),
        ValueError,
    )
    validate_evaluation_planner_compatibility(
        {**checkpoint_safety, "enabled": True, "alpha": 0.0},
        inherited,
        mpc_enabled=False,
    )

    reference = _agent(checkpoint_config, seed=7001)
    checkpoint_state = copy.deepcopy(reference.state_dict())
    checkpoint_metadata = copy.deepcopy(checkpoint_state["safety_mpc_config"])
    fixed_settings = resolve_evaluation_safety_mpc_config(
        checkpoint_config,
        override="enabled",
        alpha=0.1,
    )
    disabled_integral = resolve_evaluation_integral_lagrangian_config(
        checkpoint_config,
        override="disabled",
        initial_alpha=None,
        integral_gain=None,
        cost_limit=None,
        alpha_max=None,
        rolling_window_episodes=None,
        warmup_episodes=None,
    )
    assert not disabled_integral["enabled"]

    observation = np.linspace(-0.8, 0.8, 14, dtype=np.float32)
    actions = []
    for _ in range(2):
        agent = _agent(checkpoint_config, seed=7002)
        agent.load_state_dict(checkpoint_state)
        apply_evaluation_safety_mpc_config(
            agent,
            checkpoint_settings=checkpoint_safety,
            evaluation_settings=fixed_settings,
        )
        torch.manual_seed(7003)
        actions.append(
            agent.act(observation, first_step=True, eval_mode=True)
        )
    np.testing.assert_array_equal(actions[0], actions[1])

    agent = _agent(checkpoint_config, seed=7004)
    agent.load_state_dict(checkpoint_state)
    model_hash = state_sha256(agent.model.state_dict())
    model_optimizer_hash = state_sha256(agent.model_optimizer.state_dict())
    policy_optimizer_hash = state_sha256(agent.policy_optimizer.state_dict())
    scale_hash = state_sha256(agent.scale.state_dict())
    update_count = agent.update_count
    apply_evaluation_safety_mpc_config(
        agent,
        checkpoint_settings=checkpoint_safety,
        evaluation_settings=fixed_settings,
    )
    assert agent.safety_mpc_alpha == 0.0
    assert agent.safety_mpc_runtime_alpha == 0.1
    assert agent.state_dict()["safety_mpc_config"] == checkpoint_metadata

    controller = IntegralLagrangianController(
        _config(
            warmup_episodes=1,
            rolling_window_episodes=1,
            integral_gain=0.1,
            cost_limit=0.0,
        )
    )
    agent.set_safety_mpc_runtime_alpha(
        controller.current_alpha,
        enabled=True,
    )
    controller.observe_episode_cost(1.0)
    update_record = controller.observe_episode_cost(1.0)
    # Completing an episode updates only the controller. Runtime alpha remains
    # fixed until the next explicit episode-boundary handoff.
    assert controller.current_alpha > agent.safety_mpc_runtime_alpha
    _assert_close(agent.safety_mpc_runtime_alpha, 0.1)
    agent.set_safety_mpc_runtime_alpha(
        update_record["alpha_after_update"],
        enabled=True,
    )
    _assert_close(agent.safety_mpc_runtime_alpha, controller.current_alpha)

    with patch.object(
        agent.model.safety_curvature_head,
        "forward",
        side_effect=AssertionError(
            "Curvature must remain monitoring-only in Safety-MPC"
        ),
    ):
        torch.manual_seed(7005)
        action = agent.act(observation, first_step=True, eval_mode=True)
    assert np.all(np.isfinite(action))
    assert state_sha256(agent.model.state_dict()) == model_hash
    assert state_sha256(agent.model_optimizer.state_dict()) == (
        model_optimizer_hash
    )
    assert state_sha256(agent.policy_optimizer.state_dict()) == (
        policy_optimizer_hash
    )
    assert state_sha256(agent.scale.state_dict()) == scale_hash
    assert agent.update_count == update_count
    assert agent.state_dict()["safety_mpc_config"] == checkpoint_metadata
    assert all(parameter.grad is None for parameter in agent.model.parameters())

    for invalid_alpha, expected_error in (
        (True, TypeError),
        ("0.1", TypeError),
        (-0.1, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (1.0e-300, ValueError),
        (1.0e300, ValueError),
    ):
        runtime_snapshot = (
            agent.safety_mpc_runtime_enabled,
            agent.safety_mpc_runtime_alpha,
            agent.safety_mpc_active,
            copy.deepcopy(agent.last_safety_mpc_metrics),
        )
        _assert_raises(
            lambda invalid_alpha=invalid_alpha: (
                agent.set_safety_mpc_runtime_alpha(invalid_alpha)
            ),
            expected_error,
        )
        assert (
            agent.safety_mpc_runtime_enabled,
            agent.safety_mpc_runtime_alpha,
            agent.safety_mpc_active,
            agent.last_safety_mpc_metrics,
        ) == runtime_snapshot

    toggling = IntegralLagrangianController(
        _config(
            initial_alpha=0.05,
            integral_gain=1.0,
            cost_limit=0.5,
            alpha_max=0.5,
            rolling_window_episodes=1,
            warmup_episodes=1,
        )
    )
    toggling.observe_episode_cost(0.0)
    toggling.observe_episode_cost(0.0)
    agent.set_safety_mpc_runtime_alpha(toggling.current_alpha, enabled=True)
    assert not agent.safety_mpc_active
    toggling.observe_episode_cost(1.0)
    agent.set_safety_mpc_runtime_alpha(toggling.current_alpha, enabled=True)
    assert agent.safety_mpc_active

    # The execution-layer intervention implementation remains byte-for-byte
    # outside this feature's modification surface.
    _test_intervention_mask_unchanged()


def main() -> None:
    _test_exact_update_warmup_and_window()
    _test_clipping_and_integral_direction()
    _test_zero_gain_equal_limit_and_reset()
    _test_six_episode_warmup_boundary_and_json()
    _test_config_and_runtime_validation()
    _test_evaluation_override_and_planner_isolation()
    print(
        "PASS: exact episode-level Integral Lagrangian updates, warmup/window, "
        "bounds, reset, strict config/JSON trajectories, evaluation-only "
        "runtime-alpha isolation, fixed-alpha determinism, Translation-only "
        "planning, and unchanged model/optimizers/intervention mask."
    )


if __name__ == "__main__":
    main()
