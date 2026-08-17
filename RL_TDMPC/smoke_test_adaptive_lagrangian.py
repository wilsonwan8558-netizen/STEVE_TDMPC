"""Simulation-free checks for evaluation-only adaptive Lagrangian Safety-MPC."""

from __future__ import annotations

import copy
import json
import math
import warnings
from typing import Any, Callable

import numpy as np
import torch

from evaluate import (
    resolve_evaluation_adaptive_lagrangian_config,
    resolve_evaluation_safety_mpc_config,
    state_sha256,
)
from smoke_test_safety_mpc_active import _agent, _full_config
from tdmpc2.adaptive_lagrangian import (
    DEFAULT_ADAPTIVE_LAGRANGIAN_CONFIG,
    AdaptiveLagrangianController,
    validate_adaptive_lagrangian_config,
)
from tdmpc2.shadow_planner import capture_device_torch_rng_state


def _assert_raises(function: Callable[[], Any], exception_type: type) -> None:
    try:
        function()
    except exception_type:
        return
    raise AssertionError(
        f"Expected {exception_type.__name__}, but no exception was raised"
    )


def _config(**overrides: float) -> dict[str, float]:
    config = copy.deepcopy(DEFAULT_ADAPTIVE_LAGRANGIAN_CONFIG)
    config.update(overrides)
    return config


def _test_exact_update_bounds_reset_and_json() -> None:
    controller = AdaptiveLagrangianController(
        _config(
            epsilon=0.1,
            eta=0.2,
            lambda_initial=0.1,
            lambda_min=0.0,
            lambda_max=0.2,
        )
    )
    first = controller.observe_step_risk(0.3)
    np.testing.assert_allclose(first["constraint_residual"], 0.2)
    np.testing.assert_allclose(first["requested_lambda_update"], 0.04)
    np.testing.assert_allclose(first["lambda_before_update"], 0.1)
    np.testing.assert_allclose(first["lambda_after_update"], 0.14)
    assert not first["lower_clipping_active"]
    assert not first["upper_clipping_active"]

    second = controller.observe_step_risk(0.0)
    np.testing.assert_allclose(second["lambda_after_update"], 0.12)
    third = controller.observe_step_risk(0.1)
    np.testing.assert_allclose(third["lambda_after_update"], 0.12)
    assert third["lambda_update_amount"] == 0.0

    upper = controller.observe_step_risk(10.0)
    assert upper["upper_clipping_active"]
    assert upper["lambda_after_update"] == 0.2
    assert controller.current_lambda == 0.2
    summary = controller.summary()
    assert summary["step_update_count"] == 4
    assert len(summary["lambda_state_trajectory"]) == 5
    for name in (
        "applied_lambda_trajectory",
        "predicted_risk_trajectory",
        "constraint_residual_trajectory",
        "lambda_update_trajectory",
    ):
        assert len(summary[name]) == 4
    assert summary["upper_clip_count"] == 1
    assert summary["applied_lambda_min"] >= 0.0
    assert summary["applied_lambda_max"] <= 0.2
    json.loads(json.dumps(summary, allow_nan=False))

    controller.reset_episode()
    reset_state = controller.state_dict()
    assert controller.current_lambda == 0.1
    assert reset_state["lambda_state_trajectory"] == [0.1]
    assert reset_state["applied_lambda_trajectory"] == []
    assert reset_state["predicted_risk_trajectory"] == []
    assert reset_state["constraint_residual_trajectory"] == []
    assert reset_state["lambda_update_trajectory"] == []
    assert reset_state["lower_clip_count"] == 0
    assert reset_state["upper_clip_count"] == 0

    lower_controller = AdaptiveLagrangianController(
        _config(
            epsilon=1.0,
            eta=1.0,
            lambda_initial=0.1,
            lambda_min=0.0,
            lambda_max=0.2,
        )
    )
    lower = lower_controller.observe_step_risk(0.0)
    assert lower["lower_clipping_active"]
    assert lower["lambda_after_update"] == 0.0
    assert lower_controller.current_lambda >= 0.0

    zero_gain = AdaptiveLagrangianController(
        _config(eta=0.0, lambda_initial=0.07)
    )
    for risk in (0.0, 0.005, 1.0):
        record = zero_gain.observe_step_risk(risk)
        assert record["lambda_before_update"] == 0.07
        assert record["lambda_after_update"] == 0.07


def _test_strict_transactional_validation() -> None:
    assert validate_adaptive_lagrangian_config(
        DEFAULT_ADAPTIVE_LAGRANGIAN_CONFIG
    ) == DEFAULT_ADAPTIVE_LAGRANGIAN_CONFIG
    invalid_configs = []
    missing = _config()
    missing.pop("eta")
    invalid_configs.append((missing, KeyError))
    extra = _config()
    extra["warmup"] = 1.0
    invalid_configs.append((extra, ValueError))
    for key in DEFAULT_ADAPTIVE_LAGRANGIAN_CONFIG:
        for invalid in (True, float("nan"), float("inf"), -0.1):
            candidate = _config()
            candidate[key] = invalid
            invalid_configs.append(
                (candidate, TypeError if invalid is True else ValueError)
            )
    invalid_configs.extend(
        (
            (_config(lambda_min=0.2, lambda_max=0.2), ValueError),
            (_config(lambda_initial=0.3), ValueError),
            (_config(lambda_max=1.0e100), ValueError),
            (
                _config(
                    lambda_min=0.1,
                    lambda_initial=0.1000000001,
                    lambda_max=0.1000000002,
                ),
                ValueError,
            ),
        )
    )
    for candidate, error in invalid_configs:
        _assert_raises(
            lambda candidate=candidate: validate_adaptive_lagrangian_config(
                candidate
            ),
            error,
        )

    controller = AdaptiveLagrangianController(_config())
    before = controller.state_dict()
    for invalid, error in (
        (True, TypeError),
        (-0.1, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
    ):
        _assert_raises(
            lambda invalid=invalid: controller.observe_step_risk(invalid),
            error,
        )
        assert controller.state_dict() == before


def _run_plans(agent, observations: list[np.ndarray], *, seed: int):
    torch.manual_seed(seed)
    actions = [
        agent.act(
            observation,
            first_step=index == 0,
            eval_mode=True,
        ).copy()
        for index, observation in enumerate(observations)
    ]
    return (
        actions,
        agent.previous_mean.detach().clone(),
        capture_device_torch_rng_state(agent.device),
    )


def _assert_plan_sequences_equal(first, second) -> None:
    first_actions, first_mean, first_rng = first
    second_actions, second_mean, second_rng = second
    assert len(first_actions) == len(second_actions)
    for first_action, second_action in zip(first_actions, second_actions):
        np.testing.assert_array_equal(first_action, second_action)
    assert torch.equal(first_mean, second_mean)
    assert torch.equal(first_rng, second_rng)


def _test_force_active_zero_and_constant_lambda_bitwise_isolation() -> None:
    observations = [
        np.linspace(-0.8, 0.8, 14, dtype=np.float32),
        np.linspace(0.6, -0.6, 14, dtype=np.float32),
    ]

    zero_config = _full_config(enabled=True, alpha=0.0, iterations=3)
    inactive = _agent(zero_config, seed=7201)
    force_zero = _agent(zero_config, seed=7202)
    force_zero.load_state_dict(copy.deepcopy(inactive.state_dict()))
    configured_hash = state_sha256(force_zero.state_dict())
    force_zero.set_safety_mpc_runtime_alpha(
        0.0,
        enabled=True,
        force_active=True,
    )
    assert force_zero.safety_mpc_active
    assert force_zero.safety_mpc_runtime_force_active
    assert state_sha256(force_zero.state_dict()) == configured_hash
    _assert_plan_sequences_equal(
        _run_plans(inactive, observations, seed=7203),
        _run_plans(force_zero, observations, seed=7203),
    )
    zero_metrics = force_zero.last_safety_mpc_metrics
    assert zero_metrics is not None
    assert zero_metrics["safety_mpc_selected_penalty"] == 0.0
    assert zero_metrics["safety_mpc_penalty_to_task_scale"] == 0.0
    assert "safety_mpc_safe_top_task_score" in zero_metrics
    assert "safety_mpc_safe_top_planner_score" in zero_metrics

    fixed_config = _full_config(enabled=True, alpha=0.1, iterations=3)
    fixed = _agent(fixed_config, seed=7210)
    force_fixed = _agent(fixed_config, seed=7211)
    force_fixed.load_state_dict(copy.deepcopy(fixed.state_dict()))
    force_fixed.set_safety_mpc_runtime_alpha(
        0.1,
        enabled=True,
        force_active=True,
    )
    _assert_plan_sequences_equal(
        _run_plans(fixed, observations, seed=7212),
        _run_plans(force_fixed, observations, seed=7212),
    )
    fixed_metric_names = {
        "safety_mpc_task_scale",
        "safety_mpc_selected_risk",
        "safety_mpc_selected_penalty",
        "safety_mpc_candidate_risk_mean",
        "safety_mpc_candidate_risk_max",
        "safety_mpc_penalty_to_task_scale",
        "safety_mpc_same_population_task_sacrifice",
    }
    assert set(fixed.last_safety_mpc_metrics) == fixed_metric_names
    assert set(force_fixed.last_safety_mpc_metrics) == fixed_metric_names | {
        "safety_mpc_safe_top_task_score",
        "safety_mpc_safe_top_planner_score",
    }


def _test_runtime_override_numerical_semantics() -> None:
    observations = [
        np.linspace(-0.7, 0.7, 14, dtype=np.float32),
        np.linspace(0.5, -0.5, 14, dtype=np.float32),
    ]

    configured_zero = _agent(
        _full_config(enabled=True, alpha=0.0, iterations=3),
        seed=7250,
    )
    with torch.no_grad():
        configured_zero.model.safety_translation_error_head.weight.zero_()
        configured_zero.model.safety_translation_error_head.bias.fill_(0.2)
    native_fixed = _agent(
        _full_config(enabled=True, alpha=0.1, iterations=3),
        seed=7251,
    )
    native_fixed.model.load_state_dict(copy.deepcopy(configured_zero.model.state_dict()))
    native_fixed.scale.load_state_dict(copy.deepcopy(configured_zero.scale.state_dict()))
    native_fixed.previous_mean = configured_zero.previous_mean.detach().clone()
    configured_zero.set_safety_mpc_runtime_alpha(0.1, enabled=True)
    _assert_plan_sequences_equal(
        _run_plans(configured_zero, observations, seed=7252),
        _run_plans(native_fixed, observations, seed=7252),
    )
    metrics = configured_zero.last_safety_mpc_metrics
    assert metrics is not None
    assert metrics["safety_mpc_selected_risk"] > 0.0
    np.testing.assert_allclose(
        metrics["safety_mpc_penalty_to_task_scale"],
        0.1 * metrics["safety_mpc_selected_risk"],
        rtol=1.0e-6,
        atol=1.0e-8,
    )
    assert metrics["safety_mpc_selected_penalty"] > 0.0

    configured_active = _agent(
        _full_config(enabled=True, alpha=0.2, iterations=3),
        seed=7260,
    )
    native_disabled = _agent(
        _full_config(enabled=False, alpha=0.0, iterations=3),
        seed=7261,
    )
    native_disabled.model.load_state_dict(
        copy.deepcopy(configured_active.model.state_dict())
    )
    native_disabled.scale.load_state_dict(
        copy.deepcopy(configured_active.scale.state_dict())
    )
    native_disabled.previous_mean = configured_active.previous_mean.detach().clone()
    configured_active.set_safety_mpc_runtime_alpha(0.0, enabled=False)
    _assert_plan_sequences_equal(
        _run_plans(configured_active, observations, seed=7262),
        _run_plans(native_disabled, observations, seed=7262),
    )
    assert configured_active.last_safety_mpc_metrics is None


def _test_runtime_setter_checkpoint_isolation_and_feedback_recovery() -> None:
    config = _full_config(enabled=False, alpha=0.2)
    agent = _agent(config, seed=7301)
    configured_state = copy.deepcopy(agent.state_dict())
    configured_hash = state_sha256(configured_state)
    runtime_before = (
        agent.safety_mpc_runtime_enabled,
        agent.safety_mpc_runtime_alpha,
        agent.safety_mpc_runtime_force_active,
        agent.safety_mpc_active,
    )
    for callback, error in (
        (
            lambda: agent.set_safety_mpc_runtime_alpha(float("nan")),
            ValueError,
        ),
        (
            lambda: agent.set_safety_mpc_runtime_alpha(-0.1),
            ValueError,
        ),
        (
            lambda: agent.set_safety_mpc_runtime_alpha(True),
            TypeError,
        ),
        (
            lambda: agent.set_safety_mpc_runtime_alpha(0.1, enabled=1),
            TypeError,
        ),
        (
            lambda: agent.set_safety_mpc_runtime_alpha(
                0.0, enabled=False, force_active=True
            ),
            ValueError,
        ),
        (
            lambda: agent.set_safety_mpc_runtime_alpha(1.0e-300),
            ValueError,
        ),
        (
            lambda: agent.set_safety_mpc_runtime_alpha(1.0e300),
            ValueError,
        ),
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _assert_raises(callback, error)
        assert runtime_before == (
            agent.safety_mpc_runtime_enabled,
            agent.safety_mpc_runtime_alpha,
            agent.safety_mpc_runtime_force_active,
            agent.safety_mpc_active,
        )

    agent.set_safety_mpc_runtime_alpha(
        0.0,
        enabled=True,
        force_active=True,
    )
    assert state_sha256(agent.state_dict()) == configured_hash
    assert agent.state_dict()["safety_mpc_config"] == configured_state[
        "safety_mpc_config"
    ]
    clone = _agent(config, seed=7302)
    clone.load_state_dict(copy.deepcopy(agent.state_dict()))
    assert not clone.safety_mpc_runtime_enabled
    assert clone.safety_mpc_runtime_alpha == 0.2
    assert not clone.safety_mpc_runtime_force_active
    assert not clone.safety_mpc_active

    controller = AdaptiveLagrangianController(
        _config(
            epsilon=0.1,
            eta=0.5,
            lambda_initial=0.0,
            lambda_min=0.0,
            lambda_max=0.2,
        )
    )
    assert controller.current_lambda == 0.0
    controller.observe_step_risk(0.3)
    assert controller.current_lambda > 0.0
    agent.set_safety_mpc_runtime_alpha(
        controller.current_lambda,
        enabled=True,
        force_active=True,
    )
    assert agent.safety_mpc_active
    assert agent.safety_mpc_runtime_alpha > 0.0


def _test_evaluation_mode_resolution() -> None:
    defaults = resolve_evaluation_adaptive_lagrangian_config(
        mode="lagrangian",
        epsilon=None,
        eta=None,
        lambda_initial=None,
        lambda_min=None,
        lambda_max=None,
    )
    assert defaults == DEFAULT_ADAPTIVE_LAGRANGIAN_CONFIG
    resolved = resolve_evaluation_adaptive_lagrangian_config(
        mode="lagrangian",
        epsilon=0.01,
        eta=0.05,
        lambda_initial=0.08,
        lambda_min=0.0,
        lambda_max=0.15,
    )
    assert resolved == {
        "epsilon": 0.01,
        "eta": 0.05,
        "lambda_initial": 0.08,
        "lambda_min": 0.0,
        "lambda_max": 0.15,
    }
    _assert_raises(
        lambda: resolve_evaluation_adaptive_lagrangian_config(
            mode="enabled",
            epsilon=0.01,
            eta=None,
            lambda_initial=None,
            lambda_min=None,
            lambda_max=None,
        ),
        ValueError,
    )
    checkpoint_config = _full_config(enabled=False, alpha=0.2)
    lagrangian = resolve_evaluation_safety_mpc_config(
        checkpoint_config,
        override="lagrangian",
        alpha=None,
        lagrangian_initial=0.08,
    )
    assert lagrangian["enabled"] is True
    assert lagrangian["alpha"] == 0.08
    _assert_raises(
        lambda: resolve_evaluation_safety_mpc_config(
            checkpoint_config,
            override="lagrangian",
            alpha=0.1,
            lagrangian_initial=0.08,
        ),
        ValueError,
    )


def main() -> None:
    _test_exact_update_bounds_reset_and_json()
    _test_strict_transactional_validation()
    _test_force_active_zero_and_constant_lambda_bitwise_isolation()
    _test_runtime_override_numerical_semantics()
    _test_runtime_setter_checkpoint_isolation_and_feedback_recovery()
    _test_evaluation_mode_resolution()
    print(
        "PASS: adaptive dual equation/bounds/reset, strict finite config, "
        "zero-lambda feedback continuity, fixed/zero-penalty bitwise planner "
        "isolation, runtime override semantics, RNG stability, checkpoint "
        "isolation, and CLI mode resolution"
    )


if __name__ == "__main__":
    main()
