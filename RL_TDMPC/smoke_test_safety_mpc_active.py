"""Focused, simulation-free checks for active Translation-only Safety-MPC."""

from __future__ import annotations

import copy
import inspect
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import eve.intervention.monoplanestatic as monoplane_module
from eve.intervention import TRANSLATION_BLOCK_REASON_NAMES
from eve.intervention.monoplanestatic import MonoPlaneStatic
from evaluate import (
    apply_evaluation_safety_mpc_config,
    paths_alias,
    resolve_evaluation_safety_mpc_config,
    state_sha256,
)
from tdmpc2.agent import TDMPC2Agent
from tdmpc2.common import (
    DEFAULT_SAFETY_MPC_CONFIG,
    build_safety_mpc_agent_config,
    load_config,
    load_torch_checkpoint,
)
from tdmpc2.replay_buffer import EpisodeReplayBuffer, REPLAY_SAFETY_COST_NAMES
from tdmpc2.shadow_planner import (
    capture_device_torch_rng_state,
    capture_planner_snapshot,
    make_planner_noise_schedule,
    run_translation_shadow_plan,
)
from train import (
    DEFAULT_CONFIG,
    SAFETY_MPC_CONFIG_SCHEMA_VERSION,
    build_agent_config,
    resolved_safety_mpc_config,
    save_checkpoint,
    validate_checkpoint_schema,
)


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


def _full_config(
    *,
    enabled: bool,
    alpha: float,
    horizon: int = 3,
    iterations: int = 2,
    num_samples: int = 8,
    num_elites: int = 3,
    num_pi_trajs: int = 2,
) -> dict:
    config = load_config(DEFAULT_CONFIG)
    config["model"].update(
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
    config["training"].update(
        {
            "horizon": int(horizon),
            "batch_size": 2,
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
    config["planning"].update(
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
    config["safety"].update(
        {
            "loss_coef": 1.0,
            "curvature_loss_coef": 1.0,
            "translation_error_loss_coef": 1.0,
            "curvature_scale_mm_inv": 0.1,
            "translation_error_scale": 1.0,
        }
    )
    config["safety_mpc"] = {
        "enabled": bool(enabled),
        "alpha": float(alpha),
        "translation_risk_cap": 1.0,
        "minimum_task_scale": 1.0e-6,
        "aggregation": "max",
    }
    return config


def _agent(config: dict, *, seed: int) -> TDMPC2Agent:
    torch.manual_seed(seed)
    return TDMPC2Agent(
        14,
        2,
        build_agent_config(config),
        episode_length=20,
        device=torch.device("cpu"),
    )


def _test_strict_config_builder() -> None:
    expected_default = {
        "safety_mpc_enabled": False,
        "safety_mpc_alpha": 0.2,
        "safety_mpc_translation_risk_cap": 1.0,
        "safety_mpc_minimum_task_scale": 1.0e-6,
        "safety_mpc_aggregation": "max",
    }
    assert build_safety_mpc_agent_config({}) == expected_default
    assert DEFAULT_SAFETY_MPC_CONFIG == {
        "enabled": False,
        "alpha": 0.2,
        "translation_risk_cap": 1.0,
        "minimum_task_scale": 1.0e-6,
        "aggregation": "max",
    }

    valid = copy.deepcopy(DEFAULT_SAFETY_MPC_CONFIG)
    valid.update({"enabled": True, "alpha": np.float32(0.3)})
    parsed = build_safety_mpc_agent_config({"safety_mpc": valid})
    assert parsed["safety_mpc_enabled"] is True
    np.testing.assert_allclose(parsed["safety_mpc_alpha"], 0.3)

    for invalid_section in (None, [], "enabled"):
        _assert_raises(
            lambda invalid_section=invalid_section: (
                build_safety_mpc_agent_config(
                    {"safety_mpc": invalid_section}
                )
            ),
            TypeError,
        )

    for key in DEFAULT_SAFETY_MPC_CONFIG:
        missing = copy.deepcopy(DEFAULT_SAFETY_MPC_CONFIG)
        missing.pop(key)
        _assert_raises(
            lambda missing=missing: build_safety_mpc_agent_config(
                {"safety_mpc": missing}
            ),
            KeyError,
        )
    extra = copy.deepcopy(DEFAULT_SAFETY_MPC_CONFIG)
    extra["curvature_weight"] = 1.0
    _assert_raises(
        lambda: build_safety_mpc_agent_config({"safety_mpc": extra}),
        ValueError,
    )

    for invalid_enabled in (0, 1, np.bool_(True), "false"):
        candidate = copy.deepcopy(DEFAULT_SAFETY_MPC_CONFIG)
        candidate["enabled"] = invalid_enabled
        _assert_raises(
            lambda candidate=candidate: build_safety_mpc_agent_config(
                {"safety_mpc": candidate}
            ),
            TypeError,
        )
    for key, invalid_values, expected_error in (
        ("alpha", (-0.1, float("nan"), float("inf")), ValueError),
        (
            "translation_risk_cap",
            (0.0, -1.0, float("nan"), float("inf")),
            ValueError,
        ),
        (
            "minimum_task_scale",
            (0.0, -1.0, float("nan"), float("inf")),
            ValueError,
        ),
        ("alpha", (True, "0.2", None), TypeError),
    ):
        for invalid in invalid_values:
            candidate = copy.deepcopy(DEFAULT_SAFETY_MPC_CONFIG)
            candidate[key] = invalid
            _assert_raises(
                lambda candidate=candidate: build_safety_mpc_agent_config(
                    {"safety_mpc": candidate}
                ),
                expected_error,
            )
    for aggregation in ("discounted_sum", "", 1, None):
        candidate = copy.deepcopy(DEFAULT_SAFETY_MPC_CONFIG)
        candidate["aggregation"] = aggregation
        _assert_raises(
            lambda candidate=candidate: build_safety_mpc_agent_config(
                {"safety_mpc": candidate}
            ),
            (TypeError, ValueError),
        )


def _assert_inactive_bitwise_identity(*, enabled: bool, alpha: float) -> None:
    config = _full_config(enabled=enabled, alpha=alpha, iterations=3)
    agent = _agent(config, seed=4801)
    assert not agent.safety_mpc_active
    observation = np.linspace(-1.0, 1.0, 14, dtype=np.float32)
    previous = torch.tensor(
        [[0.25, -0.15], [0.10, 0.30], [-0.20, 0.05]],
        dtype=torch.float32,
    )
    agent.previous_mean.copy_(previous)
    torch.manual_seed(4802)
    initial_rng = torch.get_rng_state().clone()

    snapshot = capture_planner_snapshot(agent, observation, first_step=False)
    schedule = make_planner_noise_schedule(
        agent,
        initial_torch_state=initial_rng,
    )
    baseline = run_translation_shadow_plan(
        agent,
        snapshot,
        schedule,
        safety_weight=0.0,
        aggregation="max",
    )
    assert torch.equal(agent.previous_mean, previous)
    assert torch.equal(torch.get_rng_state(), initial_rng)

    agent.previous_mean.copy_(previous)
    torch.set_rng_state(initial_rng)
    with patch.object(
        agent.model,
        "translation_safety_transformed",
        side_effect=AssertionError("Inactive planning called Translation Safety"),
    ) as prediction_spy, patch.object(
        agent.model,
        "decode_translation_safety_transformed",
        side_effect=AssertionError("Inactive planning decoded Translation Safety"),
    ) as decode_spy, patch.object(
        agent,
        "_apply_safety_mpc_penalty",
        side_effect=AssertionError("Inactive planning applied a Safety penalty"),
    ) as penalty_spy:
        action = agent._plan(
            agent._tensor_observation(observation),
            first_step=False,
            eval_mode=True,
        )
    assert prediction_spy.call_count == 0
    assert decode_spy.call_count == 0
    assert penalty_spy.call_count == 0
    assert torch.equal(action, baseline.action)
    assert torch.equal(agent.previous_mean, baseline.mean)
    assert torch.equal(torch.get_rng_state(), schedule.core_end_torch_state)
    assert agent.last_safety_mpc_metrics is None


def _test_disabled_and_alpha_zero_identity() -> None:
    _assert_inactive_bitwise_identity(enabled=False, alpha=0.2)
    _assert_inactive_bitwise_identity(enabled=True, alpha=0.0)

    # Safety-MPC is an MPPI feature only. A policy-only agent must not call it.
    config = _full_config(enabled=True, alpha=0.2)
    config["planning"]["mpc"] = False
    agent = _agent(config, seed=4803)
    with patch.object(
        agent.model,
        "translation_safety_transformed",
        side_effect=AssertionError("Policy-only action called Safety-MPC"),
    ) as spy:
        action = agent.act(
            np.zeros(14, dtype=np.float32),
            first_step=True,
            eval_mode=True,
        )
    assert spy.call_count == 0
    assert agent.last_safety_mpc_metrics is None
    assert action.shape == (2,) and action.dtype == np.float32
    assert np.all(np.isfinite(action))


def _test_penalty_formula_scale_floor_and_finiteness() -> None:
    config = _full_config(enabled=True, alpha=0.2)
    agent = _agent(config, seed=4810)
    task = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    risk = torch.tensor([0.0, 0.25, 0.5, 1.0])
    planner, scale = agent._apply_safety_mpc_penalty(task, risk)
    expected_scale = torch.tensor(math.sqrt(1.25))
    expected_penalty = 0.2 * expected_scale * risk
    torch.testing.assert_close(scale, expected_scale)
    torch.testing.assert_close(
        planner,
        task - expected_penalty.unsqueeze(-1),
    )

    for constant_task in (
        torch.full((4, 1), 2.0),
        torch.full((1, 1), -3.0),
    ):
        constant_risk = torch.ones(constant_task.shape[0])
        constant_planner, constant_scale = agent._apply_safety_mpc_penalty(
            constant_task,
            constant_risk,
        )
        torch.testing.assert_close(
            constant_scale,
            torch.tensor(agent.safety_mpc_minimum_task_scale),
        )
        assert torch.isfinite(constant_planner).all()
        torch.testing.assert_close(
            constant_planner,
            constant_task
            - (
                agent.safety_mpc_alpha
                * agent.safety_mpc_minimum_task_scale
                * constant_risk
            ).unsqueeze(-1),
        )

    inactive = _agent(
        _full_config(enabled=False, alpha=0.2),
        seed=4811,
    )
    _assert_raises(
        lambda: inactive._apply_safety_mpc_penalty(task, risk),
        RuntimeError,
    )
    for invalid_task, invalid_risk, error in (
        (task.squeeze(-1), risk, ValueError),
        (task, risk.unsqueeze(-1), ValueError),
        (
            torch.tensor([[0.0], [float("nan")], [1.0], [2.0]]),
            risk,
            FloatingPointError,
        ),
        (
            task,
            torch.tensor([0.0, float("inf"), 0.0, 0.0]),
            FloatingPointError,
        ),
        (
            task,
            torch.tensor([0.0, -0.1, 0.0, 0.0]),
            FloatingPointError,
        ),
        (
            task,
            torch.tensor([0.0, 1.01, 0.0, 0.0]),
            FloatingPointError,
        ),
    ):
        _assert_raises(
            lambda invalid_task=invalid_task, invalid_risk=invalid_risk: (
                agent._apply_safety_mpc_penalty(
                    invalid_task,
                    invalid_risk,
                )
            ),
            error,
        )


def _test_active_calls_translation_only_and_recomputes_scale() -> None:
    config = _full_config(enabled=True, alpha=0.2, iterations=3)
    agent = _agent(config, seed=4820)
    assert agent.safety_mpc_active
    observation = np.linspace(0.8, -0.8, 14, dtype=np.float32)
    torch.manual_seed(4821)
    initial_rng = capture_device_torch_rng_state(agent.device)
    schedule = make_planner_noise_schedule(
        agent,
        initial_torch_state=initial_rng,
    )

    original_penalty = agent._apply_safety_mpc_penalty
    with patch.object(
        agent.model.safety_curvature_head,
        "forward",
        side_effect=AssertionError("Active planning called Curvature branch"),
    ) as curvature_spy, patch.object(
        agent.model,
        "safety_transformed",
        side_effect=AssertionError("Active planning called two-channel Safety"),
    ) as two_channel_spy, patch.object(
        agent.model,
        "translation_safety_transformed",
        wraps=agent.model.translation_safety_transformed,
    ) as translation_spy, patch.object(
        agent.model,
        "decode_translation_safety_transformed",
        wraps=agent.model.decode_translation_safety_transformed,
    ) as decode_spy, patch.object(
        agent,
        "_apply_safety_mpc_penalty",
        wraps=original_penalty,
    ) as penalty_spy:
        action = agent.act(
            observation,
            first_step=True,
            eval_mode=True,
        )

    assert curvature_spy.call_count == 0
    assert two_channel_spy.call_count == 0
    assert penalty_spy.call_count == int(agent.config["iterations"])
    expected_prediction_calls = (
        int(agent.config["iterations"]) + 1
    ) * agent.horizon
    assert translation_spy.call_count == expected_prediction_calls
    assert decode_spy.call_count == expected_prediction_calls
    batch_sizes = [
        int(call.args[0].shape[0]) for call in translation_spy.call_args_list
    ]
    assert batch_sizes.count(int(agent.config["num_samples"])) == (
        int(agent.config["iterations"]) * agent.horizon
    )
    assert batch_sizes.count(1) == agent.horizon

    for call in penalty_spy.call_args_list:
        task_values, candidate_risk = call.args
        assert task_values.shape == (int(agent.config["num_samples"]), 1)
        assert candidate_risk.shape == (int(agent.config["num_samples"]),)
        assert torch.isfinite(task_values).all()
        assert torch.isfinite(candidate_risk).all()
        assert torch.all(candidate_risk >= 0.0)
        assert torch.all(
            candidate_risk <= agent.safety_mpc_translation_risk_cap
        )

    metrics = agent.last_safety_mpc_metrics
    assert metrics is not None
    expected_metric_names = {
        "safety_mpc_task_scale",
        "safety_mpc_selected_risk",
        "safety_mpc_selected_penalty",
        "safety_mpc_candidate_risk_mean",
        "safety_mpc_candidate_risk_max",
        "safety_mpc_penalty_to_task_scale",
        "safety_mpc_same_population_task_sacrifice",
    }
    assert set(metrics) == expected_metric_names
    assert all(math.isfinite(value) for value in metrics.values())
    final_task_values = penalty_spy.call_args_list[-1].args[0]
    expected_final_scale = max(
        float(final_task_values.squeeze(-1).std(unbiased=False)),
        agent.safety_mpc_minimum_task_scale,
    )
    np.testing.assert_allclose(
        metrics["safety_mpc_task_scale"],
        expected_final_scale,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        metrics["safety_mpc_penalty_to_task_scale"],
        (
            agent.safety_mpc_alpha
            * metrics["safety_mpc_selected_risk"]
        ),
        rtol=1e-6,
    )
    assert torch.equal(torch.get_rng_state(), schedule.core_end_torch_state)
    assert action.shape == (2,) and action.dtype == np.float32
    assert np.all(np.isfinite(action))
    assert np.all(np.abs(action) <= 1.0)


def _prediction_of(value: float):
    def prediction(latent, action):
        del action
        return latent.new_full((latent.shape[0], 1), float(value))

    return prediction


def _test_translation_effect_risk_cap_and_invalid_predictions() -> None:
    config = _full_config(enabled=True, alpha=0.2)
    agent = _agent(config, seed=4830)
    latent = torch.zeros(4, agent.model.latent_dim)
    actions = torch.zeros(agent.horizon, 4, agent.action_dim)

    # Curvature is not merely ignored in the score: its branch is not executed.
    with patch.object(
        agent.model.safety_curvature_head,
        "forward",
        side_effect=AssertionError("Translation risk called Curvature"),
    ) as curvature_spy, patch.object(
        agent.model,
        "translation_safety_transformed",
        side_effect=_prediction_of(math.log(2.0)),
    ):
        risk = agent._estimate_translation_trajectory_risk(latent, actions)
    assert curvature_spy.call_count == 0
    torch.testing.assert_close(risk, torch.ones(4))

    # An arbitrarily large finite decoded prediction is capped for planning.
    with patch.object(
        agent.model,
        "translation_safety_transformed",
        side_effect=_prediction_of(80.0),
    ):
        capped = agent._estimate_translation_trajectory_risk(latent, actions)
    torch.testing.assert_close(
        capped,
        torch.full_like(capped, agent.safety_mpc_translation_risk_cap),
    )
    assert torch.isfinite(capped).all()

    task = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    zero_score, _ = agent._apply_safety_mpc_penalty(
        task,
        torch.zeros(4),
    )
    positive_score, _ = agent._apply_safety_mpc_penalty(
        task,
        torch.tensor([0.0, 0.25, 0.5, 1.0]),
    )
    assert torch.equal(zero_score, task)
    assert not torch.equal(positive_score, zero_score)

    previous = agent.previous_mean.detach().clone()
    with patch.object(
        agent.model,
        "translation_safety_transformed",
        side_effect=_prediction_of(float("nan")),
    ):
        _assert_raises(
            lambda: agent.act(
                np.zeros(14, dtype=np.float32),
                first_step=True,
                eval_mode=True,
            ),
            FloatingPointError,
        )
    assert torch.equal(agent.previous_mean, previous)

    with patch.object(
        agent.model,
        "translation_safety_transformed",
        side_effect=_prediction_of(0.0),
    ), patch.object(
        agent.model,
        "decode_translation_safety_transformed",
        side_effect=lambda value: -torch.ones_like(value),
    ):
        _assert_raises(
            lambda: agent._estimate_translation_trajectory_risk(
                latent,
                actions,
            ),
            ValueError,
        )


class _DummyDevice:
    def __init__(self) -> None:
        self.velocity_limit = np.asarray([50.0, 3.14])
        self.length = 100.0
        self.sofa_device = SimpleNamespace(radius=0.5)


class _DummySimulation:
    def __init__(self, inserted_length: float) -> None:
        self.inserted_lengths = [float(inserted_length)]
        self.rotations = [0.0]
        self.dof_positions = [np.zeros(3, dtype=np.float64)]
        self.calls = []

    def step(self, action, duration) -> None:
        self.calls.append((np.asarray(action).copy(), float(duration)))


class _StepComponent:
    def __init__(self, *, image_frequency=None) -> None:
        self.step_count = 0
        if image_frequency is not None:
            self.image_frequency = float(image_frequency)

    def step(self) -> None:
        self.step_count += 1


def _dummy_intervention(inserted_length: float) -> MonoPlaneStatic:
    return MonoPlaneStatic(
        vessel_tree=_StepComponent(),
        devices=[_DummyDevice()],
        simulation=_DummySimulation(inserted_length),
        fluoroscopy=_StepComponent(image_frequency=7.5),
        target=_StepComponent(),
    )


def _test_intervention_mask_unchanged() -> None:
    source = inspect.getsource(MonoPlaneStatic.step)
    assert "safety_mpc" not in source.lower()
    assert "tdmpc" not in source.lower()
    lower_id = TRANSLATION_BLOCK_REASON_NAMES.index(
        "lower_insertion_boundary"
    )
    tree_id = TRANSLATION_BLOCK_REASON_NAMES.index("vessel_tree_end")

    lower = _dummy_intervention(0.0)
    with patch.object(monoplane_module, "at_tree_end", return_value=False):
        lower.step(np.asarray([[-50.0, 1.25]]))
    np.testing.assert_array_equal(
        lower.requested_action,
        np.asarray([[-50.0, 1.25]]),
    )
    np.testing.assert_array_equal(
        lower.applied_action,
        np.asarray([[0.0, 1.25]]),
    )
    assert int(lower.translation_block_reason_ids[0]) == lower_id
    np.testing.assert_array_equal(
        lower.simulation.calls[-1][0],
        np.asarray([[0.0, 1.25]]),
    )

    tree = _dummy_intervention(20.0)
    with patch.object(monoplane_module, "at_tree_end", return_value=True):
        tree.step(np.asarray([[50.0, -0.75]]))
    np.testing.assert_array_equal(
        tree.requested_action,
        np.asarray([[50.0, -0.75]]),
    )
    np.testing.assert_array_equal(
        tree.applied_action,
        np.asarray([[0.0, -0.75]]),
    )
    assert int(tree.translation_block_reason_ids[0]) == tree_id
    np.testing.assert_array_equal(
        tree.simulation.calls[-1][0],
        np.asarray([[0.0, -0.75]]),
    )


def _changed_safety_mpc_value(key: str, value):
    if key == "enabled":
        return not bool(value)
    if key == "alpha":
        return float(value) + 0.1
    if key == "translation_risk_cap":
        return float(value) + 0.5
    if key == "minimum_task_scale":
        return float(value) * 10.0
    if key == "aggregation":
        return "discounted_sum"
    raise AssertionError(key)


def _test_agent_and_checkpoint_strictness() -> None:
    config = _full_config(
        enabled=True,
        alpha=0.2,
        horizon=2,
        iterations=1,
        num_samples=4,
        num_elites=2,
        num_pi_trajs=1,
    )
    agent = _agent(config, seed=4840)
    state = copy.deepcopy(agent.state_dict())
    expected = resolved_safety_mpc_config(config)
    assert state["safety_mpc_config"] == expected

    clone = _agent(config, seed=4841)
    clone.load_state_dict(state, load_optimizers=False)
    assert clone._safety_mpc_state_config() == expected
    for key in expected:
        invalid = copy.deepcopy(state)
        invalid["safety_mpc_config"][key] = _changed_safety_mpc_value(
            key,
            expected[key],
        )
        _assert_raises(
            lambda invalid=invalid: clone.load_state_dict(
                invalid,
                load_optimizers=False,
            ),
            (TypeError, ValueError),
        )
    missing = copy.deepcopy(state)
    missing["safety_mpc_config"].pop("alpha")
    _assert_raises(
        lambda: clone.load_state_dict(missing, load_optimizers=False),
        ValueError,
    )
    unexpected = copy.deepcopy(state)
    unexpected["safety_mpc_config"]["curvature_weight"] = 1.0
    _assert_raises(
        lambda: clone.load_state_dict(unexpected, load_optimizers=False),
        ValueError,
    )

    disabled_config = _full_config(
        enabled=False,
        alpha=0.2,
        horizon=2,
        iterations=1,
        num_samples=4,
        num_elites=2,
        num_pi_trajs=1,
    )
    disabled = _agent(disabled_config, seed=4842)
    legacy_state = copy.deepcopy(disabled.state_dict())
    legacy_state.pop("safety_mpc_config")
    disabled_clone = _agent(disabled_config, seed=4843)
    disabled_clone.load_state_dict(legacy_state, load_optimizers=False)
    _assert_raises(
        lambda: clone.load_state_dict(legacy_state, load_optimizers=False),
        ValueError,
    )

    replay = EpisodeReplayBuffer(
        32,
        14,
        2,
        2,
        2,
        safety_cost_names=REPLAY_SAFETY_COST_NAMES,
        seed=4844,
    )
    with TemporaryDirectory(prefix="steve-active-mpc-smoke-") as directory:
        checkpoint_path = Path(directory) / "active.pt"
        save_checkpoint(
            path=checkpoint_path,
            config=config,
            agent=agent,
            replay=replay,
            total_env_steps=10,
            episode_index=1,
            success_count=0,
            include_replay=False,
        )
        checkpoint = load_torch_checkpoint(
            checkpoint_path,
            map_location="cpu",
        )
    validate_checkpoint_schema(
        checkpoint,
        config=config,
        source="Active Safety-MPC smoke checkpoint",
    )
    assert (
        checkpoint["safety_mpc_config_schema_version"]
        == SAFETY_MPC_CONFIG_SCHEMA_VERSION
    )
    assert checkpoint["safety_mpc_config"] == expected
    assert checkpoint["config"]["safety_mpc"] == expected
    assert checkpoint["agent"]["safety_mpc_config"] == expected

    locations = (
        ("top-level", lambda value: value["safety_mpc_config"]),
        ("embedded", lambda value: value["config"]["safety_mpc"]),
        ("agent", lambda value: value["agent"]["safety_mpc_config"]),
    )
    for _, getter in locations:
        for key in expected:
            invalid = copy.deepcopy(checkpoint)
            target = getter(invalid)
            target[key] = _changed_safety_mpc_value(key, expected[key])
            _assert_raises(
                lambda invalid=invalid: validate_checkpoint_schema(
                    invalid,
                    source="Mutated active Safety-MPC checkpoint",
                ),
                (TypeError, ValueError),
            )
    for location in ("top", "embedded", "agent"):
        partial = copy.deepcopy(checkpoint)
        if location == "top":
            partial.pop("safety_mpc_config")
        elif location == "embedded":
            partial["config"].pop("safety_mpc")
        else:
            partial["agent"].pop("safety_mpc_config")
        _assert_raises(
            lambda partial=partial: validate_checkpoint_schema(
                partial,
                source="Partial active Safety-MPC checkpoint",
            ),
            ValueError,
        )
    bad_version = copy.deepcopy(checkpoint)
    bad_version["safety_mpc_config_schema_version"] += 1
    _assert_raises(
        lambda: validate_checkpoint_schema(
            bad_version,
            source="Wrong Safety-MPC schema checkpoint",
        ),
        ValueError,
    )
    requested_mismatch = copy.deepcopy(config)
    requested_mismatch["safety_mpc"]["alpha"] = 0.3
    _assert_raises(
        lambda: validate_checkpoint_schema(
            checkpoint,
            config=requested_mismatch,
            source="Mismatched requested Safety-MPC checkpoint",
        ),
        ValueError,
    )


def _test_evaluation_only_override_isolation() -> None:
    assert paths_alias(Path("/tmp/eval-checkpoint.pt"), Path("/tmp/eval-checkpoint.pt"))
    assert not paths_alias(
        Path("/tmp/eval-checkpoint.pt"),
        Path("/tmp/eval-output.json"),
    )
    checkpoint_config = _full_config(enabled=True, alpha=0.0)
    checkpoint_settings = resolved_safety_mpc_config(checkpoint_config)
    original_config = copy.deepcopy(checkpoint_config)

    assert resolve_evaluation_safety_mpc_config(
        checkpoint_config,
        override="checkpoint",
        alpha=None,
    ) == checkpoint_settings
    disabled_settings = resolve_evaluation_safety_mpc_config(
        checkpoint_config,
        override="disabled",
        alpha=None,
    )
    assert disabled_settings == {
        **checkpoint_settings,
        "enabled": False,
    }
    enabled_settings = resolve_evaluation_safety_mpc_config(
        checkpoint_config,
        override="enabled",
        alpha=0.1,
    )
    assert enabled_settings == {
        **checkpoint_settings,
        "enabled": True,
        "alpha": 0.1,
    }
    assert checkpoint_config == original_config

    invalid_arguments = (
        ("checkpoint", 0.1),
        ("disabled", 0.1),
        ("enabled", None),
        ("enabled", 0.0),
        ("enabled", -0.1),
        ("enabled", float("nan")),
        ("other", None),
    )
    for override, alpha in invalid_arguments:
        _assert_raises(
            lambda override=override, alpha=alpha: (
                resolve_evaluation_safety_mpc_config(
                    checkpoint_config,
                    override=override,
                    alpha=alpha,
                )
            ),
            ValueError,
        )

    reference = _agent(checkpoint_config, seed=1701)
    checkpoint_state = copy.deepcopy(reference.state_dict())
    checkpoint_model_hash = state_sha256(checkpoint_state["model"])
    observation = np.linspace(-0.7, 0.7, 14, dtype=np.float32)

    disabled_actions = []
    for seed in (1801, 1801):
        candidate = _agent(checkpoint_config, seed=1702)
        candidate.load_state_dict(checkpoint_state)
        config_hash = state_sha256(candidate.config)
        model_hash = state_sha256(candidate.model.state_dict())
        model_optimizer_hash = state_sha256(
            candidate.model_optimizer.state_dict()
        )
        policy_optimizer_hash = state_sha256(
            candidate.policy_optimizer.state_dict()
        )
        apply_evaluation_safety_mpc_config(
            candidate,
            checkpoint_settings=checkpoint_settings,
            evaluation_settings=disabled_settings,
        )
        assert not candidate.safety_mpc_enabled
        assert not candidate.safety_mpc_active
        assert candidate.safety_mpc_alpha == 0.0
        assert state_sha256(candidate.config) == config_hash
        torch.manual_seed(seed)
        disabled_actions.append(
            candidate.act(
                observation,
                first_step=True,
                eval_mode=True,
            )
        )
        assert candidate.last_safety_mpc_metrics is None
        assert state_sha256(candidate.model.state_dict()) == model_hash
        assert state_sha256(candidate.model.state_dict()) == checkpoint_model_hash
        assert (
            state_sha256(candidate.model_optimizer.state_dict())
            == model_optimizer_hash
        )
        assert (
            state_sha256(candidate.policy_optimizer.state_dict())
            == policy_optimizer_hash
        )
        assert all(
            parameter.grad is None
            for parameter in candidate.model.parameters()
        )
    np.testing.assert_array_equal(disabled_actions[0], disabled_actions[1])

    active = _agent(checkpoint_config, seed=1703)
    active.load_state_dict(checkpoint_state)
    active_model_hash = state_sha256(active.model.state_dict())
    active_model_optimizer_hash = state_sha256(
        active.model_optimizer.state_dict()
    )
    active_policy_optimizer_hash = state_sha256(
        active.policy_optimizer.state_dict()
    )
    active_config_hash = state_sha256(active.config)
    apply_evaluation_safety_mpc_config(
        active,
        checkpoint_settings=checkpoint_settings,
        evaluation_settings=enabled_settings,
    )
    assert active.safety_mpc_enabled
    assert active.safety_mpc_active
    assert active.safety_mpc_alpha == 0.1
    assert state_sha256(active.config) == active_config_hash
    with patch.object(
        active.model.safety_curvature_head,
        "forward",
        side_effect=AssertionError(
            "Curvature branch must remain monitoring-only"
        ),
    ):
        with patch.object(
            active.model,
            "safety_transformed",
            side_effect=AssertionError(
                "Two-channel Safety API must not be used by active planning"
            ),
        ):
            torch.manual_seed(1801)
            action = active.act(
                observation,
                first_step=True,
                eval_mode=True,
            )
    assert np.all(np.isfinite(action))
    assert active.last_safety_mpc_metrics is not None
    assert state_sha256(active.model.state_dict()) == active_model_hash
    assert active_model_hash == checkpoint_model_hash
    assert (
        state_sha256(active.model_optimizer.state_dict())
        == active_model_optimizer_hash
    )
    assert (
        state_sha256(active.policy_optimizer.state_dict())
        == active_policy_optimizer_hash
    )
    assert all(
        parameter.grad is None for parameter in active.model.parameters()
    )

    changed_cap = copy.deepcopy(enabled_settings)
    changed_cap["translation_risk_cap"] = 0.5
    _assert_raises(
        lambda: apply_evaluation_safety_mpc_config(
            active,
            checkpoint_settings=checkpoint_settings,
            evaluation_settings=changed_cap,
        ),
        ValueError,
    )


def main() -> None:
    _test_strict_config_builder()
    _test_disabled_and_alpha_zero_identity()
    _test_penalty_formula_scale_floor_and_finiteness()
    _test_active_calls_translation_only_and_recomputes_scale()
    _test_translation_effect_risk_cap_and_invalid_predictions()
    _test_intervention_mask_unchanged()
    _test_agent_and_checkpoint_strictness()
    _test_evaluation_only_override_isolation()
    print(
        "PASS: strict Safety-MPC config/checkpoint state, disabled and alpha-zero "
        "bitwise isolation, active population-scale penalty/floor, per-iteration "
        "Translation-only planning, finite capped risk/actions/metrics, unchanged "
        "lower/tree-end intervention masking, and isolated deterministic "
        "evaluation-only planner overrides."
    )


if __name__ == "__main__":
    main()
