#!/usr/bin/env python3
"""Simulation-free smoke tests for configurable Safety cost schemas.

The real stEVE/SOFA and NVIDIA/Warp simulators are deliberately not started.
The NVIDIA adapter is exercised with a strict fake external environment, and
fresh subprocesses verify that the environment factory imports only the
selected backend.
"""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Mapping, Sequence, Type
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch

from envs.nvidia_guided_env import NvidiaGuidedEnv, _REQUIRED_CT_FILES
from safety_schema import (
    LEGACY_STEVE_PRIMARY_RISK_CHANNEL,
    LEGACY_STEVE_SAFETY_COST_NAMES,
    NVIDIA_GUIDED_PRIMARY_RISK_CHANNEL,
    NVIDIA_GUIDED_SAFETY_COST_NAMES,
    UNSUPPORTED_LEGACY_THREE_CHANNEL_SAFETY_COST_NAMES,
    validate_safety_cost_names,
    validate_safety_schema,
)
from tdmpc2.agent import TDMPC2Agent
from tdmpc2.common import build_safety_agent_config, load_config
from tdmpc2.networks import WorldModel
from tdmpc2.replay_buffer import EpisodeReplayBuffer
from train import (
    build_agent_config,
    save_checkpoint,
    validate_checkpoint_schema,
)


PROJECT_DIR = Path(__file__).resolve().parent
OBSERVATION_DIM = 14
ACTION_DIM = 2
HORIZON = 3
BATCH_SIZE = 2


def _expect_error(
    error_type: Type[BaseException],
    message_fragments: Sequence[str],
    operation: Callable[[], Any],
) -> str:
    try:
        operation()
    except error_type as exc:
        message = str(exc)
        for fragment in message_fragments:
            assert fragment.lower() in message.lower(), (
                fragment,
                type(exc).__name__,
                message,
            )
        return message
    except Exception as exc:  # pragma: no cover - makes smoke failures clearer
        raise AssertionError(
            f"Expected {error_type.__name__}, got {type(exc).__name__}: {exc}"
        ) from exc
    raise AssertionError(f"Expected {error_type.__name__}")


def _assert_nested_equal(left: Any, right: Any) -> None:
    assert type(left) is type(right), (type(left), type(right))
    if isinstance(left, Mapping):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(left_item, right_item)
    elif isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    else:
        assert left == right


def _small_config(path: Path) -> dict[str, Any]:
    config = load_config(path)
    config["model"].update(
        {
            "latent_dim": 16,
            "enc_dim": 16,
            "mlp_dim": 32,
            "num_enc_layers": 2,
            "simnorm_dim": 4,
            "num_q": 2,
            "dropout": 0.0,
            "num_bins": 11,
            "vmin": -5.0,
            "vmax": 5.0,
        }
    )
    config["training"].update(
        {
            "horizon": HORIZON,
            "batch_size": BATCH_SIZE,
            "device": "cpu",
        }
    )
    config["planning"].update(
        {
            "horizon": HORIZON,
            "num_samples": 8,
            "num_elites": 2,
            "num_pi_trajs": 2,
            "iterations": 1,
        }
    )
    config["diagnostics"]["validation_interval"] = 100
    config["diagnostics"]["gradient_interval"] = 1
    return config


def _episode_arrays(
    safety_dim: int,
    *,
    length: int = HORIZON,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    observations = rng.uniform(
        -0.9,
        0.9,
        size=(length + 1, OBSERVATION_DIM),
    ).astype(np.float32)
    actions = rng.uniform(-0.9, 0.9, size=(length, ACTION_DIM)).astype(
        np.float32
    )
    rewards = rng.normal(0.0, 0.25, size=length).astype(np.float32)
    terminated = np.zeros(length, dtype=bool)
    terminated[-1] = True
    safety_cost = rng.uniform(
        0.05,
        0.95,
        size=(length, safety_dim),
    ).astype(np.float32)
    return observations, actions, rewards, terminated, safety_cost


def _make_replay(
    names: Sequence[str],
    *,
    seed: int,
) -> EpisodeReplayBuffer:
    replay = EpisodeReplayBuffer(
        capacity=100,
        observation_dim=OBSERVATION_DIM,
        action_dim=ACTION_DIM,
        horizon=HORIZON,
        batch_size=BATCH_SIZE,
        safety_cost_names=names,
        seed=seed,
    )
    replay.add_episode(*_episode_arrays(len(tuple(names)), seed=seed + 1))
    return replay


def test_schemas_and_config() -> None:
    assert validate_safety_schema(
        LEGACY_STEVE_SAFETY_COST_NAMES,
        2,
    ) == LEGACY_STEVE_SAFETY_COST_NAMES
    assert validate_safety_schema(
        NVIDIA_GUIDED_SAFETY_COST_NAMES,
        6,
    ) == NVIDIA_GUIDED_SAFETY_COST_NAMES

    _expect_error(ValueError, ("must not be empty",), lambda: (
        validate_safety_cost_names(())
    ))
    _expect_error(ValueError, ("nonempty",), lambda: (
        validate_safety_cost_names(("valid", "  "))
    ))
    _expect_error(ValueError, ("unique",), lambda: (
        validate_safety_cost_names(("duplicate", "duplicate"))
    ))
    _expect_error(TypeError, ("must be a string",), lambda: (
        validate_safety_cost_names(("valid", 3))
    ))
    _expect_error(ValueError, ("does not match",), lambda: (
        validate_safety_schema(("one", "two"), 3)
    ))
    _expect_error(
        ValueError,
        ("unsupported legacy three-channel",),
        lambda: validate_safety_cost_names(
            UNSUPPORTED_LEGACY_THREE_CHANNEL_SAFETY_COST_NAMES
        ),
    )

    legacy_config = _small_config(PROJECT_DIR / "configs" / "steve.yaml")
    legacy_agent_config = build_safety_agent_config(legacy_config)
    assert legacy_agent_config["safety_cost_names"] == (
        LEGACY_STEVE_SAFETY_COST_NAMES
    )
    assert legacy_agent_config["safety_dim"] == 2
    assert legacy_agent_config["safety_channel_loss_coefs"] == (1.0, 1.0)
    assert legacy_agent_config["safety_channel_scales"] == (0.1, 1.0)
    assert legacy_agent_config["safety_primary_risk_channel"] == (
        LEGACY_STEVE_PRIMARY_RISK_CHANNEL
    )

    legacy_embedded_config = copy.deepcopy(legacy_config)
    legacy_embedded_config.pop("backend", None)
    legacy_embedded_config.pop("safety_cost_names", None)
    legacy_embedded_config.pop("safety_dim", None)
    assert build_safety_agent_config(legacy_embedded_config) == (
        legacy_agent_config
    )

    nvidia_config = _small_config(
        PROJECT_DIR / "configs" / "nvidia_guided.yaml"
    )
    nvidia_agent_config = build_safety_agent_config(nvidia_config)
    assert nvidia_agent_config["safety_cost_names"] == (
        NVIDIA_GUIDED_SAFETY_COST_NAMES
    )
    assert nvidia_agent_config["safety_dim"] == 6
    assert nvidia_agent_config["safety_channel_loss_coefs"] == (1.0,) * 6
    assert nvidia_agent_config["safety_channel_scales"] == (1.0,) * 6
    assert nvidia_agent_config["safety_primary_risk_channel"] == (
        NVIDIA_GUIDED_PRIMARY_RISK_CHANNEL
    )


class _FakeNvidiaEnvironment(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": [None, "rgb_array"]}

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(ACTION_DIM,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(OBSERVATION_DIM,),
            dtype=np.float32,
        )
        self.last_action: np.ndarray | None = None
        self.closed = False

    @staticmethod
    def _info(*, success: bool) -> dict[str, Any]:
        safety = np.linspace(0.0, 0.5, 6, dtype=np.float64)
        return {
            "safety_component_order": NVIDIA_GUIDED_SAFETY_COST_NAMES,
            "safety_cost_vector": safety,
            "safety_cost": float(safety.sum()),
            "hard_safety_violation": False,
            "success": success,
        }

    def reset(self, *, seed=None, options=None):
        del seed, options
        return np.zeros(OBSERVATION_DIM, dtype=np.float64), self._info(
            success=False
        )

    def step(self, action):
        self.last_action = np.asarray(action).copy()
        return (
            np.full(OBSERVATION_DIM, 0.25, dtype=np.float64),
            1.25,
            True,
            False,
            self._info(success=True),
        )

    def render(self):
        return np.zeros((4, 4, 3), dtype=np.uint8)

    def close(self) -> None:
        self.closed = True


def test_fake_nvidia_adapter() -> None:
    _expect_error(
        ValueError,
        ("safety_config", "normalization constants"),
        lambda: NvidiaGuidedEnv(safety_config={}),
    )
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        workflow_root = root / "workflow"
        ct_cache = root / "cache"
        workflow_root.mkdir()
        ct_cache.mkdir()
        for name in _REQUIRED_CT_FILES:
            (ct_cache / name).touch()

        with patch(
            "envs.nvidia_guided_env._load_nvidia_environment_class",
            return_value=_FakeNvidiaEnvironment,
        ):
            environment = NvidiaGuidedEnv(
                workflow_root=workflow_root,
                ct_cache_path=ct_cache,
                max_episode_steps=7,
            )

        assert environment.observation_space.shape == (OBSERVATION_DIM,)
        assert environment.action_space.shape == (ACTION_DIM,)
        observation, reset_info = environment.reset(seed=17)
        assert observation.shape == (OBSERVATION_DIM,)
        assert observation.dtype == np.float32
        assert np.isfinite(observation).all()
        assert reset_info["safety_cost_names"] == (
            NVIDIA_GUIDED_SAFETY_COST_NAMES
        )
        reset_cost = reset_info["safety_cost"]
        assert reset_cost.shape == (6,)
        assert reset_cost.dtype == np.float32
        assert np.isfinite(reset_cost).all() and np.all(reset_cost >= 0.0)
        expected_cost = np.linspace(0.0, 0.5, 6, dtype=np.float32)
        np.testing.assert_array_equal(reset_cost, expected_cost)
        np.testing.assert_allclose(
            reset_info["safety_cost_scalar"],
            expected_cost.astype(np.float64).sum(),
        )

        next_observation, reward, terminated, truncated, step_info = (
            environment.step(np.asarray([1.5, -2.0], dtype=np.float32))
        )
        assert next_observation.shape == (OBSERVATION_DIM,)
        assert np.isfinite(next_observation).all()
        assert np.isfinite(reward)
        assert terminated and not truncated
        assert step_info["is_success"]
        assert not step_info["hard_safety_violation"]
        assert step_info["safety_cost"].shape == (6,)
        assert step_info["safety_cost"].dtype == np.float32
        np.testing.assert_array_equal(step_info["safety_cost"], expected_cost)
        np.testing.assert_array_equal(
            environment._env.last_action,
            np.asarray([1.0, -1.0], dtype=np.float32),
        )
        assert environment.render().shape == (4, 4, 3)
        environment.close()
        assert environment._env.closed


def test_six_channel_replay_exact_values() -> EpisodeReplayBuffer:
    replay = EpisodeReplayBuffer(
        capacity=20,
        observation_dim=OBSERVATION_DIM,
        action_dim=ACTION_DIM,
        horizon=HORIZON,
        batch_size=BATCH_SIZE,
        safety_cost_names=NVIDIA_GUIDED_SAFETY_COST_NAMES,
        seed=11,
    )
    observations = np.arange(
        (HORIZON + 1) * OBSERVATION_DIM,
        dtype=np.float32,
    ).reshape(HORIZON + 1, OBSERVATION_DIM)
    actions = np.arange(HORIZON * ACTION_DIM, dtype=np.float32).reshape(
        HORIZON,
        ACTION_DIM,
    )
    rewards = np.arange(HORIZON, dtype=np.float32)
    terminated = np.asarray([False, False, True])
    safety_cost = (
        np.arange(HORIZON * 6, dtype=np.float32).reshape(HORIZON, 6) / 20.0
    )
    replay.add_episode(
        observations,
        actions,
        rewards,
        terminated,
        safety_cost,
    )

    sampled = replay.sample(torch.device("cpu"))
    assert [tuple(tensor.shape) for tensor in sampled] == [
        (HORIZON + 1, BATCH_SIZE, OBSERVATION_DIM),
        (HORIZON, BATCH_SIZE, ACTION_DIM),
        (HORIZON, BATCH_SIZE, 1),
        (HORIZON, BATCH_SIZE, 1),
        (HORIZON, BATCH_SIZE, 6),
    ]
    expected_safety = np.repeat(safety_cost[:, None, :], BATCH_SIZE, axis=1)
    np.testing.assert_array_equal(sampled[-1].numpy(), expected_safety)
    return replay


def _agent_and_replay(
    config: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[TDMPC2Agent, EpisodeReplayBuffer, dict[str, float]]:
    agent_config = build_agent_config(config)
    torch.manual_seed(seed)
    agent = TDMPC2Agent(
        OBSERVATION_DIM,
        ACTION_DIM,
        agent_config,
        episode_length=20,
        device=torch.device("cpu"),
    )
    replay = _make_replay(agent.safety_cost_names, seed=seed)
    safety_parameters = agent.model.safety_head_parameters()
    before = [parameter.detach().clone() for parameter in safety_parameters]
    metrics = agent.update(replay)
    for name in (
        "total_loss",
        "safety_loss",
        "model_grad_norm",
        "policy_loss",
    ):
        assert name in metrics and np.isfinite(metrics[name]), (name, metrics)
    gradients = [
        parameter.grad
        for parameter in safety_parameters
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients)
    assert any(
        not torch.equal(expected, parameter.detach())
        for expected, parameter in zip(before, safety_parameters)
    )
    return agent, replay, metrics


def test_world_model_agent_and_synthetic_update() -> tuple[
    dict[str, Any],
    TDMPC2Agent,
    EpisodeReplayBuffer,
]:
    config = _small_config(PROJECT_DIR / "configs" / "nvidia_guided.yaml")
    agent_config = build_agent_config(config)
    fractional_dim_config = copy.deepcopy(agent_config)
    fractional_dim_config["safety_dim"] = 6.5
    _expect_error(
        TypeError,
        ("safety_dim", "integer"),
        lambda: WorldModel(
            OBSERVATION_DIM,
            ACTION_DIM,
            fractional_dim_config,
        ),
    )
    _expect_error(
        TypeError,
        ("safety_dim", "integer"),
        lambda: TDMPC2Agent(
            OBSERVATION_DIM,
            ACTION_DIM,
            fractional_dim_config,
            episode_length=20,
            device=torch.device("cpu"),
        ),
    )
    string_names_config = copy.deepcopy(agent_config)
    string_names_config["safety_cost_names"] = "tracking"
    _expect_error(
        TypeError,
        ("sequence of strings",),
        lambda: WorldModel(
            OBSERVATION_DIM,
            ACTION_DIM,
            string_names_config,
        ),
    )
    world_model = WorldModel(OBSERVATION_DIM, ACTION_DIM, agent_config)
    latent = world_model.encode(torch.zeros(4, OBSERVATION_DIM))
    prediction = world_model.safety(latent, torch.zeros(4, ACTION_DIM))
    assert prediction.shape == (4, 6)
    assert prediction.dtype == torch.float32
    assert torch.isfinite(prediction).all()
    assert torch.all(prediction >= 0.0)
    assert world_model.safety_primary_risk_index == 5

    agent, replay, metrics = _agent_and_replay(config, seed=23)
    assert agent.safety_cost_names == NVIDIA_GUIDED_SAFETY_COST_NAMES
    assert agent.safety_dim == 6
    assert agent.model.safety_primary_risk_index == 5
    assert np.isfinite(metrics["safety_loss"])
    return config, agent, replay


def _save_reload_and_verify(
    config: Mapping[str, Any],
    agent: TDMPC2Agent,
    replay: EpisodeReplayBuffer,
    destination: Path,
) -> dict[str, Any]:
    saved_path = save_checkpoint(
        path=destination,
        config=config,
        agent=agent,
        replay=replay,
        total_env_steps=19,
        episode_index=3,
        success_count=2,
        include_replay=True,
    )
    assert saved_path == destination and destination.is_file()
    checkpoint = torch.load(
        destination,
        map_location="cpu",
        weights_only=False,
    )
    validate_checkpoint_schema(
        checkpoint,
        config=config,
        source=f"Smoke checkpoint {destination.name}",
    )

    restored_agent = TDMPC2Agent(
        OBSERVATION_DIM,
        ACTION_DIM,
        build_agent_config(config),
        episode_length=20,
        device=torch.device("cpu"),
    )
    restored_agent.load_state_dict(checkpoint["agent"])
    _assert_nested_equal(agent.state_dict(), restored_agent.state_dict())

    restored_replay = EpisodeReplayBuffer(
        capacity=100,
        observation_dim=OBSERVATION_DIM,
        action_dim=ACTION_DIM,
        horizon=HORIZON,
        batch_size=BATCH_SIZE,
        safety_cost_names=agent.safety_cost_names,
        seed=999,
    )
    restored_replay.load_state_dict(checkpoint["replay"])
    _assert_nested_equal(replay.state_dict(), restored_replay.state_dict())
    return checkpoint


def test_checkpoint_round_trips_and_schema_mismatch(
    nvidia_config: Mapping[str, Any],
    nvidia_agent: TDMPC2Agent,
    nvidia_replay: EpisodeReplayBuffer,
) -> None:
    legacy_config = _small_config(PROJECT_DIR / "configs" / "steve.yaml")
    legacy_agent, legacy_replay, legacy_metrics = _agent_and_replay(
        legacy_config,
        seed=29,
    )
    assert legacy_agent.safety_cost_names == LEGACY_STEVE_SAFETY_COST_NAMES
    assert legacy_agent.safety_dim == 2
    assert np.isfinite(legacy_metrics["total_loss"])

    with TemporaryDirectory() as temporary_directory:
        checkpoint_directory = Path(temporary_directory)
        legacy_checkpoint = _save_reload_and_verify(
            legacy_config,
            legacy_agent,
            legacy_replay,
            checkpoint_directory / "legacy.pt",
        )
        nvidia_checkpoint = _save_reload_and_verify(
            nvidia_config,
            nvidia_agent,
            nvidia_replay,
            checkpoint_directory / "nvidia.pt",
        )

        agent_mismatch_message = _expect_error(
            ValueError,
            ("expected", "received"),
            lambda: legacy_agent.load_state_dict(nvidia_checkpoint["agent"]),
        )
        for channel_name in LEGACY_STEVE_SAFETY_COST_NAMES:
            assert channel_name in agent_mismatch_message
        for channel_name in NVIDIA_GUIDED_SAFETY_COST_NAMES:
            assert channel_name in agent_mismatch_message

        for checkpoint, incompatible_config in (
            (legacy_checkpoint, nvidia_config),
            (nvidia_checkpoint, legacy_config),
        ):
            message = _expect_error(
                ValueError,
                ("expected", "received"),
                lambda checkpoint=checkpoint, incompatible_config=(
                    incompatible_config
                ): validate_checkpoint_schema(
                    checkpoint,
                    config=incompatible_config,
                    source="Incompatible smoke checkpoint",
                ),
            )
            for channel_name in checkpoint["safety_cost_names"]:
                assert channel_name in message
            for channel_name in incompatible_config["safety_cost_names"]:
                assert channel_name in message


def _run_isolation_subprocess(script: str) -> None:
    environment = dict(os.environ)
    python_path = [str(PROJECT_DIR)]
    existing_python_path = environment.get("PYTHONPATH")
    if existing_python_path:
        python_path.append(existing_python_path)
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_DIR.parent,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            "Lazy-import subprocess failed\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def test_lazy_backend_import_isolation() -> None:
    _run_isolation_subprocess(
        """
import sys
import types
import envs

assert not any(name == "warp" or name.startswith("warp.") for name in sys.modules)
fake = types.ModuleType("envs.steve_env")
fake.make_steve_env = lambda config: ("steve", dict(config))
sys.modules["envs.steve_env"] = fake
result = envs.make_env({"max_episode_steps": 2}, backend="steve")
assert result == ("steve", {"max_episode_steps": 2})
assert "envs.nvidia_guided_env" not in sys.modules
assert not any(name == "warp" or name.startswith("warp.") for name in sys.modules)
"""
    )
    _run_isolation_subprocess(
        """
import importlib
import sys
import envs

nvidia = importlib.import_module("envs.nvidia_guided_env")
nvidia.make_nvidia_guided_env = lambda config: ("nvidia", dict(config))
result = envs.make_env({"max_episode_steps": 2}, backend="nvidia-guided")
assert result == ("nvidia", {"max_episode_steps": 2})
assert "envs.steve_env" not in sys.modules
assert not any(
    name == "Sofa"
    or name.startswith("Sofa.")
    or name == "SofaRuntime"
    or name.startswith("SofaRuntime.")
    for name in sys.modules
)
"""
    )


def main() -> None:
    test_schemas_and_config()
    print("PASS: legacy and NVIDIA schemas/config validation")
    test_fake_nvidia_adapter()
    print("PASS: fake NVIDIA adapter contract")
    test_six_channel_replay_exact_values()
    print("PASS: exact six-channel replay insertion/sampling")
    nvidia_config, nvidia_agent, nvidia_replay = (
        test_world_model_agent_and_synthetic_update()
    )
    print("PASS: six-channel WorldModel/Agent and finite synthetic update")
    test_checkpoint_round_trips_and_schema_mismatch(
        nvidia_config,
        nvidia_agent,
        nvidia_replay,
    )
    print("PASS: legacy/six-channel checkpoint round trips and strict mismatch")
    test_lazy_backend_import_isolation()
    print("PASS: lazy backend import isolation")
    print("PASS: configurable multi-metric Safety smoke suite")


if __name__ == "__main__":
    main()
