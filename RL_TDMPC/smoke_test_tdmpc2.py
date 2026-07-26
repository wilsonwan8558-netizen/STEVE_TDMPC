"""Fast dependency/shape test for the TD-MPC2 model, update, and planner."""

from __future__ import annotations

import copy
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from envs.safety import SAFETY_COST_NAMES as ENV_SAFETY_COST_NAMES
from train import (
    CHECKPOINT_FORMAT_VERSION,
    save_checkpoint,
    validate_collected_safety_cost,
    validate_checkpoint_schema,
)
from tdmpc2.agent import TDMPC2Agent
from tdmpc2.common import load_torch_checkpoint
from tdmpc2.replay_buffer import (
    LEGACY_THREE_CHANNEL_SAFETY_COST_NAMES,
    REPLAY_SAFETY_COST_NAMES,
    SAFETY_COST_SCHEMA_VERSION,
    EpisodeReplayBuffer,
)


def main() -> None:
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
    config = {
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
        "log_std_min": -10.0,
        "log_std_max": 2.0,
        "lr": 3e-4,
        "enc_lr_scale": 0.3,
        "tau": 0.01,
        "discount_denom": 5.0,
        "discount_min": 0.95,
        "discount_max": 0.995,
        "horizon": 3,
        "mpc": True,
        "num_samples": 8,
        "num_elites": 2,
        "num_pi_trajs": 2,
        "iterations": 1,
        "max_std": 2.0,
        "min_std": 0.05,
        "temperature": 0.5,
        "episodic": True,
        "rho": 0.5,
        "consistency_coef": 20.0,
        "reward_coef": 0.1,
        "value_coef": 0.1,
        "termination_coef": 1.0,
        "entropy_coef": 1e-4,
        "grad_clip_norm": 20.0,
    }
    device = torch.device("cpu")
    agent = TDMPC2Agent(14, 2, config, episode_length=200, device=device)
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

    class SafetyPoisonReplay:
        """Return valid baseline data plus a safety tensor that must be ignored."""

        def sample(self, sample_device: torch.device):
            tensors = list(replay.sample(sample_device))
            tensors[-1] = torch.full_like(tensors[-1], float("nan"))
            return tuple(tensors)

    metrics = agent.update(SafetyPoisonReplay())
    assert all(np.isfinite(value) for value in metrics.values())
    assert all("safety" not in key for key in metrics)
    action = agent.act(observations[0], first_step=True, eval_mode=True)
    assert action.shape == (2,)
    assert action.dtype == np.float32
    assert np.all(np.isfinite(action))
    assert np.all(np.abs(action) <= 1.0)

    clone = TDMPC2Agent(14, 2, config, episode_length=200, device=device)
    clone.load_state_dict(agent.state_dict(), load_optimizers=True)
    assert clone.update_count == agent.update_count
    with TemporaryDirectory(prefix="steve-tdmpc2-checkpoint-", dir="/tmp") as temp_dir:
        checkpoint_path = Path(temp_dir) / "schema_v2.pt"
        save_checkpoint(
            path=checkpoint_path,
            config={"test": True},
            agent=agent,
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
        validate_checkpoint_schema(
            checkpoint,
            source="Smoke-test checkpoint",
        )
        assert checkpoint["format_version"] == CHECKPOINT_FORMAT_VERSION
        assert (
            checkpoint["safety_cost_schema_version"]
            == SAFETY_COST_SCHEMA_VERSION
        )
        assert tuple(checkpoint["safety_cost_names"]) == safety_cost_names
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
            config,
            episode_length=200,
            device=device,
        )
        checkpoint_agent.load_state_dict(
            checkpoint["agent"],
            load_optimizers=True,
        )
        assert checkpoint_agent.update_count == agent.update_count

        old_checkpoint = copy.deepcopy(checkpoint)
        old_checkpoint["format_version"] = 1
        old_checkpoint.pop("safety_cost_names")
        old_checkpoint.pop("safety_cost_schema_version")
        try:
            validate_checkpoint_schema(
                old_checkpoint,
                source="Old smoke-test checkpoint",
            )
        except ValueError as exc:
            assert "predate" in str(exc)
            assert "two-channel" in str(exc)
        else:
            raise AssertionError("Training accepted a format-v1 checkpoint")

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

    print(
        "PASS: aligned two-channel safety replay sampling, strict schema "
        "validation/checkpoint restore, and baseline TD-MPC2 safety isolation."
    )


if __name__ == "__main__":
    main()
