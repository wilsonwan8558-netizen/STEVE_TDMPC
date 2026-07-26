"""Fast dependency/shape test for the TD-MPC2 model, update, and planner."""

from __future__ import annotations

import copy

import numpy as np
import torch

from envs.safety import SAFETY_COST_NAMES
from tdmpc2.agent import TDMPC2Agent
from tdmpc2.replay_buffer import EpisodeReplayBuffer


def main() -> None:
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
    replay = EpisodeReplayBuffer(
        100,
        14,
        2,
        3,
        4,
        safety_cost_names=SAFETY_COST_NAMES,
        seed=1,
    )
    rng = np.random.default_rng(1)
    observations = rng.uniform(-1, 1, size=(9, 14)).astype(np.float32)
    actions = rng.uniform(-1, 1, size=(8, 2)).astype(np.float32)
    rewards = (np.arange(8, dtype=np.float32) / 10.0).astype(np.float32)
    terminated = np.zeros(8, dtype=bool)
    terminated[-1] = True
    safety_cost = np.stack(
        (rewards, rewards + 1.0, rewards + 2.0), axis=-1
    ).astype(np.float32)

    try:
        replay.add_episode(observations, actions, rewards, terminated)
    except ValueError as exc:
        assert "safety_cost is required" in str(exc)
    else:
        raise AssertionError("Replay accepted an episode without safety_cost")

    for invalid_cost, expected_error in (
        (np.zeros((8, 2), dtype=np.float32), ValueError),
        (
            np.full(
                (8, len(SAFETY_COST_NAMES)),
                np.nan,
                dtype=np.float32,
            ),
            FloatingPointError,
        ),
        (
            -np.ones(
                (8, len(SAFETY_COST_NAMES)),
                dtype=np.float32,
            ),
            ValueError,
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
    assert sampled_safety_cost.shape == (3, 4, len(SAFETY_COST_NAMES))
    assert sampled_safety_cost.dtype == torch.float32
    assert torch.all(torch.isfinite(sampled_safety_cost))
    assert torch.all(sampled_safety_cost >= 0.0)
    torch.testing.assert_close(
        sampled_safety_cost[..., 0], sampled_rewards[..., 0]
    )
    torch.testing.assert_close(
        sampled_safety_cost[..., 1], sampled_rewards[..., 0] + 1.0
    )
    torch.testing.assert_close(
        sampled_safety_cost[..., 2], sampled_rewards[..., 0] + 2.0
    )

    replay_state = replay.state_dict()
    stored_cost = replay_state["episodes"][0]["safety_cost"]
    assert stored_cost.shape == (8, len(SAFETY_COST_NAMES))
    assert stored_cost.dtype == np.float32
    restored_replay = EpisodeReplayBuffer(
        100,
        14,
        2,
        3,
        4,
        safety_cost_names=SAFETY_COST_NAMES,
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

    reordered_state = copy.deepcopy(replay_state)
    reordered_state["safety_cost_names"] = tuple(
        reversed(SAFETY_COST_NAMES)
    )
    try:
        restored_replay.load_state_dict(reordered_state)
    except ValueError as exc:
        assert "do not match current" in str(exc)
    else:
        raise AssertionError("Replay accepted reordered safety-cost channels")

    metrics = agent.update(replay)
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
    print(
        "PASS: aligned safety-cost replay sampling, validation/state restore, "
        "unchanged TD-MPC2 update, planning, and agent state restore."
    )


if __name__ == "__main__":
    main()
