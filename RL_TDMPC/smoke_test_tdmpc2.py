"""Fast dependency/shape test for the TD-MPC2 model, update, and planner."""

from __future__ import annotations

import numpy as np
import torch

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
    replay = EpisodeReplayBuffer(100, 14, 2, 3, 4, seed=1)
    rng = np.random.default_rng(1)
    observations = rng.uniform(-1, 1, size=(9, 14)).astype(np.float32)
    actions = rng.uniform(-1, 1, size=(8, 2)).astype(np.float32)
    rewards = rng.normal(size=8).astype(np.float32)
    terminated = np.zeros(8, dtype=bool)
    terminated[-1] = True
    replay.add_episode(observations, actions, rewards, terminated)

    metrics = agent.update(replay)
    assert all(np.isfinite(value) for value in metrics.values())
    action = agent.act(observations[0], first_step=True, eval_mode=True)
    assert action.shape == (2,)
    assert action.dtype == np.float32
    assert np.all(np.isfinite(action))
    assert np.all(np.abs(action) <= 1.0)

    clone = TDMPC2Agent(14, 2, config, episode_length=200, device=device)
    clone.load_state_dict(agent.state_dict(), load_optimizers=True)
    assert clone.update_count == agent.update_count
    print("PASS: TD-MPC2 replay sampling, one update, planning, and state restore.")


if __name__ == "__main__":
    main()

