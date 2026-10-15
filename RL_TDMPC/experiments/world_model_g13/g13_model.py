"""One geometry-enhanced TD-MPC2 world model jointly supervised for force."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tdmpc2.agent import TDMPC2Agent
from tdmpc2.networks import WorldModel, mlp, soft_cross_entropy, weight_init
from tdmpc2.replay_buffer import EpisodeReplayBuffer


class GeometryForceWorldModel(WorldModel):
    """Original WorldModel plus a nonnegative post-transition force head."""

    def __init__(self, observation_dim: int, action_dim: int, config: Dict) -> None:
        super().__init__(observation_dim, action_dim, config)
        mlp_dim = int(config["mlp_dim"])
        self.force_head = nn.Sequential(
            mlp(self.latent_dim, [mlp_dim, mlp_dim], 1), nn.Softplus()
        )
        self.force_head.apply(weight_init)

    def force(self, next_latent: torch.Tensor) -> torch.Tensor:
        """Predict F_(t+1) from WorldModel.next(z_t, a_t)."""
        return self.force_head(next_latent)


class G13ReplayBuffer(EpisodeReplayBuffer):
    """Episode-aware baseline replay with aligned next-interval force labels."""

    def add_episode(self, observations, actions, rewards, terminated, forces) -> None:
        force_array = np.asarray(forces, dtype=np.float32)
        if force_array.shape != (len(actions),) or not np.all(np.isfinite(force_array)) or np.any(force_array < 0):
            raise ValueError("G13 force targets must be finite nonnegative and one per action")
        super().add_episode(observations, actions, rewards, terminated)
        stored = self._episodes[-1]
        stored["forces"] = force_array[-len(stored["actions"]):].copy()

    def sample(self, device: torch.device):
        counts = np.asarray([
            max(0, episode["actions"].shape[0] - self.horizon + 1)
            for episode in self._episodes
        ], dtype=np.int64)
        total = int(counts.sum())
        if total == 0:
            raise RuntimeError("G13 replay lacks a complete sequence")
        cumulative = np.cumsum(counts)
        obs = np.empty((self.horizon + 1, self.batch_size, self.observation_dim), dtype=np.float32)
        actions = np.empty((self.horizon, self.batch_size, self.action_dim), dtype=np.float32)
        rewards = np.empty((self.horizon, self.batch_size, 1), dtype=np.float32)
        terminated = np.empty_like(rewards)
        forces = np.empty_like(rewards)
        episodes = list(self._episodes)
        sampled = self._rng.integers(0, total, size=self.batch_size)
        for batch_index, global_index in enumerate(sampled):
            episode_index = int(np.searchsorted(cumulative, global_index, side="right"))
            previous = 0 if episode_index == 0 else int(cumulative[episode_index - 1])
            start = int(global_index - previous)
            end = start + self.horizon
            episode = episodes[episode_index]
            if "forces" not in episode:
                raise RuntimeError("G13 replay episode lacks force targets")
            obs[:, batch_index] = episode["observations"][start:end + 1]
            actions[:, batch_index] = episode["actions"][start:end]
            rewards[:, batch_index, 0] = episode["rewards"][start:end]
            terminated[:, batch_index, 0] = episode["terminated"][start:end]
            forces[:, batch_index, 0] = episode["forces"][start:end]
        return tuple(torch.as_tensor(array, device=device)
                     for array in (obs, actions, rewards, terminated, forces))


class G13Agent(TDMPC2Agent):
    """The baseline planner/policy with joint post-transition Huber force loss."""

    def __init__(self, observation_dim: int, action_dim: int,
                 config: Mapping[str, Any], *, episode_length: int,
                 device: torch.device) -> None:
        if observation_dim != 22:
            raise ValueError("G13 primary observation must be 22-D")
        super().__init__(observation_dim, action_dim, config,
                         episode_length=episode_length, device=device)
        # Replace only the model instantiation and optimizer groups. All
        # TD-MPC2 network shapes besides the encoder input and added force
        # head remain exactly the original architecture.
        self.model = GeometryForceWorldModel(observation_dim, action_dim,
                                             self.config).to(device)
        self.model.train(False)
        lr = float(self.config["lr"])
        self.model_optimizer = torch.optim.Adam([
            {"params": self.model.encoder.parameters(),
             "lr": lr * float(self.config["enc_lr_scale"])},
            {"params": self.model.dynamics.parameters()},
            {"params": self.model.reward_head.parameters()},
            {"params": self.model.termination_head.parameters()},
            {"params": self.model.q_ensemble.parameters()},
            {"params": self.model.force_head.parameters()},
        ], lr=lr)
        self.policy_optimizer = torch.optim.Adam(
            self.model.policy.parameters(), lr=lr, eps=1e-5)
        self.lambda_force_train = float(self.config["lambda_force_train"])
        if not self.lambda_force_train > 0:
            raise ValueError("G13 force training coefficient must be positive")

    def loss_terms(self, batch):
        observations, actions, rewards, terminated, forces = batch
        rho = float(self.config["rho"])
        weights = torch.pow(torch.tensor(rho, device=self.device),
                            torch.arange(self.horizon, device=self.device))
        with torch.no_grad():
            next_latent_targets = self.model.encode(observations[1:])
            td_targets = self._td_target(next_latent_targets, rewards, terminated)
        self.model.train(True)
        latent = self.model.encode(observations[0])
        latent_rollout = [latent]
        consistency_loss = torch.zeros((), device=self.device)
        for step in range(self.horizon):
            latent = self.model.next(latent, actions[step])
            consistency_loss = consistency_loss + weights[step] * F.mse_loss(
                latent, next_latent_targets[step])
            latent_rollout.append(latent)
        latent_rollout_tensor = torch.stack(latent_rollout, dim=0)
        rollout_latent = latent_rollout_tensor[:-1]
        reward_logits = self.model.reward(rollout_latent, actions)
        q_logits = self.model.q(rollout_latent, actions, reduction="all")
        termination_logits = self.model.termination(latent_rollout_tensor[1:], logits=True)
        force_predictions = self.model.force(latent_rollout_tensor[1:])
        reward_element = soft_cross_entropy(
            reward_logits, rewards, self.model.num_bins,
            self.model.value_min, self.model.value_max).mean(dim=(1, 2))
        value_element = soft_cross_entropy(
            q_logits,
            td_targets.unsqueeze(0).expand(self.model.num_q, -1, -1, -1),
            self.model.num_bins, self.model.value_min,
            self.model.value_max).mean(dim=(0, 2, 3))
        force_element = F.smooth_l1_loss(force_predictions, forces,
                                         beta=1.0, reduction="none").mean(dim=(1, 2))
        terms = {
            "consistency_loss": consistency_loss / self.horizon,
            "reward_loss": (reward_element * weights).sum() / self.horizon,
            "value_loss": (value_element * weights).sum() / self.horizon,
            "termination_loss": F.binary_cross_entropy_with_logits(termination_logits, terminated),
            "force_loss": (force_element * weights).sum() / self.horizon,
        }
        original = sum(float(self.config[name.replace("_loss", "_coef")]) * terms[name]
                       for name in ("consistency_loss", "reward_loss", "value_loss", "termination_loss"))
        terms["original_total_loss"] = original
        terms["total_loss"] = original + self.lambda_force_train * terms["force_loss"]
        return terms, latent_rollout_tensor

    def update(self, replay_buffer) -> Dict[str, float]:
        batch = replay_buffer.sample(self.device)
        terms, latent_rollout = self.loss_terms(batch)
        if not all(torch.isfinite(value).all() for value in terms.values()):
            raise FloatingPointError("G13 joint world-model loss is nonfinite")
        self.model_optimizer.zero_grad(set_to_none=True)
        terms["total_loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for group in self.model_optimizer.param_groups for p in group["params"]],
            float(self.config["grad_clip_norm"]))
        self.model_optimizer.step()
        policy_info = self._update_policy(latent_rollout.detach())
        self.model.soft_update_target_q(float(self.config["tau"]))
        self.model.train(False)
        self.update_count += 1
        return {**{key: float(value.detach().cpu()) for key, value in terms.items()},
                "model_grad_norm": float(grad_norm.detach().cpu()),
                **policy_info}
