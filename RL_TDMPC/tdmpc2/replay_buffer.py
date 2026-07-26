"""Episode-aware replay buffer for fixed-horizon TD-MPC2 sequences."""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


class EpisodeReplayBuffer:
    """Store completed episodes and uniformly sample valid subsequences.

    Keeping episode boundaries explicit prevents model rollouts from crossing
    a reset. This replaces the official TorchRL ``SliceSampler`` dependency
    while preserving its fixed-horizon sampling semantics.
    """

    def __init__(
        self,
        capacity: int,
        observation_dim: int,
        action_dim: int,
        horizon: int,
        batch_size: int,
        *,
        safety_cost_names: Sequence[str],
        seed: int = 0,
    ) -> None:
        self.capacity = int(capacity)
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.safety_cost_names = tuple(str(name) for name in safety_cost_names)
        self.safety_cost_dim = len(self.safety_cost_names)
        self.horizon = int(horizon)
        self.batch_size = int(batch_size)
        if self.safety_cost_dim != 3:
            raise ValueError("safety_cost_names must define exactly three channels")
        if len(set(self.safety_cost_names)) != self.safety_cost_dim:
            raise ValueError("safety_cost_names must be unique")
        if (
            min(
                self.capacity,
                self.horizon,
                self.batch_size,
            )
            <= 0
        ):
            raise ValueError(
                "capacity, horizon, and batch_size must be positive"
            )
        self._episodes: Deque[Dict[str, np.ndarray]] = deque()
        self._size = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    @property
    def num_sequences(self) -> int:
        return sum(
            max(0, episode["actions"].shape[0] - self.horizon + 1)
            for episode in self._episodes
        )

    def can_sample(self) -> bool:
        return self.num_sequences > 0

    def add_episode(
        self,
        observations: Sequence[np.ndarray],
        actions: Sequence[np.ndarray],
        rewards: Sequence[float],
        terminated: Sequence[bool],
        safety_cost: Optional[Sequence[np.ndarray]] = None,
    ) -> None:
        if safety_cost is None:
            raise ValueError(
                "safety_cost is required and must contain one vector per action"
            )
        observations_array = np.asarray(observations, dtype=np.float32)
        actions_array = np.asarray(actions, dtype=np.float32)
        rewards_array = np.asarray(rewards, dtype=np.float32)
        terminated_array = np.asarray(terminated, dtype=np.float32)
        try:
            safety_cost_array = np.asarray(safety_cost, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "safety_cost must be a rectangular array of finite vectors"
            ) from exc
        length = actions_array.shape[0]
        if observations_array.shape != (length + 1, self.observation_dim):
            raise ValueError(
                "observations must have shape "
                f"({length + 1}, {self.observation_dim}), got {observations_array.shape}"
            )
        if actions_array.shape != (length, self.action_dim):
            raise ValueError(
                f"actions must have shape ({length}, {self.action_dim}), "
                f"got {actions_array.shape}"
            )
        if rewards_array.shape != (length,) or terminated_array.shape != (length,):
            raise ValueError("rewards and terminated must contain one value per action")
        self._validate_safety_cost_array(
            safety_cost_array,
            length,
            source="episode",
        )
        if not all(
            np.all(np.isfinite(array))
            for array in (observations_array, actions_array, rewards_array)
        ):
            raise FloatingPointError("Cannot store non-finite replay data")

        if length > self.capacity:
            start = length - self.capacity
            observations_array = observations_array[start:]
            actions_array = actions_array[start:]
            rewards_array = rewards_array[start:]
            terminated_array = terminated_array[start:]
            safety_cost_array = safety_cost_array[start:]
            length = self.capacity

        episode = {
            "observations": observations_array.copy(),
            "actions": actions_array.copy(),
            "rewards": rewards_array.copy(),
            "terminated": terminated_array.copy(),
            "safety_cost": safety_cost_array.copy(),
        }
        while self._episodes and self._size + length > self.capacity:
            self._size -= self._episodes.popleft()["actions"].shape[0]
        self._episodes.append(episode)
        self._size += length

    def sample(
        self, device: torch.device
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        counts = np.asarray(
            [
                max(0, episode["actions"].shape[0] - self.horizon + 1)
                for episode in self._episodes
            ],
            dtype=np.int64,
        )
        total = int(counts.sum())
        if total == 0:
            raise RuntimeError("Replay buffer does not contain a complete sequence")
        cumulative = np.cumsum(counts)

        observations = np.empty(
            (self.horizon + 1, self.batch_size, self.observation_dim),
            dtype=np.float32,
        )
        actions = np.empty(
            (self.horizon, self.batch_size, self.action_dim), dtype=np.float32
        )
        rewards = np.empty((self.horizon, self.batch_size, 1), dtype=np.float32)
        terminated = np.empty((self.horizon, self.batch_size, 1), dtype=np.float32)
        safety_cost = np.empty(
            (self.horizon, self.batch_size, self.safety_cost_dim),
            dtype=np.float32,
        )
        episodes = list(self._episodes)
        sampled = self._rng.integers(0, total, size=self.batch_size)
        for batch_index, global_index in enumerate(sampled):
            episode_index = int(np.searchsorted(cumulative, global_index, side="right"))
            previous = 0 if episode_index == 0 else int(cumulative[episode_index - 1])
            start = int(global_index - previous)
            episode = episodes[episode_index]
            end = start + self.horizon
            observations[:, batch_index] = episode["observations"][start : end + 1]
            actions[:, batch_index] = episode["actions"][start:end]
            rewards[:, batch_index, 0] = episode["rewards"][start:end]
            terminated[:, batch_index, 0] = episode["terminated"][start:end]
            safety_cost[:, batch_index] = episode["safety_cost"][start:end]

        return tuple(
            torch.as_tensor(array, device=device)
            for array in (
                observations,
                actions,
                rewards,
                terminated,
                safety_cost,
            )
        )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "safety_cost_names": self.safety_cost_names,
            "horizon": self.horizon,
            "batch_size": self.batch_size,
            "episodes": list(self._episodes),
            "size": self._size,
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if "safety_cost_names" not in state:
            raise ValueError(
                "Replay checkpoint predates the required safety_cost schema; "
                "legacy replay data cannot be resumed without migration"
            )
        received_safety_names = tuple(state["safety_cost_names"])
        if received_safety_names != self.safety_cost_names:
            raise ValueError(
                "Replay safety-cost channels "
                f"{received_safety_names} do not match current "
                f"{self.safety_cost_names}"
            )
        expected = (
            self.observation_dim,
            self.action_dim,
            self.horizon,
        )
        received = (
            int(state["observation_dim"]),
            int(state["action_dim"]),
            int(state["horizon"]),
        )
        if received != expected:
            raise ValueError(
                f"Replay dimensions/horizon {received} do not match current {expected}"
            )
        episodes = list(state["episodes"])
        for index, episode in enumerate(episodes):
            if "safety_cost" not in episode:
                raise ValueError(
                    "Replay episode "
                    f"{index} is missing required safety_cost transition data"
                )
            length = np.asarray(episode["actions"]).shape[0]
            self._validate_safety_cost_array(
                np.asarray(episode["safety_cost"]),
                length,
                source=f"replay episode {index}",
            )
        self._episodes = deque(episodes)
        self._size = int(state["size"])
        self._rng.bit_generator.state = state["rng_state"]

    def _validate_safety_cost_array(
        self,
        safety_cost: np.ndarray,
        length: int,
        *,
        source: str,
    ) -> None:
        expected_shape = (length, self.safety_cost_dim)
        if safety_cost.shape != expected_shape:
            raise ValueError(
                f"{source} safety_cost must have shape {expected_shape}, "
                f"got {safety_cost.shape}"
            )
        if safety_cost.dtype != np.float32:
            raise TypeError(
                f"{source} safety_cost must use float32, got {safety_cost.dtype}"
            )
        if not np.all(np.isfinite(safety_cost)):
            raise FloatingPointError(
                f"{source} safety_cost contains NaN or infinity"
            )
        if np.any(safety_cost < 0.0):
            raise ValueError(f"{source} safety_cost values must be nonnegative")
