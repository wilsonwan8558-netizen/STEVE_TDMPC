"""Transition replay for future balanced Safety Head supervision.

This buffer is intentionally independent from ``EpisodeReplayBuffer``. It
stores single, temporally aligned ``(observation_t, action_t, safety_cost_t)``
records and owns a separate NumPy random generator.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from envs.safety import (
    CURVATURE_STRATUM_NAMES,
    SAFETY_COST_NAMES,
    validate_curvature_boundaries,
)
from eve.intervention import TRANSLATION_BLOCK_REASON_NAMES

from .replay_buffer import validate_safety_cost_names


SAFETY_AUX_REPLAY_SCHEMA_VERSION = 1
_FLOAT_FIELDS = ("observation", "action", "safety_cost")
_ID_FIELDS = ("translation_block_reason_id", "curvature_stratum_id")
_OPTIONAL_FIELDS = ("terminated", "truncated", "episode_step")


class SafetyAuxReplayBuffer:
    """Fixed-capacity ring buffer with uniform and group-balanced sampling."""

    def __init__(
        self,
        capacity: int,
        observation_dim: int,
        action_dim: int,
        *,
        safety_cost_names: Sequence[str] = SAFETY_COST_NAMES,
        curvature_boundaries_mm_inv: Sequence[float],
        seed: int = 0,
    ) -> None:
        self.capacity = self._positive_integer(capacity, "capacity")
        self.observation_dim = self._positive_integer(
            observation_dim, "observation_dim"
        )
        self.action_dim = self._positive_integer(action_dim, "action_dim")
        self.safety_cost_names = validate_safety_cost_names(
            safety_cost_names,
            source="Safety auxiliary replay",
        )
        self.safety_cost_dim = len(self.safety_cost_names)
        self.translation_block_reason_names = tuple(
            TRANSLATION_BLOCK_REASON_NAMES
        )
        self.curvature_stratum_names = tuple(CURVATURE_STRATUM_NAMES)
        self.curvature_boundaries_mm_inv = validate_curvature_boundaries(
            curvature_boundaries_mm_inv,
            source="Safety auxiliary replay curvature boundaries",
        )

        self._observations = np.empty(
            (self.capacity, self.observation_dim), dtype=np.float32
        )
        self._actions = np.empty(
            (self.capacity, self.action_dim), dtype=np.float32
        )
        self._safety_cost = np.empty(
            (self.capacity, self.safety_cost_dim), dtype=np.float32
        )
        self._translation_block_reason_ids = np.empty(
            self.capacity, dtype=np.int64
        )
        self._curvature_stratum_ids = np.empty(
            self.capacity, dtype=np.int64
        )
        self._terminated = np.empty(self.capacity, dtype=np.bool_)
        self._truncated = np.empty(self.capacity, dtype=np.bool_)
        self._episode_steps = np.empty(self.capacity, dtype=np.int64)
        self._size = 0
        self._next_index = 0
        self._total_added = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    @property
    def total_added(self) -> int:
        return self._total_added

    @property
    def overwritten_count(self) -> int:
        return max(0, self._total_added - self._size)

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        safety_cost: np.ndarray,
        translation_block_reason_id: int,
        curvature_stratum_id: int,
        *,
        terminated: bool = False,
        truncated: bool = False,
        episode_step: int = 0,
    ) -> None:
        """Validate and insert one pre-step-state/action/post-step-cost record."""

        observation_array = self._validated_float_array(
            observation,
            (self.observation_dim,),
            "observation",
        )
        action_array = self._validated_float_array(
            action,
            (self.action_dim,),
            "action",
        )
        safety_cost_array = self._validated_float_array(
            safety_cost,
            (self.safety_cost_dim,),
            "safety_cost",
            nonnegative=True,
        )
        reason_id = self._validated_group_id(
            translation_block_reason_id,
            self.translation_block_reason_names,
            "translation_block_reason_id",
        )
        stratum_id = self._validated_group_id(
            curvature_stratum_id,
            self.curvature_stratum_names,
            "curvature_stratum_id",
        )
        terminated_value = self._validated_bool(terminated, "terminated")
        truncated_value = self._validated_bool(truncated, "truncated")
        episode_step_value = self._nonnegative_integer(
            episode_step, "episode_step"
        )

        index = self._next_index
        self._observations[index] = observation_array
        self._actions[index] = action_array
        self._safety_cost[index] = safety_cost_array
        self._translation_block_reason_ids[index] = reason_id
        self._curvature_stratum_ids[index] = stratum_id
        self._terminated[index] = terminated_value
        self._truncated[index] = truncated_value
        self._episode_steps[index] = episode_step_value
        self._next_index = (index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        self._total_added += 1

    def sample_uniform(
        self, batch_size: int
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Sample stored transitions uniformly with replacement."""

        parsed_batch_size = self._validated_batch_size(batch_size)
        indices = self._rng.integers(
            0, self._size, size=parsed_batch_size, dtype=np.int64
        )
        return self._batch(indices), self._sampling_metadata(
            mode="uniform",
            indices=indices,
            requested_batch_size=parsed_batch_size,
            component_counts={"uniform": parsed_batch_size},
            missing_translation_ids=(),
            missing_curvature_ids=(),
        )

    def sample_stratified(
        self,
        batch_size: int,
        *,
        mode: str,
        translation_fraction: float = 0.5,
        curvature_fraction: float = 0.5,
        translation_reason_ids: Optional[Sequence[int]] = None,
        curvature_stratum_ids: Optional[Sequence[int]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Sample across available canonical groups without inventing labels."""

        parsed_batch_size = self._validated_batch_size(batch_size)
        normalized_mode = str(mode).strip().lower().replace("_", "-")
        supported_modes = {
            "translation-balanced",
            "curvature-balanced",
            "mixed",
        }
        if normalized_mode not in supported_modes:
            raise ValueError(
                f"Unsupported stratified mode {mode!r}; expected one of "
                f"{sorted(supported_modes)}"
            )
        requested_translation = self._validated_requested_groups(
            translation_reason_ids,
            self.translation_block_reason_names,
            "translation_reason_ids",
        )
        requested_curvature = self._validated_requested_groups(
            curvature_stratum_ids,
            self.curvature_stratum_names,
            "curvature_stratum_ids",
        )

        missing_translation: Tuple[int, ...] = ()
        missing_curvature: Tuple[int, ...] = ()
        if normalized_mode == "translation-balanced":
            indices, missing_translation = self._balanced_indices(
                self._translation_block_reason_ids,
                requested_translation,
                parsed_batch_size,
                "translation",
            )
            component_counts = {"translation": parsed_batch_size, "curvature": 0}
        elif normalized_mode == "curvature-balanced":
            indices, missing_curvature = self._balanced_indices(
                self._curvature_stratum_ids,
                requested_curvature,
                parsed_batch_size,
                "curvature",
            )
            component_counts = {"translation": 0, "curvature": parsed_batch_size}
        else:
            translation_value = self._nonnegative_finite_float(
                translation_fraction, "translation_fraction"
            )
            curvature_value = self._nonnegative_finite_float(
                curvature_fraction, "curvature_fraction"
            )
            if not np.isclose(
                translation_value + curvature_value,
                1.0,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    "translation_fraction and curvature_fraction must sum to 1 "
                    "for mixed sampling"
                )
            translation_count, curvature_count = self._mixture_counts(
                parsed_batch_size,
                translation_value,
                curvature_value,
            )
            translation_indices, missing_translation = self._balanced_indices(
                self._translation_block_reason_ids,
                requested_translation,
                translation_count,
                "translation",
            )
            curvature_indices, missing_curvature = self._balanced_indices(
                self._curvature_stratum_ids,
                requested_curvature,
                curvature_count,
                "curvature",
            )
            indices = np.concatenate(
                (translation_indices, curvature_indices)
            ).astype(np.int64, copy=False)
            if indices.size:
                indices = indices[self._rng.permutation(indices.size)]
            component_counts = {
                "translation": translation_count,
                "curvature": curvature_count,
            }

        return self._batch(indices), self._sampling_metadata(
            mode=normalized_mode,
            indices=indices,
            requested_batch_size=parsed_batch_size,
            component_counts=component_counts,
            missing_translation_ids=missing_translation,
            missing_curvature_ids=missing_curvature,
        )

    def state_dict(self) -> Dict[str, Any]:
        """Return a strict, self-describing, deep-copied replay snapshot."""

        valid = slice(0, self._size)
        return {
            "schema_version": SAFETY_AUX_REPLAY_SCHEMA_VERSION,
            "capacity": self.capacity,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "safety_cost_names": self.safety_cost_names,
            "safety_cost_dim": self.safety_cost_dim,
            "translation_block_reason_names": (
                self.translation_block_reason_names
            ),
            "curvature_stratum_names": self.curvature_stratum_names,
            "curvature_boundaries_mm_inv": (
                self.curvature_boundaries_mm_inv
            ),
            "size": self._size,
            "next_index": self._next_index,
            "total_added": self._total_added,
            "observation": self._observations[valid].copy(),
            "action": self._actions[valid].copy(),
            "safety_cost": self._safety_cost[valid].copy(),
            "translation_block_reason_id": (
                self._translation_block_reason_ids[valid].copy()
            ),
            "curvature_stratum_id": (
                self._curvature_stratum_ids[valid].copy()
            ),
            "terminated": self._terminated[valid].copy(),
            "truncated": self._truncated[valid].copy(),
            "episode_step": self._episode_steps[valid].copy(),
            "rng_bit_generator": self._rng.bit_generator.__class__.__name__,
            "rng_state": copy.deepcopy(self._rng.bit_generator.state),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Strictly validate a snapshot before atomically replacing this state."""

        if not isinstance(state, Mapping):
            raise TypeError("Safety auxiliary replay state must be a mapping")
        expected_keys = {
            "schema_version",
            "capacity",
            "observation_dim",
            "action_dim",
            "safety_cost_names",
            "safety_cost_dim",
            "translation_block_reason_names",
            "curvature_stratum_names",
            "curvature_boundaries_mm_inv",
            "size",
            "next_index",
            "total_added",
            *_FLOAT_FIELDS,
            *_ID_FIELDS,
            *_OPTIONAL_FIELDS,
            "rng_bit_generator",
            "rng_state",
        }
        missing = sorted(expected_keys - state.keys())
        unexpected = sorted(state.keys() - expected_keys)
        if missing or unexpected:
            raise ValueError(
                "Safety auxiliary replay state keys mismatch; "
                f"missing={missing}, unexpected={unexpected}"
            )

        schema_version = self._nonnegative_integer(
            state["schema_version"], "state schema_version"
        )
        if schema_version != SAFETY_AUX_REPLAY_SCHEMA_VERSION:
            raise ValueError(
                "Safety auxiliary replay schema version "
                f"{schema_version} does not match required "
                f"{SAFETY_AUX_REPLAY_SCHEMA_VERSION}"
            )
        for key, expected in (
            ("capacity", self.capacity),
            ("observation_dim", self.observation_dim),
            ("action_dim", self.action_dim),
            ("safety_cost_dim", self.safety_cost_dim),
        ):
            received = self._positive_integer(state[key], f"state {key}")
            if received != expected:
                raise ValueError(
                    f"Safety auxiliary replay {key} {received} does not match "
                    f"current {expected}"
                )
        received_cost_names = tuple(state["safety_cost_names"])
        if received_cost_names != self.safety_cost_names:
            raise ValueError(
                "Safety auxiliary replay safety-cost channel order mismatch: "
                f"{received_cost_names} != {self.safety_cost_names}"
            )
        received_reason_names = tuple(state["translation_block_reason_names"])
        if received_reason_names != self.translation_block_reason_names:
            raise ValueError(
                "Safety auxiliary replay translation-block reason order "
                f"mismatch: {received_reason_names} != "
                f"{self.translation_block_reason_names}"
            )
        received_stratum_names = tuple(state["curvature_stratum_names"])
        if received_stratum_names != self.curvature_stratum_names:
            raise ValueError(
                "Safety auxiliary replay curvature-stratum order mismatch: "
                f"{received_stratum_names} != {self.curvature_stratum_names}"
            )
        received_boundaries = validate_curvature_boundaries(
            state["curvature_boundaries_mm_inv"],
            source="Safety auxiliary replay state curvature boundaries",
        )
        if received_boundaries != self.curvature_boundaries_mm_inv:
            raise ValueError(
                "Safety auxiliary replay curvature-boundary mismatch: "
                f"{received_boundaries} != "
                f"{self.curvature_boundaries_mm_inv}"
            )

        size = self._nonnegative_integer(state["size"], "state size")
        if size > self.capacity:
            raise ValueError(
                f"Safety auxiliary replay size {size} exceeds capacity "
                f"{self.capacity}"
            )
        next_index = self._nonnegative_integer(
            state["next_index"], "state next_index"
        )
        expected_next = size if size < self.capacity else None
        if (
            (expected_next is not None and next_index != expected_next)
            or (size == self.capacity and next_index >= self.capacity)
        ):
            raise ValueError(
                f"Safety auxiliary replay next_index {next_index} is invalid "
                f"for size={size}, capacity={self.capacity}"
            )
        total_added = self._nonnegative_integer(
            state["total_added"], "state total_added"
        )
        if size < self.capacity and total_added != size:
            raise ValueError(
                "Safety auxiliary replay total_added must equal size before "
                "the first capacity wrap"
            )
        if size == self.capacity and total_added < self.capacity:
            raise ValueError(
                "Safety auxiliary replay total_added cannot be smaller than "
                "a full buffer's capacity"
            )
        if next_index != total_added % self.capacity:
            raise ValueError(
                "Safety auxiliary replay next_index does not match "
                "total_added modulo capacity"
            )

        observations = self._validated_state_float_array(
            state["observation"],
            (size, self.observation_dim),
            "state observation",
        )
        actions = self._validated_state_float_array(
            state["action"],
            (size, self.action_dim),
            "state action",
        )
        safety_cost = self._validated_state_float_array(
            state["safety_cost"],
            (size, self.safety_cost_dim),
            "state safety_cost",
            nonnegative=True,
        )
        reason_ids = self._validated_state_id_array(
            state["translation_block_reason_id"],
            size,
            len(self.translation_block_reason_names),
            "state translation_block_reason_id",
        )
        stratum_ids = self._validated_state_id_array(
            state["curvature_stratum_id"],
            size,
            len(self.curvature_stratum_names),
            "state curvature_stratum_id",
        )
        terminated = self._validated_state_array(
            state["terminated"], (size,), np.dtype(np.bool_), "state terminated"
        )
        truncated = self._validated_state_array(
            state["truncated"], (size,), np.dtype(np.bool_), "state truncated"
        )
        episode_steps = self._validated_state_array(
            state["episode_step"],
            (size,),
            np.dtype(np.int64),
            "state episode_step",
        )
        if np.any(episode_steps < 0):
            raise ValueError("state episode_step values must be nonnegative")

        expected_bit_generator = self._rng.bit_generator.__class__.__name__
        if state["rng_bit_generator"] != expected_bit_generator:
            raise ValueError(
                "Safety auxiliary replay RNG bit generator "
                f"{state['rng_bit_generator']!r} does not match "
                f"{expected_bit_generator!r}"
            )
        restored_rng = np.random.default_rng()
        try:
            restored_rng.bit_generator.state = copy.deepcopy(state["rng_state"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Safety auxiliary replay RNG state is invalid"
            ) from exc

        self._observations[:size] = observations
        self._actions[:size] = actions
        self._safety_cost[:size] = safety_cost
        self._translation_block_reason_ids[:size] = reason_ids
        self._curvature_stratum_ids[:size] = stratum_ids
        self._terminated[:size] = terminated
        self._truncated[:size] = truncated
        self._episode_steps[:size] = episode_steps
        self._size = size
        self._next_index = next_index
        self._total_added = total_added
        self._rng = restored_rng

    def _balanced_indices(
        self,
        labels: np.ndarray,
        requested_ids: Tuple[int, ...],
        count: int,
        label: str,
    ) -> Tuple[np.ndarray, Tuple[int, ...]]:
        available = []
        missing = []
        candidates: Dict[int, np.ndarray] = {}
        for group_id in requested_ids:
            group_candidates = np.flatnonzero(
                labels[: self._size] == group_id
            ).astype(np.int64)
            candidates[group_id] = group_candidates
            if group_candidates.size:
                available.append(group_id)
            else:
                missing.append(group_id)
        if count == 0:
            return np.empty(0, dtype=np.int64), tuple(missing)
        if not available:
            raise RuntimeError(
                f"No stored samples belong to the requested {label} groups"
            )

        quotient, remainder = divmod(count, len(available))
        quotas = {group_id: quotient for group_id in available}
        if remainder:
            selected = self._rng.choice(
                np.asarray(available, dtype=np.int64),
                size=remainder,
                replace=False,
            )
            for group_id in selected:
                quotas[int(group_id)] += 1
        sampled = [
            self._rng.choice(
                candidates[group_id],
                size=quotas[group_id],
                replace=True,
            ).astype(np.int64)
            for group_id in available
            if quotas[group_id] > 0
        ]
        indices = (
            np.concatenate(sampled)
            if sampled
            else np.empty(0, dtype=np.int64)
        )
        if indices.size:
            indices = indices[self._rng.permutation(indices.size)]
        return indices, tuple(missing)

    def _batch(self, indices: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            "observation": self._observations[indices].copy(),
            "action": self._actions[indices].copy(),
            "safety_cost": self._safety_cost[indices].copy(),
            "translation_block_reason_id": (
                self._translation_block_reason_ids[indices].copy()
            ),
            "curvature_stratum_id": (
                self._curvature_stratum_ids[indices].copy()
            ),
            "terminated": self._terminated[indices].copy(),
            "truncated": self._truncated[indices].copy(),
            "episode_step": self._episode_steps[indices].copy(),
        }

    def _sampling_metadata(
        self,
        *,
        mode: str,
        indices: np.ndarray,
        requested_batch_size: int,
        component_counts: Mapping[str, int],
        missing_translation_ids: Sequence[int],
        missing_curvature_ids: Sequence[int],
    ) -> Dict[str, Any]:
        reason_counts = np.bincount(
            self._translation_block_reason_ids[indices],
            minlength=len(self.translation_block_reason_names),
        )
        stratum_counts = np.bincount(
            self._curvature_stratum_ids[indices],
            minlength=len(self.curvature_stratum_names),
        )
        available_reason_counts = np.bincount(
            self._translation_block_reason_ids[: self._size],
            minlength=len(self.translation_block_reason_names),
        )
        available_stratum_counts = np.bincount(
            self._curvature_stratum_ids[: self._size],
            minlength=len(self.curvature_stratum_names),
        )
        return {
            "mode": mode,
            "requested_batch_size": int(requested_batch_size),
            "returned_batch_size": int(indices.size),
            "with_replacement": True,
            "component_counts": {
                key: int(value) for key, value in component_counts.items()
            },
            "available_translation_counts": {
                name: int(available_reason_counts[index])
                for index, name in enumerate(
                    self.translation_block_reason_names
                )
            },
            "available_curvature_counts": {
                name: int(available_stratum_counts[index])
                for index, name in enumerate(self.curvature_stratum_names)
            },
            "sampled_translation_counts": {
                name: int(reason_counts[index])
                for index, name in enumerate(
                    self.translation_block_reason_names
                )
            },
            "sampled_curvature_counts": {
                name: int(stratum_counts[index])
                for index, name in enumerate(self.curvature_stratum_names)
            },
            "missing_translation_groups": [
                self.translation_block_reason_names[index]
                for index in missing_translation_ids
            ],
            "missing_curvature_groups": [
                self.curvature_stratum_names[index]
                for index in missing_curvature_ids
            ],
        }

    @staticmethod
    def _mixture_counts(
        batch_size: int,
        translation_fraction: float,
        curvature_fraction: float,
    ) -> Tuple[int, int]:
        raw = np.asarray(
            [
                batch_size * translation_fraction,
                batch_size * curvature_fraction,
            ],
            dtype=np.float64,
        )
        counts = np.floor(raw).astype(np.int64)
        remainder = int(batch_size - counts.sum())
        if remainder:
            order = np.argsort(-(raw - counts), kind="stable")
            counts[order[:remainder]] += 1
        return int(counts[0]), int(counts[1])

    def _validated_batch_size(self, batch_size: int) -> int:
        parsed = self._positive_integer(batch_size, "batch_size")
        if self._size == 0:
            raise RuntimeError("Safety auxiliary replay buffer is empty")
        return parsed

    @staticmethod
    def _validated_float_array(
        value: np.ndarray,
        shape: Tuple[int, ...],
        label: str,
        *,
        nonnegative: bool = False,
    ) -> np.ndarray:
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{label} must be a numpy.ndarray")
        if value.shape != shape:
            raise ValueError(f"{label} must have shape {shape}, got {value.shape}")
        if value.dtype != np.float32:
            raise TypeError(f"{label} must use float32, got {value.dtype}")
        if not np.all(np.isfinite(value)):
            raise FloatingPointError(f"{label} contains NaN or infinity")
        if nonnegative and np.any(value < 0.0):
            raise ValueError(f"{label} values must be nonnegative")
        return value

    @classmethod
    def _validated_state_float_array(
        cls,
        value: Any,
        shape: Tuple[int, ...],
        label: str,
        *,
        nonnegative: bool = False,
    ) -> np.ndarray:
        array = cls._validated_state_array(
            value, shape, np.dtype(np.float32), label
        )
        if not np.all(np.isfinite(array)):
            raise FloatingPointError(f"{label} contains NaN or infinity")
        if nonnegative and np.any(array < 0.0):
            raise ValueError(f"{label} values must be nonnegative")
        return array

    @staticmethod
    def _validated_state_array(
        value: Any,
        shape: Tuple[int, ...],
        dtype: np.dtype,
        label: str,
    ) -> np.ndarray:
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{label} must be a numpy.ndarray")
        if value.shape != shape:
            raise ValueError(f"{label} must have shape {shape}, got {value.shape}")
        if value.dtype != dtype:
            raise TypeError(f"{label} must use {dtype}, got {value.dtype}")
        return value.copy()

    @classmethod
    def _validated_state_id_array(
        cls,
        value: Any,
        size: int,
        group_count: int,
        label: str,
    ) -> np.ndarray:
        array = cls._validated_state_array(
            value, (size,), np.dtype(np.int64), label
        )
        if np.any(array < 0) or np.any(array >= group_count):
            raise ValueError(
                f"{label} values must be in [0, {group_count - 1}]"
            )
        return array

    @staticmethod
    def _validated_group_id(
        value: int,
        names: Sequence[str],
        label: str,
    ) -> int:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"{label} must be an integer")
        parsed = int(value)
        if not 0 <= parsed < len(names):
            raise ValueError(f"{label} must be in [0, {len(names) - 1}]")
        return parsed

    @classmethod
    def _validated_requested_groups(
        cls,
        values: Optional[Sequence[int]],
        names: Sequence[str],
        label: str,
    ) -> Tuple[int, ...]:
        if values is None:
            return tuple(range(len(names)))
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise TypeError(f"{label} must be a sequence of integer IDs")
        parsed = tuple(
            cls._validated_group_id(value, names, label) for value in values
        )
        if not parsed:
            raise ValueError(f"{label} cannot be empty")
        if len(set(parsed)) != len(parsed):
            raise ValueError(f"{label} cannot contain duplicate IDs")
        return parsed

    @staticmethod
    def _positive_integer(value: Any, label: str) -> int:
        parsed = SafetyAuxReplayBuffer._nonnegative_integer(value, label)
        if parsed <= 0:
            raise ValueError(f"{label} must be positive")
        return parsed

    @staticmethod
    def _nonnegative_integer(value: Any, label: str) -> int:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{label} must be an integer, not bool")
        if not isinstance(value, (int, np.integer)):
            raise TypeError(f"{label} must be an integer")
        parsed = int(value)
        if parsed < 0:
            raise ValueError(f"{label} must be nonnegative")
        return parsed

    @staticmethod
    def _validated_bool(value: Any, label: str) -> bool:
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{label} must be a bool")
        return bool(value)

    @staticmethod
    def _nonnegative_finite_float(value: Any, label: str) -> float:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{label} must be a real number, not bool")
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(f"{label} must be a real number") from exc
        if not np.isfinite(parsed) or parsed < 0.0:
            raise ValueError(f"{label} must be finite and nonnegative")
        return parsed
