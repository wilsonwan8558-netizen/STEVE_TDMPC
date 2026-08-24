"""Strict offline auxiliary supervision state for the TD-MPC2 Safety Head.

The supervisor owns only an immutable dataset view, a train-split sampler, and
resume metadata.  It never writes transitions and never exposes validation
samples to the optimization sampler.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from envs.safety import CURVATURE_STRATUM_NAMES
from safety_schema import TRANSLATION_BLOCK_REASON_NAMES

from .common import (
    SAFETY_AUX_TRANSLATION_GROUP_NAMES,
    resolve_project_path,
)
from .safety_aux_dataset import (
    SAFETY_AUX_DATASET_SCHEMA_VERSION,
    SafetyAuxDataset,
    load_dataset,
)
from .safety_aux_replay import (
    SAFETY_AUX_REPLAY_SCHEMA_VERSION,
    SafetyAuxReplayBuffer,
)


SAFETY_AUXILIARY_STATE_SCHEMA_VERSION = 2
SAFETY_AUXILIARY_CHECKPOINT_KEY = "safety_auxiliary"

_STATE_KEYS = {
    "schema_version",
    "enabled",
    "config",
    "calibration",
    "dataset",
    "sampler",
    "auxiliary_update_count",
    "last_normal_update_count",
}
_CALIBRATION_KEYS = {
    "curvature_loss_coef",
    "translation_loss_coef",
    "translation_group_weights",
    "translation_zero_calibration_coef",
    "validation_translation_threshold_candidates",
}
_DATASET_IDENTITY_KEYS = {
    "resolved_path",
    "fingerprint",
    "dataset_schema_version",
    "replay_schema_version",
    "observation_dim",
    "action_dim",
    "safety_cost_names",
    "translation_block_reason_names",
    "curvature_stratum_names",
    "curvature_boundaries_mm_inv",
    "split",
}


def _exact_nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{label} must be an integer, not {type(value).__name__}")
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{label} must be nonnegative")
    return parsed


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    missing = sorted(expected - set(value))
    unexpected = sorted(set(value) - expected)
    if missing or unexpected:
        raise ValueError(
            f"{label} keys mismatch; missing={missing}, unexpected={unexpected}"
        )


def _arrays_equal(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return bool(np.array_equal(np.asarray(left), np.asarray(right)))
        except (TypeError, ValueError):
            return False
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return False
        return all(_arrays_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(
            _arrays_equal(a, b) for a, b in zip(left, right)
        )
    return bool(left == right)


class SafetyAuxiliarySupervisor:
    """Own the strict train-only sampler and deterministic validation view."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        observation_dim: int,
        action_dim: int,
        safety_cost_names: Sequence[str],
        curvature_boundaries_mm_inv: Sequence[float],
        seed: int,
    ) -> None:
        if not isinstance(config, Mapping):
            raise TypeError("Resolved safety_aux config must be a mapping")
        if config.get("enabled") is not True:
            raise ValueError(
                "SafetyAuxiliarySupervisor requires safety_aux.enabled=true"
            )
        self.config = copy.deepcopy(dict(config))
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.safety_cost_names = tuple(str(name) for name in safety_cost_names)
        self.curvature_boundaries_mm_inv = tuple(
            float(value) for value in curvature_boundaries_mm_inv
        )
        self.dataset_path = resolve_project_path(self.config["dataset_path"])
        if not self.dataset_path.is_file():
            raise FileNotFoundError(
                "Safety auxiliary dataset does not exist or is not a file: "
                f"{self.dataset_path}"
            )

        self.dataset = load_dataset(
            self.dataset_path,
            expected_curvature_boundaries_mm_inv=(
                self.curvature_boundaries_mm_inv
            ),
        )
        self._dataset_state = self.dataset.state_dict()
        self._validate_dataset_identity()
        if self.dataset.train_size <= 0:
            raise ValueError("Safety auxiliary training split is empty")
        if self.dataset.validation_size <= 0:
            raise ValueError("Safety auxiliary validation split is empty")
        self._validate_available_translation_weights()

        self.training_buffer = self.dataset.training_buffer()
        # The validation replay is never sampled.  Its state is a convenient,
        # immutable-by-copy representation in stored validation order.
        self._validation_state = self.dataset.validation_buffer().state_dict()
        self.auxiliary_update_count = 0
        self.last_normal_update_count = 0
        self._pending_update: Optional[Dict[str, Any]] = None

        # Dataset construction seeds the subset buffer from split metadata.
        # Auxiliary training owns a run-specific RNG, independent of the
        # dataset, main replay, environment, planning, and global NumPy RNG.
        self.training_buffer.reseed_sampler(seed)

    @property
    def fingerprint(self) -> str:
        return self.dataset.fingerprint

    @property
    def update_pending(self) -> bool:
        """Whether a sampled normal-update contribution awaits commit."""

        return self._pending_update is not None

    def _validate_dataset_identity(self) -> None:
        replay = self._dataset_state["replay_state"]
        if int(replay["observation_dim"]) != self.observation_dim:
            raise ValueError(
                "Safety auxiliary dataset observation_dim "
                f"{replay['observation_dim']} does not match agent "
                f"{self.observation_dim}"
            )
        if int(replay["action_dim"]) != self.action_dim:
            raise ValueError(
                "Safety auxiliary dataset action_dim "
                f"{replay['action_dim']} does not match agent {self.action_dim}"
            )
        if tuple(replay["safety_cost_names"]) != self.safety_cost_names:
            raise ValueError(
                "Safety auxiliary dataset Safety channel order mismatch: "
                f"{tuple(replay['safety_cost_names'])} != "
                f"{self.safety_cost_names}"
            )
        if (
            tuple(replay["translation_block_reason_names"])
            != tuple(TRANSLATION_BLOCK_REASON_NAMES)
        ):
            raise ValueError(
                "Safety auxiliary dataset translation-block reason schema "
                "does not match the simulator"
            )
        if tuple(replay["curvature_stratum_names"]) != tuple(
            CURVATURE_STRATUM_NAMES
        ):
            raise ValueError(
                "Safety auxiliary dataset curvature-stratum schema does not "
                "match the environment adapter"
            )
        if (
            tuple(float(value) for value in replay["curvature_boundaries_mm_inv"])
            != self.curvature_boundaries_mm_inv
        ):
            raise ValueError(
                "Safety auxiliary dataset curvature boundaries do not match "
                "the configured diagnostic boundaries"
            )
        requested_action = replay["action"]
        if np.any(requested_action < -1.0) or np.any(requested_action > 1.0):
            raise ValueError(
                "Safety auxiliary dataset requested actions exceed normalized "
                "bounds [-1, 1]"
            )
        applied_action = replay["applied_action"]
        if np.any(applied_action < -1.0) or np.any(applied_action > 1.0):
            raise ValueError(
                "Safety auxiliary dataset applied actions exceed normalized "
                "bounds [-1, 1]"
            )

    def _validate_available_translation_weights(self) -> None:
        replay = self._dataset_state["replay_state"]
        train_indices = self.dataset.train_indices
        train_reason_ids = replay["translation_block_reason_id"][
            train_indices
        ]
        available_reason_ids = set(
            int(value) for value in np.unique(train_reason_ids)
        )
        canonical_names = tuple(TRANSLATION_BLOCK_REASON_NAMES)
        if canonical_names != SAFETY_AUX_TRANSLATION_GROUP_NAMES:
            raise RuntimeError(
                "Configured canonical translation groups do not match the "
                "simulator translation-block reason schema"
            )
        weights = self.config["translation_group_weights"]
        self.available_translation_reason_names = tuple(
            name
            for reason_id, name in enumerate(canonical_names)
            if reason_id in available_reason_ids
        )
        self.positive_weighted_translation_reason_names = tuple(
            name
            for name in self.available_translation_reason_names
            if float(weights[name]) > 0.0
        )
        if not self.positive_weighted_translation_reason_names:
            raise ValueError(
                "Safety auxiliary training split has no available translation "
                "reason with a positive configured group weight; "
                f"available={self.available_translation_reason_names}"
            )

    def calibration_metadata(self) -> Dict[str, Any]:
        """Return the exact Commit-4.6D loss/calibration contract."""

        return {
            "curvature_loss_coef": float(
                self.config["curvature_loss_coef"]
            ),
            "translation_loss_coef": float(
                self.config["translation_loss_coef"]
            ),
            "translation_group_weights": copy.deepcopy(
                self.config["translation_group_weights"]
            ),
            "translation_zero_calibration_coef": float(
                self.config["translation_zero_calibration_coef"]
            ),
            "validation_translation_threshold_candidates": copy.deepcopy(
                self.config[
                    "validation_translation_threshold_candidates"
                ]
            ),
        }

    def dataset_identity(self) -> Dict[str, Any]:
        replay = self._dataset_state["replay_state"]
        return {
            "resolved_path": str(self.dataset_path),
            "fingerprint": self.dataset.fingerprint,
            "dataset_schema_version": int(
                self._dataset_state["schema_version"]
            ),
            "replay_schema_version": int(replay["schema_version"]),
            "observation_dim": int(replay["observation_dim"]),
            "action_dim": int(replay["action_dim"]),
            "safety_cost_names": tuple(replay["safety_cost_names"]),
            "translation_block_reason_names": tuple(
                replay["translation_block_reason_names"]
            ),
            "curvature_stratum_names": tuple(
                replay["curvature_stratum_names"]
            ),
            "curvature_boundaries_mm_inv": tuple(
                float(value)
                for value in replay["curvature_boundaries_mm_inv"]
            ),
            "split": copy.deepcopy(self._dataset_state["split"]),
        }

    def metadata_summary(self) -> Dict[str, Any]:
        identity = self.dataset_identity()
        split = identity.pop("split")
        identity.update(
            {
                "train_size": self.dataset.train_size,
                "validation_size": self.dataset.validation_size,
                "split_strategy": split["strategy"],
                "split_assignment_algorithm": split[
                    "assignment_algorithm"
                ],
                "split_seed": int(split["seed"]),
                "validation_fraction": float(split["validation_fraction"]),
                "available_translation_reason_names": list(
                    self.available_translation_reason_names
                ),
                "positive_weighted_translation_reason_names": list(
                    self.positive_weighted_translation_reason_names
                ),
                "calibration": self.calibration_metadata(),
            }
        )
        return identity

    def validation_batch(self) -> Dict[str, Any]:
        """Return every fixed validation transition once, in stored order."""

        return copy.deepcopy(self._validation_state)

    def sample_for_update(
        self,
        normal_update_count: int,
    ) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[Dict[str, Any]]]:
        """Sample and immediately commit one standalone schedule advance.

        Training code should use :meth:`begin_update`, followed by
        :meth:`commit_update` only after ``agent.update()`` succeeds.  This
        convenience method remains atomic for diagnostics and tests.
        """

        parsed_update = _exact_nonnegative_integer(
            normal_update_count, "normal_update_count"
        )
        previous_normal_count = self.last_normal_update_count
        previous_auxiliary_count = self.auxiliary_update_count
        previous_sampler_state = self.training_buffer.sampler_state_dict()
        try:
            return self._sample_for_update_impl(parsed_update)
        except BaseException:
            self.training_buffer.load_sampler_state_dict(
                previous_sampler_state
            )
            self.last_normal_update_count = previous_normal_count
            self.auxiliary_update_count = previous_auxiliary_count
            raise

    def _sample_for_update_impl(
        self,
        parsed_update: int,
    ) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[Dict[str, Any]]]:
        if parsed_update != self.last_normal_update_count + 1:
            raise ValueError(
                "Safety auxiliary update schedule must advance exactly once per "
                "normal update: expected "
                f"{self.last_normal_update_count + 1}, got {parsed_update}"
            )
        self.last_normal_update_count = parsed_update
        if parsed_update % int(self.config["update_interval"]) != 0:
            return None, None

        batch_size = int(self.config["batch_size"])
        replacement = bool(self.config["sample_with_replacement"])
        mode = str(self.config["sampling_mode"])
        if mode == "uniform":
            batch, metadata = self.training_buffer.sample_uniform(
                batch_size,
                sample_with_replacement=replacement,
            )
        else:
            batch, metadata = self.training_buffer.sample_stratified(
                batch_size,
                mode=mode,
                translation_fraction=float(
                    self.config["translation_fraction"]
                ),
                curvature_fraction=float(
                    self.config["curvature_fraction"]
                ),
                sample_with_replacement=replacement,
            )

        compact_indices = np.asarray(
            metadata["sampled_indices"], dtype=np.int64
        )
        source_indices = self.dataset.train_indices[compact_indices]
        validation_indices = self.dataset.validation_indices
        if np.intersect1d(source_indices, validation_indices).size:
            raise RuntimeError(
                "Safety auxiliary sampler exposed a validation transition"
            )
        metadata = copy.deepcopy(metadata)
        metadata["source_dataset_indices"] = source_indices.copy()
        metadata["normal_update_count"] = parsed_update
        self.auxiliary_update_count += 1
        metadata["auxiliary_update_count"] = self.auxiliary_update_count
        return batch, metadata

    def begin_update(
        self,
        normal_update_count: int,
    ) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[Dict[str, Any]]]:
        """Begin one transactional schedule advance for a normal update."""

        if self._pending_update is not None:
            raise RuntimeError(
                "A Safety auxiliary update transaction is already pending"
            )
        prior = {
            "sampler": self.training_buffer.sampler_state_dict(),
            "auxiliary_update_count": self.auxiliary_update_count,
            "last_normal_update_count": self.last_normal_update_count,
        }
        try:
            batch, metadata = self.sample_for_update(normal_update_count)
            self._pending_update = {
                **prior,
                "expected_normal_update_count": int(normal_update_count),
            }
            return batch, metadata
        except BaseException:
            # Cover asynchronous interruption between the atomic sampler call
            # and publication of the pending transaction.
            self.training_buffer.load_sampler_state_dict(prior["sampler"])
            self.auxiliary_update_count = int(
                prior["auxiliary_update_count"]
            )
            self.last_normal_update_count = int(
                prior["last_normal_update_count"]
            )
            self._pending_update = None
            raise

    def commit_update(self, normal_update_count: int) -> None:
        """Commit a pending auxiliary schedule advance after agent success."""

        if self._pending_update is None:
            raise RuntimeError("No Safety auxiliary update transaction is pending")
        parsed_update = _exact_nonnegative_integer(
            normal_update_count, "normal_update_count"
        )
        expected = int(
            self._pending_update["expected_normal_update_count"]
        )
        if (
            parsed_update != expected
            or self.last_normal_update_count != expected
        ):
            raise ValueError(
                "Safety auxiliary transaction count does not match the "
                f"completed agent update: expected {expected}, got "
                f"{parsed_update}"
            )
        self._pending_update = None

    def rollback_update(self) -> None:
        """Restore counters and sampler RNG after a failed agent update."""

        if self._pending_update is None:
            return
        pending = self._pending_update
        self.training_buffer.load_sampler_state_dict(pending["sampler"])
        self.auxiliary_update_count = int(
            pending["auxiliary_update_count"]
        )
        self.last_normal_update_count = int(
            pending["last_normal_update_count"]
        )
        self._pending_update = None

    @staticmethod
    def unavailable_sampling_metrics() -> Dict[str, float]:
        names = (
            "safety_aux_batch_size",
            "safety_aux_unique_sample_count",
            "safety_aux_duplicate_fraction",
            "safety_aux_replacement_used",
            "safety_aux_reason_none_count",
            "safety_aux_reason_lower_count",
            "safety_aux_reason_device_count",
            "safety_aux_reason_tree_end_count",
            "safety_aux_reason_other_count",
            "safety_aux_curvature_low_count",
            "safety_aux_curvature_medium_count",
            "safety_aux_curvature_high_count",
            "safety_aux_curvature_extreme_count",
        )
        return {name: float("nan") for name in names}

    @classmethod
    def sampling_metrics(
        cls, metadata: Optional[Mapping[str, Any]]
    ) -> Dict[str, float]:
        if metadata is None:
            return cls.unavailable_sampling_metrics()
        reasons = metadata["sampled_translation_counts"]
        curvature = metadata["sampled_curvature_counts"]
        return {
            "safety_aux_batch_size": float(
                metadata["returned_batch_size"]
            ),
            "safety_aux_unique_sample_count": float(
                metadata["unique_sample_count"]
            ),
            "safety_aux_duplicate_fraction": float(
                metadata["duplicate_exposure_fraction"]
            ),
            "safety_aux_replacement_used": float(
                bool(metadata["replacement_used"])
            ),
            "safety_aux_reason_none_count": float(reasons["none"]),
            "safety_aux_reason_lower_count": float(
                reasons["lower_insertion_boundary"]
            ),
            "safety_aux_reason_device_count": float(
                reasons["device_length_limit"]
            ),
            "safety_aux_reason_tree_end_count": float(
                reasons["vessel_tree_end"]
            ),
            "safety_aux_reason_other_count": float(reasons["other"]),
            "safety_aux_curvature_low_count": float(curvature["low"]),
            "safety_aux_curvature_medium_count": float(
                curvature["medium"]
            ),
            "safety_aux_curvature_high_count": float(curvature["high"]),
            "safety_aux_curvature_extreme_count": float(
                curvature["extreme"]
            ),
        }

    def state_dict(self) -> Dict[str, Any]:
        if self._pending_update is not None:
            raise RuntimeError(
                "Cannot checkpoint a pending Safety auxiliary update"
            )
        resolved_config = copy.deepcopy(self.config)
        resolved_config["dataset_path"] = str(self.dataset_path)
        return {
            "schema_version": SAFETY_AUXILIARY_STATE_SCHEMA_VERSION,
            "enabled": True,
            "config": resolved_config,
            "calibration": self.calibration_metadata(),
            "dataset": self.dataset_identity(),
            "sampler": self.training_buffer.sampler_state_dict(),
            "auxiliary_update_count": self.auxiliary_update_count,
            "last_normal_update_count": self.last_normal_update_count,
        }

    def load_state_dict(
        self,
        state: Mapping[str, Any],
        *,
        expected_normal_update_count: Optional[int] = None,
        allow_dataset_path_override: bool = False,
    ) -> None:
        if self._pending_update is not None:
            raise RuntimeError(
                "Cannot restore Safety auxiliary state during a pending update"
            )
        if not isinstance(state, Mapping):
            raise TypeError("Safety auxiliary checkpoint state must be a mapping")
        if "schema_version" not in state:
            raise ValueError(
                "Safety auxiliary checkpoint state is missing schema_version "
                "and required Commit-4.6D calibration metadata"
            )
        schema = _exact_nonnegative_integer(
            state["schema_version"], "Safety auxiliary state schema_version"
        )
        if schema != SAFETY_AUXILIARY_STATE_SCHEMA_VERSION:
            legacy_note = (
                " This enabled checkpoint predates Commit-4.6D calibration "
                "metadata and cannot be resumed without an explicit compatible "
                "configuration and retraining."
                if schema < SAFETY_AUXILIARY_STATE_SCHEMA_VERSION
                else ""
            )
            raise ValueError(
                f"Safety auxiliary state schema {schema} is incompatible with "
                f"required {SAFETY_AUXILIARY_STATE_SCHEMA_VERSION}."
                f"{legacy_note}"
            )
        _exact_keys(state, _STATE_KEYS, "Safety auxiliary checkpoint state")
        if state["enabled"] is not True:
            raise ValueError("Safety auxiliary checkpoint state must be enabled")
        if not isinstance(state["config"], Mapping):
            raise TypeError("Safety auxiliary checkpoint config must be a mapping")
        expected_config = copy.deepcopy(self.config)
        expected_config["dataset_path"] = str(self.dataset_path)
        received_config = copy.deepcopy(dict(state["config"]))
        _exact_keys(
            received_config,
            set(expected_config),
            "Safety auxiliary checkpoint config",
        )
        received_path = received_config.get("dataset_path")
        if isinstance(received_path, str):
            received_config["dataset_path"] = str(
                resolve_project_path(received_path)
            )
        received_weight_names = tuple(
            received_config.get("translation_group_weights", {})
        )
        if received_weight_names != SAFETY_AUX_TRANSLATION_GROUP_NAMES:
            raise ValueError(
                "Safety auxiliary checkpoint config translation group weights "
                "do not use canonical names and order"
            )
        config_matches = _arrays_equal(received_config, expected_config)
        if allow_dataset_path_override:
            received_config.pop("dataset_path", None)
            expected_config.pop("dataset_path", None)
            config_matches = _arrays_equal(received_config, expected_config)
        if not config_matches:
            raise ValueError(
                "Safety auxiliary checkpoint config does not match the "
                "requested resolved config"
            )
        calibration_state = state["calibration"]
        if not isinstance(calibration_state, Mapping):
            raise TypeError(
                "Safety auxiliary checkpoint calibration metadata must be a "
                "mapping"
            )
        _exact_keys(
            calibration_state,
            _CALIBRATION_KEYS,
            "Safety auxiliary checkpoint calibration metadata",
        )
        received_calibration = copy.deepcopy(dict(calibration_state))
        calibration_weight_names = tuple(
            received_calibration["translation_group_weights"]
        )
        if calibration_weight_names != SAFETY_AUX_TRANSLATION_GROUP_NAMES:
            raise ValueError(
                "Safety auxiliary checkpoint calibration translation group "
                "weights do not use canonical names and order"
            )
        if not _arrays_equal(
            received_calibration,
            self.calibration_metadata(),
        ):
            raise ValueError(
                "Safety auxiliary checkpoint calibration metadata does not "
                "match the requested Commit-4.6D configuration"
            )
        dataset_state = state["dataset"]
        if not isinstance(dataset_state, Mapping):
            raise TypeError(
                "Safety auxiliary checkpoint dataset identity must be a mapping"
            )
        _exact_keys(
            dataset_state,
            _DATASET_IDENTITY_KEYS,
            "Safety auxiliary checkpoint dataset identity",
        )
        expected_identity = self.dataset_identity()
        received_identity = copy.deepcopy(dict(dataset_state))
        if allow_dataset_path_override:
            received_identity.pop("resolved_path", None)
            expected_identity.pop("resolved_path", None)
        if not _arrays_equal(received_identity, expected_identity):
            raise ValueError(
                "Safety auxiliary checkpoint dataset fingerprint, schema, "
                "dimensions, boundaries, path, or fixed split does not match "
                "the loaded dataset"
            )

        auxiliary_updates = _exact_nonnegative_integer(
            state["auxiliary_update_count"],
            "Safety auxiliary auxiliary_update_count",
        )
        normal_updates = _exact_nonnegative_integer(
            state["last_normal_update_count"],
            "Safety auxiliary last_normal_update_count",
        )
        expected_auxiliary_updates = normal_updates // int(
            self.config["update_interval"]
        )
        if auxiliary_updates != expected_auxiliary_updates:
            raise ValueError(
                "Safety auxiliary update counter is inconsistent with the "
                "normal-update schedule"
            )
        if expected_normal_update_count is not None:
            expected_normal = _exact_nonnegative_integer(
                expected_normal_update_count, "expected_normal_update_count"
            )
            if normal_updates != expected_normal:
                raise ValueError(
                    "Safety auxiliary normal-update counter "
                    f"{normal_updates} does not match agent update_count "
                    f"{expected_normal}"
                )

        self.training_buffer.load_sampler_state_dict(state["sampler"])
        self.auxiliary_update_count = auxiliary_updates
        self.last_normal_update_count = normal_updates


__all__ = [
    "SAFETY_AUXILIARY_CHECKPOINT_KEY",
    "SAFETY_AUXILIARY_STATE_SCHEMA_VERSION",
    "SafetyAuxiliarySupervisor",
]
