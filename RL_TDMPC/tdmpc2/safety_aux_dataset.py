"""Strict offline dataset format for Safety auxiliary supervision.

The dataset owns one validated :class:`SafetyAuxReplayBuffer` snapshot and a
fixed train/validation split.  Split membership is deterministic and is never
resampled by the training or validation buffer accessors.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .safety_aux_replay import (
    SAFETY_AUX_REPLAY_SCHEMA_VERSION,
    SafetyAuxReplayBuffer,
)


SAFETY_AUX_DATASET_SCHEMA_VERSION = 1
SAFETY_AUX_DATASET_TYPE = "steve_safety_aux_dataset"
SAFETY_AUX_DATASET_FINGERPRINT_ALGORITHM = "sha256"
SAFETY_AUX_SPLIT_STRATEGY = (
    "joint_translation_block_reason_curvature_stratum"
)
SAFETY_AUX_SPLIT_ASSIGNMENT_ALGORITHM = "sha256_rank_v1"
MINIMUM_VALIDATION_JOINT_GROUP_SIZE = 5
DEFAULT_OBSERVATION_ROUND_DECIMALS = 6

_SEMANTIC_ARRAY_FIELDS = (
    "observation",
    "action",
    "applied_action",
    "safety_cost",
    "translation_block_reason_id",
    "curvature_stratum_id",
    "terminated",
    "truncated",
    "episode_step",
)
_EXACT_DUPLICATE_FIELDS = (
    "observation",
    "action",
    "safety_cost",
    "translation_block_reason_id",
    "curvature_stratum_id",
)
_OBSERVATION_ACTION_FIELDS = ("observation", "action")
_DATASET_KEYS = {
    "schema_version",
    "dataset_type",
    "replay_state",
    "split",
    "generation_seeds",
    "diagnostic_config",
    "fingerprint_algorithm",
    "fingerprint",
}
_SPLIT_KEYS = {
    "strategy",
    "assignment_algorithm",
    "seed",
    "validation_fraction",
    "minimum_validation_joint_group_size",
    "train_indices",
    "validation_indices",
}
_DIAGNOSTIC_CONFIG_KEYS = {"observation_round_decimals"}

ReplaySource = Union[SafetyAuxReplayBuffer, Mapping[str, Any]]


class SafetyAuxDataset:
    """A validated, immutable-by-interface offline Safety dataset."""

    def __init__(self, validated_state: Mapping[str, Any]) -> None:
        self._state = copy.deepcopy(dict(validated_state))

    @property
    def schema_version(self) -> int:
        return int(self._state["schema_version"])

    @property
    def fingerprint(self) -> str:
        return str(self._state["fingerprint"])

    @property
    def total_size(self) -> int:
        return int(self._state["replay_state"]["size"])

    @property
    def train_size(self) -> int:
        return int(self._state["split"]["train_indices"].size)

    @property
    def validation_size(self) -> int:
        return int(self._state["split"]["validation_indices"].size)

    @property
    def train_indices(self) -> np.ndarray:
        return self._state["split"]["train_indices"].copy()

    @property
    def validation_indices(self) -> np.ndarray:
        return self._state["split"]["validation_indices"].copy()

    def state_dict(self) -> Dict[str, Any]:
        """Return a defensive copy suitable for ``torch.save``."""

        return copy.deepcopy(self._state)

    def training_buffer(self) -> SafetyAuxReplayBuffer:
        """Return a replay containing training transitions and no validation data."""

        return _subset_buffer(
            self._state["replay_state"],
            self._state["split"]["train_indices"],
            seed=int(self._state["split"]["seed"]),
        )

    def validation_buffer(self) -> SafetyAuxReplayBuffer:
        """Return a replay containing fixed validation transitions only."""

        return _subset_buffer(
            self._state["replay_state"],
            self._state["split"]["validation_indices"],
            seed=int(self._state["split"]["seed"]) + 1,
        )

    def summary(self) -> Dict[str, Any]:
        """Build a finite/null-only JSON-compatible inspection summary."""

        replay = self._state["replay_state"]
        train_indices = self._state["split"]["train_indices"]
        validation_indices = self._state["split"]["validation_indices"]
        all_indices = np.arange(int(replay["size"]), dtype=np.int64)
        diagnostics = duplicate_diagnostics(
            replay,
            int(
                self._state["diagnostic_config"][
                    "observation_round_decimals"
                ]
            ),
        )
        return {
            "schema": {
                "dataset_type": self._state["dataset_type"],
                "dataset_schema_version": int(self._state["schema_version"]),
                "replay_schema_version": int(replay["schema_version"]),
                "fingerprint_algorithm": self._state[
                    "fingerprint_algorithm"
                ],
                "fingerprint": self._state["fingerprint"],
                "observation_dim": int(replay["observation_dim"]),
                "action_dim": int(replay["action_dim"]),
                "safety_cost_names": list(replay["safety_cost_names"]),
                "translation_block_reason_names": list(
                    replay["translation_block_reason_names"]
                ),
                "curvature_stratum_names": list(
                    replay["curvature_stratum_names"]
                ),
                "curvature_boundaries_mm_inv": [
                    float(value)
                    for value in replay["curvature_boundaries_mm_inv"]
                ],
            },
            "sizes": {
                "total": int(all_indices.size),
                "train": int(train_indices.size),
                "validation": int(validation_indices.size),
            },
            "split": {
                "strategy": self._state["split"]["strategy"],
                "assignment_algorithm": self._state["split"][
                    "assignment_algorithm"
                ],
                "seed": int(self._state["split"]["seed"]),
                "validation_fraction": float(
                    self._state["split"]["validation_fraction"]
                ),
                "minimum_validation_joint_group_size": int(
                    self._state["split"][
                        "minimum_validation_joint_group_size"
                    ]
                ),
                "integrity": _split_integrity_summary(
                    replay,
                    self._state["split"],
                ),
                "joint_groups": _joint_split_group_summary(
                    replay,
                    self._state["split"],
                ),
            },
            "generation_seeds": copy.deepcopy(
                self._state["generation_seeds"]
            ),
            "distributions": {
                "total": _distribution_summary(replay, all_indices),
                "train": _distribution_summary(replay, train_indices),
                "validation": _distribution_summary(
                    replay, validation_indices
                ),
            },
            "duplicate_diagnostics": diagnostics,
            "safety_cost_statistics": {
                "total": _safety_cost_statistics(replay, all_indices),
                "train": _safety_cost_statistics(replay, train_indices),
                "validation": _safety_cost_statistics(
                    replay, validation_indices
                ),
            },
        }


def build_dataset_state(
    buffer: ReplaySource,
    split_seed: int,
    validation_fraction: float,
    generation_seeds: Mapping[str, Any],
    observation_round_decimals: int = DEFAULT_OBSERVATION_ROUND_DECIMALS,
) -> Dict[str, Any]:
    """Build and validate a schema-v1 dataset state.

    Joint groups with at least five samples receive
    ``max(1, floor(group_size * validation_fraction))`` validation members,
    capped so that at least one training member remains.  Smaller groups remain
    train-only.  Membership is ranked with SHA256 over the split seed, joint
    labels, and source index, making it stable and independent of global RNGs.
    """

    replay_state = _validated_replay_state(buffer)
    if int(replay_state["size"]) == 0:
        raise ValueError("Safety auxiliary dataset cannot be empty")
    parsed_seed = _nonnegative_integer(split_seed, "split_seed")
    parsed_fraction = _validation_fraction(validation_fraction)
    normalized_generation_seeds = _normalize_generation_seeds(
        generation_seeds
    )
    parsed_round_decimals = _rounding_decimals(
        observation_round_decimals
    )
    train_indices, validation_indices = _joint_stratified_split(
        replay_state,
        split_seed=parsed_seed,
        validation_fraction=parsed_fraction,
        minimum_group_size=MINIMUM_VALIDATION_JOINT_GROUP_SIZE,
    )
    state: Dict[str, Any] = {
        "schema_version": SAFETY_AUX_DATASET_SCHEMA_VERSION,
        "dataset_type": SAFETY_AUX_DATASET_TYPE,
        "replay_state": replay_state,
        "split": {
            "strategy": SAFETY_AUX_SPLIT_STRATEGY,
            "assignment_algorithm": (
                SAFETY_AUX_SPLIT_ASSIGNMENT_ALGORITHM
            ),
            "seed": parsed_seed,
            "validation_fraction": parsed_fraction,
            "minimum_validation_joint_group_size": (
                MINIMUM_VALIDATION_JOINT_GROUP_SIZE
            ),
            "train_indices": train_indices,
            "validation_indices": validation_indices,
        },
        "generation_seeds": normalized_generation_seeds,
        "diagnostic_config": {
            "observation_round_decimals": parsed_round_decimals,
        },
        "fingerprint_algorithm": (
            SAFETY_AUX_DATASET_FINGERPRINT_ALGORITHM
        ),
        "fingerprint": "",
    }
    state["fingerprint"] = _dataset_fingerprint(state)
    return validate_dataset_state(state).state_dict()


def validate_dataset_state(
    state: Mapping[str, Any],
    *,
    expected_curvature_boundaries_mm_inv: Optional[
        Sequence[float]
    ] = None,
) -> SafetyAuxDataset:
    """Strictly validate a dataset mapping and return its safe interface."""

    if not isinstance(state, Mapping):
        raise TypeError("Safety auxiliary dataset state must be a mapping")
    _require_exact_keys(state, _DATASET_KEYS, "Dataset")
    schema_version = _nonnegative_integer(
        state["schema_version"], "dataset schema_version"
    )
    if schema_version != SAFETY_AUX_DATASET_SCHEMA_VERSION:
        raise ValueError(
            f"Dataset schema version {schema_version} does not match required "
            f"{SAFETY_AUX_DATASET_SCHEMA_VERSION}"
        )
    if state["dataset_type"] != SAFETY_AUX_DATASET_TYPE:
        raise ValueError(
            f"Dataset type {state['dataset_type']!r} does not match "
            f"{SAFETY_AUX_DATASET_TYPE!r}"
        )
    if (
        state["fingerprint_algorithm"]
        != SAFETY_AUX_DATASET_FINGERPRINT_ALGORITHM
    ):
        raise ValueError(
            "Unsupported Safety auxiliary dataset fingerprint algorithm "
            f"{state['fingerprint_algorithm']!r}"
        )

    replay_state = _validated_replay_state(state["replay_state"])
    if int(replay_state["size"]) == 0:
        raise ValueError("Safety auxiliary dataset cannot be empty")
    if expected_curvature_boundaries_mm_inv is not None:
        received = tuple(
            float(value)
            for value in replay_state["curvature_boundaries_mm_inv"]
        )
        expected = tuple(
            float(value)
            for value in expected_curvature_boundaries_mm_inv
        )
        if received != expected:
            raise ValueError(
                "Dataset curvature-boundary mismatch: "
                f"{received} != {expected}"
            )

    split = _validated_split(state["split"], replay_state)
    generation_seeds = _normalize_generation_seeds(
        state["generation_seeds"]
    )
    diagnostic_config = _validated_diagnostic_config(
        state["diagnostic_config"]
    )
    fingerprint = state["fingerprint"]
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise ValueError("Dataset fingerprint must be a lowercase SHA256 hex digest")

    validated: Dict[str, Any] = {
        "schema_version": schema_version,
        "dataset_type": SAFETY_AUX_DATASET_TYPE,
        "replay_state": replay_state,
        "split": split,
        "generation_seeds": generation_seeds,
        "diagnostic_config": diagnostic_config,
        "fingerprint_algorithm": (
            SAFETY_AUX_DATASET_FINGERPRINT_ALGORITHM
        ),
        "fingerprint": fingerprint,
    }
    calculated = _dataset_fingerprint(validated)
    if fingerprint != calculated:
        raise ValueError(
            "Safety auxiliary dataset fingerprint mismatch: "
            f"stored={fingerprint}, calculated={calculated}"
        )
    return SafetyAuxDataset(validated)


def load_dataset(
    path: Union[str, Path],
    *,
    expected_curvature_boundaries_mm_inv: Optional[
        Sequence[float]
    ] = None,
) -> SafetyAuxDataset:
    """Load a trusted local torch payload and strictly validate it."""

    resolved = Path(path).expanduser().resolve()
    state = torch.load(resolved, map_location="cpu", weights_only=False)
    return validate_dataset_state(
        state,
        expected_curvature_boundaries_mm_inv=(
            expected_curvature_boundaries_mm_inv
        ),
    )


def duplicate_diagnostics(
    buffer_or_state: ReplaySource,
    rounding_decimals: int = DEFAULT_OBSERVATION_ROUND_DECIMALS,
) -> Dict[str, Any]:
    """Report duplicates without deleting them.

    Exact composites follow the task definition: observation, requested
    action, safety cost, translation reason, and curvature stratum.  Repeated
    observation-action pairs use observation and requested action.  Near
    duplicate observations are counted after decimal rounding; with the
    default six decimals the implied component-wise tolerance is approximately
    half of ``10**-6`` away from decimal tie cases.
    """

    state = _validated_replay_state(buffer_or_state)
    decimals = _rounding_decimals(rounding_decimals)
    exact = _multiplicity_diagnostics(
        _row_keys(state, _EXACT_DUPLICATE_FIELDS)
    )
    observation_action = _multiplicity_diagnostics(
        _row_keys(state, _OBSERVATION_ACTION_FIELDS)
    )
    rounded_observations = np.round(
        state["observation"].astype(np.float64),
        decimals=decimals,
    )
    rounded = _multiplicity_diagnostics(
        _row_keys_from_arrays((rounded_observations,))
    )
    return {
        "sample_count": int(state["size"]),
        "exact_duplicate_count": int(exact["duplicate_sample_count"]),
        "exact_duplicate_group_count": int(exact["duplicate_group_count"]),
        "exact_unique_composite_count": int(exact["unique_count"]),
        "exact_max_multiplicity": int(exact["max_multiplicity"]),
        "repeated_observation_action_count": int(
            observation_action["duplicate_sample_count"]
        ),
        "repeated_observation_action_group_count": int(
            observation_action["duplicate_group_count"]
        ),
        "unique_observation_action_count": int(
            observation_action["unique_count"]
        ),
        "observation_action_max_multiplicity": int(
            observation_action["max_multiplicity"]
        ),
        "observation_round_decimals": decimals,
        "observation_rounding_description": (
            f"Each float observation component is rounded to {decimals} "
            "decimal places before exact row comparison."
        ),
        "unique_observation_count": int(rounded["unique_count"]),
        "rounded_observation_duplicate_count": int(
            rounded["duplicate_sample_count"]
        ),
        "rounded_observation_duplicate_group_count": int(
            rounded["duplicate_group_count"]
        ),
    }


def exact_unique_indices(buffer_or_state: ReplaySource) -> np.ndarray:
    """Return first-representative indices for exact composite deduplication."""

    state = _validated_replay_state(buffer_or_state)
    keys = _row_keys(state, _EXACT_DUPLICATE_FIELDS)
    seen = set()
    selected = []
    for index, key in enumerate(keys):
        if key not in seen:
            seen.add(key)
            selected.append(index)
    return np.asarray(selected, dtype=np.int64)


def subset_replay_state(
    buffer_or_state: ReplaySource,
    indices: np.ndarray,
    *,
    seed: int = 0,
) -> Dict[str, Any]:
    """Return a strict compact replay state containing exactly ``indices``."""

    replay_state = _validated_replay_state(buffer_or_state)
    parsed_indices = _validated_subset_indices(
        indices, int(replay_state["size"])
    )
    return _subset_buffer(
        replay_state,
        parsed_indices,
        seed=_nonnegative_integer(seed, "subset seed"),
    ).state_dict()


def _validated_replay_state(source: ReplaySource) -> Dict[str, Any]:
    if isinstance(source, SafetyAuxReplayBuffer):
        candidate = source.state_dict()
    elif isinstance(source, Mapping):
        candidate = copy.deepcopy(dict(source))
    else:
        raise TypeError(
            "Expected a SafetyAuxReplayBuffer or replay-state mapping"
        )
    for key in (
        "capacity",
        "observation_dim",
        "action_dim",
        "safety_cost_names",
        "curvature_boundaries_mm_inv",
    ):
        if key not in candidate:
            raise ValueError(f"Safety auxiliary replay state is missing {key!r}")
    capacity = _positive_integer(candidate["capacity"], "replay capacity")
    observation_dim = _positive_integer(
        candidate["observation_dim"], "replay observation_dim"
    )
    action_dim = _positive_integer(
        candidate["action_dim"], "replay action_dim"
    )
    validator = SafetyAuxReplayBuffer(
        capacity,
        observation_dim,
        action_dim,
        safety_cost_names=candidate["safety_cost_names"],
        curvature_boundaries_mm_inv=candidate[
            "curvature_boundaries_mm_inv"
        ],
        seed=0,
    )
    validator.load_state_dict(candidate)
    validated = validator.state_dict()
    for field in _SEMANTIC_ARRAY_FIELDS:
        if field not in validated:
            raise ValueError(
                f"Replay schema {SAFETY_AUX_REPLAY_SCHEMA_VERSION} is missing "
                f"semantic field {field!r}"
            )
    return validated


def _validated_split(
    value: Any,
    replay_state: Mapping[str, Any],
) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("Dataset split must be a mapping")
    _require_exact_keys(value, _SPLIT_KEYS, "Dataset split")
    if value["strategy"] != SAFETY_AUX_SPLIT_STRATEGY:
        raise ValueError(
            f"Dataset split strategy must be {SAFETY_AUX_SPLIT_STRATEGY!r}"
        )
    if (
        value["assignment_algorithm"]
        != SAFETY_AUX_SPLIT_ASSIGNMENT_ALGORITHM
    ):
        raise ValueError(
            "Dataset split assignment algorithm must be "
            f"{SAFETY_AUX_SPLIT_ASSIGNMENT_ALGORITHM!r}"
        )
    seed = _nonnegative_integer(value["seed"], "dataset split seed")
    fraction = _validation_fraction(value["validation_fraction"])
    minimum_group_size = _positive_integer(
        value["minimum_validation_joint_group_size"],
        "minimum_validation_joint_group_size",
    )
    if minimum_group_size != MINIMUM_VALIDATION_JOINT_GROUP_SIZE:
        raise ValueError(
            "minimum_validation_joint_group_size must be "
            f"{MINIMUM_VALIDATION_JOINT_GROUP_SIZE}"
        )
    size = int(replay_state["size"])
    train_indices = _validated_split_indices(
        value["train_indices"], size, "train_indices"
    )
    validation_indices = _validated_split_indices(
        value["validation_indices"], size, "validation_indices"
    )
    if np.intersect1d(train_indices, validation_indices).size:
        raise ValueError("Train and validation indices overlap")
    combined = np.sort(np.concatenate((train_indices, validation_indices)))
    expected_union = np.arange(size, dtype=np.int64)
    if not np.array_equal(combined, expected_union):
        raise ValueError(
            "Train/validation index union does not equal the complete dataset"
        )
    expected_train, expected_validation = _joint_stratified_split(
        replay_state,
        split_seed=seed,
        validation_fraction=fraction,
        minimum_group_size=minimum_group_size,
    )
    if not np.array_equal(train_indices, expected_train):
        raise ValueError(
            "Saved train indices do not match the deterministic joint split"
        )
    if not np.array_equal(validation_indices, expected_validation):
        raise ValueError(
            "Saved validation indices do not match the deterministic joint split"
        )
    return {
        "strategy": SAFETY_AUX_SPLIT_STRATEGY,
        "assignment_algorithm": SAFETY_AUX_SPLIT_ASSIGNMENT_ALGORITHM,
        "seed": seed,
        "validation_fraction": fraction,
        "minimum_validation_joint_group_size": minimum_group_size,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
    }


def _joint_stratified_split(
    replay_state: Mapping[str, Any],
    *,
    split_seed: int,
    validation_fraction: float,
    minimum_group_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    reasons = replay_state["translation_block_reason_id"]
    strata = replay_state["curvature_stratum_id"]
    train = []
    validation = []
    groups = sorted(
        {
            (int(reason), int(stratum))
            for reason, stratum in zip(reasons, strata)
        }
    )
    for reason_id, stratum_id in groups:
        members = np.flatnonzero(
            (reasons == reason_id) & (strata == stratum_id)
        ).astype(np.int64)
        if members.size < minimum_group_size:
            train.extend(int(index) for index in members)
            continue
        validation_count = max(
            1, int(np.floor(members.size * validation_fraction))
        )
        validation_count = min(validation_count, int(members.size) - 1)
        ranked = sorted(
            (int(index) for index in members),
            key=lambda index: _split_rank(
                split_seed, reason_id, stratum_id, index
            ),
        )
        validation.extend(ranked[:validation_count])
        train.extend(ranked[validation_count:])
    return (
        np.asarray(sorted(train), dtype=np.int64),
        np.asarray(sorted(validation), dtype=np.int64),
    )


def _split_rank(
    seed: int,
    reason_id: int,
    stratum_id: int,
    index: int,
) -> bytes:
    payload = (
        f"{seed}:{reason_id}:{stratum_id}:{index}".encode("ascii")
    )
    return hashlib.sha256(payload).digest()


def _dataset_fingerprint(state: Mapping[str, Any]) -> str:
    replay = state["replay_state"]
    split = state["split"]
    descriptor = {
        "dataset_type": state["dataset_type"],
        "dataset_schema_version": int(state["schema_version"]),
        "fingerprint_algorithm": state["fingerprint_algorithm"],
        "replay_schema": {
            "schema_version": int(replay["schema_version"]),
            "capacity": int(replay["capacity"]),
            "size": int(replay["size"]),
            "next_index": int(replay["next_index"]),
            "total_added": int(replay["total_added"]),
            "observation_dim": int(replay["observation_dim"]),
            "action_dim": int(replay["action_dim"]),
            "safety_cost_dim": int(replay["safety_cost_dim"]),
            "safety_cost_names": list(replay["safety_cost_names"]),
            "translation_block_reason_names": list(
                replay["translation_block_reason_names"]
            ),
            "curvature_stratum_names": list(
                replay["curvature_stratum_names"]
            ),
            "curvature_boundaries_mm_inv": [
                float(value)
                for value in replay["curvature_boundaries_mm_inv"]
            ],
        },
        "split": {
            "strategy": split["strategy"],
            "assignment_algorithm": split["assignment_algorithm"],
            "seed": int(split["seed"]),
            "validation_fraction": float(split["validation_fraction"]),
            "minimum_validation_joint_group_size": int(
                split["minimum_validation_joint_group_size"]
            ),
        },
        "generation_seeds": state["generation_seeds"],
        "diagnostic_config": state["diagnostic_config"],
    }
    digest = hashlib.sha256()
    digest.update(_canonical_json_bytes(descriptor))
    for field in _SEMANTIC_ARRAY_FIELDS:
        _update_hash_with_array(digest, field, replay[field])
    _update_hash_with_array(
        digest, "split.train_indices", split["train_indices"]
    )
    _update_hash_with_array(
        digest, "split.validation_indices", split["validation_indices"]
    )
    return digest.hexdigest()


def _update_hash_with_array(
    digest: "hashlib._Hash",
    name: str,
    value: np.ndarray,
) -> None:
    array = np.asarray(value)
    if array.dtype.itemsize > 1 and array.dtype.byteorder in ("=", ">"):
        canonical_dtype = array.dtype.newbyteorder("<")
        array = array.astype(canonical_dtype, copy=False)
    contiguous = np.ascontiguousarray(array)
    header = {
        "name": name,
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
    }
    digest.update(_canonical_json_bytes(header))
    digest.update(contiguous.tobytes(order="C"))


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _subset_buffer(
    replay_state: Mapping[str, Any],
    indices: np.ndarray,
    *,
    seed: int,
) -> SafetyAuxReplayBuffer:
    capacity = max(1, int(indices.size))
    output = SafetyAuxReplayBuffer(
        capacity,
        int(replay_state["observation_dim"]),
        int(replay_state["action_dim"]),
        safety_cost_names=replay_state["safety_cost_names"],
        curvature_boundaries_mm_inv=replay_state[
            "curvature_boundaries_mm_inv"
        ],
        seed=seed,
    )
    for source_index in indices:
        index = int(source_index)
        output.add(
            replay_state["observation"][index].copy(),
            replay_state["action"][index].copy(),
            replay_state["safety_cost"][index].copy(),
            int(replay_state["translation_block_reason_id"][index]),
            int(replay_state["curvature_stratum_id"][index]),
            applied_action=replay_state["applied_action"][index].copy(),
            terminated=bool(replay_state["terminated"][index]),
            truncated=bool(replay_state["truncated"][index]),
            episode_step=int(replay_state["episode_step"][index]),
        )
    return output


def _distribution_summary(
    replay_state: Mapping[str, Any],
    indices: np.ndarray,
) -> Dict[str, Any]:
    reason_names = tuple(replay_state["translation_block_reason_names"])
    stratum_names = tuple(replay_state["curvature_stratum_names"])
    reasons = replay_state["translation_block_reason_id"][indices]
    strata = replay_state["curvature_stratum_id"][indices]
    count = int(indices.size)
    reason_counts = np.bincount(reasons, minlength=len(reason_names))
    stratum_counts = np.bincount(strata, minlength=len(stratum_names))
    joint = np.zeros(
        (len(reason_names), len(stratum_names)), dtype=np.int64
    )
    if count:
        np.add.at(joint, (reasons, strata), 1)
    return {
        "sample_count": count,
        "translation_block_reasons": {
            name: {
                "id": reason_id,
                "count": int(reason_counts[reason_id]),
                "fraction": (
                    float(reason_counts[reason_id] / count)
                    if count
                    else None
                ),
            }
            for reason_id, name in enumerate(reason_names)
        },
        "curvature_strata": {
            name: {
                "id": stratum_id,
                "count": int(stratum_counts[stratum_id]),
                "fraction": (
                    float(stratum_counts[stratum_id] / count)
                    if count
                    else None
                ),
            }
            for stratum_id, name in enumerate(stratum_names)
        },
        "joint_reason_curvature": {
            reason_name: {
                stratum_name: {
                    "count": int(joint[reason_id, stratum_id]),
                    "fraction": (
                        float(joint[reason_id, stratum_id] / count)
                        if count
                        else None
                    ),
                }
                for stratum_id, stratum_name in enumerate(stratum_names)
            }
            for reason_id, reason_name in enumerate(reason_names)
        },
    }


def _safety_cost_statistics(
    replay_state: Mapping[str, Any],
    indices: np.ndarray,
) -> Dict[str, Any]:
    values = replay_state["safety_cost"][indices].astype(
        np.float64, copy=False
    )
    output: Dict[str, Any] = {}
    for channel_id, name in enumerate(replay_state["safety_cost_names"]):
        channel = values[:, channel_id]
        if channel.size == 0:
            output[name] = {
                "count": 0,
                "min": None,
                "max": None,
                "mean": None,
                "standard_deviation": None,
                "median": None,
                "p95": None,
                "fraction_equal_zero": None,
                "fraction_greater_zero": None,
            }
        else:
            output[name] = {
                "count": int(channel.size),
                "min": float(np.min(channel)),
                "max": float(np.max(channel)),
                "mean": float(np.mean(channel)),
                "standard_deviation": float(np.std(channel)),
                "median": float(np.median(channel)),
                "p95": float(np.percentile(channel, 95)),
                "fraction_equal_zero": float(np.mean(channel == 0.0)),
                "fraction_greater_zero": float(np.mean(channel > 0.0)),
            }
    return output


def _split_integrity_summary(
    replay_state: Mapping[str, Any],
    split: Mapping[str, Any],
) -> Dict[str, Any]:
    train = split["train_indices"]
    validation = split["validation_indices"]
    size = int(replay_state["size"])
    expected_train, expected_validation = _joint_stratified_split(
        replay_state,
        split_seed=int(split["seed"]),
        validation_fraction=float(split["validation_fraction"]),
        minimum_group_size=int(
            split["minimum_validation_joint_group_size"]
        ),
    )
    disjoint = np.intersect1d(train, validation).size == 0
    in_range = bool(
        np.all((train >= 0) & (train < size))
        and np.all((validation >= 0) & (validation < size))
    )
    union_complete = np.array_equal(
        np.sort(np.concatenate((train, validation))),
        np.arange(size, dtype=np.int64),
    )
    deterministic_match = bool(
        np.array_equal(train, expected_train)
        and np.array_equal(validation, expected_validation)
    )
    return {
        "passed": bool(
            disjoint and in_range and union_complete and deterministic_match
        ),
        "train_validation_disjoint": bool(disjoint),
        "all_indices_in_range": in_range,
        "union_equals_complete_dataset": bool(union_complete),
        "deterministic_split_matches": deterministic_match,
    }


def _joint_split_group_summary(
    replay_state: Mapping[str, Any],
    split: Mapping[str, Any],
) -> list:
    reasons = replay_state["translation_block_reason_id"]
    strata = replay_state["curvature_stratum_id"]
    train_membership = np.zeros(int(replay_state["size"]), dtype=np.bool_)
    validation_membership = np.zeros(
        int(replay_state["size"]), dtype=np.bool_
    )
    train_membership[split["train_indices"]] = True
    validation_membership[split["validation_indices"]] = True
    reason_names = replay_state["translation_block_reason_names"]
    stratum_names = replay_state["curvature_stratum_names"]
    minimum_size = int(split["minimum_validation_joint_group_size"])
    output = []
    for reason_id, stratum_id in sorted(
        {
            (int(reason), int(stratum))
            for reason, stratum in zip(reasons, strata)
        }
    ):
        members = (reasons == reason_id) & (strata == stratum_id)
        sample_count = int(np.count_nonzero(members))
        validation_count = int(
            np.count_nonzero(members & validation_membership)
        )
        output.append(
            {
                "translation_block_reason_id": reason_id,
                "translation_block_reason": reason_names[reason_id],
                "curvature_stratum_id": stratum_id,
                "curvature_stratum": stratum_names[stratum_id],
                "sample_count": sample_count,
                "train_count": int(
                    np.count_nonzero(members & train_membership)
                ),
                "validation_count": validation_count,
                "small_group_train_only": bool(
                    sample_count < minimum_size and validation_count == 0
                ),
            }
        )
    return output


def _row_keys(
    state: Mapping[str, Any],
    fields: Sequence[str],
) -> Tuple[bytes, ...]:
    return _row_keys_from_arrays(tuple(state[field] for field in fields))


def _row_keys_from_arrays(
    arrays: Sequence[np.ndarray],
) -> Tuple[bytes, ...]:
    if not arrays:
        return ()
    count = int(arrays[0].shape[0])
    if any(int(array.shape[0]) != count for array in arrays):
        raise ValueError("Duplicate-diagnostic arrays have inconsistent lengths")
    keys = []
    for index in range(count):
        parts = []
        for array in arrays:
            row = np.ascontiguousarray(array[index])
            parts.append(row.tobytes(order="C"))
        keys.append(b"".join(parts))
    return tuple(keys)


def _multiplicity_diagnostics(keys: Sequence[bytes]) -> Dict[str, int]:
    counts: Dict[bytes, int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    multiplicities = tuple(counts.values())
    return {
        "unique_count": len(counts),
        "duplicate_sample_count": sum(
            count - 1 for count in multiplicities if count > 1
        ),
        "duplicate_group_count": sum(
            1 for count in multiplicities if count > 1
        ),
        "max_multiplicity": max(multiplicities, default=0),
    }


def _validated_subset_indices(
    value: Any,
    size: int,
) -> np.ndarray:
    indices = _validated_split_indices(value, size, "subset indices")
    return indices


def _validated_split_indices(
    value: Any,
    size: int,
    label: str,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{label} must be a numpy.ndarray")
    if value.ndim != 1 or value.dtype != np.int64:
        raise TypeError(f"{label} must be a one-dimensional int64 array")
    indices = value.copy()
    if indices.size and (
        np.any(indices < 0) or np.any(indices >= int(size))
    ):
        raise ValueError(f"{label} contains out-of-range values")
    if np.unique(indices).size != indices.size:
        raise ValueError(f"{label} contains duplicate indices")
    if indices.size > 1 and np.any(indices[:-1] >= indices[1:]):
        raise ValueError(f"{label} must be strictly increasing")
    return indices


def _validated_diagnostic_config(value: Any) -> Dict[str, int]:
    if not isinstance(value, Mapping):
        raise TypeError("Dataset diagnostic_config must be a mapping")
    _require_exact_keys(
        value, _DIAGNOSTIC_CONFIG_KEYS, "Dataset diagnostic_config"
    )
    return {
        "observation_round_decimals": _rounding_decimals(
            value["observation_round_decimals"]
        )
    }


def _normalize_generation_seeds(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("generation_seeds must be a mapping")
    if not value:
        raise ValueError("generation_seeds must not be empty")
    normalized = _normalize_json_value(value, "generation_seeds")
    assert isinstance(normalized, dict)
    return normalized


def _normalize_json_value(value: Any, label: str) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    ):
        return int(value)
    if isinstance(value, (float, np.floating)):
        parsed = float(value)
        if not np.isfinite(parsed):
            raise ValueError(f"{label} contains NaN or infinity")
        return parsed
    if isinstance(value, Mapping):
        keys = tuple(value.keys())
        if any(not isinstance(key, str) for key in keys):
            raise TypeError(f"{label} mapping keys must be strings")
        output: Dict[str, Any] = {}
        for key in sorted(keys):
            output[key] = _normalize_json_value(
                value[key], f"{label}.{key}"
            )
        return output
    if isinstance(value, (list, tuple)):
        return [
            _normalize_json_value(item, f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, np.ndarray):
        return _normalize_json_value(value.tolist(), label)
    raise TypeError(
        f"{label} contains unsupported JSON value {type(value).__name__}"
    )


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set,
    label: str,
) -> None:
    missing = sorted(expected - value.keys())
    unexpected = sorted(value.keys() - expected)
    if missing or unexpected:
        raise ValueError(
            f"{label} keys mismatch; missing={missing}, "
            f"unexpected={unexpected}"
        )


def _validation_fraction(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("validation_fraction must be a real number, not bool")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("validation_fraction must be a real number") from exc
    if not np.isfinite(parsed) or not 0.0 < parsed < 1.0:
        raise ValueError("validation_fraction must be finite and in (0, 1)")
    return parsed


def _rounding_decimals(value: Any) -> int:
    parsed = _nonnegative_integer(
        value, "observation_round_decimals"
    )
    if parsed > 15:
        raise ValueError("observation_round_decimals must be at most 15")
    return parsed


def _positive_integer(value: Any, label: str) -> int:
    parsed = _nonnegative_integer(value, label)
    if parsed == 0:
        raise ValueError(f"{label} must be positive")
    return parsed


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{label} must be an integer")
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{label} must be nonnegative")
    return parsed


__all__ = [
    "DEFAULT_OBSERVATION_ROUND_DECIMALS",
    "MINIMUM_VALIDATION_JOINT_GROUP_SIZE",
    "SAFETY_AUX_DATASET_SCHEMA_VERSION",
    "SafetyAuxDataset",
    "build_dataset_state",
    "duplicate_diagnostics",
    "exact_unique_indices",
    "load_dataset",
    "subset_replay_state",
    "validate_dataset_state",
]
