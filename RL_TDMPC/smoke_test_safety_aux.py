"""Focused, simulation-free checks for Safety auxiliary data infrastructure."""

from __future__ import annotations

import copy
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from collect_safety_dataset import (
    _planned_storage_upper_bound,
    _resolved_output_paths,
    parse_args,
)
from envs.safety import (
    CURVATURE_STRATUM_NAMES,
    SAFETY_COST_NAMES,
    curvature_stratum_id,
)
from eve.intervention import TRANSLATION_BLOCK_REASON_NAMES
from tdmpc2.common import (
    build_safety_aux_config,
    curvature_boundaries_from_diagnostics,
    load_config,
)
from tdmpc2.replay_buffer import EpisodeReplayBuffer
from tdmpc2.safety_aux_dataset import (
    SAFETY_AUX_DATASET_SCHEMA_VERSION,
    build_dataset_state,
    duplicate_diagnostics,
    exact_unique_indices,
    load_dataset,
    subset_replay_state,
    validate_dataset_state,
)
from tdmpc2.safety_aux_replay import (
    SAFETY_AUX_REPLAY_SCHEMA_VERSION,
    SafetyAuxReplayBuffer,
)
from train import DEFAULT_CONFIG, build_agent_config


def _new_buffer(*, seed: int = 7, capacity: int = 32) -> SafetyAuxReplayBuffer:
    return SafetyAuxReplayBuffer(
        capacity,
        14,
        2,
        safety_cost_names=SAFETY_COST_NAMES,
        curvature_boundaries_mm_inv=(0.05, 0.10, 0.25),
        seed=seed,
    )


def _add_fixture(buffer: SafetyAuxReplayBuffer, count: int = 12) -> None:
    reasons = (0, 1, 3)
    curvatures = (0.01, 0.075, 0.15, 0.30)
    for index in range(count):
        observation = np.full(14, index / 100.0, dtype=np.float32)
        action = np.asarray(
            [(-1.0) ** index * 0.5, index / max(1, count)],
            dtype=np.float32,
        )
        curvature = curvatures[index % len(curvatures)]
        cost = np.asarray([curvature, index / 10.0], dtype=np.float32)
        buffer.add(
            observation,
            action,
            cost,
            reasons[index % len(reasons)],
            curvature_stratum_id(curvature, (0.05, 0.10, 0.25)),
            applied_action=action.copy(),
            terminated=index == count - 1,
            truncated=False,
            episode_step=index + 1,
        )
        observation.fill(-99.0)
        action.fill(-99.0)
        cost.fill(-99.0)


def _add_group_fixture(
    buffer: SafetyAuxReplayBuffer,
    *,
    reason_id: int,
    stratum_id: int,
    count: int,
    start_index: int,
) -> None:
    """Add uniquely identifiable samples from one joint split group."""

    curvatures = (0.01, 0.075, 0.15, 0.30)
    for offset in range(count):
        sample_index = start_index + offset
        observation = np.full(
            14,
            (sample_index + 1) / 1000.0,
            dtype=np.float32,
        )
        action = np.asarray(
            [
                ((sample_index % 5) - 2) / 2.0,
                ((sample_index % 7) - 3) / 3.0,
            ],
            dtype=np.float32,
        )
        applied_action = action * np.float32(0.5)
        safety_cost = np.asarray(
            [curvatures[stratum_id], sample_index / 100.0],
            dtype=np.float32,
        )
        buffer.add(
            observation,
            action,
            safety_cost,
            reason_id,
            stratum_id,
            applied_action=applied_action,
            episode_step=sample_index + 1,
        )


def _split_fixture() -> SafetyAuxReplayBuffer:
    """Build groups of sizes 10, 5, 4, and 1 for split edge cases."""

    buffer = _new_buffer(seed=19, capacity=24)
    next_index = 0
    for reason_id, stratum_id, count in (
        (0, 0, 10),
        (1, 1, 5),
        (3, 2, 4),
        (0, 3, 1),
    ):
        _add_group_fixture(
            buffer,
            reason_id=reason_id,
            stratum_id=stratum_id,
            count=count,
            start_index=next_index,
        )
        next_index += count
    return buffer


def _duplicate_fixture() -> SafetyAuxReplayBuffer:
    """Build exact, observation/action, near-observation, and unique rows."""

    buffer = _new_buffer(seed=23, capacity=8)
    observation = np.zeros(14, dtype=np.float32)
    action = np.asarray([0.25, -0.5], dtype=np.float32)
    base_cost = np.asarray([0.01, 0.2], dtype=np.float32)

    rows = (
        # The second row is an exact semantic duplicate even though the
        # applied action differs; the task's exact key uses requested action.
        (observation, action, base_cost, 0, 0, action),
        (
            observation,
            action,
            base_cost,
            0,
            0,
            np.asarray([0.0, -0.5], dtype=np.float32),
        ),
        (
            observation,
            action,
            np.asarray([0.01, 0.3], dtype=np.float32),
            0,
            0,
            action,
        ),
        (
            np.asarray(
                [4.0e-7] + [0.0] * 13,
                dtype=np.float32,
            ),
            action,
            np.asarray([0.01, 0.4], dtype=np.float32),
            0,
            0,
            action,
        ),
        (
            np.ones(14, dtype=np.float32),
            np.asarray([-0.5, 0.5], dtype=np.float32),
            np.asarray([0.30, 0.5], dtype=np.float32),
            3,
            3,
            np.asarray([-0.25, 0.25], dtype=np.float32),
        ),
    )
    for index, (
        row_observation,
        row_action,
        row_cost,
        reason_id,
        stratum_id,
        applied_action,
    ) in enumerate(rows):
        buffer.add(
            row_observation.copy(),
            row_action.copy(),
            row_cost.copy(),
            reason_id,
            stratum_id,
            applied_action=applied_action.copy(),
            episode_step=index + 1,
        )
    return buffer


def _assert_batch_equal(left, right) -> None:
    left_batch, left_metadata = left
    right_batch, right_metadata = right
    assert left_metadata == right_metadata
    assert left_batch.keys() == right_batch.keys()
    for key in left_batch:
        np.testing.assert_array_equal(left_batch[key], right_batch[key])


def _assert_raises(callable_object, expected_exception, message: str) -> None:
    try:
        callable_object()
    except expected_exception as exc:
        assert message in str(exc)
    else:
        expected_name = (
            "/".join(item.__name__ for item in expected_exception)
            if isinstance(expected_exception, tuple)
            else expected_exception.__name__
        )
        raise AssertionError(
            f"Expected {expected_name} containing {message!r}"
        )


def main() -> None:
    config = load_config(DEFAULT_CONFIG)
    boundaries = curvature_boundaries_from_diagnostics(config)
    assert boundaries == (0.05, 0.1, 0.25)
    aux_config = build_safety_aux_config(config)
    assert aux_config == {
        "enabled": False,
        "capacity": 100000,
        "translation_fraction": 0.5,
        "curvature_fraction": 0.5,
    }
    legacy_config = copy.deepcopy(config)
    legacy_config.pop("safety_aux")
    legacy_config["diagnostics"].pop("curvature_high_max_mm_inv")
    assert build_safety_aux_config(legacy_config) == aux_config
    assert curvature_boundaries_from_diagnostics(legacy_config) == boundaries
    assert build_agent_config(config) == build_agent_config(legacy_config)
    legacy_wide_diagnostics = copy.deepcopy(legacy_config)
    legacy_wide_diagnostics["diagnostics"][
        "curvature_medium_max_mm_inv"
    ] = 0.30
    assert (
        build_safety_aux_config(legacy_wide_diagnostics)["enabled"]
        is False
    )
    assert (
        build_agent_config(legacy_wide_diagnostics)[
            "curvature_medium_max_mm_inv"
        ]
        == 0.30
    )
    enabled_legacy_wide = copy.deepcopy(legacy_wide_diagnostics)
    enabled_legacy_wide["safety_aux"] = copy.deepcopy(aux_config)
    enabled_legacy_wide["safety_aux"]["enabled"] = True
    _assert_raises(
        lambda: build_safety_aux_config(enabled_legacy_wide),
        ValueError,
        "strictly increasing",
    )

    invalid_config = copy.deepcopy(config)
    invalid_config["safety_aux"]["capacity"] = 0
    _assert_raises(
        lambda: build_safety_aux_config(invalid_config),
        ValueError,
        "capacity",
    )
    invalid_config = copy.deepcopy(config)
    invalid_config["safety_aux"]["translation_fraction"] = -0.1
    _assert_raises(
        lambda: build_safety_aux_config(invalid_config),
        ValueError,
        "nonnegative",
    )
    invalid_config = copy.deepcopy(config)
    invalid_config["diagnostics"]["curvature_high_max_mm_inv"] = 0.09
    _assert_raises(
        lambda: curvature_boundaries_from_diagnostics(invalid_config),
        ValueError,
        "strictly increasing",
    )

    buffer = _new_buffer()
    _add_fixture(buffer)
    assert len(buffer) == 12
    state = buffer.state_dict()
    assert state["schema_version"] == SAFETY_AUX_REPLAY_SCHEMA_VERSION
    assert state["safety_cost_names"] == SAFETY_COST_NAMES
    assert (
        state["translation_block_reason_names"]
        == TRANSLATION_BLOCK_REASON_NAMES
    )
    assert state["curvature_stratum_names"] == CURVATURE_STRATUM_NAMES
    assert state["observation"].shape == (12, 14)
    assert state["action"].shape == (12, 2)
    assert state["applied_action"].shape == (12, 2)
    assert state["safety_cost"].shape == (12, 2)
    assert state["observation"].dtype == np.float32
    assert state["action"].dtype == np.float32
    assert state["applied_action"].dtype == np.float32
    assert state["safety_cost"].dtype == np.float32
    assert np.all(state["observation"] > -1.0)
    assert np.all(np.isfinite(state["applied_action"]))
    np.testing.assert_array_equal(state["action"], state["applied_action"])

    valid_observation = np.zeros(14, dtype=np.float32)
    valid_action = np.zeros(2, dtype=np.float32)
    valid_cost = np.zeros(2, dtype=np.float32)
    invalid_insertions = (
        (
            lambda: buffer.add(
                np.zeros(13, dtype=np.float32),
                valid_action,
                valid_cost,
                0,
                0,
                applied_action=valid_action,
            ),
            ValueError,
            "shape",
        ),
        (
            lambda: buffer.add(
                valid_observation.astype(np.float64),
                valid_action,
                valid_cost,
                0,
                0,
                applied_action=valid_action,
            ),
            TypeError,
            "float32",
        ),
        (
            lambda: buffer.add(
                valid_observation,
                np.zeros(3, dtype=np.float32),
                valid_cost,
                0,
                0,
                applied_action=valid_action,
            ),
            ValueError,
            "shape",
        ),
        (
            lambda: buffer.add(
                valid_observation,
                valid_action.astype(np.float64),
                valid_cost,
                0,
                0,
                applied_action=valid_action,
            ),
            TypeError,
            "float32",
        ),
        (
            lambda: buffer.add(
                valid_observation,
                np.asarray([np.nan, 0.0], dtype=np.float32),
                valid_cost,
                0,
                0,
                applied_action=valid_action,
            ),
            FloatingPointError,
            "NaN or infinity",
        ),
        (
            lambda: buffer.add(
                valid_observation,
                valid_action,
                np.zeros(3, dtype=np.float32),
                0,
                0,
                applied_action=valid_action,
            ),
            ValueError,
            "shape",
        ),
        (
            lambda: buffer.add(
                valid_observation,
                valid_action,
                valid_cost.astype(np.float64),
                0,
                0,
                applied_action=valid_action,
            ),
            TypeError,
            "float32",
        ),
        (
            lambda: buffer.add(
                valid_observation,
                valid_action,
                np.asarray([np.inf, 0.0], dtype=np.float32),
                0,
                0,
                applied_action=valid_action,
            ),
            FloatingPointError,
            "NaN or infinity",
        ),
        (
            lambda: buffer.add(
                valid_observation,
                valid_action,
                np.asarray([-1.0, 0.0], dtype=np.float32),
                0,
                0,
                applied_action=valid_action,
            ),
            ValueError,
            "nonnegative",
        ),
        (
            lambda: buffer.add(
                valid_observation,
                valid_action,
                valid_cost,
                len(TRANSLATION_BLOCK_REASON_NAMES),
                0,
                applied_action=valid_action,
            ),
            ValueError,
            "translation_block_reason_id",
        ),
        (
            lambda: buffer.add(
                valid_observation,
                valid_action,
                valid_cost,
                0,
                len(CURVATURE_STRATUM_NAMES),
                applied_action=valid_action,
            ),
            ValueError,
            "curvature_stratum_id",
        ),
        (
            lambda: buffer.add(
                valid_observation,
                valid_action,
                valid_cost,
                0,
                0,
                applied_action=np.zeros(3, dtype=np.float32),
            ),
            ValueError,
            "shape",
        ),
        (
            lambda: buffer.add(
                valid_observation,
                valid_action,
                valid_cost,
                0,
                0,
                applied_action=valid_action.astype(np.float64),
            ),
            TypeError,
            "float32",
        ),
        (
            lambda: buffer.add(
                valid_observation,
                valid_action,
                valid_cost,
                0,
                0,
                applied_action=np.asarray(
                    [np.inf, 0.0],
                    dtype=np.float32,
                ),
            ),
            FloatingPointError,
            "NaN or infinity",
        ),
    )
    for callable_object, error_type, message in invalid_insertions:
        _assert_raises(callable_object, error_type, message)

    uniform_batch, uniform_metadata = buffer.sample_uniform(9)
    assert uniform_metadata["returned_batch_size"] == 9
    assert uniform_batch["observation"].shape == (9, 14)
    assert uniform_batch["action"].shape == (9, 2)
    assert uniform_batch["applied_action"].shape == (9, 2)
    assert uniform_batch["safety_cost"].shape == (9, 2)
    assert uniform_batch["applied_action"].dtype == np.float32
    assert np.all(np.isfinite(uniform_batch["applied_action"]))
    assert uniform_batch["translation_block_reason_id"].dtype == np.int64
    assert uniform_batch["curvature_stratum_id"].dtype == np.int64
    uniform_batch["observation"].fill(-123.0)
    assert not np.any(buffer.state_dict()["observation"] == -123.0)

    reproducible_state = buffer.state_dict()
    first = _new_buffer(seed=100)
    second = _new_buffer(seed=200)
    first.load_state_dict(reproducible_state)
    second.load_state_dict(reproducible_state)
    _assert_batch_equal(first.sample_uniform(15), second.sample_uniform(15))

    first.load_state_dict(reproducible_state)
    second.load_state_dict(reproducible_state)
    translation_first = first.sample_stratified(
        30, mode="translation-balanced"
    )
    translation_second = second.sample_stratified(
        30, mode="translation_balanced"
    )
    _assert_batch_equal(translation_first, translation_second)
    _, translation_metadata = translation_first
    assert translation_metadata["sampled_translation_counts"] == {
        "none": 10,
        "lower_insertion_boundary": 10,
        "device_length_limit": 0,
        "vessel_tree_end": 10,
        "other": 0,
    }
    assert translation_metadata["missing_translation_groups"] == [
        "device_length_limit",
        "other",
    ]

    first.load_state_dict(reproducible_state)
    _, curvature_metadata = first.sample_stratified(
        40, mode="curvature-balanced"
    )
    assert curvature_metadata["sampled_curvature_counts"] == {
        "low": 10,
        "medium": 10,
        "high": 10,
        "extreme": 10,
    }
    assert curvature_metadata["missing_curvature_groups"] == []

    first.load_state_dict(reproducible_state)
    mixed_batch, mixed_metadata = first.sample_stratified(
        21,
        mode="mixed",
        translation_fraction=0.5,
        curvature_fraction=0.5,
    )
    assert mixed_batch["observation"].shape == (21, 14)
    assert mixed_metadata["component_counts"] == {
        "translation": 11,
        "curvature": 10,
    }
    _assert_raises(
        lambda: first.sample_stratified(
            4,
            mode="mixed",
            translation_fraction=0.8,
            curvature_fraction=0.8,
        ),
        ValueError,
        "sum to 1",
    )

    restored = _new_buffer(seed=999)
    restored.load_state_dict(buffer.state_dict())
    _assert_batch_equal(buffer.sample_uniform(8), restored.sample_uniform(8))

    legacy_replay_state = copy.deepcopy(reproducible_state)
    legacy_replay_state["schema_version"] = 1
    _assert_raises(
        lambda: restored.load_state_dict(legacy_replay_state),
        ValueError,
        "requires a distinct applied_action field",
    )

    for mutate, message in (
        (
            lambda value: value.__setitem__(
                "schema_version", SAFETY_AUX_REPLAY_SCHEMA_VERSION + 1
            ),
            "schema version",
        ),
        (
            lambda value: value.__setitem__(
                "safety_cost_names",
                tuple(reversed(SAFETY_COST_NAMES)),
            ),
            "channel order",
        ),
        (
            lambda value: value.__setitem__(
                "translation_block_reason_names",
                tuple(reversed(TRANSLATION_BLOCK_REASON_NAMES)),
            ),
            "reason order",
        ),
        (
            lambda value: value.__setitem__(
                "curvature_boundaries_mm_inv",
                (0.04, 0.10, 0.25),
            ),
            "curvature-boundary mismatch",
        ),
        (
            lambda value: value.__setitem__(
                "curvature_stratum_names",
                tuple(reversed(CURVATURE_STRATUM_NAMES)),
            ),
            "curvature-stratum order",
        ),
        (
            lambda value: value.__setitem__(
                "observation",
                value["observation"].astype(np.float64),
            ),
            "float32",
        ),
        (
            lambda value: value.__setitem__(
                "applied_action",
                value["applied_action"][:, :1],
            ),
            "shape",
        ),
        (
            lambda value: value.__setitem__(
                "applied_action",
                value["applied_action"].astype(np.float64),
            ),
            "float32",
        ),
        (
            lambda value: value["applied_action"].__setitem__(
                (0, 0),
                np.nan,
            ),
            "NaN or infinity",
        ),
    ):
        invalid_state = copy.deepcopy(reproducible_state)
        mutate(invalid_state)
        _assert_raises(
            lambda value=invalid_state: restored.load_state_dict(value),
            (TypeError, ValueError, FloatingPointError),
            message,
        )

    ring = _new_buffer(seed=3, capacity=3)
    _add_fixture(ring, count=5)
    assert len(ring) == 3
    assert ring.total_added == 5
    assert ring.overwritten_count == 2
    ring_state = ring.state_dict()
    restored_ring = _new_buffer(seed=999, capacity=3)
    restored_ring.load_state_dict(ring_state)
    _assert_batch_equal(
        ring.sample_uniform(11),
        restored_ring.sample_uniform(11),
    )
    for key in (
        "observation",
        "action",
        "applied_action",
        "safety_cost",
        "translation_block_reason_id",
        "curvature_stratum_id",
        "terminated",
        "truncated",
        "episode_step",
    ):
        np.testing.assert_array_equal(
            ring.state_dict()[key],
            restored_ring.state_dict()[key],
        )
    invalid_full_ring = copy.deepcopy(ring_state)
    invalid_full_ring["next_index"] = (
        int(invalid_full_ring["next_index"]) + 1
    ) % 3
    _assert_raises(
        lambda: restored_ring.load_state_dict(invalid_full_ring),
        ValueError,
        "next_index does not match",
    )
    invalid_full_ring = copy.deepcopy(ring_state)
    invalid_full_ring["total_added"] = 2
    _assert_raises(
        lambda: restored_ring.load_state_dict(invalid_full_ring),
        ValueError,
        "cannot be smaller",
    )

    split_buffer = _split_fixture()
    dataset_state = build_dataset_state(
        split_buffer,
        split_seed=4602,
        validation_fraction=0.20,
        generation_seeds={
            "lower-boundary": [100, 101],
            "random": [7],
        },
        observation_round_decimals=6,
    )
    repeated_dataset_state = build_dataset_state(
        split_buffer,
        split_seed=4602,
        validation_fraction=0.20,
        generation_seeds={
            "random": [7],
            "lower-boundary": [100, 101],
        },
        observation_round_decimals=6,
    )
    assert (
        dataset_state["schema_version"]
        == SAFETY_AUX_DATASET_SCHEMA_VERSION
    )
    assert dataset_state["fingerprint"] == repeated_dataset_state["fingerprint"]
    np.testing.assert_array_equal(
        dataset_state["split"]["train_indices"],
        repeated_dataset_state["split"]["train_indices"],
    )
    np.testing.assert_array_equal(
        dataset_state["split"]["validation_indices"],
        repeated_dataset_state["split"]["validation_indices"],
    )

    dataset = validate_dataset_state(
        dataset_state,
        expected_curvature_boundaries_mm_inv=boundaries,
    )
    assert dataset.total_size == 20
    assert dataset.train_size == 17
    assert dataset.validation_size == 3
    train_indices = dataset.train_indices
    validation_indices = dataset.validation_indices
    assert np.intersect1d(train_indices, validation_indices).size == 0
    np.testing.assert_array_equal(
        np.sort(np.concatenate((train_indices, validation_indices))),
        np.arange(dataset.total_size, dtype=np.int64),
    )

    split_replay_state = dataset_state["replay_state"]
    reason_ids = split_replay_state["translation_block_reason_id"]
    stratum_ids = split_replay_state["curvature_stratum_id"]
    for reason_id, stratum_id, expected_count, expected_validation in (
        (0, 0, 10, 2),
        (1, 1, 5, 1),
        (3, 2, 4, 0),
        (0, 3, 1, 0),
    ):
        members = np.flatnonzero(
            (reason_ids == reason_id) & (stratum_ids == stratum_id)
        )
        assert members.size == expected_count
        assert (
            np.intersect1d(members, validation_indices).size
            == expected_validation
        )
        if expected_count < 5:
            np.testing.assert_array_equal(
                np.intersect1d(members, train_indices),
                members,
            )

    semantic_fields = (
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
    training_state = dataset.training_buffer().state_dict()
    validation_state = dataset.validation_buffer().state_dict()
    assert training_state["size"] == train_indices.size
    assert validation_state["size"] == validation_indices.size
    for field in semantic_fields:
        np.testing.assert_array_equal(
            training_state[field],
            split_replay_state[field][train_indices],
        )
        np.testing.assert_array_equal(
            validation_state[field],
            split_replay_state[field][validation_indices],
        )
    train_observations = {
        row.tobytes() for row in training_state["observation"]
    }
    validation_observations = {
        row.tobytes() for row in validation_state["observation"]
    }
    assert train_observations.isdisjoint(validation_observations)

    with TemporaryDirectory(prefix="steve-safety-dataset-test-") as directory:
        dataset_path = Path(directory) / "roundtrip.pt"
        torch.save(dataset.state_dict(), dataset_path)
        loaded_dataset = load_dataset(
            dataset_path,
            expected_curvature_boundaries_mm_inv=boundaries,
        )
        assert loaded_dataset.fingerprint == dataset.fingerprint
        np.testing.assert_array_equal(
            loaded_dataset.train_indices,
            train_indices,
        )
        np.testing.assert_array_equal(
            loaded_dataset.validation_indices,
            validation_indices,
        )

    invalid_dataset = copy.deepcopy(dataset_state)
    del invalid_dataset["split"]["validation_indices"]
    _assert_raises(
        lambda: validate_dataset_state(invalid_dataset),
        ValueError,
        "keys mismatch",
    )
    invalid_dataset = copy.deepcopy(dataset_state)
    invalid_dataset["split"]["validation_indices"] = np.asarray(
        [train_indices[0]],
        dtype=np.int64,
    )
    _assert_raises(
        lambda: validate_dataset_state(invalid_dataset),
        ValueError,
        "overlap",
    )
    invalid_dataset = copy.deepcopy(dataset_state)
    invalid_dataset["fingerprint"] = "0" * 64
    _assert_raises(
        lambda: validate_dataset_state(invalid_dataset),
        ValueError,
        "fingerprint mismatch",
    )
    invalid_dataset = copy.deepcopy(dataset_state)
    invalid_dataset["replay_state"]["translation_block_reason_names"] = tuple(
        reversed(TRANSLATION_BLOCK_REASON_NAMES)
    )
    _assert_raises(
        lambda: validate_dataset_state(invalid_dataset),
        ValueError,
        "reason order",
    )
    _assert_raises(
        lambda: validate_dataset_state(
            dataset_state,
            expected_curvature_boundaries_mm_inv=(0.04, 0.10, 0.25),
        ),
        ValueError,
        "curvature-boundary mismatch",
    )

    duplicate_buffer = _duplicate_fixture()
    duplicate_report = duplicate_diagnostics(
        duplicate_buffer,
        rounding_decimals=6,
    )
    assert duplicate_report["sample_count"] == 5
    assert duplicate_report["exact_duplicate_count"] == 1
    assert duplicate_report["exact_duplicate_group_count"] == 1
    assert duplicate_report["exact_unique_composite_count"] == 4
    assert duplicate_report["exact_max_multiplicity"] == 2
    assert duplicate_report["repeated_observation_action_count"] == 2
    assert duplicate_report[
        "repeated_observation_action_group_count"
    ] == 1
    assert duplicate_report["unique_observation_action_count"] == 3
    assert duplicate_report["observation_action_max_multiplicity"] == 3
    assert duplicate_report["unique_observation_count"] == 2
    assert duplicate_report["rounded_observation_duplicate_count"] == 3
    assert duplicate_report[
        "rounded_observation_duplicate_group_count"
    ] == 1

    unique_indices = exact_unique_indices(duplicate_buffer)
    np.testing.assert_array_equal(
        unique_indices,
        np.asarray([0, 2, 3, 4], dtype=np.int64),
    )
    duplicate_state = duplicate_buffer.state_dict()
    deduplicated_state = subset_replay_state(
        duplicate_buffer,
        unique_indices,
        seed=29,
    )
    assert duplicate_state["size"] - deduplicated_state["size"] == 1
    assert deduplicated_state["size"] == 4
    assert duplicate_diagnostics(deduplicated_state)[
        "exact_duplicate_count"
    ] == 0
    for field in semantic_fields:
        np.testing.assert_array_equal(
            deduplicated_state[field],
            duplicate_state[field][unique_indices],
        )
    np.testing.assert_array_equal(
        deduplicated_state["applied_action"][0],
        duplicate_state["applied_action"][0],
    )
    assert not np.array_equal(
        deduplicated_state["applied_action"][0],
        duplicate_state["applied_action"][1],
    )

    main_replay = EpisodeReplayBuffer(
        100,
        14,
        2,
        3,
        4,
        safety_cost_names=SAFETY_COST_NAMES,
        seed=41,
    )
    observations = np.arange(9 * 14, dtype=np.float32).reshape(9, 14) / 100
    actions = np.arange(8 * 2, dtype=np.float32).reshape(8, 2) / 10
    rewards = np.arange(8, dtype=np.float32)
    terminated = np.zeros(8, dtype=bool)
    costs = np.stack((rewards / 100, rewards / 10), axis=-1).astype(
        np.float32
    )
    main_replay.add_episode(
        observations,
        actions,
        rewards,
        terminated,
        safety_cost=costs,
    )
    main_state = copy.deepcopy(main_replay.state_dict())
    main_clone = EpisodeReplayBuffer(
        100,
        14,
        2,
        3,
        4,
        safety_cost_names=SAFETY_COST_NAMES,
        seed=999,
    )
    main_clone.load_state_dict(main_state)
    buffer.sample_uniform(7)
    buffer.sample_stratified(7, mode="translation-balanced")
    buffer.sample_stratified(7, mode="curvature-balanced")
    buffer.sample_stratified(7, mode="mixed")
    assert main_replay.state_dict()["rng_state"] == main_state["rng_state"]
    for actual, expected in zip(
        main_replay.sample(torch.device("cpu")),
        main_clone.sample(torch.device("cpu")),
    ):
        torch.testing.assert_close(actual, expected)

    assert _planned_storage_upper_bound(
        (
            "random",
            "lower-boundary",
            "vessel-tree-end",
            "device-length",
            "curvature-coverage",
        ),
        lower_boundary_repetitions=2,
        tree_end_repetitions=3,
        device_length_repetitions=4,
        random_episodes=2,
        curvature_episodes=3,
        max_curvature_transitions=125,
        max_episode_steps=200,
    ) == 552
    same_output_args = parse_args(
        [
            "--output-dataset",
            "/tmp/same-safety-output",
            "--output-report",
            "/tmp/same-safety-output",
        ]
    )
    _assert_raises(
        lambda: _resolved_output_paths(same_output_args),
        ValueError,
        "different paths",
    )

    print(
        "PASS: Safety auxiliary replay validation, applied-action schema, "
        "uniform/stratified sampling, RNG isolation, deterministic joint "
        "train/validation split, strict dataset round-trip, corruption "
        "rejection, and duplicate diagnostics/deduplication."
    )


if __name__ == "__main__":
    main()
