"""Focused, simulation-free checks for Safety auxiliary data infrastructure."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from collect_safety_dataset import (
    CollectionRecorder,
    _planned_storage_upper_bound,
    _resolved_output_paths,
    build_dataset_report,
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
            terminated=index == count - 1,
            truncated=False,
            episode_step=index + 1,
        )
        observation.fill(-99.0)
        action.fill(-99.0)
        cost.fill(-99.0)


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
    assert state["safety_cost"].shape == (12, 2)
    assert state["observation"].dtype == np.float32
    assert state["action"].dtype == np.float32
    assert state["safety_cost"].dtype == np.float32
    assert np.all(state["observation"] > -1.0)

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
            ),
            ValueError,
            "curvature_stratum_id",
        ),
    )
    for callable_object, error_type, message in invalid_insertions:
        _assert_raises(callable_object, error_type, message)

    uniform_batch, uniform_metadata = buffer.sample_uniform(9)
    assert uniform_metadata["returned_batch_size"] == 9
    assert uniform_batch["observation"].shape == (9, 14)
    assert uniform_batch["action"].shape == (9, 2)
    assert uniform_batch["safety_cost"].shape == (9, 2)
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
    ):
        invalid_state = copy.deepcopy(reproducible_state)
        mutate(invalid_state)
        _assert_raises(
            lambda value=invalid_state: restored.load_state_dict(value),
            (TypeError, ValueError),
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
        random_episodes=2,
        curvature_episodes=3,
        max_episode_steps=200,
    ) == 1005
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

    recorder = CollectionRecorder(buffer, boundaries)
    report = build_dataset_report(
        buffer,
        recorder,
        modes=("random",),
        seeds={"base": 7},
        environment_config=config["environment"],
        safety_aux_config=aux_config,
        curvature_boundaries=boundaries,
        collection_parameters={
            "planned_storage_upper_bound": 200,
            "buffer_capacity": buffer.capacity,
        },
        dataset_path=Path("/tmp/example.safety_aux.pt"),
        report_path=None,
    )
    assert report["total_transition_count"] == len(buffer)
    assert report["collection_parameters"][
        "planned_storage_upper_bound"
    ] == 200
    assert report["storage"]["observation"]["shape"] == [len(buffer), 14]
    assert (
        sum(
            group["count"]
            for group in report["translation_block_reasons"].values()
        )
        == len(buffer)
    )
    assert (
        sum(group["count"] for group in report["curvature_strata"].values())
        == len(buffer)
    )
    print(
        "PASS: SafetyAuxReplayBuffer validation, schemas, uniform/stratified "
        "sampling, RNG isolation, round-trip restore, config defaults, and "
        "dataset reporting."
    )


if __name__ == "__main__":
    main()
