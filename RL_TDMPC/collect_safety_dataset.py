#!/usr/bin/env python3
"""Collect a standalone stEVE Safety auxiliary transition dataset."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from envs.safety import (
    CURVATURE_STRATUM_NAMES,
    SAFETY_COST_NAMES,
    curvature_stratum_id,
    safety_aux_metadata_from_metrics,
)
from envs.steve_env import StEVEEnv, make_steve_env
from eve.intervention import TRANSLATION_BLOCK_REASON_NAMES
from eve.intervention.vesseltree.vesseltree import at_tree_end
from tdmpc2.common import (
    PROJECT_DIR,
    atomic_json_save,
    atomic_torch_save,
    build_diagnostics_agent_config,
    build_safety_aux_config,
    curvature_boundaries_from_diagnostics,
    load_config,
)
from tdmpc2.safety_aux_replay import (
    SAFETY_AUX_REPLAY_SCHEMA_VERSION,
    SafetyAuxReplayBuffer,
)
from tdmpc2.safety_aux_dataset import (
    SAFETY_AUX_DATASET_SCHEMA_VERSION,
    SafetyAuxDataset,
    build_dataset_state,
    duplicate_diagnostics,
    exact_unique_indices,
    subset_replay_state,
    validate_dataset_state,
)


DEFAULT_CONFIG = PROJECT_DIR / "configs" / "steve.yaml"
DEFAULT_DATASET = Path("/tmp/steve_safety_aux_dataset.pt")
DEFAULT_REPORT = Path("/tmp/steve_safety_aux_report.json")
DEFAULT_TREE_END_SEED = 301
DEFAULT_DEVICE_LENGTH_SEED = 301
DEFAULT_SPLIT_SEED = 4602
DEFAULT_VALIDATION_FRACTION = 0.20
DEFAULT_OBSERVATION_ROUND_DECIMALS = 6
DEFAULT_COLLECTION_CAPACITY = 100000
LOWER_BOUNDARY_TOLERANCE_MM = 1.0e-3
COLLECTION_MODES: Tuple[str, ...] = (
    "random",
    "lower-boundary",
    "vessel-tree-end",
    "device-length",
    "curvature-coverage",
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["all"],
        choices=("all", *COLLECTION_MODES),
        help="Collection modes; default executes every mode",
    )
    parser.add_argument("--lower-boundary-repetitions", type=int, default=1)
    parser.add_argument("--tree-end-repetitions", type=int, default=1)
    parser.add_argument("--device-length-repetitions", type=int, default=1)
    parser.add_argument("--random-episodes", type=int, default=2)
    parser.add_argument(
        "--curvature-episodes",
        type=int,
        default=2,
        help="Maximum curvature-coverage trajectories before target stopping",
    )
    parser.add_argument(
        "--seed-start",
        "--seed",
        dest="seed_start",
        type=int,
        default=1,
        help="Base seed; --seed is retained as a backward-compatible alias",
    )
    parser.add_argument("--tree-end-seed", type=int, default=DEFAULT_TREE_END_SEED)
    parser.add_argument(
        "--device-length-seed",
        type=int,
        default=DEFAULT_DEVICE_LENGTH_SEED,
    )
    parser.add_argument("--moderate-action", type=float, default=0.5)
    parser.add_argument("--maximum-action", type=float, default=1.0)
    parser.add_argument("--target-none", type=int, default=500)
    parser.add_argument("--target-lower-boundary", type=int, default=100)
    parser.add_argument("--target-tree-end", type=int, default=100)
    parser.add_argument("--target-low", type=int, default=100)
    parser.add_argument("--target-medium", type=int, default=500)
    parser.add_argument("--target-high", type=int, default=300)
    parser.add_argument("--target-extreme", type=int, default=100)
    parser.add_argument(
        "--max-curvature-transitions",
        type=int,
        default=2000,
    )
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=DEFAULT_VALIDATION_FRACTION,
    )
    parser.add_argument(
        "--observation-round-decimals",
        type=int,
        default=DEFAULT_OBSERVATION_ROUND_DECIMALS,
    )
    parser.add_argument(
        "--deduplicate-exact",
        action="store_true",
        help="Keep the first representative of each exact semantic duplicate",
    )
    parser.add_argument(
        "--capacity",
        type=int,
        default=DEFAULT_COLLECTION_CAPACITY,
        help=(
            "Collector-owned transition capacity, independent of the training "
            "safety_aux configuration"
        ),
    )
    parser.add_argument(
        "--output-dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Torch state_dict destination (default: /tmp)",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=DEFAULT_REPORT,
        help="JSON report destination (default: /tmp)",
    )
    parser.add_argument("--no-save-dataset", action="store_true")
    parser.add_argument("--no-save-report", action="store_true")
    return parser.parse_args(argv)


class CollectionRecorder:
    """Track executed simulation transitions separately from stored samples."""

    def __init__(
        self,
        buffer: SafetyAuxReplayBuffer,
        curvature_boundaries: Sequence[float],
    ) -> None:
        self.buffer = buffer
        self.curvature_boundaries = tuple(curvature_boundaries)
        self.executed_transition_count = 0
        self.simulation_error_count = 0
        self.controlled_scenarios: List[Dict[str, Any]] = []
        self.controlled_constructions: List[Dict[str, Any]] = []
        self.unsupported_scenarios: List[Dict[str, Any]] = []
        self.seeds_used: Dict[str, List[int]] = {}
        self.curvature_coverage: Dict[str, Any] = {}

    def record_seed(self, mode: str, seed: int) -> None:
        seeds = self.seeds_used.setdefault(str(mode), [])
        parsed = int(seed)
        if parsed not in seeds:
            seeds.append(parsed)

    def observe_execution(self, info: Mapping[str, Any]) -> None:
        self.executed_transition_count += 1
        self.simulation_error_count += int(
            bool(info.get("simulation_error", False))
        )

    def store(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        info: Mapping[str, Any],
        *,
        terminated: bool,
        truncated: bool,
    ) -> Tuple[int, int, int]:
        metrics = info.get("safety_metrics")
        reason_id, _ = safety_aux_metadata_from_metrics(
            metrics,
            self.curvature_boundaries,
        )
        cost = info.get("safety_cost")
        stratum_id = curvature_stratum_id(
            float(cost[0]),
            self.curvature_boundaries,
        )
        requested_action = np.asarray(
            info["requested_action"],
            dtype=np.float32,
        )
        applied_action = np.asarray(
            info["applied_action"],
            dtype=np.float32,
        )
        if not np.allclose(
            requested_action,
            np.asarray(action, dtype=np.float32),
            rtol=0.0,
            atol=1.0e-6,
        ):
            raise RuntimeError(
                "Environment requested_action does not match the command "
                "supplied for this transition"
            )
        sample_index = self.buffer.total_added
        self.buffer.add(
            observation,
            requested_action,
            cost,
            reason_id,
            stratum_id,
            applied_action=applied_action,
            terminated=terminated,
            truncated=truncated,
            episode_step=int(info["episode_step"]),
        )
        return reason_id, stratum_id, sample_index


def _expanded_modes(values: Sequence[str]) -> Tuple[str, ...]:
    requested = tuple(values)
    if "all" in requested:
        return COLLECTION_MODES
    # Preserve CLI order while eliminating accidental duplicates.
    return tuple(dict.fromkeys(requested))


def _validate_collection_args(args: argparse.Namespace) -> None:
    nonnegative_integer_names = (
        "lower_boundary_repetitions",
        "tree_end_repetitions",
        "device_length_repetitions",
        "random_episodes",
        "curvature_episodes",
        "target_none",
        "target_lower_boundary",
        "target_tree_end",
        "target_low",
        "target_medium",
        "target_high",
        "target_extreme",
        "max_curvature_transitions",
        "seed_start",
        "tree_end_seed",
        "device_length_seed",
        "split_seed",
    )
    for name in nonnegative_integer_names:
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be nonnegative")
    moderate = float(args.moderate_action)
    maximum = float(args.maximum_action)
    if (
        not np.isfinite(moderate)
        or not np.isfinite(maximum)
        or not 0.0 < moderate < maximum <= 1.0
    ):
        raise ValueError(
            "Controlled actions must satisfy "
            "0 < --moderate-action < --maximum-action <= 1"
        )
    if args.capacity is not None and args.capacity <= 0:
        raise ValueError("--capacity must be positive")
    validation_fraction = float(args.validation_fraction)
    if (
        not np.isfinite(validation_fraction)
        or not 0.0 < validation_fraction < 1.0
    ):
        raise ValueError("--validation-fraction must be finite and in (0, 1)")
    if (
        isinstance(args.observation_round_decimals, bool)
        or not 0 <= int(args.observation_round_decimals) <= 12
    ):
        raise ValueError(
            "--observation-round-decimals must be an integer in [0, 12]"
        )


def _planned_storage_upper_bound(
    modes: Sequence[str],
    *,
    lower_boundary_repetitions: int = 1,
    tree_end_repetitions: int = 1,
    device_length_repetitions: int = 1,
    random_episodes: int,
    curvature_episodes: int,
    max_curvature_transitions: Optional[int] = None,
    max_episode_steps: int,
) -> int:
    """Return a conservative no-overwrite capacity requirement."""

    selected = set(modes)
    maximum = 0
    if "random" in selected:
        maximum += int(random_episodes) * int(max_episode_steps)
    if "curvature-coverage" in selected:
        episode_bound = int(curvature_episodes) * int(max_episode_steps)
        maximum += (
            episode_bound
            if max_curvature_transitions is None
            else min(episode_bound, int(max_curvature_transitions))
        )
    if "lower-boundary" in selected:
        maximum += 4 * int(lower_boundary_repetitions)
    if "vessel-tree-end" in selected:
        maximum += 5 * int(tree_end_repetitions)
    if "device-length" in selected:
        maximum += int(device_length_repetitions)
    return maximum


def _resolved_output_paths(
    args: argparse.Namespace,
) -> Tuple[Optional[Path], Optional[Path]]:
    dataset_path = (
        None
        if args.no_save_dataset
        else args.output_dataset.expanduser().resolve()
    )
    report_path = (
        None
        if args.no_save_report
        else args.output_report.expanduser().resolve()
    )
    if (
        dataset_path is not None
        and report_path is not None
        and dataset_path == report_path
    ):
        raise ValueError(
            "--output-dataset and --output-report must resolve to different paths"
        )
    return dataset_path, report_path


def _at_tree_end(env: StEVEEnv) -> bool:
    positions = np.asarray(
        env.intervention.simulation.dof_positions,
        dtype=np.float64,
    )
    return bool(
        positions.ndim == 2
        and positions.shape[0] > 0
        and at_tree_end(
            positions[0],
            env.intervention.vessel_tree,
        )
    )


def _step(
    env: StEVEEnv,
    action: np.ndarray,
    recorder: CollectionRecorder,
) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
    result = env.step(action)
    recorder.observe_execution(result[4])
    return result


def _collect_controlled_action(
    env: StEVEEnv,
    recorder: CollectionRecorder,
    observation: np.ndarray,
    action: np.ndarray,
    *,
    scenario: str,
    variant: str,
    repetition: int,
    seed: int,
    expected_reason: str,
    approach_steps: int,
) -> Tuple[np.ndarray, bool, bool, Mapping[str, Any], bool]:
    (
        next_observation,
        _,
        terminated,
        truncated,
        info,
    ) = _step(env, action, recorder)
    reason_id, stratum_id, sample_index = recorder.store(
        observation,
        action,
        info,
        terminated=terminated,
        truncated=truncated,
    )
    actual_reason = TRANSLATION_BLOCK_REASON_NAMES[reason_id]
    matched = actual_reason == expected_reason
    recorder.controlled_scenarios.append(
        _controlled_record(
            scenario=scenario,
            variant=variant,
            repetition=repetition,
            seed=seed,
            status="observed" if matched else "unexpected_label",
            expected_reason=expected_reason,
            approach_steps=approach_steps,
            action=action,
            metrics=info["safety_metrics"],
            safety_cost=info["safety_cost"],
            curvature_stratum_id_value=stratum_id,
            sample_index=sample_index,
            requested_action=info["requested_action"],
            applied_action=info["applied_action"],
            stored=True,
        )
    )
    return next_observation, terminated, truncated, info, matched


def collect_random_policy(
    env: StEVEEnv,
    recorder: CollectionRecorder,
    *,
    episodes: int,
    base_seed: int,
) -> None:
    for episode_index in range(episodes):
        episode_seed = int(base_seed + episode_index)
        recorder.record_seed("random", episode_seed)
        rng = np.random.default_rng(episode_seed)
        observation, _ = env.reset(seed=episode_seed)
        terminated = truncated = False
        while not (terminated or truncated):
            action = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
            (
                next_observation,
                _,
                terminated,
                truncated,
                info,
            ) = _step(env, action, recorder)
            recorder.store(
                observation,
                action,
                info,
                terminated=terminated,
                truncated=truncated,
            )
            observation = next_observation


def collect_lower_boundary(
    env: StEVEEnv,
    recorder: CollectionRecorder,
    *,
    base_seed: int,
    repetitions: int,
    magnitudes: Sequence[float],
) -> None:
    moderate, maximum = (float(value) for value in magnitudes)
    action_plan = (
        (
            "control_zero",
            np.asarray([0.0, 0.0], dtype=np.float32),
            "none",
        ),
        (
            "blocked_moderate",
            np.asarray([-moderate, 0.0], dtype=np.float32),
            "lower_insertion_boundary",
        ),
        (
            "blocked_maximum",
            np.asarray([-maximum, 0.0], dtype=np.float32),
            "lower_insertion_boundary",
        ),
        (
            "control_forward_moderate",
            np.asarray([moderate, 0.0], dtype=np.float32),
            "none",
        ),
    )
    for repetition in range(int(repetitions)):
        seed = int(base_seed + repetition)
        recorder.record_seed("lower_boundary", seed)
        observation, reset_info = env.reset(seed=seed)
        initial_length = float(
            reset_info["safety_metrics"]["inserted_length_mm"]
        )
        near_boundary = abs(initial_length) <= LOWER_BOUNDARY_TOLERANCE_MM
        construction: Dict[str, Any] = {
            "scenario": "lower_insertion_boundary",
            "repetition": repetition,
            "seed": seed,
            "initial_inserted_length_mm": initial_length,
            "lower_boundary_tolerance_mm": LOWER_BOUNDARY_TOLERANCE_MM,
            "near_lower_boundary": bool(near_boundary),
            "attempted_actions": 0,
            "matched_actions": 0,
        }
        if not near_boundary:
            construction.update(
                {
                    "status": "failed",
                    "reason": (
                        "Reset insertion length was not within the configured "
                        "lower-boundary tolerance."
                    ),
                }
            )
            recorder.controlled_constructions.append(construction)
            continue

        terminated = truncated = False
        for variant, action, expected_reason in action_plan:
            if terminated or truncated:
                break
            (
                observation,
                terminated,
                truncated,
                _,
                matched,
            ) = _collect_controlled_action(
                env,
                recorder,
                observation,
                action,
                scenario="lower_insertion_boundary",
                variant=variant,
                repetition=repetition,
                seed=seed,
                expected_reason=expected_reason,
                approach_steps=0,
            )
            construction["attempted_actions"] += 1
            construction["matched_actions"] += int(matched)
        construction["status"] = (
            "successful"
            if construction["attempted_actions"] == len(action_plan)
            and construction["matched_actions"] == len(action_plan)
            else "failed"
        )
        construction["terminated"] = bool(terminated)
        construction["truncated"] = bool(truncated)
        recorder.controlled_constructions.append(construction)


def collect_tree_end(
    env: StEVEEnv,
    recorder: CollectionRecorder,
    *,
    base_seed: int,
    repetitions: int,
    magnitudes: Sequence[float],
) -> None:
    moderate, maximum = (float(value) for value in magnitudes)
    approach_action = np.asarray([1.0, 0.0], dtype=np.float32)
    action_plan = (
        (
            "control_zero",
            np.asarray([0.0, 0.0], dtype=np.float32),
            "none",
        ),
        (
            "blocked_moderate",
            np.asarray([moderate, 0.0], dtype=np.float32),
            "vessel_tree_end",
        ),
        (
            "blocked_maximum",
            np.asarray([maximum, 0.0], dtype=np.float32),
            "vessel_tree_end",
        ),
        (
            "control_rotation_only",
            np.asarray([0.0, moderate], dtype=np.float32),
            "none",
        ),
        (
            "control_retraction_moderate",
            np.asarray([-moderate, 0.0], dtype=np.float32),
            "none",
        ),
    )
    for repetition in range(int(repetitions)):
        seed = int(base_seed + repetition)
        recorder.record_seed("vessel_tree_end", seed)
        observation, _ = env.reset(seed=seed)
        terminated = truncated = False
        approach_steps = 0
        local_simulation_errors = 0
        while not _at_tree_end(env) and not (terminated or truncated):
            (
                observation,
                _,
                terminated,
                truncated,
                info,
            ) = _step(env, approach_action, recorder)
            local_simulation_errors += int(
                bool(info.get("simulation_error", False))
            )
            approach_steps += 1
        if terminated or truncated or not _at_tree_end(env):
            recorder.controlled_constructions.append(
                {
                    "scenario": "vessel_tree_end",
                    "repetition": repetition,
                    "seed": seed,
                    "status": "failed",
                    "approach_steps": int(approach_steps),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "ended_before_tree_end": True,
                    "simulation_error_count": local_simulation_errors,
                    "reason": (
                        "Episode ended before the fixed-tree endpoint was "
                        "reached."
                    ),
                }
            )
            continue

        construction: Dict[str, Any] = {
            "scenario": "vessel_tree_end",
            "repetition": repetition,
            "seed": seed,
            "status": "successful",
            "ended_before_tree_end": False,
            "approach_steps": int(approach_steps),
            "simulation_error_count": local_simulation_errors,
            "attempted_actions": 0,
            "matched_actions": 0,
        }
        for variant, action, expected_reason in action_plan:
            if terminated or truncated:
                break
            (
                observation,
                terminated,
                truncated,
                info,
                matched,
            ) = _collect_controlled_action(
                env,
                recorder,
                observation,
                action,
                scenario="vessel_tree_end",
                variant=variant,
                repetition=repetition,
                seed=seed,
                expected_reason=expected_reason,
                approach_steps=approach_steps,
            )
            construction["attempted_actions"] += 1
            construction["matched_actions"] += int(matched)
            construction["simulation_error_count"] += int(
                bool(info.get("simulation_error", False))
            )
        if (
            construction["attempted_actions"] != len(action_plan)
            or construction["matched_actions"] != len(action_plan)
        ):
            construction["status"] = "failed"
        construction["terminated"] = bool(terminated)
        construction["truncated"] = bool(truncated)
        recorder.controlled_constructions.append(construction)


def collect_device_length_attempt(
    env: StEVEEnv,
    recorder: CollectionRecorder,
    *,
    base_seed: int,
    repetitions: int,
) -> None:
    expected_id = TRANSLATION_BLOCK_REASON_NAMES.index("device_length_limit")
    action = np.asarray([1.0, 0.0], dtype=np.float32)
    for repetition in range(int(repetitions)):
        seed = int(base_seed + repetition)
        recorder.record_seed("device_length", seed)
        observation, _ = env.reset(seed=seed)
        terminated = truncated = False
        approach_steps = 0
        observed_info: Optional[Mapping[str, Any]] = None
        observed_observation: Optional[np.ndarray] = None
        local_simulation_errors = 0
        while not (terminated or truncated):
            (
                next_observation,
                _,
                terminated,
                truncated,
                info,
            ) = _step(env, action, recorder)
            local_simulation_errors += int(
                bool(info.get("simulation_error", False))
            )
            approach_steps += 1
            metrics = info["safety_metrics"]
            if int(metrics["translation_block_reason_id"]) != 0:
                observed_info = info
                observed_observation = observation
                break
            observation = next_observation

        first_block_reason = (
            str(observed_info["safety_metrics"]["translation_block_reason"])
            if observed_info is not None
            else "none"
        )
        inserted_length = (
            float(observed_info["safety_metrics"]["inserted_length_mm"])
            if observed_info is not None
            else float(env.intervention.device_lengths_inserted[0])
        )
        device_maximum_length = float(
            env.intervention.device_lengths_maximum[0]
        )
        base_record: Dict[str, Any] = {
            "scenario": "device_length_limit",
            "variant": "blocked_maximum",
            "repetition": repetition,
            "seed": seed,
            "approach_steps": int(
                max(0, approach_steps - 1)
                if observed_info is not None
                else approach_steps
            ),
            "first_block_reason": first_block_reason,
            "inserted_length_mm": inserted_length,
            "device_maximum_length_mm": device_maximum_length,
            "simulation_error_count": local_simulation_errors,
            "stored": False,
        }
        if (
            observed_info is not None
            and int(
                observed_info["safety_metrics"][
                    "translation_block_reason_id"
                ]
            )
            == expected_id
            and observed_observation is not None
        ):
            reason_id, stratum_id, sample_index = recorder.store(
                observed_observation,
                action,
                observed_info,
                terminated=terminated,
                truncated=truncated,
            )
            record = _controlled_record(
                scenario="device_length_limit",
                variant="blocked_maximum",
                repetition=repetition,
                seed=seed,
                status="observed",
                expected_reason="device_length_limit",
                approach_steps=max(0, approach_steps - 1),
                action=action,
                metrics=observed_info["safety_metrics"],
                safety_cost=observed_info["safety_cost"],
                curvature_stratum_id_value=stratum_id,
                sample_index=sample_index,
                requested_action=observed_info["requested_action"],
                applied_action=observed_info["applied_action"],
                stored=True,
            )
            assert reason_id == expected_id
            recorder.controlled_scenarios.append(record)
            recorder.controlled_constructions.append(
                {
                    **base_record,
                    "status": "successful",
                    "stored": True,
                    "pre_dedup_sample_index": sample_index,
                }
            )
            continue

        if first_block_reason == "vessel_tree_end":
            unsupported = {
                **base_record,
                "status": "unsupported",
                "reason": (
                    "The unchanged fixed vessel reached its tree endpoint at "
                    f"{inserted_length:.6g} mm before the device-length limit "
                    f"of {device_maximum_length:.6g} mm; no sample was "
                    "fabricated."
                ),
            }
            recorder.unsupported_scenarios.append(unsupported)
            recorder.controlled_scenarios.append(
                copy.deepcopy(unsupported)
            )
            recorder.controlled_constructions.append(
                copy.deepcopy(unsupported)
            )
            continue

        reason = (
            "The episode terminated or truncated before any translation "
            "blocker was observed; physical support for device-length blockage "
            "was not established."
            if observed_info is None
            else (
                "The first observed translation blocker was "
                f"{first_block_reason!r}, not the device-length limit; no "
                "sample was fabricated."
            )
        )
        not_observed = {
            **base_record,
            "status": "not_observed",
            "reason": reason,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
        recorder.controlled_scenarios.append(not_observed)
        recorder.controlled_constructions.append(
            copy.deepcopy(not_observed)
        )


def collect_curvature_coverage(
    env: StEVEEnv,
    recorder: CollectionRecorder,
    *,
    episodes: int,
    base_seed: int,
    target_counts: Mapping[str, int],
    max_transitions: int,
) -> None:
    canonical_targets = {
        name: int(target_counts[name]) for name in CURVATURE_STRATUM_NAMES
    }
    counts_at_start = _curvature_counts(recorder.buffer)
    transitions_examined = 0
    episodes_run = 0
    stop_reason = "episode_limit"
    for episode_index in range(int(episodes)):
        current_counts = _curvature_counts(recorder.buffer)
        if all(
            current_counts[name] >= canonical_targets[name]
            for name in CURVATURE_STRATUM_NAMES
        ):
            stop_reason = "targets_reached"
            break
        if transitions_examined >= int(max_transitions):
            stop_reason = "maximum_transitions_reached"
            break
        episode_seed = int(base_seed + episode_index)
        recorder.record_seed("curvature_coverage", episode_seed)
        rng = np.random.default_rng(episode_seed)
        observation, _ = env.reset(seed=episode_seed)
        terminated = truncated = False
        step_index = 0
        episodes_run += 1
        while (
            not (terminated or truncated)
            and transitions_examined < int(max_transitions)
        ):
            current_counts = _curvature_counts(recorder.buffer)
            if all(
                current_counts[name] >= canonical_targets[name]
                for name in CURVATURE_STRATUM_NAMES
            ):
                stop_reason = "targets_reached"
                break
            action = _curvature_exploration_action(
                rng,
                episode_index=episode_index,
                step_index=step_index,
            )
            (
                next_observation,
                _,
                terminated,
                truncated,
                info,
            ) = _step(env, action, recorder)
            recorder.store(
                observation,
                action,
                info,
                terminated=terminated,
                truncated=truncated,
            )
            observation = next_observation
            step_index += 1
            transitions_examined += 1
        if stop_reason == "targets_reached":
            break
    if (
        stop_reason != "targets_reached"
        and transitions_examined >= int(max_transitions)
    ):
        stop_reason = "maximum_transitions_reached"

    collected = _curvature_counts(recorder.buffer)
    recorder.curvature_coverage = {
        "requested_count": canonical_targets,
        "count_at_coverage_start": counts_at_start,
        "collected_count": collected,
        "unmet_count": {
            name: max(0, canonical_targets[name] - collected[name])
            for name in CURVATURE_STRATUM_NAMES
        },
        "transitions_examined": int(transitions_examined),
        "maximum_transitions": int(max_transitions),
        "episode_limit": int(episodes),
        "episodes_run": int(episodes_run),
        "stop_reason": stop_reason,
    }


def _curvature_counts(
    buffer: SafetyAuxReplayBuffer,
) -> Dict[str, int]:
    state = buffer.state_dict()
    counts = np.bincount(
        state["curvature_stratum_id"],
        minlength=len(CURVATURE_STRATUM_NAMES),
    )
    return {
        name: int(counts[index])
        for index, name in enumerate(CURVATURE_STRATUM_NAMES)
    }


def _curvature_exploration_action(
    rng: np.random.Generator,
    *,
    episode_index: int,
    step_index: int,
) -> np.ndarray:
    """Generate diverse real commands without injecting curvature labels."""

    policy_variant = int(episode_index) % 4
    if policy_variant == 0:
        translation = rng.uniform(0.25, 1.0)
        rotation = np.clip(
            0.75 * np.sin(0.35 * step_index)
            + 0.25 * rng.uniform(-1.0, 1.0),
            -1.0,
            1.0,
        )
    elif policy_variant == 1:
        translation = rng.uniform(-0.25, 1.0)
        rotation = rng.uniform(-1.0, 1.0)
    elif policy_variant == 2:
        translation = rng.uniform(0.65, 1.0)
        direction = 1.0 if step_index % 2 == 0 else -1.0
        rotation = direction * rng.uniform(0.65, 1.0)
    else:
        translation = rng.uniform(-1.0, 1.0)
        rotation = rng.uniform(-1.0, 1.0)
    return np.asarray([translation, rotation], dtype=np.float32)


def _controlled_record(
    *,
    scenario: str,
    variant: str,
    repetition: int,
    seed: int,
    status: str,
    expected_reason: str,
    approach_steps: int,
    action: np.ndarray,
    metrics: Mapping[str, Any],
    safety_cost: np.ndarray,
    curvature_stratum_id_value: int,
    sample_index: int,
    requested_action: np.ndarray,
    applied_action: np.ndarray,
    stored: bool,
) -> Dict[str, Any]:
    return {
        "scenario": scenario,
        "variant": variant,
        "repetition": int(repetition),
        "seed": int(seed),
        "status": status,
        "expected_translation_block_reason": str(expected_reason),
        "approach_steps": int(approach_steps),
        "pre_dedup_sample_index": int(sample_index),
        "commanded_normalized_action": np.asarray(
            action,
            dtype=np.float32,
        ).tolist(),
        "requested_normalized_action": np.asarray(
            requested_action,
            dtype=np.float32,
        ).tolist(),
        # This is the normalized post-mask command, not measured device motion.
        "applied_normalized_action": np.asarray(
            applied_action,
            dtype=np.float32,
        ).tolist(),
        "requested_translation_speed_mm_s": float(
            metrics["requested_translation_speed_mm_s"]
        ),
        "applied_translation_speed_mm_s": float(
            metrics["applied_translation_speed_mm_s"]
        ),
        "translation_block_reason_id": int(
            metrics["translation_block_reason_id"]
        ),
        "translation_block_reason": str(
            metrics["translation_block_reason"]
        ),
        "curvature_stratum_id": int(curvature_stratum_id_value),
        "curvature_stratum": CURVATURE_STRATUM_NAMES[
            int(curvature_stratum_id_value)
        ],
        "inserted_length_mm": float(metrics["inserted_length_mm"]),
        "simulation_error": bool(metrics["simulation_error"]),
        "safety_cost": {
            name: float(safety_cost[index])
            for index, name in enumerate(SAFETY_COST_NAMES)
        },
        "stored": bool(stored),
    }


def build_dataset_report(
    dataset: SafetyAuxDataset,
    recorder: CollectionRecorder,
    *,
    modes: Sequence[str],
    seed_configuration: Mapping[str, Any],
    environment_config: Mapping[str, Any],
    safety_aux_config: Mapping[str, Any],
    collection_parameters: Mapping[str, Any],
    duplicate_diagnostics_before: Mapping[str, Any],
    duplicate_diagnostics_after: Mapping[str, Any],
    deduplication_enabled: bool,
    deduplicated_sample_count: int,
    translation_targets: Mapping[str, Optional[int]],
    curvature_targets: Mapping[str, int],
    dataset_path: Optional[Path],
    report_path: Optional[Path],
) -> Dict[str, Any]:
    summary = dataset.summary()
    state = dataset.state_dict()["replay_state"]
    total_distribution = summary["distributions"]["total"]
    reason_distribution = total_distribution["translation_block_reasons"]
    stratum_distribution = total_distribution["curvature_strata"]
    translation_coverage = {
        name: {
            "requested_count": (
                None if translation_targets[name] is None
                else int(translation_targets[name])
            ),
            "collected_count": int(reason_distribution[name]["count"]),
            "unmet_count": (
                None
                if translation_targets[name] is None
                else max(
                    0,
                    int(translation_targets[name])
                    - int(reason_distribution[name]["count"]),
                )
            ),
            "optional": translation_targets[name] is None,
        }
        for name in TRANSLATION_BLOCK_REASON_NAMES
    }
    curvature_coverage = copy.deepcopy(recorder.curvature_coverage)
    curvature_coverage["collected_count_before_optional_deduplication"] = (
        copy.deepcopy(curvature_coverage.get("collected_count", {}))
    )
    curvature_coverage["requested_count"] = {
        name: int(curvature_targets[name])
        for name in CURVATURE_STRATUM_NAMES
    }
    curvature_coverage["collected_count"] = {
        name: int(stratum_distribution[name]["count"])
        for name in CURVATURE_STRATUM_NAMES
    }
    curvature_coverage["unmet_count"] = {
        name: max(
            0,
            int(curvature_targets[name])
            - int(stratum_distribution[name]["count"]),
        )
        for name in CURVATURE_STRATUM_NAMES
    }
    return {
        "report_schema_version": 2,
        "dataset_schema_version": SAFETY_AUX_DATASET_SCHEMA_VERSION,
        "auxiliary_buffer_schema_version": SAFETY_AUX_REPLAY_SCHEMA_VERSION,
        "dataset_fingerprint": dataset.fingerprint,
        "output_paths": {
            "dataset": (
                str(dataset_path) if dataset_path is not None else None
            ),
            "report": str(report_path) if report_path is not None else None,
        },
        "modes": list(modes),
        "collection_parameters": copy.deepcopy(
            dict(collection_parameters)
        ),
        "seed_configuration": copy.deepcopy(dict(seed_configuration)),
        "seeds_used": copy.deepcopy(recorder.seeds_used),
        "dataset_generation_seeds": copy.deepcopy(
            summary["generation_seeds"]
        ),
        "environment_configuration": copy.deepcopy(
            dict(environment_config)
        ),
        "safety_aux_configuration": copy.deepcopy(
            dict(safety_aux_config)
        ),
        "schema": copy.deepcopy(summary["schema"]),
        "sizes": copy.deepcopy(summary["sizes"]),
        "split": copy.deepcopy(summary["split"]),
        "executed_transition_count": int(
            recorder.executed_transition_count
        ),
        "total_transition_count": int(dataset.total_size),
        "total_added_before_optional_deduplication": int(
            recorder.buffer.total_added
        ),
        "overwritten_transition_count": int(
            recorder.buffer.overwritten_count
        ),
        "simulation_error_count": int(recorder.simulation_error_count),
        "distributions": copy.deepcopy(summary["distributions"]),
        "translation_block_reasons": copy.deepcopy(reason_distribution),
        "curvature_strata": copy.deepcopy(stratum_distribution),
        "joint_reason_curvature": copy.deepcopy(
            total_distribution["joint_reason_curvature"]
        ),
        "target_coverage": {
            "translation_block_reasons": translation_coverage,
            "curvature_strata": curvature_coverage,
        },
        "duplicate_diagnostics": {
            "before_optional_deduplication": copy.deepcopy(
                dict(duplicate_diagnostics_before)
            ),
            "after_optional_deduplication": copy.deepcopy(
                dict(duplicate_diagnostics_after)
            ),
            "deduplication": {
                "enabled": bool(deduplication_enabled),
                "mode": (
                    "keep_first_exact_composite"
                    if deduplication_enabled
                    else "disabled"
                ),
                "removed_sample_count": int(deduplicated_sample_count),
                "exact_composite_fields": [
                    "observation",
                    "action",
                    "safety_cost",
                    "translation_block_reason_id",
                    "curvature_stratum_id",
                ],
                "near_duplicates_removed": False,
            },
        },
        "safety_cost_statistics": copy.deepcopy(
            summary["safety_cost_statistics"]
        ),
        "storage": {
            field: {
                "shape": list(state[field].shape),
                "dtype": str(state[field].dtype),
            }
            for field in (
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
        },
        "controlled_scenarios": copy.deepcopy(
            recorder.controlled_scenarios
        ),
        "controlled_scenario_status_counts": (
            _controlled_scenario_status_counts(
                recorder.controlled_scenarios
            )
        ),
        "controlled_constructions": copy.deepcopy(
            recorder.controlled_constructions
        ),
        "controlled_construction_summary": (
            _controlled_construction_summary(
                recorder.controlled_constructions
            )
        ),
        "unsupported_scenarios": copy.deepcopy(
            recorder.unsupported_scenarios
        ),
    }


def _controlled_scenario_status_counts(
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {}
    for record in records:
        scenario = str(record["scenario"])
        status = str(record["status"])
        scenario_counts = counts.setdefault(scenario, {})
        scenario_counts[status] = scenario_counts.get(status, 0) + 1
    return counts


def _controlled_construction_summary(
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for record in records:
        scenario = str(record["scenario"])
        status = str(record["status"])
        entry = output.setdefault(
            scenario,
            {
                "attempt_count": 0,
                "successful_count": 0,
                "failed_count": 0,
                "unsupported_count": 0,
                "not_observed_count": 0,
                "terminated_or_truncated_count": 0,
                "ended_before_tree_end_count": 0,
                "simulation_error_count": 0,
                "simulation_error_seeds": [],
                "successful_seeds": [],
                "failed_seeds": [],
                "unsupported_seeds": [],
                "not_observed_seeds": [],
            },
        )
        entry["attempt_count"] += 1
        status_key = f"{status}_count"
        if status_key in entry:
            entry[status_key] += 1
        seed_key = f"{status}_seeds"
        if seed_key in entry:
            entry[seed_key].append(int(record["seed"]))
        entry["terminated_or_truncated_count"] += int(
            bool(record.get("terminated", False))
            or bool(record.get("truncated", False))
        )
        entry["ended_before_tree_end_count"] += int(
            bool(record.get("ended_before_tree_end", False))
        )
        simulation_error_count = int(
            record.get("simulation_error_count", 0)
        )
        entry["simulation_error_count"] += simulation_error_count
        if simulation_error_count:
            entry["simulation_error_seeds"].append(int(record["seed"]))
    return output


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    _validate_collection_args(args)
    modes = _expanded_modes(args.modes)
    dataset_path, report_path = _resolved_output_paths(args)
    config = load_config(args.config)
    diagnostics_config = build_diagnostics_agent_config(config)
    config["diagnostics"] = copy.deepcopy(diagnostics_config)
    safety_aux_config = build_safety_aux_config(config)
    collection_capacity = int(args.capacity)
    curvature_boundaries = curvature_boundaries_from_diagnostics(config)
    max_episode_steps = int(config["environment"]["max_episode_steps"])
    planned_storage_upper_bound = _planned_storage_upper_bound(
        modes,
        lower_boundary_repetitions=int(
            args.lower_boundary_repetitions
        ),
        tree_end_repetitions=int(args.tree_end_repetitions),
        device_length_repetitions=int(
            args.device_length_repetitions
        ),
        random_episodes=int(args.random_episodes),
        curvature_episodes=int(args.curvature_episodes),
        max_curvature_transitions=int(args.max_curvature_transitions),
        max_episode_steps=max_episode_steps,
    )
    if collection_capacity < planned_storage_upper_bound:
        raise ValueError(
            "Collector capacity is too small for no-overwrite collection: "
            f"capacity={collection_capacity}, required at "
            f"least {planned_storage_upper_bound} for the selected modes"
        )

    env = make_steve_env(config["environment"])
    try:
        observation_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        buffer = SafetyAuxReplayBuffer(
            collection_capacity,
            observation_dim,
            action_dim,
            safety_cost_names=SAFETY_COST_NAMES,
            curvature_boundaries_mm_inv=curvature_boundaries,
            seed=int(args.seed_start),
        )
        recorder = CollectionRecorder(buffer, curvature_boundaries)
        seed_configuration = {
            "seed_start": int(args.seed_start),
            "random_policy_base": int(args.seed_start),
            "lower_boundary_base": int(args.seed_start + 10000),
            "vessel_tree_end": int(args.tree_end_seed),
            "device_length": int(args.device_length_seed),
            "curvature_coverage_base": int(args.seed_start + 20000),
            "split": int(args.split_seed),
        }
        magnitudes = (
            float(args.moderate_action),
            float(args.maximum_action),
        )

        if "random" in modes:
            collect_random_policy(
                env,
                recorder,
                episodes=int(args.random_episodes),
                base_seed=seed_configuration["random_policy_base"],
            )
        if "lower-boundary" in modes:
            collect_lower_boundary(
                env,
                recorder,
                base_seed=seed_configuration["lower_boundary_base"],
                repetitions=int(args.lower_boundary_repetitions),
                magnitudes=magnitudes,
            )
        if "vessel-tree-end" in modes:
            collect_tree_end(
                env,
                recorder,
                base_seed=seed_configuration["vessel_tree_end"],
                repetitions=int(args.tree_end_repetitions),
                magnitudes=magnitudes,
            )
        if "device-length" in modes:
            collect_device_length_attempt(
                env,
                recorder,
                base_seed=seed_configuration["device_length"],
                repetitions=int(args.device_length_repetitions),
            )
        curvature_targets = {
            "low": int(args.target_low),
            "medium": int(args.target_medium),
            "high": int(args.target_high),
            "extreme": int(args.target_extreme),
        }
        if "curvature-coverage" in modes:
            collect_curvature_coverage(
                env,
                recorder,
                episodes=int(args.curvature_episodes),
                base_seed=seed_configuration["curvature_coverage_base"],
                target_counts=curvature_targets,
                max_transitions=int(args.max_curvature_transitions),
            )
        else:
            recorder.curvature_coverage = {
                "requested_count": copy.deepcopy(curvature_targets),
                "count_at_coverage_start": _curvature_counts(buffer),
                "collected_count": _curvature_counts(buffer),
                "unmet_count": {
                    name: max(
                        0,
                        curvature_targets[name]
                        - _curvature_counts(buffer)[name],
                    )
                    for name in CURVATURE_STRATUM_NAMES
                },
                "transitions_examined": 0,
                "maximum_transitions": int(
                    args.max_curvature_transitions
                ),
                "episode_limit": int(args.curvature_episodes),
                "episodes_run": 0,
                "stop_reason": "mode_not_selected",
            }

        if buffer.overwritten_count:
            raise RuntimeError(
                "Safety auxiliary collection unexpectedly overwrote "
                f"{buffer.overwritten_count} transition(s)"
            )
        duplicates_before = duplicate_diagnostics(
            buffer,
            int(args.observation_round_decimals),
        )
        final_replay_state = buffer.state_dict()
        deduplicated_sample_count = 0
        if args.deduplicate_exact:
            retained_indices = exact_unique_indices(buffer)
            deduplicated_sample_count = (
                len(buffer) - int(retained_indices.size)
            )
            final_replay_state = subset_replay_state(
                buffer,
                retained_indices,
                seed=int(args.seed_start),
            )
        duplicates_after = duplicate_diagnostics(
            final_replay_state,
            int(args.observation_round_decimals),
        )
        generation_seeds = {
            "actual_by_mode": copy.deepcopy(recorder.seeds_used),
            "configuration": copy.deepcopy(seed_configuration),
        }
        dataset_state = build_dataset_state(
            final_replay_state,
            split_seed=int(args.split_seed),
            validation_fraction=float(args.validation_fraction),
            generation_seeds=generation_seeds,
            observation_round_decimals=int(
                args.observation_round_decimals
            ),
        )
        dataset = validate_dataset_state(
            dataset_state,
            expected_curvature_boundaries_mm_inv=curvature_boundaries,
        )
        translation_targets: Dict[str, Optional[int]] = {
            "none": int(args.target_none),
            "lower_insertion_boundary": int(
                args.target_lower_boundary
            ),
            "device_length_limit": None,
            "vessel_tree_end": int(args.target_tree_end),
            "other": None,
        }
        collection_parameters = {
            "lower_boundary_repetitions": int(
                args.lower_boundary_repetitions
            ),
            "tree_end_repetitions": int(args.tree_end_repetitions),
            "device_length_repetitions": int(
                args.device_length_repetitions
            ),
            "random_episodes": int(args.random_episodes),
            "curvature_episodes": int(args.curvature_episodes),
            "max_curvature_transitions": int(
                args.max_curvature_transitions
            ),
            "moderate_action": float(args.moderate_action),
            "maximum_action": float(args.maximum_action),
            "validation_fraction": float(args.validation_fraction),
            "split_seed": int(args.split_seed),
            "observation_round_decimals": int(
                args.observation_round_decimals
            ),
            "deduplicate_exact": bool(args.deduplicate_exact),
            "max_episode_steps": max_episode_steps,
            "planned_storage_upper_bound": planned_storage_upper_bound,
            "buffer_capacity": collection_capacity,
        }
        report = build_dataset_report(
            dataset,
            recorder,
            modes=modes,
            seed_configuration=seed_configuration,
            environment_config=config["environment"],
            safety_aux_config=safety_aux_config,
            collection_parameters=collection_parameters,
            duplicate_diagnostics_before=duplicates_before,
            duplicate_diagnostics_after=duplicates_after,
            deduplication_enabled=bool(args.deduplicate_exact),
            deduplicated_sample_count=deduplicated_sample_count,
            translation_targets=translation_targets,
            curvature_targets=curvature_targets,
            dataset_path=dataset_path,
            report_path=report_path,
        )
        if dataset_path is not None:
            atomic_torch_save(dataset.state_dict(), dataset_path)
        if report_path is not None:
            atomic_json_save(report, report_path)
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    finally:
        env.close()


if __name__ == "__main__":
    main()
