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


DEFAULT_CONFIG = PROJECT_DIR / "configs" / "steve.yaml"
DEFAULT_DATASET = Path("/tmp/steve_safety_aux_dataset.pt")
DEFAULT_REPORT = Path("/tmp/steve_safety_aux_report.json")
DEFAULT_TREE_END_SEED = 301
DEFAULT_DEVICE_LENGTH_SEED = 301
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
    parser.add_argument("--random-episodes", type=int, default=2)
    parser.add_argument("--curvature-episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--tree-end-seed", type=int, default=DEFAULT_TREE_END_SEED)
    parser.add_argument(
        "--device-length-seed",
        type=int,
        default=DEFAULT_DEVICE_LENGTH_SEED,
    )
    parser.add_argument("--moderate-action", type=float, default=0.5)
    parser.add_argument("--maximum-action", type=float, default=1.0)
    parser.add_argument(
        "--capacity",
        type=int,
        default=None,
        help="Override safety_aux.capacity",
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
        self.unsupported_scenarios: List[Dict[str, Any]] = []

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
    ) -> Tuple[int, int]:
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
        self.buffer.add(
            observation,
            action,
            cost,
            reason_id,
            stratum_id,
            terminated=terminated,
            truncated=truncated,
            episode_step=int(info["episode_step"]),
        )
        return reason_id, stratum_id


def _expanded_modes(values: Sequence[str]) -> Tuple[str, ...]:
    requested = tuple(values)
    if "all" in requested:
        return COLLECTION_MODES
    # Preserve CLI order while eliminating accidental duplicates.
    return tuple(dict.fromkeys(requested))


def _validate_collection_args(args: argparse.Namespace) -> None:
    for name in ("random_episodes", "curvature_episodes"):
        value = getattr(args, name)
        if isinstance(value, bool) or value < 0:
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


def _planned_storage_upper_bound(
    modes: Sequence[str],
    *,
    random_episodes: int,
    curvature_episodes: int,
    max_episode_steps: int,
) -> int:
    """Return a conservative no-overwrite capacity requirement."""

    selected = set(modes)
    maximum = 0
    if "random" in selected:
        maximum += int(random_episodes) * int(max_episode_steps)
    if "curvature-coverage" in selected:
        maximum += int(curvature_episodes) * int(max_episode_steps)
    if "lower-boundary" in selected:
        maximum += 2
    if "vessel-tree-end" in selected:
        maximum += 2
    if "device-length" in selected:
        maximum += 1
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


def collect_random_policy(
    env: StEVEEnv,
    recorder: CollectionRecorder,
    *,
    episodes: int,
    base_seed: int,
) -> None:
    for episode_index in range(episodes):
        episode_seed = int(base_seed + episode_index)
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
    magnitudes: Sequence[float],
) -> None:
    expected_id = TRANSLATION_BLOCK_REASON_NAMES.index(
        "lower_insertion_boundary"
    )
    for index, magnitude in enumerate(magnitudes):
        seed = int(base_seed)
        observation, _ = env.reset(seed=seed)
        action = np.asarray([-float(magnitude), 0.0], dtype=np.float32)
        _, _, terminated, truncated, info = _step(env, action, recorder)
        reason_id, _ = recorder.store(
            observation,
            action,
            info,
            terminated=terminated,
            truncated=truncated,
        )
        metrics = info["safety_metrics"]
        recorder.controlled_scenarios.append(
            _controlled_record(
                scenario="lower_insertion_boundary",
                variant="moderate" if index == 0 else "maximum",
                seed=seed,
                status="observed" if reason_id == expected_id else "not_observed",
                approach_steps=0,
                action=action,
                metrics=metrics,
                safety_cost=info["safety_cost"],
                stored=True,
            )
        )


def collect_tree_end(
    env: StEVEEnv,
    recorder: CollectionRecorder,
    *,
    seed: int,
    magnitudes: Sequence[float],
) -> None:
    expected_id = TRANSLATION_BLOCK_REASON_NAMES.index("vessel_tree_end")
    approach_action = np.asarray([1.0, 0.0], dtype=np.float32)
    for index, magnitude in enumerate(magnitudes):
        observation, _ = env.reset(seed=seed)
        terminated = truncated = False
        approach_steps = 0
        while not _at_tree_end(env) and not (terminated or truncated):
            (
                observation,
                _,
                terminated,
                truncated,
                _,
            ) = _step(env, approach_action, recorder)
            approach_steps += 1
        if terminated or truncated or not _at_tree_end(env):
            record = {
                "scenario": "vessel_tree_end",
                "variant": "moderate" if index == 0 else "maximum",
                "seed": int(seed),
                "status": "not_observed",
                "approach_steps": int(approach_steps),
                "reason": (
                    "Episode ended before the fixed-tree endpoint was reached."
                ),
                "stored": False,
            }
            recorder.controlled_scenarios.append(record)
            continue

        action = np.asarray([float(magnitude), 0.0], dtype=np.float32)
        _, _, terminated, truncated, info = _step(env, action, recorder)
        reason_id, _ = recorder.store(
            observation,
            action,
            info,
            terminated=terminated,
            truncated=truncated,
        )
        recorder.controlled_scenarios.append(
            _controlled_record(
                scenario="vessel_tree_end",
                variant="moderate" if index == 0 else "maximum",
                seed=seed,
                status="observed" if reason_id == expected_id else "not_observed",
                approach_steps=approach_steps,
                action=action,
                metrics=info["safety_metrics"],
                safety_cost=info["safety_cost"],
                stored=True,
            )
        )


def collect_device_length_attempt(
    env: StEVEEnv,
    recorder: CollectionRecorder,
    *,
    seed: int,
) -> None:
    expected_id = TRANSLATION_BLOCK_REASON_NAMES.index("device_length_limit")
    action = np.asarray([1.0, 0.0], dtype=np.float32)
    observation, _ = env.reset(seed=seed)
    terminated = truncated = False
    approach_steps = 0
    observed_info: Optional[Mapping[str, Any]] = None
    while not (terminated or truncated):
        (
            next_observation,
            _,
            terminated,
            truncated,
            info,
        ) = _step(env, action, recorder)
        approach_steps += 1
        metrics = info["safety_metrics"]
        if int(metrics["translation_block_reason_id"]) != 0:
            observed_info = info
            if int(metrics["translation_block_reason_id"]) == expected_id:
                recorder.store(
                    observation,
                    action,
                    info,
                    terminated=terminated,
                    truncated=truncated,
                )
            break
        observation = next_observation

    if (
        observed_info is not None
        and int(
            observed_info["safety_metrics"]["translation_block_reason_id"]
        )
        == expected_id
    ):
        recorder.controlled_scenarios.append(
            _controlled_record(
                scenario="device_length_limit",
                variant="maximum",
                seed=seed,
                status="observed",
                approach_steps=approach_steps - 1,
                action=action,
                metrics=observed_info["safety_metrics"],
                safety_cost=observed_info["safety_cost"],
                stored=True,
            )
        )
        return

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
    if first_block_reason == "vessel_tree_end":
        unsupported = {
            "scenario": "device_length_limit",
            "variant": "maximum",
            "status": "unsupported",
            "seed": int(seed),
            "approach_steps": int(max(0, approach_steps - 1)),
            "first_block_reason": first_block_reason,
            "inserted_length_mm": inserted_length,
            "device_maximum_length_mm": device_maximum_length,
            "reason": (
                "The unchanged fixed vessel reached its tree endpoint at "
                f"{inserted_length:.6g} mm before the device-length limit of "
                f"{device_maximum_length:.6g} mm; no sample was fabricated."
            ),
            "stored": False,
        }
        recorder.unsupported_scenarios.append(unsupported)
        recorder.controlled_scenarios.append(copy.deepcopy(unsupported))
        return

    reason = (
        "The episode terminated or truncated before any translation blocker "
        "was observed; physical support for device-length blockage was not "
        "established."
        if observed_info is None
        else (
            f"The first observed translation blocker was {first_block_reason!r}, "
            "not the device-length limit; no sample was fabricated."
        )
    )
    not_observed: Dict[str, Any] = {
        "scenario": "device_length_limit",
        "variant": "maximum",
        "status": "not_observed",
        "seed": int(seed),
        "approach_steps": int(
            max(0, approach_steps - 1)
            if observed_info is not None
            else approach_steps
        ),
        "first_block_reason": first_block_reason,
        "inserted_length_mm": inserted_length,
        "device_maximum_length_mm": device_maximum_length,
        "reason": reason,
        "stored": False,
    }
    if observed_info is not None:
        not_observed.update(
            _controlled_record(
                scenario="device_length_limit",
                variant="maximum",
                seed=seed,
                status="not_observed",
                approach_steps=max(0, approach_steps - 1),
                action=action,
                metrics=observed_info["safety_metrics"],
                safety_cost=observed_info["safety_cost"],
                stored=False,
            )
        )
        not_observed["first_block_reason"] = first_block_reason
        not_observed["device_maximum_length_mm"] = device_maximum_length
        not_observed["reason"] = reason
    recorder.controlled_scenarios.append(not_observed)


def collect_curvature_coverage(
    env: StEVEEnv,
    recorder: CollectionRecorder,
    *,
    episodes: int,
    base_seed: int,
) -> None:
    for episode_index in range(episodes):
        episode_seed = int(base_seed + episode_index)
        rng = np.random.default_rng(episode_seed)
        observation, _ = env.reset(seed=episode_seed)
        terminated = truncated = False
        step_index = 0
        while not (terminated or truncated):
            # Forward-biased insertion plus reproducible rotation explores
            # geometry without assigning or enforcing any clinical threshold.
            translation = rng.uniform(0.25, 1.0)
            rotation = np.clip(
                0.75 * np.sin(0.35 * step_index)
                + 0.25 * rng.uniform(-1.0, 1.0),
                -1.0,
                1.0,
            )
            action = np.asarray(
                [translation, rotation],
                dtype=np.float32,
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


def _controlled_record(
    *,
    scenario: str,
    variant: str,
    seed: int,
    status: str,
    approach_steps: int,
    action: np.ndarray,
    metrics: Mapping[str, Any],
    safety_cost: np.ndarray,
    stored: bool,
) -> Dict[str, Any]:
    return {
        "scenario": scenario,
        "variant": variant,
        "seed": int(seed),
        "status": status,
        "approach_steps": int(approach_steps),
        "normalized_action": np.asarray(action, dtype=np.float32).tolist(),
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
        "inserted_length_mm": float(metrics["inserted_length_mm"]),
        "safety_cost": {
            name: float(safety_cost[index])
            for index, name in enumerate(SAFETY_COST_NAMES)
        },
        "stored": bool(stored),
    }


def build_dataset_report(
    buffer: SafetyAuxReplayBuffer,
    recorder: CollectionRecorder,
    *,
    modes: Sequence[str],
    seeds: Mapping[str, Any],
    environment_config: Mapping[str, Any],
    safety_aux_config: Mapping[str, Any],
    curvature_boundaries: Sequence[float],
    collection_parameters: Mapping[str, Any],
    dataset_path: Optional[Path],
    report_path: Optional[Path],
) -> Dict[str, Any]:
    state = buffer.state_dict()
    count = len(buffer)
    reason_ids = state["translation_block_reason_id"]
    stratum_ids = state["curvature_stratum_id"]
    reason_counts = np.bincount(
        reason_ids,
        minlength=len(TRANSLATION_BLOCK_REASON_NAMES),
    )
    stratum_counts = np.bincount(
        stratum_ids,
        minlength=len(CURVATURE_STRATUM_NAMES),
    )
    return {
        "schema_version": 1,
        "auxiliary_buffer_schema_version": (
            SAFETY_AUX_REPLAY_SCHEMA_VERSION
        ),
        "dataset_path": str(dataset_path) if dataset_path is not None else None,
        "report_path": str(report_path) if report_path is not None else None,
        "modes": list(modes),
        "collection_parameters": copy.deepcopy(
            dict(collection_parameters)
        ),
        "random_seeds": dict(seeds),
        "environment_configuration": copy.deepcopy(
            dict(environment_config)
        ),
        "safety_aux_configuration": copy.deepcopy(
            dict(safety_aux_config)
        ),
        "curvature_boundaries_mm_inv": [
            float(value) for value in curvature_boundaries
        ],
        "executed_transition_count": int(
            recorder.executed_transition_count
        ),
        "total_transition_count": int(count),
        "total_added": int(buffer.total_added),
        "overwritten_transition_count": int(buffer.overwritten_count),
        "simulation_error_count": int(recorder.simulation_error_count),
        "translation_block_reasons": {
            name: {
                "id": index,
                "count": int(reason_counts[index]),
                "fraction": (
                    float(reason_counts[index] / count) if count else 0.0
                ),
            }
            for index, name in enumerate(
                TRANSLATION_BLOCK_REASON_NAMES
            )
        },
        "curvature_strata": {
            name: {
                "id": index,
                "count": int(stratum_counts[index]),
                "fraction": (
                    float(stratum_counts[index] / count) if count else 0.0
                ),
            }
            for index, name in enumerate(CURVATURE_STRATUM_NAMES)
        },
        "safety_cost_statistics": _safety_cost_statistics(
            state["safety_cost"]
        ),
        "storage": {
            "observation": {
                "shape": list(state["observation"].shape),
                "dtype": str(state["observation"].dtype),
            },
            "action": {
                "shape": list(state["action"].shape),
                "dtype": str(state["action"].dtype),
            },
            "safety_cost": {
                "shape": list(state["safety_cost"].shape),
                "dtype": str(state["safety_cost"].dtype),
            },
            "translation_block_reason_id": {
                "shape": list(reason_ids.shape),
                "dtype": str(reason_ids.dtype),
            },
            "curvature_stratum_id": {
                "shape": list(stratum_ids.shape),
                "dtype": str(stratum_ids.dtype),
            },
        },
        "controlled_scenarios": copy.deepcopy(
            recorder.controlled_scenarios
        ),
        "controlled_scenario_status_counts": (
            _controlled_scenario_status_counts(
                recorder.controlled_scenarios
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


def _safety_cost_statistics(values: np.ndarray) -> Dict[str, Any]:
    costs = np.asarray(values)
    if costs.ndim != 2 or costs.shape[1] != len(SAFETY_COST_NAMES):
        raise ValueError("Safety-cost report data has an invalid shape")
    output: Dict[str, Any] = {}
    for index, name in enumerate(SAFETY_COST_NAMES):
        channel = costs[:, index].astype(np.float64, copy=False)
        if channel.size == 0:
            output[name] = {
                key: None
                for key in (
                    "min",
                    "max",
                    "mean",
                    "median",
                    "p05",
                    "p25",
                    "p75",
                    "p95",
                    "p99",
                )
            }
            continue
        output[name] = {
            "min": float(np.min(channel)),
            "max": float(np.max(channel)),
            "mean": float(np.mean(channel)),
            "median": float(np.median(channel)),
            "p05": float(np.percentile(channel, 5)),
            "p25": float(np.percentile(channel, 25)),
            "p75": float(np.percentile(channel, 75)),
            "p95": float(np.percentile(channel, 95)),
            "p99": float(np.percentile(channel, 99)),
        }
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
    if args.capacity is not None:
        safety_aux_config["capacity"] = int(args.capacity)
    curvature_boundaries = curvature_boundaries_from_diagnostics(config)
    max_episode_steps = int(config["environment"]["max_episode_steps"])
    planned_storage_upper_bound = _planned_storage_upper_bound(
        modes,
        random_episodes=int(args.random_episodes),
        curvature_episodes=int(args.curvature_episodes),
        max_episode_steps=max_episode_steps,
    )
    if int(safety_aux_config["capacity"]) < planned_storage_upper_bound:
        raise ValueError(
            "Safety auxiliary replay capacity is too small for no-overwrite "
            f"collection: capacity={safety_aux_config['capacity']}, required at "
            f"least {planned_storage_upper_bound} for the selected modes"
        )

    env = make_steve_env(config["environment"])
    try:
        observation_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        buffer = SafetyAuxReplayBuffer(
            int(safety_aux_config["capacity"]),
            observation_dim,
            action_dim,
            safety_cost_names=SAFETY_COST_NAMES,
            curvature_boundaries_mm_inv=curvature_boundaries,
            seed=int(args.seed),
        )
        recorder = CollectionRecorder(buffer, curvature_boundaries)
        seeds = {
            "base": int(args.seed),
            "random_policy": int(args.seed),
            "lower_boundary": int(args.seed + 10000),
            "vessel_tree_end": int(args.tree_end_seed),
            "device_length": int(args.device_length_seed),
            "curvature_coverage": int(args.seed + 20000),
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
                base_seed=seeds["random_policy"],
            )
        if "lower-boundary" in modes:
            collect_lower_boundary(
                env,
                recorder,
                base_seed=seeds["lower_boundary"],
                magnitudes=magnitudes,
            )
        if "vessel-tree-end" in modes:
            collect_tree_end(
                env,
                recorder,
                seed=seeds["vessel_tree_end"],
                magnitudes=magnitudes,
            )
        if "device-length" in modes:
            collect_device_length_attempt(
                env,
                recorder,
                seed=seeds["device_length"],
            )
        if "curvature-coverage" in modes:
            collect_curvature_coverage(
                env,
                recorder,
                episodes=int(args.curvature_episodes),
                base_seed=seeds["curvature_coverage"],
            )

        if buffer.overwritten_count:
            raise RuntimeError(
                "Safety auxiliary collection unexpectedly overwrote "
                f"{buffer.overwritten_count} transition(s)"
            )
        collection_parameters = {
            "random_episodes": int(args.random_episodes),
            "curvature_episodes": int(args.curvature_episodes),
            "moderate_action": float(args.moderate_action),
            "maximum_action": float(args.maximum_action),
            "max_episode_steps": max_episode_steps,
            "planned_storage_upper_bound": planned_storage_upper_bound,
            "buffer_capacity": int(safety_aux_config["capacity"]),
        }
        report = build_dataset_report(
            buffer,
            recorder,
            modes=modes,
            seeds=seeds,
            environment_config=config["environment"],
            safety_aux_config=safety_aux_config,
            curvature_boundaries=curvature_boundaries,
            collection_parameters=collection_parameters,
            dataset_path=dataset_path,
            report_path=report_path,
        )
        if dataset_path is not None:
            atomic_torch_save(buffer.state_dict(), dataset_path)
        if report_path is not None:
            atomic_json_save(report, report_path)
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    finally:
        env.close()


if __name__ == "__main__":
    main()
