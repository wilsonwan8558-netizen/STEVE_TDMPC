#!/usr/bin/env python3
"""Collect a small, deterministic NVIDIA guided safety-coverage dataset.

This is an evaluation-only probe.  It constructs the NVIDIA environment via
the integration's backend factory, executes a fixed mixture of normalized
actions, validates the six-channel safety contract on every transition, and
writes aggregate diagnostics plus transition data outside the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from envs import make_env
from safety_schema import NVIDIA_GUIDED_SAFETY_COST_NAMES
from tdmpc2.common import atomic_json_save, load_config


PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PROJECT_DIR.parent
DEFAULT_CONFIG = PROJECT_DIR / "configs" / "nvidia_guided.yaml"
DEFAULT_WORKFLOW_ROOT = Path("/home/bizon/i4h-workflows")
DEFAULT_CT_CACHE = Path("/home/bizon/i4h_data/case01/cache")
DEFAULT_OUTPUT_NPZ = Path(
    "/home/bizon/i4h_outputs/case01/nvidia_safety_coverage.npz"
)
DEFAULT_OUTPUT_JSON = Path(
    "/home/bizon/i4h_outputs/case01/nvidia_safety_coverage.json"
)
EXPECTED_SCHEMA = tuple(NVIDIA_GUIDED_SAFETY_COST_NAMES)
OBSERVATION_SHAPE = (14,)
ACTION_SHAPE = (2,)
SAFETY_SHAPE = (len(EXPECTED_SCHEMA),)

STRATEGY_ORDER = (
    "aggressive_boundary_burst",
    "zero_action",
    "conservative_forward",
    "reverse_retraction",
    "positive_rotation",
    "negative_rotation",
    "random_action",
)

# Every numeric telemetry value exposed by the guided environment is retained
# in a dense matrix plus an availability mask.  The complete JSON-safe info
# mapping is also retained per transition in raw_info_json.
RAW_TELEMETRY_FIELDS = (
    "guide_s_m",
    "guide_max_root_s_m",
    "step",
    "success",
    "progress_mm",
    "velocity_cmd_m_s",
    "rotation_cmd_rad_s",
    "pre_contact_count",
    "pre_penetration_mm",
    "post_penetration_mm",
    "closest_clearance_mm",
    "max_curvature_1_m",
    "solver_curvature_1_m",
    "raw_force_proxy_n",
    "corrected_particle_count",
    "outside_count",
    "translation_error_mm",
    "excess_force_proxy_n",
    "curvature_excess_1_m",
    "hard_safety_violation",
    "is_success",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--workflow-root", type=Path, default=DEFAULT_WORKFLOW_ROOT
    )
    parser.add_argument("--ct-cache", type=Path, default=DEFAULT_CT_CACHE)
    parser.add_argument("--transitions", type=int, default=500)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--base-seed", type=int, default=41000)
    parser.add_argument("--output-npz", type=Path, default=DEFAULT_OUTPUT_NPZ)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Run only the lightweight contract/statistics self-check.",
    )
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    """Convert simulator values into strict-JSON-compatible objects."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    return repr(value)


def _git_head(repository: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _validate_output_location(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError:
        return resolved
    raise ValueError(f"Probe outputs must be outside the Git repository: {resolved}")


def _atomic_npz_save(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _channel_statistics(values: np.ndarray) -> dict[str, dict[str, Any]]:
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != len(EXPECTED_SCHEMA):
        raise ValueError(
            f"Safety statistics require shape (N, {len(EXPECTED_SCHEMA)}), "
            f"got {values.shape}"
        )
    output: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(EXPECTED_SCHEMA):
        channel = values[:, index].astype(np.float64, copy=False)
        if channel.size == 0:
            raise ValueError("Cannot calculate safety statistics for zero samples")
        nonzero_count = int(np.count_nonzero(channel > 0.0))
        percentiles = np.percentile(channel, [50.0, 90.0, 95.0, 99.0])
        output[name] = {
            "sample_count": int(channel.size),
            "nonzero_count": nonzero_count,
            "nonzero_ratio": float(nonzero_count / channel.size),
            "mean": float(np.mean(channel)),
            "standard_deviation": float(np.std(channel, ddof=0)),
            "maximum": float(np.max(channel)),
            "percentile_50": float(percentiles[0]),
            "percentile_90": float(percentiles[1]),
            "percentile_95": float(percentiles[2]),
            "percentile_99": float(percentiles[3]),
        }
    return output


def _correlation_statistics(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    variances = np.var(values, axis=0)
    matrix: list[list[float | None]] = []
    defined_pairs: list[dict[str, Any]] = []
    undefined_pairs: list[dict[str, str]] = []
    for row_index, row_name in enumerate(EXPECTED_SCHEMA):
        row: list[float | None] = []
        for column_index, column_name in enumerate(EXPECTED_SCHEMA):
            if variances[row_index] > 0.0 and variances[column_index] > 0.0:
                correlation = float(
                    np.corrcoef(
                        values[:, row_index], values[:, column_index]
                    )[0, 1]
                )
                row.append(correlation if math.isfinite(correlation) else None)
            else:
                row.append(None)
        matrix.append(row)
        for column_index in range(row_index + 1, len(EXPECTED_SCHEMA)):
            column_name = EXPECTED_SCHEMA[column_index]
            value = matrix[row_index][column_index]
            if value is None:
                undefined_pairs.append(
                    {"channel_a": row_name, "channel_b": column_name}
                )
            else:
                defined_pairs.append(
                    {
                        "channel_a": row_name,
                        "channel_b": column_name,
                        "correlation": value,
                    }
                )
    return {
        "channel_order": list(EXPECTED_SCHEMA),
        "population_variances": [float(value) for value in variances],
        "matrix": matrix,
        "defined_pairs": defined_pairs,
        "undefined_pairs": undefined_pairs,
    }


def _validate_observation(observation: Any, *, context: str) -> np.ndarray:
    value = np.asarray(observation)
    if value.shape != OBSERVATION_SHAPE:
        raise ValueError(
            f"{context}: observation shape must be {OBSERVATION_SHAPE}, "
            f"got {value.shape}"
        )
    if value.dtype != np.dtype(np.float32):
        raise TypeError(
            f"{context}: observation dtype must be float32, got {value.dtype}"
        )
    if not np.all(np.isfinite(value)):
        raise FloatingPointError(f"{context}: observation is not finite")
    return value.copy()


def _validate_safety_info(
    info: Mapping[str, Any], *, context: str
) -> tuple[np.ndarray, float]:
    if not isinstance(info, Mapping):
        raise TypeError(f"{context}: info must be a mapping")
    names = tuple(info.get("safety_cost_names", ()))
    component_order = tuple(info.get("safety_component_order", ()))
    if names != EXPECTED_SCHEMA:
        raise ValueError(
            f"{context}: safety_cost_names changed: {names!r} != {EXPECTED_SCHEMA!r}"
        )
    if component_order != EXPECTED_SCHEMA:
        raise ValueError(
            f"{context}: raw safety_component_order changed: "
            f"{component_order!r} != {EXPECTED_SCHEMA!r}"
        )
    safety = np.asarray(info.get("safety_cost"))
    if safety.shape != SAFETY_SHAPE:
        raise ValueError(
            f"{context}: safety cost shape must be {SAFETY_SHAPE}, got {safety.shape}"
        )
    if safety.dtype != np.dtype(np.float32):
        raise TypeError(
            f"{context}: safety cost dtype must be float32, got {safety.dtype}"
        )
    if not np.all(np.isfinite(safety)):
        raise FloatingPointError(f"{context}: safety cost is not finite")
    if np.any(safety < 0.0):
        raise ValueError(f"{context}: safety cost contains negative values")
    scalar_raw = info.get("safety_cost_scalar", None)
    if scalar_raw is None:
        scalar = float("nan")
    else:
        scalar = float(scalar_raw)
        if not math.isfinite(scalar) or scalar < 0.0:
            raise ValueError(
                f"{context}: scalar safety cost must be finite and nonnegative"
            )
    return safety.copy(), scalar


def _strategy_action(
    step_index: int, rng: np.random.Generator
) -> tuple[str, np.ndarray]:
    """Return one deterministic 40-step action-mixture element.

    Two boundary-valued actions form each aggressive burst.  The first burst
    after reset retracts at the lower insertion boundary; later bursts
    alternate direction.  Random actions come only from the episode-seeded
    generator.
    """

    cycle, slot = divmod(step_index, 40)
    if slot < 2:
        direction = -1.0 if cycle % 3 == 0 else 1.0
        rotation = 1.0 if slot == 0 else -1.0
        return "aggressive_boundary_burst", np.array(
            [direction, rotation], dtype=np.float32
        )
    if slot < 4:
        return "zero_action", np.zeros(2, dtype=np.float32)
    if slot < 24:
        return "conservative_forward", np.array(
            [0.65, 0.0], dtype=np.float32
        )
    if slot < 27:
        return "reverse_retraction", np.array(
            [-0.60, 0.0], dtype=np.float32
        )
    if slot < 31:
        return "positive_rotation", np.array(
            [0.0, 0.75], dtype=np.float32
        )
    if slot < 35:
        return "negative_rotation", np.array(
            [0.0, -0.75], dtype=np.float32
        )
    return "random_action", rng.uniform(-1.0, 1.0, size=2).astype(np.float32)


def _validate_action(action: np.ndarray, *, context: str) -> np.ndarray:
    value = np.asarray(action)
    if value.shape != ACTION_SHAPE:
        raise ValueError(
            f"{context}: action shape must be {ACTION_SHAPE}, got {value.shape}"
        )
    if value.dtype != np.dtype(np.float32):
        raise TypeError(f"{context}: action dtype must be float32, got {value.dtype}")
    if not np.all(np.isfinite(value)):
        raise FloatingPointError(f"{context}: action is not finite")
    if np.any(value < -1.0) or np.any(value > 1.0):
        raise ValueError(f"{context}: normalized action lies outside [-1, 1]")
    return value.copy()


def _raw_telemetry_row(info: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    values = np.full(len(RAW_TELEMETRY_FIELDS), np.nan, dtype=np.float64)
    available = np.zeros(len(RAW_TELEMETRY_FIELDS), dtype=np.bool_)
    for index, field in enumerate(RAW_TELEMETRY_FIELDS):
        if field not in info or info[field] is None:
            continue
        raw = info[field]
        if isinstance(raw, (bool, np.bool_)):
            values[index] = float(bool(raw))
            available[index] = True
        elif isinstance(raw, (int, float, np.integer, np.floating)):
            numeric = float(raw)
            values[index] = numeric
            available[index] = math.isfinite(numeric)
    return values, available


def _build_per_strategy_statistics(
    strategies: np.ndarray,
    safety_costs: np.ndarray,
    rewards: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    hard_violations: np.ndarray,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name in STRATEGY_ORDER:
        mask = strategies == name
        count = int(np.count_nonzero(mask))
        if count == 0:
            raise RuntimeError(f"Action strategy {name!r} received no samples")
        strategy_costs = safety_costs[mask]
        output[name] = {
            "transition_count": count,
            "reward_sum": float(np.sum(rewards[mask], dtype=np.float64)),
            "reward_mean": float(np.mean(rewards[mask])),
            "terminated_count": int(np.count_nonzero(terminated[mask])),
            "truncated_count": int(np.count_nonzero(truncated[mask])),
            "hard_violation_count": int(np.count_nonzero(hard_violations[mask])),
            "any_safety_nonzero_count": int(
                np.count_nonzero(np.any(strategy_costs > 0.0, axis=1))
            ),
            "channel_statistics": _channel_statistics(strategy_costs),
        }
    return output


def _coverage_assessment(channel_stats: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    channels: dict[str, Any] = {}
    controlled_curriculum: list[str] = []
    for name in EXPECTED_SCHEMA:
        nonzero_count = int(channel_stats[name]["nonzero_count"])
        nonzero_ratio = float(channel_stats[name]["nonzero_ratio"])
        if nonzero_count == 0:
            status = "absent"
        elif nonzero_count < 10 or nonzero_ratio < 0.02:
            status = "sparse"
        else:
            status = "observed"
        channels[name] = {
            "status": status,
            "nonzero_count": nonzero_count,
            "nonzero_ratio": nonzero_ratio,
        }
        if status != "observed":
            controlled_curriculum.append(name)
    return {
        "criterion": (
            "A channel is sparse when it has fewer than 10 positive samples or "
            "a positive ratio below 2%; zero positives is absent."
        ),
        "channels": channels,
        "all_channels_observed": not controlled_curriculum,
        "sufficiently_diverse_for_initial_six_channel_training": not controlled_curriculum,
        "channels_needing_controlled_curriculum": controlled_curriculum,
        "limitation": (
            "This 500-transition action probe diagnoses label coverage only; it "
            "does not establish dataset sufficiency or balanced supervision."
        ),
    }


def _synthetic_validation() -> None:
    synthetic = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.2, 0.0, 0.3, 0.4],
            [0.2, 0.0, 0.4, 0.1, 0.2, 0.8],
            [0.3, 0.0, 0.6, 0.2, 0.1, 1.0],
        ],
        dtype=np.float32,
    )
    stats = _channel_statistics(synthetic)
    correlation = _correlation_statistics(synthetic)
    assert stats["contact_force"]["nonzero_count"] == 3
    assert stats["pre_penetration"]["nonzero_count"] == 0
    assert correlation["matrix"][1][1] is None
    with tempfile.TemporaryDirectory(prefix="nvidia-safety-probe-") as directory:
        directory_path = Path(directory)
        npz_path = directory_path / "self_check.npz"
        json_path = directory_path / "self_check.json"
        _atomic_npz_save(npz_path, {"safety_cost": synthetic})
        atomic_json_save(
            {"statistics": stats, "correlation": correlation}, json_path
        )
        with np.load(npz_path, allow_pickle=False) as archive:
            assert np.array_equal(archive["safety_cost"], synthetic)
        with json_path.open("r", encoding="utf-8") as stream:
            json.load(stream)
    print("Synthetic validation: PASS", flush=True)


def _print_report(report: Mapping[str, Any]) -> None:
    aggregate = report["aggregate_statistics"]
    print("\nPer-channel safety coverage", flush=True)
    print(
        "channel                 N  nonzero    ratio       mean        std"
        "        max        p50        p90        p95        p99",
        flush=True,
    )
    for name in EXPECTED_SCHEMA:
        item = aggregate["channel_statistics"][name]
        print(
            f"{name:22s} {item['sample_count']:4d} {item['nonzero_count']:8d} "
            f"{item['nonzero_ratio']:8.3%} {item['mean']:10.6f} "
            f"{item['standard_deviation']:10.6f} {item['maximum']:10.6f} "
            f"{item['percentile_50']:10.6f} {item['percentile_90']:10.6f} "
            f"{item['percentile_95']:10.6f} {item['percentile_99']:10.6f}",
            flush=True,
        )

    termination = report["termination_statistics"]
    print("\nRun totals", flush=True)
    print(f"  total reward: {aggregate['total_reward']:.6f}", flush=True)
    print(f"  episodes started: {termination['episode_count']}", flush=True)
    print(f"  terminated transitions: {termination['terminated_count']}", flush=True)
    print(f"  truncated transitions: {termination['truncated_count']}", flush=True)
    print(f"  hard violations: {termination['hard_violation_count']}", flush=True)
    print(
        "  transitions with any nonzero safety channel: "
        f"{aggregate['any_safety_nonzero_count']}",
        flush=True,
    )

    print("\nAction-strategy counts", flush=True)
    for name in STRATEGY_ORDER:
        item = report["per_strategy_statistics"][name]
        print(
            f"  {name:27s} {item['transition_count']:4d} "
            f"(any-safety={item['any_safety_nonzero_count']:4d})",
            flush=True,
        )

    correlation = aggregate["pairwise_safety_channel_correlation"]
    print("\nPairwise safety-channel correlations", flush=True)
    for item in correlation["defined_pairs"]:
        print(
            f"  {item['channel_a']} vs {item['channel_b']}: "
            f"{item['correlation']:.6f}",
            flush=True,
        )
    for item in correlation["undefined_pairs"]:
        print(
            f"  {item['channel_a']} vs {item['channel_b']}: "
            "undefined (zero variance)",
            flush=True,
        )

    assessment = report["coverage_assessment"]
    print("\nCoverage assessment", flush=True)
    print(
        "  sufficiently diverse for all six channels: "
        f"{assessment['sufficiently_diverse_for_initial_six_channel_training']}",
        flush=True,
    )
    print(
        "  controlled-contact curriculum needed: "
        f"{assessment['channels_needing_controlled_curriculum']}",
        flush=True,
    )


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    if args.transitions < 1:
        raise ValueError("--transitions must be at least 1")
    if args.max_episode_steps < 1:
        raise ValueError("--max-episode-steps must be at least 1")

    config_path = args.config.expanduser().resolve()
    workflow_root = args.workflow_root.expanduser().resolve()
    ct_cache = args.ct_cache.expanduser().resolve()
    output_npz = _validate_output_location(args.output_npz)
    output_json = _validate_output_location(args.output_json)
    if output_npz == output_json:
        raise ValueError("NPZ and JSON output paths must differ")

    config = load_config(config_path)
    if config.get("backend") != "nvidia-guided":
        raise ValueError(
            f"Probe config must select backend 'nvidia-guided', got {config.get('backend')!r}"
        )
    configured_schema = tuple(config.get("safety_cost_names", ()))
    if configured_schema != EXPECTED_SCHEMA:
        raise ValueError(
            f"Config safety schema {configured_schema!r} != {EXPECTED_SCHEMA!r}"
        )
    if int(config.get("safety_dim", -1)) != len(EXPECTED_SCHEMA):
        raise ValueError("Config safety_dim must be exactly 6")

    environment_config = dict(config["environment"])
    environment_config.update(
        {
            "workflow_root": workflow_root,
            "ct_cache_path": ct_cache,
            "max_episode_steps": int(args.max_episode_steps),
        }
    )

    started_at = time.time()
    started_monotonic = time.perf_counter()
    print("Constructing NVIDIA guided environment via envs.make_env ...", flush=True)
    environment = make_env(environment_config, backend="nvidia-guided")
    if tuple(environment.safety_cost_names) != EXPECTED_SCHEMA:
        raise RuntimeError("Constructed environment exposes the wrong safety schema")

    observations: list[np.ndarray] = []
    next_observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    terminated_flags: list[bool] = []
    truncated_flags: list[bool] = []
    safety_costs: list[np.ndarray] = []
    safety_scalars: list[float] = []
    hard_violations: list[bool] = []
    episode_indices: list[int] = []
    step_indices: list[int] = []
    episode_seeds: list[int] = []
    action_strategies: list[str] = []
    safety_names_per_transition: list[tuple[str, ...]] = []
    raw_telemetry_values: list[np.ndarray] = []
    raw_telemetry_available: list[np.ndarray] = []
    raw_info_json: list[str] = []
    navigation_modes: list[str] = []
    seeds_used: list[int] = []

    episode_index = 0
    step_index = 0
    current_seed = int(args.base_seed)
    episode_rng = np.random.default_rng(current_seed)
    seeds_used.append(current_seed)

    try:
        observation, reset_info = environment.reset(seed=current_seed)
        observation = _validate_observation(observation, context="reset")
        _validate_safety_info(reset_info, context="reset")

        while len(rewards) < args.transitions:
            strategy, action = _strategy_action(step_index, episode_rng)
            action = _validate_action(action, context=f"transition {len(rewards)}")
            next_observation, reward, terminated, truncated, info = environment.step(
                action
            )
            context = f"transition {len(rewards)}"
            next_observation = _validate_observation(
                next_observation, context=context
            )
            safety_cost, safety_scalar = _validate_safety_info(
                info, context=context
            )
            finite_reward = float(reward)
            if not math.isfinite(finite_reward):
                raise FloatingPointError(f"{context}: reward is not finite")
            hard_violation = bool(
                info.get("hard_violation", info.get("hard_safety_violation", False))
            )
            raw_values, raw_available = _raw_telemetry_row(info)

            observations.append(observation)
            next_observations.append(next_observation)
            actions.append(action)
            rewards.append(finite_reward)
            terminated_flags.append(bool(terminated))
            truncated_flags.append(bool(truncated))
            safety_costs.append(safety_cost)
            safety_scalars.append(safety_scalar)
            hard_violations.append(hard_violation)
            episode_indices.append(episode_index)
            step_indices.append(step_index)
            episode_seeds.append(current_seed)
            action_strategies.append(strategy)
            safety_names_per_transition.append(EXPECTED_SCHEMA)
            raw_telemetry_values.append(raw_values)
            raw_telemetry_available.append(raw_available)
            raw_info_json.append(
                json.dumps(
                    _json_safe(info), sort_keys=True, separators=(",", ":"), allow_nan=False
                )
            )
            navigation_modes.append(str(info.get("navigation_mode", "")))

            collected = len(rewards)
            if collected % 50 == 0 or collected == args.transitions:
                print(
                    f"Collected {collected}/{args.transitions} transitions "
                    f"(episode={episode_index}, step={step_index}, "
                    f"hard={sum(hard_violations)})",
                    flush=True,
                )

            observation = next_observation
            step_index += 1
            if bool(terminated) or bool(truncated) or hard_violation:
                episode_index += 1
                step_index = 0
                current_seed = int(args.base_seed) + episode_index
                episode_rng = np.random.default_rng(current_seed)
                if collected < args.transitions:
                    seeds_used.append(current_seed)
                    observation, reset_info = environment.reset(seed=current_seed)
                    observation = _validate_observation(
                        observation, context=f"reset episode {episode_index}"
                    )
                    _validate_safety_info(
                        reset_info, context=f"reset episode {episode_index}"
                    )
    finally:
        environment.close()

    observation_array = np.stack(observations).astype(np.float32, copy=False)
    next_observation_array = np.stack(next_observations).astype(
        np.float32, copy=False
    )
    action_array = np.stack(actions).astype(np.float32, copy=False)
    reward_array = np.asarray(rewards, dtype=np.float64)
    terminated_array = np.asarray(terminated_flags, dtype=np.bool_)
    truncated_array = np.asarray(truncated_flags, dtype=np.bool_)
    safety_array = np.stack(safety_costs).astype(np.float32, copy=False)
    scalar_array = np.asarray(safety_scalars, dtype=np.float32)
    hard_array = np.asarray(hard_violations, dtype=np.bool_)
    episode_index_array = np.asarray(episode_indices, dtype=np.int64)
    step_index_array = np.asarray(step_indices, dtype=np.int64)
    seed_array = np.asarray(episode_seeds, dtype=np.int64)
    strategy_array = np.asarray(action_strategies, dtype=np.str_)
    raw_values_array = np.stack(raw_telemetry_values)
    raw_available_array = np.stack(raw_telemetry_available)

    if observation_array.shape != (args.transitions, *OBSERVATION_SHAPE):
        raise AssertionError(f"Unexpected observation dataset shape {observation_array.shape}")
    if action_array.shape != (args.transitions, *ACTION_SHAPE):
        raise AssertionError(f"Unexpected action dataset shape {action_array.shape}")
    if safety_array.shape != (args.transitions, *SAFETY_SHAPE):
        raise AssertionError(f"Unexpected safety dataset shape {safety_array.shape}")
    if safety_array.dtype != np.dtype(np.float32):
        raise AssertionError(f"Unexpected safety dataset dtype {safety_array.dtype}")
    if not np.all(np.isfinite(safety_array)) or np.any(safety_array < 0.0):
        raise AssertionError("Final safety dataset is non-finite or negative")
    if not np.all(np.isfinite(action_array)):
        raise AssertionError("Final action dataset is non-finite")

    channel_stats = _channel_statistics(safety_array)
    per_strategy = _build_per_strategy_statistics(
        strategy_array,
        safety_array,
        reward_array,
        terminated_array,
        truncated_array,
        hard_array,
    )
    episode_count = int(np.max(episode_index_array)) + 1
    episode_lengths = Counter(int(value) for value in episode_index_array)
    duration_seconds = float(time.perf_counter() - started_monotonic)
    aggregate = {
        "transition_count": int(args.transitions),
        "total_reward": float(np.sum(reward_array, dtype=np.float64)),
        "mean_reward": float(np.mean(reward_array)),
        "any_safety_nonzero_count": int(
            np.count_nonzero(np.any(safety_array > 0.0, axis=1))
        ),
        "any_safety_nonzero_ratio": float(
            np.mean(np.any(safety_array > 0.0, axis=1))
        ),
        "channel_statistics": channel_stats,
        "pairwise_safety_channel_correlation": _correlation_statistics(
            safety_array
        ),
    }
    termination_stats = {
        "episode_count": episode_count,
        "completed_episode_count": int(
            np.count_nonzero(terminated_array | truncated_array | hard_array)
        ),
        "terminated_count": int(np.count_nonzero(terminated_array)),
        "truncated_count": int(np.count_nonzero(truncated_array)),
        "hard_violation_count": int(np.count_nonzero(hard_array)),
        "success_count": int(
            sum(json.loads(value).get("is_success", False) for value in raw_info_json)
        ),
        "episode_transition_counts": {
            str(key): int(value) for key, value in sorted(episode_lengths.items())
        },
        "reset_rule": "terminated or truncated or hard_safety_violation",
    }
    assessment = _coverage_assessment(channel_stats)

    npz_arrays = {
        "observations": observation_array,
        "next_observations": next_observation_array,
        "actions": action_array,
        "rewards": reward_array,
        "terminated": terminated_array,
        "truncated": truncated_array,
        "safety_cost": safety_array,
        "safety_cost_scalar": scalar_array,
        "safety_cost_names": np.asarray(EXPECTED_SCHEMA, dtype=np.str_),
        "safety_cost_names_per_transition": np.asarray(
            safety_names_per_transition, dtype=np.str_
        ),
        "hard_violation": hard_array,
        "episode_index": episode_index_array,
        "step_index": step_index_array,
        "episode_seed": seed_array,
        "action_strategy": strategy_array,
        "raw_telemetry_field_names": np.asarray(
            RAW_TELEMETRY_FIELDS, dtype=np.str_
        ),
        "raw_telemetry_values": raw_values_array,
        "raw_telemetry_available": raw_available_array,
        "raw_info_json": np.asarray(raw_info_json, dtype=np.str_),
        "navigation_mode": np.asarray(navigation_modes, dtype=np.str_),
    }
    _atomic_npz_save(output_npz, npz_arrays)

    report: dict[str, Any] = {
        "format_version": 1,
        "schema": {
            "safety_cost_names": list(EXPECTED_SCHEMA),
            "safety_dim": len(EXPECTED_SCHEMA),
            "observation_shape": list(OBSERVATION_SHAPE),
            "action_shape": list(ACTION_SHAPE),
            "safety_dtype": "float32",
            "channel_order_validated_every_transition": True,
        },
        "runtime_configuration": {
            "started_unix_seconds": float(started_at),
            "duration_seconds": duration_seconds,
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "package_versions": {
                "numpy": np.__version__,
                "gymnasium": _package_version("gymnasium"),
                "torch": _package_version("torch"),
                "warp-lang": _package_version("warp-lang"),
            },
            "config_path": str(config_path),
            "config_sha256": _file_sha256(config_path),
            "workflow_root": str(workflow_root),
            "ct_cache_path": str(ct_cache),
            "backend": "nvidia-guided",
            "transitions": int(args.transitions),
            "max_episode_steps": int(args.max_episode_steps),
            "base_seed": int(args.base_seed),
            "environment_config": _json_safe(environment_config),
            "action_schedule": {
                "period_steps": 40,
                "strategy_order": list(STRATEGY_ORDER),
                "description": (
                    "Two-step normalized-boundary bursts, zeros, forward "
                    "insertion, retraction, isolated signed rotation, and "
                    "episode-seeded uniform random actions."
                ),
            },
            "repository_root": str(REPOSITORY_ROOT),
            "repository_commit": _git_head(REPOSITORY_ROOT),
            "nvidia_repository_commit": _git_head(workflow_root),
            "environment_variables": {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            },
            "command_argv": [sys.executable, *sys.argv],
        },
        "seeds": {
            "base_seed": int(args.base_seed),
            "episode_seed_rule": "base_seed + zero_based_episode_index",
            "ordered_episode_seeds": seeds_used,
            "per_transition_seed_saved_in_npz": True,
        },
        "aggregate_statistics": aggregate,
        "per_strategy_statistics": per_strategy,
        "termination_statistics": termination_stats,
        "coverage_assessment": assessment,
        "outputs": {
            "npz": str(output_npz),
            "json": str(output_json),
        },
        "npz_fields": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in npz_arrays.items()
        },
    }
    atomic_json_save(report, output_json)

    # Verify that both final artifacts are independently readable without
    # pickle and that their on-disk schema agrees with the in-memory result.
    with np.load(output_npz, allow_pickle=False) as archive:
        if tuple(archive["safety_cost_names"].tolist()) != EXPECTED_SCHEMA:
            raise RuntimeError("Saved NPZ safety schema failed round-trip validation")
        if archive["safety_cost"].shape != safety_array.shape:
            raise RuntimeError("Saved NPZ safety tensor failed round-trip validation")
    with output_json.open("r", encoding="utf-8") as stream:
        saved_report = json.load(stream)
    if tuple(saved_report["schema"]["safety_cost_names"]) != EXPECTED_SCHEMA:
        raise RuntimeError("Saved JSON safety schema failed round-trip validation")

    _print_report(report)
    print(f"\nSaved NPZ: {output_npz}", flush=True)
    print(f"Saved JSON: {output_json}", flush=True)
    print(f"Runtime: {duration_seconds:.3f} seconds", flush=True)
    return report


def main() -> int:
    args = parse_args()
    _synthetic_validation()
    if args.synthetic_only:
        return 0
    run_probe(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
