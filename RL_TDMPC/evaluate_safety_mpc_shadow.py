#!/usr/bin/env python3
"""Evaluate Translation-only soft Safety MPPI without executing shadow actions.

The evaluator has one strict control rule: only the unchanged production
TD-MPC2 action may be sent to stEVE on policy trajectories.  In random and
controlled-boundary modes, the predefined behavior command is sent instead.
Every Safety-aware action is counterfactual planner output used only for
analysis.

Translation Safety means predicted controller action-feasibility risk: the
intervention may modify or block a requested insertion/retraction command.
It is not contact force, vessel injury, guidewire slippage, or a clinical
safety measure.  The exact intervention action mask remains the final
execution-layer protection.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import random
import resource
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from unittest import mock

import numpy as np
import torch

from collect_safety_dataset import _at_tree_end
from envs.steve_env import StEVEEnv, make_steve_env
from evaluate_safety_head import load_evaluation_model
from evaluate_safety_rollout import strict_json_dumps, write_strict_json
from tdmpc2.common import set_seed
from tdmpc2.shadow_planner import (
    SUPPORTED_AGGREGATIONS,
    PlannerSnapshot,
    ShadowPlanTrace,
    capture_device_torch_rng_state,
    capture_planner_snapshot,
    compare_shadow_trace,
    make_planner_noise_schedule,
    run_translation_shadow_plan,
    run_translation_shadow_sweep,
)


DEFAULT_CHECKPOINTS = (
    Path(
        "/tmp/steve_commit46e_robustness/runs/seed_1/"
        "checkpoints/step_3000.pt"
    ),
    Path(
        "/tmp/steve_commit46e_robustness/runs/seed_11/"
        "checkpoints/step_3000.pt"
    ),
    Path(
        "/tmp/steve_commit46e_robustness/runs/seed_21/"
        "checkpoints/step_3000.pt"
    ),
)
DEFAULT_WEIGHTS = (0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5)
DEFAULT_SEEDS = (41000, 51000, 61000, 71000)
DEFAULT_OUTPUT_ROOT = Path("/tmp/steve_commit47b_shadow")
TRANSLATION_INDEX = 1


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        default=list(DEFAULT_CHECKPOINTS),
        help=(
            "Strict step checkpoints (default: the seed 1/11/21 robustness "
            "step-3000 checkpoints)"
        ),
    )
    parser.add_argument("--policy-episodes", type=int, default=5)
    parser.add_argument("--random-episodes", type=int, default=5)
    parser.add_argument("--lower-boundary-repetitions", type=int, default=20)
    parser.add_argument("--tree-end-repetitions", type=int, default=20)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs=4,
        metavar=("POLICY", "RANDOM", "LOWER", "TREE"),
        default=list(DEFAULT_SEEDS),
        help="Four disjoint base seeds, reused identically for every checkpoint",
    )
    parser.add_argument(
        "--planner-seed",
        type=int,
        default=91000,
        help="Base private planner-noise seed for non-policy behavior states",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=list(DEFAULT_WEIGHTS),
    )
    parser.add_argument(
        "--aggregation",
        choices=list(SUPPORTED_AGGREGATIONS),
        nargs="+",
        default=list(SUPPORTED_AGGREGATIONS),
    )
    parser.add_argument("--safety-discount", type=float, default=1.0)
    parser.add_argument("--translation-risk-cap", type=float, default=1.0)
    parser.add_argument(
        "--diagnostic-threshold",
        type=float,
        default=0.2,
        help="Reporting threshold only; never used for planner scoring",
    )
    parser.add_argument(
        "--translation-action-tolerance",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--rotation-action-tolerance",
        type=float,
        default=1e-3,
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "report.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "REPORT.md",
    )
    parser.add_argument(
        "--details-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "details",
        help="Compressed per-planning-call numeric records",
    )
    parser.add_argument(
        "--max-planning-calls-per-checkpoint",
        type=int,
        default=None,
        help=(
            "Developer-only early stop for a quick pipeline check. Omit for "
            "the required complete evaluation."
        ),
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if not args.checkpoints:
        raise ValueError("At least one checkpoint is required")
    missing = [str(path) for path in args.checkpoints if not path.expanduser().exists()]
    if missing:
        raise FileNotFoundError(
            "Expected checkpoint(s) unavailable: " + ", ".join(missing)
        )
    for name in (
        "policy_episodes",
        "random_episodes",
        "lower_boundary_repetitions",
        "tree_end_repetitions",
    ):
        if int(getattr(args, name)) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be nonnegative")
    if args.max_planning_calls_per_checkpoint is not None and (
        int(args.max_planning_calls_per_checkpoint) <= 0
    ):
        raise ValueError("--max-planning-calls-per-checkpoint must be positive")
    weights = tuple(float(value) for value in args.weights)
    if (
        not weights
        or any(not math.isfinite(value) or value < 0.0 for value in weights)
        or len(set(weights)) != len(weights)
    ):
        raise ValueError("--weights must be unique finite nonnegative values")
    if 0.0 not in weights:
        raise ValueError("--weights must include 0.0 for baseline identity")
    if len(set(args.aggregation)) != len(args.aggregation):
        raise ValueError("--aggregation entries must be unique")
    if len(set(int(value) for value in args.seeds)) != 4:
        raise ValueError("--seeds must contain four disjoint values")
    for value, name, lower, upper in (
        (args.safety_discount, "--safety-discount", 0.0, 1.0),
        (args.diagnostic_threshold, "--diagnostic-threshold", 0.0, None),
        (
            args.translation_action_tolerance,
            "--translation-action-tolerance",
            0.0,
            None,
        ),
        (
            args.rotation_action_tolerance,
            "--rotation-action-tolerance",
            0.0,
            None,
        ),
    ):
        parsed = float(value)
        if (
            not math.isfinite(parsed)
            or parsed < lower
            or (upper is not None and parsed > upper)
        ):
            raise ValueError(f"{name} has an invalid value")
    if (
        not math.isfinite(float(args.translation_risk_cap))
        or float(args.translation_risk_cap) <= 0.0
    ):
        raise ValueError("--translation-risk-cap must be finite and positive")


def _checkpoint_label(path: Path) -> str:
    resolved = path.expanduser().resolve()
    parts = resolved.parts
    seed = next(
        (part for part in reversed(parts) if part.startswith("seed_")),
        resolved.parent.parent.name,
    )
    return f"{seed}_{resolved.stem}"


def _numpy_legacy_state_bytes() -> bytes:
    return pickle.dumps(np.random.get_state(), protocol=pickle.HIGHEST_PROTOCOL)


def _generator_state_bytes(generator: Any) -> bytes:
    return pickle.dumps(
        generator.bit_generator.state,
        protocol=pickle.HIGHEST_PROTOCOL,
    )


def _known_environment_rng_states(env: StEVEEnv) -> Dict[str, bytes]:
    """Capture known Gym/stEVE generators without traversing arbitrary state."""

    candidates = {
        "env.np_random": getattr(env, "np_random", None),
        "action_space.np_random": getattr(env.action_space, "np_random", None),
        "intervention._np_random": getattr(env._env, "_np_random", None),
        "simulation._rng": getattr(env._simulation, "_rng", None),
    }
    states = {}
    for name, candidate in candidates.items():
        if isinstance(candidate, np.random.Generator):
            states[name] = _generator_state_bytes(candidate)
    return states


def _capture_shadow_isolation_state(
    agent: Any,
    env: StEVEEnv,
) -> Dict[str, Any]:
    return {
        "python": pickle.dumps(
            random.getstate(),
            protocol=pickle.HIGHEST_PROTOCOL,
        ),
        "numpy": _numpy_legacy_state_bytes(),
        "torch": torch.get_rng_state().clone(),
        "cuda": (
            [value.clone() for value in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        ),
        "environment": _known_environment_rng_states(env),
        "previous_mean": agent.previous_mean.detach().clone(),
        "parameter_versions": tuple(
            int(parameter._version) for parameter in agent.model.parameters()
        ),
        "training_flags": tuple(
            bool(module.training) for module in agent.model.modules()
        ),
    }


def _assert_shadow_isolation(
    before: Mapping[str, Any],
    agent: Any,
    env: StEVEEnv,
) -> None:
    if before["python"] != pickle.dumps(
        random.getstate(),
        protocol=pickle.HIGHEST_PROTOCOL,
    ):
        raise RuntimeError("Shadow planning modified Python RNG state")
    if before["numpy"] != _numpy_legacy_state_bytes():
        raise RuntimeError("Shadow planning modified NumPy RNG state")
    if not torch.equal(before["torch"], torch.get_rng_state()):
        raise RuntimeError("Shadow planning modified CPU Torch RNG state")
    current_cuda = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    )
    if len(current_cuda) != len(before["cuda"]) or any(
        not torch.equal(left, right)
        for left, right in zip(before["cuda"], current_cuda)
    ):
        raise RuntimeError("Shadow planning modified CUDA RNG state")
    if before["environment"] != _known_environment_rng_states(env):
        raise RuntimeError("Shadow planning modified an environment RNG state")
    if not torch.equal(before["previous_mean"], agent.previous_mean):
        raise RuntimeError("Shadow planning modified agent.previous_mean")
    if before["parameter_versions"] != tuple(
        int(parameter._version) for parameter in agent.model.parameters()
    ):
        raise RuntimeError("Shadow planning modified model parameters")
    if before["training_flags"] != tuple(
        bool(module.training) for module in agent.model.modules()
    ):
        raise RuntimeError("Shadow planning modified model train/eval state")


def _trace_equal(left: ShadowPlanTrace, right: ShadowPlanTrace) -> bool:
    tensor_fields = (
        "action",
        "mean",
        "std",
        "final_actions",
        "final_task_values",
        "final_uncapped_risks",
        "final_capped_risks",
        "final_scores",
        "final_elite_indices",
        "final_elite_weights",
        "selected_task_value",
        "selected_uncapped_risk",
        "selected_capped_risk",
        "selected_step_uncapped_risk",
        "selected_step_capped_risk",
    )
    return all(
        torch.equal(getattr(left, name), getattr(right, name))
        for name in tensor_fields
    ) and all(
        torch.equal(a, b)
        for left_values, right_values in (
            (left.iteration_means, right.iteration_means),
            (left.iteration_stds, right.iteration_stds),
            (left.iteration_elites, right.iteration_elites),
        )
        for a, b in zip(left_values, right_values)
    )


def _finite_or_none(value: Any) -> Optional[float]:
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _safe_mean(values: Iterable[float]) -> Optional[float]:
    array = np.asarray(list(values), dtype=np.float64)
    return None if array.size == 0 else _finite_or_none(np.mean(array))


def _safe_median(values: Iterable[float]) -> Optional[float]:
    array = np.asarray(list(values), dtype=np.float64)
    return None if array.size == 0 else _finite_or_none(np.median(array))


def _safe_std(values: Iterable[float]) -> Optional[float]:
    array = np.asarray(list(values), dtype=np.float64)
    return None if array.size == 0 else _finite_or_none(np.std(array))


def _safe_percentile(
    values: Iterable[float],
    percentile: float,
) -> Optional[float]:
    array = np.asarray(list(values), dtype=np.float64)
    return (
        None
        if array.size == 0
        else _finite_or_none(np.percentile(array, percentile))
    )


def _fraction(flags: Iterable[bool]) -> Optional[float]:
    values = list(bool(value) for value in flags)
    return None if not values else float(sum(values) / len(values))


def _relative(numerator: float, denominator: float) -> Optional[float]:
    denominator = float(denominator)
    if not math.isfinite(denominator) or denominator == 0.0:
        return None
    value = float(numerator) / denominator
    return value if math.isfinite(value) else None


def _action_direction(value: float, tolerance: float) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def action_divergence(
    baseline: np.ndarray,
    shadow: np.ndarray,
    *,
    translation_tolerance: float,
    rotation_tolerance: float,
) -> Dict[str, Any]:
    baseline = np.asarray(baseline, dtype=np.float64)
    shadow = np.asarray(shadow, dtype=np.float64)
    if baseline.shape != (2,) or shadow.shape != (2,):
        raise ValueError("Actions must both have shape (2,)")
    delta = shadow - baseline
    l2 = float(np.linalg.norm(delta))
    baseline_direction = _action_direction(
        float(baseline[0]),
        translation_tolerance,
    )
    shadow_direction = _action_direction(
        float(shadow[0]),
        translation_tolerance,
    )
    return {
        "translation_abs_difference": abs(float(delta[0])),
        "rotation_abs_difference": abs(float(delta[1])),
        "action_l2": l2,
        "action_diverged": bool(
            abs(float(delta[0])) > translation_tolerance
            or abs(float(delta[1])) > rotation_tolerance
        ),
        "l2_gt_0_01": bool(l2 > 0.01),
        "l2_gt_0_05": bool(l2 > 0.05),
        "l2_gt_0_10": bool(l2 > 0.10),
        "translation_sign_change": bool(
            baseline_direction * shadow_direction < 0
        ),
        "baseline_translation_direction": baseline_direction,
        "shadow_translation_direction": shadow_direction,
    }


class PlanningRecordStore:
    """Compact in-memory records later written as non-object NPZ arrays."""

    def __init__(
        self,
        *,
        horizon: int,
        weights: Sequence[float],
        aggregations: Sequence[str],
    ) -> None:
        self.horizon = int(horizon)
        self.weights = tuple(float(value) for value in weights)
        self.aggregations = tuple(str(value) for value in aggregations)
        self.records: List[Dict[str, Any]] = []

    @property
    def count(self) -> int:
        return len(self.records)

    def add(self, record: Mapping[str, Any]) -> None:
        self.records.append(dict(record))

    def labels(self) -> Dict[str, int]:
        available = future = lower = tree = no_block = 0
        by_episode: Dict[str, List[int]] = {}
        for index, record in enumerate(self.records):
            by_episode.setdefault(str(record["episode_id"]), []).append(index)
        for indices in by_episode.values():
            for position, index in enumerate(indices):
                record = self.records[index]
                window_indices = indices[position : position + self.horizon]
                if len(window_indices) != self.horizon:
                    record.update(
                        {
                            "label_available": False,
                            "true_block_within_h": False,
                            "lower_block_within_h": False,
                            "tree_block_within_h": False,
                            "no_block_window": False,
                            "steps_to_next_block": -1,
                        }
                    )
                    continue
                window = [self.records[value] for value in window_indices]
                positive_offsets = [
                    offset
                    for offset, item in enumerate(window)
                    if float(item["translation_safety_cost"]) > 0.0
                ]
                true_value = bool(positive_offsets)
                lower_value = any(
                    str(item["block_reason"]) == "lower_insertion_boundary"
                    and float(item["translation_safety_cost"]) > 0.0
                    for item in window
                )
                tree_value = any(
                    str(item["block_reason"]) == "vessel_tree_end"
                    and float(item["translation_safety_cost"]) > 0.0
                    for item in window
                )
                record.update(
                    {
                        "label_available": True,
                        "true_block_within_h": true_value,
                        "lower_block_within_h": lower_value,
                        "tree_block_within_h": tree_value,
                        "no_block_window": not true_value,
                        "steps_to_next_block": (
                            int(positive_offsets[0]) if positive_offsets else -1
                        ),
                    }
                )
                available += 1
                future += int(true_value)
                lower += int(lower_value)
                tree += int(tree_value)
                no_block += int(not true_value)
        for record in self.records:
            record["boundary_phase"] = _boundary_phase(record)
        return {
            "planning_calls": self.count,
            "label_available": available,
            "future_block": future,
            "no_block": no_block,
            "lower_future_block": lower,
            "tree_future_block": tree,
            "incomplete_terminal_windows": self.count - available,
        }

    def write_npz(self, path: Path) -> Path:
        destination = path.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        records = self.records
        count = len(records)
        if count == 0:
            raise RuntimeError("Cannot write an empty planning-call dataset")
        arrays: Dict[str, np.ndarray] = {}
        base_string = (
            "episode_id",
            "mode",
            "variant",
            "block_reason",
            "boundary_phase",
            "noise_fingerprint",
        )
        base_int = (
            "episode_step",
            "environment_seed",
            "steps_to_next_block",
        )
        base_bool = (
            "policy_baseline_executed",
            "label_available",
            "true_block_within_h",
            "lower_block_within_h",
            "tree_block_within_h",
            "no_block_window",
        )
        for name in base_string:
            arrays[name] = np.asarray(
                [str(item[name]) for item in records],
                dtype=np.str_,
            )
        for name in base_int:
            arrays[name] = np.asarray(
                [int(item[name]) for item in records],
                dtype=np.int64,
            )
        for name in base_bool:
            arrays[name] = np.asarray(
                [bool(item[name]) for item in records],
                dtype=np.bool_,
            )
        for name in (
            "behavior_action",
            "baseline_action",
            "baseline_sequence",
        ):
            arrays[name] = np.stack([item[name] for item in records]).astype(
                np.float32,
                copy=False,
            )
        arrays["translation_safety_cost"] = np.asarray(
            [item["translation_safety_cost"] for item in records],
            dtype=np.float32,
        )
        arrays["baseline_planning_seconds"] = np.asarray(
            [item["baseline_planning_seconds"] for item in records],
            dtype=np.float64,
        )
        for aggregation in self.aggregations:
            for weight in self.weights:
                key = _combo_key(aggregation, weight)
                details = [item["shadow"][key] for item in records]
                prefix = f"{aggregation}__w_{weight:g}__"
                arrays[prefix + "action"] = np.stack(
                    [item["action"] for item in details]
                ).astype(np.float32, copy=False)
                arrays[prefix + "sequence"] = np.stack(
                    [item["sequence"] for item in details]
                ).astype(np.float32, copy=False)
                for field in _DETAIL_FLOAT_FIELDS:
                    arrays[prefix + field] = np.asarray(
                        [item[field] for item in details],
                        dtype=np.float64,
                    )
                for field in _DETAIL_BOOL_FIELDS:
                    arrays[prefix + field] = np.asarray(
                        [item[field] for item in details],
                        dtype=np.bool_,
                    )
        np.savez_compressed(destination, **arrays)
        return destination


_DETAIL_FLOAT_FIELDS = (
    "translation_abs_difference",
    "rotation_abs_difference",
    "action_l2",
    "sequence_l2",
    "baseline_risk_uncapped",
    "shadow_risk_uncapped",
    "uncapped_risk_reduction",
    "baseline_risk_capped",
    "shadow_risk_capped",
    "capped_risk_reduction",
    "relative_capped_risk_reduction",
    "baseline_selected_task_value",
    "shadow_selected_task_value",
    "task_value_sacrifice",
    "shadow_selected_penalized_score",
    "candidate_task_mean",
    "candidate_task_std",
    "candidate_task_min",
    "candidate_task_max",
    "candidate_uncapped_risk_mean",
    "candidate_uncapped_risk_std",
    "candidate_uncapped_risk_min",
    "candidate_uncapped_risk_max",
    "candidate_capped_risk_mean",
    "candidate_capped_risk_std",
    "candidate_capped_risk_min",
    "candidate_capped_risk_max",
    "penalty_mean",
    "penalty_std",
    "penalty_min",
    "penalty_max",
    "penalty_over_task_std",
    "rank_correlation",
    "elite_overlap_fraction",
    "baseline_sequence_risk_rank",
    "shadow_sequence_task_rank",
    "shadow_planning_seconds",
    "safety_inference_seconds",
    "active_core_seconds",
    "candidate_safety_inference_seconds",
    "selected_diagnostic_seconds",
)
_DETAIL_BOOL_FIELDS = (
    "action_diverged",
    "l2_gt_0_01",
    "l2_gt_0_05",
    "l2_gt_0_10",
    "translation_sign_change",
    "top1_agreement",
    "relative_capped_risk_reduction_available",
    "penalty_over_task_std_available",
    "rank_correlation_available",
)
_OPTIONAL_DETAIL_AVAILABILITY = {
    "relative_capped_risk_reduction": (
        "relative_capped_risk_reduction_available"
    ),
    "penalty_over_task_std": "penalty_over_task_std_available",
    "rank_correlation": "rank_correlation_available",
}


def _combo_key(aggregation: str, weight: float) -> str:
    return f"{aggregation}|{float(weight):.12g}"


def _boundary_phase(record: Mapping[str, Any]) -> str:
    mode = str(record["mode"])
    variant = str(record["variant"])
    if mode == "lower_boundary":
        if variant in ("approach_retract_1", "approach_retract_2"):
            return "several_before"
        if variant == "return_near_boundary":
            return "one_before"
        if variant.startswith("blocked_retraction"):
            return "blocked"
        return "matched_control"
    if mode == "tree_end":
        if variant.startswith("blocked_forward"):
            return "blocked"
        if variant == "tree_end_zero":
            return "one_before+matched_control"
        if variant == "tree_end_approach":
            distance = int(record.get("steps_to_next_block", -1))
            return "one_before" if distance == 1 else "several_before"
        return "matched_control"
    return "not_boundary"


def _record_in_boundary_phase(
    record: Mapping[str, Any],
    *,
    mode: str,
    phase: str,
) -> bool:
    if record["mode"] != mode:
        return False
    variant = str(record["variant"])
    if mode == "lower_boundary":
        if phase == "several_before":
            return variant in ("approach_retract_1", "approach_retract_2")
        if phase == "one_before":
            return variant == "return_near_boundary"
        if phase == "blocked":
            return variant.startswith("blocked_retraction")
        if phase == "matched_control":
            return variant not in (
                "approach_retract_1",
                "approach_retract_2",
                "return_near_boundary",
                "blocked_retraction_moderate",
                "blocked_retraction_maximum",
            )
    elif mode == "tree_end":
        if phase == "several_before":
            return (
                variant == "tree_end_approach"
                and int(record.get("steps_to_next_block", -1)) != 1
            )
        if phase == "one_before":
            return (
                variant == "tree_end_zero"
                or (
                    variant == "tree_end_approach"
                    and int(record.get("steps_to_next_block", -1)) == 1
                )
            )
        if phase == "blocked":
            return variant.startswith("blocked_forward")
        if phase == "matched_control":
            return variant in (
                "tree_end_zero",
                "tree_end_rotation_only",
                "tree_end_retraction",
            )
    return False


def _synchronize_device(device: torch.device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))


def _trace_detail(
    baseline: ShadowPlanTrace,
    shadow: ShadowPlanTrace,
    *,
    translation_tolerance: float,
    rotation_tolerance: float,
) -> Dict[str, Any]:
    comparison = compare_shadow_trace(baseline, shadow)
    baseline_action = baseline.action.detach().cpu().numpy()
    shadow_action = shadow.action.detach().cpu().numpy()
    divergence = action_divergence(
        baseline_action,
        shadow_action,
        translation_tolerance=translation_tolerance,
        rotation_tolerance=rotation_tolerance,
    )
    task = shadow.final_task_values.detach().cpu().numpy().astype(np.float64)
    uncapped = (
        shadow.final_uncapped_risks.detach().cpu().numpy().astype(np.float64)
    )
    capped = shadow.final_capped_risks.detach().cpu().numpy().astype(np.float64)
    penalty = float(shadow.safety_weight) * capped
    task_std = float(np.std(task))
    baseline_capped = float(baseline.selected_capped_risk.item())
    capped_reduction = float(comparison["capped_risk_reduction"])
    sequence_l2 = float(
        torch.linalg.vector_norm(shadow.mean - baseline.mean).item()
    )
    rank = comparison["task_penalized_spearman"]
    relative_risk = _relative(capped_reduction, baseline_capped)
    penalty_ratio = _relative(float(np.mean(np.abs(penalty))), task_std)
    detail: Dict[str, Any] = {
        "action": shadow.action.cpu().numpy().astype(np.float32, copy=True),
        "sequence": shadow.mean.cpu().numpy().astype(np.float32, copy=True),
        **divergence,
        "sequence_l2": sequence_l2,
        "baseline_risk_uncapped": float(
            baseline.selected_uncapped_risk.item()
        ),
        "shadow_risk_uncapped": float(shadow.selected_uncapped_risk.item()),
        "uncapped_risk_reduction": float(
            comparison["uncapped_risk_reduction"]
        ),
        "baseline_risk_capped": baseline_capped,
        "shadow_risk_capped": float(shadow.selected_capped_risk.item()),
        "capped_risk_reduction": capped_reduction,
        "relative_capped_risk_reduction": (
            0.0 if relative_risk is None else float(relative_risk)
        ),
        "relative_capped_risk_reduction_available": (
            relative_risk is not None
        ),
        "baseline_selected_task_value": float(
            baseline.selected_task_value.item()
        ),
        "shadow_selected_task_value": float(
            shadow.selected_task_value.item()
        ),
        "task_value_sacrifice": float(comparison["task_value_sacrifice"]),
        "shadow_selected_penalized_score": float(
            shadow.selected_task_value.item()
            - shadow.safety_weight * shadow.selected_capped_risk.item()
        ),
        "candidate_task_mean": float(np.mean(task)),
        "candidate_task_std": task_std,
        "candidate_task_min": float(np.min(task)),
        "candidate_task_max": float(np.max(task)),
        "candidate_uncapped_risk_mean": float(np.mean(uncapped)),
        "candidate_uncapped_risk_std": float(np.std(uncapped)),
        "candidate_uncapped_risk_min": float(np.min(uncapped)),
        "candidate_uncapped_risk_max": float(np.max(uncapped)),
        "candidate_capped_risk_mean": float(np.mean(capped)),
        "candidate_capped_risk_std": float(np.std(capped)),
        "candidate_capped_risk_min": float(np.min(capped)),
        "candidate_capped_risk_max": float(np.max(capped)),
        "penalty_mean": float(np.mean(penalty)),
        "penalty_std": float(np.std(penalty)),
        "penalty_min": float(np.min(penalty)),
        "penalty_max": float(np.max(penalty)),
        "penalty_over_task_std": (
            0.0 if penalty_ratio is None else float(penalty_ratio)
        ),
        "penalty_over_task_std_available": penalty_ratio is not None,
        "rank_correlation": 0.0 if rank is None else float(rank),
        "rank_correlation_available": rank is not None,
        "top1_agreement": bool(comparison["top1_agreement"]),
        "elite_overlap_fraction": float(
            comparison["elite_overlap_fraction"]
        ),
        "baseline_sequence_risk_rank": float(
            comparison["baseline_sequence_risk_rank"]
        ),
        "shadow_sequence_task_rank": float(
            comparison["shadow_sequence_task_rank"]
        ),
        "shadow_planning_seconds": float(shadow.planner_seconds),
        "safety_inference_seconds": float(shadow.safety_inference_seconds),
        "active_core_seconds": float(shadow.core_planner_seconds),
        "candidate_safety_inference_seconds": float(
            shadow.candidate_safety_inference_seconds
        ),
        "selected_diagnostic_seconds": float(
            shadow.selected_diagnostic_seconds
        ),
    }
    for field in _DETAIL_FLOAT_FIELDS:
        value = float(detail[field])
        if not math.isfinite(value):
            raise FloatingPointError(f"Invalid detail field {field}")
    return detail


def _assert_weight_zero_identity(
    traces: Mapping[str, Mapping[float, ShadowPlanTrace]],
) -> ShadowPlanTrace:
    baseline: Optional[ShadowPlanTrace] = None
    for aggregation, values in traces.items():
        if 0.0 not in values:
            raise RuntimeError(f"Aggregation {aggregation} lacks weight zero")
        candidate = values[0.0]
        if baseline is None:
            baseline = candidate
        elif not (
            torch.equal(candidate.action, baseline.action)
            and torch.equal(candidate.mean, baseline.mean)
            and torch.equal(
                candidate.final_task_values,
                baseline.final_task_values,
            )
            and torch.equal(candidate.final_actions, baseline.final_actions)
        ):
            raise RuntimeError(
                "Weight-zero baseline changed across risk aggregations"
            )
    assert baseline is not None
    return baseline


def _evaluate_planning_state(
    *,
    env: StEVEEnv,
    agent: Any,
    observation: np.ndarray,
    first_step: bool,
    analysis_previous_mean: Optional[torch.Tensor],
    production_policy: bool,
    private_planner_seed: int,
    weights: Sequence[float],
    aggregations: Sequence[str],
    safety_discount: float,
    translation_risk_cap: float,
    translation_tolerance: float,
    rotation_tolerance: float,
    episode_id: str,
    episode_step: int,
    environment_seed: int,
    mode: str,
    variant: str,
) -> Tuple[Dict[str, Any], np.ndarray, torch.Tensor]:
    """Evaluate one state, returning record, baseline action, next warm mean."""

    snapshot = capture_planner_snapshot(
        agent,
        observation,
        first_step=first_step,
    )
    if analysis_previous_mean is not None:
        snapshot = PlannerSnapshot(
            latent_single=snapshot.latent_single,
            previous_mean=analysis_previous_mean.detach().clone(),
            first_step=bool(first_step),
        )

    production_seconds = 0.0
    if production_policy:
        pre_device_rng = capture_device_torch_rng_state(agent.device)
        schedule = make_planner_noise_schedule(
            agent,
            initial_torch_state=pre_device_rng,
        )
        _synchronize_device(agent.device)
        started = time.perf_counter()
        executed_action = np.asarray(
            agent.act(
                observation,
                first_step=first_step,
                eval_mode=True,
            ),
            dtype=np.float32,
        ).copy()
        _synchronize_device(agent.device)
        production_seconds = time.perf_counter() - started
        if not torch.equal(
            capture_device_torch_rng_state(agent.device),
            schedule.core_end_torch_state,
        ):
            raise RuntimeError(
                "Production planner RNG consumption differs from explicit schedule"
            )
    else:
        schedule = make_planner_noise_schedule(
            agent,
            seed=int(private_planner_seed),
        )
        executed_action = np.empty((2,), dtype=np.float32)

    isolation = _capture_shadow_isolation_state(agent, env)
    traces = run_translation_shadow_sweep(
        agent,
        snapshot,
        schedule,
        safety_weights=weights,
        aggregations=aggregations,
        aggregation_discount=safety_discount,
        planner_cap=translation_risk_cap,
    )
    _assert_shadow_isolation(isolation, agent, env)
    baseline = _assert_weight_zero_identity(traces)
    baseline_action = baseline.action.cpu().numpy().astype(np.float32, copy=True)
    if production_policy:
        if not np.array_equal(executed_action, baseline_action):
            raise RuntimeError(
                "Weight-zero shadow replica does not reproduce executed action"
            )
        if not torch.equal(agent.previous_mean, baseline.mean):
            raise RuntimeError(
                "Weight-zero shadow replica does not reproduce production mean"
            )
    else:
        production_seconds = float(baseline.planner_seconds)

    detail_map: Dict[str, Dict[str, Any]] = {}
    for aggregation, values in traces.items():
        aggregation_baseline = values[0.0]
        for weight, trace in values.items():
            detail_map[_combo_key(aggregation, weight)] = _trace_detail(
                aggregation_baseline,
                trace,
                translation_tolerance=translation_tolerance,
                rotation_tolerance=rotation_tolerance,
            )
    record = {
        "episode_id": str(episode_id),
        "episode_step": int(episode_step),
        "environment_seed": int(environment_seed),
        "mode": str(mode),
        "variant": str(variant),
        "policy_baseline_executed": bool(production_policy),
        "behavior_action": np.zeros((2,), dtype=np.float32),
        "baseline_action": baseline_action,
        "baseline_sequence": baseline.mean.cpu().numpy().astype(
            np.float32,
            copy=True,
        ),
        "baseline_planning_seconds": float(production_seconds),
        "noise_fingerprint": schedule.fingerprint,
        "translation_safety_cost": 0.0,
        "block_reason": "none",
        "shadow": detail_map,
    }
    return record, baseline_action, baseline.mean.detach().clone()


def _finish_environment_step(
    *,
    env: StEVEEnv,
    store: PlanningRecordStore,
    record: Dict[str, Any],
    behavior_action: np.ndarray,
) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
    command = np.asarray(behavior_action, dtype=np.float32).copy()
    if command.shape != (2,) or not np.all(np.isfinite(command)):
        raise ValueError("Behavior action must be finite shape (2,)")
    result = env.step(command)
    info = result[4]
    safety_cost = np.asarray(info["safety_cost"], dtype=np.float32)
    if safety_cost.shape != (2,) or not np.all(np.isfinite(safety_cost)):
        raise RuntimeError("Environment returned invalid Safety cost")
    if np.any(safety_cost < 0.0):
        raise RuntimeError("Environment returned negative Safety cost")
    if not np.allclose(
        command,
        np.asarray(info["requested_action"], dtype=np.float32),
        rtol=0.0,
        atol=1e-6,
    ):
        raise RuntimeError("Behavior command/requested action alignment failed")
    record["behavior_action"] = command
    record["translation_safety_cost"] = float(safety_cost[TRANSLATION_INDEX])
    record["block_reason"] = str(
        info["safety_metrics"]["translation_block_reason"]
    )
    store.add(record)
    return result


class _PlanningCallLimit(RuntimeError):
    pass


def _enforce_limit(
    store: PlanningRecordStore,
    maximum: Optional[int],
) -> None:
    if maximum is not None and store.count >= int(maximum):
        raise _PlanningCallLimit


def _plan_and_step(
    *,
    env: StEVEEnv,
    agent: Any,
    store: PlanningRecordStore,
    observation: np.ndarray,
    behavior_action: Optional[np.ndarray],
    first_step: bool,
    analysis_previous_mean: Optional[torch.Tensor],
    production_policy: bool,
    args: argparse.Namespace,
    episode_id: str,
    episode_step: int,
    environment_seed: int,
    mode: str,
    variant: str,
) -> Tuple[
    np.ndarray,
    float,
    bool,
    bool,
    Dict[str, Any],
    torch.Tensor,
]:
    _enforce_limit(store, args.max_planning_calls_per_checkpoint)
    record, baseline_action, next_mean = _evaluate_planning_state(
        env=env,
        agent=agent,
        observation=observation,
        first_step=first_step,
        analysis_previous_mean=analysis_previous_mean,
        production_policy=production_policy,
        private_planner_seed=int(args.planner_seed + store.count),
        weights=args.weights,
        aggregations=args.aggregation,
        safety_discount=float(args.safety_discount),
        translation_risk_cap=float(args.translation_risk_cap),
        translation_tolerance=float(args.translation_action_tolerance),
        rotation_tolerance=float(args.rotation_action_tolerance),
        episode_id=episode_id,
        episode_step=episode_step,
        environment_seed=environment_seed,
        mode=mode,
        variant=variant,
    )
    if production_policy:
        command = baseline_action
    else:
        if behavior_action is None:
            raise ValueError("Non-policy state requires a behavior action")
        command = np.asarray(behavior_action, dtype=np.float32)
    result = _finish_environment_step(
        env=env,
        store=store,
        record=record,
        behavior_action=command,
    )
    return (*result, next_mean)


def _collect_policy(
    env: StEVEEnv,
    agent: Any,
    store: PlanningRecordStore,
    args: argparse.Namespace,
    checkpoint_label: str,
) -> List[Dict[str, Any]]:
    episodes = []
    for episode_index in range(int(args.policy_episodes)):
        seed = int(args.seeds[0] + episode_index)
        set_seed(seed)
        observation, _ = env.reset(seed=seed)
        terminated = truncated = False
        step = 0
        reward_sum = 0.0
        episode_id = f"policy:{checkpoint_label}:{seed}"
        while not (terminated or truncated):
            (
                observation,
                reward,
                terminated,
                truncated,
                _,
                _,
            ) = _plan_and_step(
                env=env,
                agent=agent,
                store=store,
                observation=observation,
                behavior_action=None,
                first_step=step == 0,
                analysis_previous_mean=None,
                production_policy=True,
                args=args,
                episode_id=episode_id,
                episode_step=step + 1,
                environment_seed=seed,
                mode="policy",
                variant="policy_action",
            )
            reward_sum += float(reward)
            step += 1
        episodes.append(
            {
                "episode_id": episode_id,
                "seed": seed,
                "length": step,
                "reward": reward_sum,
            }
        )
        print(
            f"shadow collect checkpoint={checkpoint_label} mode=policy "
            f"episode={episode_index + 1}/{args.policy_episodes} "
            f"seed={seed} length={step}"
        )
    return episodes


def _collect_random(
    env: StEVEEnv,
    agent: Any,
    store: PlanningRecordStore,
    args: argparse.Namespace,
    checkpoint_label: str,
) -> List[Dict[str, Any]]:
    episodes = []
    for episode_index in range(int(args.random_episodes)):
        seed = int(args.seeds[1] + episode_index)
        rng = np.random.default_rng(seed)
        observation, _ = env.reset(seed=seed)
        previous_mean = torch.zeros_like(agent.previous_mean)
        terminated = truncated = False
        step = 0
        episode_id = f"random:{seed}"
        while not (terminated or truncated):
            behavior = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
            (
                observation,
                _,
                terminated,
                truncated,
                _,
                previous_mean,
            ) = _plan_and_step(
                env=env,
                agent=agent,
                store=store,
                observation=observation,
                behavior_action=behavior,
                first_step=step == 0,
                analysis_previous_mean=previous_mean,
                production_policy=False,
                args=args,
                episode_id=episode_id,
                episode_step=step + 1,
                environment_seed=seed,
                mode="random",
                variant="uniform_random_action",
            )
            step += 1
        episodes.append(
            {"episode_id": episode_id, "seed": seed, "length": step}
        )
        print(
            f"shadow collect checkpoint={checkpoint_label} mode=random "
            f"episode={episode_index + 1}/{args.random_episodes} "
            f"seed={seed} length={step}"
        )
    return episodes


_LOWER_PLAN = (
    ("pre_zero", np.asarray([0.0, 0.0], dtype=np.float32), "none"),
    ("pre_forward", np.asarray([0.5, 0.0], dtype=np.float32), "none"),
    ("approach_retract_1", np.asarray([-0.2, 0.0], dtype=np.float32), "none"),
    ("approach_retract_2", np.asarray([-0.2, 0.0], dtype=np.float32), "none"),
    ("near_boundary_zero", np.asarray([0.0, 0.0], dtype=np.float32), "none"),
    (
        "near_boundary_forward",
        np.asarray([0.2, 0.0], dtype=np.float32),
        "none",
    ),
    (
        "return_near_boundary",
        np.asarray([-0.2, 0.0], dtype=np.float32),
        "none",
    ),
    (
        "blocked_retraction_moderate",
        np.asarray([-0.5, 0.0], dtype=np.float32),
        "lower_insertion_boundary",
    ),
    (
        "blocked_retraction_maximum",
        np.asarray([-1.0, 0.0], dtype=np.float32),
        "lower_insertion_boundary",
    ),
    ("post_block_zero", np.asarray([0.0, 0.0], dtype=np.float32), "none"),
)


def _collect_lower_boundary(
    env: StEVEEnv,
    agent: Any,
    store: PlanningRecordStore,
    args: argparse.Namespace,
    checkpoint_label: str,
) -> List[Dict[str, Any]]:
    sequences = []
    for repetition in range(int(args.lower_boundary_repetitions)):
        seed = int(args.seeds[2] + repetition)
        observation, _ = env.reset(seed=seed)
        previous_mean = torch.zeros_like(agent.previous_mean)
        terminated = truncated = False
        episode_id = f"lower_boundary:{seed}"
        for step, (variant, behavior, expected_reason) in enumerate(
            _LOWER_PLAN
        ):
            (
                observation,
                _,
                terminated,
                truncated,
                info,
                previous_mean,
            ) = _plan_and_step(
                env=env,
                agent=agent,
                store=store,
                observation=observation,
                behavior_action=behavior,
                first_step=step == 0,
                analysis_previous_mean=previous_mean,
                production_policy=False,
                args=args,
                episode_id=episode_id,
                episode_step=step + 1,
                environment_seed=seed,
                mode="lower_boundary",
                variant=variant,
            )
            reason = str(info["safety_metrics"]["translation_block_reason"])
            if reason != expected_reason:
                raise RuntimeError(
                    f"Lower-boundary seed {seed} variant {variant} expected "
                    f"{expected_reason}, got {reason}"
                )
            if terminated or truncated:
                raise RuntimeError(
                    f"Lower-boundary construction ended at {variant}"
                )
        sequences.append(
            {
                "episode_id": episode_id,
                "seed": seed,
                "transition_count": len(_LOWER_PLAN),
                "matched": True,
            }
        )
        print(
            f"shadow collect checkpoint={checkpoint_label} mode=lower "
            f"sequence={repetition + 1}/{args.lower_boundary_repetitions} "
            f"seed={seed}"
        )
    return sequences


_TREE_CONTROLS = (
    ("tree_end_zero", np.asarray([0.0, 0.0], dtype=np.float32), "none"),
    (
        "blocked_forward_moderate",
        np.asarray([0.5, 0.0], dtype=np.float32),
        "vessel_tree_end",
    ),
    (
        "blocked_forward_maximum",
        np.asarray([1.0, 0.0], dtype=np.float32),
        "vessel_tree_end",
    ),
    (
        "tree_end_rotation_only",
        np.asarray([0.0, 0.5], dtype=np.float32),
        "none",
    ),
    (
        "tree_end_retraction",
        np.asarray([-0.5, 0.0], dtype=np.float32),
        "none",
    ),
)


def _collect_tree_end(
    env: StEVEEnv,
    agent: Any,
    store: PlanningRecordStore,
    args: argparse.Namespace,
    checkpoint_label: str,
) -> Dict[str, Any]:
    successes = []
    failures = []
    attempt = 0
    repetitions = int(args.tree_end_repetitions)
    maximum_attempts = max(20 * repetitions, repetitions)
    approach_action = np.asarray([1.0, 0.0], dtype=np.float32)
    while len(successes) < repetitions and attempt < maximum_attempts:
        seed = int(args.seeds[3] + attempt)
        attempt += 1
        observation, _ = env.reset(seed=seed)
        previous_mean = torch.zeros_like(agent.previous_mean)
        terminated = truncated = False
        step = 0
        episode_id = f"tree_end:{seed}"
        while not _at_tree_end(env) and not (terminated or truncated):
            (
                observation,
                _,
                terminated,
                truncated,
                _,
                previous_mean,
            ) = _plan_and_step(
                env=env,
                agent=agent,
                store=store,
                observation=observation,
                behavior_action=approach_action,
                first_step=step == 0,
                analysis_previous_mean=previous_mean,
                production_policy=False,
                args=args,
                episode_id=episode_id,
                episode_step=step + 1,
                environment_seed=seed,
                mode="tree_end",
                variant="tree_end_approach",
            )
            step += 1
        if terminated or truncated or not _at_tree_end(env):
            failures.append(
                {
                    "episode_id": episode_id,
                    "seed": seed,
                    "approach_steps": step,
                    "ended_before_endpoint": True,
                }
            )
            continue
        mismatches = []
        for variant, behavior, expected_reason in _TREE_CONTROLS:
            (
                observation,
                _,
                terminated,
                truncated,
                info,
                previous_mean,
            ) = _plan_and_step(
                env=env,
                agent=agent,
                store=store,
                observation=observation,
                behavior_action=behavior,
                first_step=False,
                analysis_previous_mean=previous_mean,
                production_policy=False,
                args=args,
                episode_id=episode_id,
                episode_step=step + 1,
                environment_seed=seed,
                mode="tree_end",
                variant=variant,
            )
            step += 1
            reason = str(info["safety_metrics"]["translation_block_reason"])
            if reason != expected_reason:
                mismatches.append(
                    {
                        "variant": variant,
                        "expected": expected_reason,
                        "actual": reason,
                    }
                )
            if terminated or truncated:
                break
        if mismatches or terminated or truncated:
            failures.append(
                {
                    "episode_id": episode_id,
                    "seed": seed,
                    "approach_steps": step - len(_TREE_CONTROLS),
                    "mismatches": mismatches,
                    "ended_during_controls": bool(terminated or truncated),
                }
            )
            continue
        successes.append(
            {
                "episode_id": episode_id,
                "seed": seed,
                "approach_steps": step - len(_TREE_CONTROLS),
                "matched": True,
            }
        )
        print(
            f"shadow collect checkpoint={checkpoint_label} mode=tree "
            f"sequence={len(successes)}/{repetitions} seed={seed} "
            f"approach_steps={step - len(_TREE_CONTROLS)}"
        )
    if len(successes) != repetitions:
        raise RuntimeError(
            f"Collected only {len(successes)}/{repetitions} successful "
            f"tree-end sequences after {attempt} disjoint seeds"
        )
    return {"successful_sequences": successes, "failed_attempts": failures}


def _finite_detail_values(
    records: Sequence[Mapping[str, Any]],
    key: str,
    field: str,
) -> List[float]:
    values = []
    for record in records:
        detail = record["shadow"][key]
        availability_field = _OPTIONAL_DETAIL_AVAILABILITY.get(field)
        if availability_field is not None and not bool(
            detail[availability_field]
        ):
            continue
        value = float(detail[field])
        if math.isfinite(value):
            values.append(value)
    return values


def _group_summary(
    records: Sequence[Mapping[str, Any]],
    key: str,
) -> Dict[str, Any]:
    if not records:
        return {
            "count": 0,
            "action_divergence_rate": None,
            "action_l2_mean": None,
            "action_l2_median": None,
            "translation_sign_change_rate": None,
            "capped_risk_reduction_mean": None,
            "uncapped_risk_reduction_mean": None,
            "task_score_sacrifice_mean": None,
            "task_sacrifice_over_candidate_std_mean": None,
            "l2_gt_0_01_rate": None,
            "l2_gt_0_05_rate": None,
            "l2_gt_0_10_rate": None,
        }
    ratios = []
    for record in records:
        detail = record["shadow"][key]
        candidate_std = float(detail["candidate_task_std"])
        sacrifice = float(detail["task_value_sacrifice"])
        if candidate_std > 0.0:
            ratios.append(sacrifice / candidate_std)
    return {
        "count": len(records),
        "action_divergence_rate": _fraction(
            record["shadow"][key]["action_diverged"] for record in records
        ),
        "action_l2_mean": _safe_mean(
            record["shadow"][key]["action_l2"] for record in records
        ),
        "action_l2_median": _safe_median(
            record["shadow"][key]["action_l2"] for record in records
        ),
        "translation_sign_change_rate": _fraction(
            record["shadow"][key]["translation_sign_change"]
            for record in records
        ),
        "capped_risk_reduction_mean": _safe_mean(
            record["shadow"][key]["capped_risk_reduction"]
            for record in records
        ),
        "uncapped_risk_reduction_mean": _safe_mean(
            record["shadow"][key]["uncapped_risk_reduction"]
            for record in records
        ),
        "task_score_sacrifice_mean": _safe_mean(
            record["shadow"][key]["task_value_sacrifice"]
            for record in records
        ),
        "task_sacrifice_over_candidate_std_mean": _safe_mean(ratios),
        "l2_gt_0_01_rate": _fraction(
            record["shadow"][key]["l2_gt_0_01"] for record in records
        ),
        "l2_gt_0_05_rate": _fraction(
            record["shadow"][key]["l2_gt_0_05"] for record in records
        ),
        "l2_gt_0_10_rate": _fraction(
            record["shadow"][key]["l2_gt_0_10"] for record in records
        ),
    }


def _direction_transition_summary(
    records: Sequence[Mapping[str, Any]],
    key: str,
    *,
    tolerance: float,
) -> Dict[str, Any]:
    forward = {
        "eligible": 0,
        "smaller_forward": 0,
        "zero": 0,
        "retraction": 0,
        "larger_or_equal_forward": 0,
    }
    retraction = {
        "eligible": 0,
        "smaller_retraction": 0,
        "zero": 0,
        "forward": 0,
        "larger_or_equal_retraction": 0,
    }
    for record in records:
        baseline = float(record["baseline_action"][0])
        shadow = float(record["shadow"][key]["action"][0])
        base_direction = _action_direction(baseline, tolerance)
        shadow_direction = _action_direction(shadow, tolerance)
        if base_direction > 0:
            forward["eligible"] += 1
            if shadow_direction < 0:
                forward["retraction"] += 1
            elif shadow_direction == 0:
                forward["zero"] += 1
            elif shadow < baseline - tolerance:
                forward["smaller_forward"] += 1
            else:
                forward["larger_or_equal_forward"] += 1
        elif base_direction < 0:
            retraction["eligible"] += 1
            if shadow_direction > 0:
                retraction["forward"] += 1
            elif shadow_direction == 0:
                retraction["zero"] += 1
            elif shadow > baseline + tolerance:
                retraction["smaller_retraction"] += 1
            else:
                retraction["larger_or_equal_retraction"] += 1
    for summary in (forward, retraction):
        denominator = int(summary["eligible"])
        summary["fractions"] = {
            name: (
                None
                if denominator == 0
                else float(value / denominator)
            )
            for name, value in summary.items()
            if name != "eligible"
        }
    return {"baseline_forward": forward, "baseline_retraction": retraction}


def _policy_divergence_runs(
    records: Sequence[Mapping[str, Any]],
    key: str,
) -> Dict[str, Any]:
    by_episode: Dict[str, List[bool]] = {}
    for record in records:
        if record["mode"] == "policy":
            by_episode.setdefault(str(record["episode_id"]), []).append(
                bool(record["shadow"][key]["action_diverged"])
            )
    lengths = []
    policy_steps = 0
    for flags in by_episode.values():
        policy_steps += len(flags)
        run = 0
        for flag in flags + [False]:
            if flag:
                run += 1
            elif run:
                lengths.append(run)
                run = 0
    return {
        "policy_steps": policy_steps,
        "run_count": len(lengths),
        "runs_per_100_policy_steps": (
            None
            if policy_steps == 0
            else float(100.0 * len(lengths) / policy_steps)
        ),
        "run_length_mean": _safe_mean(lengths),
        "run_length_max": None if not lengths else int(max(lengths)),
        "run_length_p95": _safe_percentile(lengths, 95.0),
    }


def _false_conservatism_summary(
    records: Sequence[Mapping[str, Any]],
    key: str,
    *,
    tolerance: float,
) -> Dict[str, Any]:
    no_block = [
        record
        for record in records
        if bool(record.get("label_available", False))
        and bool(record.get("no_block_window", False))
    ]
    both = [
        record
        for record in no_block
        if float(record["shadow"][key]["capped_risk_reduction"]) > 0.0
        and float(record["shadow"][key]["task_value_sacrifice"]) > 0.0
    ]
    diverged = [
        record
        for record in no_block
        if bool(record["shadow"][key]["action_diverged"])
    ]
    both_diverged = [
        record
        for record in both
        if bool(record["shadow"][key]["action_diverged"])
    ]
    absolute_translation_reduction = [
        abs(float(record["baseline_action"][0]))
        - abs(float(record["shadow"][key]["action"][0]))
        for record in no_block
    ]
    return {
        "count": len(no_block),
        "action_divergence_rate": _fraction(
            record["shadow"][key]["action_diverged"] for record in no_block
        ),
        "risk_lower_and_task_lower_fraction_all": (
            None if not no_block else float(len(both) / len(no_block))
        ),
        "risk_lower_and_task_lower_fraction_diverged": (
            None
            if not diverged
            else float(len(both_diverged) / len(diverged))
        ),
        "task_score_sacrifice_mean": _safe_mean(
            record["shadow"][key]["task_value_sacrifice"]
            for record in no_block
        ),
        "absolute_translation_reduction_mean": _safe_mean(
            absolute_translation_reduction
        ),
        "direction_transitions": _direction_transition_summary(
            no_block,
            key,
            tolerance=tolerance,
        ),
        "policy_divergence_runs": _policy_divergence_runs(records, key),
    }


def _boundary_summary(
    records: Sequence[Mapping[str, Any]],
    key: str,
    *,
    mode: str,
    tolerance: float,
) -> Dict[str, Any]:
    output = {}
    for phase in (
        "several_before",
        "one_before",
        "blocked",
        "matched_control",
    ):
        selected = [
            record
            for record in records
            if _record_in_boundary_phase(
                record,
                mode=mode,
                phase=phase,
            )
        ]
        improved = []
        zero = []
        opposite = []
        for record in selected:
            baseline = float(record["baseline_action"][0])
            shadow = float(record["shadow"][key]["action"][0])
            if mode == "lower_boundary":
                improved.append(shadow > baseline + tolerance)
                opposite.append(shadow > tolerance)
            else:
                improved.append(shadow < baseline - tolerance)
                opposite.append(shadow < -tolerance)
            zero.append(abs(shadow) <= tolerance)
        output[phase] = {
            **_group_summary(selected, key),
            "desired_direction_change_rate": _fraction(improved),
            "zero_translation_rate": _fraction(zero),
            (
                "forward_translation_rate"
                if mode == "lower_boundary"
                else "retraction_rate"
            ): _fraction(opposite),
        }
    return output


def _score_scale_summary(
    records: Sequence[Mapping[str, Any]],
    key: str,
) -> Dict[str, Any]:
    return {
        "baseline_score_mean": _safe_mean(
            record["shadow"][key]["candidate_task_mean"]
            for record in records
        ),
        "baseline_score_std_mean": _safe_mean(
            record["shadow"][key]["candidate_task_std"]
            for record in records
        ),
        "baseline_score_min": (
            None
            if not records
            else min(
                float(record["shadow"][key]["candidate_task_min"])
                for record in records
            )
        ),
        "baseline_score_max": (
            None
            if not records
            else max(
                float(record["shadow"][key]["candidate_task_max"])
                for record in records
            )
        ),
        "uncapped_risk_mean": _safe_mean(
            record["shadow"][key]["candidate_uncapped_risk_mean"]
            for record in records
        ),
        "uncapped_risk_std_mean": _safe_mean(
            record["shadow"][key]["candidate_uncapped_risk_std"]
            for record in records
        ),
        "uncapped_risk_min": (
            None
            if not records
            else min(
                float(record["shadow"][key]["candidate_uncapped_risk_min"])
                for record in records
            )
        ),
        "uncapped_risk_max": (
            None
            if not records
            else max(
                float(record["shadow"][key]["candidate_uncapped_risk_max"])
                for record in records
            )
        ),
        "capped_risk_mean": _safe_mean(
            record["shadow"][key]["candidate_capped_risk_mean"]
            for record in records
        ),
        "capped_risk_std_mean": _safe_mean(
            record["shadow"][key]["candidate_capped_risk_std"]
            for record in records
        ),
        "capped_risk_min": (
            None
            if not records
            else min(
                float(record["shadow"][key]["candidate_capped_risk_min"])
                for record in records
            )
        ),
        "capped_risk_max": (
            None
            if not records
            else max(
                float(record["shadow"][key]["candidate_capped_risk_max"])
                for record in records
            )
        ),
        "penalty_mean": _safe_mean(
            record["shadow"][key]["penalty_mean"] for record in records
        ),
        "penalty_std_mean": _safe_mean(
            record["shadow"][key]["penalty_std"] for record in records
        ),
        "penalty_min": (
            None
            if not records
            else min(
                float(record["shadow"][key]["penalty_min"])
                for record in records
            )
        ),
        "penalty_max": (
            None
            if not records
            else max(
                float(record["shadow"][key]["penalty_max"])
                for record in records
            )
        ),
        "penalty_over_baseline_score_std_mean": _safe_mean(
            _finite_detail_values(records, key, "penalty_over_task_std")
        ),
    }


def _ranking_summary(
    records: Sequence[Mapping[str, Any]],
    key: str,
) -> Dict[str, Any]:
    return {
        "rank_correlation_mean": _safe_mean(
            _finite_detail_values(records, key, "rank_correlation")
        ),
        "top1_agreement_rate": _fraction(
            record["shadow"][key]["top1_agreement"] for record in records
        ),
        "elite_overlap_fraction_mean": _safe_mean(
            record["shadow"][key]["elite_overlap_fraction"]
            for record in records
        ),
        "baseline_sequence_risk_rank_mean": _safe_mean(
            record["shadow"][key]["baseline_sequence_risk_rank"]
            for record in records
        ),
        "shadow_sequence_task_rank_mean": _safe_mean(
            record["shadow"][key]["shadow_sequence_task_rank"]
            for record in records
        ),
    }


def _runtime_summary(
    records: Sequence[Mapping[str, Any]],
    key: str,
) -> Dict[str, Any]:
    policy = [
        record for record in records if record["policy_baseline_executed"]
    ]
    baseline_seconds = _safe_mean(
        record["baseline_planning_seconds"] for record in policy
    )
    candidate_safety_seconds = _safe_mean(
        record["shadow"][key]["candidate_safety_inference_seconds"]
        for record in records
    )
    return {
        "production_baseline_policy_seconds_mean": baseline_seconds,
        "shadow_total_with_selected_diagnostics_seconds_mean": _safe_mean(
            record["shadow"][key]["shadow_planning_seconds"]
            for record in records
        ),
        "estimated_single_weight_active_core_seconds_mean": _safe_mean(
            record["shadow"][key]["active_core_seconds"] for record in records
        ),
        "candidate_safety_inference_seconds_mean": candidate_safety_seconds,
        "projected_single_weight_total_seconds_mean": (
            None
            if baseline_seconds is None or candidate_safety_seconds is None
            else float(baseline_seconds + candidate_safety_seconds)
        ),
        "projected_incremental_overhead_fraction": (
            None
            if (
                baseline_seconds is None
                or baseline_seconds == 0.0
                or candidate_safety_seconds is None
            )
            else float(candidate_safety_seconds / baseline_seconds)
        ),
        "selected_mean_diagnostic_seconds_mean": _safe_mean(
            record["shadow"][key]["selected_diagnostic_seconds"]
            for record in records
        ),
        "note": (
            "The active-core estimate starts from the already encoded latent, "
            "includes faithful candidate task/Safety rollouts, and excludes "
            "observation encoding plus post-hoc selected-mean diagnostics. "
            "Candidate Safety inference is the most direct additive-overhead "
            "estimate. The standalone utility performs a second dynamics "
            "rollout for Safety, so a fused implementation could be faster."
        ),
    }


def analyze_records(
    store: PlanningRecordStore,
    *,
    translation_tolerance: float,
) -> Dict[str, Any]:
    records = store.records
    combinations: Dict[str, Dict[str, Any]] = {}
    for aggregation in store.aggregations:
        aggregation_report = {}
        for weight in store.weights:
            key = _combo_key(aggregation, weight)
            groups = {
                "all": records,
                "future_block": [
                    record
                    for record in records
                    if bool(record.get("label_available", False))
                    and bool(record.get("true_block_within_h", False))
                ],
                "no_block": [
                    record
                    for record in records
                    if bool(record.get("label_available", False))
                    and bool(record.get("no_block_window", False))
                ],
                "lower_boundary": [
                    record
                    for record in records
                    if record["mode"] == "lower_boundary"
                ],
                "tree_end": [
                    record for record in records if record["mode"] == "tree_end"
                ],
                "policy": [
                    record for record in records if record["mode"] == "policy"
                ],
            }
            group_report = {
                name: _group_summary(selected, key)
                for name, selected in groups.items()
            }
            future_rate = group_report["future_block"][
                "action_divergence_rate"
            ]
            no_block_rate = group_report["no_block"]["action_divergence_rate"]
            aggregation_report[f"{weight:g}"] = {
                "weight": float(weight),
                "groups": group_report,
                "selective_divergence_gap": (
                    None
                    if future_rate is None or no_block_rate is None
                    else float(future_rate - no_block_rate)
                ),
                "score_scale": _score_scale_summary(records, key),
                "ranking": _ranking_summary(records, key),
                "false_conservatism": _false_conservatism_summary(
                    records,
                    key,
                    tolerance=translation_tolerance,
                ),
                "lower_boundary": _boundary_summary(
                    records,
                    key,
                    mode="lower_boundary",
                    tolerance=translation_tolerance,
                ),
                "tree_end": _boundary_summary(
                    records,
                    key,
                    mode="tree_end",
                    tolerance=translation_tolerance,
                ),
                "runtime": _runtime_summary(records, key),
            }
        combinations[aggregation] = aggregation_report
    per_state_shadow_seconds = []
    per_state_safety_seconds = []
    for record in records:
        details = list(record["shadow"].values())
        per_state_shadow_seconds.append(
            sum(float(item["shadow_planning_seconds"]) for item in details)
        )
        per_state_safety_seconds.append(
            sum(float(item["safety_inference_seconds"]) for item in details)
        )
    return {
        "combinations": combinations,
        "complete_shadow_grid_runtime": {
            "weights_times_aggregations": (
                len(store.weights) * len(store.aggregations)
            ),
            "total_shadow_seconds_per_state_mean": _safe_mean(
                per_state_shadow_seconds
            ),
            "total_shadow_seconds_per_state_median": _safe_median(
                per_state_shadow_seconds
            ),
            "total_shadow_seconds_per_state_p95": _safe_percentile(
                per_state_shadow_seconds,
                95.0,
            ),
            "total_safety_inference_seconds_per_state_mean": _safe_mean(
                per_state_safety_seconds
            ),
            "note": (
                "This is the full offline grid including selected-mean "
                "diagnostics for every weight/aggregation, not projected "
                "single-weight active-planner cost."
            ),
        },
    }


def _curvature_isolation_spy(
    agent: Any,
    *,
    weight: float,
    aggregation: str,
    safety_discount: float,
    planner_cap: float,
) -> Dict[str, Any]:
    observation = np.zeros((agent.observation_dim,), dtype=np.float32)
    snapshot = capture_planner_snapshot(agent, observation, first_step=True)
    schedule = make_planner_noise_schedule(agent, seed=470070)
    baseline = run_translation_shadow_plan(
        agent,
        snapshot,
        schedule,
        safety_weight=weight,
        aggregation=aggregation,
        aggregation_discount=safety_discount,
        planner_cap=planner_cap,
    )
    original = type(agent.model).safety_transformed

    def changed_curvature(
        model: torch.nn.Module,
        latent: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        value = original(model, latent, action).clone()
        value[..., 0] = float("nan")
        return value

    with mock.patch.object(
        type(agent.model),
        "safety_transformed",
        new=changed_curvature,
    ):
        changed = run_translation_shadow_plan(
            agent,
            snapshot,
            schedule,
            safety_weight=weight,
            aggregation=aggregation,
            aggregation_discount=safety_discount,
            planner_cap=planner_cap,
        )
    passed = _trace_equal(baseline, changed)
    if not passed:
        raise RuntimeError(
            "Changing only Curvature output changed Translation shadow planning"
        )
    return {
        "passed": True,
        "spy_change": (
            "curvature transformed output replaced with a non-finite sentinel"
        ),
        "translation_risk_score_elite_mean_action_bitwise_unchanged": True,
    }


def _model_state_clone(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _assert_model_state_equal(
    expected: Mapping[str, torch.Tensor],
    model: torch.nn.Module,
) -> None:
    actual = model.state_dict()
    if set(expected) != set(actual):
        raise RuntimeError("Model state keys changed during evaluation")
    for name, value in expected.items():
        if not torch.equal(value, actual[name].detach().cpu()):
            raise RuntimeError(
                f"Model state tensor {name!r} changed during evaluation"
            )


def evaluate_checkpoint(
    checkpoint_path: Path,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    checkpoint, config, agent, device = load_evaluation_model(
        checkpoint_path,
        requested_device=args.device,
    )
    del checkpoint
    if not bool(agent.config.get("mpc", True)):
        raise ValueError("Shadow evaluation requires checkpoint planning.mpc=true")
    if agent.action_dim != 2:
        raise ValueError("stEVE shadow evaluation requires two actions")
    label = _checkpoint_label(checkpoint_path)
    model_before = _model_state_clone(agent.model)
    memory_before_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    spy = _curvature_isolation_spy(
        agent,
        weight=max(float(value) for value in args.weights),
        aggregation=str(args.aggregation[0]),
        safety_discount=float(args.safety_discount),
        planner_cap=float(args.translation_risk_cap),
    )
    env = make_steve_env(config["environment"])
    store = PlanningRecordStore(
        horizon=agent.horizon,
        weights=args.weights,
        aggregations=args.aggregation,
    )
    collection: Dict[str, Any] = {}
    incomplete_due_to_limit = False
    started = time.perf_counter()
    try:
        observation_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        if (
            observation_dim != agent.observation_dim
            or action_dim != agent.action_dim
        ):
            raise ValueError(
                "Environment dimensions do not match checkpoint dimensions"
            )
        try:
            collection["policy"] = _collect_policy(
                env,
                agent,
                store,
                args,
                label,
            )
            collection["random"] = _collect_random(
                env,
                agent,
                store,
                args,
                label,
            )
            collection["lower_boundary"] = _collect_lower_boundary(
                env,
                agent,
                store,
                args,
                label,
            )
            collection["tree_end"] = _collect_tree_end(
                env,
                agent,
                store,
                args,
                label,
            )
        except _PlanningCallLimit:
            incomplete_due_to_limit = True
            collection["developer_limit_reached"] = int(
                args.max_planning_calls_per_checkpoint
            )
    finally:
        env.close()
    elapsed = time.perf_counter() - started
    if store.count == 0:
        raise RuntimeError(f"Checkpoint {label} produced no planning calls")
    labels = store.labels()
    details_path = (
        args.details_dir.expanduser().resolve() / f"{label}.npz"
    )
    store.write_npz(details_path)
    analysis = analyze_records(
        store,
        translation_tolerance=float(args.translation_action_tolerance),
    )
    _assert_model_state_equal(model_before, agent.model)
    memory_after_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    report = {
        "checkpoint_label": label,
        "checkpoint_path": str(checkpoint_path.expanduser().resolve()),
        "device": str(device),
        "planner": {
            "horizon": int(agent.horizon),
            "iterations": int(agent.config["iterations"]),
            "num_samples": int(agent.config["num_samples"]),
            "num_elites": int(agent.config["num_elites"]),
            "num_policy_trajectories": int(agent.config["num_pi_trajs"]),
            "temperature": float(agent.config["temperature"]),
            "min_std": float(agent.config["min_std"]),
            "max_std": float(agent.config["max_std"]),
            "discount": float(agent.discount),
        },
        "collection": collection,
        "label_counts": labels,
        "complete_required_evaluation": not incomplete_due_to_limit,
        "details_npz": str(details_path),
        "curvature_isolation_spy": spy,
        "isolation": {
            "per_call_python_numpy_torch_environment_rng_asserted": True,
            "per_call_previous_mean_and_parameter_versions_asserted": True,
            "full_model_state_bitwise_unchanged": True,
            "optimizer_state": (
                "not loaded by the read-only checkpoint evaluator; therefore "
                "no optimizer object exists to mutate"
            ),
            "replay_rng": (
                "no replay buffer exists in evaluation; focused smoke test "
                "asserts a real EpisodeReplayBuffer RNG is unchanged"
            ),
        },
        "runtime": {
            "total_checkpoint_evaluation_seconds": float(elapsed),
            "process_peak_rss_before_kib": memory_before_kib,
            "process_peak_rss_after_kib": memory_after_kib,
            "process_peak_rss_increment_kib": max(
                0,
                memory_after_kib - memory_before_kib,
            ),
        },
        **analysis,
    }
    return report, config


def _nonnull_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    filtered = [float(value) for value in values if value is not None]
    return _safe_mean(filtered)


def _nonnull_min(values: Iterable[Optional[float]]) -> Optional[float]:
    filtered = [float(value) for value in values if value is not None]
    return None if not filtered else min(filtered)


def _nonnull_max(values: Iterable[Optional[float]]) -> Optional[float]:
    filtered = [float(value) for value in values if value is not None]
    return None if not filtered else max(filtered)


def _boundary_direction_value(
    combination: Mapping[str, Any],
    boundary: str,
) -> Optional[float]:
    values = []
    for phase in ("several_before", "one_before", "blocked"):
        value = combination[boundary][phase]["desired_direction_change_rate"]
        count = int(combination[boundary][phase]["count"])
        if value is not None and count > 0:
            values.extend([float(value)] * count)
    return _safe_mean(values)


def build_cross_checkpoint_pareto(
    checkpoint_reports: Sequence[Mapping[str, Any]],
    aggregations: Sequence[str],
    weights: Sequence[float],
) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for aggregation in aggregations:
        rows = []
        for weight in weights:
            combinations = [
                report["combinations"][aggregation][f"{float(weight):g}"]
                for report in checkpoint_reports
            ]
            future_risk = [
                item["groups"]["future_block"][
                    "capped_risk_reduction_mean"
                ]
                for item in combinations
            ]
            task_sacrifice = [
                item["groups"]["future_block"][
                    "task_score_sacrifice_mean"
                ]
                for item in combinations
            ]
            task_scaled = [
                item["groups"]["future_block"][
                    "task_sacrifice_over_candidate_std_mean"
                ]
                for item in combinations
            ]
            overall_divergence = [
                item["groups"]["all"]["action_divergence_rate"]
                for item in combinations
            ]
            no_block_divergence = [
                item["groups"]["no_block"]["action_divergence_rate"]
                for item in combinations
            ]
            future_divergence = [
                item["groups"]["future_block"]["action_divergence_rate"]
                for item in combinations
            ]
            selectivity = [
                item["selective_divergence_gap"] for item in combinations
            ]
            policy_divergence = [
                item["groups"]["policy"]["action_divergence_rate"]
                for item in combinations
            ]
            lower_direction = [
                _boundary_direction_value(item, "lower_boundary")
                for item in combinations
            ]
            tree_direction = [
                _boundary_direction_value(item, "tree_end")
                for item in combinations
            ]
            active_seconds = [
                item["runtime"][
                    "estimated_single_weight_active_core_seconds_mean"
                ]
                for item in combinations
            ]
            row = {
                "weight": float(weight),
                "predicted_future_block_risk_reduction_mean": _nonnull_mean(
                    future_risk
                ),
                "predicted_future_block_risk_reduction_min": _nonnull_min(
                    future_risk
                ),
                "predicted_future_block_risk_reduction_max": _nonnull_max(
                    future_risk
                ),
                "task_score_sacrifice_mean": _nonnull_mean(task_sacrifice),
                "task_sacrifice_over_candidate_std_mean": _nonnull_mean(
                    task_scaled
                ),
                "action_divergence_rate_mean": _nonnull_mean(
                    overall_divergence
                ),
                "no_block_divergence_rate_mean": _nonnull_mean(
                    no_block_divergence
                ),
                "future_block_divergence_rate_mean": _nonnull_mean(
                    future_divergence
                ),
                "selective_divergence_gap_mean": _nonnull_mean(selectivity),
                "selective_divergence_gap_min": _nonnull_min(selectivity),
                "ordinary_policy_divergence_rate_mean": _nonnull_mean(
                    policy_divergence
                ),
                "lower_boundary_directional_improvement_mean": _nonnull_mean(
                    lower_direction
                ),
                "tree_end_directional_improvement_mean": _nonnull_mean(
                    tree_direction
                ),
                "estimated_single_weight_active_core_seconds_mean": (
                    _nonnull_mean(active_seconds)
                ),
                "near_total_action_replacement_any_checkpoint": any(
                    value is not None and float(value) >= 0.95
                    for value in overall_divergence
                ),
                "per_checkpoint": {
                    str(report["checkpoint_label"]): {
                        "future_block_risk_reduction": future_risk[index],
                        "task_score_sacrifice": task_sacrifice[index],
                        "task_sacrifice_over_candidate_std": task_scaled[index],
                        "all_divergence": overall_divergence[index],
                        "no_block_divergence": no_block_divergence[index],
                        "future_block_divergence": future_divergence[index],
                        "selectivity_gap": selectivity[index],
                        "policy_divergence": policy_divergence[index],
                        "lower_directional_improvement": lower_direction[index],
                        "tree_directional_improvement": tree_direction[index],
                    }
                    for index, report in enumerate(checkpoint_reports)
                },
            }
            rows.append(row)

        for row in rows:
            dominated = False
            if row["weight"] != 0.0:
                for other in rows:
                    if other is row or other["weight"] == 0.0:
                        continue
                    risk = row["predicted_future_block_risk_reduction_mean"]
                    other_risk = other[
                        "predicted_future_block_risk_reduction_mean"
                    ]
                    sacrifice = row["task_score_sacrifice_mean"]
                    other_sacrifice = other["task_score_sacrifice_mean"]
                    no_block = row["no_block_divergence_rate_mean"]
                    other_no_block = other["no_block_divergence_rate_mean"]
                    if None in (
                        risk,
                        other_risk,
                        sacrifice,
                        other_sacrifice,
                        no_block,
                        other_no_block,
                    ):
                        continue
                    weak = (
                        other_risk >= risk
                        and other_sacrifice <= sacrifice
                        and other_no_block <= no_block
                    )
                    strict = (
                        other_risk > risk
                        or other_sacrifice < sacrifice
                        or other_no_block < no_block
                    )
                    if weak and strict:
                        dominated = True
                        break
            row["pareto_dominated"] = dominated
        output[aggregation] = rows
    return output


def choose_engineering_decision(
    pareto: Mapping[str, Sequence[Mapping[str, Any]]],
    checkpoint_reports: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    candidates = []
    positive_rows = []
    for aggregation, rows in pareto.items():
        for row in rows:
            if float(row["weight"]) == 0.0:
                continue
            positive_rows.append((aggregation, row))
            criteria = {
                "risk_reduction_every_checkpoint": (
                    row["predicted_future_block_risk_reduction_min"] is not None
                    and row["predicted_future_block_risk_reduction_min"] > 0.0
                ),
                "positive_selectivity_every_checkpoint": (
                    row["selective_divergence_gap_min"] is not None
                    and row["selective_divergence_gap_min"] > 0.0
                ),
                "ordinary_policy_divergence_below_half": (
                    row["ordinary_policy_divergence_rate_mean"] is not None
                    and row["ordinary_policy_divergence_rate_mean"] < 0.5
                ),
                "scaled_task_sacrifice_below_half_std": (
                    row["task_sacrifice_over_candidate_std_mean"] is not None
                    and row["task_sacrifice_over_candidate_std_mean"] < 0.5
                ),
                "no_near_total_replacement": not row[
                    "near_total_action_replacement_any_checkpoint"
                ],
                "both_boundary_directions_majority": (
                    row["lower_boundary_directional_improvement_mean"] is not None
                    and row["lower_boundary_directional_improvement_mean"] > 0.5
                    and row["tree_end_directional_improvement_mean"] is not None
                    and row["tree_end_directional_improvement_mean"] > 0.5
                ),
            }
            if all(criteria.values()):
                candidates.append((aggregation, row, criteria))
    if candidates:
        aggregation, row, criteria = min(
            candidates,
            key=lambda item: (
                float(item[1]["weight"]),
                -float(
                    item[1]["predicted_future_block_risk_reduction_mean"]
                ),
            ),
        )
        return {
            "outcome": "A",
            "label": "PROCEED TO ACTIVE TRANSLATION-ONLY SOFT MPC",
            "recommended_aggregation": aggregation,
            "recommended_weight": float(row["weight"]),
            "criteria": criteria,
            "reason": (
                "This low weight met every conservative cross-checkpoint "
                "engineering criterion. This is not a clinical safety claim."
            ),
        }

    score_spreads = []
    for report in checkpoint_reports:
        first_aggregation = next(iter(report["combinations"]))
        first_weight = next(iter(report["combinations"][first_aggregation]))
        value = report["combinations"][first_aggregation][first_weight][
            "score_scale"
        ]["baseline_score_std_mean"]
        if value is not None and float(value) > 0.0:
            score_spreads.append(float(value))
    scale_ratio = (
        None
        if len(score_spreads) < 2
        else float(max(score_spreads) / min(score_spreads))
    )
    any_risk_reduction = any(
        row["predicted_future_block_risk_reduction_mean"] is not None
        and row["predicted_future_block_risk_reduction_mean"] > 0.0
        for _, row in positive_rows
    )
    penalty_impact_ratios = []
    for aggregation, row in positive_rows:
        weight_key = f"{float(row['weight']):g}"
        impacts = []
        for checkpoint in checkpoint_reports:
            value = checkpoint["combinations"][aggregation][weight_key][
                "score_scale"
            ]["penalty_over_baseline_score_std_mean"]
            if value is not None and float(value) > 0.0:
                impacts.append(float(value))
        if len(impacts) >= 2:
            penalty_impact_ratios.append(
                {
                    "aggregation": aggregation,
                    "weight": float(row["weight"]),
                    "ratio": float(max(impacts) / min(impacts)),
                    "minimum": min(impacts),
                    "maximum": max(impacts),
                }
            )
    largest_penalty_impact_ratio = (
        None
        if not penalty_impact_ratios
        else max(penalty_impact_ratios, key=lambda item: item["ratio"])
    )
    materially_different_scale = (
        (scale_ratio is not None and scale_ratio >= 2.0)
        or (
            largest_penalty_impact_ratio is not None
            and largest_penalty_impact_ratio["ratio"] >= 2.0
        )
    )
    if any_risk_reduction and materially_different_scale:
        return {
            "outcome": "B",
            "label": "NORMALIZE THE SAFETY PENALTY BEFORE ACTIVE MPC",
            "recommended_aggregation": None,
            "recommended_weight": None,
            "task_score_spread_ratio_across_checkpoints": scale_ratio,
            "largest_penalty_to_score_std_ratio_across_checkpoints": (
                largest_penalty_impact_ratio
            ),
            "reason": (
                "The effective penalty-to-task-score scale differs by at "
                "least 2x across checkpoints for the raw weight sweep, so "
                "one unnormalized lambda is not a stable active-control "
                "setting. Normalization is necessary but not sufficient: "
                "selectivity, false-conservatism, and phase-specific boundary "
                "behavior must all be re-evaluated before active control."
            ),
        }
    false_conservative = any(
        row["no_block_divergence_rate_mean"] is not None
        and row["no_block_divergence_rate_mean"] >= 0.3
        and (
            row["selective_divergence_gap_mean"] is None
            or row["selective_divergence_gap_mean"] <= 0.05
        )
        for _, row in positive_rows
    )
    if any_risk_reduction and false_conservative:
        return {
            "outcome": "D",
            "label": "IMPROVE TRANSLATION CALIBRATION/CONTEXT",
            "recommended_aggregation": None,
            "recommended_weight": None,
            "task_score_spread_ratio_across_checkpoints": scale_ratio,
            "reason": (
                "The penalty reduces predicted risk but changes too many "
                "no-block actions without clear selective divergence."
            ),
        }
    if any_risk_reduction:
        return {
            "outcome": "C",
            "label": "ADD DIRECT FUTURE-BLOCKAGE HEAD",
            "recommended_aggregation": None,
            "recommended_weight": None,
            "task_score_spread_ratio_across_checkpoints": scale_ratio,
            "reason": (
                "Immediate-risk rollout changes planning, but no raw weight "
                "meets the complete cross-checkpoint directional/selectivity "
                "criteria."
            ),
        }
    return {
        "outcome": "E",
        "label": "DO NOT PROCEED",
        "recommended_aggregation": None,
        "recommended_weight": None,
        "task_score_spread_ratio_across_checkpoints": scale_ratio,
        "reason": "No positive raw weight produced useful predicted-risk reduction.",
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Translation-Only Soft Safety-Aware MPC Shadow Evaluation",
        "",
        (
            "Only unchanged baseline TD-MPC2 policy actions were executed. "
            "All Safety-aware actions are shadow outputs."
        ),
        "",
        (
            "Translation Safety is controller action-feasibility risk, not "
            "contact force, vascular injury, or a clinical safety guarantee."
        ),
        "",
        "## Decision",
        "",
        (
            f"**{report['decision']['outcome']}. "
            f"{report['decision']['label']}**"
        ),
        "",
        str(report["decision"]["reason"]),
        "",
        "Curvature remains monitoring-only and has no active-MPC recommendation.",
        "",
        "## Cross-checkpoint Pareto summary",
        "",
    ]
    for aggregation, rows in report["pareto"].items():
        lines.extend(
            [
                f"### {aggregation}",
                "",
                (
                    "| λ | future risk Δ | task sacrifice | future div. | "
                    "no-block div. | policy div. | selectivity | lower dir. | "
                    "tree dir. | dominated |"
                ),
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
            ]
        )
        for row in rows:
            def fmt(value: Optional[float]) -> str:
                return "—" if value is None else f"{float(value):.4g}"

            lines.append(
                "| "
                + " | ".join(
                    (
                        f"{row['weight']:g}",
                        fmt(
                            row[
                                "predicted_future_block_risk_reduction_mean"
                            ]
                        ),
                        fmt(row["task_score_sacrifice_mean"]),
                        fmt(row["future_block_divergence_rate_mean"]),
                        fmt(row["no_block_divergence_rate_mean"]),
                        fmt(row["ordinary_policy_divergence_rate_mean"]),
                        fmt(row["selective_divergence_gap_mean"]),
                        fmt(
                            row[
                                "lower_boundary_directional_improvement_mean"
                            ]
                        ),
                        fmt(
                            row[
                                "tree_end_directional_improvement_mean"
                            ]
                        ),
                        "yes" if row["pareto_dominated"] else "no",
                    )
                )
                + " |"
            )
        lines.append("")
    lines.extend(
        [
            "## Coverage and isolation",
            "",
        ]
    )
    for checkpoint in report["checkpoints"]:
        counts = checkpoint["label_counts"]
        lines.append(
            f"- `{checkpoint['checkpoint_label']}`: "
            f"{counts['planning_calls']} calls, "
            f"{counts['future_block']} valid future-block windows, "
            f"{counts['no_block']} valid no-block windows; details: "
            f"`{checkpoint['details_npz']}`."
        )
    lines.extend(
        [
            "",
            (
                "Every call asserted Python/NumPy/Torch/environment RNG and "
                "planner-state isolation. The focused smoke test additionally "
                "covers replay RNG and optimizer states."
            ),
            "",
            "## Important limitations",
            "",
            (
                "- Real future labels come from the executed behavior "
                "trajectory. They are not counterfactual outcomes of a shadow "
                "action."
            ),
            (
                "- Policy trajectories often contain no positive blockage "
                "window; their main use here is false-conservatism analysis."
            ),
            (
                "- Candidate Safety rollout does not mask risk after a learned "
                "termination prediction."
            ),
            (
                "- Lower-boundary baseline planner outputs were already almost "
                "entirely forward, so a low change-rate there is not evidence "
                "of failure. Tree `several_before` includes long approach "
                "coverage and is not a near-block-only metric; use the "
                "one-before/blocked phase tables for boundary conclusions."
            ),
            (
                "- Runtime is a standalone-evaluator measurement. A fused "
                "active implementation may reuse dynamics computations."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    _validate_args(args)
    checkpoint_reports = []
    reference_environment: Optional[str] = None
    for checkpoint_path in args.checkpoints:
        report, config = evaluate_checkpoint(checkpoint_path, args)
        environment_identity = strict_json_dumps(config["environment"])
        if reference_environment is None:
            reference_environment = environment_identity
        elif environment_identity != reference_environment:
            raise RuntimeError(
                "Checkpoint environment configs differ; identical-state "
                "cross-checkpoint comparison is invalid"
            )
        checkpoint_reports.append(report)
    pareto = build_cross_checkpoint_pareto(
        checkpoint_reports,
        args.aggregation,
        args.weights,
    )
    decision = choose_engineering_decision(pareto, checkpoint_reports)
    report = {
        "schema_version": 1,
        "evaluation": "translation_only_soft_safety_mpc_shadow",
        "semantics": {
            "translation_safety": (
                "future controller action-feasibility risk for intervention "
                "translation constraints"
            ),
            "not_measured": [
                "contact force",
                "vessel-wall collision",
                "guidewire slippage",
                "tissue damage",
                "vascular injury",
                "clinical safety",
            ],
            "executed_action": (
                "unchanged production baseline on policy trajectories; "
                "predefined behavior action in random/boundary trajectories"
            ),
            "shadow_action_executed": False,
            "curvature_used_in_planning": False,
            "intervention_action_mask_changed": False,
        },
        "settings": {
            "checkpoints": [
                str(path.expanduser().resolve()) for path in args.checkpoints
            ],
            "policy_episodes": int(args.policy_episodes),
            "random_episodes": int(args.random_episodes),
            "lower_boundary_repetitions": int(
                args.lower_boundary_repetitions
            ),
            "tree_end_repetitions": int(args.tree_end_repetitions),
            "seeds": [int(value) for value in args.seeds],
            "planner_seed": int(args.planner_seed),
            "weights": [float(value) for value in args.weights],
            "aggregations": list(args.aggregation),
            "safety_discount": float(args.safety_discount),
            "translation_risk_cap": float(args.translation_risk_cap),
            "diagnostic_threshold_reporting_only": float(
                args.diagnostic_threshold
            ),
            "translation_action_tolerance": float(
                args.translation_action_tolerance
            ),
            "rotation_action_tolerance": float(
                args.rotation_action_tolerance
            ),
            "max_planning_calls_per_checkpoint": (
                None
                if args.max_planning_calls_per_checkpoint is None
                else int(args.max_planning_calls_per_checkpoint)
            ),
        },
        "risk_formula": {
            "channel": "normalized_requested_applied_translation_error",
            "per_step": "finite nonnegative decoded prediction",
            "bounded_per_step": "clamp(prediction, 0, translation_risk_cap)",
            "max": "max_k bounded_per_step[k]",
            "discounted_sum": (
                "sum_k safety_discount**k * bounded_per_step[k]"
            ),
            "score": "task_value - weight * bounded_trajectory_risk",
            "hard_threshold_used": False,
        },
        "mppi_fidelity": {
            "shared_exogenous_noise": [
                "policy-prior Gaussian",
                "per-iteration candidate Gaussian",
                "terminal-policy Gaussian",
                "Q-ensemble randperm indices",
            ],
            "independent_mean_and_std_updates_per_weight": True,
            "warm_start_cloned_from_pre_call_previous_mean": True,
            "selection": "elite exponential weighting and weighted mean",
            "not_final_population_argmax": True,
            "weight_zero_production_identity_asserted_per_policy_call": True,
        },
        "checkpoints": checkpoint_reports,
        "pareto": pareto,
        "decision": decision,
        "normal_tdmpc2_behavior": {
            "agent_source_modified": False,
            "training_source_modified": False,
            "ordinary_plan_imports_shadow_utility": False,
            "ordinary_planning_safety_calls_added": False,
            "shadow_disabled_overhead": "zero",
        },
    }
    output_json = args.output_json.expanduser().resolve()
    output_markdown = args.output_markdown.expanduser().resolve()
    write_strict_json(output_json, report)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(render_markdown(report), encoding="utf-8")
    print(strict_json_dumps({
        "output_json": str(output_json),
        "output_markdown": str(output_markdown),
        "decision": decision,
        "planning_calls": {
            item["checkpoint_label"]: item["label_counts"]["planning_calls"]
            for item in checkpoint_reports
        },
    }))


if __name__ == "__main__":
    main()
