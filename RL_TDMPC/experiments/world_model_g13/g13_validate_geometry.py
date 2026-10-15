"""Validate the frozen G13 geometry features on already-used G2 rollouts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
sys.path.insert(0, str(RL))
sys.path.insert(0, str(RL.parent))
sys.path.insert(0, str(RL / "experiments/generic_solver_g2"))
from envs.steve_env import make_steve_env  # noqa: E402
from envs.geometry_features import GeometryFeatureExtractor, OBSERVATION_DIM  # noqa: E402
from tdmpc2.common import set_seed  # noqa: E402
from collect import check_scene  # noqa: E402

OUT = RL / "logs/world_model_force_aware_planning_g13"
G2 = RL / "logs/generic_solver_phase_g2_force_baseline/step_level_force.csv"
SEEDS = tuple(range(90000, 90010))


def main() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("G13 validation requires CUDA_VISIBLE_DEVICES=0")
    by_seed = {seed: [] for seed in SEEDS}
    with G2.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            seed = int(row["seed"])
            if seed in by_seed:
                by_seed[seed].append(row)
    if any(not by_seed[seed] for seed in SEEDS):
        raise RuntimeError("Archived validation seed rows missing")
    set_seed(SEEDS[0])
    env = make_steve_env()
    extractor = GeometryFeatureExtractor()
    rows = []
    max_repeated_difference = 0.0
    try:
        for seed in SEEDS:
            set_seed(seed)
            env.action_space.seed(seed)
            observation, _ = env.reset(seed=seed)
            check_scene(env)
            previous = None
            for decision, reference in enumerate(by_seed[seed]):
                archived = np.asarray(json.loads(reference["decision_observation_json"]), dtype=np.float32)
                if not np.array_equal(observation, archived):
                    raise RuntimeError(f"G13 pre-action archived observation mismatch {seed}/{decision}")
                result = extractor.compute(env)
                repeat = extractor.compute(env)
                difference = float(np.max(np.abs(result.normalized - repeat.normalized)))
                max_repeated_difference = max(max_repeated_difference, difference)
                if difference != 0:
                    raise RuntimeError(f"Nondeterministic current-geometry query {seed}/{decision}")
                augmented = np.concatenate((observation, result.normalized))
                if augmented.shape != (OBSERVATION_DIM,) or not np.all(np.isfinite(augmented)):
                    raise RuntimeError(f"Invalid G13 observation {seed}/{decision}")
                continuity = float(np.linalg.norm(result.normalized - previous.normalized)) if previous is not None else 0.0
                row = {
                    "seed": seed, "decision_index": decision,
                    "representative_labels": "",
                    "d0_mm": float(result.clearance_mm[0]),
                    "d1_mm": float(result.clearance_mm[1]),
                    "d2_mm": float(result.clearance_mm[2]),
                    "d0_normalized": float(result.normalized[0]),
                    "d1_normalized": float(result.normalized[1]),
                    "d2_normalized": float(result.normalized[2]),
                    "sin_delta_theta": result.sin_delta_theta,
                    "cos_delta_theta": result.cos_delta_theta,
                    "bend_angle_rad": result.bend_angle_rad,
                    "kappa_distal_normalized": float(result.normalized[5]),
                    "r_local_mm": result.radius_mm,
                    "r_local_normalized": float(result.normalized[6]),
                    "direction_valid": int(result.direction_valid),
                    "selected_branch": result.selected_branch,
                    "selected_segment": result.selected_segment,
                    "tracking_alignment_error_mm": result.tracking_alignment_error_mm,
                    "normalized_geometry_step_L2_change": continuity,
                    "d0_step_change_mm": float(result.clearance_mm[0] - previous.clearance_mm[0]) if previous is not None else 0.0,
                    "r_local_step_change_mm": float(result.radius_mm - previous.radius_mm) if previous is not None else 0.0,
                    "tip_force_next_max_N_diagnostic_only": float(reference["distal_contact_force_max_N"]),
                    "all_finite": True, "repeat_query_exact": True,
                    "pre_action_only": True, "archived_parent_observation_exact": True,
                }
                rows.append(row)
                action = np.asarray(json.loads(reference["selected_action_json"]), dtype=np.float32)
                next_observation, reward, terminated, truncated, info = env.step(action)
                if (not np.array_equal(next_observation, np.asarray(json.loads(reference["next_observation_json"]), dtype=np.float32)) or
                        abs(float(reward) - float(reference["reward"])) > 1e-9 or
                        bool(terminated) != (reference["terminated"] == "True") or
                        bool(truncated) != (reference["truncated"] == "True") or
                        not np.array_equal(np.asarray(info["raw_action"], dtype=np.float32),
                                           np.asarray(json.loads(reference["applied_action_json"]), dtype=np.float32))):
                    raise RuntimeError(f"G13 geometry validation changed archived transition {seed}/{decision}")
                observation = np.asarray(next_observation, dtype=np.float32)
                previous = result
            print("G13_GEOMETRY_VALIDATION_SEED", seed, len(by_seed[seed]), flush=True)
    finally:
        env.close()
    valid = [r for r in rows if r["direction_valid"]]
    if len(valid) < 100:
        raise RuntimeError("Too few valid distal angle states")
    exemplar_indices = {
        "center_lumen": max(range(len(rows)), key=lambda i: rows[i]["d0_mm"]),
        "near_wall": min(range(len(rows)), key=lambda i: abs(rows[i]["d0_mm"])),
        "bent_distal": max(range(len(rows)), key=lambda i: rows[i]["bend_angle_rad"]),
        "aligned": max((i for i, r in enumerate(rows) if r["direction_valid"]),
                       key=lambda i: rows[i]["cos_delta_theta"]),
        "wall_directed": max((i for i, r in enumerate(rows) if r["direction_valid"]),
                             key=lambda i: abs(rows[i]["sin_delta_theta"])),
        "direction_degenerate": next(i for i, r in enumerate(rows) if not r["direction_valid"]),
    }
    for label, index in exemplar_indices.items():
        rows[index]["representative_labels"] += ("|" if rows[index]["representative_labels"] else "") + label
    with (OUT / "geometry_feature_validation.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    changes = np.asarray([r["normalized_geometry_step_L2_change"] for r in rows])
    clearance = np.asarray([r["d0_mm"] for r in rows])
    with (OUT / "geometry_feature_validation_summary.json").open("x", encoding="utf-8") as stream:
        json.dump({
            "status": "PASS_FINITE_DETERMINISTIC_NO_FUTURE_READ_AND_EXACT_G2_PARENT_REPLAY",
            "seeds": list(SEEDS), "states": len(rows),
            "direction_valid_states": len(valid),
            "max_tracking_alignment_error_mm": max(r["tracking_alignment_error_mm"] for r in rows),
            "max_repeat_query_difference": max_repeated_difference,
            "tip_clearance_mm_range": [float(clearance.min()), float(clearance.max())],
            "normalized_geometry_step_L2_change_q50_q95_q99_max": [
                float(np.quantile(changes, q)) for q in (.5, .95, .99, 1)],
            "representatives": {label: {"seed": rows[i]["seed"], "decision_index": rows[i]["decision_index"]}
                                for label, i in exemplar_indices.items()},
            "force_column_usage": "diagnostic only for representative-state interpretation; no force input enters geometry features",
        }, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print("G13_GEOMETRY_VALIDATION_COMPLETE", len(rows), exemplar_indices, flush=True)


if __name__ == "__main__":
    main()
