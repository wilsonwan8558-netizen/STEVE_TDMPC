"""Pair open-loop solver traces and write G0 numerical diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    folder = args.output_dir.resolve()
    generic = json.loads((folder / "generic_rollout.json").read_text(encoding="utf-8"))
    lcp = json.loads((folder / "lcp_rollout.json").read_text(encoding="utf-8"))
    if set(generic["scenarios"]) != set(lcp["scenarios"]):
        raise RuntimeError("LCP/Generic open-loop scenarios do not match")
    descriptive = []
    stability = {"status": "COMPLETE_OPEN_LOOP_SCREEN",
                 "sample_timing": "solver fields sampled after the final SOFA substep of each RL action; within-control-interval peaks not observed",
                 "penetration_limit": "Exact guidewire-vessel signed surface penetration not measured by this screen",
                 "geometry_screen": "finite DOFs, maximum absolute coordinate, adjacent parent DOF spacing, tip jump, and simulation_error flag",
                 "scenarios": {}}
    for scenario in generic["scenarios"]:
        g = generic["scenarios"][scenario]
        l = lcp["scenarios"][scenario]
        gt, lt = g["trajectory"], l["trajectory"]
        if len(gt) != len(lt) or any(a["action"] != b["action"] for a, b in zip(gt, lt)):
            raise RuntimeError(f"Open-loop action/length pairing changed: {scenario}")
        tip_distance = [float(np.linalg.norm(np.asarray(a["tip_position_mm"])-np.asarray(b["tip_position_mm"])))
                        for a, b in zip(gt, lt)]
        residual_ratios = [a["currentError"]/(1e-4*max(a["currentNumConstraints"], 1))
                           for a in gt if a["currentNumConstraints"] is not None]
        stability["scenarios"][scenario] = {
            "steps": len(gt), "generic_all_finite": g["finite_all_steps"],
            "generic_simulation_error_any": g["simulation_error_any"],
            "generic_max_tip_displacement_mm": g["max_tip_displacement_mm"],
            "generic_max_adjacent_parent_dof_spacing_mm": g["max_adjacent_parent_dof_spacing_mm"],
            "generic_max_abs_parent_position_mm": max(r["max_abs_parent_position_mm"] for r in gt),
            "generic_max_multiplier": g["max_abs_constraint_multiplier"],
            "generic_max_current_iterations": g["max_solver_iterations"],
            "generic_max_current_error": g["max_solver_residual"],
            "generic_max_error_divided_by_scaled_tolerance": max(residual_ratios),
            "generic_contact_sampled_steps": g["contact_sampled_steps"],
            "terminated": g["terminated"], "truncated": g["truncated"],
            "pathology_flags": {
                "nonfinite": not g["finite_all_steps"],
                "simulation_error": g["simulation_error_any"],
                "tip_jump_gt_25_mm_per_RL_action": g["max_tip_displacement_mm"] > 25.,
                "adjacent_parent_dof_spacing_gt_30_mm": g["max_adjacent_parent_dof_spacing_mm"] > 30.,
                "abs_parent_coordinate_gt_1000_mm": max(r["max_abs_parent_position_mm"] for r in gt) > 1000.,
                "iterations_reached_2000": g["max_solver_iterations"] >= 2000}}
        descriptive.append({"scenario": scenario, "seed": g["seed"], "paired_steps": len(gt),
                            "LCP_sampled_contact_steps": l["contact_sampled_steps"],
                            "Generic_sampled_contact_steps": g["contact_sampled_steps"],
                            "LCP_max_raw_multiplier": l["max_abs_constraint_multiplier"],
                            "Generic_max_raw_multiplier": g["max_abs_constraint_multiplier"],
                            "LCP_final_insertion_mm": l["final_insertion_length_mm"],
                            "Generic_final_insertion_mm": g["final_insertion_length_mm"],
                            "final_tip_distance_mm": tip_distance[-1],
                            "mean_tip_distance_mm": float(np.mean(tip_distance)),
                            "max_tip_distance_mm": max(tip_distance),
                            "LCP_total_reward": sum(r["reward"] for r in lt),
                            "Generic_total_reward": sum(r["reward"] for r in gt),
                            "LCP_success": l["success"], "Generic_success": g["success"],
                            "both_finite": g["finite_all_steps"] and l["finite_all_steps"]})
    with (folder / "lcp_vs_generic_diagnostic.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(descriptive[0]))
        writer.writeheader()
        writer.writerows(descriptive)
    (folder / "numerical_stability_results.json").write_text(
        json.dumps(stability, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: {"contact": v["generic_contact_sampled_steps"],
                           "max_iterations": v["generic_max_current_iterations"],
                           "pathology": v["pathology_flags"]}
                      for k, v in stability["scenarios"].items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
