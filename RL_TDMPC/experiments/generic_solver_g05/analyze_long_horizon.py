"""Merge G0.5 shards and summarize force, penetration, solver and geometry traces."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write an empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def values(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def stats(array: np.ndarray) -> dict[str, float]:
    return {"median": float(np.median(array)), "q95": float(np.quantile(array, .95)),
            "q99": float(np.quantile(array, .99)), "max": float(np.max(array))}


def longest_run(flags: np.ndarray) -> tuple[int, int]:
    longest = current = episodes = 0
    for flag in flags:
        if flag:
            current += 1
            if current == 1:
                episodes += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest, episodes


def analyze_case(rows: list[dict], *, family: str, seed: int) -> dict:
    rows.sort(key=lambda row: int(row["step"]))
    steps = [int(row["step"]) for row in rows]
    if steps != list(range(1, len(rows)+1)):
        raise RuntimeError(f"Missing or repeated control steps in {family}/{seed}")
    force = values(rows, "distal_force_N")
    global_force = values(rows, "global_contact_reaction_N")
    distal_pen = values(rows, "distal_penetration_max_mm")
    global_pen = values(rows, "global_penetration_max_mm")
    signed_nearest_wall = values(rows, "signed_exterior_max_mm")
    # The mechanical beam radius varies from 0.35 mm at the tip to 0.445 mm
    # proximally. The collision model itself is a zero-radius line/point set.
    # Radius offsets are a sampled spherical-envelope diagnostic, not measured
    # tissue/beam intersection. The .445 mm offset bounds the signed exterior
    # distance of any point within that radius of a sampled centerline point.
    tip_radius_offset = np.maximum(signed_nearest_wall + .35, 0.0)
    max_radius_offset = np.maximum(signed_nearest_wall + .445, 0.0)
    contact = values(rows, "contact_groups")
    iterations = values(rows, "solver_iterations")
    solver_error = values(rows, "solver_error")
    scaled_error = values(rows, "solver_error_scaled_ratio")
    force_q95 = float(np.quantile(force, .95))
    nonzero_force = force > 1e-10
    high_force = force > force_q95
    distal_outside = distal_pen > 1e-6
    global_outside = global_pen > 1e-6
    longest_nonzero, nonzero_runs = longest_run(nonzero_force)
    longest_high, high_runs = longest_run(high_force)
    distal_longest, distal_runs = longest_run(distal_outside)
    global_longest, global_runs = longest_run(global_outside)
    q99 = float(np.quantile(force, .99))
    extreme_indices = np.flatnonzero(force >= q99) if q99 > 0 else np.asarray([], dtype=int)
    contact_changes = np.r_[False, contact[1:] != contact[:-1]]
    tip = np.column_stack([values(rows, f"tip_{axis}_mm") for axis in "xyz"])
    quarter = max(1, len(rows)//4)
    if np.std(force) > 0 and np.std(global_pen) > 0:
        correlation = float(np.corrcoef(force, global_pen)[0, 1])
    else:
        correlation = None
    return {
        "family": family, "seed": seed, "steps": len(rows),
        "simulated_time_s": float(rows[-1]["simulated_time_s"]),
        "nominal_horizon_reached": any(row["truncated"] == "True" for row in rows),
        "success_any": any(row["success"] == "True" for row in rows),
        "nonfinite_steps": sum(row["all_finite"] != "True" for row in rows),
        "simulation_error_steps": sum(row["simulation_error"] == "True" for row in rows),
        "contact_steps": int(np.count_nonzero(contact)),
        "max_simultaneous_contact_groups": int(np.max(contact)),
        "force_median_N": stats(force)["median"], "force_q95_N": stats(force)["q95"],
        "force_q99_N": stats(force)["q99"], "force_max_N": stats(force)["max"],
        "global_reaction_max_N": float(np.max(global_force)),
        "longest_nonzero_force_run_steps": longest_nonzero,
        "longest_nonzero_force_run_s": longest_nonzero * .132,
        "nonzero_force_runs": nonzero_runs,
        "high_force_definition": "within-trajectory > Q95 force",
        "longest_high_force_run_steps": longest_high,
        "high_force_runs": high_runs,
        "force_without_contact_steps": sum(row["force_without_contact"] == "True" for row in rows),
        "first_quarter_force_median_N": float(np.median(force[:quarter])),
        "last_quarter_force_median_N": float(np.median(force[-quarter:])),
        "extreme_force_steps_at_or_above_q99": int(len(extreme_indices)),
        "extreme_force_steps_with_contact_change": int(np.count_nonzero(contact_changes[extreme_indices])),
        "extreme_force_steps_with_global_penetration": int(np.count_nonzero(global_outside[extreme_indices])),
        "global_penetration_max_mm": stats(global_pen)["max"],
        "global_tip_radius_offset_proxy_max_mm": float(np.max(tip_radius_offset)),
        "global_max_radius_offset_bound_max_mm": float(np.max(max_radius_offset)),
        "global_max_radius_offset_bound_q95_mm": float(np.quantile(max_radius_offset, .95)),
        "global_penetration_median_stepmax_mm": stats(global_pen)["median"],
        "global_penetration_q95_stepmax_mm": stats(global_pen)["q95"],
        "global_penetration_q99_stepmax_mm": stats(global_pen)["q99"],
        "global_penetration_mean_sample_mm_max": float(np.max(values(rows, "global_penetration_mean_mm"))),
        "global_penetration_q95_sample_mm_max": float(np.max(values(rows, "global_penetration_q95_mm"))),
        "global_penetrating_steps": int(np.count_nonzero(global_outside)),
        "global_longest_penetrating_run_steps": global_longest,
        "global_longest_penetrating_run_s": global_longest*.132,
        "global_penetrating_runs": global_runs,
        "global_max_penetrating_samples_simultaneous": int(np.max(values(rows, "global_penetrating_samples"))),
        "distal_penetration_max_mm": stats(distal_pen)["max"],
        "distal_penetration_median_stepmax_mm": stats(distal_pen)["median"],
        "distal_penetration_q95_stepmax_mm": stats(distal_pen)["q95"],
        "distal_penetration_q99_stepmax_mm": stats(distal_pen)["q99"],
        "distal_penetrating_steps": int(np.count_nonzero(distal_outside)),
        "distal_longest_penetrating_run_steps": distal_longest,
        "distal_longest_penetrating_run_s": distal_longest*.132,
        "distal_penetrating_runs": distal_runs,
        "distal_max_penetrating_samples_simultaneous": int(np.max(values(rows, "distal_penetrating_samples"))),
        "max_contact_groups_with_exterior_nodes": int(np.max(values(rows, "contact_groups_with_exterior_nodes"))),
        "solver_iterations_median": stats(iterations)["median"],
        "solver_iterations_q95": stats(iterations)["q95"],
        "solver_iterations_q99": stats(iterations)["q99"],
        "solver_iterations_max": stats(iterations)["max"],
        "solver_error_median": stats(solver_error)["median"],
        "solver_error_q95": stats(solver_error)["q95"],
        "solver_error_q99": stats(solver_error)["q99"],
        "solver_error_max": stats(solver_error)["max"],
        "solver_scaled_error_ratio_max": float(np.max(scaled_error)),
        "solver_iteration_cap_steps": int(np.count_nonzero(iterations >= 2000)),
        "solver_scaled_error_exceedance_steps": int(np.count_nonzero(scaled_error > 1)),
        "max_parent_quaternion_norm_error": float(np.max(values(rows, "max_parent_quaternion_norm_error"))),
        "max_parent_tip_to_collision_tip_mm": float(np.max(values(rows, "parent_tip_collision_tip_mismatch_mm"))),
        "max_collision_edge_mm": float(np.max(values(rows, "max_collision_edge_mm"))),
        "min_nonzero_collision_edge_mm": float(np.min(values(rows, "min_active_collision_edge_mm")[values(rows, "min_active_collision_edge_mm") > 0])) if np.any(values(rows, "min_active_collision_edge_mm") > 0) else 0.0,
        "max_collision_turn_deg": float(np.max(values(rows, "max_collision_turn_deg"))),
        "initial_inserted_length_mm": float(rows[0]["inserted_length_mm"]),
        "final_inserted_length_mm": float(rows[-1]["inserted_length_mm"]),
        "initial_active_collision_length_mm": float(rows[0]["active_collision_length_mm"]),
        "final_active_collision_length_mm": float(rows[-1]["active_collision_length_mm"]),
        "tip_net_displacement_mm": float(np.linalg.norm(tip[-1] - tip[0])),
        "tip_cumulative_travel_mm": float(np.sum(np.linalg.norm(np.diff(tip, axis=0), axis=1))),
        "force_global_penetration_pearson": correlation,
    }


def plot_case(rows: list[dict], label: str, root: Path) -> None:
    time = values(rows, "simulated_time_s")
    distal_force = values(rows, "distal_force_N")
    distal_pen = values(rows, "distal_penetration_max_mm")
    global_pen = values(rows, "global_penetration_max_mm")
    radius_offset_bound = np.maximum(values(rows, "signed_exterior_max_mm") + .445, 0.0)
    groups = values(rows, "contact_groups")
    folder = root / "plots"
    for child in ("force_time_series", "penetration_time_series", "force_vs_penetration", "representative_failures"):
        (folder / child).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(time, distal_force, lw=.9, color="tab:red")
    ax.set(xlabel="Simulated time (s)", ylabel="Distal reaction (nominal N)", title=label)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(folder / "force_time_series" / f"{label}.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(time, global_pen, label="global", lw=.9)
    ax.plot(time, distal_pen, label="distal 20 mm", lw=.9)
    ax.plot(time, radius_offset_bound, label="sampled max-radius offset bound", lw=.9,
            ls="--", color="tab:orange")
    ax.set(xlabel="Simulated time (s)", ylabel="Centerline wall crossing (mm)", title=label)
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(folder / "penetration_time_series" / f"{label}.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True,
                             layout="constrained")
    scatter = axes[0].scatter(global_pen, distal_force, s=8, c=groups,
                              cmap="viridis", alpha=.7)
    axes[0].set(xlabel="Centerline wall crossing (mm)",
                ylabel="Distal reaction (nominal N)")
    axes[1].scatter(radius_offset_bound, distal_force, s=8, c=groups,
                    cmap="viridis", alpha=.7)
    axes[1].set(xlabel="Sampled max-radius offset bound (mm)")
    fig.colorbar(scatter, ax=axes, label="Active contact groups", shrink=.8)
    fig.suptitle(label)
    fig.savefig(folder / "force_vs_penetration" / f"{label}.png", dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    scripted_paths = sorted((root / "shards").glob("scripted_*/long_horizon_steps.csv"))
    random_paths = sorted((root / "shards").glob("random_independent_*/randomized_steps.csv"))
    if not scripted_paths or not random_paths:
        raise RuntimeError("Scripted or random shard files missing")
    scripted = [row for path in scripted_paths for row in read_csv(path)]
    random = [row for path in random_paths for row in read_csv(path)]
    scripted_groups = defaultdict(list)
    random_groups = defaultdict(list)
    for row in scripted:
        scripted_groups[(row["family"], int(row["seed"]))].append(row)
    for row in random:
        random_groups[(row["family"], int(row["seed"]))].append(row)
    if len(scripted_groups) != 7 or len(random_groups) != 50:
        raise RuntimeError(f"Need 7 scripted families and 50 random seeds; got {len(scripted_groups)}, {len(random_groups)}")
    if sorted(seed for _, seed in random_groups) != list(range(91000, 91050)):
        raise RuntimeError("Randomized seed set does not match the frozen 91000..91049 plan")
    substep_path = root / "substep_contact_heavy_summary.json"
    substep = json.loads(substep_path.read_text(encoding="utf-8"))
    if not (substep["status"] == "COMPLETE" and
            substep["passive_monitor_behaviorally_equivalent_at_endpoints"] and
            substep["all_finite"] and substep["control_steps"] == 344 and
            substep["sofa_substeps"] == 344*22):
        raise RuntimeError("Matched-lifecycle contact-heavy substep audit did not pass")
    write_csv(root / "long_horizon_steps.csv", scripted)
    write_csv(root / "randomized_steps.csv", random)
    scripted_summary = [analyze_case(rows, family=family, seed=seed)
                        for (family, seed), rows in sorted(scripted_groups.items())]
    random_summary = [analyze_case(rows, family=family, seed=seed)
                      for (family, seed), rows in sorted(random_groups.items())]
    write_csv(root / "long_horizon_results.csv", scripted_summary)
    write_csv(root / "randomized_seed_results.csv", random_summary)
    write_json(root / "randomized_seed_plan.json", {
        "environment_seeds": list(range(91000, 91050)),
        "action_rng_seed_rule": "environment_seed + 730000",
        "action_program": "Every 20 steps sample insertion target uniformly in [40,145] mm; each step translation drives toward target clipped to [-0.3,0.3], rotation sampled uniformly in [-0.5,0.5].",
        "nominal_horizon_steps": 200,
        "policy_used": False,
        "scene_lifecycle": "One freshly created SOFA scene per seed. The interrupted reused-scene shard is preserved as diagnostic data and excluded.",
        "shards": [str(path.parent.relative_to(root)) for path in random_paths],
    })
    for family in ("contact_heavy", "alternating_translation", "translation_plus_rotation"):
        rows = next(rows for (name, _), rows in scripted_groups.items() if name == family)
        plot_case(rows, family, root)
    worst_force = max(random_summary, key=lambda row: row["force_max_N"])
    worst_pen = max(random_summary, key=lambda row: row["global_penetration_max_mm"])
    for seed in {worst_force["seed"], worst_pen["seed"]}:
        plot_case(random_groups[("randomized", seed)], f"randomized_{seed}", root)
    all_rows = scripted + random
    worst_error = max(all_rows, key=lambda row: float(row["solver_error_scaled_ratio"]))
    run = (worst_error["family"], int(worst_error["seed"]))
    run_rows = scripted_groups.get(run, random_groups.get(run))
    step = int(worst_error["step"])
    window = [row for row in run_rows if abs(int(row["step"]) - step) <= 20]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(values(window, "step"), values(window, "solver_error_scaled_ratio"), label="scaled residual ratio")
    ax.plot(values(window, "step"), values(window, "distal_force_N"), label="distal force (N)")
    ax.set(xlabel="Control step", title=f"Worst observed solver stress: {run[0]} seed {run[1]}")
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(root / "plots" / "representative_failures" / "worst_solver_stress.png", dpi=140)
    plt.close(fig)

    def maxima(key: str, summaries: list[dict]) -> float:
        return max(float(row[key]) for row in summaries)

    write_json(root / "penetration_summary.json", {
        "metric": "positive = mapped centerline outside closed vessel surface; mm",
        "sampling": "all collision vertices plus active edge midpoints at RL-control endpoints",
        "scripted": {row["family"]: {key: row[key] for key in (
            "distal_penetration_max_mm", "distal_penetration_q95_stepmax_mm", "distal_penetrating_steps",
            "distal_longest_penetrating_run_s", "global_penetration_max_mm",
            "global_penetration_q95_stepmax_mm", "global_penetrating_steps",
            "global_longest_penetrating_run_s", "global_max_penetrating_samples_simultaneous",
            "global_tip_radius_offset_proxy_max_mm",
            "global_max_radius_offset_bound_max_mm",
            "global_max_radius_offset_bound_q95_mm")}
            for row in scripted_summary},
        "randomized": {"seeds": 50, "seeds_with_global_penetration": sum(row["global_penetrating_steps"] > 0 for row in random_summary),
                       "seeds_with_distal_penetration": sum(row["distal_penetrating_steps"] > 0 for row in random_summary),
                       "maximum_global_mm": maxima("global_penetration_max_mm", random_summary),
                       "maximum_distal_mm": maxima("distal_penetration_max_mm", random_summary),
                       "maximum_global_tip_radius_offset_proxy_mm": maxima("global_tip_radius_offset_proxy_max_mm", random_summary),
                       "maximum_global_max_radius_offset_bound_mm": maxima("global_max_radius_offset_bound_max_mm", random_summary)},
        "finite_radius_offset_definition": "At each endpoint, clamp(max sampled signed exterior centerline distance plus 0.35 or 0.445 mm, 0). The 0.35 mm term is only a tip-radius spherical-offset proxy, NOT a lower bound on true beam-surface overlap. The 0.445 mm term upper-bounds possible signed exterior excursion for points within 0.445 mm of each sampled centerline point, but is not a measured overlap or a bound along unsampled continuous segments. Actual beam cross-section orientation and full collision geometry are not resolved.",
        "contact_heavy_substep_audit": {key: substep[key] for key in (
            "sofa_substeps", "all_substeps_max_centerline_crossing_mm",
            "substeps_with_centerline_crossing", "passive_monitor_behaviorally_equivalent_at_endpoints")},
    })
    write_json(root / "force_stability_summary.json", {
        "units": "nominal N after J_contact^T lambda / 0.006 s",
        "high_force_run_definition": "consecutive control endpoints above each trajectory's own Q95 force; descriptive, not a safety threshold",
        "pooled_scripted_endpoint_force_N": stats(values(scripted, "distal_force_N")),
        "pooled_randomized_endpoint_force_N": stats(values(random, "distal_force_N")),
        "scripted": {row["family"]: {key: row[key] for key in (
            "force_median_N", "force_q95_N", "force_q99_N", "force_max_N",
            "longest_nonzero_force_run_s", "longest_high_force_run_steps",
            "force_without_contact_steps", "first_quarter_force_median_N", "last_quarter_force_median_N",
            "extreme_force_steps_at_or_above_q99", "extreme_force_steps_with_contact_change",
            "extreme_force_steps_with_global_penetration")}
            for row in scripted_summary},
        "randomized": {"maximum_force_N": maxima("force_max_N", random_summary),
                       "seeds_with_force_without_contact": sum(row["force_without_contact_steps"] > 0 for row in random_summary)},
        "contact_heavy_substep_audit": {key: substep[key] for key in (
            "all_substeps_max_distal_force_N", "endpoint_max_distal_force_N",
            "substeps_force_without_contact")},
    })
    write_json(root / "geometry_integrity_summary.json", {
        "scripted": {row["family"]: {key: row[key] for key in (
            "initial_inserted_length_mm", "final_inserted_length_mm",
            "initial_active_collision_length_mm", "final_active_collision_length_mm",
            "max_collision_edge_mm", "min_nonzero_collision_edge_mm", "max_collision_turn_deg",
            "max_parent_quaternion_norm_error", "max_parent_tip_to_collision_tip_mm",
            "tip_net_displacement_mm", "tip_cumulative_travel_mm")}
            for row in scripted_summary},
        "randomized_max_quaternion_norm_error": maxima("max_parent_quaternion_norm_error", random_summary),
        "randomized_max_parent_tip_to_collision_tip_mm": maxima("max_parent_tip_to_collision_tip_mm", random_summary),
        "nonfinite_cases": sum(row["nonfinite_steps"] > 0 for row in scripted_summary+random_summary),
        "self_intersection": "Not directly measured; sampled curvature/segment length and finite/mapping checks do not prove absence of self-intersection.",
    })
    write_json(root / "solver_convergence_summary.json", {
        "scaleTolerance": True, "tolerance": 1e-4, "maxIterations": 2000,
        "pooled_scripted_endpoint_iterations": stats(values(scripted, "solver_iterations")),
        "pooled_randomized_endpoint_iterations": stats(values(random, "solver_iterations")),
        "scripted": {row["family"]: {key: row[key] for key in (
            "solver_iterations_median", "solver_iterations_q95", "solver_iterations_q99",
            "solver_iterations_max", "solver_error_median", "solver_error_q95",
            "solver_error_q99", "solver_error_max", "solver_scaled_error_ratio_max",
            "solver_iteration_cap_steps", "solver_scaled_error_exceedance_steps")}
            for row in scripted_summary},
        "randomized_cap_hit_seeds": sum(row["solver_iteration_cap_steps"] > 0 for row in random_summary),
        "randomized_scaled_error_exceedance_seeds": sum(row["solver_scaled_error_exceedance_steps"] > 0 for row in random_summary),
        "randomized_max_iterations": maxima("solver_iterations_max", random_summary),
        "randomized_max_scaled_error_ratio": maxima("solver_scaled_error_ratio_max", random_summary),
        "randomized_nonfinite_or_simulation_error_seeds": sum(row["nonfinite_steps"] > 0 or row["simulation_error_steps"] > 0 for row in random_summary),
        "contact_heavy_substep_max_iterations": substep["max_solver_iterations"],
        "contact_heavy_substep_max_scaled_error_ratio": substep["max_scaled_solver_error_ratio"],
    })
    print(f"Merged {len(scripted_summary)} scripted trajectories and {len(random_summary)} randomized seeds", flush=True)


if __name__ == "__main__":
    main()
