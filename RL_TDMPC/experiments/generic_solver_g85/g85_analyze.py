"""Read-only analysis of completed G8.5 true-SOFA oracle forks (no predictor)."""

from __future__ import annotations

import csv
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
OUT = RL / "logs/generic_solver_phase_g85_continuous_oracle_feasibility"
G8_STATES = RL / "logs/generic_solver_phase_g8_offline_quantile_shield/confirmation_candidates"
F_LIM = 3.685331040756903
FIELDS = ("state_id", "episode_seed", "decision_index", "translation", "rotation", "F_true_N", "true_safe")


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def cast(row):
    return {**row, "state_id": int(row["state_id"]), "episode_seed": int(row["episode_seed"]),
        "decision_index": int(row["decision_index"]), "translation": float(row["translation"]),
        "rotation": float(row["rotation"]), "F_true_N": float(row["F_true_N"]),
        "true_safe": int(row["true_safe"])}


def csv_new(path, rows, fields=None):
    if not rows:
        raise RuntimeError(f"No rows for {path}")
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def json_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def xy(row):
    return (row["translation"], row["rotation"])


def min_force(rows):
    return min(rows, key=lambda r: (r["F_true_N"], r["translation"], r["rotation"]))


def nearest_safe(rows, nominal):
    safe = [r for r in rows if r["true_safe"]]
    return min(safe, key=lambda r: ((r["translation"]-nominal[0])**2 +
        (r["rotation"]-nominal[1])**2, r["translation"], r["rotation"])) if safe else None


def connected_components(mask):
    seen = set()
    sizes = []
    for i, j in zip(*np.where(mask)):
        if (int(i), int(j)) in seen:
            continue
        queue = deque([(int(i), int(j))])
        seen.add((int(i), int(j)))
        size = 0
        while queue:
            x, y = queue.popleft()
            size += 1
            for nx, ny in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
                if 0 <= nx < 41 and 0 <= ny < 41 and mask[nx,ny] and (nx,ny) not in seen:
                    seen.add((nx,ny))
                    queue.append((nx,ny))
        sizes.append(size)
    return sorted(sizes, reverse=True)


def plot_landscape(state_id, matrix, nominal, nearest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6,5))
    vmax = max(F_LIM, float(np.quantile(matrix, 0.99)))
    im = ax.imshow(matrix, origin="lower", extent=(-1,1,-1,1), aspect="auto",
        vmin=0, vmax=vmax, interpolation="nearest")
    if float(np.min(matrix)) <= F_LIM <= float(np.max(matrix)):
        ax.contour(np.linspace(-1,1,41), np.linspace(-1,1,41), matrix,
            levels=[F_LIM], colors="white", linewidths=1)
    ax.scatter([nominal[1]], [nominal[0]], marker="x", color="red", s=55, label="nominal")
    ax.scatter([0], [-1], marker="+", color="cyan", s=65, label="strong withdrawal")
    if nearest is not None:
        ax.scatter([nearest["rotation"]], [nearest["translation"]],
            marker="o", facecolors="none", edgecolors="white", s=70, label="nearest sampled safe")
    ax.set_xlabel("rotation action")
    ax.set_ylabel("translation action")
    ax.set_title(f"G8.5 state {state_id}: true next-interval distal peak force")
    ax.legend(loc="upper right", fontsize=7)
    fig.colorbar(im, ax=ax, label="nominal N")
    fig.tight_layout()
    fig.savefig(OUT / "force_landscapes" / f"state_{state_id:04d}.png", dpi=150)
    plt.close(fig)


def main():
    replay = json.loads((OUT / "replay_audit.json").read_text())
    search = json.loads((OUT / "search_parent_audit.json").read_text())
    if replay["status"] != "PASS" or search["status"] != "PASS" or search["parent_steps_compared"] != 1848:
        raise RuntimeError("G8.5 replay/search parent integrity not passed")
    manifest = json.loads((OUT / "recovered_g8_state_manifest.json").read_text())["states"]
    if len(manifest) != 16:
        raise RuntimeError("G8.5 archived state count changed")
    global_all, local_all = [], []
    withdrawal, oracle, comparison, geometry = [], [], [], []
    for state in manifest:
        sid = state["state_id"]
        global_rows = [cast(r) for r in read_csv(OUT / "per_state" / f"state_{sid:04d}_global.csv")]
        local_rows = [cast(r) for r in read_csv(OUT / "per_state" / f"state_{sid:04d}_local.csv")]
        if len(global_rows) != 1681 or not local_rows or len({xy(r) for r in global_rows}) != 1681:
            raise RuntimeError(f"Incomplete G8.5 state {sid}")
        for row in global_rows + local_rows:
            if row["state_id"] != sid or row["episode_seed"] != state["episode_seed"] or row["decision_index"] != state["decision_index"]:
                raise RuntimeError(f"G8.5 state identity mismatch {sid}")
            if row["true_safe"] != int(row["F_true_N"] <= F_LIM) or row["F_true_N"] < 0:
                raise RuntimeError(f"G8.5 force/safe classification mismatch {sid}")
        if len({xy(r) for r in global_rows + local_rows}) != len(global_rows) + len(local_rows):
            raise RuntimeError(f"Duplicate G8.5 realized action {sid}")
        global_all.extend(global_rows)
        local_all.extend(local_rows)
        archived = json.loads((G8_STATES / f"state_{sid:04d}.json").read_text())
        old_rows = []
        for candidate in archived["candidates"]:
            old_rows.append({"state_id":sid,"episode_seed":state["episode_seed"],
                "decision_index":state["decision_index"],
                "translation":float(candidate["action"][0]),"rotation":float(candidate["action"][1]),
                "F_true_N":float(candidate["force_next_max_N"]),
                "true_safe":int(float(candidate["force_next_max_N"]) <= F_LIM),
                "action_source":"old21_reused_label"})
        if len(old_rows) != 21 or sum(r["true_safe"] for r in old_rows) != state["old21_true_safe_count"]:
            raise RuntimeError(f"G8 old 21 labels changed {sid}")
        both = [{**r,"action_source":"global_grid"} for r in global_rows]
        both += [{**r,"action_source":"local_refinement"} for r in local_rows]
        existing = {xy(r):r for r in both}
        for old in old_rows:
            prior = existing.get(xy(old))
            if prior is not None:
                if abs(prior["F_true_N"]-old["F_true_N"]) > 1e-8:
                    raise RuntimeError(f"G8/G8.5 duplicate-action force mismatch {sid}")
            else:
                both.append(old)
                existing[xy(old)] = old
        nominal = state["nominal_action"]
        nearest = nearest_safe(both, nominal)
        minimum = min_force(both)
        minimum_grid = min_force(global_rows)
        negative = [r for r in both if r["translation"] < 0]
        nonpositive = [r for r in both if r["translation"] <= 0]
        best_negative = min_force(negative)
        best_nonpositive = min_force(nonpositive)
        strong = next((r for r in global_rows if r["translation"] == -1.0 and r["rotation"] == 0.0), None)
        if strong is None:
            raise RuntimeError("Strong withdrawal grid action missing")
        w = {"state_id":sid, "episode_seed":state["episode_seed"],
            "withdrawal_action_count":len(negative), "withdrawal_safe_count":sum(r["true_safe"] for r in negative),
            "any_withdrawal_safe":int(any(r["true_safe"] for r in negative)),
            "minimum_withdrawal_force_N":best_negative["F_true_N"],
            "best_withdrawal_translation":best_negative["translation"],
            "best_withdrawal_rotation":best_negative["rotation"],
            "best_nonpositive_translation":best_nonpositive["translation"],
            "best_nonpositive_rotation":best_nonpositive["rotation"],
            "best_nonpositive_force_N":best_nonpositive["F_true_N"],
            "strong_withdrawal_translation":-1.0, "strong_withdrawal_rotation":0.0,
            "strong_withdrawal_force_N":strong["F_true_N"],
            "strong_withdrawal_safe":strong["true_safe"]}
        withdrawal.append(w)
        safe_grid = [r for r in global_rows if r["true_safe"]]
        safe_local = [r for r in local_rows if r["true_safe"]]
        deviation = float(np.linalg.norm(np.asarray(xy(nearest))-nominal)) if nearest is not None else None
        oracle.append({"state_id":sid,"episode_seed":state["episode_seed"],
            "decision_index":state["decision_index"], "nominal_translation":nominal[0],
            "nominal_rotation":nominal[1], "nominal_F_true_N":state["nominal_F_true_N"],
            "global_grid_min_F_true_N":minimum_grid["F_true_N"],
            "minimum_searched_F_true_N":minimum["F_true_N"],
            "minimum_force_translation":minimum["translation"],
            "minimum_force_rotation":minimum["rotation"],
            "global_grid_safe_count":len(safe_grid), "local_safe_count":len(safe_local),
            "continuous_oracle_feasible":int(nearest is not None),
            "nearest_safe_translation":nearest["translation"] if nearest else None,
            "nearest_safe_rotation":nearest["rotation"] if nearest else None,
            "nearest_safe_F_true_N":nearest["F_true_N"] if nearest else None,
            "nearest_safe_action_source":nearest["action_source"] if nearest else None,
            "nearest_safe_action_distance_l2":deviation,
            "delta_translation":nearest["translation"]-nominal[0] if nearest else None,
            "delta_rotation":nearest["rotation"]-nominal[1] if nearest else None,
            "force_change_N":nearest["F_true_N"]-state["nominal_F_true_N"] if nearest else None,
            "global_evaluations":len(global_rows), "local_evaluations":len(local_rows),
            "old21_reused_labels":len(old_rows)})
        old = not state["old21_infeasible"]
        new = nearest is not None
        category = "ALREADY_REPAIRABLE" if old else ("OLD_SET_FALSE_INFEASIBLE" if new else "NO_SAFE_FOUND_IN_SEARCH")
        comparison.append({"state_id":sid,"episode_seed":state["episode_seed"],
            "old21_true_safe_count":state["old21_true_safe_count"],
            "old21_feasible":int(old),"searched_continuous_feasible":int(new),
            "category":category,"nearest_safe_action_distance_l2":deviation,
            "minimum_searched_F_true_N":minimum["F_true_N"]})
        matrix = np.asarray([r["F_true_N"] for r in global_rows],dtype=float).reshape(41,41)
        mask = matrix <= F_LIM
        components = connected_components(mask)
        safe_coordinates = np.argwhere(mask)
        rot_range = np.max(matrix,axis=1)-np.min(matrix,axis=1)
        rot_benefit = matrix[:,20]-np.min(matrix,axis=1)
        geometry.append({"state_id":sid,"grid_safe_count":int(np.sum(mask)),
            "grid_safe_fraction":float(np.mean(mask)),
            "safe_components_4_connected":len(components),
            "largest_safe_component_grid_points":components[0] if components else 0,
            "safe_translation_min":float(-1+safe_coordinates[:,0].min()*0.05) if len(safe_coordinates) else None,
            "safe_translation_max":float(-1+safe_coordinates[:,0].max()*0.05) if len(safe_coordinates) else None,
            "safe_rotation_min":float(-1+safe_coordinates[:,1].min()*0.05) if len(safe_coordinates) else None,
            "safe_rotation_max":float(-1+safe_coordinates[:,1].max()*0.05) if len(safe_coordinates) else None,
            "safe_grid_negative_translation_count":sum(r["true_safe"] for r in global_rows if r["translation"]<0),
            "median_force_range_over_rotation_N":float(np.median(rot_range)),
            "max_force_range_over_rotation_N":float(np.max(rot_range)),
            "median_best_rotation_benefit_vs_zero_N":float(np.median(rot_benefit)),
            "max_best_rotation_benefit_vs_zero_N":float(np.max(rot_benefit))})
        plot_landscape(sid, matrix, nominal, nearest)
        print(f"G8.5 analyzed state={sid} grid_safe={len(safe_grid)} local_safe={len(safe_local)} minimum={minimum['F_true_N']:.6f}",flush=True)
    csv_new(OUT / "global_grid_results.csv",global_all,FIELDS)
    csv_new(OUT / "local_refinement_results.csv",local_all,FIELDS)
    csv_new(OUT / "withdrawal_analysis.csv",withdrawal)
    csv_new(OUT / "continuous_oracle_repair.csv",oracle)
    csv_new(OUT / "old21_vs_continuous_comparison.csv",comparison)
    csv_new(OUT / "force_landscape_geometry.csv",geometry)
    distances=[r["nearest_safe_action_distance_l2"] for r in oracle if r["continuous_oracle_feasible"]]
    old_false=[r for r in comparison if r["category"]=="OLD_SET_FALSE_INFEASIBLE"]
    no_safe=[r for r in comparison if r["category"]=="NO_SAFE_FOUND_IN_SEARCH"]
    summary={"status":"COMPLETE_TRUE_SOFA_FINITE_CONTINUOUS_ACTION_SEARCH",
        "nominal_unsafe_state_count":16,"state_ids":[r["state_id"] for r in manifest],
        "old21_repairable_count":4,"old21_infeasible_count":12,
        "searched_continuous_feasible_count":sum(r["continuous_oracle_feasible"] for r in oracle),
        "no_safe_found_count":len(no_safe),"old21_false_infeasible_count":len(old_false),
        "old21_false_infeasible_state_ids":[r["state_id"] for r in old_false],
        "old21_still_no_safe_found_count":len(no_safe),
        "old21_still_no_safe_found_state_ids":[r["state_id"] for r in no_safe],
        "global_grid_evaluations":len(global_all),"local_refinement_evaluations":len(local_all),
        "total_search_oracle_evaluations":len(global_all)+len(local_all),
        "old21_reused_oracle_labels":16*21,
        "replay_nominal_audit_forks":replay["unsafe_nominal_fork_count"],
        "search_nominal_integrity_forks":search["unsafe_nominal_fork_count"],
        "withdrawal_safe_state_count":sum(r["any_withdrawal_safe"] for r in withdrawal),
        "strong_withdrawal_safe_state_count":sum(r["strong_withdrawal_safe"] for r in withdrawal),
        "nearest_safe_distance_median":float(np.median(distances)) if distances else None,
        "nearest_safe_distance_max":float(np.max(distances)) if distances else None,
        "finite_search_caveat":"No-safe-found on a 41x41 grid plus deterministic local refinement is not a proof that no continuous safe action exists."}
    json_new(OUT / "recoverability_summary.json",summary)
    print("G8.5 summary",summary,flush=True)


if __name__ == "__main__":
    main()
