"""Exact-action nearest-state force disagreement, independent of predictors."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generic_solver_g3"))
from g4_data import OUT, REPRESENTATIONS, GROUPS, source, physical_lookup, candidate_states, state_vector
from g3_evaluate import quantiles


def exact_action_key(action):
    return tuple(float(v) for v in np.asarray(action, dtype=np.float32))


def candidate_records(representation, states, vector_for):
    records = []
    for state in states:
        vector = vector_for(representation, state)
        for candidate in state["candidates"]:
            records.append({"group": state["group"], "seed": state["seed"], "decision_index": state["decision_index"],
                            "state_id": state["state_id"], "candidate_id": candidate["candidate_id"],
                            "action": exact_action_key(candidate["action"]), "vector": vector,
                            "force_next_max_N": float(candidate["force_next_max_N"])})
    return records


def nearest(records, mean, scale, q95, reference=None, exclude_same_state=True):
    """Match exact float32 2-D actions, never candidate IDs or rounded actions."""
    reference = records if reference is None else reference
    buckets = defaultdict(list)
    for i, item in enumerate(reference):
        buckets[item["action"]].append(i)
    trees = {}
    for action, indices in buckets.items():
        vectors = np.asarray([(reference[i]["vector"]-mean)/scale for i in indices])
        trees[action] = (indices, cKDTree(vectors))
    output = []
    for item in records:
        match = None
        bucket = trees.get(item["action"])
        if bucket is not None:
            indices, tree = bucket
            query = (item["vector"]-mean)/scale
            # Query all same-action peers in distance order when necessary to
            # exclude the identical state/candidate; coverage is explicit.
            distances, positions = tree.query(query, k=len(indices))
            for distance, position in zip(np.atleast_1d(distances), np.atleast_1d(positions)):
                peer = reference[indices[int(position)]]
                if exclude_same_state and (peer["seed"], peer["decision_index"]) == (item["seed"], item["decision_index"]):
                    continue
                match = (float(distance), peer)
                break
        record = {"representation": None, "group": item["group"], "seed": item["seed"],
                  "decision_index": item["decision_index"], "state_id": item["state_id"],
                  "candidate_id": item["candidate_id"], "action_translation": item["action"][0],
                  "action_rotation": item["action"][1], "force_next_max_N": item["force_next_max_N"],
                  "same_action_peer_found": match is not None, "neighbor_group": None,
                  "neighbor_seed": None, "neighbor_decision_index": None, "neighbor_state_id": None,
                  "neighbor_candidate_id": None, "neighbor_force_next_max_N": None,
                  "neighbor_distance": None, "absolute_force_difference_N": None,
                  "Q95_cross_boundary": None}
        if match is not None:
            distance, peer = match
            record.update({"neighbor_group": peer["group"], "neighbor_seed": peer["seed"],
                           "neighbor_decision_index": peer["decision_index"], "neighbor_state_id": peer["state_id"],
                           "neighbor_candidate_id": peer["candidate_id"], "neighbor_force_next_max_N": peer["force_next_max_N"],
                           "neighbor_distance": distance,
                           "absolute_force_difference_N": abs(item["force_next_max_N"]-peer["force_next_max_N"]),
                           "Q95_cross_boundary": (item["force_next_max_N"] > q95) != (peer["force_next_max_N"] > q95)})
        output.append(record)
    return output


def summary(records):
    matched = [r for r in records if r["same_action_peer_found"]]
    if not matched:
        return {"query_count": len(records), "matched_query_count": 0, "coverage": 0.,
                "warning": "No exact-action peer; observability cannot be assessed by this diagnostic"}
    distance = np.asarray([r["neighbor_distance"] for r in matched], dtype=float)
    difference = np.asarray([r["absolute_force_difference_N"] for r in matched], dtype=float)
    cutoffs = np.quantile(distance, (.1, .25, .5))
    bins = []
    for label, cutoff in zip(("nearest_10pct", "nearest_25pct", "nearest_50pct"), cutoffs):
        mask = distance <= cutoff
        bins.append({"bin": label, "maximum_normalized_state_distance": float(cutoff),
                     "n": int(np.sum(mask)), "median_absolute_force_difference_N": float(np.median(difference[mask])),
                     "fraction_difference_gt_1_N": float(np.mean(difference[mask] > 1.)),
                     "Q95_cross_boundary_fraction": float(np.mean(np.asarray([r["Q95_cross_boundary"] for r in matched])[mask]))})
    return {"query_count": len(records), "matched_query_count": len(matched), "coverage": len(matched)/len(records),
            "unique_matched_exact_actions": sorted({(r["action_translation"], r["action_rotation"]) for r in matched}),
            "unique_matched_exact_action_count": len({(r["action_translation"], r["action_rotation"]) for r in matched}),
            "median_absolute_force_difference_N": float(np.median(difference)),
            "Q90_absolute_force_difference_N": float(np.quantile(difference, .9)),
            "Q95_absolute_force_difference_N": float(np.quantile(difference, .95)),
            "fraction_difference_gt_1_N": float(np.mean(difference > 1.)),
            "Q95_cross_boundary_fraction": float(np.mean([r["Q95_cross_boundary"] for r in matched])),
            "median_neighbor_distance": float(np.median(distance)),
            "small_distance_bins": bins,
            "warning": "Exact float32 candidate actions are continuous and mostly unique; conclusions are conditional on matched-query coverage"}


def write_csv(path, rows):
    if path.exists():
        raise FileExistsError(path)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    arrays, split, index = source()
    physical = physical_lookup()
    states = [s for group in GROUPS for s in candidate_states(group, split)]
    if len(states) != 800:
        raise RuntimeError("Development aliasing requires 800 states")
    q95 = quantiles()["q95"]
    rows, summaries = [], {}
    for representation in REPRESENTATIONS:
        norm = json.loads((OUT / "models" / representation / "normalization.json").read_text())
        mean, scale = np.asarray(norm["input_mean"][:-2]), np.asarray(norm["input_std"][:-2])
        def vector_for(r, state):
            return state_vector(r, state["seed"], state["decision_index"], arrays, index, physical)
        records = candidate_records(representation, states, vector_for)
        result = nearest(records, mean, scale, q95, exclude_same_state=True)
        for record in result:
            record["representation"] = representation
        rows.extend(result)
        summaries[representation] = summary(result)
    write_csv(OUT / "nearest_neighbor_aliasing.csv", rows)
    path = OUT / "aliasing_summary.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps({"method": "leave-one-state-out nearest exact-action neighbor among 800 development states / 16800 candidate rows; representation TRAIN-behavior normalization; includes train, validation, and development-test states",
                                "normalization_excludes_action_columns": True, "force_reference_Q95_N": q95,
                                "representations": summaries}, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print("G4 development exact-action aliasing complete", flush=True)


if __name__ == "__main__":
    main()
