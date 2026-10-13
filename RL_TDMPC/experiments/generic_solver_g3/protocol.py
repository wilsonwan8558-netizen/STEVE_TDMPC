"""Predeclared G3 state and 21-action candidate protocol; no outcome labels."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from prepare import G2, OUT

CANDIDATES = 21
CANDIDATE_SEED_BASE = 260106
GROUP_OFFSET = {"train": 200000, "validation": 300000, "development_test": 100000, "confirmation": 400000}


def choose_states(seeds: list[int], rows_by_seed: dict[int, list[dict]], group: str) -> list[dict]:
    """Old 2D study's deterministic four-per-episode progress/prior-force strata."""
    selected = []
    for episode_index, seed in enumerate(sorted(seeds)):
        rows = rows_by_seed[seed]
        length = len(rows)
        if length < 12:
            raise RuntimeError(f"Episode {seed} too short")
        valid = list(range(1, length - 2))
        prior = {d: float(rows[d-1]["distal_contact_force_max_N"]) for d in valid}
        thresholds = np.quantile([prior[d] for d in valid], (.2, .5, .8, .95))
        chosen: set[int] = set()
        for slot, (low, high) in enumerate(((0, 1/3), (1/3, 2/3), (2/3, 1))):
            available = [d for d in valid if low <= d/length < high and d not in chosen]
            if not available:
                raise RuntimeError(f"Missing progress stratum {seed}/{slot}")
            target = float(thresholds[(slot + episode_index) % 3])
            midpoint = (low + high)/2
            d = min(available, key=lambda value: (abs(prior[value]-target), abs(value/length-midpoint), value))
            chosen.add(d)
            selected.append({"group": group, "state_id": len(selected), "seed": seed, "decision_index": d,
                             "slot": ("early", "middle", "late")[slot], "episode_progress": d/length,
                             "previous_executed_force_N": prior[d], "episode_length": length})
        d = min((value for value in valid if value not in chosen),
                key=lambda value: (abs(prior[value]-float(thresholds[3])), abs(value/length-.5), value))
        selected.append({"group": group, "state_id": len(selected), "seed": seed, "decision_index": d,
                         "slot": "prior_force_Q95", "episode_progress": d/length,
                         "previous_executed_force_N": prior[d], "episode_length": length})
    if len(selected) != 4*len(seeds) or len({(s["seed"], s["decision_index"]) for s in selected}) != len(selected):
        raise RuntimeError("State selection incomplete or duplicated")
    return selected


def generate_candidates(behavior_action, state_id: int, group: str) -> list[dict]:
    """Exact old 21-action local-axis/diagonal/broad design, new disjoint RNG streams."""
    behavior = np.asarray(behavior_action, dtype=np.float32).reshape(2)
    if np.any(~np.isfinite(behavior)) or np.any(np.abs(behavior) > 1+1e-6):
        raise RuntimeError("Infeasible behavior action")
    candidates = []
    seen = set()

    def add(action, kind):
        action = np.clip(np.asarray(action, dtype=np.float32), -1, 1)
        key = tuple(float(value) for value in action)
        if key in seen:
            return
        seen.add(key)
        candidates.append({"candidate_id": len(candidates), "kind": kind, "action": action.tolist(),
                           "action_delta": (action-behavior).tolist(),
                           "action_distance_l2": float(np.linalg.norm(action-behavior))})

    add(behavior, "behavior")
    for component in (0, 1):
        for offset in (-.4, -.2, .2, .4):
            delta = np.zeros(2, dtype=np.float32)
            delta[component] = offset
            add(behavior+delta, "local_axis")
    for magnitude in (.2, .4):
        for translation_sign in (-1, 1):
            for rotation_sign in (-1, 1):
                add(behavior+np.asarray([translation_sign*magnitude, rotation_sign*magnitude], dtype=np.float32), "local_diagonal")
    rng = np.random.default_rng(CANDIDATE_SEED_BASE + GROUP_OFFSET[group] + state_id)
    attempts = 0
    while len(candidates) < CANDIDATES:
        add(rng.uniform(-1, 1, size=2), "broad_uniform")
        attempts += 1
        if attempts > 200:
            raise RuntimeError("Cannot generate 21 unique candidates")
    return candidates


def freeze_development() -> None:
    split = json.loads((OUT / "development_split.json").read_text())
    with (G2 / "step_level_force.csv").open(newline="") as stream:
        rows_by_seed: dict[int, list[dict]] = {}
        for row in csv.DictReader(stream):
            rows_by_seed.setdefault(int(row["seed"]), []).append(row)
    manifest = {group: choose_states(split[group], rows_by_seed, group)
                for group in ("train", "validation", "development_test")}
    for group, states in manifest.items():
        for state in states:
            row = rows_by_seed[state["seed"]][state["decision_index"]]
            actions = generate_candidates(json.loads(row["selected_action_json"]), state["state_id"], group)
            if len(actions) != 21 or actions[0]["kind"] != "behavior":
                raise RuntimeError("Candidate protocol validation failed")
    path = OUT / "development_state_selection.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps({"selection_rule": "three progress strata matched to prior executed force quantiles plus one prior-force Q95 state; no future counterfactual label", "states": manifest}, indent=2, sort_keys=True)+"\n")
    path = OUT / "candidate_action_protocol.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps({"candidate_count": 21, "behavior_candidate_id": 0, "action_bounds": [-1, 1],
                                "local_axis_offsets": [-.4, -.2, .2, .4], "local_diagonal_magnitudes": [.2, .4],
                                "broad_fill": "NumPy default_rng(260106 + group_offset + state_id).uniform(-1,1,size=2) until 21 unique float32 actions",
                                "group_offsets": GROUP_OFFSET, "clipping": "componentwise [-1,1]", "duplicate_handling": "exact float32 duplicate removal",
                                "source_design": "LCP Phase 2C 21-action protocol; compatible with Generic normalized 2D action space",
                                "frozen_before_counterfactual_force_outcomes": True}, indent=2, sort_keys=True)+"\n")
    print("G3 development states and candidate protocol frozen", flush=True)


if __name__ == "__main__":
    freeze_development()
