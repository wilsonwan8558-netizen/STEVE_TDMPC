"""Predeclare true-high and control G12-held-out SOFA parent states for G13 forks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
OUT = RL / "logs/world_model_force_aware_planning_g13"
G12 = RL / "logs/world_model_force_aware_planning_g12"
F_LIM = 3.685331040756903


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    path = OUT / "candidate_state_manifest.json"
    if path.exists():
        raise FileExistsError(path)
    with np.load(G12 / "force_transitions.npz", allow_pickle=False) as data:
        seeds = data["seed"]
        decisions = data["decision_index"]
        forces = data["force_next_max_N"]
    split = json.loads((G12 / "force_dataset_split.json").read_text())
    heldout = set(int(seed) for seed in split["heldout_test_seeds"])
    by_seed = {}
    for index, (seed, force) in enumerate(zip(seeds, forces)):
        if int(seed) in heldout:
            by_seed.setdefault(int(seed), []).append(index)
    high = []
    for seed, indices in by_seed.items():
        high_indices = [index for index in indices if forces[index] > F_LIM]
        if high_indices:
            best = max(high_indices, key=lambda index: (float(forces[index]), -int(decisions[index])))
            high.append(best)
    if len(high) < 20:
        raise RuntimeError(f"Not enough distinct held-out high-force seeds: {len(high)}")
    # True-force-based enrichment is declared before any G13 predictions and
    # is not used as a population prevalence estimate.
    high = sorted(high, key=lambda index: (-float(forces[index]), int(seeds[index])))[:20]
    states = []
    for index in high:
        seed = int(seeds[index])
        decision = int(decisions[index])
        states.append({"seed": seed, "decision_index": decision,
                       "stratum": "archived_true_high_anchor",
                       "archived_behavior_force_N": float(forces[index])})
        controls = [i for i in by_seed[seed] if forces[i] <= 0.5 * F_LIM
                    and abs(int(decisions[i]) - decision) >= 10]
        if not controls:
            controls = [i for i in by_seed[seed] if forces[i] <= 0.5 * F_LIM]
        if not controls:
            raise RuntimeError(f"No safe control for held-out seed {seed}")
        control = min(controls, key=lambda i: (abs(int(decisions[i]) - decision), int(decisions[i])))
        states.append({"seed": seed, "decision_index": int(decisions[control]),
                       "stratum": "matched_low_force_control",
                       "archived_behavior_force_N": float(forces[control])})
    states.sort(key=lambda state: (state["seed"], state["decision_index"]))
    if len(states) != 40 or len({(row["seed"], row["decision_index"]) for row in states}) != 40:
        raise RuntimeError("Candidate state manifest has duplicate or missing parents")
    manifest = {
        "status": "FROZEN_BEFORE_G13_PREDICTIONS_OR_TRUE_SOFA_CANDIDATE_FORKS",
        "F_lim_N": F_LIM,
        "source": str((G12 / "force_transitions.npz").resolve()),
        "source_sha256": sha256(G12 / "force_transitions.npz"),
        "split_sha256": sha256(G12 / "force_dataset_split.json"),
        "selection": "one maximal above-limit behavior transition from each of the 20 highest-peak distinct held-out seeds, plus one matched low-force control from each seed",
        "analysis_caveat": "true-force-enriched parent selection; candidate-level results are conditional, not population prevalence",
        "states": states,
        "state_count": len(states),
        "normal_planner_candidate_count_per_state": 512,
        "planned_true_sofa_forks_per_state": 32,
        "candidate_fork_sampling": "predeclared deterministic rank-spread of final planner candidates, including elites and nonelites; archived behavior action separately as positive/high or control anchor",
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("G13_CANDIDATE_PARENTS_FROZEN", len(states), flush=True)


if __name__ == "__main__":
    main()
