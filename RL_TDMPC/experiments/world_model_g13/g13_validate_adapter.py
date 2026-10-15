"""Verify G13 observation/force wrapper is passive versus archived Generic G2."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
sys.path.insert(0, str(RL))
sys.path.insert(0, str(RL.parent))
from tdmpc2.common import set_seed  # noqa: E402
from g13_env import make_g13_env  # noqa: E402

OUT = RL / "logs/world_model_force_aware_planning_g13"
G2 = RL / "logs/generic_solver_phase_g2_force_baseline/step_level_force.csv"
SEEDS = tuple(range(90000, 90010))


def main() -> None:
    rows_by_seed = {seed: [] for seed in SEEDS}
    with G2.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            seed = int(row["seed"])
            if seed in rows_by_seed:
                rows_by_seed[seed].append(row)
    set_seed(SEEDS[0])
    env = make_g13_env()
    results = []
    try:
        for seed in SEEDS:
            set_seed(seed)
            env.action_space.seed(seed)
            observation, _ = env.reset(seed=seed)
            max_force_diff = 0.0
            steps = rows_by_seed[seed]
            for decision, row in enumerate(steps):
                expected = np.asarray(json.loads(row["decision_observation_json"]), dtype=np.float32)
                if observation.shape != (22,) or not np.array_equal(observation[:14], expected):
                    raise RuntimeError(f"G13 adapter changed pre-action observation {seed}/{decision}")
                action = np.asarray(json.loads(row["selected_action_json"]), dtype=np.float32)
                observation, reward, terminated, truncated, info = env.step(action)
                next_expected = np.asarray(json.loads(row["next_observation_json"]), dtype=np.float32)
                if (not np.array_equal(observation[:14], next_expected) or
                        abs(float(reward)-float(row["reward"])) > 1e-9 or
                        bool(terminated) != (row["terminated"] == "True") or
                        bool(truncated) != (row["truncated"] == "True") or
                        not np.array_equal(np.asarray(info["raw_action"], dtype=np.float32),
                                           np.asarray(json.loads(row["applied_action_json"]), dtype=np.float32))):
                    raise RuntimeError(f"G13 adapter changed task transition {seed}/{decision}")
                force_diff = abs(info["distal_contact_force_max_N"] - float(row["distal_contact_force_max_N"]))
                max_force_diff = max(max_force_diff, force_diff)
                if force_diff > 1e-8 or info["sofa_substeps"] != int(row["sofa_substeps"]):
                    raise RuntimeError(f"G13 adapter changed frozen force target {seed}/{decision}: {force_diff}")
            results.append({"seed": seed, "steps": len(steps), "max_force_abs_diff_N": max_force_diff,
                            "original_task_observation_exact": True,
                            "task_transition_exact": True,
                            "force_target_within_1e_8_N": True})
            print("G13_ADAPTER_EQUIVALENCE", seed, len(steps), max_force_diff, flush=True)
    finally:
        env.close()
    with (OUT / "g13_adapter_equivalence.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print("G13_ADAPTER_EQUIVALENCE_PASS", sum(r["steps"] for r in results), flush=True)


if __name__ == "__main__":
    main()
