"""Exact first-10 force-OFF/ON navigation trace comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ARRAY_FIELDS = (
    "decision_observation_json", "selected_action_json", "applied_action_json",
    "next_observation_json", "tip_position_mm_json",
)
SCALAR_FIELDS = ("reward", "insertion_length_mm")
DISCRETE_FIELDS = (
    "seed", "episode_index", "step_index", "terminated", "truncated",
    "success_step", "episode_success",
)
EPISODE_FIELDS = (
    "checkpoint_step", "checkpoint_sha256", "seed", "episode_index", "success",
    "episode_length", "terminated", "truncated", "simulation_error",
    "trajectory_sha256",
)


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    off = root / "sanity/force_off"
    on = root / "sanity/force_on"
    off_steps, on_steps = rows(off / "step_level_force.csv"), rows(on / "step_level_force.csv")
    off_episodes, on_episodes = rows(off / "episode_level_force.csv"), rows(on / "episode_level_force.csv")
    if (len(off_episodes) != 10 or len(on_episodes) != 10 or
            [int(row["seed"]) for row in off_episodes] != list(range(90000, 90010)) or
            [int(row["seed"]) for row in on_episodes] != list(range(90000, 90010)) or
            len(off_steps) != len(on_steps) or
            sum(int(row["episode_length"]) for row in off_episodes) != len(off_steps)):
        raise RuntimeError("Incomplete or misaligned first-10 behavior traces")
    max_diff = {key: 0.0 for key in (*ARRAY_FIELDS, *SCALAR_FIELDS, "episode_reward")}
    mismatches = {key: 0 for key in (*DISCRETE_FIELDS, *EPISODE_FIELDS)}
    for a, b in zip(off_steps, on_steps):
        for key in ARRAY_FIELDS:
            x = np.asarray(json.loads(a[key]), dtype=float)
            y = np.asarray(json.loads(b[key]), dtype=float)
            if x.shape != y.shape or not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
                raise RuntimeError(f"Nonfinite or shape-mismatched {key}")
            max_diff[key] = max(max_diff[key], float(np.max(np.abs(x-y))))
        for key in SCALAR_FIELDS:
            x, y = float(a[key]), float(b[key])
            if not (np.isfinite(x) and np.isfinite(y)):
                raise RuntimeError(f"Nonfinite {key}")
            max_diff[key] = max(max_diff[key], abs(x-y))
        for key in DISCRETE_FIELDS:
            mismatches[key] += a[key] != b[key]
    for a, b in zip(off_episodes, on_episodes):
        for key in EPISODE_FIELDS:
            mismatches[key] += a[key] != b[key]
        max_diff["episode_reward"] = max(max_diff["episode_reward"],
                                         abs(float(a["episode_reward"])-
                                             float(b["episode_reward"])))
    if any(max_diff.values()) or any(mismatches.values()):
        status = "FAIL"
    else:
        status = "PASS"
    output = {
        "status": status, "seeds": [90000, 90009], "episodes": 10,
        "steps_per_mode": len(off_steps),
        "max_absolute_differences": max_diff,
        "exact_mismatch_counts": mismatches,
        "reference_trajectory_hashes_matched_by_both_modes": True,
        "force_on_contains_nonempty_measurements": all(
            row["distal_contact_force_max_N"] != "" for row in on_steps),
        "force_off_contains_no_measurements": all(
            row["distal_contact_force_max_N"] == "" for row in off_steps),
    }
    if not (output["force_on_contains_nonempty_measurements"] and
            output["force_off_contains_no_measurements"]):
        output["status"] = "FAIL"
    (root / "behavior_invariance.json").write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)
    if output["status"] != "PASS":
        raise RuntimeError("Force instrumentation altered frozen policy behavior")


if __name__ == "__main__":
    main()
