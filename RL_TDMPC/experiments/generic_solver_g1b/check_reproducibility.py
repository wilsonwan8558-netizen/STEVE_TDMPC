"""Compare two independent reused-scene G1B evaluation replays seed by seed."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    first = read(args.first)
    second = read(args.second)
    if len(first) != 5 or len(second) != 5:
        raise RuntimeError("G1B deterministic gate requires two complete five-seed replays")
    if [int(row["seed"]) for row in first] != list(range(91000, 91005)):
        raise RuntimeError("First deterministic replay used wrong seeds/order")
    if [int(row["seed"]) for row in second] != list(range(91000, 91005)):
        raise RuntimeError("Second deterministic replay used wrong seeds/order")
    comparisons = []
    for a, b in zip(first, second):
        if a["checkpoint_step"] != b["checkpoint_step"] or a["checkpoint_sha256"] != b["checkpoint_sha256"]:
            raise RuntimeError("Different checkpoint used for paired reproducibility test")
        comparisons.append({
            "seed": int(a["seed"]),
            "success_match": a["success"] == b["success"],
            "length_match": a["episode_length"] == b["episode_length"],
            "reward_absolute_difference": abs(float(a["episode_reward"])-float(b["episode_reward"])),
            "trajectory_hash_match": a["trajectory_sha256"] == b["trajectory_sha256"],
        })
    pass_gate = all(row["success_match"] and row["length_match"] and
                    row["reward_absolute_difference"] <= 1e-9 and
                    row["trajectory_hash_match"] for row in comparisons)
    result = {
        "status": "PASS" if pass_gate else "FAIL",
        "checkpoint_step": int(first[0]["checkpoint_step"]),
        "seeds": list(range(91000, 91005)),
        "max_reward_absolute_difference": max(row["reward_absolute_difference"] for row in comparisons),
        "success_mismatches": sum(not row["success_match"] for row in comparisons),
        "length_mismatches": sum(not row["length_match"] for row in comparisons),
        "trajectory_hash_mismatches": sum(not row["trajectory_hash_match"] for row in comparisons),
        "per_seed": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if not pass_gate:
        raise RuntimeError("G1B deterministic evaluator gate failed")


if __name__ == "__main__":
    main()
