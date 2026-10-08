"""Characterize the one frozen Generic baseline on the untouched 200-seed set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    selected_path = root / "selected_checkpoint.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    if selected.get("status") != "FROZEN_BEFORE_CONFIRMATION":
        raise RuntimeError("Checkpoint selection was not frozen before confirmation")
    with (root / "confirmation_200_episode_results.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 200 or [int(row["seed"]) for row in rows] != list(range(90000, 90200)):
        raise RuntimeError("Confirmation set missing seeds, duplicated, or out of order")
    if any(int(row["checkpoint_step"]) != selected["step"] or
           row["checkpoint_sha256"] != selected["sha256"] for row in rows):
        raise RuntimeError("Confirmation did not use the frozen selected checkpoint")
    if any(row["simulation_error"] != "False" for row in rows):
        raise RuntimeError("Confirmation encountered a simulator error")
    success = np.asarray([row["success"] == "True" for row in rows], dtype=int)
    length = np.asarray([int(row["episode_length"]) for row in rows], dtype=float)
    reward = np.asarray([float(row["episode_reward"]) for row in rows], dtype=float)
    if not np.all(np.isfinite(reward)):
        raise RuntimeError("Nonfinite confirmation reward")
    summary = {
        "status": "COMPLETE",
        "checkpoint_step": selected["step"],
        "checkpoint_path": selected["path"],
        "checkpoint_sha256": selected["sha256"],
        "selected_checkpoint_manifest_sha256": sha256(selected_path),
        "seeds": [90000, 90199], "episodes": 200,
        "success_count": int(np.sum(success)),
        "success_rate": float(np.mean(success)),
        "mean_episode_length": float(np.mean(length)),
        "median_episode_length": float(np.median(length)),
        "mean_episode_reward": float(np.mean(reward)),
        "median_episode_reward": float(np.median(reward)),
        "mean_success_episode_length": float(np.mean(length[success == 1])) if np.any(success == 1) else None,
        "mean_failure_episode_length": float(np.mean(length[success == 0])) if np.any(success == 0) else None,
        "selection_set_success_rate": selected["selection_metrics"]["success_rate"],
        "confirmation_minus_selection_success_rate_pp": float(100*(np.mean(success)-selected["selection_metrics"]["success_rate"])),
        "selection_set_mean_length": selected["selection_metrics"]["mean_episode_length"],
        "confirmation_minus_selection_mean_length_steps": float(np.mean(length)-selected["selection_metrics"]["mean_episode_length"]),
        "selection_set_mean_reward": selected["selection_metrics"]["mean_episode_reward"],
        "confirmation_minus_selection_mean_reward": float(np.mean(reward)-selected["selection_metrics"]["mean_episode_reward"]),
    }
    (root / "confirmation_200_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    historical = json.loads((root / "historical_lcp_reference_verified.json").read_text())
    if historical.get("status") != "VERIFIED_FROM_ARCHIVED_PER_EPISODE_REPORT":
        raise RuntimeError("Historical LCP reference not verified")
    comparison = {
        "status": "DESCRIPTIVE_ONLY",
        "generic_solver": "GenericConstraintSolver",
        "generic_checkpoint_step": selected["step"],
        "generic_checkpoint_sha256": selected["sha256"],
        "generic_success_count": summary["success_count"],
        "generic_success_rate": summary["success_rate"],
        "generic_mean_episode_length": summary["mean_episode_length"],
        "generic_median_episode_length": summary["median_episode_length"],
        "generic_mean_episode_reward": summary["mean_episode_reward"],
        "generic_median_episode_reward": summary["median_episode_reward"],
        "historical_lcp_reference_source": historical["source"],
        "historical_lcp_reference_sha256": historical["source_sha256"],
        "historical_lcp_success_count": historical["success_count"],
        "historical_lcp_success_rate": historical["success_rate"],
        "historical_lcp_mean_episode_length": historical["mean_episode_length"],
        "historical_lcp_median_episode_length": historical["median_episode_length"],
        "historical_lcp_mean_episode_reward": historical["mean_episode_reward"],
        "historical_lcp_median_episode_reward": historical["median_episode_reward"],
        "generic_minus_historical_lcp_success_rate_pp": float(100*(summary["success_rate"]-historical["success_rate"])),
        "generic_minus_historical_lcp_mean_length_steps": float(summary["mean_episode_length"]-historical["mean_episode_length"]),
        "generic_minus_historical_lcp_mean_reward": float(summary["mean_episode_reward"]-historical["mean_episode_reward"]),
        "interpretation": "Different underlying contact dynamics; descriptive navigation comparison between simulation configurations, not a controlled solver-only or algorithm ablation. Historical 90000-90199 results did not influence Generic checkpoint selection.",
    }
    (root / "historical_lcp_comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"confirmation": summary, "historical_lcp_comparison": comparison},
                     indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
