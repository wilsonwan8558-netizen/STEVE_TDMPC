"""Window G1 navigation training metrics from 180k through 300k."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def numeric(rows: list[dict[str, str]], field: str) -> np.ndarray:
    return np.asarray([float(row[field]) for row in rows], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    episodes = read(args.logs / "episodes.csv")
    updates = read(args.logs / "updates.csv")
    if not episodes or not updates or int(episodes[-1]["total_env_steps"]) != 300000:
        raise RuntimeError("G1 training logs are absent/incomplete")
    windows = []
    for end in range(200000, 300001, 20000):
        start = end - 20000
        e = [row for row in episodes if start < int(row["total_env_steps"]) <= end]
        u = [row for row in updates if start < int(row["total_env_steps"]) <= end]
        if not e or not u:
            raise RuntimeError(f"Missing training metrics in ({start},{end}]")
        rewards = numeric(e, "episode_reward")
        lengths = numeric(e, "episode_length")
        successes = np.asarray([row["success"] == "True" for row in e], dtype=float)
        losses = numeric(u, "total_loss")
        if not (np.all(np.isfinite(rewards)) and np.all(np.isfinite(losses))):
            raise RuntimeError(f"Nonfinite training reward/loss in ({start},{end}]")
        windows.append({
            "start_exclusive": start, "end_inclusive": end,
            "episodes": len(e), "update_records": len(u),
            "success_rate": float(np.mean(successes)),
            "mean_episode_reward": float(np.mean(rewards)),
            "median_episode_reward": float(np.median(rewards)),
            "mean_episode_length": float(np.mean(lengths)),
            "total_loss_mean": float(np.mean(losses)),
            "total_loss_median": float(np.median(losses)),
            "total_loss_q95": float(np.quantile(losses, .95)),
            "total_loss_max": float(np.max(losses)),
            "simulation_error_episodes": sum(row["simulation_error"] == "True" for row in e),
        })
    output = {
        "status": "COMPLETE",
        "training_final_step": 300000,
        "windows": windows,
        "first_three_windows_mean_success_rate": float(np.mean([w["success_rate"] for w in windows[:3]])),
        "last_three_windows_mean_success_rate": float(np.mean([w["success_rate"] for w in windows[3:]])),
        "first_three_windows_mean_reward": float(np.mean([w["mean_episode_reward"] for w in windows[:3]])),
        "last_three_windows_mean_reward": float(np.mean([w["mean_episode_reward"] for w in windows[3:]])),
        "first_three_windows_mean_loss": float(np.mean([w["total_loss_mean"] for w in windows[:3]])),
        "last_three_windows_mean_loss": float(np.mean([w["total_loss_mean"] for w in windows[3:]])),
        "note": "Training metrics are noisy support only; checkpoint selection uses the separate 91000-91099 evaluation set.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
