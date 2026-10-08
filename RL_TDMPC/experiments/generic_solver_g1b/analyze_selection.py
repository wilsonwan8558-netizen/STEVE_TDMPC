"""Navigation-only six-checkpoint summary and 1,000 paired bootstrap comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

STEPS = (200000, 220000, 240000, 260000, 280000, 300000)
SEEDS = tuple(range(91000, 91100))
BOOTSTRAP_SEED = 20261007
BOOTSTRAP_REPLICATES = 1000


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "checkpoint_manifest.json").read_text())
    expected_sha = {int(row["step"]): row["sha256"] for row in manifest["checkpoint_records"]}
    groups = {}
    for step in STEPS:
        rows = read(root / "selection" / f"step_{step}.csv")
        if len(rows) != 100 or [int(row["seed"]) for row in rows] != list(SEEDS):
            raise RuntimeError(f"Wrong selection seed set/order or incomplete step {step}")
        if any(int(row["checkpoint_step"]) != step or row["checkpoint_sha256"] != expected_sha[step]
               for row in rows):
            raise RuntimeError(f"Checkpoint identity mismatch at step {step}")
        if any(row["simulation_error"] != "False" for row in rows):
            raise RuntimeError(f"Simulation error at step {step}; investigate before selection")
        if any(not np.isfinite(float(row["episode_reward"])) for row in rows):
            raise RuntimeError(f"Nonfinite reward at step {step}")
        groups[step] = rows
    merged = [row for step in STEPS for row in groups[step]]
    write(root / "selection_episode_results.csv", merged)
    summaries = []
    for step in STEPS:
        rows = groups[step]
        success = np.asarray([row["success"] == "True" for row in rows], dtype=bool)
        length = metric(rows, "episode_length")
        reward = metric(rows, "episode_reward")
        summaries.append({
            "checkpoint_step": step, "checkpoint_sha256": expected_sha[step],
            "episodes": 100, "success_count": int(np.count_nonzero(success)),
            "success_rate": float(np.mean(success)),
            "mean_episode_length": float(np.mean(length)),
            "median_episode_length": float(np.median(length)),
            "mean_episode_reward": float(np.mean(reward)),
            "median_episode_reward": float(np.median(reward)),
            "mean_success_episode_length": float(np.mean(length[success])) if np.any(success) else None,
            "mean_failure_episode_length": float(np.mean(length[~success])) if np.any(~success) else None,
        })
    write(root / "selection_checkpoint_summary.csv", summaries)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_indices = rng.integers(0, 100, size=(BOOTSTRAP_REPLICATES, 100))
    pair_rows = []
    bootstrap = []
    for earlier, later in zip(STEPS[:-1], STEPS[1:]):
        a = groups[earlier]
        b = groups[later]
        a_success = np.asarray([row["success"] == "True" for row in a], dtype=int)
        b_success = np.asarray([row["success"] == "True" for row in b], dtype=int)
        ds = b_success-a_success
        dl = metric(b, "episode_length")-metric(a, "episode_length")
        dr = metric(b, "episode_reward")-metric(a, "episode_reward")
        bs = np.mean(ds[bootstrap_indices], axis=1)
        bl = np.mean(dl[bootstrap_indices], axis=1)
        br = np.mean(dr[bootstrap_indices], axis=1)
        ci_s = np.quantile(bs, (.025, .975))*100
        ci_l = np.quantile(bl, (.025, .975))
        ci_r = np.quantile(br, (.025, .975))
        pair = {
            "earlier_step": earlier, "later_step": later,
            "both_success": int(np.count_nonzero((a_success == 1) & (b_success == 1))),
            "earlier_only_success": int(np.count_nonzero((a_success == 1) & (b_success == 0))),
            "later_only_success": int(np.count_nonzero((a_success == 0) & (b_success == 1))),
            "both_failure": int(np.count_nonzero((a_success == 0) & (b_success == 0))),
            "later_minus_earlier_success_rate_pp": float(100*np.mean(ds)),
            "later_minus_earlier_mean_length_steps": float(np.mean(dl)),
            "later_minus_earlier_mean_reward": float(np.mean(dr)),
            "paired_length_difference_median_steps": float(np.median(dl)),
            "paired_reward_difference_median": float(np.median(dr)),
            "success_diff_ci95_low_pp": float(ci_s[0]),
            "success_diff_ci95_high_pp": float(ci_s[1]),
            "length_diff_ci95_low_steps": float(ci_l[0]),
            "length_diff_ci95_high_steps": float(ci_l[1]),
            "reward_diff_ci95_low": float(ci_r[0]),
            "reward_diff_ci95_high": float(ci_r[1]),
        }
        pair_rows.append(pair)
        bootstrap.append({
            "earlier_step": earlier, "later_step": later,
            "success_rate_difference_pp": {"observed": pair["later_minus_earlier_success_rate_pp"],
                                           "ci95": [float(ci_s[0]), float(ci_s[1])],
                                           "replicates_later_higher_fraction": float(np.mean(bs > 0))},
            "mean_length_difference_steps": {"observed": pair["later_minus_earlier_mean_length_steps"],
                                             "ci95": [float(ci_l[0]), float(ci_l[1])],
                                             "replicates_later_shorter_fraction": float(np.mean(bl < 0))},
            "mean_reward_difference": {"observed": pair["later_minus_earlier_mean_reward"],
                                       "ci95": [float(ci_r[0]), float(ci_r[1])],
                                       "replicates_later_higher_fraction": float(np.mean(br > 0))},
        })
    write(root / "paired_checkpoint_comparison.csv", pair_rows)
    (root / "bootstrap_summary.json").write_text(json.dumps({
        "method": "Paired episode-level nonparametric bootstrap, same 100-seed index resample for each metric/comparison; percentile 95% intervals.",
        "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES,
        "selection_seeds": [SEEDS[0], SEEDS[-1]], "comparisons": bootstrap,
    }, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"summaries": summaries, "paired": pair_rows}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
