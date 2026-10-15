"""Summarize frozen G12.1 development true-force and navigation outcomes."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from g121_planner import OUT, F_LIM


def write_csv(name: str, rows: list[dict]) -> None:
    with (OUT / name).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def events(flags: list[bool]) -> list[int]:
    result = []
    run = 0
    for flag in flags:
        if flag:
            run += 1
        elif run:
            result.append(run)
            run = 0
    if run:
        result.append(run)
    return result


def main() -> None:
    manifest = json.loads((OUT / "development_seed_manifest.json").read_text())
    weights = json.loads((OUT / "lambda_scale_definition.json").read_text())["fixed_lambda_set"]
    seeds = manifest["ordered_seeds"]
    episodes = []
    steps = []
    for arm_index, weight in enumerate(weights):
        folder = OUT / "development_arms" / f"arm_{arm_index}"
        paths = sorted(folder.glob("seed_*.json"))
        if [int(path.stem.split("_")[1]) for path in paths] != seeds:
            raise RuntimeError(f"G12.1 development arm {arm_index} missing/out-of-order seeds")
        for path in paths:
            episode = json.loads(path.read_text())
            if episode["seed"] != int(path.stem.split("_")[1]) or episode["lambda_F"] != weight:
                raise RuntimeError("G12.1 development episode identity mismatch")
            rows = episode.pop("steps")
            if len(rows) != episode["episode_length"]:
                raise RuntimeError("G12.1 episode step count mismatch")
            forces = np.asarray([step["true_distal_force_max_N"] for step in rows], dtype=np.float64)
            highs = [force > F_LIM for force in forces]
            durations = events(highs)
            episodes.append({**episode,
                             "episode_peak_force_N": float(np.max(forces)),
                             "episode_above_F_lim_steps": int(sum(highs)),
                             "episode_high_force_event_count": len(durations),
                             "episode_max_high_force_event_duration_steps": max(durations, default=0),
                             "episode_force_Q95_N": float(np.quantile(forces, .95)),
                             "episode_force_Q99_N": float(np.quantile(forces, .99))})
            steps.extend({"arm":episode["arm"], "lambda_F": weight, **row} for row in rows)
    write_csv("development_episode_results.csv", episodes)
    write_csv("development_step_results.csv", steps)
    force_rows = []
    task_rows = []
    for weight in weights:
        selected_steps = [step for step in steps if step["lambda_F"] == weight]
        selected_eps = [ep for ep in episodes if ep["lambda_F"] == weight]
        if len(selected_eps) != len(seeds):
            raise RuntimeError("G12.1 development paired cohort incomplete")
        force = np.asarray([row["true_distal_force_max_N"] for row in selected_steps], dtype=np.float64)
        high = force > F_LIM
        durations = [length for ep in selected_eps for length in events([
            step["true_distal_force_max_N"] > F_LIM
            for step in selected_steps if step["seed"] == ep["seed"]])]
        force_rows.append({
            "lambda_F": weight, "arm": "B0" if weight == 0 else "B1",
            "episode_count": len(selected_eps), "RL_step_count": len(force),
            "true_force_Q95_N": float(np.quantile(force,.95)),
            "true_force_Q99_N": float(np.quantile(force,.99)),
            "true_force_max_N": float(np.max(force)),
            "true_above_F_lim_steps": int(high.sum()),
            "true_above_F_lim_step_fraction": float(np.mean(high)),
            "high_force_event_count": len(durations),
            "mean_high_force_event_duration_steps": float(np.mean(durations)) if durations else None,
            "Q90_high_force_event_duration_steps": float(np.quantile(durations,.9)) if durations else None,
            "mean_high_force_event_duration_s": float(np.mean(durations)*.132) if durations else None,
            "Q90_high_force_event_duration_s": float(np.quantile(durations,.9)*.132) if durations else None,
        })
        success_times = [ep["time_to_target_s"] for ep in selected_eps if ep["success"]]
        changes = np.asarray([step["first_action_changed_vs_same_state_baseline"] for step in selected_steps], dtype=bool)
        task_rows.append({
            "lambda_F": weight, "arm": "B0" if weight == 0 else "B1",
            "episode_count": len(selected_eps),
            "success_count": int(sum(ep["success"] for ep in selected_eps)),
            "success_rate": float(np.mean([ep["success"] for ep in selected_eps])),
            "mean_episode_reward": float(np.mean([ep["episode_reward"] for ep in selected_eps])),
            "median_episode_reward": float(np.median([ep["episode_reward"] for ep in selected_eps])),
            "mean_episode_length_steps": float(np.mean([ep["episode_length"] for ep in selected_eps])),
            "median_episode_length_steps": float(np.median([ep["episode_length"] for ep in selected_eps])),
            "mean_time_to_target_s_successes_only": float(np.mean(success_times)) if success_times else None,
            "control_step_count": len(selected_steps),
            "different_first_action_fraction_same_state": float(np.mean(changes)),
            "mean_first_action_L2_difference_same_state": float(np.mean([step["first_action_L2_difference"] for step in selected_steps])),
            "mean_abs_translation_difference_same_state": float(np.mean([abs(step["translation_difference"]) for step in selected_steps])),
            "mean_abs_rotation_difference_same_state": float(np.mean([abs(step["rotation_difference"]) for step in selected_steps])),
        })
    write_csv("development_force_metrics.csv", force_rows)
    write_csv("development_task_metrics.csv", task_rows)
    print("G121_DEVELOPMENT_ANALYZED", force_rows, task_rows, flush=True)


if __name__ == "__main__":
    main()
