"""Compare selected G13 task-only navigation with the frozen Generic baseline."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
sys.path.insert(0, str(RL))
from envs.steve_env import make_steve_env  # noqa: E402
from tdmpc2.agent import TDMPC2Agent  # noqa: E402
from tdmpc2.common import load_torch_checkpoint, set_seed  # noqa: E402
from train import build_agent_config  # noqa: E402

OUT = RL / "logs/world_model_force_aware_planning_g13"
OLD = RL / "runs/generic_solver_tdmpc2_baseline_300k_20261006/checkpoints/step_280000.pt"
FIELDS = ("arm", "seed", "success", "episode_length", "episode_reward",
          "simulation_error", "terminated", "truncated", "time_to_target_s")
RL_INTERVAL_SECONDS = 22 * 0.006


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Navigation comparison requires CUDA_VISIBLE_DEVICES=0")
    selected = json.loads((OUT / "checkpoint_selection.json").read_text())
    seeds = selected["ordered_development_seeds"]
    selection_rows = list(csv.DictReader(
        (OUT / "navigation_checkpoint_candidates.csv").open(newline="", encoding="utf-8")))
    g13_rows = [row for row in selection_rows
                if int(row["checkpoint_step"]) == selected["selected_step"]]
    if [int(row["seed"]) for row in g13_rows] != seeds:
        raise RuntimeError("Selected checkpoint navigation records incomplete")
    expected_old_sha = json.loads((OUT / "metadata.json").read_text())["old_tdmpc2_checkpoint_sha256"]
    if sha256(OLD) != expected_old_sha:
        raise RuntimeError("Frozen Generic baseline checkpoint SHA changed")
    if (OUT / "navigation_baseline_summary.json").exists():
        raise FileExistsError("G13 navigation comparison already summarized")
    checkpoint = load_torch_checkpoint(OLD, map_location="cpu")
    config = checkpoint["config"]
    set_seed(seeds[0])
    env = make_steve_env(config["environment"])
    agent = TDMPC2Agent(14, 2, build_agent_config(config),
                        episode_length=int(config["environment"]["max_episode_steps"]),
                        device=torch.device("cuda:0"))
    agent.load_state_dict(checkpoint["agent"], load_optimizers=False)
    del checkpoint
    path = OUT / "navigation_baseline_metrics.csv"
    prior = list(csv.DictReader(path.open(newline="", encoding="utf-8"))) if path.exists() else []
    expected_prefix = ([ ("G13_task_only", seed) for seed in seeds ] +
                       [ ("Generic_frozen", seed) for seed in seeds ])
    if [(row["arm"], int(row["seed"])) for row in prior] != expected_prefix[:len(prior)]:
        raise RuntimeError("Saved navigation comparison rows are not an ordered prefix")
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if not prior:
            writer.writeheader()
        for index, row in enumerate(g13_rows):
            record = {"arm": "G13_task_only", "seed": int(row["seed"]),
                      "success": int(row["success"]),
                      "episode_length": int(row["episode_length"]),
                      "episode_reward": float(row["episode_reward"]),
                      "simulation_error": int(row["simulation_error"]),
                      "terminated": int(row["terminated"]),
                      "truncated": int(row["truncated"]),
                      "time_to_target_s": int(row["episode_length"]) * RL_INTERVAL_SECONDS
                      if int(row["success"]) else ""}
            if index < len(prior):
                earlier = prior[index]
                if any(str(earlier[key]) != str(record[key]) for key in FIELDS):
                    raise RuntimeError(f"Saved G13 navigation row changed at seed {row['seed']}")
            else:
                writer.writerow(record)
        stream.flush()
        try:
            for seed in seeds:
                set_seed(seed)
                observation, _ = env.reset(seed=seed)
                terminated = truncated = success = simulation_error = False
                total_reward = 0.0
                length = 0
                while not (terminated or truncated):
                    action = agent.act(observation, first_step=length == 0, eval_mode=True)
                    observation, reward, terminated, truncated, info = env.step(action)
                    total_reward += float(reward)
                    length += 1
                    success = success or bool(info.get("is_success", False))
                    simulation_error = simulation_error or bool(info.get("simulation_error", False))
                record = {"arm": "Generic_frozen", "seed": seed,
                          "success": int(success), "episode_length": length,
                          "episode_reward": total_reward,
                          "simulation_error": int(simulation_error),
                          "terminated": int(terminated), "truncated": int(truncated),
                          "time_to_target_s": length * RL_INTERVAL_SECONDS if success else ""}
                index = len(seeds) + seeds.index(seed)
                if index < len(prior):
                    earlier = prior[index]
                    for key in FIELDS:
                        if key == "episode_reward":
                            if abs(float(earlier[key]) - float(record[key])) > 1e-9:
                                raise RuntimeError(f"Saved Generic replay mismatch {seed}/{key}")
                        elif str(earlier[key]) != str(record[key]):
                            raise RuntimeError(f"Saved Generic replay mismatch {seed}/{key}")
                else:
                    writer.writerow(record)
                    stream.flush()
                print("G13_NAV_COMPARISON", seed, int(success), length,
                      round(total_reward, 5), flush=True)
        finally:
            env.close()
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    metrics = {}
    for arm in ("G13_task_only", "Generic_frozen"):
        subset = [row for row in rows if row["arm"] == arm]
        if [int(row["seed"]) for row in subset] != seeds:
            raise RuntimeError(f"Incomplete {arm} navigation rows")
        metrics[arm] = {
            "successes": sum(int(row["success"]) for row in subset),
            "success_rate": float(np.mean([int(row["success"]) for row in subset])),
            "mean_reward": float(np.mean([float(row["episode_reward"]) for row in subset])),
            "mean_episode_length": float(np.mean([int(row["episode_length"]) for row in subset])),
            "simulation_errors": sum(int(row["simulation_error"]) for row in subset),
            "truncated_episodes": sum(int(row["truncated"]) for row in subset),
            "mean_success_only_time_to_target_s": float(np.mean([
                float(row["time_to_target_s"]) for row in subset if int(row["success"])]))
                if any(int(row["success"]) for row in subset) else None,
        }
    (OUT / "navigation_baseline_summary.json").write_text(
        json.dumps({"seeds": seeds, "metrics": metrics,
                    "comparison": "paired development seeds, one reused scene per controller; G13 task-only planner"},
                   indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("G13_NAV_SUMMARY", metrics, flush=True)


if __name__ == "__main__":
    main()
