"""Evaluate the four frozen G13 checkpoints on task-only development seeds."""

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
sys.path.insert(0, str(HERE))
from tdmpc2.common import load_torch_checkpoint, resolve_project_path, set_seed  # noqa: E402
from train import build_agent_config  # noqa: E402
from g13_env import make_g13_env  # noqa: E402
from g13_model import G13Agent  # noqa: E402

OUT = RL / "logs/world_model_force_aware_planning_g13"
CANDIDATES = (240000, 260000, 280000, 300000)
FIELDS = ("checkpoint_step", "seed", "success", "episode_length",
          "episode_reward", "simulation_error", "terminated", "truncated")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("G13 checkpoint selection requires CUDA_VISIBLE_DEVICES=0")
    audit = json.loads((OUT / "checkpoint_selection_seed_audit.json").read_text())
    manifest = json.loads((OUT / "checkpoint_selection_seed_manifest.json").read_text())
    frozen = json.loads((OUT / "g13_training_config.json").read_text())
    if (frozen["selected_checkpoint_candidates"] != list(CANDIDATES)
            or frozen["config_yaml_sha256"] != sha256(HERE / "g13_config.yaml")
            or frozen["geometry_feature_source_sha256"] != sha256(RL / "envs/geometry_features.py")
            or frozen["joint_model_source_sha256"] != sha256(HERE / "g13_model.py")):
        raise RuntimeError("Frozen G13 candidate set, config, or model identity changed")
    if audit["status"] != "PASS_UNUSED_IN_SCANNED_ARTIFACTS":
        raise RuntimeError("Development seed audit did not pass")
    seeds = [int(seed) for seed in manifest["ordered_seeds"]]
    if seeds != list(range(350000, 350030)):
        raise RuntimeError("Frozen checkpoint-selection seeds changed")
    if (OUT / "checkpoint_selection.json").exists():
        raise FileExistsError("G13 checkpoint has already been selected and frozen")
    path = OUT / "navigation_checkpoint_candidates.csv"
    saved = list(csv.DictReader(path.open(newline="", encoding="utf-8"))) if path.exists() else []
    expected_order = [(step, seed) for step in CANDIDATES for seed in seeds]
    saved_order = [(int(row["checkpoint_step"]), int(row["seed"])) for row in saved]
    if saved_order != expected_order[:len(saved_order)]:
        raise RuntimeError("Saved checkpoint evaluations are not an ordered prefix")
    run = RL / "runs/geometry_force_tdmpc2_g13_300k_20261014/checkpoints"
    for step in CANDIDATES:
        if not (run / f"step_{step}.pt").is_file():
            raise FileNotFoundError(run / f"step_{step}.pt")
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if not saved:
            writer.writeheader()
        for step in CANDIDATES:
            checkpoint = load_torch_checkpoint(run / f"step_{step}.pt", map_location="cpu")
            if (checkpoint.get("algorithm") != "Geometry-Force-TD-MPC2"
                    or checkpoint["total_env_steps"] != step
                    or checkpoint.get("observation_dim") != 22
                    or checkpoint.get("lambda_force_train") != 0.5
                    or checkpoint["config"] != frozen["config"]):
                raise RuntimeError(f"Unexpected G13 checkpoint format at {step}")
            config = checkpoint["config"]
            set_seed(seeds[0])
            env = make_g13_env(config["environment"])
            agent = G13Agent(22, 2, build_agent_config(config),
                             episode_length=int(config["environment"]["max_episode_steps"]),
                             device=torch.device("cuda:0"))
            agent.load_state_dict(checkpoint["agent"], load_optimizers=False)
            del checkpoint
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
                    record = {"checkpoint_step": step, "seed": seed,
                              "success": int(success), "episode_length": length,
                              "episode_reward": total_reward,
                              "simulation_error": int(simulation_error),
                              "terminated": int(terminated), "truncated": int(truncated)}
                    record_index = expected_order.index((step, seed))
                    if record_index < len(saved):
                        earlier = saved[record_index]
                        for key in FIELDS:
                            if key == "episode_reward":
                                if abs(float(earlier[key]) - float(record[key])) > 1e-9:
                                    raise RuntimeError(f"Checkpoint replay mismatch {step}/{seed}/{key}")
                            elif str(earlier[key]) != str(record[key]):
                                raise RuntimeError(f"Checkpoint replay mismatch {step}/{seed}/{key}")
                    else:
                        writer.writerow(record)
                        stream.flush()
                    print("G13_CHECKPOINT_EVAL", step, seed, int(success), length,
                          round(total_reward, 5), flush=True)
            finally:
                env.close()
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    aggregate = []
    for step in CANDIDATES:
        current = [row for row in rows if int(row["checkpoint_step"]) == step]
        if [int(row["seed"]) for row in current] != seeds:
            raise RuntimeError(f"Incomplete checkpoint evaluation at {step}")
        rewards = [float(row["episode_reward"]) for row in current]
        successes = [int(row["success"]) for row in current]
        lengths = [int(row["episode_length"]) for row in current]
        aggregate.append({"checkpoint_step": step, "successes": sum(successes),
                          "success_rate": float(np.mean(successes)),
                          "mean_reward": float(np.mean(rewards)),
                          "mean_episode_length": float(np.mean(lengths)),
                          "simulation_errors": sum(int(row["simulation_error"]) for row in current)})
    # This selection uses task-only navigation outcomes; no force prediction
    # or force-planning outcome enters the ordering.
    ranked = sorted(aggregate, key=lambda row: (
        row["successes"], row["mean_reward"], -row["simulation_errors"],
        row["checkpoint_step"]), reverse=True)
    selected = ranked[0]
    selected_path = run / f"step_{selected['checkpoint_step']}.pt"
    result = {
        "status": "SELECTED_FROM_PREDECLARED_TASK_ONLY_DEVELOPMENT_SET",
        "ordered_development_seeds": seeds,
        "candidate_steps": list(CANDIDATES),
        "selection_order": "highest successes, then highest mean reward, then fewest simulation errors, then latest step",
        "candidate_navigation_metrics": aggregate,
        "selected_checkpoint": str(selected_path.resolve()),
        "selected_checkpoint_sha256": sha256(selected_path),
        "selected_step": selected["checkpoint_step"],
        "force_planning_used_in_selection": False,
        "normal_task_planner_only": True,
    }
    (OUT / "checkpoint_selection.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("G13_SELECTED_CHECKPOINT", selected["checkpoint_step"], flush=True)


if __name__ == "__main__":
    main()
