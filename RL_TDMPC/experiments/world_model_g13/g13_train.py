"""Train one 300k-step geometry/force-supervised TD-MPC2 model from scratch."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
sys.path.insert(0, str(RL))
sys.path.insert(0, str(HERE))
from tdmpc2.common import (MetricLogger, atomic_torch_save, capture_rng_state,
                           resolve_project_path, set_seed)  # noqa: E402
from train import LossAccumulator, build_agent_config  # noqa: E402
from g13_env import make_g13_env  # noqa: E402
from g13_model import G13Agent, G13ReplayBuffer  # noqa: E402

OUT = RL / "logs/world_model_force_aware_planning_g13"
CONFIG = HERE / "g13_config.yaml"
CHECKPOINT_STEPS_FOR_NAVIGATION_SELECTION = [240000, 260000, 280000, 300000]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_checkpoint(path: Path, config: dict, agent: G13Agent,
                    replay: G13ReplayBuffer, total_steps: int,
                    episode_index: int, success_count: int) -> None:
    payload = {
        "format_version": 2,
        "algorithm": "Geometry-Force-TD-MPC2",
        "config": copy.deepcopy(config),
        "agent": agent.state_dict(),
        "total_env_steps": total_steps,
        "episode_index": episode_index,
        "success_count": success_count,
        "rng_state": capture_rng_state(),
        "saved_at_unix": time.time(),
        "observation_dim": 22,
        "force_target": "next-RL-interval max-substep distal contact force, nominal N",
        "force_head_attachment": "predicted next latent WorldModel.next(z_t,a_t)",
        "lambda_force_train": agent.lambda_force_train,
        "replay": replay.state_dict(),
    }
    atomic_torch_save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Diagnostic smoke cap only; omit for the single 300k primary run")
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG.read_text())
    training = config["training"]
    total_budget = int(training["total_steps"])
    if args.max_steps is not None:
        if not 0 < args.max_steps <= total_budget:
            raise ValueError("Invalid smoke cap")
        total_budget = args.max_steps
    if (training["device"] != "cuda:0" or not torch.cuda.is_available() or
            torch.cuda.device_count() != 1 or "RTX 5090" not in torch.cuda.get_device_name(0)):
        raise RuntimeError("G13 primary training requires CUDA_VISIBLE_DEVICES=0 RTX 5090")
    if float(training["lambda_force_train"]) != 0.5:
        raise RuntimeError("Frozen G13 force-loss coefficient changed")
    primary_run_dir = resolve_project_path(config["checkpoint"]["directory"]).parent
    run_dir = (primary_run_dir if args.max_steps is None else
               primary_run_dir.with_name(primary_run_dir.name + f"_smoke_{args.max_steps}"))
    checkpoint_dir = run_dir / "checkpoints"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"G13 training run already exists; refusing second primary run: {run_dir}")
    freeze_path = (OUT / "g13_training_config.json" if args.max_steps is None else
                   run_dir / "g13_smoke_config.json")
    if freeze_path.exists():
        raise FileExistsError("G13 training config was already frozen")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    frozen = {
        "status": "FROZEN_BEFORE_ONE_PRIMARY_G13_TRAINING_RUN",
        "config": config,
        "config_yaml_sha256": sha(CONFIG),
        "geometry_feature_definition_sha256": sha(OUT / "geometry_feature_definition.json"),
        "geometry_normalization_sha256": sha(OUT / "geometry_normalization.json"),
        "geometry_feature_source_sha256": sha(RL / "envs/geometry_features.py"),
        "geometry_adapter_source_sha256": sha(HERE / "g13_env.py"),
        "joint_model_source_sha256": sha(HERE / "g13_model.py"),
        "selected_checkpoint_candidates": CHECKPOINT_STEPS_FOR_NAVIGATION_SELECTION,
        "selection_primary": "development navigation success, stable reward, no obvious collapse; force-planning outcomes excluded",
        "single_primary_model": True,
        "CUDA_VISIBLE_DEVICES": "0",
        "CUDA_index": 0,
        "CUDA_name": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "python": sys.executable,
        "smoke_cap": args.max_steps,
    }
    with freeze_path.open("x", encoding="utf-8") as stream:
        json.dump(frozen, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    set_seed(int(training["seed"]))
    env = make_g13_env(config["environment"])
    agent = G13Agent(22, 2, build_agent_config(config),
                     episode_length=int(config["environment"]["max_episode_steps"]),
                     device=torch.device("cuda:0"))
    replay = G13ReplayBuffer(int(training["replay_capacity"]), 22, 2,
                             int(training["horizon"]), int(training["batch_size"]),
                             seed=int(training["seed"]))
    log_directory = (config["logging"]["directory"] if args.max_steps is None else
                     str(run_dir / "logs"))
    logger = MetricLogger(log_directory)
    loss_accumulator = LossAccumulator()
    total_env_steps = 0
    episode_index = 0
    successes = 0
    start_time = time.time()
    try:
        while total_env_steps < total_budget:
            episode_seed = int(training["seed"]) + episode_index
            env.action_space.seed(episode_seed)
            observation, _ = env.reset(seed=episode_seed)
            observations = [observation.copy()]
            actions, rewards, terminations, forces = [], [], [], []
            episode_reward = 0.0
            success = simulation_error = False
            terminated = truncated = False
            while not (terminated or truncated) and total_env_steps < total_budget:
                random_action = (total_env_steps < int(training["seed_steps"]) or
                                 agent.update_count == 0)
                action = (env.action_space.sample().astype(np.float32) if random_action
                          else agent.act(observation, first_step=len(actions) == 0,
                                         eval_mode=False))
                next_observation, reward, terminated, truncated, info = env.step(action)
                total_env_steps += 1
                if total_env_steps >= total_budget and not (terminated or truncated):
                    truncated = True
                force = float(info["distal_contact_force_max_N"])
                if not np.isfinite(force) or force < 0:
                    raise FloatingPointError("G13 frozen distal force label invalid")
                observations.append(next_observation.copy())
                actions.append(action.copy())
                rewards.append(float(reward))
                terminations.append(bool(terminated))
                forces.append(force)
                episode_reward += float(reward)
                success = success or bool(info.get("is_success", False))
                simulation_error = simulation_error or bool(info.get("simulation_error", False))
                observation = next_observation
                episode_done = terminated or truncated or total_env_steps >= total_budget
                if episode_done:
                    replay.add_episode(observations, actions, rewards,
                                       terminations, forces)
                if total_env_steps >= int(training["seed_steps"]) and replay.can_sample():
                    updates = (int(training["initial_updates"]) if agent.update_count == 0
                               else int(training["updates_per_step"]))
                    for _ in range(updates):
                        loss_accumulator.add(agent.update(replay))
                if agent.update_count and total_env_steps % int(config["logging"]["update_interval"]) == 0:
                    metrics = loss_accumulator.pop_means()
                    if metrics:
                        metrics.update({"total_env_steps": total_env_steps,
                                        "update_count": agent.update_count,
                                        "replay_size": len(replay),
                                        "elapsed_seconds": time.time() - start_time})
                        logger.log("updates", metrics)
                        if total_env_steps % 1000 == 0:
                            print("G13_TRAIN_STEP", total_env_steps,
                                  "loss", round(metrics["total_loss"], 5),
                                  "force_loss", round(metrics["force_loss"], 5),
                                  "success_rate", round(successes / max(1, episode_index), 3),
                                  flush=True)
                if (total_env_steps % int(config["checkpoint"]["interval"]) == 0 and
                        args.max_steps is None):
                    path = checkpoint_dir / f"step_{total_env_steps}.pt"
                    save_checkpoint(path, config, agent, replay, total_env_steps,
                                    episode_index + (1 if episode_done else 0),
                                    successes + (int(success) if episode_done else 0))
                    print("G13_TRAIN_CHECKPOINT", path, flush=True)
            successes += int(success)
            episode_index += 1
            logger.log("episodes", {
                "total_env_steps": total_env_steps,
                "episode": episode_index,
                "episode_reward": episode_reward,
                "episode_length": len(actions),
                "success": success,
                "success_rate": successes / episode_index,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "simulation_error": simulation_error,
                "replay_size": len(replay),
                "episode_peak_force_N": float(max(forces)),
                "episode_above_F_lim_steps": sum(f > 3.685331040756903 for f in forces),
            })
            if episode_index % 50 == 0:
                print("G13_TRAIN_EPISODE", episode_index,
                      "steps", total_env_steps,
                      "success_rate", round(successes / episode_index, 4), flush=True)
    finally:
        env.close()
    if args.max_steps is None:
        final = checkpoint_dir / f"step_{total_env_steps}.pt"
        if not final.exists():
            save_checkpoint(final, config, agent, replay, total_env_steps,
                            episode_index, successes)
        print("G13_TRAIN_COMPLETE", total_env_steps, episode_index,
              successes / max(1, episode_index), time.time() - start_time, flush=True)
    else:
        print("G13_TRAIN_SMOKE_COMPLETE", total_env_steps, flush=True)


if __name__ == "__main__":
    main()
