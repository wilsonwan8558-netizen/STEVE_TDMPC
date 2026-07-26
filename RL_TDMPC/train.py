#!/usr/bin/env python3
"""Train TD-MPC2 on the normalized stEVE endovascular environment."""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch

from envs.safety import SAFETY_COST_NAMES
from envs.steve_env import make_steve_env
from tdmpc2.agent import TDMPC2Agent
from tdmpc2.common import (
    PROJECT_DIR,
    MetricLogger,
    apply_cli_overrides,
    atomic_torch_save,
    capture_rng_state,
    load_torch_checkpoint,
    load_config,
    resolve_project_path,
    restore_rng_state,
    select_device,
    set_seed,
)
from tdmpc2.replay_buffer import EpisodeReplayBuffer


DEFAULT_CONFIG = PROJECT_DIR / "configs" / "steve.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "YAML config (default: saved checkpoint config when resuming, "
            "otherwise configs/steve.yaml)"
        ),
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=None, help="Override total steps")
    parser.add_argument("--seed-steps", type=int, default=None)
    parser.add_argument("--initial-updates", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, help="auto, cpu, or cuda")
    return parser.parse_args()


def build_agent_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    training_horizon = int(config["training"]["horizon"])
    planning_horizon = int(config["planning"]["horizon"])
    if training_horizon != planning_horizon:
        raise ValueError("training.horizon and planning.horizon must match")
    return {
        **dict(config["model"]),
        **dict(config["training"]),
        **dict(config["planning"]),
    }


def save_checkpoint(
    *,
    path: Path,
    config: Mapping[str, Any],
    agent: TDMPC2Agent,
    replay: EpisodeReplayBuffer,
    total_env_steps: int,
    episode_index: int,
    success_count: int,
    include_replay: bool,
) -> Path:
    payload: Dict[str, Any] = {
        "format_version": 1,
        "algorithm": "TD-MPC2",
        "official_reference_commit": "e9f59321933cbc8e11a002b842adc7d4ffae8ff1",
        "config": copy.deepcopy(dict(config)),
        "agent": agent.state_dict(),
        "total_env_steps": int(total_env_steps),
        "episode_index": int(episode_index),
        "success_count": int(success_count),
        "rng_state": capture_rng_state(),
        "saved_at_unix": time.time(),
    }
    if include_replay:
        payload["replay"] = replay.state_dict()
    return atomic_torch_save(payload, path)


class LossAccumulator:
    def __init__(self) -> None:
        self.sums: Dict[str, float] = {}
        self.count = 0

    def add(self, metrics: Mapping[str, float]) -> None:
        for key, value in metrics.items():
            self.sums[key] = self.sums.get(key, 0.0) + float(value)
        self.count += 1

    def pop_means(self) -> Dict[str, float]:
        if self.count == 0:
            return {}
        means = {key: value / self.count for key, value in self.sums.items()}
        self.sums.clear()
        self.count = 0
        return means


def train(config: Dict[str, Any], resume_path: Optional[Path] = None) -> None:
    training = config["training"]
    checkpoint_config = config["checkpoint"]
    seed = int(training["seed"])
    set_seed(seed)
    device = select_device(training["device"])
    print(f"Device: {device}")

    env = make_steve_env(config["environment"])
    observation_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    agent = TDMPC2Agent(
        observation_dim,
        action_dim,
        build_agent_config(config),
        episode_length=int(config["environment"]["max_episode_steps"]),
        device=device,
    )
    replay = EpisodeReplayBuffer(
        int(training["replay_capacity"]),
        observation_dim,
        action_dim,
        int(training["horizon"]),
        int(training["batch_size"]),
        safety_cost_names=SAFETY_COST_NAMES,
        seed=seed,
    )
    logger = MetricLogger(config["logging"]["directory"])

    total_env_steps = 0
    episode_index = 0
    successes = 0
    if resume_path is not None:
        resolved_resume = resume_path.expanduser().resolve()
        checkpoint = load_torch_checkpoint(resolved_resume, map_location=device)
        agent.load_state_dict(checkpoint["agent"], load_optimizers=True)
        if "replay" in checkpoint:
            replay.load_state_dict(checkpoint["replay"])
        total_env_steps = int(checkpoint["total_env_steps"])
        episode_index = int(checkpoint.get("episode_index", 0))
        successes = int(checkpoint.get("success_count", 0))
        restore_rng_state(checkpoint.get("rng_state"))
        print(
            f"Resumed {resolved_resume} at step {total_env_steps:,}; "
            f"replay={len(replay):,}, updates={agent.update_count:,}."
        )

    total_steps = int(training["total_steps"])
    if total_env_steps >= total_steps:
        env.close()
        raise ValueError(
            f"Checkpoint is already at step {total_env_steps}, which is not below "
            f"training.total_steps={total_steps}. Use --steps with a larger value."
        )

    checkpoint_dir = resolve_project_path(checkpoint_config["directory"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_interval = int(checkpoint_config["interval"])
    update_log_interval = max(1, int(config["logging"]["update_interval"]))
    loss_accumulator = LossAccumulator()
    start_time = time.time()
    interrupted = False

    try:
        while total_env_steps < total_steps:
            episode_seed = seed + episode_index
            env.action_space.seed(episode_seed)
            observation, _ = env.reset(seed=episode_seed)
            episode_observations: List[np.ndarray] = [observation.copy()]
            episode_actions: List[np.ndarray] = []
            episode_rewards: List[float] = []
            episode_safety_costs: List[np.ndarray] = []
            episode_terminated: List[bool] = []
            episode_reward = 0.0
            success = False
            simulation_error = False

            terminated = truncated = False
            while not (terminated or truncated) and total_env_steps < total_steps:
                use_random_action = (
                    total_env_steps < int(training["seed_steps"])
                    or agent.update_count == 0
                )
                if use_random_action:
                    action = env.action_space.sample().astype(np.float32)
                else:
                    action = agent.act(
                        observation,
                        first_step=len(episode_actions) == 0,
                        eval_mode=False,
                    )

                next_observation, reward, terminated, truncated, info = env.step(action)
                total_env_steps += 1
                if total_env_steps >= total_steps and not (terminated or truncated):
                    # The requested training budget cuts this physical episode
                    # short. Treat it as a time-limit truncation for logging;
                    # it still bootstraps in replay because terminated=False.
                    truncated = True
                episode_observations.append(next_observation.copy())
                episode_actions.append(action.copy())
                episode_rewards.append(float(reward))
                episode_safety_costs.append(info["safety_cost"].copy())
                episode_terminated.append(bool(terminated))
                episode_reward += float(reward)
                success = success or bool(info.get("is_success", False))
                simulation_error = simulation_error or bool(
                    info.get("simulation_error", False)
                )
                observation = next_observation

                episode_done = terminated or truncated or total_env_steps >= total_steps
                if episode_done:
                    replay.add_episode(
                        episode_observations,
                        episode_actions,
                        episode_rewards,
                        episode_terminated,
                        safety_cost=episode_safety_costs,
                    )

                if (
                    total_env_steps >= int(training["seed_steps"])
                    and replay.can_sample()
                ):
                    if agent.update_count == 0:
                        updates = int(training["initial_updates"])
                        print(f"Pretraining world model for {updates:,} updates...")
                    else:
                        updates = int(training["updates_per_step"])
                    for _ in range(updates):
                        loss_accumulator.add(agent.update(replay))

                if (
                    agent.update_count
                    and total_env_steps % update_log_interval == 0
                ):
                    update_metrics = loss_accumulator.pop_means()
                    if update_metrics:
                        update_metrics.update(
                            {
                                "total_env_steps": total_env_steps,
                                "update_count": agent.update_count,
                                "replay_size": len(replay),
                                "elapsed_seconds": time.time() - start_time,
                            }
                        )
                        record = logger.log("updates", update_metrics)
                        print(
                            f"step={total_env_steps:,} "
                            f"loss={record['total_loss']:.4f} "
                            f"replay={len(replay):,}"
                        )

                if (
                    checkpoint_interval > 0
                    and total_env_steps % checkpoint_interval == 0
                ):
                    path = checkpoint_dir / f"step_{total_env_steps}.pt"
                    save_checkpoint(
                        path=path,
                        config=config,
                        agent=agent,
                        replay=replay,
                        total_env_steps=total_env_steps,
                        # A live SOFA state is intentionally not serialized;
                        # resume therefore starts the following episode seed.
                        episode_index=episode_index + 1,
                        success_count=(
                            successes + int(success) if episode_done else successes
                        ),
                        include_replay=bool(checkpoint_config["save_replay"]),
                    )
                    print(f"Saved checkpoint: {path}")

            successes += int(success)
            episode_index += 1
            episode_metrics = {
                "total_env_steps": total_env_steps,
                "episode": episode_index,
                "episode_reward": episode_reward,
                "episode_length": len(episode_actions),
                "success": success,
                "success_rate": successes / episode_index,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "simulation_error": simulation_error,
                "replay_size": len(replay),
            }
            logger.log("episodes", episode_metrics)
            print(
                f"episode={episode_index} step={total_env_steps:,} "
                f"reward={episode_reward:.3f} length={len(episode_actions)} "
                f"success={int(success)}"
            )
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted; saving a resumable checkpoint...")
    finally:
        if bool(checkpoint_config.get("save_final", True)) or interrupted:
            final_path = checkpoint_dir / f"step_{total_env_steps}.pt"
            save_checkpoint(
                path=final_path,
                config=config,
                agent=agent,
                replay=replay,
                total_env_steps=total_env_steps,
                episode_index=episode_index,
                success_count=successes,
                include_replay=bool(checkpoint_config["save_replay"]),
            )
            print(f"Saved final checkpoint: {final_path}")
        env.close()


def main() -> None:
    args = parse_args()
    if args.config is not None:
        config = load_config(args.config)
    elif args.resume is not None:
        resume_metadata = load_torch_checkpoint(args.resume, map_location="cpu")
        if "config" not in resume_metadata:
            raise KeyError("Resume checkpoint has no embedded config; pass --config")
        config = copy.deepcopy(resume_metadata["config"])
    else:
        config = load_config(DEFAULT_CONFIG)
    apply_cli_overrides(
        config,
        total_steps=args.steps,
        checkpoint_interval=args.checkpoint_interval,
        device=args.device,
    )
    if args.seed_steps is not None:
        config["training"]["seed_steps"] = args.seed_steps
    if args.initial_updates is not None:
        config["training"]["initial_updates"] = args.initial_updates
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.num_samples is not None:
        config["planning"]["num_samples"] = args.num_samples
        config["planning"]["num_elites"] = min(
            int(config["planning"]["num_elites"]), args.num_samples
        )
        config["planning"]["num_pi_trajs"] = min(
            int(config["planning"]["num_pi_trajs"]), args.num_samples
        )
    if args.iterations is not None:
        config["planning"]["iterations"] = args.iterations
    train(config, args.resume)


if __name__ == "__main__":
    main()
