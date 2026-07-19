#!/usr/bin/env python3
"""Load a TD-MPC2 checkpoint and run reproducible evaluation episodes."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import cv2
import numpy as np

from envs.steve_env import make_steve_env
from tdmpc2.agent import TDMPC2Agent
from tdmpc2.common import (
    MetricLogger,
    load_config,
    load_torch_checkpoint,
    select_device,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional config override; checkpoint config is used by default",
    )
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--render",
        action="store_true",
        help="Display evaluation in a SofaPygame OpenGL window",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Save rendered evaluation to an MP4 file (also enables rendering)",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=None,
        help="Output video frame rate (default: environment render FPS)",
    )
    return parser.parse_args()


def build_agent_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        **dict(config["model"]),
        **dict(config["training"]),
        **dict(config["planning"]),
    }


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = load_torch_checkpoint(checkpoint_path, map_location="cpu")
    if args.config is not None:
        config = load_config(args.config)
    elif "config" in checkpoint:
        config = copy.deepcopy(checkpoint["config"])
    else:
        raise KeyError("Checkpoint has no saved config; pass --config explicitly")

    evaluation = config["evaluation"]
    episodes = int(args.episodes if args.episodes is not None else evaluation["episodes"])
    base_seed = int(args.seed if args.seed is not None else evaluation["seed"])
    if episodes <= 0:
        raise ValueError("Evaluation episode count must be positive")
    requested_device = args.device or config["training"]["device"]
    device = select_device(requested_device)
    set_seed(base_seed)

    record_video = args.video is not None
    render_enabled = args.render or record_video
    environment_config = dict(config["environment"])
    if render_enabled:
        environment_config["render_mode"] = "human"
    env = make_steve_env(environment_config)
    observation_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    agent = TDMPC2Agent(
        observation_dim,
        action_dim,
        build_agent_config(config),
        episode_length=int(config["environment"]["max_episode_steps"]),
        device=device,
    )
    agent.load_state_dict(checkpoint["agent"], load_optimizers=False)
    logger = MetricLogger(evaluation["log_directory"])

    rewards = []
    lengths = []
    successes = []
    video_writer: Optional[cv2.VideoWriter] = None
    video_path = args.video.expanduser().resolve() if record_video else None
    video_fps = float(
        args.video_fps
        if args.video_fps is not None
        else env.metadata.get("render_fps", 7.5)
    )
    if video_fps <= 0:
        raise ValueError("Video FPS must be positive")

    def render_frame() -> None:
        nonlocal video_writer
        if not render_enabled:
            return
        frame = env.render()
        if not record_video:
            return
        if frame is None:
            raise RuntimeError("The environment did not return a frame for recording")
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise RuntimeError(f"Expected an HxWx3 RGB frame, got {frame.shape}")
        frame = np.clip(frame, 0, 255).astype(np.uint8, copy=False)
        if video_writer is None:
            assert video_path is not None
            video_path.parent.mkdir(parents=True, exist_ok=True)
            height, width = frame.shape[:2]
            video_writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                video_fps,
                (width, height),
            )
            if not video_writer.isOpened():
                video_writer.release()
                video_writer = None
                raise RuntimeError(f"Could not open MP4 video writer: {video_path}")
        # OpenCV video encoders expect BGR rather than the renderer's RGB.
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    try:
        for episode in range(episodes):
            episode_seed = base_seed + episode
            # The world-model planner is sampling-based. Reseeding makes the
            # no-exploration eval trajectory reproducible across program runs.
            set_seed(episode_seed)
            observation, _ = env.reset(seed=episode_seed)
            render_frame()
            terminated = truncated = False
            episode_reward = 0.0
            episode_length = 0
            success = False
            simulation_error = False
            while not (terminated or truncated):
                action = agent.act(
                    observation,
                    first_step=episode_length == 0,
                    eval_mode=True,
                )
                observation, reward, terminated, truncated, info = env.step(action)
                render_frame()
                episode_reward += float(reward)
                episode_length += 1
                success = success or bool(info.get("is_success", False))
                simulation_error = simulation_error or bool(
                    info.get("simulation_error", False)
                )
            rewards.append(episode_reward)
            lengths.append(episode_length)
            successes.append(float(success))
            logger.log(
                "episodes",
                {
                    "checkpoint_step": int(checkpoint.get("total_env_steps", -1)),
                    "episode": episode + 1,
                    "seed": episode_seed,
                    "episode_reward": episode_reward,
                    "episode_length": episode_length,
                    "success": success,
                    "simulation_error": simulation_error,
                },
            )
            print(
                f"episode={episode + 1} reward={episode_reward:.3f} "
                f"length={episode_length} success={int(success)}"
            )
    finally:
        if video_writer is not None:
            video_writer.release()
        env.close()

    if video_path is not None:
        print(f"Saved evaluation video: {video_path}")

    summary = {
        "checkpoint_step": int(checkpoint.get("total_env_steps", -1)),
        "episodes": episodes,
        "mean_episode_reward": float(np.mean(rewards)),
        "std_episode_reward": float(np.std(rewards)),
        "mean_episode_length": float(np.mean(lengths)),
        "success_rate": float(np.mean(successes)),
    }
    logger.log("summary", summary)
    print(
        "Evaluation summary: "
        f"reward={summary['mean_episode_reward']:.3f} +/- "
        f"{summary['std_episode_reward']:.3f}, "
        f"length={summary['mean_episode_length']:.1f}, "
        f"success_rate={summary['success_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
