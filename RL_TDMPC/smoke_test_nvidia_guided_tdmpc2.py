#!/usr/bin/env python3
"""Run one real NVIDIA-guided collection and one TD-MPC2 update.

This is deliberately a smoke test, not a training entry point. It collects a
few deterministic guided transitions, verifies the complete six-channel
safety path through replay, and calls ``TDMPC2Agent.update`` exactly once.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import torch

from envs import make_env
from safety_schema import NVIDIA_GUIDED_SAFETY_COST_NAMES
from tdmpc2.agent import TDMPC2Agent
from tdmpc2.common import load_config
from tdmpc2.replay_buffer import EpisodeReplayBuffer
from train import build_agent_config


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_DIR / "configs" / "nvidia_guided.yaml"
_SOFA_MODULE_PREFIXES = ("Sofa", "SofaRuntime")
_FINITE_UPDATE_METRICS = (
    "total_loss",
    "consistency_loss",
    "reward_loss",
    "value_loss",
    "termination_loss",
    "model_grad_norm",
    "safety_loss",
    "policy_loss",
    "policy_grad_norm",
    "policy_entropy",
    "policy_scale",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workflow-root",
        type=Path,
        required=True,
        help="Local i4h-workflows repository root",
    )
    parser.add_argument(
        "--ct-cache",
        type=Path,
        required=True,
        help="NVIDIA case cache containing the guided CT assets",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="NVIDIA TD-MPC2 config (default: configs/nvidia_guided.yaml)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for the one agent update: cpu, cuda, cuda:0, or auto",
    )
    parser.add_argument(
        "--collection-steps",
        type=int,
        default=4,
        help="Number of real guided transitions (must cover the horizon)",
    )
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def _assert_sofa_not_imported(stage: str) -> None:
    loaded = sorted(
        name
        for name in sys.modules
        if any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in _SOFA_MODULE_PREFIXES
        )
    )
    if loaded:
        raise AssertionError(
            f"SOFA was imported during NVIDIA smoke test ({stage}): {loaded}"
        )


def _resolve_device(requested: str) -> torch.device:
    normalized = requested.strip().lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is unavailable")
    return device


def _safety_cost(
    info: Mapping[str, object],
    *,
    stage: str,
) -> np.ndarray:
    names = tuple(info.get("safety_cost_names", ()))
    if names != NVIDIA_GUIDED_SAFETY_COST_NAMES:
        raise AssertionError(
            f"{stage} safety names {names} do not match "
            f"{NVIDIA_GUIDED_SAFETY_COST_NAMES}"
        )
    cost = np.asarray(info["safety_cost"])
    if cost.shape != (6,):
        raise AssertionError(f"{stage} safety cost shape is {cost.shape}, not (6,)")
    if cost.dtype != np.float32:
        raise AssertionError(
            f"{stage} safety cost dtype is {cost.dtype}, not numpy.float32"
        )
    if not np.all(np.isfinite(cost)) or np.any(cost < 0.0):
        raise AssertionError(f"{stage} safety cost is invalid: {cost}")
    return cost.copy()


def _finite_update_metrics(metrics: Mapping[str, float]) -> Dict[str, float]:
    missing = sorted(set(_FINITE_UPDATE_METRICS) - set(metrics))
    if missing:
        raise AssertionError(f"Agent update omitted metrics: {missing}")
    selected = {name: float(metrics[name]) for name in _FINITE_UPDATE_METRICS}
    for channel_name in NVIDIA_GUIDED_SAFETY_COST_NAMES:
        key = f"safety_channel_loss_{channel_name}"
        if key not in metrics:
            raise AssertionError(f"Agent update omitted metric {key!r}")
        selected[key] = float(metrics[key])
    nonfinite = {
        name: value for name, value in selected.items() if not np.isfinite(value)
    }
    if nonfinite:
        raise FloatingPointError(
            f"Agent update produced non-finite metrics: {nonfinite}"
        )
    return selected


def main() -> None:
    args = parse_args()
    _assert_sofa_not_imported("startup")

    device = _resolve_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(1)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = load_config(args.config)
    if config.get("backend") != "nvidia-guided":
        raise ValueError(
            "Smoke config backend must be 'nvidia-guided', got "
            f"{config.get('backend')!r}"
        )
    configured_names = tuple(config.get("safety_cost_names", ()))
    if configured_names != NVIDIA_GUIDED_SAFETY_COST_NAMES:
        raise ValueError(
            f"Smoke config safety schema is {configured_names}, expected "
            f"{NVIDIA_GUIDED_SAFETY_COST_NAMES}"
        )
    if int(config.get("safety_dim", -1)) != 6:
        raise ValueError("Smoke config safety_dim must be 6")

    smoke_config = copy.deepcopy(config)
    horizon = int(smoke_config["training"]["horizon"])
    if horizon != int(smoke_config["planning"]["horizon"]):
        raise ValueError("Training and planning horizons must match")
    if args.collection_steps < horizon:
        raise ValueError(
            f"collection-steps must be at least horizon={horizon}, "
            f"got {args.collection_steps}"
        )

    # Preserve the committed NVIDIA model/training hyperparameters while
    # shrinking only the replay/update batch needed for this one-step smoke.
    smoke_batch_size = 2
    smoke_config["training"]["batch_size"] = smoke_batch_size
    smoke_config["training"]["replay_capacity"] = max(
        16,
        args.collection_steps,
    )
    smoke_config["training"]["device"] = str(device)
    agent_config = build_agent_config(smoke_config)

    environment_config = dict(smoke_config["environment"])
    environment_config["workflow_root"] = args.workflow_root
    environment_config["ct_cache_path"] = args.ct_cache

    env = make_env(environment_config, backend="nvidia-guided")
    try:
        _assert_sofa_not_imported("after environment construction")
        if env.observation_space.shape != (14,):
            raise AssertionError(
                f"Observation space is {env.observation_space.shape}, not (14,)"
            )
        if env.action_space.shape != (2,):
            raise AssertionError(
                f"Action space is {env.action_space.shape}, not (2,)"
            )
        if tuple(env.safety_cost_names) != NVIDIA_GUIDED_SAFETY_COST_NAMES:
            raise AssertionError("Environment safety schema is not the NVIDIA schema")

        observation_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        agent = TDMPC2Agent(
            observation_dim,
            action_dim,
            agent_config,
            episode_length=int(environment_config["max_episode_steps"]),
            device=device,
        )
        replay = EpisodeReplayBuffer(
            int(smoke_config["training"]["replay_capacity"]),
            observation_dim,
            action_dim,
            horizon,
            smoke_batch_size,
            safety_cost_names=NVIDIA_GUIDED_SAFETY_COST_NAMES,
            seed=args.seed,
        )

        observation, reset_info = env.reset(seed=args.seed)
        if observation.shape != (14,) or observation.dtype != np.float32:
            raise AssertionError(
                "Reset observation must have shape (14,) and dtype float32"
            )
        if not np.all(np.isfinite(observation)):
            raise FloatingPointError("Reset observation is non-finite")
        _safety_cost(reset_info, stage="reset")
        if bool(reset_info["hard_safety_violation"]):
            raise AssertionError("Guided reset unexpectedly has a hard violation")

        observations = [observation.copy()]
        actions = []
        rewards = []
        terminated_flags = []
        safety_costs = []
        safe_action = np.array([1.0, 0.0], dtype=np.float32)
        if safe_action.shape != (2,) or not env.action_space.contains(safe_action):
            raise AssertionError("Safe guided action is not a valid (2,) action")

        for step_index in range(args.collection_steps):
            next_observation, reward, terminated, truncated, info = env.step(
                safe_action
            )
            if bool(info["hard_safety_violation"]):
                raise AssertionError(
                    f"Safe guided rollout hit a hard violation at step {step_index + 1}"
                )
            cost = _safety_cost(info, stage=f"step {step_index + 1}")
            if next_observation.shape != (14,) or next_observation.dtype != np.float32:
                raise AssertionError("Step observation shape/dtype changed")
            if not np.isfinite(float(reward)):
                raise FloatingPointError("Environment reward is non-finite")

            actions.append(safe_action.copy())
            rewards.append(float(reward))
            terminated_flags.append(bool(terminated))
            safety_costs.append(cost)
            observations.append(next_observation.copy())

            if terminated or truncated:
                break

        if len(actions) < horizon:
            raise AssertionError(
                "Safe guided rollout ended before one replay horizon: "
                f"collected {len(actions)}, need {horizon}"
            )

        replay.add_episode(
            observations,
            actions,
            rewards,
            terminated_flags,
            safety_cost=safety_costs,
        )
        if not replay.can_sample():
            raise AssertionError("Replay cannot sample the collected guided episode")
        sampled_batch = replay.sample_diagnostics(device, seed=args.seed)
        sampled_observations, sampled_actions, _, _, sampled_safety = sampled_batch
        expected_observation_shape = (
            horizon + 1,
            smoke_batch_size,
            observation_dim,
        )
        expected_action_shape = (horizon, smoke_batch_size, action_dim)
        expected_safety_shape = (horizon, smoke_batch_size, 6)
        if tuple(sampled_observations.shape) != expected_observation_shape:
            raise AssertionError(
                f"Replay observations are {tuple(sampled_observations.shape)}, "
                f"expected {expected_observation_shape}"
            )
        if tuple(sampled_actions.shape) != expected_action_shape:
            raise AssertionError(
                f"Replay actions are {tuple(sampled_actions.shape)}, "
                f"expected {expected_action_shape}"
            )
        if tuple(sampled_safety.shape) != expected_safety_shape:
            raise AssertionError(
                f"Replay safety is {tuple(sampled_safety.shape)}, "
                f"expected {expected_safety_shape}"
            )
        if sampled_safety.dtype != torch.float32:
            raise AssertionError("Replay safety tensor must be torch.float32")
        if not bool(torch.isfinite(sampled_safety).all()) or bool(
            torch.any(sampled_safety < 0.0)
        ):
            raise AssertionError("Replay safety tensor is invalid")

        if agent.update_count != 0:
            raise AssertionError("Fresh smoke agent update_count is not zero")
        update_metrics = agent.update(replay)
        if agent.update_count != 1:
            raise AssertionError(
                "Smoke test must execute exactly one agent update; "
                f"update_count is {agent.update_count}"
            )
        finite_metrics = _finite_update_metrics(update_metrics)
        _assert_sofa_not_imported("after replay update")

        summary = {
            "observation_shape": list(observation.shape),
            "action_shape": list(safe_action.shape),
            "safety_cost_shape": list(safety_costs[0].shape),
            "safety_cost_names": list(NVIDIA_GUIDED_SAFETY_COST_NAMES),
            "collected_steps": len(actions),
            "hard_violation": False,
            "replay_safety_shape": list(sampled_safety.shape),
            "replay_safety_dim": int(sampled_safety.shape[-1]),
            "agent_device": str(device),
            "agent_updates": agent.update_count,
            "finite_update_metrics": finite_metrics,
            "sofa_imported": False,
        }
        print("NVIDIA GUIDED TD-MPC2 SMOKE TEST")
        print(json.dumps(summary, indent=2, sort_keys=True))
        print("NVIDIA GUIDED TD-MPC2 SMOKE TEST PASSED")
    finally:
        env.close()


if __name__ == "__main__":
    main()
