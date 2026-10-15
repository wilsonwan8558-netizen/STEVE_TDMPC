"""Frozen paired G12.1 closed-loop Generic-SOFA development, one arm per run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
sys.path.insert(0, str(RL))
sys.path.insert(0, str(RL.parent))
sys.path.insert(0, str(RL / "experiments/generic_solver_g2"))
from envs.steve_env import make_steve_env  # noqa: E402
from tdmpc2.common import load_torch_checkpoint, set_seed  # noqa: E402
from substep_monitor import DistalForceMonitor  # noqa: E402
from collect import check_scene  # noqa: E402
from g121_planner import OUT, CHECKPOINT, EXPECTED_CHECKPOINT_SHA, F_LIM, load_agent, sha  # noqa: E402

CONTROL_INTERVAL_S = 22 * 0.006


def save_json(path: Path, data: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def evaluate_one(env, monitor, agent, seed: int, weight: float) -> dict:
    set_seed(seed)
    env.action_space.seed(seed)
    observation, _ = env.reset(seed=seed)
    check_scene(env)
    if not hasattr(monitor, "_g121_added"):
        env._simulation.root.addObject(monitor)
        monitor._g121_added = True
    steps = []
    total_reward = 0.0
    success = False
    terminated = truncated = False
    while not (terminated or truncated):
        index = len(steps)
        if index >= env.max_episode_steps:
            raise RuntimeError("G12.1 episode exceeded frozen step limit")
        if weight == 0:
            action = agent.act(observation, first_step=index == 0, eval_mode=True)
            nominal = action.copy()
        else:
            mean_before = agent.previous_mean.detach().clone()
            cpu_rng = torch.get_rng_state().clone()
            cuda_rng = torch.cuda.get_rng_state().clone()
            agent.lambda_F = 0.0
            nominal = agent.act(observation, first_step=index == 0, eval_mode=True)
            agent.previous_mean.copy_(mean_before)
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng)
            agent.lambda_F = weight
            action = agent.act(observation, first_step=index == 0, eval_mode=True)
        monitor.begin_step()
        next_observation, reward, terminated, truncated, info = env.step(action)
        force = monitor.finish_step()
        if force.substep_count != 22 or bool(info.get("simulation_error", False)):
            raise RuntimeError(f"G12.1 simulation/force error {seed}/{index}")
        total_reward += float(reward)
        success = success or bool(info.get("is_success", False))
        difference = action - nominal
        steps.append({
            "seed": seed, "step_index": index,
            "selected_action": action.tolist(), "same_state_lambda_zero_action": nominal.tolist(),
            "applied_action": np.asarray(info["raw_action"], dtype=np.float32).reshape(-1).tolist(),
            "first_action_changed_vs_same_state_baseline": bool(not np.array_equal(action, nominal)),
            "first_action_L2_difference": float(np.linalg.norm(difference)),
            "translation_difference": float(difference[0]),
            "rotation_difference": float(difference[1]),
            "reward": float(reward), "terminated": bool(terminated), "truncated": bool(truncated),
            "success_step": bool(info.get("is_success", False)),
            "true_distal_force_max_N": force.maximum_N,
            "true_distal_force_mean_N": force.mean_N,
            "true_force_time_integral_Ns": force.integral_Ns,
            "true_above_F_lim": bool(force.maximum_N > F_LIM),
            "tip_position_mm": np.asarray(env._simulation.dof_positions[0], dtype=np.float32).tolist(),
            "insertion_length_mm": float(env._simulation.inserted_lengths[0]),
        })
        observation = next_observation
    if not np.isfinite(total_reward):
        raise FloatingPointError("G12.1 nonfinite episode reward")
    return {
        "seed": seed, "lambda_F": weight, "arm": "B0" if weight == 0 else "B1",
        "success": bool(success), "episode_length": len(steps), "episode_reward": total_reward,
        "time_to_target_s": len(steps)*CONTROL_INTERVAL_S if success else None,
        "terminated": bool(terminated), "truncated": bool(truncated),
        "simulation_error": False,
        "steps": steps,
    }


def main(arm_index: int) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or "RTX 5090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("G12.1 development requires CUDA_VISIBLE_DEVICES=0 RTX 5090")
    if sha(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("Frozen TD-MPC2 checkpoint changed")
    manifest = json.loads((OUT / "development_seed_manifest.json").read_text())
    audit = json.loads((OUT / "development_seed_audit.json").read_text())
    gate = json.loads((OUT / "plan_fork_gate_decision.json").read_text())
    scale = json.loads((OUT / "lambda_scale_definition.json").read_text())
    if (manifest["status"] != "FROZEN_BEFORE_G121_CLOSED_LOOP_DEVELOPMENT" or
            audit["status"] != "PASS_UNUSED_IN_SCANNED_ARTIFACTS" or
            gate["decision"] != "PASS_NARROWLY_FOR_FRESH_PAIRED_DEVELOPMENT_ONLY" or
            scale["H_force"] != 1):
        raise RuntimeError("G12.1 development gate/freeze failed")
    weights = scale["fixed_lambda_set"]
    if arm_index not in range(len(weights)):
        raise ValueError("G12.1 arm index outside frozen four-weight set")
    weight = float(weights[arm_index])
    seeds = manifest["ordered_seeds"]
    folder = OUT / "development_arms" / f"arm_{arm_index}"
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = load_torch_checkpoint(CHECKPOINT, map_location="cpu")
    config = checkpoint["config"]
    set_seed(seeds[0])
    env = make_steve_env(config["environment"])
    agent = load_agent(weight)
    monitor = DistalForceMonitor(env)
    try:
        for seed in seeds:
            episode = evaluate_one(env, monitor, agent, seed, weight)
            target = folder / f"seed_{seed}.json"
            if target.exists():
                prior = json.loads(target.read_text())
                if prior != episode:
                    raise RuntimeError(f"G12.1 replay-resume episode mismatch arm={arm_index} seed={seed}")
            else:
                save_json(target, episode)
            print("G121_DEVELOPMENT_EPISODE", arm_index, seed,
                  episode["episode_length"], episode["success"],
                  max(step["true_distal_force_max_N"] for step in episode["steps"]), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm-index", type=int, required=True)
    main(parser.parse_args().arm_index)
