"""G1 integration smoke using the final Original TD-MPC2 hyperparameters.

This intentionally forces ONE optimizer update before the configured 5,000-step
seed period solely to exercise the update path. It never resumes into training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

RL_ROOT = Path(__file__).resolve().parents[2]
REPO = RL_ROOT.parent
sys.path.insert(0, str(RL_ROOT))
sys.path.insert(0, str(REPO))

from envs.steve_env import make_steve_env  # noqa: E402
from tdmpc2.agent import TDMPC2Agent  # noqa: E402
from tdmpc2.common import load_config, load_torch_checkpoint, select_device, set_seed  # noqa: E402
from tdmpc2.replay_buffer import EpisodeReplayBuffer  # noqa: E402
from train import build_agent_config, save_checkpoint  # noqa: E402

EXPECTED_ADAPTER_SHA256 = "f6e397c8f43cfe0d32d4d5e0b834ae6f8f0ebb7119a2f3f5756ef47aa5e27b67"


def assert_frozen_scene(env) -> dict:
    sim = env._simulation
    solver = sim.root.getObject("Generic")
    contact = sim.root.getObject("DefaultContactManager")
    if solver is None or solver.getClassName() != "GenericConstraintSolver":
        raise RuntimeError("Frozen Generic solver missing")
    if contact is None:
        raise RuntimeError("Frozen contact manager missing")
    fields = {
        "resolutionMethod": str(solver.findData("resolutionMethod").value),
        "tolerance": float(solver.findData("tolerance").value),
        "maxIterations": int(solver.findData("maxIterations").value),
        "computeConstraintForces": bool(solver.findData("computeConstraintForces").value),
        "response": str(contact.findData("response").value),
        "responseParams": str(contact.findData("responseParams").value),
        "dt_s": float(sim.root.dt.value),
    }
    expected = {
        "resolutionMethod": "ProjectedGaussSeidel", "tolerance": 1e-4,
        "maxIterations": 2000, "computeConstraintForces": True,
        "response": "FrictionContactConstraint", "responseParams": "mu=0.001",
        "dt_s": 0.006,
    }
    if fields != expected:
        raise RuntimeError(f"Frozen scene mismatch: {fields!r} != {expected!r}")
    return fields


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    adapter_hash = hashlib.sha256(
        (REPO / "eve/intervention/simulation/sofabeamadapter.py").read_bytes()
    ).hexdigest()
    if adapter_hash != EXPECTED_ADAPTER_SHA256:
        raise RuntimeError("G0/G0.5 production solver adapter SHA256 changed")
    training = config["training"]
    if training["seed"] != 1 or training["total_steps"] != 300000:
        raise RuntimeError("Unexpected G1 seed or total step budget")
    if training["device"] != "auto" or not torch.cuda.is_available():
        raise RuntimeError("Historical device=auto must resolve to CUDA for G1")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Smoke expects CUDA_VISIBLE_DEVICES=0 and one visible GPU")
    device = select_device(training["device"])
    if device.type != "cuda":
        raise RuntimeError("Refusing CPU smoke test")
    set_seed(int(training["seed"]))
    env = make_steve_env(config["environment"])
    try:
        observation_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        agent = TDMPC2Agent(
            observation_dim, action_dim, build_agent_config(config),
            episode_length=int(config["environment"]["max_episode_steps"]),
            device=device,
        )
        if agent.update_count != 0:
            raise RuntimeError("Agent did not initialize from scratch")
        replay = EpisodeReplayBuffer(
            int(training["replay_capacity"]), observation_dim, action_dim,
            int(training["horizon"]), int(training["batch_size"]),
            seed=int(training["seed"]),
        )
        env.action_space.seed(int(training["seed"]))
        observation, _ = env.reset(seed=int(training["seed"]))
        scene = assert_frozen_scene(env)
        observations = [observation.copy()]
        actions = []
        rewards = []
        terminated_flags = []
        terminated = truncated = False
        while not (terminated or truncated):
            action = env.action_space.sample().astype(np.float32)
            observation, reward, terminated, truncated, info = env.step(action)
            if info.get("simulation_error", False):
                raise RuntimeError("SOFA simulation error during smoke rollout")
            observations.append(observation.copy())
            actions.append(action.copy())
            rewards.append(float(reward))
            terminated_flags.append(bool(terminated))
        replay.add_episode(observations, actions, rewards, terminated_flags)
        if not replay.can_sample() or len(replay) != len(actions):
            raise RuntimeError("Replay failed to fill from the real environment")
        metrics = agent.update(replay)
        if agent.update_count != 1 or not all(np.isfinite(float(v)) for v in metrics.values()):
            raise RuntimeError(f"Nonfinite/absent optimizer update: {metrics}")
        planned_action = agent.act(observation, first_step=True, eval_mode=False)
        if planned_action.shape != (action_dim,) or not np.all(np.isfinite(planned_action)):
            raise RuntimeError("Training planner produced invalid action")
        checkpoint_path = output / "smoke_checkpoint.pt"
        save_checkpoint(
            path=checkpoint_path, config=config, agent=agent, replay=replay,
            total_env_steps=len(actions), episode_index=1,
            success_count=int(info.get("is_success", False)), include_replay=True,
        )
        saved = load_torch_checkpoint(checkpoint_path, map_location="cpu")
        if (saved["total_env_steps"] != len(actions) or
                saved["agent"]["update_count"] != 1 or
                saved["replay"]["size"] != len(actions)):
            raise RuntimeError("Smoke checkpoint did not round-trip")
        summary = {
            "status": "SMOKE_TRAINING_PASS",
            "note": "One diagnostic update was forced before the configured 5000 seed steps; the final 300k run starts fresh at step 0 with the unchanged schedule.",
            "config": str(args.config.resolve()),
            "adapter_sha256": adapter_hash,
            "python": sys.executable,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_visible_device_index": 0,
            "cuda_device_name": torch.cuda.get_device_name(0),
            "observation_dim": observation_dim,
            "action_dim": action_dim,
            "scene": scene,
            "real_environment_steps": len(actions),
            "replay_size": len(replay),
            "replay_sequences": replay.num_sequences,
            "optimizer_updates": agent.update_count,
            "finite_losses": True,
            "losses": {k: float(v) for k, v in metrics.items()},
            "planned_action_finite": True,
            "checkpoint": str(checkpoint_path),
            "checkpoint_round_trip": True,
            "safety_objective_active": False,
        }
        (output / "smoke_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
