"""Navigation-only, historical-lifecycle TD-MPC2 evaluator for G1B.

One checkpoint, one SOFA scene, ordered seeds. Replays saved prefixes normally
when --resume is requested; no episode is skipped with a new scene.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

RL_ROOT = Path(__file__).resolve().parents[2]
REPO = RL_ROOT.parent
sys.path.insert(0, str(RL_ROOT))
sys.path.insert(0, str(REPO))

from envs.steve_env import make_steve_env  # noqa: E402
from tdmpc2.agent import TDMPC2Agent  # noqa: E402
from tdmpc2.common import load_torch_checkpoint, set_seed  # noqa: E402
from train import build_agent_config  # noqa: E402

FROZEN_ADAPTER_SHA = "f6e397c8f43cfe0d32d4d5e0b834ae6f8f0ebb7119a2f3f5756ef47aa5e27b67"
G1_CONFIG = RL_ROOT / "runs/generic_solver_tdmpc2_baseline_300k_20261006/config.yaml"
FIELDS = (
    "checkpoint_step", "checkpoint_sha256", "seed", "success", "episode_length",
    "episode_reward", "terminated", "truncated", "termination_reason",
    "simulation_error", "trajectory_sha256",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_scene(env) -> None:
    sim = env._simulation
    solver = sim.root.getObject("Generic")
    contact = sim.root.getObject("DefaultContactManager")
    if solver is None or solver.getClassName() != "GenericConstraintSolver" or contact is None:
        raise RuntimeError("Generic constraint scene missing")
    actual = (
        str(solver.findData("resolutionMethod").value),
        float(solver.findData("tolerance").value),
        int(solver.findData("maxIterations").value),
        bool(solver.findData("computeConstraintForces").value),
        str(contact.findData("response").value),
        str(contact.findData("responseParams").value),
        float(sim.root.dt.value),
    )
    expected = ("ProjectedGaussSeidel", 1e-4, 2000, True,
                "FrictionContactConstraint", "mu=0.001", 0.006)
    if actual != expected:
        raise RuntimeError(f"G1B scene differs from G0/G0.5: {actual!r}")


def evaluate_one(env, agent: TDMPC2Agent, seed: int, step: int, checkpoint_sha: str) -> dict:
    # Same RNG reseed/order and reused-scene reset as the archived Original
    # evaluator; action_space.seed is explicit for the G1B protocol.
    set_seed(seed)
    env.action_space.seed(seed)
    observation, _ = env.reset(seed=seed)
    if observation.shape != (14,) or env.action_space.shape != (2,):
        raise RuntimeError("Historical Original task observation/action shape changed")
    digest = hashlib.sha256()
    digest.update(np.asarray(observation, dtype=np.float32).tobytes())
    terminated = truncated = False
    length = 0
    reward_sum = 0.0
    success = False
    simulation_error = False
    while not (terminated or truncated):
        action = agent.act(observation, first_step=length == 0, eval_mode=True)
        observation, reward, terminated, truncated, info = env.step(action)
        length += 1
        reward_sum += float(reward)
        success = success or bool(info.get("is_success", False))
        simulation_error = simulation_error or bool(info.get("simulation_error", False))
        digest.update(np.asarray(action, dtype=np.float32).tobytes())
        digest.update(np.asarray(observation, dtype=np.float32).tobytes())
        digest.update(np.asarray([reward], dtype=np.float64).tobytes())
        digest.update(bytes((bool(terminated), bool(truncated))))
    if not np.isfinite(reward_sum):
        raise FloatingPointError(f"Nonfinite reward on seed {seed}")
    reason = ("simulation_error" if simulation_error else "success" if success
              else "time_limit" if truncated else "other_terminal")
    return {
        "checkpoint_step": step, "checkpoint_sha256": checkpoint_sha,
        "seed": seed, "success": success, "episode_length": length,
        "episode_reward": reward_sum, "terminated": terminated,
        "truncated": truncated, "termination_reason": reason,
        "simulation_error": simulation_error, "trajectory_sha256": digest.hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    checkpoint_path = args.checkpoint.resolve()
    checkpoint_sha = sha256(checkpoint_path)
    if checkpoint_sha != args.expected_sha256:
        raise RuntimeError("Checkpoint SHA256 differs from locked manifest")
    if sha256(REPO / "eve/intervention/simulation/sofabeamadapter.py") != FROZEN_ADAPTER_SHA:
        raise RuntimeError("Frozen Generic adapter SHA256 changed")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("G1B requires CUDA_VISIBLE_DEVICES=0, one visible RTX 5090")
    if "RTX 5090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Unexpected CUDA device")
    device = torch.device("cuda:0")
    checkpoint = load_torch_checkpoint(checkpoint_path, map_location="cpu")
    step = int(checkpoint["total_env_steps"])
    if checkpoint.get("format_version") != 1 or checkpoint.get("algorithm") != "TD-MPC2":
        raise RuntimeError("Unexpected TD-MPC2 checkpoint format")
    if checkpoint_path.name != f"step_{step}.pt":
        raise RuntimeError("Checkpoint filename and internal step disagree")
    with G1_CONFIG.open(encoding="utf-8") as stream:
        expected_config = yaml.safe_load(stream)
    config = checkpoint["config"]
    if config != expected_config or config["training"]["seed"] != 1:
        raise RuntimeError("Checkpoint configuration differs from frozen G1 training config")
    existing = []
    output = args.output.resolve()
    if output.exists():
        if not args.resume:
            raise FileExistsError(f"Output exists; use --resume to replay its saved seed prefix: {output}")
        with output.open(newline="", encoding="utf-8") as stream:
            existing = list(csv.DictReader(stream))
        if [int(row["seed"]) for row in existing] != list(range(args.seed_start, args.seed_start+len(existing))):
            raise RuntimeError("Saved output is not a contiguous ordered prefix")
        if len(existing) > args.episodes:
            raise RuntimeError("Saved output exceeds requested episode count")
    output.parent.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed_start)
    env = make_steve_env(config["environment"])
    agent = TDMPC2Agent(14, 2, build_agent_config(config),
                        episode_length=int(config["environment"]["max_episode_steps"]),
                        device=device)
    agent.load_state_dict(checkpoint["agent"], load_optimizers=False)
    try:
        with output.open("a" if existing else "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(FIELDS))
            if not existing:
                writer.writeheader()
                stream.flush()
            for index in range(args.episodes):
                seed = args.seed_start + index
                row = evaluate_one(env, agent, seed, step, checkpoint_sha)
                if index == 0:
                    check_scene(env)
                if index < len(existing):
                    saved = existing[index]
                    if (saved["success"] != str(row["success"]) or
                            int(saved["episode_length"]) != row["episode_length"] or
                            abs(float(saved["episode_reward"])-row["episode_reward"]) > 1e-9 or
                            saved["trajectory_sha256"] != row["trajectory_sha256"]):
                        raise RuntimeError(f"Scene-history replay mismatch at seed {seed}")
                    print(f"replayed seed={seed} step={step} OK", flush=True)
                    continue
                writer.writerow(row)
                stream.flush()
                print(f"seed={seed} step={step} success={int(row['success'])} "
                      f"length={row['episode_length']} reward={row['episode_reward']:.9f}",
                      flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
