"""Frozen Generic 280k navigation with optional passive 3 mm force monitoring.

One scene is reused across ordered seeds. Every episode is checked against the
frozen G1B trajectory hash before being saved. Force never enters policy input.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
RL_ROOT = HERE.parents[1]
REPO = RL_ROOT.parent
sys.path.insert(0, str(RL_ROOT))
sys.path.insert(0, str(REPO))

from envs.steve_env import make_steve_env  # noqa: E402
from tdmpc2.agent import TDMPC2Agent  # noqa: E402
from tdmpc2.common import load_torch_checkpoint, set_seed  # noqa: E402
from train import build_agent_config  # noqa: E402
from substep_monitor import DistalForceMonitor  # noqa: E402


ROOT = RL_ROOT / "logs/generic_solver_phase_g2_force_baseline"
G1B = RL_ROOT / "logs/generic_solver_phase_g1b_checkpoint_sweep"
CONFIG = RL_ROOT / "runs/generic_solver_tdmpc2_baseline_300k_20261006/config.yaml"
ADAPTER_SHA = "f6e397c8f43cfe0d32d4d5e0b834ae6f8f0ebb7119a2f3f5756ef47aa5e27b67"
EPISODE_FIELDS = (
    "checkpoint_step", "checkpoint_sha256", "seed", "episode_index", "success",
    "episode_length", "episode_reward", "terminated", "truncated",
    "simulation_error", "trajectory_sha256", "episode_peak_force_N",
    "episode_mean_force_N", "episode_total_fti_Ns", "contact_steps",
    "contact_fraction", "longest_contact_run",
)
STEP_FIELDS = (
    "seed", "episode_index", "step_index", "decision_observation_json",
    "selected_action_json", "applied_action_json", "next_observation_json",
    "reward", "terminated", "truncated", "success_step", "episode_success",
    "tip_position_mm_json", "insertion_length_mm",
    "distal_contact_force_max_N", "distal_contact_force_mean_N",
    "distal_contact_force_final_N", "distal_force_time_integral_Ns",
    "distal_contact_count", "distal_contact_substeps",
    "force_vector_at_max_N_json", "sofa_substeps", "max_constraint_rows",
    "max_contact_groups",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compact_array(value) -> str:
    return json.dumps(np.asarray(value, dtype=np.float32).tolist(),
                      separators=(",", ":"), allow_nan=False)


def check_scene(env) -> None:
    sim = env._simulation
    solver = sim.root.getObject("Generic")
    contact = sim.root.getObject("DefaultContactManager")
    if solver is None or contact is None:
        raise RuntimeError("Frozen Generic scene missing")
    actual = (
        solver.getClassName(), str(solver.findData("resolutionMethod").value),
        float(solver.findData("tolerance").value),
        int(solver.findData("maxIterations").value),
        bool(solver.findData("computeConstraintForces").value),
        str(contact.findData("response").value),
        str(contact.findData("responseParams").value), float(sim.root.dt.value),
    )
    expected = ("GenericConstraintSolver", "ProjectedGaussSeidel", 1e-4,
                2000, True, "FrictionContactConstraint", "mu=0.001", 0.006)
    if actual != expected:
        raise RuntimeError(f"Frozen Generic scene changed: {actual!r}")


def reference_rows(path: Path, seed_start: int, episodes: int, checkpoint_sha: str):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if [int(row["seed"]) for row in rows] != list(range(90000, 90200)):
        raise RuntimeError("Frozen G1B reference seed order changed")
    if any(row["checkpoint_sha256"] != checkpoint_sha or
           int(row["checkpoint_step"]) != 280000 for row in rows):
        raise RuntimeError("Frozen G1B reference checkpoint changed")
    return rows[seed_start-90000:seed_start-90000+episodes]


def evaluate_one(env, agent, seed: int, episode_index: int, checkpoint_sha: str,
                 monitor: DistalForceMonitor | None):
    set_seed(seed)
    env.action_space.seed(seed)
    observation, _ = env.reset(seed=seed)
    if observation.shape != (14,) or env.action_space.shape != (2,):
        raise RuntimeError("Frozen task observation/action shape changed")
    check_scene(env)
    if monitor is None:
        active_monitor = None
    else:
        # Add to the already constructed scene after its first historical reset.
        if monitor.env is not env:
            raise RuntimeError("Monitor attached to another scene")
        if not hasattr(monitor, "_g2_added"):
            env._simulation.root.addObject(monitor)
            monitor._g2_added = True
        active_monitor = monitor
    digest = hashlib.sha256()
    digest.update(np.asarray(observation, dtype=np.float32).tobytes())
    step_rows = []
    terminated = truncated = success = simulation_error = False
    total_reward = 0.0
    while not (terminated or truncated):
        index = len(step_rows)
        state = observation
        action = agent.act(state, first_step=index == 0, eval_mode=True)
        if active_monitor is not None:
            active_monitor.begin_step()
        observation, reward, terminated, truncated, info = env.step(action)
        force = active_monitor.finish_step() if active_monitor is not None else None
        total_reward += float(reward)
        success = success or bool(info.get("is_success", False))
        simulation_error = simulation_error or bool(info.get("simulation_error", False))
        digest.update(np.asarray(action, dtype=np.float32).tobytes())
        digest.update(np.asarray(observation, dtype=np.float32).tobytes())
        digest.update(np.asarray([reward], dtype=np.float64).tobytes())
        digest.update(bytes((bool(terminated), bool(truncated))))
        sim = env._simulation
        row = {
            "seed": seed, "episode_index": episode_index, "step_index": index,
            "decision_observation_json": compact_array(state),
            "selected_action_json": compact_array(action),
            "applied_action_json": compact_array(info["raw_action"]),
            "next_observation_json": compact_array(observation),
            "reward": float(reward), "terminated": bool(terminated),
            "truncated": bool(truncated),
            "success_step": bool(info.get("is_success", False)),
            "episode_success": False,  # filled after episode end
            "tip_position_mm_json": compact_array(sim.dof_positions[0]),
            "insertion_length_mm": float(sim.inserted_lengths[0]),
            "distal_contact_force_max_N": force.maximum_N if force else "",
            "distal_contact_force_mean_N": force.mean_N if force else "",
            "distal_contact_force_final_N": force.final_N if force else "",
            "distal_force_time_integral_Ns": force.integral_Ns if force else "",
            "distal_contact_count": force.contact_count if force else "",
            "distal_contact_substeps": force.contact_substeps if force else "",
            "force_vector_at_max_N_json": compact_array(force.max_vector_N) if force else "",
            "sofa_substeps": force.substep_count if force else "",
            "max_constraint_rows": force.max_constraint_rows if force else "",
            "max_contact_groups": force.max_contact_groups if force else "",
        }
        step_rows.append(row)
    if not np.isfinite(total_reward):
        raise FloatingPointError("Nonfinite episode reward")
    for row in step_rows:
        row["episode_success"] = bool(success)
    if active_monitor is not None:
        forces = np.asarray([float(row["distal_contact_force_max_N"])
                             for row in step_rows], dtype=float)
        contact = [int(row["distal_contact_count"]) > 0 for row in step_rows]
        run = longest = 0
        for flag in contact:
            run = run+1 if flag else 0
            longest = max(longest, run)
        peak = float(np.max(forces))
        mean = float(np.mean(forces))
        fti = float(sum(float(row["distal_force_time_integral_Ns"])
                        for row in step_rows))
        contact_steps = sum(contact)
        contact_fraction = contact_steps/len(step_rows)
    else:
        peak = mean = fti = contact_steps = contact_fraction = longest = ""
    episode = {
        "checkpoint_step": 280000, "checkpoint_sha256": checkpoint_sha,
        "seed": seed, "episode_index": episode_index, "success": bool(success),
        "episode_length": len(step_rows), "episode_reward": total_reward,
        "terminated": bool(terminated), "truncated": bool(truncated),
        "simulation_error": bool(simulation_error),
        "trajectory_sha256": digest.hexdigest(),
        "episode_peak_force_N": peak, "episode_mean_force_N": mean,
        "episode_total_fti_Ns": fti, "contact_steps": contact_steps,
        "contact_fraction": contact_fraction, "longest_contact_run": longest,
    }
    return episode, step_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--force", choices=("off", "on"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.seed_start < 90000 or args.seed_start+args.episodes > 90200 or args.episodes <= 0:
        raise ValueError("Only the frozen 90000-90199 confirmation seed range is permitted")
    selected = json.loads((G1B / "selected_checkpoint.json").read_text())
    checkpoint_path = Path(selected["path"])
    checkpoint_sha = sha256(checkpoint_path)
    if selected["step"] != 280000 or checkpoint_sha != selected["sha256"]:
        raise RuntimeError("Frozen 280k checkpoint SHA mismatch")
    if sha256(REPO / "eve/intervention/simulation/sofabeamadapter.py") != ADAPTER_SHA:
        raise RuntimeError("Frozen Generic adapter changed")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or \
            "RTX 5090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("G2 requires CUDA_VISIBLE_DEVICES=0 RTX 5090")
    checkpoint = load_torch_checkpoint(checkpoint_path, map_location="cpu")
    if (checkpoint.get("format_version") != 1 or
            checkpoint.get("algorithm") != "TD-MPC2" or
            int(checkpoint["total_env_steps"]) != 280000):
        raise RuntimeError("Frozen checkpoint format/step changed")
    config = checkpoint["config"]
    with CONFIG.open(encoding="utf-8") as stream:
        if config != yaml.safe_load(stream):
            raise RuntimeError("Frozen G1 config changed")
    reference = reference_rows(G1B / "confirmation_200_episode_results.csv",
                               args.seed_start, args.episodes, checkpoint_sha)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    episode_path = out / "episode_level_force.csv"
    step_path = out / "step_level_force.csv"
    if episode_path.exists() or step_path.exists():
        raise FileExistsError("Output already exists; rerun from seed 90000 in a new directory")
    runtime = {
        "mode": args.force, "seed_start": args.seed_start, "episodes": args.episodes,
        "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": checkpoint_sha,
        "python_executable": sys.executable, "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "visible_cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name_0": torch.cuda.get_device_name(0),
        "physical_cuda_device_index": 0, "visible_cuda_device_index": 0,
        "scene_lifecycle": "one scene, ordered reused-scene resets",
    }
    (out / "runtime.json").write_text(json.dumps(runtime, indent=2, sort_keys=True)+"\n")
    print(json.dumps(runtime, sort_keys=True), flush=True)
    set_seed(args.seed_start)
    env = make_steve_env(config["environment"])
    agent = TDMPC2Agent(14, 2, build_agent_config(config),
                        episode_length=int(config["environment"]["max_episode_steps"]),
                        device=torch.device("cuda:0"))
    agent.load_state_dict(checkpoint["agent"], load_optimizers=False)
    monitor = DistalForceMonitor(env) if args.force == "on" else None
    try:
        with episode_path.open("w", newline="", encoding="utf-8") as episode_stream, \
                step_path.open("w", newline="", encoding="utf-8") as step_stream:
            ew = csv.DictWriter(episode_stream, fieldnames=EPISODE_FIELDS)
            sw = csv.DictWriter(step_stream, fieldnames=STEP_FIELDS)
            ew.writeheader()
            sw.writeheader()
            episode_stream.flush()
            step_stream.flush()
            for episode_index in range(args.episodes):
                seed = args.seed_start + episode_index
                episode, steps = evaluate_one(env, agent, seed, episode_index,
                                              checkpoint_sha, monitor)
                expected = reference[episode_index]
                if (episode["success"] != (expected["success"] == "True") or
                        episode["episode_length"] != int(expected["episode_length"]) or
                        abs(episode["episode_reward"]-float(expected["episode_reward"])) > 1e-9 or
                        episode["trajectory_sha256"] != expected["trajectory_sha256"]):
                    raise RuntimeError(f"Frozen G1B trajectory regression at seed {seed}")
                if episode["simulation_error"]:
                    raise RuntimeError(f"Simulation error at seed {seed}")
                sw.writerows(steps)
                ew.writerow(episode)
                step_stream.flush()
                episode_stream.flush()
                print(f"seed={seed} force={args.force} success={int(episode['success'])} "
                      f"length={episode['episode_length']} reward={episode['episode_reward']:.9f}",
                      flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
