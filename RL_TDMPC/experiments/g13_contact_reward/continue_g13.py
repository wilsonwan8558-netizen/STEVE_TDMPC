"""Opt-in G13 B0 continuation scaffold with episode-boundary scene replay.

The default --audit path never interacts with SOFA or updates a model.
Training requires a separately frozen and enabled B0 pilot config plus
--execute. B1 is intentionally non-executable in F2H.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
ROOT = RL.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RL))
sys.path.insert(0, str(RL / "experiments/world_model_g13"))
from tdmpc2.common import (atomic_torch_save, capture_rng_state,
                           load_torch_checkpoint, restore_rng_state, set_seed)  # noqa: E402
from train import build_agent_config  # noqa: E402
from g13_model import G13Agent, G13ReplayBuffer  # noqa: E402

EXPECTED_SOURCE_SHA = {
    "experiments/world_model_g13/g13_config.yaml": "806b339f71254aa0cc5fc994c6e59054e449388f043e160eb24760bf382af48a",
    "experiments/world_model_g13/g13_env.py": "0ce85a157ee985d3b72cb4b3bd90e3ce27006514cd06f10e5ad45ec12ea6ccee",
    "experiments/world_model_g13/g13_model.py": "d2584083c84d40d7b39737d8f3ed415ca4d44fed5d55ba5f70e0c3a92cc3b3b5",
    "envs/geometry_features.py": "b0082e44da810287d412b3935b60eebb9ea58cae398e98d00e97e1f4e5452b46",
    "experiments/generic_solver_g2/substep_monitor.py": "744759896f36d284c9ba039a537a9f5334a5e6f5d99752f036062ae55ced3534",
    "experiments/generic_solver_g2/force_reconstruction.py": "76c920ce5122cb8da97ba100851985267e734b2877fc41ab6726217cdb13fa26",
    "tdmpc2/agent.py": "b7ead501ad7607f2ce1b1a3a46c23d2f3e16827302a3d9f8c8862900e6b27ef5",
    "tdmpc2/networks.py": "4c03b5f414f015bdc406885d98f220842d9a80e6933a2dafaebdf5b81cdefb19",
    "tdmpc2/replay_buffer.py": "460e70efb68de8d917928587fd732d3101df4f98b408e498ec29904e009921b8",
    "tdmpc2/common.py": "95163d00ea79f9c544ac6c8c6d59b9bb6c7a9d5be6581b9ff0a704327d0d7153",
    "train.py": "fa51042ebb772dabd09d9f2157dd965f5170c71ead1d0b837c63135167a415f4",
    "envs/steve_env.py": "ee340640bcc546b17a5effd6caf01f0d80a435c1f1e984a8a211d98395b24cfe",
    "../eve/intervention/simulation/sofabeamadapter.py": "f6e397c8f43cfe0d32d4d5e0b834ae6f8f0ebb7119a2f3f5756ef47aa5e27b67",
}
SOURCE_CHECKPOINT_SHA = "b2bba8be2e2722886c20952b15712deaa31ab612825999f1b3469a34fd6ef7be"
FORMAT = "F2H-G13-episode-boundary-v1"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def check_cuda() -> dict:
    if (os.getenv("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available() or
            torch.cuda.device_count() != 1 or "RTX 5090" not in torch.cuda.get_device_name(0)):
        raise RuntimeError("G13 continuation requires CUDA_VISIBLE_DEVICES=0 RTX 5090; no CPU fallback")
    return {"python": sys.executable, "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}


def validate_config(config: dict, *, execution: bool) -> None:
    if (config.get("source_checkpoint_sha256") != SOURCE_CHECKPOINT_SHA or
            config.get("auxiliary_force_head_coefficient") != 0.5 or
            config.get("updates_per_environment_step") != 1):
        raise RuntimeError("G13 foundation checkpoint/auxiliary-loss/update semantics drift")
    arm = config.get("arm")
    if arm not in ("B0", "B1"):
        raise RuntimeError("Unknown G13 experimental arm")
    if arm == "B0" and (config.get("reward_mode") != "original_task_only" or
                        config.get("contact_penalty_beta") != 0.0):
        raise RuntimeError("B0 must use the raw G13 task reward with no contact penalty")
    if arm == "B1" and (config.get("reward_mode") != "task_minus_fixed_normalized_contact_exposure" or
                        config.get("contact_exposure_metric") != "distal_force_time_integral_Ns"):
        raise RuntimeError("B1 future configuration is not the G2 contact-exposure definition")
    if execution:
        if arm != "B0":
            raise RuntimeError("F2H does not implement or activate B1 training")
        if config.get("status") != "FROZEN_FOR_SEPARATELY_AUTHORIZED_PILOT" or config.get("enabled") is not True:
            raise RuntimeError("Template cannot train: freeze and enable a separately authorized pilot protocol")
        if not isinstance(config.get("training_seed"), int) or not isinstance(config.get("max_additional_environment_steps"), int):
            raise RuntimeError("Training seed and bounded step budget must be fixed before execution")
        if not 0 < config["max_additional_environment_steps"] <= 10000:
            raise RuntimeError("G13 pilot runner permits only a bounded <=10000-step study")
        milestones = config.get("eval_snapshot_steps")
        if not isinstance(milestones, list) or any(not isinstance(x, int) or x <= 0 or x > config["max_additional_environment_steps"] for x in milestones):
            raise RuntimeError("Invalid fixed evaluation snapshot milestones")
        if sorted(set(milestones)) != milestones:
            raise RuntimeError("Evaluation snapshot milestones must be sorted and unique")


def load_foundation(path: Path, expected_sha: str):
    if expected_sha != SOURCE_CHECKPOINT_SHA or sha(path) != expected_sha:
        raise RuntimeError("Historical G13 checkpoint SHA256 mismatch")
    frozen = json.loads((RL / "logs/world_model_force_aware_planning_g13/g13_training_config.json").read_text())
    for relative, digest in EXPECTED_SOURCE_SHA.items():
        if sha(RL / relative) != digest:
            raise RuntimeError(f"Frozen G13/Generic source drift: {relative}")
    source = load_torch_checkpoint(path, map_location="cpu")
    if (source.get("format_version") != 2 or source.get("algorithm") != "Geometry-Force-TD-MPC2" or
            source.get("observation_dim") != 22 or source.get("total_env_steps") != 260000 or
            source.get("lambda_force_train") != 0.5 or source.get("config") != frozen["config"] or
            source["agent"]["action_dim"] != 2 or source["agent"]["update_count"] != 260000):
        raise RuntimeError("Historical G13 checkpoint structure or original training config mismatch")
    return source


def build_agent(source):
    c = source["config"]
    agent = G13Agent(22, 2, build_agent_config(c),
                     episode_length=int(c["environment"]["max_episode_steps"]),
                     device=torch.device("cuda:0"))
    agent.load_state_dict(source["agent"], load_optimizers=True)
    check_optimizer_devices(agent)
    return agent


def check_optimizer_devices(agent) -> None:
    for name, optimizer in (("model", agent.model_optimizer), ("policy", agent.policy_optimizer)):
        if not optimizer.state:
            raise RuntimeError(f"Missing restored {name} Adam state")
        for parameter, state in optimizer.state.items():
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state and state[key].device != parameter.device:
                    raise RuntimeError(f"Restored {name} Adam {key} is on wrong device")


def build_replay(source, seed: int):
    c = source["config"]["training"]
    replay = G13ReplayBuffer(int(c["replay_capacity"]), 22, 2, int(c["horizon"]),
                             int(c["batch_size"]), seed=seed)
    replay.load_state_dict(source["replay"])
    if len(replay) != 259987 or replay.num_episodes != 3800 or replay.num_sequences != 252387:
        raise RuntimeError("Historical G13 replay coverage changed")
    if any("forces" not in episode or len(episode["forces"]) != len(episode["actions"])
           for episode in replay._episodes):
        raise RuntimeError("Historical G13 replay force targets misaligned")
    return replay


def audit(config_path: Path, checkpoint_path: Path, output: Path | None = None) -> dict:
    config = json.loads(config_path.read_text())
    validate_config(config, execution=False)
    runtime = check_cuda()
    source = load_foundation(checkpoint_path, config["source_checkpoint_sha256"])
    set_seed(741912)
    agent = build_agent(source)
    replay = build_replay(source, seed=741912)
    result = {"status": "PASS", "mode": "AUDIT_NO_TRAINING", "arm": config["arm"],
              "configuration_status": config["status"], "execution_enabled": config.get("enabled") is True,
              "source_checkpoint": str(checkpoint_path.resolve()),
              "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA,
              "checkpoint_config_matches_archived_training_freeze": True,
              "observation_dim": 22, "action_dim": 2,
              "original_auxiliary_force_loss_coefficient": agent.lambda_force_train,
              "original_task_only_planning": True,
              "optimizer_model_lr_groups": [float(g["lr"]) for g in agent.model_optimizer.param_groups],
              "optimizer_policy_lr_groups": [float(g["lr"]) for g in agent.policy_optimizer.param_groups],
              "model_optimizer_state_entries": len(agent.model_optimizer.state),
              "policy_optimizer_state_entries": len(agent.policy_optimizer.state),
              "update_count": agent.update_count,
              "replay": {"transitions": len(replay), "episodes": replay.num_episodes,
                         "eligible_H3_sequences": replay.num_sequences,
                         "stored_episode_fields": sorted(replay._episodes[0].keys()),
                         "contact_FTI_labels_available": False,
                         "sampler_rng_restored_from_checkpoint": True},
              "source_has_global_rng_state": all(k in source["rng_state"] for k in ("python", "numpy", "torch", "cuda")),
              "mid_episode_original_scene_available": False,
              "B1_training_supported_in_F2H": False,
              "runtime": runtime}
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, result)
    return result


def read_step(env, action) -> tuple[np.ndarray, float, bool, bool, dict]:
    next_observation, task_reward, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
    trace = np.asarray([sample.magnitude_N for sample, _, _ in env.monitor.samples], dtype=float)
    if (next_observation.shape != (22,) or not np.all(np.isfinite(next_observation)) or
            not np.isfinite(task_reward) or trace.shape != (22,) or
            not np.all(np.isfinite(trace)) or np.any(trace < 0) or
            info.get("sofa_substeps") != 22 or info.get("simulation_error")):
        raise RuntimeError("Invalid G13 observation, task reward, force telemetry or SOFA step")
    fti = float(np.sum(trace) * 0.006)
    if (not np.isclose(fti, info["distal_force_time_integral_Ns"], rtol=0, atol=1e-9) or
            not np.isclose(float(np.max(trace)), info["distal_contact_force_max_N"], rtol=0, atol=1e-9)):
        raise RuntimeError("G13 force telemetry does not align with completed SOFA substeps")
    row = {"action": np.asarray(action, dtype=float).tolist(),
           "observation_after": next_observation.tolist(),
           "task_reward": float(task_reward), "training_reward": float(task_reward),
           "force_peak_nominal_N": float(np.max(trace)), "contact_exposure_nominal_Ns": fti,
           "terminated": bool(terminated), "truncated": bool(truncated),
           "success": bool(info.get("is_success", False))}
    return next_observation, float(task_reward), bool(terminated), bool(truncated), row


def restore_scene(env, history: list[dict]) -> int:
    """Replay accepted actions; verify observable state, not hidden solver internals."""
    replayed = 0
    for episode in history:
        seed = int(episode["seed"])
        env.action_space.seed(seed)
        obs, _ = env.reset(seed=seed)
        if not np.allclose(obs, episode["initial_observation"], rtol=0, atol=1e-6):
            raise RuntimeError(f"Scene replay initial observation mismatch: {seed}")
        target = env.unwrapped.intervention.target.coordinates3d
        if not np.allclose(target, episode["target_tracking3d_mm"], rtol=0, atol=1e-6):
            raise RuntimeError(f"Scene replay target mismatch: {seed}")
        for old in episode["steps"]:
            obs, _, terminated, truncated, new = read_step(env, old["action"])
            for key in ("observation_after", "action"):
                if not np.allclose(new[key], old[key], rtol=0, atol=1e-6):
                    raise RuntimeError(f"Scene replay {key} mismatch: {seed}/{replayed}")
            for key in ("task_reward", "force_peak_nominal_N", "contact_exposure_nominal_Ns"):
                if not np.isclose(new[key], old[key], rtol=0, atol=1e-6):
                    raise RuntimeError(f"Scene replay {key} mismatch: {seed}/{replayed}")
            if terminated != old["terminated"] or (truncated != old["truncated"] and not old.get("budget_truncated", False)):
                raise RuntimeError(f"Scene replay termination mismatch: {seed}/{replayed}")
            replayed += 1
    return replayed


def verify_restored_rng(saved: dict) -> None:
    import random
    current_numpy = np.random.get_state()
    old_numpy = saved["numpy"]
    numpy_equal = (current_numpy[0] == old_numpy[0] and
                   np.array_equal(current_numpy[1], old_numpy[1]) and
                   current_numpy[2:] == old_numpy[2:])
    if (random.getstate() != saved["python"] or
            not numpy_equal or
            not torch.equal(torch.get_rng_state(), saved["torch"]) or
            len(saved.get("cuda", [])) != torch.cuda.device_count() or
            any(not torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), saved["cuda"]))):
        raise RuntimeError("Python/NumPy/PyTorch/CUDA RNG restoration failed")


def run(config_path: Path, checkpoint_path: Path, run_dir: Path,
        *, resume: bool, max_new_episodes: int | None) -> None:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text())
    validate_config(config, execution=True)
    check_cuda()
    gate = json.loads((RL / "logs/contact_aware_joint_mppi/stage_f2h/baseline_equivalence.json").read_text())
    if gate.get("status") != "PASS":
        raise RuntimeError("F2H zero-update equivalence gate must pass before continuation")
    source = load_foundation(checkpoint_path, config["source_checkpoint_sha256"])
    from g13_env import make_g13_env  # SOFA is loaded only after all execution gates pass.
    seed = int(config["training_seed"])
    budget = int(config["max_additional_environment_steps"])
    run_dir = run_dir.resolve()
    latest = run_dir / "latest.pt"
    if resume:
        if not latest.is_file():
            raise FileNotFoundError(f"No episode-boundary checkpoint to resume: {latest}")
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty pilot run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    env = make_g13_env(source["config"]["environment"])
    agent = build_agent(source)
    replay = build_replay(source, seed)
    history: list[dict] = []
    total = successes = 0
    try:
        if resume:
            state = load_torch_checkpoint(latest, map_location="cpu")
            if (state.get("format") != FORMAT or state.get("source_checkpoint_sha256") != SOURCE_CHECKPOINT_SHA or
                    state.get("config_sha256") != sha(config_path) or state.get("seed") != seed or
                    state.get("arm") != "B0" or state.get("budget") != budget):
                raise RuntimeError("Resume checkpoint protocol/arm/seed/budget mismatch")
            agent.load_state_dict(state["agent"], load_optimizers=True)
            check_optimizer_devices(agent)
            replay.load_state_dict(state["replay"])
            history = state["scene_history"]
            total = int(state["additional_environment_steps"])
            successes = int(state["new_successes"])
            replayed = restore_scene(env, history)
            if (replayed != total or len(history) != int(state["new_episodes"]) or
                    len(replay) != 259987 + total):
                raise RuntimeError("Scene replay length does not match checkpoint boundary")
            restore_rng_state(state["rng_state"])
            verify_restored_rng(state["rng_state"])
            print(json.dumps({"resume_reconstructed_scene_steps_not_new_training_data": replayed,
                              "accepted_additional_steps": total}), flush=True)
        if total >= budget:
            raise RuntimeError("Pilot already exhausted its frozen step budget")
        if not resume:
            atomic_torch_save({"format": "F2H-G13-eval-only-v1", "agent": agent.state_dict(),
                               "config": source["config"], "additional_environment_steps": 0,
                               "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA}, run_dir / "eval_step_0.pt")
        new_episodes = 0
        while total < budget:
            episode_seed = seed + len(history)
            env.action_space.seed(episode_seed)
            obs, _ = env.reset(seed=episode_seed)
            if obs.shape != (22,):
                raise RuntimeError("G13 observation interface drift")
            episode = {"seed": episode_seed, "initial_observation": obs.tolist(),
                       "target_tracking3d_mm": env.unwrapped.intervention.target.coordinates3d.tolist(),
                       "steps": []}
            observations = [obs.copy()]
            actions, rewards, terminated_flags, forces = [], [], [], []
            done = success = False
            while not done and total < budget:
                action = agent.act(obs, first_step=not actions, eval_mode=False)
                next_obs, task_reward, terminated, truncated, row = read_step(env, action)
                total += 1
                if total == budget and not (terminated or truncated):
                    truncated = True
                    row["truncated"] = True
                    row["budget_truncated"] = True
                observations.append(next_obs.copy())
                actions.append(action.copy())
                rewards.append(task_reward)
                terminated_flags.append(bool(terminated))
                forces.append(row["force_peak_nominal_N"])
                episode["steps"].append(row)
                success = success or row["success"]
                obs = next_obs
                done = bool(terminated or truncated)
                if done:
                    replay.add_episode(observations, actions, rewards, terminated_flags, forces)
                if not replay.can_sample():
                    raise RuntimeError("Restored G13 replay unexpectedly cannot sample")
                loss = agent.update(replay)
                if not all(np.isfinite(float(v)) for v in loss.values()):
                    raise FloatingPointError("Nonfinite unchanged G13 update")
                if total in config["eval_snapshot_steps"]:
                    atomic_torch_save({"format": "F2H-G13-eval-only-v1", "agent": agent.state_dict(),
                                       "config": source["config"], "additional_environment_steps": total,
                                       "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA},
                                      run_dir / f"eval_step_{total}.pt")
                if total % 25 == 0:
                    print(json.dumps({"arm": "B0", "seed": seed, "additional_steps_attempted": total,
                                      "latest_episode_boundary_steps": total - len(actions),
                                      "episode_seed": episode_seed, "replay_eligible_sequences": replay.num_sequences,
                                      "update_count": agent.update_count}), flush=True)
            episode["success"] = success
            episode["length"] = len(actions)
            history.append(episode)
            successes += int(success)
            payload = {"format": FORMAT, "arm": "B0", "seed": seed, "budget": budget,
                       "config_sha256": sha(config_path), "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA,
                       "agent": agent.state_dict(), "replay": replay.state_dict(),
                       "rng_state": capture_rng_state(), "scene_history": history,
                       "additional_environment_steps": total, "new_episodes": len(history),
                       "new_successes": successes,
                       "resume_scope": "Observable scene-history replay from the new F2H branch start; not exact historical 260k mid-episode resume."}
            atomic_torch_save(payload, latest)
            write_json(run_dir / "progress.json", {"arm": "B0", "seed": seed,
                       "accepted_additional_steps": total, "budget": budget,
                       "new_episodes": len(history), "new_successes": successes,
                       "replay_eligible_sequences": replay.num_sequences,
                       "checkpoint": str(latest)})
            print(json.dumps({"completed_new_episode": len(history), "accepted_additional_steps": total,
                              "success": success, "episode_seed": episode_seed}), flush=True)
            new_episodes += 1
            if max_new_episodes is not None and new_episodes >= max_new_episodes:
                return
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--execute", action="store_true", help="Requires a frozen, enabled B0 pilot config")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-episodes", type=int)
    args = parser.parse_args()
    if not args.execute:
        if args.resume or args.run_dir or args.max_new_episodes is not None:
            raise RuntimeError("Execution-only flags supplied without --execute")
        print(json.dumps(audit(args.config, args.checkpoint, args.audit_output), sort_keys=True), flush=True)
        return
    if args.run_dir is None:
        raise RuntimeError("--run-dir is required for explicit training execution")
    if args.max_new_episodes is not None and args.max_new_episodes <= 0:
        raise ValueError("--max-new-episodes must be positive")
    run(args.config, args.checkpoint, args.run_dir, resume=args.resume,
        max_new_episodes=args.max_new_episodes)


if __name__ == "__main__":
    main()
