"""Gate G3 on Generic full-process copy-on-write one-step fork equivalence."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import select
import signal
import sys
import traceback

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(RL / "experiments/generic_solver_g2"))
sys.path.insert(0, str(RL))
sys.path.insert(0, str(RL.parent))

from envs.steve_env import make_steve_env  # noqa: E402
from tdmpc2.agent import TDMPC2Agent  # noqa: E402
from tdmpc2.common import load_torch_checkpoint, set_seed  # noqa: E402
from train import build_agent_config  # noqa: E402
from substep_monitor import DistalForceMonitor  # noqa: E402
from collect import check_scene  # noqa: E402
from prepare import G1B, G2, OUT, CONFIG, EXPECTED_CHECKPOINT_SHA, EXPECTED_G2_SHA, sha, check_force_source  # noqa: E402

TOLERANCE = 1e-8
AUDIT_SEEDS = (90000, 90001, 90002, 90003)


def signature(env, monitor, action):
    monitor.begin_step()
    observation, reward, terminated, truncated, info = env.step(action)
    force = monitor.finish_step()
    return {"observation": np.asarray(observation, dtype=np.float32).tolist(),
            "reward": float(reward), "terminated": bool(terminated), "truncated": bool(truncated),
            "success": bool(info.get("is_success", False)), "force_next_max_N": force.maximum_N,
            # G2 serialized the tip with compact_array(), which casts to float32.
            "tip_position_mm": np.asarray(env._simulation.dof_positions[0], dtype=np.float32).tolist(),
            "insertion_length_mm": float(env._simulation.inserted_lengths[0]),
            "applied_action": np.asarray(info["raw_action"], dtype=np.float32).reshape(-1).tolist(),
            "substeps": force.substep_count}


def fork_step(env, monitor, action):
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        try:
            with open(os.devnull, "w") as null:
                os.dup2(null.fileno(), 1)
                os.dup2(null.fileno(), 2)
            payload = {"status": "PASS", "signature": signature(env, monitor, action.copy())}
        except BaseException:
            payload = {"status": "ERROR", "traceback": traceback.format_exc()}
        try:
            os.write(write_fd, json.dumps(payload).encode())
        finally:
            os._exit(0)
    os.close(write_fd)
    ready, _, _ = select.select([read_fd], [], [], 120)
    if not ready:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
        os.close(read_fd)
        raise RuntimeError("Generic child SOFA step timed out")
    payload_bytes = os.read(read_fd, 1_000_000)
    os.close(read_fd)
    _, wait_status = os.waitpid(pid, 0)
    if not payload_bytes:
        raise RuntimeError(f"Generic child returned no data; status={wait_status}")
    payload = json.loads(payload_bytes)
    if payload["status"] != "PASS":
        raise RuntimeError(payload["traceback"])
    return payload["signature"]


def compare(actual, expected, differences, location):
    for key in expected:
        a, b = actual[key], expected[key]
        if isinstance(b, bool) or key == "substeps":
            if a != b:
                raise RuntimeError(f"{location}: {key} mismatch {a} vs {b}")
        else:
            delta = float(np.max(np.abs(np.asarray(a, dtype=float)-np.asarray(b, dtype=float))))
            differences[key] = max(differences.get(key, 0.0), delta)
            if not np.isfinite(delta) or delta > TOLERANCE:
                raise RuntimeError(f"{location}: {key} mismatch {delta}")


def g2_signature(row):
    return {"observation": json.loads(row["next_observation_json"]), "reward": float(row["reward"]),
            "terminated": row["terminated"] == "True", "truncated": row["truncated"] == "True",
            "success": row["success_step"] == "True", "force_next_max_N": float(row["distal_contact_force_max_N"]),
            "tip_position_mm": json.loads(row["tip_position_mm_json"]),
            "insertion_length_mm": float(row["insertion_length_mm"]),
            "applied_action": json.loads(row["applied_action_json"]), "substeps": int(row["sofa_substeps"])}


def main():
    report_path = OUT / "counterfactual_restore_audit.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    report = {"status": "RUNNING", "method": "Linux fork copy-on-write of live Generic SOFA scene; child executes exactly one env.step and exits",
              "seeds": AUDIT_SEEDS, "states": [], "tolerance": TOLERANCE, "max_abs_difference": {},
              "g2_force_source_sha256": EXPECTED_G2_SHA}
    try:
        check_force_source()
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("CUDA_VISIBLE_DEVICES=0 is required")
        checkpoint_path = Path(json.loads((G1B / "selected_checkpoint.json").read_text())["path"])
        if sha(checkpoint_path) != EXPECTED_CHECKPOINT_SHA or sha(G2 / "step_level_force.csv") != EXPECTED_G2_SHA:
            raise RuntimeError("Checkpoint or G2 force dataset hash changed")
        with (G2 / "step_level_force.csv").open(newline="") as stream:
            rows_by_seed = {}
            for row in csv.DictReader(stream):
                seed = int(row["seed"])
                if seed in AUDIT_SEEDS:
                    rows_by_seed.setdefault(seed, []).append(row)
        with np.load(OUT / "behavior_features.npz") as features:
            feature_lookup = {(int(seed), int(decision)): i for i, (seed, decision) in enumerate(zip(features["seed"], features["decision_index"]))}
            cached_latents = features["latent"]
        checkpoint = load_torch_checkpoint(checkpoint_path, map_location="cpu")
        config = checkpoint["config"]
        if config != yaml.safe_load(CONFIG.read_text()):
            raise RuntimeError("Frozen config mismatch")
        set_seed(AUDIT_SEEDS[0])
        env = make_steve_env(config["environment"])
        agent = TDMPC2Agent(14, 2, build_agent_config(config), episode_length=int(config["environment"]["max_episode_steps"]), device=torch.device("cuda:0"))
        agent.load_state_dict(checkpoint["agent"], load_optimizers=False)
        monitor = DistalForceMonitor(env)
        try:
            for seed in AUDIT_SEEDS:
                set_seed(seed)
                env.action_space.seed(seed)
                observation, _ = env.reset(seed=seed)
                check_scene(env)
                if not hasattr(monitor, "_g3_added"):
                    env._simulation.root.addObject(monitor)
                    monitor._g3_added = True
                rows = rows_by_seed[seed]
                selected = {5, min(15, len(rows)-3), min(len(rows)//2, len(rows)-3)}
                terminated = truncated = False
                decision = 0
                while not (terminated or truncated):
                    if decision >= len(rows):
                        raise RuntimeError("Parent exceeded G2 episode length")
                    row = rows[decision]
                    archived_observation = np.asarray(json.loads(row["decision_observation_json"]), dtype=np.float32)
                    if not np.array_equal(observation, archived_observation):
                        raise RuntimeError(f"Parent G2 decision observation differs {seed}/{decision}")
                    with torch.inference_mode():
                        latent = agent.model.encode(agent._tensor_observation(observation))[0].cpu().numpy()
                    if not np.array_equal(latent, cached_latents[feature_lookup[(seed, decision)]]):
                        raise RuntimeError(f"Live latent differs from passive extraction {seed}/{decision}")
                    action = agent.act(observation, first_step=decision == 0, eval_mode=True)
                    if not np.array_equal(action, np.asarray(json.loads(row["selected_action_json"]), dtype=np.float32)):
                        raise RuntimeError(f"Parent G2 action differs {seed}/{decision}")
                    if decision in selected:
                        first = fork_step(env, monitor, action)
                        second = fork_step(env, monitor, action)
                        compare(first, second, report["max_abs_difference"], f"repeat {seed}/{decision}")
                    parent = signature(env, monitor, action)
                    compare(parent, g2_signature(row), report["max_abs_difference"], f"parent G2 {seed}/{decision}")
                    if decision in selected:
                        compare(first, parent, report["max_abs_difference"], f"fork parent {seed}/{decision}")
                        report["states"].append({"seed": seed, "decision_index": decision,
                                                 "behavior_force_next_max_N": parent["force_next_max_N"]})
                    observation = np.asarray(parent["observation"], dtype=np.float32)
                    terminated, truncated = parent["terminated"], parent["truncated"]
                    decision += 1
                if decision != len(rows):
                    raise RuntimeError(f"Parent G2 episode length differs: {seed}")
                print(f"Generic fork audit seed={seed} parent_steps={decision} states={len(selected)}", flush=True)
        finally:
            env.close()
        report["status"] = "PASS_GENERIC_FORK_RESTORE_PARENT_AND_G2_FORCE"
        report["state_count"] = len(report["states"])
    except BaseException:
        report["status"] = "FAIL_STOP"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
