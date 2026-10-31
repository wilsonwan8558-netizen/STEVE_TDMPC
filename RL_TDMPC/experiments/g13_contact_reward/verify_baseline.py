"""F2H zero-update G13 regression against archived and F2G references.

No gradient updates or checkpoint writes occur here.
"""

from __future__ import annotations

import csv
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
from tdmpc2.common import load_torch_checkpoint, set_seed  # noqa: E402
from train import build_agent_config  # noqa: E402
from g13_env import make_g13_env  # noqa: E402
from g13_model import G13Agent  # noqa: E402
import eve  # noqa: E402

OUT = RL / "logs/contact_aware_joint_mppi/stage_f2h"
P = json.loads((OUT / "baseline_protocol.json").read_text())
F2G = Path(P["f2g_reference_root"])


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def write(data: dict) -> None:
    (OUT / "baseline_equivalence.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def check_telemetry(env, info: dict) -> dict:
    samples = np.asarray([row[0].magnitude_N for row in env.monitor.samples], dtype=float)
    if (samples.shape != (22,) or not np.all(np.isfinite(samples)) or np.any(samples < 0)
            or int(info.get("sofa_substeps", -1)) != 22):
        raise RuntimeError("G13 Generic contact telemetry is incomplete or invalid")
    fti = float(np.sum(samples) * 0.006)
    if not np.isclose(fti, info["distal_force_time_integral_Ns"], rtol=0, atol=1e-9):
        raise RuntimeError("G13 contact FTI does not align with executed substeps")
    if not np.isclose(np.max(samples), info["distal_contact_force_max_N"], rtol=0, atol=1e-9):
        raise RuntimeError("G13 peak force does not align with executed substeps")
    if info.get("simulation_error", False):
        raise RuntimeError("SOFA simulation error during baseline gate")
    return {"peak_N": float(np.max(samples)), "fti_Ns": fti,
            "contact_substeps": int(np.count_nonzero(samples > 0))}


def new_agent(checkpoint, *, optimizers: bool = False):
    config = checkpoint["config"]
    agent = G13Agent(22, 2, build_agent_config(config),
                     episode_length=int(config["environment"]["max_episode_steps"]),
                     device=torch.device("cuda:0"))
    agent.load_state_dict(checkpoint["agent"], load_optimizers=optimizers)
    return agent


def model_probe(checkpoint) -> dict:
    observation = checkpoint["replay"]["episodes"][0]["observations"][0]
    set_seed(762334)
    agent = new_agent(checkpoint)
    obs = torch.as_tensor(observation, device="cuda:0").unsqueeze(0)
    with torch.no_grad():
        latent = agent.model.encode(obs)
        candidates = torch.linspace(-1, 1, 3 * 512 * 2, device="cuda:0").reshape(3, 512, 2)
        values = agent._estimate_value(latent.repeat(512, 1), candidates)
        reward_logits = agent.model.reward(latent, candidates[0, :1])
        q = agent.model.q(latent, candidates[0, :1], reduction="avg")
    set_seed(993019)
    action = agent.act(observation, first_step=True, eval_mode=True)
    expected = json.loads((F2G / "zero_update_model_probe.json").read_text())
    observed = {"candidate_score_sha256": tensor_sha(values),
                "latent_sha256": tensor_sha(latent),
                "reward_logits_sha256": tensor_sha(reward_logits),
                "q_sha256": tensor_sha(q),
                "selected_joint_action": action.tolist()}
    matches = {key: observed[key] == expected[key] for key in observed}
    return {"status": "PASS" if all(matches.values()) else "FAIL",
            "observed": observed, "matches_f2g": matches}


def historical_gate(checkpoint) -> dict:
    reference = {int(row["seed"]): row for row in csv.DictReader(
        (ROOT / P["historical_reference"]).open(newline=""))
        if int(row["checkpoint_step"]) == 260000 and int(row["seed"]) in P["historical_gate_seeds"]}
    seeds = P["historical_gate_seeds"]
    if list(reference) != seeds:
        raise RuntimeError("Archived evaluator rows missing or reordered")
    set_seed(seeds[0])
    env = make_g13_env(checkpoint["config"]["environment"])
    agent = new_agent(checkpoint)
    rows = []
    try:
        for seed in seeds:
            set_seed(seed)
            observation, _ = env.reset(seed=seed)
            if observation.shape != (22,):
                raise RuntimeError("G13 observation shape drift")
            terminated = truncated = success = False
            total_reward = peak_N = total_fti = 0.0
            steps = 0
            while not (terminated or truncated):
                action = agent.act(observation, first_step=steps == 0, eval_mode=True)
                observation, reward, terminated, truncated, info = env.step(action)
                force = check_telemetry(env, info)
                total_reward += float(reward)
                peak_N = max(peak_N, force["peak_N"])
                total_fti += force["fti_Ns"]
                steps += 1
                success = success or bool(info.get("is_success", False))
            archived = reference[seed]
            row = {"seed": seed, "archived_success": int(archived["success"]),
                   "success": int(success), "archived_length": int(archived["episode_length"]),
                   "length": steps, "archived_reward": float(archived["episode_reward"]),
                   "reward": total_reward,
                   "reward_abs_diff": abs(total_reward - float(archived["episode_reward"])),
                   "episode_peak_force_nominal_N": peak_N,
                   "episode_contact_FTI_nominal_Ns": total_fti}
            row["match"] = (row["success"] == row["archived_success"] and
                            row["length"] == row["archived_length"] and
                            row["reward_abs_diff"] <= 1e-6)
            rows.append(row)
            (OUT / "historical_gate_progress.json").write_text(json.dumps(rows, indent=2) + "\n")
            print("F2H_HISTORICAL", seed, int(row["match"]), steps, total_reward, flush=True)
            if not row["match"]:
                break
    finally:
        env.close()
    return {"status": "PASS" if len(rows) == len(seeds) and all(r["match"] for r in rows) else "FAIL",
            "archived_steps_checked": sum(r["archived_length"] for r in rows),
            "reproduced_steps_checked": sum(r["length"] for r in rows),
            "max_reward_abs_diff": max((r["reward_abs_diff"] for r in rows), default=None),
            "rows": rows}


def f2g_regression(checkpoint) -> dict:
    seed = int(P["f2g_regression_seed"])
    expected_rows = json.loads((F2G / "evaluation_baseline.json").read_text())
    expected = next(row for row in expected_rows if row["seed"] == seed)
    init_hash = json.loads((F2G / "model_drift_analysis.json").read_text())["fixed_observation_sha256"]
    set_seed(seed)
    env = make_g13_env(checkpoint["config"]["environment"])
    agent = new_agent(checkpoint)
    try:
        set_seed(seed)
        observation, _ = env.reset(seed=seed)
        initial_observation_sha256 = hashlib.sha256(observation.tobytes()).hexdigest()
        initial_target = env.unwrapped.intervention.target.coordinates3d.tolist()
        actions = []
        total_reward = peak_N = total_fti = 0.0
        terminated = truncated = success = False
        while not (terminated or truncated):
            action = agent.act(observation, first_step=not actions, eval_mode=True)
            observation, reward, terminated, truncated, info = env.step(action)
            force = check_telemetry(env, info)
            actions.append(action.tolist())
            total_reward += float(reward)
            peak_N = max(peak_N, force["peak_N"])
            total_fti += force["fti_Ns"]
            success = success or bool(info.get("is_success", False))
    finally:
        env.close()
    comparisons = {"initial_observation_sha256": initial_observation_sha256 == init_hash,
                   "actions_exact": np.array_equal(np.asarray(actions), np.asarray(expected["actions"])),
                   "success": int(success) == expected["success"],
                   "length": len(actions) == expected["length"],
                   "reward": abs(total_reward - expected["raw_task_reward"]) <= 1e-9}
    return {"status": "PASS" if all(comparisons.values()) else "FAIL",
            "seed": seed, "matches_f2g": comparisons,
            "initial_target_tracking3d_mm": initial_target,
            "steps": len(actions), "reward": total_reward, "success": int(success),
            "peak_force_nominal_N": peak_N, "contact_FTI_nominal_Ns": total_fti,
            "scope": "Regression only; reused F2G case is not an independent confirmation."}


def main() -> None:
    if (os.getenv("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available() or
            torch.cuda.device_count() != 1 or "RTX 5090" not in torch.cuda.get_device_name(0)):
        raise RuntimeError("F2H requires explicit CUDA_VISIBLE_DEVICES=0 RTX 5090")
    if not Path(eve.__file__).resolve().is_relative_to(ROOT):
        raise RuntimeError(f"eve import resolved outside new G13 worktree: {eve.__file__}")
    if sha(Path(P["checkpoint"])) != P["checkpoint_sha256"]:
        raise RuntimeError("G13 checkpoint SHA256 mismatch")
    frozen = json.loads((RL / "logs/world_model_force_aware_planning_g13/g13_training_config.json").read_text())
    if (sha(RL / "experiments/world_model_g13/g13_config.yaml") != frozen["config_yaml_sha256"] or
            sha(RL / "experiments/world_model_g13/g13_model.py") != frozen["joint_model_source_sha256"] or
            sha(RL / "experiments/world_model_g13/g13_env.py") != frozen["geometry_adapter_source_sha256"] or
            sha(RL / "envs/geometry_features.py") != frozen["geometry_feature_source_sha256"]):
        raise RuntimeError("G13 source differs from archived pre-training freeze")
    checkpoint = load_torch_checkpoint(P["checkpoint"], map_location="cpu")
    if (checkpoint.get("format_version") != 2 or checkpoint.get("algorithm") != "Geometry-Force-TD-MPC2"
            or checkpoint.get("observation_dim") != 22 or checkpoint.get("total_env_steps") != 260000
            or checkpoint.get("lambda_force_train") != 0.5 or checkpoint["config"] != frozen["config"]):
        raise RuntimeError("G13 checkpoint metadata/configuration mismatch")
    result = {"status": "RUNNING", "checkpoint_sha256": P["checkpoint_sha256"],
              "source_commit": P["source_commit"],
              "runtime": {"python": sys.executable, "torch": torch.__version__,
                          "torch_cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                          "eve_origin": eve.__file__}}
    write(result)
    result["historical_gate"] = historical_gate(checkpoint)
    write(result)
    if result["historical_gate"]["status"] != "PASS":
        result["status"] = "FAIL_STOP"
        write(result)
        raise SystemExit(2)
    result["model_probe"] = model_probe(checkpoint)
    write(result)
    if result["model_probe"]["status"] != "PASS":
        result["status"] = "FAIL_STOP"
        write(result)
        raise SystemExit(2)
    result["f2g_action_and_telemetry_regression"] = f2g_regression(checkpoint)
    write(result)
    if result["f2g_action_and_telemetry_regression"]["status"] != "PASS":
        result["status"] = "FAIL_STOP"
        write(result)
        raise SystemExit(2)
    result["status"] = "PASS"
    write(result)
    print("F2H_BASELINE_EQUIVALENCE_PASS", flush=True)


if __name__ == "__main__":
    main()
