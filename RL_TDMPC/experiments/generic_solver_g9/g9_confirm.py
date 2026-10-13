"""Freeze G9, then collect untouched 102000-102029 reused-scene confirmation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
for folder in ("generic_solver_g6a", "generic_solver_g5", "generic_solver_g3", "generic_solver_g2"):
    sys.path.insert(0, str(RL / "experiments" / folder))

from g9_estimate import Transition, ukf_episode  # noqa: E402
from g9_events import encoder, map_physical  # noqa: E402
from g9_risk import OUT, ESTIMATORS, HORIZONS, F_LIM, cuda_gate, json_new, sha, load_model, predict, metrics  # noqa: E402
from g6_collect import setup  # noqa: E402
from collect import evaluate_one, EPISODE_FIELDS, STEP_FIELDS  # noqa: E402
import g5_confirm  # noqa: E402

CONFIRM = tuple(range(102000, 102030))
EPISODES = OUT / "confirmation_behavior_episodes.csv"
STEPS = OUT / "confirmation_behavior_steps.csv"
SOURCES = ("g9_data.py", "g9_estimate.py", "g9_risk.py", "g9_select.py",
           "g9_events.py", "g9_event_link.py", "g9_oracle.py", "g9_plans.py",
           "g9_filter_noise.py", "g9_precursor.py", "g9_confirm.py")
REQUIRED = ("metadata.json", "dataset_split.json", "model_manifest.json",
            "validation_horizon_selection.json", "horizon_prediction_metrics.csv",
            "quantile_coverage_metrics.csv", "state_estimation_metrics.csv",
            "lead_time_event_linked.csv", "lead_time_preexisting_force_safe.csv",
            "oracle_parent_replay_audit.json",
            "oracle_early_recoverability.csv", "prospective_plan_prediction_metrics.csv",
            "filter_noise_sensitivity.json")


def csv_new(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def freeze_value():
    selection = json.loads((OUT / "validation_horizon_selection.json").read_text())
    if selection["focus_horizon_by_estimator"] != {e: 2 for e in ESTIMATORS}:
        raise RuntimeError("G9 validation-only horizon selection changed")
    split = json.loads((OUT / "dataset_split.json").read_text())
    if split["untouched_confirmation"] != list(CONFIRM) or not split["confirmation_unopened"]:
        raise RuntimeError("G9 untouched confirmation seed declaration changed")
    oracle = json.loads((OUT / "oracle_parent_replay_audit.json").read_text())
    if oracle["status"] != "PASS" or oracle["parent_steps_compared"] != 1848:
        raise RuntimeError("G9 archived parent replay not passed")
    models = json.loads((OUT / "model_manifest.json").read_text())
    if len(models["checkpoint_sha256"]) != 12:
        raise RuntimeError("G9 model manifest incomplete")
    return {"status": "G9_ALL_MODEL_AND_HORIZON_CHOICES_FROZEN_BEFORE_102000_CONFIRMATION",
            "confirmation_seeds_ordered": CONFIRM,
            "scene_lifecycle": "one reused Generic-SOFA scene; seed/reset each episode in ascending order",
            "device": "CUDA_VISIBLE_DEVICES=0, cuda:0, no CPU fallback",
            "controller_checkpoint_sha256": g5_confirm.EXPECTED_CHECKPOINT_SHA,
            "F_lim_N": F_LIM,
            "focus_horizon_by_estimator": selection["focus_horizon_by_estimator"],
            "all_estimators_and_horizons_reported": True,
            "confirmation_protocol": "frozen-policy behavior trajectories; E0/E1/E2 computed from current/past observations/actions only; complete d>=2 and d+5<=episode length; direct R_H for H1/2/3/5; fixed validation Q95 offsets; no model selection on confirmation",
            "prospective_plan_protocol": "matched true-SOFA 3-step TD-MPC2 plan forks evaluated only in archived G8 independent diagnostic; no H5 native planner claim",
            "source_sha256": {name: sha(HERE / name) for name in SOURCES},
            "required_development_artifact_sha256": {name: sha(OUT / name) for name in REQUIRED},
            "model_checkpoint_sha256": models["checkpoint_sha256"],
            "model_normalization_sha256": models["normalization_sha256"]}


def freeze():
    cuda_gate()
    if EPISODES.exists() or STEPS.exists() or (OUT / "pre_confirmation_freeze.json").exists():
        raise FileExistsError("Confirmation already opened or freeze already exists")
    value = freeze_value()
    json_new(OUT / "pre_confirmation_freeze.json", value)
    print("G9 pre-confirmation freeze saved; 102000-102029 unopened", flush=True)


def verify_freeze():
    frozen = json.loads((OUT / "pre_confirmation_freeze.json").read_text())
    if frozen != json.loads(json.dumps(freeze_value())):
        raise RuntimeError("G9 source/development/frozen-model hash changed after confirmation freeze")
    manifest = json.loads((OUT / "model_manifest.json").read_text())
    for key, checksum in manifest["checkpoint_sha256"].items():
        name, h = key.split("_H")
        path = OUT / "models/risk" / name / f"H{h}" / "model.pt"
        if sha(path) != checksum:
            raise RuntimeError(f"Frozen G9 model changed {key}")
    return frozen


def collect():
    cuda_gate()
    frozen = verify_freeze()
    if EPISODES.exists() or STEPS.exists():
        raise FileExistsError("Confirmation behavior already exists; cannot skip directly in reused scene")
    json_new(OUT / "confirmation_seed_manifest.json", {"seeds": CONFIRM,
        "pre_confirmation_freeze_sha256": sha(OUT / "pre_confirmation_freeze.json"),
        "scene_lifecycle": frozen["scene_lifecycle"],
        "checkpoint_sha256": frozen["controller_checkpoint_sha256"],
        "python": sys.executable, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "cuda_device_index": 0, "cuda_device": torch.cuda.get_device_name(0)})
    env, agent, monitor = setup(list(CONFIRM))
    try:
        with EPISODES.open("x", newline="") as ep_stream, STEPS.open("x", newline="") as step_stream:
            ew = csv.DictWriter(ep_stream, fieldnames=EPISODE_FIELDS)
            sw = csv.DictWriter(step_stream, fieldnames=STEP_FIELDS)
            ew.writeheader()
            sw.writeheader()
            for index, seed in enumerate(CONFIRM):
                episode, rows = evaluate_one(env, agent, seed, index,
                                             g5_confirm.EXPECTED_CHECKPOINT_SHA, monitor)
                if episode["simulation_error"]:
                    raise RuntimeError(f"G9 confirmation simulation error seed {seed}")
                ew.writerow(episode)
                sw.writerows(rows)
                ep_stream.flush()
                step_stream.flush()
                print(f"G9 confirmation seed={seed} steps={len(rows)}", flush=True)
    finally:
        env.close()


def analyze():
    cuda_gate()
    verify_freeze()
    manifest = json.loads((OUT / "confirmation_seed_manifest.json").read_text())
    if manifest["seeds"] != list(CONFIRM) or manifest["pre_confirmation_freeze_sha256"] != sha(OUT / "pre_confirmation_freeze.json"):
        raise RuntimeError("G9 confirmation seed protocol changed")
    with EPISODES.open(newline="") as stream:
        episodes = list(csv.DictReader(stream))
    if [int(r["seed"]) for r in episodes] != list(CONFIRM):
        raise RuntimeError("G9 confirmation episodes incomplete or out of order")
    by_seed = {}
    with STEPS.open(newline="") as stream:
        for row in csv.DictReader(stream):
            by_seed.setdefault(int(row["seed"]), []).append(row)
    if sorted(by_seed) != list(CONFIRM):
        raise RuntimeError("G9 confirmation behavior steps incomplete")
    agent = encoder()
    transition = Transition()
    transition.load_state_dict(torch.load(OUT / "models/transition.pt", map_location="cpu", weights_only=True))
    transition.eval()
    filter_def = json.loads((OUT / "filter_definition.json").read_text())
    Q = np.asarray(filter_def["process_covariance_diagonal"])
    R = float(filter_def["measurement_covariance_diagonal"][0])
    records = {k: [] for k in ("seed", "decision_index", "latent", "E0", "E1", "E2", "action_sequence", "force_sequence")}
    for seed in CONFIRM:
        rows = by_seed[seed]
        if ([int(r["step_index"]) for r in rows] != list(range(len(rows))) or
                len(rows) != int(episodes[seed-CONFIRM[0]]["episode_length"])):
            raise RuntimeError(f"G9 confirmation episode indexing changed {seed}")
        obs = np.asarray([json.loads(r["decision_observation_json"]) for r in rows], dtype=np.float32)
        action = np.asarray([json.loads(r["selected_action_json"]) for r in rows], dtype=np.float32)
        force = np.asarray([float(r["distal_contact_force_max_N"]) for r in rows], dtype=np.float32)
        with torch.inference_mode():
            latent = np.asarray([agent.model.encode(agent._tensor_observation(o))[0].cpu().numpy() for o in obs])
        ukf, _ = ukf_episode(transition, obs, action, Q, R)
        decisions = list(range(2, len(rows)-4))
        if not decisions:
            raise RuntimeError(f"G9 confirmation episode too short {seed}")
        e0 = map_physical("E0", obs[decisions])
        h = np.asarray([np.r_[obs[d-2:d+1].reshape(-1), action[d-2:d].reshape(-1)] for d in decisions])
        e1 = map_physical("E1", h)
        e2 = map_physical("E2", ukf[decisions])
        for j, d in enumerate(decisions):
            if d + 5 > len(rows):
                raise RuntimeError("G9 confirmation future horizon crosses episode end")
            records["seed"].append(seed)
            records["decision_index"].append(d)
            records["latent"].append(latent[d])
            for name, value in (("E0", e0), ("E1", e1), ("E2", e2)):
                records[name].append(value[j])
            records["action_sequence"].append(action[d:d+5])
            records["force_sequence"].append(force[d:d+5])
    arrays = {key: np.asarray(value, dtype=(np.int32 if key in ("seed", "decision_index") else np.float32))
              for key, value in records.items()}
    np.savez_compressed(OUT / "confirmation_temporal_behavior.npz", **arrays)
    report = []
    for name in ESTIMATORS:
        for horizon in HORIZONS:
            x = np.concatenate((arrays["latent"], arrays[name], arrays["action_sequence"][:, :horizon].reshape(len(arrays["seed"]), -1)), axis=1)
            y = np.max(arrays["force_sequence"][:, :horizon], axis=1)
            net, mean, scale, delta = load_model(name, horizon)
            p = predict(net, x, mean, scale)
            report.append(metrics(y, p, delta, name, horizon, "untouched_confirmation_behavior"))
    csv_new(OUT / "confirmation_metrics.csv", report)
    print(f"G9 untouched confirmation analyzed: {len(CONFIRM)} episodes, {len(arrays['seed'])} H5-complete decision rows", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("freeze", "collect", "analyze"), required=True)
    stage = parser.parse_args().stage
    {"freeze": freeze, "collect": collect, "analyze": analyze}[stage]()


if __name__ == "__main__":
    main()
