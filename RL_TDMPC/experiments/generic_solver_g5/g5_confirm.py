"""Untouched G5 Generic-SOFA confirmation, only after full development freeze."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
sys.path.insert(0, str(RL / "experiments/generic_solver_g3"))
sys.path.insert(0, str(RL / "experiments/generic_solver_g2"))
sys.path.insert(0, str(RL / "experiments/generic_solver_g4"))
sys.path.insert(0, str(RL))
sys.path.insert(0, str(RL.parent))

from envs.steve_env import make_steve_env  # noqa: E402
from tdmpc2.agent import TDMPC2Agent  # noqa: E402
from tdmpc2.common import load_torch_checkpoint, set_seed  # noqa: E402
from train import build_agent_config  # noqa: E402
from substep_monitor import DistalForceMonitor  # noqa: E402
from collect import check_scene, evaluate_one, STEP_FIELDS, EPISODE_FIELDS  # noqa: E402
from prepare import G1B, CONFIG, EXPECTED_CHECKPOINT_SHA, sha, check_force_source  # noqa: E402
from protocol import choose_states, generate_candidates  # noqa: E402
from restore_audit import fork_step, signature, compare, g2_signature  # noqa: E402
from physical_features import extract  # noqa: E402
from g5_data import OUT, PRIMARY, spec, minimal_groups  # noqa: E402
from g5_prepare import CONFIRM, SEEDS  # noqa: E402
from g5_metrics import load_predictor, predict, evaluate_candidate, classification, regression, quantiles  # noqa: E402
from g5_development import write_csv  # noqa: E402
from g5_freeze import verify  # noqa: E402

BEHAVIOR_EPISODES = OUT / "confirmation_behavior_episodes.csv"
BEHAVIOR_STEPS = OUT / "confirmation_behavior_steps.csv"
CANDIDATES = OUT / "confirmation_candidates"
MODELS = PRIMARY + ("Z1", "P1", "Pmin1", "Pmin", "ZPmin")


def new_json(path, data):
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def setup():
    check_force_source()
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available() or
            torch.cuda.device_count() != 1 or "RTX 5090" not in torch.cuda.get_device_name(0)):
        raise RuntimeError("Frozen CUDA_VISIBLE_DEVICES=0 required")
    checkpoint_path = Path(json.loads((G1B / "selected_checkpoint.json").read_text())["path"])
    if sha(checkpoint_path) != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("Frozen TD-MPC2 checkpoint changed")
    checkpoint = load_torch_checkpoint(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    if config != yaml.safe_load(CONFIG.read_text()):
        raise RuntimeError("Frozen Generic environment config changed")
    set_seed(CONFIRM[0])
    env = make_steve_env(config["environment"])
    agent = TDMPC2Agent(14, 2, build_agent_config(config), episode_length=int(config["environment"]["max_episode_steps"]), device=torch.device("cuda:0"))
    agent.load_state_dict(checkpoint["agent"], load_optimizers=False)
    monitor = DistalForceMonitor(env)
    return env, agent, monitor


def behavior():
    verify()
    for path in (BEHAVIOR_EPISODES, BEHAVIOR_STEPS, OUT / "confirmation_seed_manifest.json"):
        if path.exists():
            raise FileExistsError(path)
    new_json(OUT / "confirmation_seed_manifest.json", {"seeds": CONFIRM,
        "predeclared_seed_protocol_sha256": sha(OUT / "confirmation_seed_protocol.json"),
        "pre_confirmation_freeze_sha256": sha(OUT / "pre_confirmation_freeze.json"),
        "frozen_navigation_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "scene_lifecycle": "one reused Generic scene, ordered resets 94000-94029",
        "all_models_features_metrics_frozen_before_open": True})
    env, agent, monitor = setup()
    try:
        with BEHAVIOR_EPISODES.open("w", newline="") as ep_stream, BEHAVIOR_STEPS.open("w", newline="") as step_stream:
            ep_writer, step_writer = csv.DictWriter(ep_stream, EPISODE_FIELDS), csv.DictWriter(step_stream, STEP_FIELDS)
            ep_writer.writeheader()
            step_writer.writeheader()
            for i, seed in enumerate(CONFIRM):
                episode, steps = evaluate_one(env, agent, seed, i, EXPECTED_CHECKPOINT_SHA, monitor)
                if episode["simulation_error"]:
                    raise RuntimeError(f"Confirmation simulation error {seed}")
                ep_writer.writerow(episode)
                step_writer.writerows(steps)
                ep_stream.flush()
                step_stream.flush()
                print(f"G5 confirmation behavior seed={seed} steps={len(steps)}", flush=True)
    finally:
        env.close()


def candidate_pass():
    verify()
    manifest = json.loads((OUT / "confirmation_seed_manifest.json").read_text())
    if manifest["pre_confirmation_freeze_sha256"] != sha(OUT / "pre_confirmation_freeze.json"):
        raise RuntimeError("G5 freeze changed after confirmation opened")
    rows_by_seed = {}
    with BEHAVIOR_STEPS.open(newline="") as stream:
        for row in csv.DictReader(stream):
            rows_by_seed.setdefault(int(row["seed"]), []).append(row)
    if sorted(rows_by_seed) != CONFIRM:
        raise RuntimeError("Confirmation behavior set incomplete")
    chosen = choose_states(CONFIRM, rows_by_seed, "confirmation")
    selected = {(state["seed"], state["decision_index"]): state for state in chosen}
    if len(selected) != 120:
        raise RuntimeError("Confirmation selection must have 120 unique states")
    CANDIDATES.mkdir(parents=True, exist_ok=False)
    env, agent, monitor = setup()
    differences, steps_compared = {}, 0
    try:
        for seed in CONFIRM:
            set_seed(seed)
            env.action_space.seed(seed)
            observation, _ = env.reset(seed=seed)
            check_scene(env)
            if not hasattr(monitor, "_g5_added"):
                env._simulation.root.addObject(monitor)
                monitor._g5_added = True
            rows = rows_by_seed[seed]
            terminated = truncated = False
            d = 0
            while not (terminated or truncated):
                if d >= len(rows):
                    raise RuntimeError(f"Replay overran behavior episode {seed}")
                row = rows[d]
                if not np.array_equal(observation, np.asarray(json.loads(row["decision_observation_json"]), dtype=np.float32)):
                    raise RuntimeError(f"Observation replay mismatch {seed}/{d}")
                state = selected.get((seed, d))
                if state is not None:
                    if d < 1:
                        raise RuntimeError("Physical history undefined at d=0")
                    physical, audit = extract(env)
                    delta = abs(audit["current_distal_force_final_N"] - float(rows[d-1]["distal_contact_force_final_N"]))
                    differences["preaction_previous_final_force_N"] = max(differences.get("preaction_previous_final_force_N", 0.), delta)
                    if delta > 1e-8:
                        raise RuntimeError(f"Physical pre-action alignment mismatch {seed}/{d}")
                    with torch.inference_mode():
                        latent = agent.model.encode(agent._tensor_observation(observation))[0].cpu().numpy()
                action = agent.act(observation, first_step=d == 0, eval_mode=True)
                if not np.array_equal(action, np.asarray(json.loads(row["selected_action_json"]), dtype=np.float32)):
                    raise RuntimeError(f"Action replay mismatch {seed}/{d}")
                forked_behavior = None
                if state is not None:
                    outcomes = []
                    for candidate in generate_candidates(action, state["state_id"], "confirmation"):
                        result = fork_step(env, monitor, np.asarray(candidate["action"], dtype=np.float32))
                        if candidate["candidate_id"] == 0:
                            forked_behavior = result
                            compare(result, g2_signature(row), differences, f"confirmation behavior fork {seed}/{d}")
                        outcomes.append({**candidate, "force_next_max_N": result["force_next_max_N"], "outcome": result})
                    new_json(CANDIDATES / f"state_{state['state_id']:04d}.json", {**state,
                        "observation": observation.tolist(), "latent": latent.tolist(), "physical": physical.tolist(),
                        "behavior_action": action.tolist(), "candidates": outcomes})
                    print(f"G5 confirmation candidate state {state['state_id']+1}/120 seed={seed} d={d}", flush=True)
                parent = signature(env, monitor, action)
                compare(parent, g2_signature(row), differences, f"confirmation parent {seed}/{d}")
                if forked_behavior is not None:
                    compare(parent, forked_behavior, differences, f"confirmation parent-fork {seed}/{d}")
                observation = np.asarray(parent["observation"], dtype=np.float32)
                terminated, truncated = parent["terminated"], parent["truncated"]
                d += 1
                steps_compared += 1
            if d != len(rows):
                raise RuntimeError(f"Episode length changed {seed}")
    finally:
        env.close()
    if len(list(CANDIDATES.glob("state_*.json"))) != 120:
        raise RuntimeError("Confirmation candidate set incomplete")
    new_json(OUT / "confirmation_replay_audit.json", {"status": "PASS", "parent_steps_compared": steps_compared,
        "candidate_states": 120, "candidate_evaluations": 2520, "max_abs_difference": differences})


def confirmation_x(name, states, groups):
    include_z, indices = spec(name, groups)
    x = []
    for state in states:
        parts = []
        if include_z:
            parts.append(np.asarray(state["latent"], dtype=np.float64))
        if indices:
            p = np.asarray(state["physical"], dtype=np.float64)
            if p.shape != (38,):
                raise RuntimeError("Confirmation P dimension changed")
            parts.append(p[indices])
        vector = np.concatenate(parts)
        for candidate in state["candidates"]:
            x.append(np.r_[vector, candidate["action"]])
    return np.asarray(x, dtype=np.float64)


def evaluate():
    verify()
    audit = json.loads((OUT / "confirmation_replay_audit.json").read_text())
    if audit["status"] != "PASS":
        raise RuntimeError("Confirmation replay gate failed")
    paths = sorted(CANDIDATES.glob("state_*.json"))
    if len(paths) != 120:
        raise RuntimeError("Confirmation labels incomplete")
    states = [json.loads(path.read_text()) for path in paths]
    if [state["state_id"] for state in states] != list(range(120)):
        raise RuntimeError("Confirmation state order differs")
    y = np.asarray([candidate["force_next_max_N"] for state in states for candidate in state["candidates"]], dtype=np.float64)
    contact = np.repeat(np.asarray([float(state["physical"][34]) > 0 for state in states], dtype=bool), 21)
    if len(y) != 2520:
        raise RuntimeError("Confirmation label count differs")
    q = quantiles()
    groups = minimal_groups()
    device = torch.device("cuda:0")
    results, summary_rows, state_rows, contact_rows, emergence_rows, boundary_rows = [], [], [], [], [], []
    for name in MODELS:
        x = confirmation_x(name, states, groups)
        for seed in SEEDS:
            prediction = predict(load_predictor(name, seed, device, groups), x, device)
            individual, aggregate, boundaries = evaluate_candidate(states, y, prediction, q)
            label = {"model": name, "training_seed": seed}
            state_rows.extend({**label, **record} for record in individual)
            b = boundaries["q95"]
            summary_rows.append({**label, **aggregate, "Q95_true_unsafe": b["true_exceedances"],
                "Q95_false_safe_count": b["FN_false_safe"], "Q95_false_safe_rate": b["false_safe_rate_P_pred_safe_given_true_unsafe"],
                "Q95_AUROC": b["AUROC"], "Q95_AUPRC": b["AUPRC"]})
            boundary_rows.extend({**label, "reference": reference, "reference_N": q[reference], **values} for reference, values in boundaries.items())
            for stratum, mask in (("no_current_contact", ~contact), ("current_contact", contact)):
                ids = np.flatnonzero(mask[::21])
                if not len(ids):
                    raise RuntimeError(f"Empty confirmation contact stratum: {stratum}")
                b = classification(y[mask], prediction[mask], q["q95"])
                contact_rows.append({**label, "stratum": stratum, "state_count": len(ids), **regression(y[mask], prediction[mask]),
                    "mean_pairwise_accuracy": float(np.nanmean([individual[i]["pairwise_accuracy"] if individual[i]["pairwise_accuracy"] is not None else np.nan for i in ids])),
                    "mean_safe_choice_regret_N": float(np.mean([individual[i]["safe_choice_regret_N"] for i in ids])),
                    "Q95_true_exceedances": b["true_exceedances"], "Q95_false_safe_count": b["FN_false_safe"],
                    "Q95_false_safe_rate": b["false_safe_rate_P_pred_safe_given_true_unsafe"],
                    "Q95_AUROC": b["AUROC"], "Q95_AUPRC": b["AUPRC"]})
            b = classification(y[~contact], prediction[~contact], q["q95"])
            emergence_rows.append({**label, "no_contact_states": int(np.sum(~contact[::21])), "new_high_force_candidates": b["true_exceedances"],
                "false_safe_count": b["FN_false_safe"], "false_safe_rate": b["false_safe_rate_P_pred_safe_given_true_unsafe"],
                "unsafe_recall": b["unsafe_recall"], "AUROC": b["AUROC"], "AUPRC": b["AUPRC"]})
            for i, state in enumerate(states):
                for j, candidate in enumerate(state["candidates"]):
                    results.append({**label, "episode_seed": state["seed"], "decision_index": state["decision_index"],
                        "state_id": state["state_id"], "candidate_id": j, "candidate_kind": candidate["kind"],
                        "action_translation": candidate["action"][0], "action_rotation": candidate["action"][1],
                        "current_contact": bool(contact[i*21+j]), "force_next_max_N": y[i*21+j],
                        "predicted_force_next_max_N": prediction[i*21+j]})
    write_csv(OUT / "confirmation_candidate_results.csv", results)
    write_csv(OUT / "confirmation_metrics.csv", summary_rows)
    write_csv(OUT / "confirmation_boundary_metrics.csv", boundary_rows)
    write_csv(OUT / "confirmation_state_metrics.csv", state_rows)
    write_csv(OUT / "confirmation_contact_conditioned_metrics.csv", contact_rows)
    write_csv(OUT / "confirmation_no_contact_emergence_metrics.csv", emergence_rows)
    new_json(OUT / "confirmation_evaluation_summary.json", {"status": "COMPLETE_FROZEN_G5_UNTOUCHED_CONFIRMATION",
        "seeds": CONFIRM, "states": 120, "candidate_labels": 2520, "models": list(MODELS),
        "current_contact_states": int(np.sum(contact[::21])), "no_current_contact_states": int(np.sum(~contact[::21])),
        "P_min_groups": list(groups), "all_model_weights_and_features_unchanged_after_freeze": True})
    print("G5 untouched confirmation evaluation complete; no retraining/recalibration", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("behavior", "candidates", "evaluate"), required=True)
    args = parser.parse_args()
    {"behavior": behavior, "candidates": candidate_pass, "evaluate": evaluate}[args.stage]()


if __name__ == "__main__":
    main()
