"""Open untouched G4 seeds only after identity freeze; no fitting or tuning."""

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
from g4_data import OUT, REPRESENTATIONS, source, physical_lookup, candidate_states, state_vector  # noqa: E402
from g4_train import SEEDS  # noqa: E402
from g4_evaluate import load_predictor, predict, candidate_metrics, write_csv, write_json, three_seed_summary  # noqa: E402
from g4_aliasing import candidate_records, nearest, summary  # noqa: E402
from g3_evaluate import classification, quantiles  # noqa: E402
from g4_freeze import verify, SEEDS_CONFIRM  # noqa: E402

BEHAVIOR_EPISODES = OUT / "confirmation_behavior_episodes.csv"
BEHAVIOR_STEPS = OUT / "confirmation_behavior_steps.csv"
CANDIDATES = OUT / "confirmation_candidates"


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
    set_seed(SEEDS_CONFIRM[0])
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
    write_json(OUT / "confirmation_seed_manifest.json", {"seeds": SEEDS_CONFIRM,
        "predeclared_seed_protocol_sha256": sha(OUT / "confirmation_seed_protocol.json"),
        "pre_confirmation_freeze_sha256": sha(OUT / "pre_confirmation_freeze.json"),
        "frozen_navigation_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "scene_lifecycle": "one reused Generic scene, ordered resets 93000-93029",
        "all_models_features_metrics_frozen_before_open": True})
    env, agent, monitor = setup()
    try:
        with BEHAVIOR_EPISODES.open("w", newline="") as episode_stream, BEHAVIOR_STEPS.open("w", newline="") as step_stream:
            episode_writer, step_writer = csv.DictWriter(episode_stream, EPISODE_FIELDS), csv.DictWriter(step_stream, STEP_FIELDS)
            episode_writer.writeheader()
            step_writer.writeheader()
            for index, seed in enumerate(SEEDS_CONFIRM):
                episode, steps = evaluate_one(env, agent, seed, index, EXPECTED_CHECKPOINT_SHA, monitor)
                if episode["simulation_error"]:
                    raise RuntimeError(f"New confirmation simulation error {seed}")
                episode_writer.writerow(episode)
                step_writer.writerows(steps)
                episode_stream.flush()
                step_stream.flush()
                print(f"G4 confirmation behavior seed={seed} steps={len(steps)}", flush=True)
    finally:
        env.close()


def candidate_pass():
    verify()
    manifest = json.loads((OUT / "confirmation_seed_manifest.json").read_text())
    if manifest["pre_confirmation_freeze_sha256"] != sha(OUT / "pre_confirmation_freeze.json"):
        raise RuntimeError("G4 freeze changed after confirmation opened")
    with BEHAVIOR_STEPS.open(newline="") as stream:
        rows_by_seed = {}
        for row in csv.DictReader(stream):
            rows_by_seed.setdefault(int(row["seed"]), []).append(row)
    if sorted(rows_by_seed) != SEEDS_CONFIRM:
        raise RuntimeError("Confirmation behavior set incomplete")
    chosen = choose_states(SEEDS_CONFIRM, rows_by_seed, "confirmation")
    selected = {(s["seed"], s["decision_index"]): s for s in chosen}
    if len(selected) != 120:
        raise RuntimeError("Confirmation selection not 120 unique states")
    CANDIDATES.mkdir(parents=True, exist_ok=False)
    env, agent, monitor = setup()
    differences = {}
    steps_compared = 0
    try:
        for seed in SEEDS_CONFIRM:
            set_seed(seed)
            env.action_space.seed(seed)
            observation, _ = env.reset(seed=seed)
            check_scene(env)
            if not hasattr(monitor, "_g4_added"):
                env._simulation.root.addObject(monitor)
                monitor._g4_added = True
            rows = rows_by_seed[seed]
            terminated = truncated = False
            d = 0
            while not (terminated or truncated):
                if d >= len(rows):
                    raise RuntimeError(f"Confirmation replay overran behavior episode {seed}")
                row = rows[d]
                if not np.array_equal(observation, np.asarray(json.loads(row["decision_observation_json"]), dtype=np.float32)):
                    raise RuntimeError(f"Confirmation observation changed {seed}/{d}")
                state = selected.get((seed, d))
                if state is not None:
                    if d < 1:
                        raise RuntimeError("History representation undefined at d=0")
                    physical, audit = extract(env)
                    delta = abs(audit["current_distal_force_final_N"]-float(rows[d-1]["distal_contact_force_final_N"]))
                    differences["preaction_previous_final_force_N"] = max(differences.get("preaction_previous_final_force_N", 0.), delta)
                    if delta > 1e-8:
                        raise RuntimeError(f"P decision alignment changed {seed}/{d}")
                    with torch.inference_mode():
                        latent = agent.model.encode(agent._tensor_observation(observation))[0].cpu().numpy()
                    history = np.r_[np.asarray(json.loads(rows[d-1]["decision_observation_json"]), np.float32),
                                    observation, np.asarray(json.loads(rows[d-1]["selected_action_json"]), np.float32)]
                    if history.shape != (30,):
                        raise RuntimeError("H vector changed")
                action = agent.act(observation, first_step=d == 0, eval_mode=True)
                if not np.array_equal(action, np.asarray(json.loads(row["selected_action_json"]), dtype=np.float32)):
                    raise RuntimeError(f"Confirmation behavior action changed {seed}/{d}")
                forked_behavior = None
                if state is not None:
                    outcomes = []
                    for candidate in generate_candidates(action, state["state_id"], "confirmation"):
                        result = fork_step(env, monitor, np.asarray(candidate["action"], dtype=np.float32))
                        if candidate["candidate_id"] == 0:
                            forked_behavior = result
                            compare(result, g2_signature(row), differences, f"confirmation behavior fork {seed}/{d}")
                        outcomes.append({**candidate, "force_next_max_N": result["force_next_max_N"], "outcome": result})
                    write_json(CANDIDATES / f"state_{state['state_id']:04d}.json",
                               {**state, "observation": observation.tolist(), "latent": latent.tolist(),
                                "history": history.tolist(), "physical": physical.tolist(),
                                "behavior_action": action.tolist(), "candidates": outcomes})
                    print(f"G4 confirmation candidate state {state['state_id']+1}/120 seed={seed} d={d}", flush=True)
                parent = signature(env, monitor, action)
                compare(parent, g2_signature(row), differences, f"confirmation parent {seed}/{d}")
                if forked_behavior is not None:
                    compare(parent, forked_behavior, differences, f"confirmation parent-fork {seed}/{d}")
                observation = np.asarray(parent["observation"], dtype=np.float32)
                terminated, truncated = parent["terminated"], parent["truncated"]
                d += 1
                steps_compared += 1
            if d != len(rows):
                raise RuntimeError(f"Confirmation episode length changed {seed}")
    finally:
        env.close()
    if len(list(CANDIDATES.glob("state_*.json"))) != 120:
        raise RuntimeError("Confirmation candidate count incomplete")
    write_json(OUT / "confirmation_replay_audit.json", {"status": "PASS", "parent_steps_compared": steps_compared,
        "candidate_states": 120, "candidate_evaluations": 2520, "max_abs_difference": differences})


def evaluate():
    verify()
    audit = json.loads((OUT / "confirmation_replay_audit.json").read_text())
    if audit["status"] != "PASS":
        raise RuntimeError("New confirmation replay gate failed")
    paths = sorted(CANDIDATES.glob("state_*.json"))
    if len(paths) != 120:
        raise RuntimeError("New confirmation candidate labels incomplete")
    states = [json.loads(path.read_text()) for path in paths]
    q = quantiles()
    device = torch.device("cuda:0")
    results, summary_rows, state_rows = [], [], []
    for representation in REPRESENTATIONS:
        field = {"Z": "latent", "O": "observation", "H": "history", "P": "physical"}[representation]
        x = np.asarray([np.r_[state[field], candidate["action"]] for state in states for candidate in state["candidates"]], dtype=np.float64)
        y = np.asarray([candidate["force_next_max_N"] for state in states for candidate in state["candidates"]], dtype=np.float64)
        for supervision in (0, 1):
            cell = f"{representation}{supervision}"
            for seed in SEEDS:
                prediction = predict(load_predictor(cell, seed, device), x, device)
                individual, aggregate = candidate_metrics(states, y, prediction, cell, seed)
                state_rows.extend(individual)
                boundary = classification(y, prediction, q["q95"])
                summary_rows.append({"model": cell, "training_seed": seed,
                    **aggregate, "Q95_true_unsafe": boundary["true_exceedances"],
                    "Q95_false_safe_count": boundary["FN_false_safe"],
                    "Q95_false_safe_rate": boundary["false_safe_rate_P_pred_safe_given_true_unsafe"],
                    "Q95_AUROC": boundary["AUROC"], "Q95_AUPRC": boundary["AUPRC"]})
                for i, state in enumerate(states):
                    for j, candidate in enumerate(state["candidates"]):
                        results.append({"model": cell, "training_seed": seed, "episode_seed": state["seed"],
                                        "decision_index": state["decision_index"], "state_id": state["state_id"],
                                        "candidate_id": j, "candidate_kind": candidate["kind"],
                                        "action_translation": candidate["action"][0], "action_rotation": candidate["action"][1],
                                        "force_next_max_N": y[i*21+j], "predicted_force_next_max_N": prediction[i*21+j]})
    write_csv(OUT / "confirmation_candidate_results.csv", results)
    write_csv(OUT / "confirmation_metrics.csv", summary_rows)
    write_csv(OUT / "confirmation_state_metrics.csv", state_rows)
    write_json(OUT / "confirmation_three_seed_summary.json", {
        "candidate": three_seed_summary(summary_rows),
        "standard_deviation": "sample SD across three independent training seeds, ddof=1"})
    arrays, split, index = source()
    physical_lookup_data = physical_lookup()
    development_states = [s for group in ("train", "validation", "development_test") for s in candidate_states(group, split)]
    alias_summary = {}
    for representation in REPRESENTATIONS:
        norm = json.loads((OUT / "models" / representation / "normalization.json").read_text())
        mean, scale = np.asarray(norm["input_mean"][:-2]), np.asarray(norm["input_std"][:-2])
        field = {"Z": "latent", "O": "observation", "H": "history", "P": "physical"}[representation]
        query = candidate_records(representation, states, lambda _, state: np.asarray(state[field], dtype=np.float64))
        reference = candidate_records(representation, development_states,
            lambda r, state: state_vector(r, state["seed"], state["decision_index"], arrays, index, physical_lookup_data))
        within = nearest(query, mean, scale, q["q95"], exclude_same_state=True)
        cross = nearest(query, mean, scale, q["q95"], reference=reference, exclude_same_state=True)
        alias_summary[representation] = {"within_untouched_confirmation": summary(within),
                                         "confirmation_to_frozen_development_reference": summary(cross)}
    write_json(OUT / "confirmation_aliasing_summary.json", {"method": "exact float32 same-action nearest states with TRAIN-behavior normalization; both within new confirmation and against frozen development candidate references",
        "reference_Q95_N": q["q95"], "representations": alias_summary})
    print("G4 untouched confirmation evaluation complete; no retraining or recalibration", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("behavior", "candidates", "evaluate"), required=True)
    args = parser.parse_args()
    {"behavior": behavior, "candidates": candidate_pass, "evaluate": evaluate}[args.stage]()


if __name__ == "__main__":
    main()
