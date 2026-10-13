"""Post-freeze, untouched 92000-92029 behavior and one-step confirmation."""

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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(RL / "experiments/generic_solver_g2"))
sys.path.insert(0, str(RL))
sys.path.insert(0, str(RL.parent))

from envs.steve_env import make_steve_env  # noqa: E402
from tdmpc2.agent import TDMPC2Agent  # noqa: E402
from tdmpc2.common import load_torch_checkpoint, set_seed  # noqa: E402
from train import build_agent_config  # noqa: E402
from substep_monitor import DistalForceMonitor  # noqa: E402
from collect import check_scene, evaluate_one, STEP_FIELDS, EPISODE_FIELDS  # noqa: E402
from prepare import G1B, OUT, CONFIG, EXPECTED_CHECKPOINT_SHA, sha, check_force_source  # noqa: E402
from protocol import choose_states, generate_candidates  # noqa: E402
from restore_audit import fork_step, signature, compare, g2_signature  # noqa: E402
from g3_evaluate import MODELS, load_predictor, predict, regression, classification, state_metrics, aggregate_candidate, quantiles, csv_new, json_new  # noqa: E402

SEEDS = list(range(92000, 92030))


def frozen():
    protocol = json.loads((OUT / "confirmation_seed_protocol.json").read_text())
    if protocol["seeds"] != SEEDS or protocol["expected_candidate_labels"] != 2520:
        raise RuntimeError("Predeclared confirmation seed/action protocol changed")
    if not (OUT / "bootstrap_summary.json").exists():
        raise RuntimeError("Development-test analysis not complete; confirmation remains sealed")
    manifest = json.loads((OUT / "frozen_model_manifest.json").read_text())
    if manifest["confirmation_seeds_opened_before_freeze"] or manifest["primary_selected_model"] not in MODELS:
        raise RuntimeError("Invalid predictor freeze")
    for field, filename in (("model_selection_protocol_sha256", "model_selection_protocol.json"),
                            ("confirmation_seed_protocol_sha256", "confirmation_seed_protocol.json"),
                            ("development_split_sha256", "development_split.json"),
                            ("candidate_action_protocol_sha256", "candidate_action_protocol.json")):
        if sha(OUT / filename) != manifest[field]:
            raise RuntimeError(f"Frozen G3 protocol changed: {filename}")
    for name in MODELS:
        if sha(OUT / "models" / name / "model.pt") != manifest["checkpoint_sha256"][name]:
            raise RuntimeError("Frozen model checkpoint changed")
        if sha(OUT / "models" / name / "normalization.json") != manifest["normalization_sha256"][name]:
            raise RuntimeError("Frozen normalization changed")
    return manifest


def setup():
    check_force_source()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Frozen RTX 5090 CUDA_VISIBLE_DEVICES=0 required")
    checkpoint_path = Path(json.loads((G1B / "selected_checkpoint.json").read_text())["path"])
    if sha(checkpoint_path) != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("Frozen navigation checkpoint changed")
    checkpoint = load_torch_checkpoint(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    if config != yaml.safe_load(CONFIG.read_text()):
        raise RuntimeError("Frozen environment config changed")
    set_seed(SEEDS[0])
    env = make_steve_env(config["environment"])
    agent = TDMPC2Agent(14, 2, build_agent_config(config), episode_length=int(config["environment"]["max_episode_steps"]), device=torch.device("cuda:0"))
    agent.load_state_dict(checkpoint["agent"], load_optimizers=False)
    monitor = DistalForceMonitor(env)
    return env, agent, monitor


def behavior():
    frozen()
    episode_path = OUT / "confirmation_behavior_episodes.csv"
    step_path = OUT / "confirmation_behavior_steps.csv"
    manifest_path = OUT / "confirmation_seed_manifest.json"
    for path in (episode_path, step_path, manifest_path):
        if path.exists():
            raise FileExistsError(path)
    json_new(manifest_path, {"seeds": SEEDS, "episode_count": 30, "predeclared_before_training": True,
                             "confirmation_seed_protocol_sha256": sha(OUT / "confirmation_seed_protocol.json"),
                             "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
                             "scene_lifecycle": "one Generic scene, ordered seed resets 92000-92029",
                             "models_frozen_before_open": True,
                             "frozen_model_manifest_sha256": sha(OUT / "frozen_model_manifest.json")})
    env, agent, monitor = setup()
    try:
        with episode_path.open("w", newline="") as episode_stream, step_path.open("w", newline="") as step_stream:
            ew, sw = csv.DictWriter(episode_stream, EPISODE_FIELDS), csv.DictWriter(step_stream, STEP_FIELDS)
            ew.writeheader(); sw.writeheader()
            for index, seed in enumerate(SEEDS):
                episode, steps = evaluate_one(env, agent, seed, index, EXPECTED_CHECKPOINT_SHA, monitor)
                if episode["simulation_error"]:
                    raise RuntimeError(f"Confirmation simulation error {seed}")
                ew.writerow(episode); sw.writerows(steps)
                episode_stream.flush(); step_stream.flush()
                print(f"confirmation behavior seed={seed} length={len(steps)}", flush=True)
    finally:
        env.close()
    print("Confirmation behavior pass complete", flush=True)


def candidates():
    frozen()
    if not (OUT / "confirmation_seed_manifest.json").exists():
        raise RuntimeError("Confirmation behavior pass has not started")
    if json.loads((OUT / "confirmation_seed_manifest.json").read_text())["frozen_model_manifest_sha256"] != sha(OUT / "frozen_model_manifest.json"):
        raise RuntimeError("Frozen manifest changed after confirmation started")
    with (OUT / "confirmation_behavior_steps.csv").open(newline="") as stream:
        rows_by_seed = {}
        for row in csv.DictReader(stream):
            rows_by_seed.setdefault(int(row["seed"]), []).append(row)
    if set(rows_by_seed) != set(SEEDS):
        raise RuntimeError("Incomplete untouched behavior seed range")
    chosen = choose_states(SEEDS, rows_by_seed, "confirmation")
    selected = {(s["seed"], s["decision_index"]): s for s in chosen}
    if len(selected) != 120:
        raise RuntimeError("Expected 120 confirmation states")
    folder = OUT / "confirmation_candidates"
    folder.mkdir(parents=True, exist_ok=False)
    env, agent, monitor = setup()
    differences = {}
    steps_compared = 0
    try:
        for seed in SEEDS:
            set_seed(seed)
            env.action_space.seed(seed)
            observation, _ = env.reset(seed=seed)
            check_scene(env)
            if not hasattr(monitor, "_g3_added"):
                env._simulation.root.addObject(monitor)
                monitor._g3_added = True
            rows = rows_by_seed[seed]
            terminated = truncated = False
            d = 0
            while not (terminated or truncated):
                if d >= len(rows):
                    raise RuntimeError(f"Confirmation replay overrun {seed}")
                row = rows[d]
                if not np.array_equal(observation, np.asarray(json.loads(row["decision_observation_json"]), dtype=np.float32)):
                    raise RuntimeError(f"Confirmation replay observation differs {seed}/{d}")
                action = agent.act(observation, first_step=d == 0, eval_mode=True)
                if not np.array_equal(action, np.asarray(json.loads(row["selected_action_json"]), dtype=np.float32)):
                    raise RuntimeError(f"Confirmation replay behavior action differs {seed}/{d}")
                state = selected.get((seed, d))
                forked_behavior = None
                if state is not None:
                    with torch.inference_mode():
                        latent = agent.model.encode(agent._tensor_observation(observation))[0].cpu().numpy()
                    outcomes = []
                    for candidate in generate_candidates(action, state["state_id"], "confirmation"):
                        outcome = fork_step(env, monitor, np.asarray(candidate["action"], dtype=np.float32))
                        if candidate["candidate_id"] == 0:
                            forked_behavior = outcome
                            compare(outcome, g2_signature(row), differences, f"confirmation behavior fork {seed}/{d}")
                        outcomes.append({**candidate, "force_next_max_N": outcome["force_next_max_N"], "outcome": outcome})
                    path = folder / f"state_{state['state_id']:04d}.json"
                    json_new(path, {**state, "observation": observation.tolist(), "latent": latent.tolist(),
                                    "behavior_action": action.tolist(), "candidates": outcomes})
                    print(f"confirmation state {state['state_id']+1}/120 seed={seed} d={d}", flush=True)
                parent = signature(env, monitor, action)
                compare(parent, g2_signature(row), differences, f"confirmation parent {seed}/{d}")
                if forked_behavior is not None:
                    compare(forked_behavior, parent, differences, f"confirmation behavior parent {seed}/{d}")
                observation = np.asarray(parent["observation"], dtype=np.float32)
                terminated, truncated = parent["terminated"], parent["truncated"]
                d += 1
                steps_compared += 1
            if d != len(rows):
                raise RuntimeError(f"Confirmation replay episode length differs {seed}")
    finally:
        env.close()
    json_new(OUT / "confirmation_replay_audit.json", {"status": "PASS", "parent_steps_compared": steps_compared,
                                                        "candidate_states": 120, "candidate_evaluations": 2520,
                                                        "max_abs_difference": differences})


def evaluate():
    manifest = frozen()
    if json.loads((OUT / "confirmation_replay_audit.json").read_text())["status"] != "PASS":
        raise RuntimeError("Untouched confirmation replay gate failed")
    paths = sorted((OUT / "confirmation_candidates").glob("state_*.json"))
    if len(paths) != 120:
        raise RuntimeError("Incomplete confirmation candidates")
    states = [json.loads(path.read_text()) for path in paths]
    x = np.asarray([np.r_[state["latent"], candidate["action"]] for state in states for candidate in state["candidates"]], dtype=float)
    y = np.asarray([candidate["force_next_max_N"] for state in states for candidate in state["candidates"]], dtype=float)
    q = quantiles()
    result_rows, state_rows, summary = [], [], {}
    for name in MODELS:
        p, seconds = predict(load_predictor(name, torch.device("cuda:0")), x, torch.device("cuda:0"))
        metrics = [{"model": name, **state_metrics(state, p[i*21:(i+1)*21])} for i, state in enumerate(states)]
        state_rows.extend(metrics)
        for i, state in enumerate(states):
            for j, candidate in enumerate(state["candidates"]):
                k = i*21+j
                result_rows.append({"model": name, "seed": state["seed"], "decision_index": state["decision_index"],
                                    "state_id": state["state_id"], "candidate_id": j, "candidate_kind": candidate["kind"],
                                    "action_translation": candidate["action"][0], "action_rotation": candidate["action"][1],
                                    "force_next_max_N": y[k], "predicted_force_next_max_N": p[k]})
        agg = aggregate_candidate({"truth": y, "prediction": p, "state_metrics": metrics, "inference_seconds": seconds})
        summary[name] = {**agg, "candidate_regression": regression(y, p),
                         "boundaries": {key: classification(y, p, level) for key, level in q.items()}}
    csv_new(OUT / "confirmation_candidate_results.csv", result_rows)
    csv_new(OUT / "confirmation_state_metrics.csv", state_rows)
    json_new(OUT / "confirmation_summary.json", {"primary_selected_model": manifest["primary_selected_model"],
                                                 "models_frozen_before_confirmation": True,
                                                 "seeds": [SEEDS[0], SEEDS[-1]], "state_count": 120, "candidate_count": 2520,
                                                 "nominal_N_references": q, "models": summary})
    print("Untouched confirmation evaluation complete; no retraining/recalibration", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("behavior", "candidates", "evaluate"), required=True)
    args = parser.parse_args()
    {"behavior": behavior, "candidates": candidates, "evaluate": evaluate}[args.stage]()


if __name__ == "__main__":
    main()
