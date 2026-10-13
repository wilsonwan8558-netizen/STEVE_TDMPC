"""Freeze G9.5 before 103000-series access; collect real planner-fork confirmation."""

from __future__ import annotations

import argparse
import csv
import json
import sys

import numpy as np
import torch

from g95_common import (HERE, OUT, G9, CONFIRM, F_LIM, PLANS_PER_STATE,
                        E1Representation, captured_plan, choose_plans, fork_two,
                        cuda_gate, json_new, sha, source_gate)
from g95_collect import validate_prepared
from g6_collect import setup
from collect import evaluate_one, EPISODE_FIELDS, STEP_FIELDS, check_scene
from protocol import choose_states
from restore_audit import signature, compare, g2_signature
from tdmpc2.common import set_seed
import g5_confirm

EPISODES = OUT / "confirmation_behavior_episodes.csv"
STEPS = OUT / "confirmation_behavior_steps.csv"
FROZEN_SCRIPTS = ("g95_common.py", "g95_prepare.py", "g95_collect.py", "g95_dataset.py",
                  "g95_train.py", "g95_eval.py", "g95_confirmation_data.py", "g95_confirmation_eval.py")
FROZEN_FILES = ("metadata.json", "g9_model_audit.json", "planner_candidate_protocol.json",
                "two_step_fork_audit.json", "dataset_manifest.json", "dataset_split.json",
                "development_parent_replay_audit.json", "development_planner_dataset.npz",
                "development_dataset_summary.json", "model_training_manifest.json",
                "development_plan_metrics.csv", "development_state_metrics.csv",
                "training_seed_robustness.json")


def freeze_value():
    source = validate_prepared()
    if not all((HERE / name).is_file() for name in FROZEN_SCRIPTS):
        raise RuntimeError("G9.5 confirmation scripts incomplete before freeze")
    if not all((OUT / name).is_file() for name in FROZEN_FILES):
        raise RuntimeError("G9.5 development artifacts incomplete before freeze")
    parent = json.loads((OUT / "development_parent_replay_audit.json").read_text())
    if parent["status"] != "PASS" or parent["states"] != 800 or parent["planner_candidate_forks"] != 7200:
        raise RuntimeError("G9.5 development SOFA parent audit failed")
    model_manifest = json.loads((OUT / "model_training_manifest.json").read_text())
    if model_manifest["status"] != "THREE_FIXED_REPLICAS_PER_MODEL_BEFORE_DEVELOPMENT_TEST":
        raise RuntimeError("G9.5 model replicas incomplete")
    for kind in ("M0_behavior", "M1_planner_conditioned"):
        if set(model_manifest["models"][kind]) != {"260312", "260313", "260314"}:
            raise RuntimeError(f"G9.5 {kind} training seed set changed")
    checkpoints = {}
    for kind, seeds in model_manifest["models"].items():
        for seed, record in seeds.items():
            path = OUT / "models" / kind / f"seed_{seed}.pt"
            if sha(path) != record["checkpoint_sha256"]:
                raise RuntimeError(f"G9.5 model checkpoint changed {kind}/{seed}")
            checkpoints[f"{kind}/{seed}"] = record["checkpoint_sha256"]
    return {"status": "G95_ALL_MODELS_AND_EVALUATION_RULES_FROZEN_BEFORE_103000_CONFIRMATION",
        "confirmation_seeds_ordered": list(CONFIRM),
        "one_reused_scene_per_ordered_pass": True,
        "controller_checkpoint_sha256": source["checkpoint_sha256"],
        "planner_source_sha256": source["planner_source_sha256"],
        "G9_E1_mapper_sha256": source["g9_E1_mapper_sha256"],
        "G9_E1_H2_normalization_sha256": source["g9_risk_normalization_sha256"],
        "G9_old_E1_H2_checkpoint_sha256": source["g9_model_sha256"],
        "F_lim_N": F_LIM, "horizon": 2, "quantiles": [.50, .90, .95, .99],
        "candidate_count_per_state": PLANS_PER_STATE,
        "state_selection": "frozen G3 choose_states four per episode; progress/prior-force only; no candidate outcome labels",
        "prospective_plan_label": "original captured first and second action executed open-loop in copy-on-write true Generic-SOFA child; no future behavior substitution",
        "censored_rule": "one-step terminal has no F2/R2 and is excluded from complete-target metrics",
        "Q95_threshold": "validation-frozen raw Q95 plus nonnegative validation residual correction, accepted iff <= F_lim",
        "ranking_rule": "per-state Spearman, pairwise ordering excluding equal true risks, top3 high/low overlap",
        "no_retraining_recalibration_or_threshold_change_on_confirmation": True,
        "model_checkpoint_sha256": checkpoints,
        "source_sha256": {name: sha(HERE / name) for name in FROZEN_SCRIPTS},
        "development_artifact_sha256": {name: sha(OUT / name) for name in FROZEN_FILES}}


def freeze():
    cuda_gate()
    if any(path.exists() for path in (OUT / "pre_confirmation_freeze.json", EPISODES, STEPS,
                                      OUT / "confirmation_seed_manifest.json")):
        raise FileExistsError("G9.5 confirmation or freeze already exists")
    json_new(OUT / "pre_confirmation_freeze.json", freeze_value())
    print("G9.5 pre-confirmation freeze saved; 103000-103029 unopened", flush=True)


def verify_freeze():
    cuda_gate()
    frozen = json.loads((OUT / "pre_confirmation_freeze.json").read_text())
    if frozen != json.loads(json.dumps(freeze_value())):
        raise RuntimeError("G9.5 frozen source/model/development evidence changed after freeze")
    return frozen


def behavior():
    frozen = verify_freeze()
    if EPISODES.exists() or STEPS.exists() or (OUT / "confirmation_seed_manifest.json").exists():
        raise FileExistsError("G9.5 confirmation behavior already opened; no direct skip/resume")
    json_new(OUT / "confirmation_seed_manifest.json", {"seeds": list(CONFIRM),
        "pre_confirmation_freeze_sha256": sha(OUT / "pre_confirmation_freeze.json"),
        "checkpoint_sha256": frozen["controller_checkpoint_sha256"],
        "scene_lifecycle": "one reused Generic scene, ordered 103000-103029 resets",
        "python": sys.executable, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "cuda_device_index": 0, "cuda_device_name": torch.cuda.get_device_name(0)})
    env, agent, monitor = setup(list(CONFIRM))
    try:
        with EPISODES.open("x", newline="") as episode_stream, STEPS.open("x", newline="") as step_stream:
            ew = csv.DictWriter(episode_stream, fieldnames=EPISODE_FIELDS)
            sw = csv.DictWriter(step_stream, fieldnames=STEP_FIELDS)
            ew.writeheader()
            sw.writeheader()
            for index, seed in enumerate(CONFIRM):
                episode, steps = evaluate_one(env, agent, seed, index,
                                              g5_confirm.EXPECTED_CHECKPOINT_SHA, monitor)
                if episode["simulation_error"]:
                    raise RuntimeError(f"G9.5 confirmation simulation error {seed}")
                ew.writerow(episode)
                sw.writerows(steps)
                episode_stream.flush()
                step_stream.flush()
                print(f"G9.5 confirmation behavior seed={seed} steps={len(steps)}", flush=True)
    finally:
        env.close()


def confirmation_rows():
    manifest = json.loads((OUT / "confirmation_seed_manifest.json").read_text())
    if manifest["seeds"] != list(CONFIRM) or manifest["pre_confirmation_freeze_sha256"] != sha(OUT / "pre_confirmation_freeze.json"):
        raise RuntimeError("G9.5 confirmation seed manifest changed")
    with EPISODES.open(newline="") as stream:
        episodes = list(csv.DictReader(stream))
    if [int(row["seed"]) for row in episodes] != list(CONFIRM):
        raise RuntimeError("G9.5 confirmation episodes incomplete/out of order")
    by_seed = {seed: [] for seed in CONFIRM}
    with STEPS.open(newline="") as stream:
        for row in csv.DictReader(stream):
            seed, d = int(row["seed"]), int(row["step_index"])
            if seed not in by_seed or d != len(by_seed[seed]):
                raise RuntimeError(f"G9.5 confirmation step indexing changed {seed}/{d}")
            by_seed[seed].append(row)
    for episode in episodes:
        if len(by_seed[int(episode["seed"])]) != int(episode["episode_length"]):
            raise RuntimeError("G9.5 confirmation episode/step length mismatch")
    return episodes, by_seed


def select():
    verify_freeze()
    if (OUT / "confirmation_state_selection.json").exists():
        raise FileExistsError(OUT / "confirmation_state_selection.json")
    _, by_seed = confirmation_rows()
    states = choose_states(list(CONFIRM), by_seed, "confirmation")
    if (len(states) != 120 or any(s["decision_index"] < 2 or
            s["decision_index"]+2 > s["episode_length"] for s in states)):
        raise RuntimeError("G9.5 confirmation E1/H2 selected state coverage invalid")
    json_new(OUT / "confirmation_state_selection.json", {"status": "SELECTED_WITHOUT_PLANNER_FORCE_OUTCOMES",
        "source_behavior_episodes_sha256": sha(EPISODES),
        "source_behavior_steps_sha256": sha(STEPS),
        "rule": "frozen G3 four-per-episode progress/prior-force strata, deterministic seed order",
        "states": states, "count": 120,
        "candidate_outcome_labels_used": False})
    print("G9.5 confirmation selected 120 planner states without candidate outcomes", flush=True)


def plans():
    verify_freeze()
    selection = json.loads((OUT / "confirmation_state_selection.json").read_text())
    if (selection["count"] != 120 or selection["source_behavior_steps_sha256"] != sha(STEPS) or
            selection["source_behavior_episodes_sha256"] != sha(EPISODES)):
        raise RuntimeError("G9.5 confirmation selected states changed")
    if (OUT / "confirmation_parent_replay_audit.json").exists():
        raise FileExistsError(OUT / "confirmation_parent_replay_audit.json")
    _, by_seed = confirmation_rows()
    selected = {(int(s["seed"]), int(s["decision_index"])): s for s in selection["states"]}
    if len(selected) != 120:
        raise RuntimeError("G9.5 confirmation state keys duplicate")
    mapper = E1Representation()
    env, agent, monitor = setup(list(CONFIRM))
    parent_steps = state_count = fork_count = censored = 0
    differences = {}
    try:
        for seed in CONFIRM:
            set_seed(seed)
            env.action_space.seed(seed)
            observation, _ = env.reset(seed=seed)
            check_scene(env)
            if not hasattr(monitor, "_g95_confirmation_added"):
                env._simulation.root.addObject(monitor)
                monitor._g95_confirmation_added = True
            rows = by_seed[seed]
            obs_history, action_history = [], []
            terminated = truncated = False
            decision = 0
            while not (terminated or truncated):
                if decision >= len(rows) or not np.array_equal(observation,
                    np.asarray(json.loads(rows[decision]["decision_observation_json"]), dtype=np.float32)):
                    raise RuntimeError(f"G9.5 confirmation parent observation mismatch {seed}/{decision}")
                obs_history.append(observation.copy())
                state = selected.get((seed, decision))
                if state:
                    action, final_actions, scores, mean = captured_plan(agent, observation, decision == 0)
                else:
                    action = agent.act(observation, first_step=decision == 0, eval_mode=True)
                if not np.array_equal(action,
                    np.asarray(json.loads(rows[decision]["selected_action_json"]), dtype=np.float32)):
                    raise RuntimeError(f"G9.5 passive capture changed confirmation behavior action {seed}/{decision}")
                if state:
                    history_obs = np.asarray(obs_history[decision-2:decision+1], dtype=np.float32)
                    history_action = np.asarray(action_history[decision-2:decision], dtype=np.float32)
                    if history_obs.shape != (3, 14) or history_action.shape != (2, 2):
                        raise RuntimeError("G9.5 confirmation pre-action history unavailable")
                    for j in range(3):
                        if not np.array_equal(history_obs[j], np.asarray(json.loads(rows[decision-2+j]["decision_observation_json"]), dtype=np.float32)):
                            raise RuntimeError("G9.5 confirmation pre-action observation history mismatch")
                    for j in range(2):
                        if not np.array_equal(history_action[j], np.asarray(json.loads(rows[decision-2+j]["selected_action_json"]), dtype=np.float32)):
                            raise RuntimeError("G9.5 confirmation pre-action action history mismatch")
                    path = OUT / "confirmation_states" / f"state_{int(state['state_id']):04d}.json"
                    planner_candidates = choose_plans(final_actions, scores, mean)
                    if path.exists():
                        prior = json.loads(path.read_text())
                        if ((prior["state_id"], prior["seed"], prior["decision_index"]) !=
                                (state["state_id"], seed, decision) or
                                len(prior["plans"]) != PLANS_PER_STATE or
                                any(not np.array_equal(np.asarray(prior["plans"][j]["actions"], dtype=np.float32),
                                                           np.asarray(planner_candidates[j]["actions"], dtype=np.float32))
                                    for j in range(PLANS_PER_STATE))):
                            raise RuntimeError(f"G9.5 confirmation replay-resume candidate mismatch {path}")
                        item = prior
                        print(f"G9.5 confirmation replay-resume state {state['state_id']}", flush=True)
                    else:
                        with torch.inference_mode():
                            latent = agent.model.encode(agent._tensor_observation(observation))[0].cpu().numpy()
                        estimated = mapper.estimate(history_obs, history_action)
                        outcomes = []
                        for candidate in planner_candidates:
                            outcomes.append({**candidate,
                                "fork": fork_two(env, monitor, np.asarray(candidate["actions"], dtype=np.float32))})
                        item = {"group": "confirmation", "state_id": state["state_id"],
                            "seed": seed, "decision_index": decision, "slot": state["slot"],
                            "episode_progress": state["episode_progress"],
                            "previous_executed_force_N": state["previous_executed_force_N"],
                            "history_observation": history_obs.tolist(),
                            "history_action": history_action.tolist(),
                            "current_observation": observation.tolist(),
                            "current_latent": latent.tolist(),
                            "E1_current_physical_estimate": estimated.tolist(),
                            "planner_mean_action": action.tolist(),
                            "plans": outcomes,
                            "future_behavior_actions_used_as_candidate": False}
                        json_new(path, item)
                    state_count += 1
                    fork_count += len(item["plans"])
                    censored += sum(not p["fork"]["complete_two_intervals"] for p in item["plans"])
                    if state_count % 20 == 0:
                        print(f"G9.5 confirmation planner states {state_count}/120 forks={fork_count}", flush=True)
                parent = signature(env, monitor, action)
                compare(parent, g2_signature(rows[decision]), differences,
                        f"G9.5 confirmation parent {seed}/{decision}")
                observation = np.asarray(parent["observation"], dtype=np.float32)
                terminated, truncated = parent["terminated"], parent["truncated"]
                action_history.append(action.copy())
                decision += 1
                parent_steps += 1
            if decision != len(rows):
                raise RuntimeError(f"G9.5 confirmation episode length changed {seed}")
            print(f"G9.5 confirmation parent seed={seed} steps={decision}", flush=True)
    finally:
        env.close()
    expected = sum(len(by_seed[seed]) for seed in CONFIRM)
    if (parent_steps != expected or state_count != 120 or fork_count != 120*PLANS_PER_STATE or
            len(list((OUT / "confirmation_states").glob("state_*.json"))) != 120):
        raise RuntimeError("G9.5 confirmation parent/plan pass incomplete")
    json_new(OUT / "confirmation_parent_replay_audit.json", {"status": "PASS",
        "ordered_reused_scene_seeds": list(CONFIRM),
        "parent_steps_compared": parent_steps, "planner_states": state_count,
        "planner_candidate_forks": fork_count, "complete_R2_forks": fork_count-censored,
        "censored_one_step_terminal_forks": censored,
        "future_behavior_action_pairs_not_used_as_candidate_labels": True,
        "max_abs_difference": differences, "tolerance": 1e-8})
    print(f"G9.5 confirmation true-plan forks PASS: {state_count} states, {fork_count} plans", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("freeze", "behavior", "select", "plans"), required=True)
    stage = parser.parse_args().stage
    {"freeze": freeze, "behavior": behavior, "select": select, "plans": plans}[stage]()


if __name__ == "__main__":
    main()
