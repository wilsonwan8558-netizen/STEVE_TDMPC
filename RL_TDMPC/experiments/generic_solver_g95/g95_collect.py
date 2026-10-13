"""Audit then collect exact TD-MPC2 planner two-action true-SOFA forks."""

from __future__ import annotations

import argparse
import csv
import json
import sys

import numpy as np
import torch

from g95_common import (OUT, G2, G3, CONFIRM, PLANS_PER_STATE, E1Representation,
                        captured_plan, choose_plans, compare_two, fork_two,
                        json_new, sha, source_gate)
from g6_collect import setup
from restore_audit import signature, compare, g2_signature
from collect import check_scene
from tdmpc2.common import set_seed

AUDIT_SEEDS = tuple(range(90000, 90004))
FULL_SEEDS = tuple(range(90000, 90200))


def archived():
    with (G2 / "step_level_force.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_seed = {seed: [] for seed in FULL_SEEDS}
    for row in rows:
        seed, decision = int(row["seed"]), int(row["step_index"])
        if seed not in by_seed or decision != len(by_seed[seed]):
            raise RuntimeError(f"G9.5 archived behavior indexing changed {seed}/{decision}")
        by_seed[seed].append(row)
    if len(rows) != 15500 or any(not episode for episode in by_seed.values()):
        raise RuntimeError("G9.5 archived G2 200-episode behavior incomplete")
    with np.load(G3 / "behavior_features.npz") as z:
        cached_seed = z["seed"].copy()
        cached_decision = z["decision_index"].copy()
        cached_latent = z["latent"].copy()
    latent_lookup = {(int(s), int(d)): cached_latent[i]
                     for i, (s, d) in enumerate(zip(cached_seed, cached_decision))}
    if len(latent_lookup) != 15500:
        raise RuntimeError("G9.5 G3 frozen latent lookup incomplete")
    return by_seed, latent_lookup


def selected_states():
    split = json.loads((OUT / "dataset_split.json").read_text())
    mapping = {}
    for group, rows in split["selected_development_states"].items():
        for state in rows:
            key = (int(state["seed"]), int(state["decision_index"]))
            if key in mapping:
                raise RuntimeError("G9.5 selected state duplicate")
            mapping[key] = state
    if len(mapping) != 800:
        raise RuntimeError("G9.5 expected 800 selected development states")
    return mapping


def validate_prepared():
    source = source_gate()
    meta = json.loads((OUT / "metadata.json").read_text())
    if source != meta["frozen_sources"] or meta["horizon"] != 2 or meta["confirm_seeds_declared_but_unopened"] != list(CONFIRM):
        raise RuntimeError("G9.5 prepared frozen source/protocol changed")
    protocol = json.loads((OUT / "planner_candidate_protocol.json").read_text())
    if (protocol["planner_source_sha256"] != source["planner_source_sha256"] or
            protocol["candidate_count_per_state"] != PLANS_PER_STATE or
            protocol["subsample_without_force_labels"] !=
            "two highest model-value, two nearest median rank, two lowest, two greedy max-min diverse in flattened first-two-action 4D normalized space; ties by original population index; plus exact optimized mean plan"):
        raise RuntimeError("G9.5 planner candidate protocol changed")
    return source


def record_state(env, agent, monitor, mapper, state, observation, obs_history, action_history,
                 rows, latent_lookup, action, final_actions, scores, mean, differences, audit_only):
    seed, decision = int(state["seed"]), int(state["decision_index"])
    history_obs = np.asarray(obs_history[decision-2:decision+1], dtype=np.float32)
    history_action = np.asarray(action_history[decision-2:decision], dtype=np.float32)
    if history_obs.shape != (3, 14) or history_action.shape != (2, 2):
        raise RuntimeError(f"G9.5 E1 history missing {seed}/{decision}")
    for j in range(3):
        archived_obs = np.asarray(json.loads(rows[decision-2+j]["decision_observation_json"]), dtype=np.float32)
        if not np.array_equal(history_obs[j], archived_obs):
            raise RuntimeError(f"G9.5 pre-action history observation mismatch {seed}/{decision}")
    for j in range(2):
        archived_action = np.asarray(json.loads(rows[decision-2+j]["selected_action_json"]), dtype=np.float32)
        if not np.array_equal(history_action[j], archived_action):
            raise RuntimeError(f"G9.5 pre-action history action mismatch {seed}/{decision}")
    with torch.inference_mode():
        latent = agent.model.encode(agent._tensor_observation(observation))[0].cpu().numpy()
    if not np.array_equal(latent, latent_lookup[(seed, decision)]):
        raise RuntimeError(f"G9.5 frozen TD-MPC2 latent mismatch {seed}/{decision}")
    estimated = mapper.estimate(history_obs, history_action)
    plans = choose_plans(final_actions, scores, mean)
    behavior_pair = np.asarray([json.loads(rows[decision+j]["selected_action_json"]) for j in (0, 1)], dtype=np.float32)
    if not np.array_equal(behavior_pair[0], action):
        raise RuntimeError(f"G9.5 behavior first-action mismatch {seed}/{decision}")
    archived_fork = fork_two(env, monitor, behavior_pair)
    if not archived_fork["complete_two_intervals"]:
        raise RuntimeError(f"G9.5 archived behavior pair unexpectedly censored {seed}/{decision}")
    compare(archived_fork["outcomes"][0], g2_signature(rows[decision]), differences,
            f"G9.5 archived behavior fork first {seed}/{decision}")
    compare(archived_fork["outcomes"][1], g2_signature(rows[decision+1]), differences,
            f"G9.5 archived behavior fork second {seed}/{decision}")
    if audit_only:
        repeat = fork_two(env, monitor, np.asarray(plans[-1]["actions"], dtype=np.float32))
        first = fork_two(env, monitor, np.asarray(plans[-1]["actions"], dtype=np.float32))
        compare_two(first, repeat, differences, f"G9.5 identical two-action repeat {seed}/{decision}")
        return {"state_id": state["state_id"], "seed": seed, "decision_index": decision,
                "planner_mean_action": action.tolist(), "candidate_count": len(plans),
                "repeat_two_step_complete": first["complete_two_intervals"],
                "behavior_pair_R2_N": archived_fork["R2_true_N"]}
    output = []
    for item in plans:
        outcome = fork_two(env, monitor, np.asarray(item["actions"], dtype=np.float32))
        output.append({**item, "fork": outcome})
    return {"group": state["group"], "state_id": state["state_id"],
            "seed": seed, "decision_index": decision, "slot": state["slot"],
            "episode_progress": state["episode_progress"],
            "previous_executed_force_N": state["previous_executed_force_N"],
            "history_observation": history_obs.tolist(), "history_action": history_action.tolist(),
            "current_observation": observation.tolist(), "current_latent": latent.tolist(),
            "E1_current_physical_estimate": estimated.tolist(),
            "planner_mean_action": action.tolist(),
            "planner_final_population_count": len(scores),
            "planner_final_score_min": float(np.min(scores)),
            "planner_final_score_median": float(np.median(scores)),
            "planner_final_score_max": float(np.max(scores)),
            "archived_behavior_two_step_R2_N": archived_fork["R2_true_N"],
            "plans": output}


def run(mode):
    validate_prepared()
    if mode == "audit":
        report_path = OUT / "two_step_fork_audit.json"
        if report_path.exists():
            raise FileExistsError(report_path)
        seeds = AUDIT_SEEDS
    else:
        audit = json.loads((OUT / "two_step_fork_audit.json").read_text())
        if audit["status"] != "PASS" or audit["states_checked"] != 16:
            raise RuntimeError("G9.5 two-step fork audit must pass before development labels")
        report_path = OUT / "development_parent_replay_audit.json"
        if report_path.exists():
            raise FileExistsError(report_path)
        seeds = FULL_SEEDS
    by_seed, latent_lookup = archived()
    selected = selected_states()
    mapper = E1Representation()
    env, agent, monitor = setup(list(seeds))
    differences, audit_states = {}, []
    parent_steps = state_count = fork_count = censored = 0
    try:
        for seed in seeds:
            set_seed(seed)
            env.action_space.seed(seed)
            observation, _ = env.reset(seed=seed)
            check_scene(env)
            if not hasattr(monitor, "_g95_added"):
                env._simulation.root.addObject(monitor)
                monitor._g95_added = True
            rows = by_seed[seed]
            obs_history, action_history = [], []
            terminated = truncated = False
            decision = 0
            while not (terminated or truncated):
                if decision >= len(rows) or not np.array_equal(observation,
                    np.asarray(json.loads(rows[decision]["decision_observation_json"]), dtype=np.float32)):
                    raise RuntimeError(f"G9.5 reused-scene parent observation mismatch {seed}/{decision}")
                obs_history.append(observation.copy())
                state = selected.get((seed, decision))
                if state is not None:
                    action, final_actions, scores, mean = captured_plan(agent, observation, decision == 0)
                else:
                    action = agent.act(observation, first_step=decision == 0, eval_mode=True)
                if not np.array_equal(action,
                    np.asarray(json.loads(rows[decision]["selected_action_json"]), dtype=np.float32)):
                    raise RuntimeError(f"G9.5 passive candidate capture changed nominal action {seed}/{decision}")
                if state is not None:
                    if mode == "audit":
                        item = record_state(env, agent, monitor, mapper, state, observation,
                                            obs_history, action_history, rows, latent_lookup,
                                            action, final_actions, scores, mean, differences, True)
                        audit_states.append(item)
                        state_count += 1
                    else:
                        path = OUT / "development_states" / f"state_{state['group']}_{int(state['state_id']):04d}.json"
                        if path.exists():
                            prior = json.loads(path.read_text())
                            if ((prior["group"], prior["state_id"], prior["seed"], prior["decision_index"]) !=
                                    (state["group"], state["state_id"], seed, decision) or
                                    len(prior["plans"]) != PLANS_PER_STATE or
                                    not np.array_equal(np.asarray(prior["planner_mean_action"], dtype=np.float32), action)):
                                raise RuntimeError(f"G9.5 resumed state corrupt {path}")
                            item = prior
                            print(f"G9.5 replay-resume state {state['group']}/{state['state_id']}", flush=True)
                        else:
                            item = record_state(env, agent, monitor, mapper, state, observation,
                                                obs_history, action_history, rows, latent_lookup,
                                                action, final_actions, scores, mean, differences, False)
                            json_new(path, item)
                        state_count += 1
                        fork_count += len(item["plans"])
                        censored += sum(not p["fork"]["complete_two_intervals"] for p in item["plans"])
                        if state_count % 20 == 0:
                            print(f"G9.5 development planner states {state_count}/800, forks={fork_count}, censored={censored}", flush=True)
                parent = signature(env, monitor, action)
                compare(parent, g2_signature(rows[decision]), differences,
                        f"G9.5 reused-scene parent {seed}/{decision}")
                observation = np.asarray(parent["observation"], dtype=np.float32)
                terminated, truncated = parent["terminated"], parent["truncated"]
                action_history.append(action.copy())
                decision += 1
                parent_steps += 1
            if decision != len(rows):
                raise RuntimeError(f"G9.5 parent episode length changed {seed}")
            print(f"G9.5 {mode} parent seed={seed} steps={decision}", flush=True)
    finally:
        env.close()
    expected_steps = sum(len(by_seed[seed]) for seed in seeds)
    if parent_steps != expected_steps or state_count != 4*len(seeds):
        raise RuntimeError(f"G9.5 {mode} incomplete parent steps={parent_steps}/{expected_steps}, states={state_count}")
    if mode == "audit":
        json_new(report_path, {"status": "PASS", "ordered_reused_scene_seeds": seeds,
            "parent_steps_compared": parent_steps, "states_checked": state_count,
            "identical_two_action_repeats": state_count,
            "archived_behavior_pairs_compared": state_count,
            "all_parent_steps_match_archived_G2": True,
            "feature_history_aligned_to_preaction_decision": True,
            "passive_planner_capture_changes_no_nominal_actions": True,
            "max_abs_difference": differences, "states": audit_states})
    else:
        if fork_count != 800*PLANS_PER_STATE or len(list((OUT / "development_states").glob("state_*.json"))) != 800:
            raise RuntimeError("G9.5 planner development state/plan count incomplete")
        json_new(report_path, {"status": "PASS", "ordered_reused_scene_seeds": seeds,
            "parent_steps_compared": parent_steps, "states": state_count,
            "planner_candidate_forks": fork_count, "censored_one_step_terminal_forks": censored,
            "complete_R2_forks": fork_count-censored,
            "max_abs_difference": differences, "tolerance": 1e-8})
    print(f"G9.5 {mode} PASS parent steps={parent_steps} selected states={state_count}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("audit", "development"), required=True)
    run(parser.parse_args().stage)


if __name__ == "__main__":
    main()
