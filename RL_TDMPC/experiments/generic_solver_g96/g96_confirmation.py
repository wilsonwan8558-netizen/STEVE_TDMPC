"""Freeze then collect/evaluate untouched 105000-series true planner-plan profiles."""

from __future__ import annotations

import argparse
import csv
import json
import sys

import numpy as np
import torch

from g96_common import (HERE, OUT, G95, SEEDS, F_LIM, assert_prepared,
                        frozen_sources, json_new, sha)
from g96_eval import evaluate_arrays, robustness, csv_new
from g95_common import (E1Representation, fork_two,
                        PLANS_PER_STATE)
from g96_protocol_v2 import (captured_plan_v2, canonical_plans, NEW_CONFIRMATION_SEEDS, ATOL, RTOL)
from g6_collect import setup
from collect import evaluate_one, EPISODE_FIELDS, STEP_FIELDS, check_scene
from protocol import choose_states
from restore_audit import signature, compare, g2_signature
from tdmpc2.common import set_seed

CONFIRM = NEW_CONFIRMATION_SEEDS
BASE = OUT / "confirmation_v2"
FREEZE = OUT / "pre_confirmation_freeze_v2.json"
EPISODES = BASE / "confirmation_behavior_episodes.csv"
STEPS = BASE / "confirmation_behavior_steps.csv"
SCRIPTS = ("g96_common.py", "g96_prepare.py", "g96_train.py", "g96_eval.py",
           "g96_confirmation.py", "g96_protocol_v2.py")
FROZEN_FILES = ("metadata.json", "g95_data_audit.json", "model_definition.json",
                "calibration_definition.json", "training_manifest.json",
                "calibration_result.json", "development_stepwise_coverage.csv",
                "development_plan_metrics.csv", "temporal_event_metrics.csv",
                "state_ranking_metrics.csv", "development_plan_results.csv",
                "training_seed_robustness.json")


def freeze_value():
    source = assert_prepared()
    amendment_path = OUT / "protocol_amendment.json"
    reaudit_path = OUT / "protocol_reaudit.json"
    old_freeze_path = OUT / "pre_confirmation_freeze.json"
    old_stop_path = OUT / "confirmation_protocol_stop.json"
    if any(not path.is_file() for path in (amendment_path, reaudit_path, old_freeze_path, old_stop_path)):
        raise RuntimeError("G9.6 v2 numerical repair or v1 stop evidence absent")
    amendment = json.loads(amendment_path.read_text())
    reaudit = json.loads(reaudit_path.read_text())
    if (amendment["status"] != "PREDECLARED_G96_EXECUTABLE_OPTIMIZED_MEAN_REPAIR_BEFORE_105000_CONFIRMATION" or
            amendment["atol"] != ATOL or amendment["rtol"] != RTOL or
            reaudit["status"] != "PASS_ZERO_AMBIGUOUS_CANDIDATES_EXACT_DEVELOPMENT_REPLAY" or
            reaudit["amendment_sha256"] != sha(amendment_path) or
            reaudit["planner_states_checked"] != 800 or
            reaudit["parent_steps_compared"] != 15500 or
            reaudit["amendment_introduced_ambiguous_mean_sampled_full_plan_matches"] != 0 or
            reaudit["previously_valid_candidate_identities_changed"] != 0 or
            reaudit["105000_series_accessed"] is not False):
        raise RuntimeError("G9.6 v2 development numerical protocol reaudit not passed")
    if any(not (HERE / name).is_file() for name in SCRIPTS):
        raise RuntimeError("G9.6 confirmation source incomplete")
    if any(not (OUT / name).is_file() for name in FROZEN_FILES):
        raise RuntimeError("G9.6 development/definition files incomplete")
    training = json.loads((OUT / "training_manifest.json").read_text())
    calibration = json.loads((OUT / "calibration_result.json").read_text())
    if (training["status"] != "THREE_FIXED_G96_REPLICAS_VALIDATION_SELECTED" or
            training["seeds"] != list(SEEDS) or
            calibration["status"] != "VALIDATION_ONLY_SHARED_STEP_CORRECTION_FROZEN_BEFORE_DEVELOPMENT_TEST" or
            calibration["training_manifest_sha256"] != sha(OUT / "training_manifest.json")):
        raise RuntimeError("G9.6 training/calibration not frozen from validation")
    checkpoint_hashes = {}
    for index, seed in enumerate(SEEDS, 1):
        path = OUT / "models" / f"stepwise_seed_{index}" / "model.pt"
        selection = path.with_name("selection.json")
        record = training["models"][str(seed)]
        if sha(path) != record["checkpoint_sha256"] or sha(selection) != record["selection_sha256"]:
            raise RuntimeError(f"G9.6 model changed {seed}")
        checkpoint_hashes[str(seed)] = sha(path)
    if source["scalar_baseline"] != json.loads((OUT / "models/scalar_baseline/manifest.json").read_text())["replicas"]:
        raise RuntimeError("G9.6 scalar baseline manifest changed")
    return {"status": "G96_PROFILE_SCALAR_AND_EVALUATION_FROZEN_BEFORE_105000_CONFIRMATION",
        "confirmation_seeds_ordered": list(CONFIRM),
        "one_reused_scene_per_ordered_pass": True,
        "F_lim_N": F_LIM, "H": 2, "quantiles": [.50, .90, .95, .99],
        "controller_checkpoint_sha256": source["controller_checkpoint_sha256"],
        "planner_source_sha256": source["planner_source_sha256"],
        "G95_dataset_sha256": source["g95_development_dataset_sha256"],
        "G95_candidate_protocol_sha256": sha(G95 / "planner_candidate_protocol.json"),
        "G95_scalar_baseline": source["scalar_baseline"],
        "G96_stepwise_checkpoint_sha256": checkpoint_hashes,
        "G9_input_normalization_sha256": source["g9_input_normalization_sha256"],
        "shared_delta1_N": calibration["shared_delta1_N"],
        "shared_delta2_N": calibration["shared_delta2_N"],
        "stepwise_decision": "both separately corrected Q95(F1) and Q95(F2) <= F_lim",
        "scalar_decision": "frozen G9.5 M1 corrected Q95(R2) <= F_lim",
        "state_selection": "frozen G3 progress/prior-force four per episode, no candidate force outcomes",
        "candidate_selection": "exact frozen G9.5 eight sampled plans plus clipped executable optimized mean ninth plan",
        "optimized_mean_canonicalization": "componentwise np.clip(raw optimized mean first two actions,-1,1)",
        "optimized_mean_executed_action_match": "np.allclose(canonical first action, executed action, rtol=0, atol=1e-6)",
        "optimized_mean_match_rtol": RTOL, "optimized_mean_match_atol": ATOL,
        "sampled_eight_candidate_semantics_changed": False,
        "protocol_amendment_sha256": sha(amendment_path),
        "protocol_reaudit_sha256": sha(reaudit_path),
        "original_v1_pre_confirmation_freeze_sha256": sha(old_freeze_path),
        "original_v1_confirmation_protocol_stop_sha256": sha(old_stop_path),
        "fork": "original planner first two actions executed open-loop true SOFA; no second-step replan",
        "no_retrain_recalibration_threshold_change_on_confirmation": True,
        "source_sha256": {name: sha(HERE / name) for name in SCRIPTS},
        "development_artifact_sha256": {name: sha(OUT / name) for name in FROZEN_FILES},
        "scalar_manifest_sha256": sha(OUT / "models/scalar_baseline/manifest.json")}


def freeze():
    if any(path.exists() for path in (FREEZE, EPISODES,
        STEPS, BASE / "confirmation_seed_manifest.json")):
        raise FileExistsError("G9.6 freeze or confirmation already exists")
    value = freeze_value()
    BASE.mkdir(exist_ok=False)
    json_new(FREEZE, value)
    print("G9.6 v2 pre-confirmation freeze saved; 105000-105029 unopened", flush=True)


def verify_freeze():
    frozen = json.loads(FREEZE.read_text())
    if frozen != json.loads(json.dumps(freeze_value())):
        raise RuntimeError("G9.6 source/model/development evidence changed after freeze")
    return frozen


def behavior():
    frozen = verify_freeze()
    if any(path.exists() for path in (EPISODES, STEPS, BASE / "confirmation_seed_manifest.json")):
        raise FileExistsError("G9.6 confirmation behavior already opened; no direct skip")
    json_new(BASE / "confirmation_seed_manifest.json", {"seeds": list(CONFIRM),
        "pre_confirmation_freeze_sha256": sha(FREEZE),
        "controller_checkpoint_sha256": frozen["controller_checkpoint_sha256"],
        "scene_lifecycle": "one reused Generic scene, ordered 105000-105029 resets",
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
                    frozen["controller_checkpoint_sha256"], monitor)
                if episode["simulation_error"]:
                    raise RuntimeError(f"G9.6 confirmation simulation error {seed}")
                ew.writerow(episode)
                sw.writerows(steps)
                episode_stream.flush()
                step_stream.flush()
                print(f"G9.6 confirmation behavior seed={seed} steps={len(steps)}", flush=True)
    finally:
        env.close()


def confirmation_rows():
    manifest = json.loads((BASE / "confirmation_seed_manifest.json").read_text())
    if (manifest["seeds"] != list(CONFIRM) or
            manifest["pre_confirmation_freeze_sha256"] != sha(FREEZE)):
        raise RuntimeError("G9.6 confirmation seed manifest changed")
    with EPISODES.open(newline="") as stream:
        episodes = list(csv.DictReader(stream))
    if [int(row["seed"]) for row in episodes] != list(CONFIRM):
        raise RuntimeError("G9.6 confirmation episodes incomplete/out of order")
    by_seed = {seed: [] for seed in CONFIRM}
    with STEPS.open(newline="") as stream:
        for row in csv.DictReader(stream):
            seed, d = int(row["seed"]), int(row["step_index"])
            if seed not in by_seed or d != len(by_seed[seed]):
                raise RuntimeError(f"G9.6 confirmation step indexing changed {seed}/{d}")
            by_seed[seed].append(row)
    for episode in episodes:
        if len(by_seed[int(episode["seed"])]) != int(episode["episode_length"]):
            raise RuntimeError("G9.6 confirmation episode/step length mismatch")
    return episodes, by_seed


def select():
    verify_freeze()
    if (BASE / "confirmation_state_selection.json").exists():
        raise FileExistsError(BASE / "confirmation_state_selection.json")
    _, by_seed = confirmation_rows()
    states = choose_states(list(CONFIRM), by_seed, "confirmation")
    if (len(states) != 120 or any(state["decision_index"] < 2 or
            state["decision_index"]+2 > state["episode_length"] for state in states)):
        raise RuntimeError("G9.6 confirmation selected E1/H2 state coverage invalid")
    json_new(BASE / "confirmation_state_selection.json", {"status": "SELECTED_WITHOUT_CANDIDATE_FORCE_OUTCOMES",
        "source_behavior_episodes_sha256": sha(EPISODES),
        "source_behavior_steps_sha256": sha(STEPS),
        "rule": "frozen G3 four per episode progress/prior-force strata",
        "states": states, "count": 120, "candidate_outcome_labels_used": False})
    print("G9.6 selected 120 confirmation states without candidate outcomes", flush=True)


def plans():
    verify_freeze()
    selection = json.loads((BASE / "confirmation_state_selection.json").read_text())
    if (selection["count"] != 120 or selection["source_behavior_steps_sha256"] != sha(STEPS) or
            selection["source_behavior_episodes_sha256"] != sha(EPISODES)):
        raise RuntimeError("G9.6 confirmation selected state set changed")
    if (BASE / "confirmation_parent_replay_audit.json").exists():
        raise FileExistsError(BASE / "confirmation_parent_replay_audit.json")
    _, by_seed = confirmation_rows()
    selected = {(int(s["seed"]), int(s["decision_index"])): s for s in selection["states"]}
    if len(selected) != 120:
        raise RuntimeError("G9.6 confirmation selected state keys duplicated")
    (BASE / "confirmation_states").mkdir(exist_ok=True)
    mapper = E1Representation()
    env, agent, monitor = setup(list(CONFIRM))
    parent_steps = state_count = fork_count = censored = clipped = tolerance_only = 0
    differences = {}
    try:
        for seed in CONFIRM:
            set_seed(seed)
            env.action_space.seed(seed)
            observation, _ = env.reset(seed=seed)
            check_scene(env)
            if not hasattr(monitor, "_g96_added"):
                env._simulation.root.addObject(monitor)
                monitor._g96_added = True
            rows = by_seed[seed]
            obs_history, action_history = [], []
            terminated = truncated = False
            decision = 0
            while not (terminated or truncated):
                if (decision >= len(rows) or not np.array_equal(observation,
                    np.asarray(json.loads(rows[decision]["decision_observation_json"]), dtype=np.float32))):
                    raise RuntimeError(f"G9.6 confirmation parent observation mismatch {seed}/{decision}")
                obs_history.append(observation.copy())
                state = selected.get((seed, decision))
                if state is not None:
                    action, final_actions, scores, mean = captured_plan_v2(agent, observation, decision == 0)
                else:
                    action = agent.act(observation, first_step=decision == 0, eval_mode=True)
                if not np.array_equal(action,
                    np.asarray(json.loads(rows[decision]["selected_action_json"]), dtype=np.float32)):
                    raise RuntimeError(f"G9.6 passive planner capture changed action {seed}/{decision}")
                if state is not None:
                    history_obs = np.asarray(obs_history[decision-2:decision+1], dtype=np.float32)
                    history_action = np.asarray(action_history[decision-2:decision], dtype=np.float32)
                    if history_obs.shape != (3, 14) or history_action.shape != (2, 2):
                        raise RuntimeError("G9.6 E1 pre-action history unavailable")
                    for j in range(3):
                        if not np.array_equal(history_obs[j], np.asarray(json.loads(
                                rows[decision-2+j]["decision_observation_json"]), dtype=np.float32)):
                            raise RuntimeError("G9.6 pre-action history observation mismatch")
                    for j in range(2):
                        if not np.array_equal(history_action[j], np.asarray(json.loads(
                                rows[decision-2+j]["selected_action_json"]), dtype=np.float32)):
                            raise RuntimeError("G9.6 pre-action history action mismatch")
                    candidates, numerical = canonical_plans(final_actions, scores, mean, action)
                    path = BASE / "confirmation_states" / f"state_{int(state['state_id']):04d}.json"
                    if path.exists():
                        item = json.loads(path.read_text())
                        if ((item["state_id"], item["seed"], item["decision_index"]) !=
                                (state["state_id"], seed, decision) or
                                len(item["plans"]) != PLANS_PER_STATE or
                                item.get("optimized_mean_numerical_semantics") != numerical or
                                any(not np.array_equal(np.asarray(item["plans"][j]["actions"], dtype=np.float32),
                                                       np.asarray(candidates[j]["actions"], dtype=np.float32))
                                    for j in range(PLANS_PER_STATE))):
                            raise RuntimeError(f"G9.6 replay-resume candidate mismatch {path}")
                        print(f"G9.6 replay-resume state {state['state_id']}", flush=True)
                    else:
                        with torch.inference_mode():
                            latent = agent.model.encode(agent._tensor_observation(observation))[0].cpu().numpy()
                        estimate = mapper.estimate(history_obs, history_action)
                        outcomes = [{**candidate, "fork": fork_two(env, monitor,
                            np.asarray(candidate["actions"], dtype=np.float32))} for candidate in candidates]
                        item = {"group": "confirmation", "state_id": state["state_id"],
                            "seed": seed, "decision_index": decision, "slot": state["slot"],
                            "episode_progress": state["episode_progress"],
                            "previous_executed_force_N": state["previous_executed_force_N"],
                            "history_observation": history_obs.tolist(),
                            "history_action": history_action.tolist(),
                            "current_observation": observation.tolist(),
                            "current_latent": latent.tolist(),
                            "E1_current_physical_estimate": estimate.tolist(),
                            "planner_mean_action": action.tolist(),
                            "optimized_mean_numerical_semantics": numerical,
                            "plans": outcomes, "future_behavior_actions_used_as_candidate": False}
                        json_new(path, item)
                    clipped += numerical["changed_by_clipping"]
                    tolerance_only += numerical["changed_by_tolerance_only"]
                    state_count += 1
                    fork_count += len(item["plans"])
                    censored += sum(not p["fork"]["complete_two_intervals"] for p in item["plans"])
                    if state_count % 20 == 0:
                        print(f"G9.6 confirmation states {state_count}/120 forks={fork_count} censored={censored}", flush=True)
                parent = signature(env, monitor, action)
                compare(parent, g2_signature(rows[decision]), differences,
                        f"G9.6 confirmation parent {seed}/{decision}")
                observation = np.asarray(parent["observation"], dtype=np.float32)
                terminated, truncated = parent["terminated"], parent["truncated"]
                action_history.append(action.copy())
                decision += 1
                parent_steps += 1
            if decision != len(rows):
                raise RuntimeError(f"G9.6 confirmation episode length mismatch {seed}")
            print(f"G9.6 confirmation parent seed={seed} steps={decision}", flush=True)
    finally:
        env.close()
    expected = sum(len(by_seed[seed]) for seed in CONFIRM)
    if (parent_steps != expected or state_count != 120 or fork_count != 1080 or
            len(list((BASE / "confirmation_states").glob("state_*.json"))) != 120):
        raise RuntimeError("G9.6 confirmation parent/planner pass incomplete")
    json_new(BASE / "confirmation_parent_replay_audit.json", {"status": "PASS",
        "ordered_reused_scene_seeds": list(CONFIRM),
        "parent_steps_compared": parent_steps, "planner_states": state_count,
        "planner_candidate_forks": fork_count, "complete_R2_forks": fork_count-censored,
        "censored_one_step_terminal_forks": censored,
        "optimized_mean_plans_changed_by_clipping": clipped,
        "optimized_mean_executed_matches_by_tolerance_only": tolerance_only,
        "ambiguous_full_two_action_matches": 0,
        "future_behavior_action_pairs_not_used_as_candidate_labels": True,
        "max_abs_difference": differences, "tolerance": 1e-8})
    print(f"G9.6 confirmation true-plan forks PASS: {state_count} states, {fork_count} plans", flush=True)


def assemble():
    verify_freeze()
    audit = json.loads((BASE / "confirmation_parent_replay_audit.json").read_text())
    if (audit["status"] != "PASS" or audit["planner_states"] != 120 or
            audit["planner_candidate_forks"] != 1080 or
            audit["ordered_reused_scene_seeds"] != list(CONFIRM)):
        raise RuntimeError("G9.6 confirmation parent/candidate audit incomplete")
    selection = json.loads((BASE / "confirmation_state_selection.json").read_text())
    fields = ("x", "F1_true_N", "F2_true_N", "R2_true_N", "seed", "decision_index",
              "state_global_id", "candidate_id", "source", "action_pair")
    records = {key: [] for key in fields}
    censored = 0
    state_summary = []
    for state in selection["states"]:
        sid = int(state["state_id"])
        item = json.loads((BASE / "confirmation_states" / f"state_{sid:04d}.json").read_text())
        if ((item["state_id"], item["seed"], item["decision_index"]) !=
                (sid, state["seed"], state["decision_index"]) or len(item["plans"]) != PLANS_PER_STATE):
            raise RuntimeError(f"G9.6 confirmation state identity changed {sid}")
        latent = np.asarray(item["current_latent"], dtype=np.float32)
        estimate = np.asarray(item["E1_current_physical_estimate"], dtype=np.float32)
        if latent.shape != (256,) or estimate.shape != (38,):
            raise RuntimeError("G9.6 E1 current-state feature dimension changed")
        complete = safe = 0
        for candidate_id, plan in enumerate(item["plans"]):
            if plan["candidate_id"] != candidate_id:
                raise RuntimeError("G9.6 confirmation candidate indexing changed")
            fork = plan["fork"]
            if not fork["complete_two_intervals"]:
                if fork["F2_true_N"] is not None or fork["R2_true_N"] is not None:
                    raise RuntimeError("G9.6 censored plan incorrectly given H2 target")
                censored += 1
                continue
            action = np.asarray(plan["actions"], dtype=np.float32)
            x = np.r_[latent, estimate, action.reshape(-1)]
            f1, f2, risk = [np.float32(fork[key]) for key in ("F1_true_N", "F2_true_N", "R2_true_N")]
            if (action.shape != (2, 2) or x.shape != (298,) or not np.all(np.isfinite(x)) or
                    risk != max(f1, f2)):
                raise RuntimeError("G9.6 confirmation original planner action/profile target changed")
            for key, value in (("x", x), ("F1_true_N", f1), ("F2_true_N", f2),
                               ("R2_true_N", risk), ("seed", item["seed"]),
                               ("decision_index", item["decision_index"]),
                               ("state_global_id", sid), ("candidate_id", candidate_id),
                               ("source", plan["source"]), ("action_pair", action)):
                records[key].append(value)
            complete += 1
            safe += risk <= F_LIM
        state_summary.append({"state_id": sid, "seed": item["seed"],
            "decision_index": item["decision_index"], "candidate_forks": PLANS_PER_STATE,
            "complete_R2": complete, "censored": PLANS_PER_STATE-complete,
            "true_safe_plans": safe})
    if len(records["x"])+censored != 1080 or censored != audit["censored_one_step_terminal_forks"]:
        raise RuntimeError("G9.6 confirmation complete/censored count changed")
    arrays = {}
    for key, values in records.items():
        dtype = ("U40" if key == "source" else np.int32 if key in
                 ("seed", "decision_index", "state_global_id", "candidate_id") else np.float32)
        arrays[key] = np.asarray(values, dtype=dtype)
    np.savez_compressed(BASE / "confirmation_planner_dataset.npz", **arrays)
    json_new(BASE / "confirmation_dataset_summary.json", {"status": "UNTOUCHED_G96_PROSPECTIVE_PLANNER_PROFILES",
        "states": 120, "candidate_forks": 1080,
        "complete_R2_rows": len(arrays["x"]), "censored_one_step_terminal_rows": censored,
        "state_summary": state_summary,
        "no_future_behavior_actions_as_candidate_input": True})
    print(f"G9.6 untouched planner profile dataset: {len(arrays['x'])} complete, {censored} censored", flush=True)


def evaluate():
    verify_freeze()
    with np.load(BASE / "confirmation_planner_dataset.npz") as z:
        arrays = {key: z[key].copy() for key in z.files}
    if len(set(arrays["state_global_id"])) != 120:
        raise RuntimeError("G9.6 confirmation planner states incomplete")
    model_rows, events, states, plans, coverage = evaluate_arrays(arrays, "confirmation")
    csv_new(BASE / "confirmation_plan_results.csv", plans)
    csv_new(BASE / "confirmation_temporal_event_metrics.csv", events)
    csv_new(BASE / "confirmation_state_metrics.csv", states)
    csv_new(BASE / "confirmation_model_metrics.csv", model_rows)
    csv_new(BASE / "confirmation_stepwise_coverage.csv", coverage)
    paired = []
    for seed in SEEDS:
        scalar = next(row for row in model_rows if row["model"] == "M_scalar" and row["training_seed"] == seed)
        step = next(row for row in model_rows if row["model"] == "M_step" and row["training_seed"] == seed)
        se = next(row for row in events if row["model"] == "M_scalar" and row["training_seed"] == seed and row["temporal_category"] == "SAFE-HIGH")
        pe = next(row for row in events if row["model"] == "M_step" and row["training_seed"] == seed and row["temporal_category"] == "SAFE-HIGH")
        paired.append({"training_seed": seed,
            "scalar_safe_rejection": scalar["safe_plan_rejection_rate"],
            "step_safe_rejection": step["safe_plan_rejection_rate"],
            "step_minus_scalar_safe_rejection": step["safe_plan_rejection_rate"]-scalar["safe_plan_rejection_rate"],
            "scalar_unsafe_acceptance": scalar["unsafe_plan_acceptance_rate"],
            "step_unsafe_acceptance": step["unsafe_plan_acceptance_rate"],
            "step_minus_scalar_unsafe_acceptance": step["unsafe_plan_acceptance_rate"]-scalar["unsafe_plan_acceptance_rate"],
            "scalar_SAFE_HIGH_misses": se["miss_count_if_true_unsafe"],
            "step_SAFE_HIGH_misses": pe["miss_count_if_true_unsafe"],
            "SAFE_HIGH_total": pe["count"],
            "scalar_rank_spearman": scalar["mean_within_state_spearman"],
            "step_rank_spearman": step["mean_within_state_spearman"],
            "scalar_false_predicted_infeasibility": scalar["false_predicted_infeasibility_rate"],
            "step_false_predicted_infeasibility": step["false_predicted_infeasibility_rate"],
            "scalar_usable_feasible_states": scalar["states_with_usable_true_safe_plan"],
            "step_usable_feasible_states": step["states_with_usable_true_safe_plan"]})
    json_new(BASE / "stepwise_vs_scalar_comparison.json", {"status": "UNTOUCHED_G96_ALL_THREE_SEEDS_MATCHED_PLANS",
        "same_complete_candidate_rows_per_model": len(arrays["x"]),
        "same_planner_states_per_model": len(set(arrays["state_global_id"])),
        "paired_seed_comparisons": paired,
        "training_seed_robustness_on_confirmation": robustness(model_rows, events),
        "no_best_seed_selection": True,
        "no_confirmation_training_or_recalibration": True,
        "classification": "See PHASE_G96_REPORT.md after evidence interpretation"})
    first_by_category = {}
    for index in range(len(arrays["x"])):
        category = plans[index]["temporal_category"]
        first_by_category.setdefault(category, index)
    examples = []
    for category in ("SAFE-SAFE", "SAFE-HIGH", "HIGH-SAFE", "HIGH-HIGH"):
        if category not in first_by_category:
            examples.append({"category": category, "available": False})
            continue
        index = first_by_category[category]
        key = (int(arrays["seed"][index]), int(arrays["state_global_id"][index]),
               int(arrays["candidate_id"][index]))
        scalar = next(p for p in plans if p["model"] == "M_scalar" and p["training_seed"] == SEEDS[0]
                      and (p["episode_seed"], p["state_id"], p["candidate_id"]) == key)
        step = next(p for p in plans if p["model"] == "M_step" and p["training_seed"] == SEEDS[0]
                    and (p["episode_seed"], p["state_id"], p["candidate_id"]) == key)
        examples.append({"category": category, "available": True,
            "selection": "first complete confirmation plan in frozen state/candidate order with this category",
            "training_seed": SEEDS[0], "episode_seed": key[0], "state_id": key[1],
            "candidate_id": key[2], "true_F1_N": scalar["F1_true_N"],
            "true_F2_N": scalar["F2_true_N"], "true_R2_N": scalar["R2_true_N"],
            "scalar_raw_Q95_R2_N": scalar["scalar_Q95_N"],
            "scalar_corrected_Q95_R2_N": scalar["decision_risk_score_N"],
            "scalar_predicted_safe": bool(scalar["predicted_safe"]),
            "step_raw_Q95_F1_N": step["step_F1_Q95_N"],
            "step_raw_Q95_F2_N": step["step_F2_Q95_N"],
            "step_corrected_Q95_F1_N": step["step_F1_Q95_corrected_N"],
            "step_corrected_Q95_F2_N": step["step_F2_Q95_corrected_N"],
            "step_predicted_safe": bool(step["predicted_safe"])})
    json_new(BASE / "representative_planner_examples.json", {"selection_frozen_without_model_performance_cherry_picking": True,
        "examples": examples})
    print("G9.6 all frozen scalar/stepwise confirmation results and examples saved", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("freeze", "behavior", "select", "plans", "assemble", "evaluate"), required=True)
    {"freeze": freeze, "behavior": behavior, "select": select, "plans": plans,
     "assemble": assemble, "evaluate": evaluate}[parser.parse_args().stage]()


if __name__ == "__main__":
    main()
