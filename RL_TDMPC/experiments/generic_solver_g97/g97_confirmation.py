"""Freeze then collect untouched 106000-series matched-plan privileged ceiling."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
for folder in ("generic_solver_g97", "generic_solver_g96", "generic_solver_g95",
               "generic_solver_g6a", "generic_solver_g3", "generic_solver_g2"):
    sys.path.insert(0, str(RL / "experiments" / folder))
sys.path.insert(0, str(RL))

from g97_prepare import (OUT, G95, CONFIRM, SEEDS, F_LIM,
                         source_gate)  # noqa: E402
from g97_state import PrivilegedExtractor, RAW_DIM  # noqa: E402
from g97_train import load_replica  # noqa: E402
from g97_eval import score_arrays, paired_comparison, robustness  # noqa: E402
from g95_common import (E1Representation, fork_two, PLANS_PER_STATE,
                        json_new, sha)  # noqa: E402
from g96_protocol_v2 import canonical_plans, captured_plan_v2, ATOL, RTOL  # noqa: E402
from g6_collect import setup  # noqa: E402
from collect import evaluate_one, EPISODE_FIELDS, STEP_FIELDS, check_scene  # noqa: E402
from protocol import choose_states  # noqa: E402
from restore_audit import signature, compare, g2_signature  # noqa: E402
from tdmpc2.common import set_seed  # noqa: E402
from g95_eval import csv_new  # noqa: E402

FREEZE = OUT / "pre_confirmation_freeze.json"
EPISODES = OUT / "confirmation_behavior_episodes.csv"
STEPS = OUT / "confirmation_behavior_steps.csv"
SOURCES = ("g97_state.py", "g97_prepare.py", "g97_collect.py",
           "g97_train.py", "g97_eval.py", "g97_confirmation.py")
FROZEN_ARTIFACTS = ("metadata.json", "deployable_state_definition.json",
    "privileged_state_definition.json", "privileged_state_audit.json",
    "dataset_manifest.json", "dataset_split.json", "dataset_alignment_audit.json",
    "development_privileged_replay_audit.json", "development_privileged_states.npz",
    "repeatability_protocol.json", "repeatability_results.csv", "repeatability_summary.json",
    "model_protocol.json", "privileged_input_adapter.npz", "privileged_input_adapter_manifest.json",
    "training_manifest.json", "development_metrics.csv", "temporal_category_metrics.csv",
    "state_ranking_metrics.csv", "development_plan_results.csv",
    "identifiability_analysis.csv", "identifiability_summary.json",
    "training_seed_robustness.json", "development_deploy_vs_privileged_comparison.json")


def freeze_value():
    source, _, _ = source_gate()
    g96_dir = RL / "logs/generic_solver_phase_g96_stepwise_h2_force_profile"
    g96_freeze = json.loads((g96_dir / "pre_confirmation_freeze_v2.json").read_text())
    if (g96_freeze["protocol_amendment_sha256"] != sha(g96_dir / "protocol_amendment.json") or
            g96_freeze["protocol_reaudit_sha256"] != sha(g96_dir / "protocol_reaudit.json") or
            g96_freeze["source_sha256"]["g96_protocol_v2.py"] !=
            sha(RL / "experiments/generic_solver_g96/g96_protocol_v2.py")):
        raise RuntimeError("G9.6 validated executable planner protocol changed")
    if any(not (HERE / name).is_file() for name in SOURCES):
        raise RuntimeError("G9.7 confirmation source incomplete")
    if any(not (OUT / name).is_file() for name in FROZEN_ARTIFACTS):
        raise RuntimeError("G9.7 development/evaluation artifacts incomplete")
    replay = json.loads((OUT / "development_privileged_replay_audit.json").read_text())
    alignment = json.loads((OUT / "dataset_alignment_audit.json").read_text())
    repeat = json.loads((OUT / "repeatability_summary.json").read_text())
    training = json.loads((OUT / "training_manifest.json").read_text())
    if (replay["status"] != "PASS_EXACT_G2_G4_G95_PREACTION_ALIGNMENT" or
            replay["selected_states"] != 800 or replay["parent_steps_compared"] != 15500 or
            alignment["status"] != "PASS_ORIGINAL_7193_G95_LABELS_ACTIONS_AND_800_PRIVILEGED_STATES_ALIGNED" or
            repeat["status"] != "THREE_CASES_10_REPEATS_EACH" or
            not repeat["all_force_and_observable_next_state_max_abs_zero"] or
            training["status"] != "SIX_FIXED_MATCHED_CAPACITY_REPLICAS_SELECTED_ON_VALIDATION_ONLY" or
            training["training_seeds"] != list(SEEDS) or
            training["G95_dataset_sha256"] != sha(G95 / "development_planner_dataset.npz") or
            training["privileged_adapter_sha256"] != sha(OUT / "privileged_input_adapter.npz") or
            training["training_source_sha256"] != sha(HERE / "g97_train.py")):
        raise RuntimeError("G9.7 development data/model/repeatability evidence changed")
    checkpoints = {}
    normalizations = {}
    calibrations = {}
    for family in ("M_DEPLOY", "M_PRIV"):
        checkpoints[family], normalizations[family], calibrations[family] = {}, {}, {}
        for seed in SEEDS:
            item = training["models"][family][str(seed)]
            folder = OUT / "models" / family / f"seed_{seed}"
            checkpoint, norm = folder / "model.pt", folder / "normalization.json"
            if sha(checkpoint) != item["checkpoint_sha256"] or sha(norm) != item["normalization_sha256"]:
                raise RuntimeError(f"G9.7 checkpoint/normalization changed {family}/{seed}")
            checkpoints[family][str(seed)] = sha(checkpoint)
            normalizations[family][str(seed)] = sha(norm)
            calibrations[family][str(seed)] = item["validation_only_Q95_delta_N"]
    return {"status": "G97_SIX_MODELS_STATE_PROTOCOL_AND_METRICS_FROZEN_BEFORE_106000_CONFIRMATION",
        "ordered_confirmation_seeds": list(CONFIRM),
        "one_reused_scene_per_ordered_behavior_and_parent_fork_pass": True,
        "controller_checkpoint_sha256": source["controller_checkpoint_sha256"],
        "planner_source_sha256": source["planner_source_sha256"],
        "G95_dataset_sha256": sha(G95 / "development_planner_dataset.npz"),
        "G95_candidate_protocol_sha256": sha(G95 / "planner_candidate_protocol.json"),
        "G96_authorized_executable_mean_protocol_sha256": sha(RL / "logs/generic_solver_phase_g96_stepwise_h2_force_profile/protocol_amendment.json"),
        "G96_validated_executable_mean_source_sha256": sha(RL / "experiments/generic_solver_g96/g96_protocol_v2.py"),
        "optimized_mean_canonicalization": "componentwise clip first two optimized-mean actions to [-1,1]",
        "optimized_mean_match_atol": ATOL, "optimized_mean_match_rtol": RTOL,
        "sampled_eight_candidate_semantics_unchanged": True,
        "H": 2, "target": "R2=max(F1,F2)", "F_lim_N": F_LIM,
        "quantiles": [0.5, 0.9, 0.95, 0.99],
        "checkpoint_sha256": checkpoints,
        "normalization_sha256": normalizations,
        "validation_only_Q95_delta_N": calibrations,
        "same_model_parameter_count": training["same_parameter_count"],
        "privileged_adapter_sha256": sha(OUT / "privileged_input_adapter.npz"),
        "source_sha256": {name: sha(HERE / name) for name in SOURCES},
        "development_artifact_sha256": {name: sha(OUT / name) for name in FROZEN_ARTIFACTS},
        "state_selection": "exact frozen G3 four-per-episode progress/prior-force rule; no candidate outcomes",
        "fork": "nine exact frozen G9.5/G9.6-v2 planner candidates, first two actions open-loop true SOFA, no second-step replan",
        "no_retraining_recalibration_or_threshold_change_on_confirmation": True,
        "103000_104000_105000_cohorts_not_used_for_G97_final_confirmation": True}


def freeze():
    if any(path.exists() for path in (FREEZE, EPISODES, STEPS,
                                     OUT / "confirmation_seed_manifest.json")):
        raise FileExistsError("G9.7 freeze or confirmation already exists")
    json_new(FREEZE, freeze_value())
    print("G9.7 pre-confirmation freeze saved; 106000-106029 unopened", flush=True)


def verify_freeze():
    frozen = json.loads(FREEZE.read_text())
    if frozen != json.loads(json.dumps(freeze_value())):
        raise RuntimeError("G9.7 source/models/development evidence changed after freeze")
    return frozen


def behavior():
    frozen = verify_freeze()
    manifest_path = OUT / "confirmation_seed_manifest.json"
    if any(path.exists() for path in (manifest_path, EPISODES, STEPS)):
        raise FileExistsError("G9.7 confirmation behavior already opened")
    json_new(manifest_path, {"seeds": list(CONFIRM),
        "pre_confirmation_freeze_sha256": sha(FREEZE),
        "controller_checkpoint_sha256": frozen["controller_checkpoint_sha256"],
        "scene_lifecycle": "one reused Generic scene, ordered 106000-106029 resets",
        "python": sys.executable, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "cuda_device_index": 0, "cuda_device_name": torch.cuda.get_device_name(0)})
    env, agent, monitor = setup(list(CONFIRM))
    try:
        with EPISODES.open("x", newline="") as episode_stream, STEPS.open("x", newline="") as step_stream:
            ew, sw = (csv.DictWriter(episode_stream, fieldnames=EPISODE_FIELDS),
                      csv.DictWriter(step_stream, fieldnames=STEP_FIELDS))
            ew.writeheader()
            sw.writeheader()
            for index, seed in enumerate(CONFIRM):
                episode, steps = evaluate_one(env, agent, seed, index,
                    frozen["controller_checkpoint_sha256"], monitor)
                if episode["simulation_error"]:
                    raise RuntimeError(f"G9.7 confirmation simulation error {seed}")
                ew.writerow(episode)
                sw.writerows(steps)
                episode_stream.flush()
                step_stream.flush()
                print(f"G9.7 behavior seed={seed} steps={len(steps)}", flush=True)
    finally:
        env.close()


def confirmation_rows():
    manifest = json.loads((OUT / "confirmation_seed_manifest.json").read_text())
    if (manifest["seeds"] != list(CONFIRM) or
            manifest["pre_confirmation_freeze_sha256"] != sha(FREEZE)):
        raise RuntimeError("G9.7 confirmation seed manifest changed")
    with EPISODES.open(newline="") as stream:
        episodes = list(csv.DictReader(stream))
    if [int(row["seed"]) for row in episodes] != list(CONFIRM):
        raise RuntimeError("G9.7 confirmation episodes incomplete/out of order")
    by_seed = {seed: [] for seed in CONFIRM}
    with STEPS.open(newline="") as stream:
        for row in csv.DictReader(stream):
            seed, decision = int(row["seed"]), int(row["step_index"])
            if seed not in by_seed or decision != len(by_seed[seed]):
                raise RuntimeError(f"G9.7 confirmation step order changed {seed}/{decision}")
            by_seed[seed].append(row)
    for episode in episodes:
        if len(by_seed[int(episode["seed"])]) != int(episode["episode_length"]):
            raise RuntimeError("G9.7 confirmation episode/step length mismatch")
    return episodes, by_seed


def select():
    verify_freeze()
    path = OUT / "confirmation_state_selection.json"
    if path.exists():
        raise FileExistsError(path)
    _, by_seed = confirmation_rows()
    states = choose_states(list(CONFIRM), by_seed, "confirmation")
    if (len(states) != 120 or any(int(s["decision_index"]) < 2 or
            int(s["decision_index"])+2 > int(s["episode_length"]) for s in states)):
        raise RuntimeError("G9.7 confirmation E1/H2 state selection invalid")
    json_new(path, {"status": "SELECTED_WITHOUT_CANDIDATE_OUTCOMES",
        "source_behavior_episodes_sha256": sha(EPISODES),
        "source_behavior_steps_sha256": sha(STEPS),
        "rule": "exact frozen G3 four per episode progress/prior-force strata",
        "candidate_outcome_labels_used": False,
        "states": states, "count": 120})
    print("G9.7 selected 120 confirmation states without candidate outcomes", flush=True)


def plans():
    verify_freeze()
    selection = json.loads((OUT / "confirmation_state_selection.json").read_text())
    if (selection["count"] != 120 or
            selection["source_behavior_steps_sha256"] != sha(STEPS) or
            selection["source_behavior_episodes_sha256"] != sha(EPISODES)):
        raise RuntimeError("G9.7 confirmation state selection changed")
    audit_path = OUT / "confirmation_parent_replay_audit.json"
    if audit_path.exists():
        raise FileExistsError(audit_path)
    _, by_seed = confirmation_rows()
    selected = {(int(s["seed"]), int(s["decision_index"])): s for s in selection["states"]}
    if len(selected) != 120:
        raise RuntimeError("G9.7 duplicate selected state keys")
    state_folder = OUT / "confirmation_states"
    raw_folder = OUT / "confirmation_privileged_states"
    state_folder.mkdir(exist_ok=True)
    raw_folder.mkdir(exist_ok=True)
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
            if not hasattr(monitor, "_g97_confirmation_added"):
                env._simulation.root.addObject(monitor)
                monitor._g97_confirmation_added = True
            extractor = PrivilegedExtractor(env)
            rows = by_seed[seed]
            obs_history, action_history = [], []
            terminated = truncated = False
            decision = 0
            while not (terminated or truncated):
                if (decision >= len(rows) or not np.array_equal(observation,
                    np.asarray(json.loads(rows[decision]["decision_observation_json"]), dtype=np.float32))):
                    raise RuntimeError(f"G9.7 confirmation parent observation mismatch {seed}/{decision}")
                obs_history.append(observation.copy())
                state = selected.get((seed, decision))
                if state is not None:
                    raw, physical_audit = extractor.extract()
                    action, final_actions, scores, mean = captured_plan_v2(
                        agent, observation, decision == 0)
                    candidates, numerical = canonical_plans(final_actions, scores, mean, action)
                else:
                    action = agent.act(observation, first_step=decision == 0, eval_mode=True)
                if not np.array_equal(action,
                    np.asarray(json.loads(rows[decision]["selected_action_json"]), dtype=np.float32)):
                    raise RuntimeError(f"G9.7 passive planner capture changed action {seed}/{decision}")
                if state is not None:
                    sid = int(state["state_id"])
                    history_obs = np.asarray(obs_history[decision-2:decision+1], dtype=np.float32)
                    history_action = np.asarray(action_history[decision-2:decision], dtype=np.float32)
                    if history_obs.shape != (3, 14) or history_action.shape != (2, 2):
                        raise RuntimeError("G9.7 E1 pre-action history unavailable")
                    for j in range(3):
                        if not np.array_equal(history_obs[j], np.asarray(json.loads(
                                rows[decision-2+j]["decision_observation_json"]), dtype=np.float32)):
                            raise RuntimeError("G9.7 E1 history observation changed")
                    for j in range(2):
                        if not np.array_equal(history_action[j], np.asarray(json.loads(
                                rows[decision-2+j]["selected_action_json"]), dtype=np.float32)):
                            raise RuntimeError("G9.7 E1 history action changed")
                    with torch.inference_mode():
                        latent = agent.model.encode(agent._tensor_observation(observation))[0].cpu().numpy()
                    estimate = mapper.estimate(history_obs, history_action)
                    path = state_folder / f"state_{sid:04d}.json"
                    raw_path = raw_folder / f"state_{sid:04d}.npz"
                    if raw_path.exists():
                        with np.load(raw_path) as saved:
                            if (int(saved["seed"]) != seed or int(saved["decision_index"]) != decision or
                                    not np.array_equal(saved["raw"], raw)):
                                raise RuntimeError(f"G9.7 confirmation replay-resume raw state mismatch {raw_path}")
                    else:
                        np.savez_compressed(raw_path, raw=raw, seed=np.int32(seed),
                                            decision_index=np.int32(decision),
                                            state_id=np.int32(sid))
                    if path.exists():
                        item = json.loads(path.read_text())
                        if ((item["state_id"], item["seed"], item["decision_index"]) !=
                                (sid, seed, decision) or len(item["plans"]) != PLANS_PER_STATE or
                                item["optimized_mean_numerical_semantics"] != numerical or
                                any(not np.array_equal(np.asarray(item["plans"][j]["actions"], dtype=np.float32),
                                                       np.asarray(candidates[j]["actions"], dtype=np.float32))
                                    for j in range(PLANS_PER_STATE))):
                            raise RuntimeError(f"G9.7 confirmation replay-resume candidates changed {path}")
                    else:
                        outcomes = [{**candidate, "fork": fork_two(env, monitor,
                            np.asarray(candidate["actions"], dtype=np.float32))}
                            for candidate in candidates]
                        item = {"group": "confirmation", "state_id": sid,
                            "seed": seed, "decision_index": decision,
                            "slot": state["slot"], "episode_progress": state["episode_progress"],
                            "previous_executed_force_N": state["previous_executed_force_N"],
                            "history_observation": history_obs.tolist(),
                            "history_action": history_action.tolist(),
                            "current_observation": observation.tolist(),
                            "current_latent": latent.tolist(),
                            "E1_current_physical_estimate": estimate.tolist(),
                            "planner_mean_action": action.tolist(),
                            "current_privileged_physical_audit": {key: value for key, value in physical_audit.items()
                                if key != "physical_first_38"},
                            "optimized_mean_numerical_semantics": numerical,
                            "plans": outcomes,
                            "future_behavior_actions_used_as_candidate": False}
                        json_new(path, item)
                    clipped += numerical["changed_by_clipping"]
                    tolerance_only += numerical["changed_by_tolerance_only"]
                    state_count += 1
                    fork_count += len(item["plans"])
                    censored += sum(not p["fork"]["complete_two_intervals"] for p in item["plans"])
                    if state_count % 20 == 0:
                        print(f"G9.7 confirmation states={state_count}/120 forks={fork_count} censored={censored}", flush=True)
                parent = signature(env, monitor, action)
                compare(parent, g2_signature(rows[decision]), differences,
                        f"G9.7 confirmation parent {seed}/{decision}")
                observation = np.asarray(parent["observation"], dtype=np.float32)
                terminated, truncated = parent["terminated"], parent["truncated"]
                action_history.append(action.copy())
                decision += 1
                parent_steps += 1
            if decision != len(rows):
                raise RuntimeError(f"G9.7 confirmation episode length mismatch {seed}")
            print(f"G9.7 confirmation parent seed={seed} steps={decision}", flush=True)
    finally:
        env.close()
    expected = sum(len(by_seed[seed]) for seed in CONFIRM)
    if (parent_steps != expected or state_count != 120 or fork_count != 1080 or
            len(list(state_folder.glob("state_*.json"))) != 120 or
            len(list(raw_folder.glob("state_*.npz"))) != 120):
        raise RuntimeError("G9.7 confirmation parent/planner/privileged pass incomplete")
    json_new(audit_path, {"status": "PASS", "ordered_reused_scene_seeds": list(CONFIRM),
        "parent_steps_compared": parent_steps, "planner_states": state_count,
        "planner_candidate_forks": fork_count, "complete_R2_forks": fork_count-censored,
        "censored_one_step_terminal_forks": censored,
        "optimized_mean_plans_changed_by_clipping": clipped,
        "optimized_mean_executed_matches_by_tolerance_only": tolerance_only,
        "amendment_introduced_ambiguous_mean_sampled_full_plan_matches": 0,
        "raw_privileged_dimension": RAW_DIM,
        "future_behavior_action_pairs_not_used_as_candidate_labels": True,
        "max_abs_parent_difference": differences, "tolerance": 1e-8})
    print("G9.7 confirmation 120 true-plan state forks PASS", flush=True)


def assemble():
    verify_freeze()
    audit = json.loads((OUT / "confirmation_parent_replay_audit.json").read_text())
    if (audit["status"] != "PASS" or audit["planner_states"] != 120 or
            audit["planner_candidate_forks"] != 1080 or
            audit["ordered_reused_scene_seeds"] != list(CONFIRM)):
        raise RuntimeError("G9.7 confirmation parent/candidate audit incomplete")
    selection = json.loads((OUT / "confirmation_state_selection.json").read_text())
    fields = ("x", "F1_true_N", "F2_true_N", "R2_true_N", "seed",
              "decision_index", "state_global_id", "candidate_id", "action_pair")
    records = {key: [] for key in fields}
    raw_states = []
    summary = []
    censored = 0
    for state in selection["states"]:
        sid = int(state["state_id"])
        item = json.loads((OUT / "confirmation_states" / f"state_{sid:04d}.json").read_text())
        with np.load(OUT / "confirmation_privileged_states" / f"state_{sid:04d}.npz") as z:
            raw = z["raw"].copy()
            if (int(z["seed"]) != int(state["seed"]) or
                    int(z["decision_index"]) != int(state["decision_index"]) or
                    int(z["state_id"]) != sid or raw.shape != (RAW_DIM,)):
                raise RuntimeError(f"G9.7 confirmation raw preaction state changed {sid}")
        if ((item["state_id"], item["seed"], item["decision_index"]) !=
                (sid, state["seed"], state["decision_index"]) or len(item["plans"]) != PLANS_PER_STATE):
            raise RuntimeError(f"G9.7 confirmation selected/planner identity changed {sid}")
        raw_states.append(raw)
        latent = np.asarray(item["current_latent"], dtype=np.float32)
        estimate = np.asarray(item["E1_current_physical_estimate"], dtype=np.float32)
        if latent.shape != (256,) or estimate.shape != (38,):
            raise RuntimeError("G9.7 confirmation E1 dimension changed")
        complete = safe = 0
        for candidate_id, plan in enumerate(item["plans"]):
            if plan["candidate_id"] != candidate_id:
                raise RuntimeError("G9.7 confirmation planner candidate indexing changed")
            fork = plan["fork"]
            if not fork["complete_two_intervals"]:
                if fork["F2_true_N"] is not None or fork["R2_true_N"] is not None:
                    raise RuntimeError("G9.7 censored fork incorrectly assigned H2 label")
                censored += 1
                continue
            action = np.asarray(plan["actions"], dtype=np.float32)
            x = np.r_[latent, estimate, action.reshape(-1)]
            f1, f2, risk = (np.float32(fork[key]) for key in
                            ("F1_true_N", "F2_true_N", "R2_true_N"))
            if (action.shape != (2, 2) or x.shape != (298,) or
                    not np.all(np.isfinite(x)) or risk != max(f1, f2)):
                raise RuntimeError("G9.7 exact scalar H2 planner action/target changed")
            for key, value in (("x", x), ("F1_true_N", f1), ("F2_true_N", f2),
                ("R2_true_N", risk), ("seed", item["seed"]),
                ("decision_index", item["decision_index"]),
                ("state_global_id", sid), ("candidate_id", candidate_id),
                ("action_pair", action)):
                records[key].append(value)
            complete += 1
            safe += risk <= F_LIM
        summary.append({"state_id": sid, "seed": int(item["seed"]),
            "decision_index": int(item["decision_index"]),
            "candidate_forks": PLANS_PER_STATE, "complete_R2": complete,
            "censored": PLANS_PER_STATE-complete, "true_safe_plans": int(safe)})
    if len(records["x"])+censored != 1080 or censored != audit["censored_one_step_terminal_forks"]:
        raise RuntimeError("G9.7 confirmation complete/censored count changed")
    arrays = {key: np.asarray(values, dtype=(np.int32 if key in
        ("seed", "decision_index", "state_global_id", "candidate_id") else np.float32))
        for key, values in records.items()}
    raw_matrix = np.asarray(raw_states, dtype=np.float32)
    if raw_matrix.shape != (120, RAW_DIM):
        raise RuntimeError("G9.7 confirmation privileged state matrix incomplete")
    np.savez_compressed(OUT / "confirmation_planner_dataset.npz", **arrays)
    np.savez_compressed(OUT / "confirmation_privileged_states.npz", raw=raw_matrix,
                        state_id=np.arange(120, dtype=np.int32))
    json_new(OUT / "confirmation_dataset_summary.json", {
        "status": "UNTOUCHED_G97_MATCHED_PLANNER_PROFILES_AND_PREACTION_PRIVILEGED_STATE",
        "states": 120, "candidate_forks": 1080,
        "complete_R2_rows": len(arrays["x"]),
        "censored_one_step_terminal_rows": censored,
        "state_summary": summary,
        "no_future_behavior_actions_as_candidate_input": True,
        "no_future_or_post_action_predictor_state": True})
    print(f"G9.7 confirmation dataset {len(arrays['x'])} complete/{censored} censored", flush=True)


def evaluate():
    verify_freeze()
    with np.load(OUT / "confirmation_planner_dataset.npz") as z:
        arrays = {key: z[key].copy() for key in z.files}
    with np.load(OUT / "confirmation_privileged_states.npz") as z:
        raw = z["raw"].copy()
        if not np.array_equal(z["state_id"], np.arange(120, dtype=np.int32)):
            raise RuntimeError("G9.7 confirmation privileged state order changed")
    if len(set(arrays["state_global_id"])) != 120:
        raise RuntimeError("G9.7 confirmation planner states incomplete")
    rows, events, states, plans = score_arrays(arrays, raw, "confirmation")
    csv_new(OUT / "confirmation_metrics.csv", rows)
    csv_new(OUT / "confirmation_temporal_category_metrics.csv", events)
    csv_new(OUT / "confirmation_state_ranking_metrics.csv", states)
    csv_new(OUT / "confirmation_plan_results.csv", plans)
    json_new(OUT / "deploy_vs_privileged_comparison.json",
             paired_comparison(rows, events, "untouched_106000_106029_confirmation"))
    json_new(OUT / "confirmation_training_seed_robustness.json", robustness(rows, events))
    print("G9.7 all six frozen deploy/privileged models scored on same confirmation plans", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("freeze", "behavior", "select", "plans", "assemble", "evaluate"), required=True)
    {"freeze": freeze, "behavior": behavior, "select": select,
     "plans": plans, "assemble": assemble, "evaluate": evaluate}[parser.parse_args().stage]()


if __name__ == "__main__":
    main()
