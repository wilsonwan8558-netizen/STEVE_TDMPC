"""Exact reused-scene 1D predicted/true force curves at the G10 parent states."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
for name in ("generic_solver_g101", "generic_solver_g10", "generic_solver_g96",
             "generic_solver_g95", "generic_solver_g85", "generic_solver_g8",
             "generic_solver_g6a", "generic_solver_g3", "generic_solver_g2"):
    sys.path.insert(0, str(RL / "experiments" / name))
sys.path.insert(0, str(RL))

from g101_prepare import OUT, G10, G95, PRED_AXIS, TRUE_AXIS, LOCAL_DELTAS, F_LIM, gate  # noqa: E402
from g10_repair import FrozenRepair  # noqa: E402
from g10_offline import current_state, DEVELOPMENT_SEEDS  # noqa: E402
from g95_collect import archived  # noqa: E402
from g95_common import E1Representation, fork_two, json_new, sha  # noqa: E402
from g96_protocol_v2 import captured_plan_v2, canonical_plans  # noqa: E402
from g85_oracle import inputs as g85_inputs  # noqa: E402
from g8_prepare import TEST_SEEDS  # noqa: E402
from g6_collect import setup  # noqa: E402
from restore_audit import signature, compare, g2_signature  # noqa: E402
from collect import check_scene  # noqa: E402
from tdmpc2.common import set_seed  # noqa: E402


def records():
    gate()
    source = json.loads((OUT / "state_manifest.json").read_text())
    if (source["count"] != 40 or source["heldout_unsafe_nominal_count"] != 18 or
            source["difficult_event_prior_state_count"] != 22 or
            source["G10_state_manifest_sha256"] != sha(G10 / "offline_state_manifest.json")):
        raise RuntimeError("G10.1 frozen exact-state manifest changed")
    return source["states"]


def vkey(value):
    return float(np.float32(value))


def joined_points(axis, nominal):
    points = {vkey(v) for v in axis}
    points.add(vkey(nominal))
    for delta in LOCAL_DELTAS:
        points.add(vkey(np.clip(np.float32(nominal)-np.float32(delta), -1, 1)))
        points.add(vkey(np.clip(np.float32(nominal)+np.float32(delta), -1, 1)))
    return np.asarray(sorted(points), dtype=np.float32)


def curve(env, monitor, predictor, record, x, pair, prior):
    nominal, second = np.asarray(pair, dtype=np.float32)
    v_nom, w_nom = float(nominal[0]), float(nominal[1])
    pvs = joined_points(PRED_AXIS, v_nom)
    actions = np.c_[pvs, np.full(len(pvs), w_nom, dtype=np.float32)]
    scores = predictor.scores(x, actions, second)
    nominal_index = np.flatnonzero(pvs == np.float32(v_nom))
    if len(nominal_index) != 1:
        raise RuntimeError("G10.1 exact nominal translation missing from predicted curve")
    nominal_index = int(nominal_index[0])
    # G10 evaluated its nominal plan as a single action. CUDA float32 kernels can
    # round differently when that same action is part of a sweep batch.
    nominal_single = predictor.scores(x, nominal[None, :], second)[0]
    if not np.allclose(nominal_single,
                       prior["proposal"]["nominal_corrected_Q95_N"], rtol=0, atol=1e-7):
        raise RuntimeError("G10.1 frozen single-action nominal predictor differs from G10")
    nominal_batch_difference = float(np.max(np.abs(scores[nominal_index] - nominal_single)))
    if nominal_batch_difference > 1e-4:
        raise RuntimeError("G10.1 anomalous nominal CUDA batch-size difference")
    scores[nominal_index] = nominal_single
    safe = np.all(scores <= F_LIM, axis=1)
    if np.any(safe):
        j = min(np.flatnonzero(safe), key=lambda k: (abs(float(pvs[k])-v_nom), float(pvs[k])))
        v_pred = float(pvs[j])
    else:
        v_pred = None
    predicted = [{"v": float(v), "Q95_corrected_replicas_N": row.tolist(),
                  "max_corrected_Q95_N": float(np.max(row)),
                  "predicted_safe": bool(flag)}
                 for v, row, flag in zip(pvs, scores, safe)]
    tvs = joined_points(TRUE_AXIS, v_nom)
    if v_pred is not None and vkey(v_pred) not in {vkey(v) for v in tvs}:
        tvs = np.sort(np.r_[tvs, np.float32(v_pred)])
    true = []
    for index, v in enumerate(tvs):
        first = np.asarray((v, w_nom), dtype=np.float32)
        result = fork_two(env, monitor, np.stack((first, second)))
        true.append({"v": float(v), "F1_true_N": result["F1_true_N"],
            "F2_true_N": result["F2_true_N"], "R2_true_N": result["R2_true_N"],
            "complete_two_intervals": result["complete_two_intervals"],
            "true_safe": bool(result["complete_two_intervals"] and result["R2_true_N"] <= F_LIM),
            "censor_reason": result["censor_reason"]})
        if (index+1) % 50 == 0:
            print(f"G10.1 {record['cohort']} state={record['state_id']}" +
                  (f"/-{record['lookback_steps']}" if record.get("lookback_steps") else "") +
                  f" true forks {index+1}/{len(tvs)}", flush=True)
    p_by_v = {vkey(row["v"]): row for row in predicted}
    t_by_v = {vkey(row["v"]): row for row in true}
    nominal_key = vkey(v_nom)
    if nominal_key not in p_by_v or nominal_key not in t_by_v:
        raise RuntimeError("G10.1 exact nominal action missing from translation curves")
    reference = prior["nominal_true"]
    now = t_by_v[nominal_key]
    if (not now["complete_two_intervals"] or
            any(np.float32(now[key]) != np.float32(reference[old]) for key, old in
                (("F1_true_N", "F1_N"), ("F2_true_N", "F2_N"), ("R2_true_N", "R2_N")))):
        raise RuntimeError("G10.1 true nominal R2 differs from G10 true-SOFA fork")
    safe_true = [row for row in true if row["true_safe"]]
    best_true = min(safe_true, key=lambda row: (abs(row["v"]-v_nom), row["v"])) if safe_true else None
    return {"cohort": record["cohort"], "state_id": record["state_id"],
        "seed": record["seed"], "decision_index": record["decision_index"],
        "event_decision_index": record.get("event_decision_index"),
        "lookback_steps": record.get("lookback_steps"),
        "G10_result_sha256": record["G10_result_sha256"],
        "nominal_pair": pair.tolist(), "v_nom": v_nom, "w_nom_fixed": w_nom,
        "second_action_fixed": second.tolist(),
        "predicted_uniform_grid_points": len(PRED_AXIS),
        "true_uniform_grid_points": len(TRUE_AXIS),
        "predicted_curve": predicted, "true_curve": true,
        "predicted_closest_safe_v": v_pred,
        "predicted_closest_safe_abs_delta_v": abs(v_pred-v_nom) if v_pred is not None else None,
        "true_closest_tested_safe_v": best_true["v"] if best_true else None,
        "true_closest_tested_safe_abs_delta_v": abs(best_true["v"]-v_nom) if best_true else None,
        "predicted_safe_translation_count": int(np.sum(safe)),
        "true_safe_translation_count": len(safe_true),
        "true_fork_count": len(true),
        "nominal_true_R2_N": now["R2_true_N"],
        "nominal_predicted_max_corrected_Q95_N": p_by_v[nominal_key]["max_corrected_Q95_N"],
        "nominal_CUDA_batch_vs_single_max_abs_difference_N": nominal_batch_difference,
        "G10_zero_translation_zero_rotation_true_safe": prior["fallback_zero_true"]["true_safe"],
        "G10_zero_translation_zero_rotation_R2_N": prior["fallback_zero_true"]["R2_N"],
        "no_repaired_action_applied_to_parent": True}


def run(cohort):
    all_states = records()
    label = "heldout_development_unsafe_nominal" if cohort == "heldout" else "difficult_event_early_intervention"
    subset = [row for row in all_states if row["cohort"] == label]
    selected = {(row["seed"], row["decision_index"]): row for row in subset}
    if cohort == "heldout":
        by_seed, _ = archived()
        seeds = list(DEVELOPMENT_SEEDS)
        old_folder = G10 / "heldout_state_results"
    else:
        _, _, by_seed = g85_inputs()
        seeds = TEST_SEEDS
        old_folder = G10 / "difficult_event_state_results"
    folder = OUT / f"{cohort}_state_curves"
    folder.mkdir(exist_ok=True)
    mapper = E1Representation()
    predictor = FrozenRepair()
    env, agent, monitor = setup(seeds)
    checked = parent_steps = 0
    differences = {}
    try:
        for seed in seeds:
            set_seed(seed)
            env.action_space.seed(seed)
            observation, _ = env.reset(seed=seed)
            check_scene(env)
            if not hasattr(monitor, "_g101_added"):
                env._simulation.root.addObject(monitor)
                monitor._g101_added = True
            rows = by_seed[seed]
            obs_history, action_history = [], []
            terminated = truncated = False
            decision = 0
            while not (terminated or truncated):
                if (decision >= len(rows) or not np.array_equal(observation,
                        np.asarray(json.loads(rows[decision]["decision_observation_json"]), dtype=np.float32))):
                    raise RuntimeError(f"G10.1 pre-action parent observation changed {seed}/{decision}")
                obs_history.append(observation.copy())
                record = selected.get((seed, decision))
                if record is not None:
                    action, final_actions, scores, mean = captured_plan_v2(agent, observation, decision == 0)
                    candidates, _ = canonical_plans(final_actions, scores, mean, action)
                    pair = np.asarray(candidates[8]["actions"], dtype=np.float32)
                    old_name = (f"state_{record['state_id']:04d}.json" if cohort == "heldout" else
                                f"state_{record['state_id']:04d}_minus_{record['lookback_steps']}.json")
                    old_path = old_folder / old_name
                    if sha(old_path) != record["G10_result_sha256"]:
                        raise RuntimeError("G10.1 G10 source state/result changed")
                    old = json.loads(old_path.read_text())
                    if not np.array_equal(pair, np.asarray(old["nominal_pair"], dtype=np.float32)):
                        raise RuntimeError("G10.1 fixed nominal two-action plan changed")
                    x = current_state(agent, mapper, observation, obs_history, action_history, decision)
                    if cohort == "heldout":
                        archived_state = json.loads((G95 / "development_states" /
                            f"state_development_test_{record['state_id']:04d}.json").read_text())
                        archived_x = np.r_[archived_state["current_latent"],
                                            archived_state["E1_current_physical_estimate"]].astype(np.float32)
                        if not np.array_equal(x, archived_x):
                            raise RuntimeError("G10.1 G9.5 heldout latent/E1 state changed")
                    path = folder / old_name
                    if path.exists():
                        item = json.loads(path.read_text())
                        if (item["seed"], item["decision_index"], item["nominal_pair"],
                                item["G10_result_sha256"]) != (
                                seed, decision, pair.tolist(), record["G10_result_sha256"]):
                            raise RuntimeError("G10.1 replay-resume curve identity changed")
                    else:
                        item = curve(env, monitor, predictor, record, x, pair, old)
                        json_new(path, item)
                    checked += 1
                    print(f"G10.1 {cohort} {checked}/{len(subset)} state={record['state_id']}" +
                          (f"/-{record['lookback_steps']}" if record.get("lookback_steps") else "") +
                          f" true_safe={item['true_safe_translation_count']} pred_safe={item['predicted_safe_translation_count']}", flush=True)
                else:
                    action = agent.act(observation, first_step=decision == 0, eval_mode=True)
                if not np.array_equal(action,
                        np.asarray(json.loads(rows[decision]["selected_action_json"]), dtype=np.float32)):
                    raise RuntimeError("G10.1 nominal TD-MPC2 parent action changed")
                parent = signature(env, monitor, action)
                compare(parent, g2_signature(rows[decision]), differences,
                        f"G10.1 parent {seed}/{decision}")
                observation = np.asarray(parent["observation"], dtype=np.float32)
                terminated, truncated = parent["terminated"], parent["truncated"]
                action_history.append(action.copy())
                decision += 1
                parent_steps += 1
            if decision != len(rows):
                raise RuntimeError("G10.1 parent episode length changed")
    finally:
        env.close()
    if checked != len(subset) or len(list(folder.glob("state_*.json"))) != len(subset):
        raise RuntimeError("G10.1 translation-force curves incomplete")
    json_new(OUT / f"{cohort}_parent_replay_audit.json", {"status": "PASS_EXACT_FROZEN_PARENT_AND_G10_NOMINAL",
        "ordered_reused_scene_seeds": seeds, "parent_steps_compared": parent_steps,
        "states": checked, "max_abs_parent_difference": differences,
        "no_curve_action_applied_to_parent": True,
        "107000_107199_accessed": False})
    print(f"G10.1 {cohort} curves PASS {checked} states", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("heldout", "events"), required=True)
    run(parser.parse_args().cohort)
