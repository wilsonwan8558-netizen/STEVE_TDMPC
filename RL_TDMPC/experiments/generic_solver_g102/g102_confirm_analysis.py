"""Analyze untouched, pre-frozen G10.2 confirmation without retraining."""

from __future__ import annotations

import csv
import json
import sys

import numpy as np
from scipy.stats import spearmanr

HERE = __import__("pathlib").Path(__file__).resolve().parent
RL = HERE.parents[1]
for name in ("generic_solver_g102", "generic_solver_g101", "generic_solver_g10", "generic_solver_g95"):
    sys.path.insert(0, str(RL / "experiments" / name))

from g102_prepare import OUT, SEEDS, CONFIRM_SEEDS, F_LIM  # noqa: E402
from g102_confirm import freeze_gate  # noqa: E402
from g102_eval_common import closest_safe, key  # noqa: E402
from g95_common import json_new, sha  # noqa: E402


def csv_new(path, rows):
    if path.exists():
        raise FileExistsError(path)
    if not rows:
        raise RuntimeError(f"No rows for {path}")
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def sign(value):
    return 0 if abs(value) <= 1e-6 else (1 if value > 0 else -1)


def rho(a, b):
    if len(a) < 3 or np.ptp(a) < 1e-9 or np.ptp(b) < 1e-9:
        return None
    value = float(spearmanr(a, b).statistic)
    return value if np.isfinite(value) else None


def analyze():
    frozen = freeze_gate()
    if (OUT / "confirmation_repair_results.csv").exists():
        raise FileExistsError("G10.2 confirmation already analyzed")
    audit = json.loads((OUT / "confirmation_parent_replay_audit.json").read_text())
    selection = json.loads((OUT / "confirmation_state_selection.json").read_text())
    if (audit["status"] != "PASS_EXACT_PASSIVE_PARENT_AND_FORK_REPLAY" or
            audit["states"] != 60 or any(x != 0 for x in audit["max_abs_parent_difference"].values()) or
            selection["ordered_seeds"] != list(CONFIRM_SEEDS) or
            audit["pre_confirmation_freeze_sha256"] != sha(OUT / "pre_confirmation_freeze.json")):
        raise RuntimeError("G10.2 confirmation replay/freeze audit failed")
    responses, boundaries, repairs = [], [], []
    solo_repair = {seed: {"proposed": 0, "true_safe": 0, "false_safe": 0,
        "censored_unverifiable": 0, "abstain": 0, "false_infeasible": 0} for seed in SEEDS}
    for selected in selection["selected_states"]:
        path = OUT / "confirmation_state_curves" / (
            f"seed_{selected['seed']}_decision_{selected['decision_index']:04d}.json")
        item = json.loads(path.read_text())
        if (item["seed"], item["decision_index"], item["source_episode_sha256"]) != (
                selected["seed"], selected["decision_index"], selected["episode_sha256"]):
            raise RuntimeError("G10.2 confirmation selected-state curve changed")
        vs = np.asarray(item["tested_v"], dtype=np.float32)
        q = np.asarray(item["predicted_delta_quantiles_replicas_N"], dtype=np.float64)
        absolute = np.asarray(item["predicted_absolute_R2_N"], dtype=np.float64)
        predicted_safe = np.asarray(item["predicted_safe"], dtype=bool)
        true = item["true_curve"]
        if (q.shape != (len(vs), 3, 3) or len(true) != len(vs) or
                not all(key(v) == key(r["v"]) for v, r in zip(vs, true)) or
                not np.array_equal(predicted_safe,
                    (absolute <= F_LIM) & np.asarray(item["predicted_uncertainty_constraint_pass"], dtype=bool))):
            raise RuntimeError("G10.2 confirmation curve/prediction alignment invalid")
        nominal_indices = np.flatnonzero(vs == np.float32(item["v_nom"]))
        if len(nominal_indices) != 1:
            raise RuntimeError("G10.2 confirmation exact nominal translation missing")
        nominal = true[int(nominal_indices[0])]
        true_safe_mask = np.asarray([r["true_safe"] for r in true], dtype=bool)
        valid = np.asarray([r["complete_two_intervals"] for r in true], dtype=bool)
        best_true = closest_safe(vs, true_safe_mask, item["v_nom"])
        best_pred = closest_safe(vs, predicted_safe, item["v_nom"])
        if nominal["complete_two_intervals"]:
            diagnostic_safe = ((nominal["R2_true_N"]+np.max(q[:, :, 2], axis=1) <= F_LIM) &
                np.asarray(item["predicted_uncertainty_constraint_pass"], dtype=bool))
            diagnostic_best = closest_safe(vs, diagnostic_safe, item["v_nom"])
        else:
            diagnostic_best = None
        true_v = float(vs[best_true]) if best_true is not None else None
        pred_v = float(vs[best_pred]) if best_pred is not None else None
        candidate = true[best_pred] if best_pred is not None else None
        diagnostic_candidate = true[diagnostic_best] if diagnostic_best is not None else None
        for replica, training_seed in enumerate(SEEDS):
            mask = item["nominal_conservative_Q95_N"]+q[:, replica, 2] <= F_LIM
            alone = closest_safe(vs, mask, item["v_nom"])
            outcome = solo_repair[training_seed]
            if alone is None:
                outcome["abstain"] += 1
                outcome["false_infeasible"] += best_true is not None
            else:
                outcome["proposed"] += 1
                witness = true[alone]
                outcome["true_safe"] += witness["true_safe"]
                outcome["false_safe"] += witness["complete_two_intervals"] and not witness["true_safe"]
                outcome["censored_unverifiable"] += not witness["complete_two_intervals"]
        state = {"seed": item["seed"], "decision_index": item["decision_index"],
            "selection_slot": item["selection_slot"], "v_nom": item["v_nom"],
            "w_nom_fixed": item["w_nom_fixed"],
            "nominal_true_R2_N": nominal["R2_true_N"],
            "nominal_R2_complete": nominal["complete_two_intervals"],
            "nominal_G10_conservative_Q95_N": item["nominal_conservative_Q95_N"],
            "v_true_closest_safe": true_v, "v_pred_closest_safe": pred_v,
            "true_recoverable_on_finite_grid": true_v is not None,
            "no_safe_found_with_censored_probes": bool(true_v is None and np.any(~valid)),
            "abstain": pred_v is None,
            "predicted_repair_R2_complete": candidate["complete_two_intervals"] if candidate else None,
            "predicted_repair_true_R2_N": candidate["R2_true_N"] if candidate else None,
            "predicted_repair_F1_true_N": candidate["F1_true_N"] if candidate else None,
            "predicted_repair_F2_true_N": candidate["F2_true_N"] if candidate else None,
            "predicted_repair_true_safe": candidate["true_safe"] if candidate else None,
            "diagnostic_true_nominal_baseline_predicted_v_NOT_DEPLOYABLE":
                float(vs[diagnostic_best]) if diagnostic_best is not None else None,
            "diagnostic_true_nominal_baseline_repair_true_safe_NOT_DEPLOYABLE":
                diagnostic_candidate["true_safe"] if diagnostic_candidate else None,
            "false_safe": candidate is not None and candidate["complete_two_intervals"] and not candidate["true_safe"],
            "repair_censored_unverifiable": candidate is not None and not candidate["complete_two_intervals"],
            "false_infeasible": pred_v is None and true_v is not None,
            "repair_action_deviation": abs(pred_v-item["v_nom"]) if pred_v is not None else None,
            "oracle_closest_safe_deviation": abs(true_v-item["v_nom"]) if true_v is not None else None,
            "one_interval_censored_tested_points": int(np.sum(~valid)),
            "true_force_source": "new frozen-protocol Generic-SOFA two-action fork",
            "no_action_applied_to_parent": True}
        repairs.append(state)
        boundaries.append({k: state[k] for k in (
            "seed", "decision_index", "selection_slot", "v_nom", "v_true_closest_safe",
            "v_pred_closest_safe", "true_recoverable_on_finite_grid", "abstain",
            "false_safe", "false_infeasible", "one_interval_censored_tested_points")}
            | {"abs_boundary_error_v": abs(pred_v-true_v) if pred_v is not None and true_v is not None else None,
               "same_side": sign(pred_v-item["v_nom"]) == sign(true_v-item["v_nom"])
                   if pred_v is not None and true_v is not None else None})
        if nominal["complete_two_intervals"]:
            delta_true = np.asarray([r["R2_true_N"]-nominal["R2_true_N"]
                if r["complete_two_intervals"] else np.nan for r in true], dtype=np.float64)
        else:
            delta_true = np.full(len(vs), np.nan, dtype=np.float64)
        local = (np.abs(vs-item["v_nom"]) <= .5+1e-7) & (vs != np.float32(item["v_nom"])) & valid & nominal["complete_two_intervals"]
        for replica, seed in enumerate(SEEDS):
            pred = q[:, replica, 0]
            err = pred[local]-delta_true[local]
            lo = key(np.clip(np.float32(item["v_nom"])-np.float32(.05), -1, 1))
            hi = key(np.clip(np.float32(item["v_nom"])+np.float32(.05), -1, 1))
            lookup = {key(v): j for j, v in enumerate(vs)}
            slope_valid = lo in lookup and hi in lookup and valid[lookup[lo]] and valid[lookup[hi]] and nominal["complete_two_intervals"]
            responses.append({"seed": item["seed"], "decision_index": item["decision_index"],
                "selection_slot": item["selection_slot"], "training_seed": seed,
                "nominal_R2_complete": nominal["complete_two_intervals"],
                "local_complete_nonnominal_rows": int(np.sum(local)),
                "delta_R2_Q50_MAE_N": float(np.mean(np.abs(err))) if len(err) else None,
                "delta_R2_Q50_RMSE_N": float(np.sqrt(np.mean(err**2))) if len(err) else None,
                "sign_correct": sum(sign(a) == sign(b) for a, b in zip(pred[local], delta_true[local])),
                "sign_valid": int(np.sum(local)),
                "reducing_translation_sign_correct": sum(sign(pred[j]) == sign(delta_true[j])
                    for j in np.flatnonzero(local & (vs < item["v_nom"]))),
                "reducing_translation_valid": int(np.sum(local & (vs < item["v_nom"]))),
                "increasing_translation_sign_correct": sum(sign(pred[j]) == sign(delta_true[j])
                    for j in np.flatnonzero(local & (vs > item["v_nom"]))),
                "increasing_translation_valid": int(np.sum(local & (vs > item["v_nom"]))),
                "true_local_Spearman_v_deltaR2": rho(vs[local], delta_true[local]),
                "predicted_local_Spearman_v_deltaR2": rho(vs[local], pred[local]),
                "true_slope_sign": sign(delta_true[lookup[hi]]-delta_true[lookup[lo]]) if slope_valid else None,
                "predicted_slope_sign": sign(pred[lookup[hi]]-pred[lookup[lo]]) if slope_valid else None,
                "slope_sign_agrees": (sign(delta_true[lookup[hi]]-delta_true[lookup[lo]]) ==
                    sign(pred[lookup[hi]]-pred[lookup[lo]])) if slope_valid else None})
    if (len(repairs), len(responses)) != (60, 180):
        raise RuntimeError("G10.2 confirmation 60-state/three-replica analysis incomplete")
    for name, rows in (("confirmation_response_metrics", responses),
                       ("confirmation_boundary_metrics", boundaries),
                       ("confirmation_repair_results", repairs)):
        csv_new(OUT / f"{name}.csv", rows)
    valid_recoverable = [r for r in repairs if r["true_recoverable_on_finite_grid"]]
    proposed = [r for r in repairs if not r["abstain"]]
    verified_proposed = [r for r in proposed if r["predicted_repair_R2_complete"]]
    matched_boundaries = [r for r in boundaries if r["abs_boundary_error_v"] is not None]
    summary = {"status": "UNTOUCHED_CONFIRMATION_EVALUATED_AFTER_FREEZE",
        "pre_confirmation_freeze_sha256": sha(OUT / "pre_confirmation_freeze.json"),
        "ordered_seeds": list(CONFIRM_SEEDS), "states": len(repairs),
        "nominal_two_interval_censored_states": sum(not r["nominal_R2_complete"] for r in repairs),
        "true_force_forks": audit["true_SOFA_two_action_forks"],
        "one_interval_censored_forks": audit["one_interval_censored_forks"],
        "translation_recoverable_on_finite_grid": len(valid_recoverable),
        "no_safe_found_on_finite_grid": len(repairs)-len(valid_recoverable),
        "no_safe_found_with_some_censored_probes": sum(r["no_safe_found_with_censored_probes"] for r in repairs),
        "predicted_repair_found": len(proposed),
        "successful_true_safe_repair": sum(r["predicted_repair_true_safe"] is True for r in repairs),
        "false_safe_repair": sum(r["false_safe"] for r in repairs),
        "censored_unverifiable_repair": sum(r["repair_censored_unverifiable"] for r in repairs),
        "abstain": sum(r["abstain"] for r in repairs),
        "false_infeasible": sum(r["false_infeasible"] for r in repairs),
        "recoverable_state_rescue_rate": sum(r["predicted_repair_true_safe"] is True for r in repairs)/len(valid_recoverable) if valid_recoverable else None,
        "false_infeasible_rate_among_recoverable": sum(r["false_infeasible"] for r in repairs)/len(valid_recoverable) if valid_recoverable else None,
        "false_safe_rate_among_verified_proposed": sum(r["false_safe"] for r in repairs)/len(verified_proposed) if verified_proposed else None,
        "abstention_rate": sum(r["abstain"] for r in repairs)/len(repairs),
        "safe_boundary_pairs": len(matched_boundaries),
        "diagnostic_only_true_future_nominal_baseline_NOT_DEPLOYABLE": {
            "predicted_repair_found": sum(r["diagnostic_true_nominal_baseline_predicted_v_NOT_DEPLOYABLE"] is not None for r in repairs),
            "true_safe_repair": sum(r["diagnostic_true_nominal_baseline_repair_true_safe_NOT_DEPLOYABLE"] is True for r in repairs),
            "false_safe_repair": sum(r["diagnostic_true_nominal_baseline_repair_true_safe_NOT_DEPLOYABLE"] is False for r in repairs)},
        "safe_boundary_mean_abs_error_v": float(np.mean([r["abs_boundary_error_v"] for r in matched_boundaries])) if matched_boundaries else None,
        "safe_boundary_median_abs_error_v": float(np.median([r["abs_boundary_error_v"] for r in matched_boundaries])) if matched_boundaries else None,
        "action_deviation_mean_proposed": float(np.mean([r["repair_action_deviation"] for r in proposed])) if proposed else None,
        "all_three_replicas_frozen": True,
        "no_retraining_recalibration_or_threshold_change": True,
        "online_107000_107199_opened": False}
    by_seed = {}
    for seed in SEEDS:
        rows = [r for r in responses if r["training_seed"] == seed]
        correct = sum(r["sign_correct"] for r in rows)
        valid_count = sum(r["sign_valid"] for r in rows)
        slope_rows = [r for r in rows if r["slope_sign_agrees"] is not None]
        by_seed[str(seed)] = {"sign_correct": correct, "sign_valid": valid_count,
            "sign_agreement_rate": correct/valid_count if valid_count else None,
            "slope_sign_agreement_count": sum(r["slope_sign_agrees"] for r in slope_rows),
            "slope_sign_valid": len(slope_rows),
            "state_mean_delta_R2_Q50_MAE_N": float(np.mean([r["delta_R2_Q50_MAE_N"] for r in rows
                if r["delta_R2_Q50_MAE_N"] is not None])) if any(r["delta_R2_Q50_MAE_N"] is not None for r in rows) else None,
            "single_replica_repair_diagnostic_not_deployment_rule": solo_repair[seed]}
    summary["per_replica_response"] = by_seed
    json_new(OUT / "confirmation_seed_robustness.json", {
        "all_fixed_replicas_reported_no_best_seed_selection": True,
        "single_replica_repair_diagnostics_omit_ensemble_disagreement_abstention": True,
        "per_replica_response": by_seed})
    json_new(OUT / "confirmation_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    analyze()
