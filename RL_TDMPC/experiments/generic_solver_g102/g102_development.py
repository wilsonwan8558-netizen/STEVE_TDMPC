"""Evaluate frozen local replicas on the 40 previously inspected G10.1 states."""

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

from g102_prepare import OUT, G101, G95, SEEDS, F_LIM, verify_source  # noqa: E402
from g102_eval_common import LocalEnsemble, search_points, closest_safe, key  # noqa: E402
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
    if len(a) < 3 or np.ptp(a) <= 1e-9 or np.ptp(b) <= 1e-9:
        return None
    result = float(spearmanr(a, b).statistic)
    return result if np.isfinite(result) else None


def features(row, item):
    if row["cohort"] == "heldout_development_unsafe_nominal":
        path = G95 / "development_states" / f"state_development_test_{int(row['state_id']):04d}.json"
        state = json.loads(path.read_text())
        if ((state["seed"], state["decision_index"]) != (row["seed"], row["decision_index"]) or
                not np.array_equal(np.asarray(state["plans"][8]["actions"], dtype=np.float32),
                                   np.asarray(item["nominal_pair"], dtype=np.float32))):
            raise RuntimeError("G10.2 G10.1/G9.5 heldout state alignment changed")
        x = np.r_[state["current_latent"], state["E1_current_physical_estimate"]].astype(np.float32)
    else:
        path = OUT / "development_event_features" / (
            f"state_{int(row['state_id']):04d}_minus_{int(row['lookback_steps'])}.json")
        state = json.loads(path.read_text())
        if (state["seed"], state["decision_index"], state["G101_curve_sha256"]) != (
                row["seed"], row["decision_index"], row["curve_sha256"]):
            raise RuntimeError("G10.2 G10.1 event state alignment changed")
        x = np.asarray(state["x_294"], dtype=np.float32)
    if x.shape != (294,) or not np.all(np.isfinite(x)):
        raise RuntimeError("G10.2 deployable feature vector invalid")
    return x


def summarize(rows):
    recoverable = [r for r in rows if r["true_recoverable_on_search_grid"]]
    safe = [r for r in rows if r["predicted_repair_true_safe"] is True]
    false_safe = [r for r in rows if r["false_safe"]]
    abstain = [r for r in rows if r["abstain"]]
    return {"states": len(rows), "true_recoverable": len(recoverable),
        "no_safe_found_on_finite_search_grid": len(rows)-len(recoverable),
        "predicted_repair_found": len(rows)-len(abstain),
        "true_safe_repair": len(safe), "false_safe_repair": len(false_safe),
        "abstain": len(abstain),
        "false_infeasible": sum(r["false_infeasible"] for r in rows),
        "recoverable_state_rescue_rate": len(safe)/len(recoverable) if recoverable else None,
        "false_infeasible_rate_among_recoverable": sum(r["false_infeasible"] for r in rows)/len(recoverable) if recoverable else None,
        "false_safe_rate_among_proposed": len(false_safe)/(len(rows)-len(abstain)) if len(rows)>len(abstain) else None,
        "abstention_rate": len(abstain)/len(rows) if rows else None}


def evaluate():
    verify_source()
    event_audit = json.loads((OUT / "development_event_feature_replay_audit.json").read_text())
    if event_audit["status"] != "PASS_EXACT_G10_1_EVENT_FEATURE_REPLAY":
        raise RuntimeError("G10.2 exact event feature replay audit required")
    if (OUT / "development_repair_results.csv").exists():
        raise FileExistsError("G10.2 development already evaluated")
    manifest = json.loads((OUT / "local_dataset_manifest.json").read_text())
    ensemble = LocalEnsemble()
    response, boundary, repair, revisit, events, witnesses = [], [], [], [], [], []
    solo = {seed: {"sign_correct": 0, "sign_valid": 0, "mae_sum": 0., "rows": 0,
                   "proposed": 0, "true_safe": 0, "false_safe": 0, "false_infeasible": 0}
            for seed in SEEDS}
    for row in manifest["development_test_states"]:
        path = __import__("pathlib").Path(row["curve_path"])
        if sha(path) != row["curve_sha256"]:
            raise RuntimeError("G10.2 frozen G10.1 true-force curve hash changed")
        item = json.loads(path.read_text())
        if (item["seed"], item["decision_index"], item["nominal_pair"]) != (
                row["seed"], row["decision_index"], row["nominal_pair"]):
            raise RuntimeError("G10.2 development curve identity changed")
        x = features(row, item)
        pair = np.asarray(item["nominal_pair"], dtype=np.float32)
        v_nom = float(pair[0, 0])
        true_by_v = {key(p["v"]): p for p in item["true_curve"]}
        vs = np.asarray(sorted(true_by_v), dtype=np.float32)
        result = ensemble.score(x, pair, vs)
        score_by_v = {key(v): k for k, v in enumerate(vs)}
        true_delta = np.asarray([true_by_v[key(v)]["R2_true_N"]-item["nominal_true_R2_N"]
                                 for v in vs], dtype=np.float64)
        complete = np.asarray([true_by_v[key(v)]["complete_two_intervals"] for v in vs], dtype=bool)
        if not np.all(complete):
            raise RuntimeError("G10.2 development true R2 censored; predeclared handling needed")
        local = (np.abs(vs-v_nom) <= .5+1e-7) & (vs != np.float32(v_nom))
        reduced = local & (vs < v_nom)
        increased = local & (vs > v_nom)
        withdrawal = reduced & (vs < 0)
        for replica, seed in enumerate(SEEDS):
            prediction = result["delta_quantiles_replicas_N"][:, replica, 0]
            err = prediction[local]-true_delta[local]
            sign_agree = np.asarray([sign(a) == sign(b) for a, b in
                                     zip(prediction[local], true_delta[local])], dtype=bool)
            m = {"cohort": item["cohort"], "state_id": item["state_id"],
                "seed": item["seed"], "decision_index": item["decision_index"],
                "lookback_steps": item["lookback_steps"], "training_seed": seed,
                "local_nonzero_rows": int(np.sum(local)),
                "delta_R2_Q50_MAE_N": float(np.mean(np.abs(err))) if len(err) else None,
                "delta_R2_Q50_RMSE_N": float(np.sqrt(np.mean(err**2))) if len(err) else None,
                "sign_agreement_count": int(np.sum(sign_agree)),
                "sign_valid": len(sign_agree),
                "reduced_translation_sign_agreement_count": sum(sign(prediction[j]) == sign(true_delta[j]) for j in np.flatnonzero(reduced)),
                "reduced_translation_valid": int(np.sum(reduced)),
                "increased_translation_sign_agreement_count": sum(sign(prediction[j]) == sign(true_delta[j]) for j in np.flatnonzero(increased)),
                "increased_translation_valid": int(np.sum(increased)),
                "withdrawal_sign_agreement_count": sum(sign(prediction[j]) == sign(true_delta[j]) for j in np.flatnonzero(withdrawal)),
                "withdrawal_valid": int(np.sum(withdrawal)),
                "true_local_Spearman_v_deltaR2": rho(vs[local], true_delta[local]),
                "predicted_local_Spearman_v_deltaR2": rho(vs[local], prediction[local])}
            lo, hi = (key(np.clip(np.float32(v_nom)+np.float32(d), -1, 1)) for d in (-.05, .05))
            if lo in score_by_v and hi in score_by_v and lo != hi:
                m["true_local_slope_sign"] = sign(true_delta[score_by_v[hi]]-true_delta[score_by_v[lo]])
                m["predicted_local_slope_sign"] = sign(prediction[score_by_v[hi]]-prediction[score_by_v[lo]])
                m["local_slope_sign_agrees"] = m["true_local_slope_sign"] == m["predicted_local_slope_sign"]
            else:
                m["true_local_slope_sign"] = m["predicted_local_slope_sign"] = m["local_slope_sign_agrees"] = None
            response.append(m)
            solo[seed]["sign_correct"] += m["sign_agreement_count"]
            solo[seed]["sign_valid"] += m["sign_valid"]
            solo[seed]["mae_sum"] += float(np.sum(np.abs(err)))
            solo[seed]["rows"] += len(err)
        search = search_points(v_nom)
        if not all(key(v) in score_by_v for v in search):
            raise RuntimeError("G10.2 search grid not fully covered by preexisting G10.1 true forks")
        indices = np.asarray([score_by_v[key(v)] for v in search])
        true_safe_mask = np.asarray([true_by_v[key(v)]["true_safe"] for v in search], dtype=bool)
        best_true = closest_safe(search, true_safe_mask, v_nom)
        best_pred = closest_safe(search, result["predicted_safe"][indices], v_nom)
        diagnostic_oracle_baseline_safe = (
            (item["nominal_true_R2_N"]+
             np.max(result["delta_quantiles_replicas_N"][indices, :, 2], axis=1) <= F_LIM) &
            result["uncertainty_constraint_pass"][indices])
        diagnostic_best = closest_safe(search, diagnostic_oracle_baseline_safe, v_nom)
        true_v = float(search[best_true]) if best_true is not None else None
        pred_v = float(search[best_pred]) if best_pred is not None else None
        diagnostic_v = float(search[diagnostic_best]) if diagnostic_best is not None else None
        pred_true = true_by_v[key(pred_v)] if pred_v is not None else None
        diagnostic_true = true_by_v[key(diagnostic_v)] if diagnostic_v is not None else None
        false_safe = pred_true is not None and not pred_true["true_safe"]
        false_infeasible = pred_v is None and true_v is not None
        base = {"cohort": item["cohort"], "state_id": item["state_id"],
            "seed": item["seed"], "decision_index": item["decision_index"],
            "lookback_steps": item["lookback_steps"], "v_nom": v_nom,
            "w_nom_fixed": float(pair[0, 1]),
            "second_action_fixed": json.dumps(pair[1].tolist()),
            "nominal_true_R2_N": item["nominal_true_R2_N"],
            "frozen_G10_nominal_conservative_Q95_N": result["nominal_conservative_Q95_N"],
            "nominal_baseline_error_N": result["nominal_conservative_Q95_N"]-item["nominal_true_R2_N"],
            "v_true_closest_safe_on_search_grid": true_v,
            "v_predicted_closest_safe": pred_v,
            "true_recoverable_on_search_grid": true_v is not None,
            "abstain": pred_v is None,
            "predicted_repair_true_safe": pred_true["true_safe"] if pred_true else None,
            "predicted_repair_true_R2_N": pred_true["R2_true_N"] if pred_true else None,
            "predicted_repair_F1_true_N": pred_true["F1_true_N"] if pred_true else None,
            "predicted_repair_F2_true_N": pred_true["F2_true_N"] if pred_true else None,
            "predicted_repair_action_deviation": abs(pred_v-v_nom) if pred_v is not None else None,
            "diagnostic_true_nominal_baseline_predicted_v_NOT_DEPLOYABLE": diagnostic_v,
            "diagnostic_true_nominal_baseline_repair_true_safe_NOT_DEPLOYABLE":
                diagnostic_true["true_safe"] if diagnostic_true else None,
            "oracle_closest_safe_action_deviation": abs(true_v-v_nom) if true_v is not None else None,
            "false_safe": false_safe, "false_infeasible": false_infeasible,
            "repair_true_force_source": "exact prior G10.1 true-SOFA fork at this same parent state/action; no duplicate fork" if pred_v is not None else None,
            "finite_grid_not_impossibility_proof": True}
        repair.append(base)
        boundary.append({**{k: base[k] for k in ("cohort", "state_id", "seed", "decision_index", "lookback_steps", "v_nom")},
            "v_true_closest_safe": true_v, "v_pred_closest_safe": pred_v,
            "abs_boundary_error_v": abs(pred_v-true_v) if pred_v is not None and true_v is not None else None,
            "same_side": sign(pred_v-v_nom) == sign(true_v-v_nom)
                if pred_v is not None and true_v is not None else None,
            "false_infeasible": false_infeasible, "false_safe": false_safe})
        for replica, seed in enumerate(SEEDS):
            solo_mask = result["nominal_conservative_Q95_N"]+result["delta_quantiles_replicas_N"][indices, replica, 2] <= F_LIM
            chosen = closest_safe(search, solo_mask, v_nom)
            solo[seed]["proposed"] += chosen is not None
            if chosen is None:
                solo[seed]["false_infeasible"] += true_v is not None
            else:
                good = true_by_v[key(search[chosen])]["true_safe"]
                solo[seed]["true_safe"] += good
                solo[seed]["false_safe"] += not good
        old_false_infeasible = (item["predicted_closest_safe_v"] is None and
                                item["true_safe_translation_count"] > 0)
        if old_false_infeasible:
            if true_v is None:
                raise RuntimeError("G10.2 search grid lost a G10.1 false-infeasible safe witness")
            oracle_index = score_by_v[key(true_v)]
            oracle_true_delta = (true_by_v[key(true_v)]["R2_true_N"]-item["nominal_true_R2_N"])
            oracle_predicted_median_Q50_delta = float(np.median(
                result["delta_quantiles_replicas_N"][oracle_index, :, 0]))
            revisit.append({**base, "G101_false_infeasible": True,
                "oracle_closest_safe_true_delta_R2_N": oracle_true_delta,
                "oracle_closest_safe_predicted_replica_median_Q50_delta_R2_N": oracle_predicted_median_Q50_delta,
                "G102_identified_oracle_safe_direction": sign(oracle_true_delta) == sign(oracle_predicted_median_Q50_delta),
                "G102_predicted_repair_found": pred_v is not None,
                "G102_true_safe_repair": pred_true["true_safe"] if pred_true else False})
        if item["cohort"] == "difficult_event_early_intervention":
            events.append({**base, "G101_true_recoverable": item["true_safe_translation_count"] > 0,
                "G101_predicted_repair_found": item["predicted_closest_safe_v"] is not None})
        for label, v in (("zero", 0.), ("withdraw_minus_0_5", -.5), ("withdraw_minus_1", -1.)):
            j = score_by_v[key(v)]
            truth = true_by_v[key(v)]
            witnesses.append({"cohort": item["cohort"], "state_id": item["state_id"],
                "lookback_steps": item["lookback_steps"], "witness": label,
                "v": v, "true_delta_R2_N": truth["R2_true_N"]-item["nominal_true_R2_N"],
                "predicted_delta_Q50_replicas_N": json.dumps(result["delta_quantiles_replicas_N"][j, :, 0].tolist()),
                "predicted_delta_corrected_Q95_max_N": float(np.max(result["delta_quantiles_replicas_N"][j, :, 2])),
                "true_R2_N": truth["R2_true_N"],
                "predicted_reconstructed_R2_N": result["predicted_absolute_R2_N"][j],
                "true_safe": truth["true_safe"], "predicted_safe": result["predicted_safe"][j]})
    if (len(repair), len(response), len(events), len(revisit)) != (40, 120, 22, 22):
        raise RuntimeError("G10.2 exact development/even/false-infeasible set not covered")
    for name, rows in (("development_response_metrics", response),
                       ("development_boundary_metrics", boundary),
                       ("development_repair_results", repair),
                       ("false_infeasible_revisit", revisit),
                       ("difficult_event_results", events),
                       ("zero_withdrawal_results", witnesses)):
        csv_new(OUT / f"{name}.csv", rows)
    robustness = {str(seed): {**result,
        "local_Q50_delta_MAE_N": result["mae_sum"]/result["rows"] if result["rows"] else None,
        "local_sign_agreement_rate": result["sign_correct"]/result["sign_valid"] if result["sign_valid"] else None}
        for seed, result in solo.items()}
    json_new(OUT / "training_seed_robustness.json", {"all_three_fixed_replicas_reported": True,
        "no_best_seed_selection": True, "per_replica_development_diagnostics": robustness,
        "per_replica_repair_diagnostics_are_non_deployable_and_omit_ensemble_disagreement_abstention": True,
        "ensemble_rule": "max three corrected Q95 delta plus validation-only spread abstention"})
    groups = {"heldout": [r for r in repair if r["cohort"] == "heldout_development_unsafe_nominal"],
        "t_star_minus_1": [r for r in repair if r["lookback_steps"] == 1],
        "t_star_minus_2": [r for r in repair if r["lookback_steps"] == 2]}
    groups["combined_descriptive_not_exchangeable"] = repair
    compare = {"G101_comparison_sha256": sha(G101 / "G10_vs_G101_comparison.json"),
        "G101_counts": json.loads((G101 / "G10_vs_G101_comparison.json").read_text())["G101_translation_only_counts"],
        "G102_development_counts": {name: summarize(rows) for name, rows in groups.items()},
        "G101_false_infeasible_22_revisit": {"states": len(revisit),
            "G102_predicted_repair_found": sum(r["G102_predicted_repair_found"] for r in revisit),
            "G102_true_safe_repair": sum(r["G102_true_safe_repair"] for r in revisit),
            "G102_false_safe": sum(r["false_safe"] for r in revisit),
            "G102_abstain": sum(r["abstain"] for r in revisit)},
        "diagnostic_only_true_future_nominal_baseline_NOT_DEPLOYABLE": {
            "predicted_repair_found": sum(r["diagnostic_true_nominal_baseline_predicted_v_NOT_DEPLOYABLE"] is not None for r in repair),
            "true_safe_repair": sum(r["diagnostic_true_nominal_baseline_repair_true_safe_NOT_DEPLOYABLE"] is True for r in repair),
            "false_safe_repair": sum(r["diagnostic_true_nominal_baseline_repair_true_safe_NOT_DEPLOYABLE"] is False for r in repair)},
        "development_previously_inspected_not_untouched": True,
        "candidate_truth_reuses_exact_G101_SOFA_forks": True,
        "confirmation_108000_108029_opened": False}
    json_new(OUT / "G101_vs_G102_comparison.json", compare)
    print(json.dumps(compare, indent=2), flush=True)


if __name__ == "__main__":
    evaluate()
