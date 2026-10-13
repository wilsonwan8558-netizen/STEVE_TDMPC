"""Matched scalar/stepwise planner-plan, temporal-category, and state metrics."""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import rankdata

from g96_common import (OUT, G95, SEEDS, F_LIM, TAUS, assert_prepared,
                        training_arrays, predict_profile, json_new, sha)
from g96_train import load_replica
from g95_eval import load_replica as load_scalar, per_state_metrics, auc_ap, csv_new
from g9_risk import predict as predict_scalar

CATEGORIES = ("SAFE-SAFE", "SAFE-HIGH", "HIGH-SAFE", "HIGH-HIGH")


def score_source(arrays, model, seed, mean, scale, delta):
    if model == "M_scalar":
        net, base_mean, base_scale, scalar_delta = load_scalar("M1_planner_conditioned", seed)
        if not np.array_equal(base_mean, mean) or not np.array_equal(base_scale, scale):
            raise RuntimeError("G9.6 scalar and stepwise input normalization differ")
        raw = predict_scalar(net, arrays["x"], base_mean, base_scale)
        score = raw[:, 2]+scalar_delta
        return raw, None, score, scalar_delta
    net = load_replica(seed)
    profile = predict_profile(net, arrays["x"], mean, scale)
    score = np.max(profile[:, :, 2]+delta[None, :], axis=1)
    return None, profile, score, None


def categories_for(arrays):
    first_high = arrays["F1_true_N"] > F_LIM
    second_high = arrays["F2_true_N"] > F_LIM
    index = first_high.astype(int)*2 + second_high.astype(int)
    return np.asarray(CATEGORIES, dtype="U10")[index]


def secondary_rank(y, score):
    severity = np.maximum(y-F_LIM, 0)
    if len(y) < 3 or np.ptp(severity) <= 1e-8 or np.ptp(score) <= 1e-8:
        spearman = None
    else:
        spearman = float(np.corrcoef(rankdata(severity), rankdata(score))[0, 1])
    pairs = good = ties = 0
    for i in range(len(y)):
        for j in range(i+1, len(y)):
            if abs(float(severity[i]-severity[j])) <= 1e-8:
                continue
            pairs += 1
            relation = float((severity[i]-severity[j])*(score[i]-score[j]))
            good += relation > 0
            ties += relation == 0
    return spearman, (good+0.5*ties)/pairs if pairs else None, pairs


def assess(arrays, model, seed, raw, profile, score, scalar_delta, shared_delta):
    y = arrays["R2_true_N"].astype(np.float64)
    f1 = arrays["F1_true_N"].astype(np.float64)
    f2 = arrays["F2_true_N"].astype(np.float64)
    if (len(y) != len(score) or not np.array_equal(np.maximum(f1, f2).astype(np.float32),
                                                    arrays["R2_true_N"]) or
            np.any(~np.isfinite(score))):
        raise RuntimeError("G9.6 true/score plan alignment failed")
    accepted = score <= F_LIM
    unsafe = y > F_LIM
    labels = categories_for(arrays)
    if int(np.sum(~unsafe)) == 0 or int(np.sum(unsafe)) == 0:
        raise RuntimeError("G9.6 plan cohort lacks both safety classes")
    state_rows = []
    for state_id in np.unique(arrays["state_global_id"]):
        idx = np.flatnonzero(arrays["state_global_id"] == state_id)
        if len(set(arrays["seed"][idx])) != 1 or len(set(arrays["decision_index"][idx])) != 1:
            raise RuntimeError("G9.6 candidate rows crossed state/episode boundary")
        row = per_state_metrics(state_id, arrays["seed"][idx[0]], arrays["decision_index"][idx[0]],
            arrays["candidate_id"][idx], y[idx], score[idx], accepted[idx], model, seed)
        secondary_spearman, secondary_pairwise, secondary_pairs = secondary_rank(y[idx], score[idx])
        row.update({"secondary_temporal_severity_spearman": secondary_spearman,
                    "secondary_temporal_severity_pairwise_accuracy": secondary_pairwise,
                    "secondary_temporal_severity_pairs": secondary_pairs})
        state_rows.append(row)
    feasible = [row for row in state_rows if row["has_true_safe_plan"]]
    finite_mean = lambda key: (float(np.mean([row[key] for row in state_rows if row[key] is not None]))
                               if any(row[key] is not None for row in state_rows) else None)
    auc, ap = auc_ap(unsafe, score)
    plan_row = {"model": model, "training_seed": seed, "candidate_rows": len(y),
        "planner_states": len(state_rows), "true_safe_plans": int(np.sum(~unsafe)),
        "true_unsafe_plans": int(np.sum(unsafe)), "predicted_safe_plans": int(np.sum(accepted)),
        "safe_plan_accepted": int(np.sum((~unsafe)&accepted)),
        "safe_plan_rejected": int(np.sum((~unsafe)&(~accepted))),
        "safe_plan_acceptance_rate": float(np.mean(accepted[~unsafe])),
        "safe_plan_rejection_rate": float(np.mean((~accepted)[~unsafe])),
        "unsafe_plan_accepted": int(np.sum(unsafe&accepted)),
        "unsafe_plan_rejected": int(np.sum(unsafe&(~accepted))),
        "unsafe_plan_acceptance_rate": float(np.mean(accepted[unsafe])),
        "unsafe_plan_rejection_rate": float(np.mean((~accepted)[unsafe])),
        "Q95_false_safe_count": int(np.sum(unsafe&accepted)),
        "Q95_false_safe_rate": float(np.mean(accepted[unsafe])),
        "derived_score_empirical_R2_coverage": float(np.mean(y <= score)),
        "AUROC_unsafe": auc, "AUPRC_unsafe": ap,
        "states_with_true_safe_plan": len(feasible),
        "fraction_states_with_true_safe_plan": len(feasible)/len(state_rows),
        "states_with_usable_true_safe_plan": sum(row["leaves_at_least_one_usable_safe_plan"] for row in feasible),
        "usable_state_fraction_among_feasible": sum(row["leaves_at_least_one_usable_safe_plan"] for row in feasible)/len(feasible) if feasible else None,
        "false_predicted_infeasible_states": sum(row["false_predicted_infeasible"] for row in feasible),
        "false_predicted_infeasibility_rate": sum(row["false_predicted_infeasible"] for row in feasible)/len(feasible) if feasible else None,
        "mean_within_state_spearman": finite_mean("within_state_spearman_Q95_vs_R2"),
        "mean_within_state_pairwise_accuracy": finite_mean("pairwise_risk_ranking_accuracy"),
        "mean_top3_high_risk_containment": finite_mean("top3_high_risk_containment"),
        "mean_top3_low_risk_containment": finite_mean("top3_low_risk_containment"),
        "mean_secondary_temporal_severity_spearman": finite_mean("secondary_temporal_severity_spearman"),
        "mean_secondary_temporal_severity_pairwise_accuracy": finite_mean("secondary_temporal_severity_pairwise_accuracy"),
        "scalar_Q95_delta_N": scalar_delta,
        "stepwise_shared_delta1_N": shared_delta[0] if model == "M_step" else None,
        "stepwise_shared_delta2_N": shared_delta[1] if model == "M_step" else None,
        "score_interpretation": "corrected scalar Q95(R2)" if model == "M_scalar" else
                                "diagnostic max of separately corrected Q95(F1),Q95(F2); not a learned scalar R2 quantile"}
    event_rows = []
    for category in CATEGORIES:
        mask = labels == category
        count = int(np.sum(mask))
        first_flag = profile[:, 0, 2]+shared_delta[0] > F_LIM if profile is not None else None
        second_flag = profile[:, 1, 2]+shared_delta[1] > F_LIM if profile is not None else None
        event_rows.append({"model": model, "training_seed": seed, "temporal_category": category,
            "count": count, "predicted_safe_count": int(np.sum(accepted[mask])),
            "predicted_unsafe_count": int(np.sum((~accepted)[mask])),
            "predicted_safe_rate": float(np.mean(accepted[mask])) if count else None,
            "miss_count_if_true_unsafe": int(np.sum(accepted[mask])) if category != "SAFE-SAFE" else None,
            "miss_rate_if_true_unsafe": float(np.mean(accepted[mask])) if category != "SAFE-SAFE" and count else None,
            "recall_if_true_unsafe": float(np.mean((~accepted)[mask])) if category != "SAFE-SAFE" and count else None,
            "safe_plan_rejection_if_safe_safe": int(np.sum((~accepted)[mask])) if category == "SAFE-SAFE" else None,
            "step1_high_flagged_count": int(np.sum(first_flag[mask])) if first_flag is not None else None,
            "step2_high_flagged_count": int(np.sum(second_flag[mask])) if second_flag is not None else None})
    plan_rows = []
    for i in range(len(y)):
        plan_rows.append({"model": model, "training_seed": seed,
            "episode_seed": int(arrays["seed"][i]), "state_id": int(arrays["state_global_id"][i]),
            "decision_index": int(arrays["decision_index"][i]),
            "candidate_id": int(arrays["candidate_id"][i]),
            "temporal_category": str(labels[i]), "F1_true_N": f1[i], "F2_true_N": f2[i],
            "R2_true_N": y[i], "true_safe": int(not unsafe[i]),
            "scalar_Q50_N": float(raw[i, 0]) if raw is not None else None,
            "scalar_Q90_N": float(raw[i, 1]) if raw is not None else None,
            "scalar_Q95_N": float(raw[i, 2]) if raw is not None else None,
            "scalar_Q99_N": float(raw[i, 3]) if raw is not None else None,
            "step_F1_Q50_N": float(profile[i, 0, 0]) if profile is not None else None,
            "step_F1_Q90_N": float(profile[i, 0, 1]) if profile is not None else None,
            "step_F1_Q95_N": float(profile[i, 0, 2]) if profile is not None else None,
            "step_F1_Q99_N": float(profile[i, 0, 3]) if profile is not None else None,
            "step_F2_Q50_N": float(profile[i, 1, 0]) if profile is not None else None,
            "step_F2_Q90_N": float(profile[i, 1, 1]) if profile is not None else None,
            "step_F2_Q95_N": float(profile[i, 1, 2]) if profile is not None else None,
            "step_F2_Q99_N": float(profile[i, 1, 3]) if profile is not None else None,
            "step_F1_Q95_corrected_N": float(profile[i, 0, 2]+shared_delta[0]) if profile is not None else None,
            "step_F2_Q95_corrected_N": float(profile[i, 1, 2]+shared_delta[1]) if profile is not None else None,
            "decision_risk_score_N": float(score[i]), "predicted_safe": int(accepted[i])})
    return plan_row, event_rows, state_rows, plan_rows


def coverage_rows(arrays, seed, profile, shared_delta):
    output = []
    for step in range(2):
        y = arrays[("F1_true_N", "F2_true_N")[step]].astype(np.float64)
        for q, tau in enumerate(TAUS):
            p = profile[:, step, q]
            coverage = float(np.mean(y <= p))
            loss = float(np.mean(np.maximum(tau*(y-p), (tau-1)*(y-p))))
            output.append({"model": "M_step", "training_seed": seed, "step": step+1,
                "quantile": f"Q{int(tau*100)}", "tau": tau, "candidate_rows": len(y),
                "empirical_coverage": coverage, "calibration_error": coverage-tau,
                "mean_pinball_N": loss,
                "corrected_Q95_coverage": float(np.mean(y <= profile[:, step, 2]+shared_delta[step])) if q == 2 else None,
                "shared_Q95_delta_N": shared_delta[step] if q == 2 else None})
    return output


def evaluate_arrays(arrays, prefix):
    _, _, _, _, _, _, mean, scale = training_arrays()
    calibration = json.loads((OUT / "calibration_result.json").read_text())
    if (calibration["status"] != "VALIDATION_ONLY_SHARED_STEP_CORRECTION_FROZEN_BEFORE_DEVELOPMENT_TEST" or
            calibration["training_manifest_sha256"] != sha(OUT / "training_manifest.json")):
        raise RuntimeError("G9.6 shared validation calibration changed")
    delta = np.asarray([calibration["shared_delta1_N"], calibration["shared_delta2_N"]], dtype=np.float64)
    model_rows, events, states, plans, coverages = [], [], [], [], []
    for model in ("M_scalar", "M_step"):
        for seed in SEEDS:
            raw, profile, score, scalar_delta = score_source(arrays, model, seed, mean, scale, delta)
            row, event_rows, state_rows, plan_rows = assess(arrays, model, seed, raw, profile,
                                                             score, scalar_delta, delta)
            model_rows.append(row)
            events.extend(event_rows)
            states.extend(state_rows)
            plans.extend(plan_rows)
            if model == "M_step":
                coverages.extend(coverage_rows(arrays, seed, profile, delta))
            sh = next(e for e in event_rows if e["temporal_category"] == "SAFE-HIGH")
            print(f"G9.6 {prefix} {model}/{seed}: safe_reject={row['safe_plan_rejection_rate']:.3f} unsafe_accept={row['unsafe_plan_acceptance_rate']:.3f} SAFE-HIGH miss={sh['miss_count_if_true_unsafe']}/{sh['count']} rank={row['mean_within_state_spearman']:.3f}", flush=True)
    return model_rows, events, states, plans, coverages


def robustness(model_rows, event_rows):
    result = {}
    higher_bad = {"safe_plan_rejection_rate", "unsafe_plan_acceptance_rate",
                  "false_predicted_infeasibility_rate", "SAFE_HIGH_miss_rate"}
    keys = ("safe_plan_rejection_rate", "unsafe_plan_acceptance_rate",
            "mean_within_state_spearman", "mean_within_state_pairwise_accuracy",
            "false_predicted_infeasibility_rate", "SAFE_HIGH_miss_rate")
    for model in ("M_scalar", "M_step"):
        replicas = {}
        for seed in SEEDS:
            row = next(row for row in model_rows if row["model"] == model and row["training_seed"] == seed)
            event = next(row for row in event_rows if row["model"] == model and
                         row["training_seed"] == seed and row["temporal_category"] == "SAFE-HIGH")
            replicas[str(seed)] = {**row, "SAFE_HIGH_miss_rate": event["miss_rate_if_true_unsafe"],
                                   "SAFE_HIGH_miss_count": event["miss_count_if_true_unsafe"],
                                   "SAFE_HIGH_count": event["count"]}
        summary = {}
        for key in keys:
            observed = [replicas[str(seed)][key] for seed in SEEDS]
            if any(value is None for value in observed):
                summary[key] = None
            else:
                values = np.asarray(observed, dtype=np.float64)
                summary[key] = {"mean": float(np.mean(values)), "std_population": float(np.std(values)),
                    "worst_replica": float(np.max(values) if key in higher_bad else np.min(values))}
        result[model] = {"replicas": replicas, "summary": summary}
    return result


def development():
    assert_prepared()
    if any((OUT / name).exists() for name in ("development_stepwise_coverage.csv",
          "development_plan_metrics.csv", "temporal_event_metrics.csv",
          "state_ranking_metrics.csv", "training_seed_robustness.json")):
        raise FileExistsError("G9.6 development evaluation already saved")
    with np.load(G95 / "development_planner_dataset.npz") as z:
        d = {key: z[key].copy() for key in z.files}
    mask = d["group_index"] == 2
    arrays = {key: value[mask] for key, value in d.items()}
    if len(arrays["x"]) != 1075 or len(set(arrays["state_global_id"])) != 120:
        raise RuntimeError("G9.6 held-out development cohort changed")
    model_rows, events, states, plans, coverage = evaluate_arrays(arrays, "development")
    csv_new(OUT / "development_stepwise_coverage.csv", coverage)
    csv_new(OUT / "development_plan_metrics.csv", model_rows)
    csv_new(OUT / "temporal_event_metrics.csv", events)
    csv_new(OUT / "state_ranking_metrics.csv", states)
    csv_new(OUT / "development_plan_results.csv", plans)
    json_new(OUT / "training_seed_robustness.json", {"status": "ALL_THREE_SEEDS_MATCHED_DEVELOPMENT_PLANS",
        "candidate_rows": len(arrays["x"]), "planner_states": 120,
        "models": robustness(model_rows, events), "no_best_seed_selection": True})
    print("G9.6 complete held-out development evaluation saved; 104000-series still unopened", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("development",), required=True)
    if parser.parse_args().stage == "development":
        development()
