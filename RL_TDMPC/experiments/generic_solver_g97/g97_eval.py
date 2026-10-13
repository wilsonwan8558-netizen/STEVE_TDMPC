"""Frozen scalar H2 prospective risk metrics and state-aliasing diagnostic."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
for folder in ("generic_solver_g97", "generic_solver_g96", "generic_solver_g95", "generic_solver_g9"):
    sys.path.insert(0, str(RL / "experiments" / folder))

from g97_prepare import OUT, G95, SEEDS, F_LIM, PCA_DIM, MODEL_INPUT_DIM, source_gate  # noqa: E402
from g97_train import FAMILIES, inputs, matrix, load_replica  # noqa: E402
from g95_common import json_new, sha  # noqa: E402
from g95_eval import calculate, csv_new  # noqa: E402
from g9_risk import predict  # noqa: E402

CATEGORIES = ("SAFE-SAFE", "SAFE-HIGH", "HIGH-SAFE", "HIGH-HIGH")
METRICS = ("Q95_false_safe_rate", "safe_plan_rejection_rate", "unsafe_plan_acceptance_rate",
           "mean_within_state_spearman", "mean_within_state_pairwise_accuracy",
           "false_predicted_infeasibility_rate", "usable_state_fraction_among_feasible",
           "Q95_raw_coverage", "Q95_corrected_coverage", "four_quantile_pinball_N")


def categories(arrays):
    high1 = arrays["F1_true_N"] > F_LIM
    high2 = arrays["F2_true_N"] > F_LIM
    return np.asarray(CATEGORIES, dtype="U10")[high1.astype(int)*2 + high2.astype(int)]


def score_arrays(arrays, raw_states, cohort):
    source_gate()
    if raw_states.shape[1] != 3582 or arrays["x"].shape[1] != 298:
        raise RuntimeError("G9.7 score input dimension changed")
    with np.load(OUT / "privileged_input_adapter.npz") as z:
        adapter = {key: z[key].copy() for key in z.files}
    manifest = json.loads((OUT / "training_manifest.json").read_text())
    if manifest["privileged_adapter_sha256"] != sha(OUT / "privileged_input_adapter.npz"):
        raise RuntimeError("G9.7 frozen adapter changed")
    state_data = {"raw": raw_states}
    labels = categories(arrays)
    y = arrays["R2_true_N"].astype(np.float64)
    model_rows, temporal_rows, state_rows, plan_rows = [], [], [], []
    for family in FAMILIES:
        x = matrix(arrays, state_data, adapter, family)
        for seed in SEEDS:
            net, mean, scale, delta = load_replica(family, seed)
            if mean.shape != (MODEL_INPUT_DIM,) or scale.shape != (MODEL_INPUT_DIM,):
                raise RuntimeError("G9.7 frozen model normalization dimension changed")
            p = predict(net, x, mean, scale)
            row, per_state = calculate(family, seed, arrays, p, delta)
            model_rows.append(row)
            state_rows.extend(per_state)
            score = p[:, 2] + delta
            safe = score <= F_LIM
            for category in CATEGORIES:
                mask = labels == category
                count = int(np.sum(mask))
                temporal_rows.append({"model": family, "training_seed": seed,
                    "temporal_category": category, "count": count,
                    "predicted_safe_count": int(np.sum(safe[mask])),
                    "predicted_unsafe_count": int(np.sum(~safe[mask])),
                    "predicted_safe_rate": float(np.mean(safe[mask])) if count else None,
                    "miss_count_if_true_unsafe": int(np.sum(safe[mask])) if category != "SAFE-SAFE" else None,
                    "miss_rate_if_true_unsafe": float(np.mean(safe[mask])) if category != "SAFE-SAFE" and count else None})
            for i in range(len(y)):
                plan_rows.append({"cohort": cohort, "model": family, "training_seed": seed,
                    "episode_seed": int(arrays["seed"][i]),
                    "state_id": int(arrays["state_global_id"][i]),
                    "decision_index": int(arrays["decision_index"][i]),
                    "candidate_id": int(arrays["candidate_id"][i]),
                    "temporal_category": str(labels[i]),
                    "F1_true_N": float(arrays["F1_true_N"][i]),
                    "F2_true_N": float(arrays["F2_true_N"][i]),
                    "R2_true_N": float(y[i]),
                    "raw_Q50_N": float(p[i, 0]), "raw_Q90_N": float(p[i, 1]),
                    "raw_Q95_N": float(p[i, 2]), "raw_Q99_N": float(p[i, 3]),
                    "validation_Q95_delta_N": float(delta),
                    "corrected_Q95_N": float(score[i]),
                    "predicted_safe": int(safe[i]),
                    "true_safe": int(y[i] <= F_LIM)})
            sh = temporal_rows[-3]
            print(f"G9.7 {cohort} {family}/{seed}: safe_reject={row['safe_plan_rejection_rate']:.3f} unsafe_accept={row['unsafe_plan_acceptance_rate']:.3f} SAFE-HIGH miss={sh['miss_count_if_true_unsafe']}/{sh['count']} rank={row['mean_within_state_spearman']:.3f}", flush=True)
    return model_rows, temporal_rows, state_rows, plan_rows


def robustness(model_rows, temporal_rows):
    result = {}
    higher_bad = {"Q95_false_safe_rate", "safe_plan_rejection_rate",
                  "unsafe_plan_acceptance_rate", "false_predicted_infeasibility_rate",
                  "four_quantile_pinball_N"}
    for family in FAMILIES:
        replicas = {}
        for seed in SEEDS:
            row = next(row for row in model_rows if row["model"] == family and row["training_seed"] == seed)
            sh = next(row for row in temporal_rows if row["model"] == family and
                      row["training_seed"] == seed and row["temporal_category"] == "SAFE-HIGH")
            replicas[str(seed)] = {**row, "SAFE_HIGH_miss_count": sh["miss_count_if_true_unsafe"],
                                   "SAFE_HIGH_miss_rate": sh["miss_rate_if_true_unsafe"]}
        summary = {}
        for key in (*METRICS, "SAFE_HIGH_miss_rate"):
            values = np.asarray([replicas[str(seed)][key] for seed in SEEDS], dtype=np.float64)
            if not np.all(np.isfinite(values)):
                summary[key] = None
            else:
                summary[key] = {"mean": float(np.mean(values)),
                    "std_population": float(np.std(values)),
                    "worst_replica": float(np.max(values) if key in higher_bad or key == "SAFE_HIGH_miss_rate" else np.min(values))}
        result[family] = {"replicas": replicas, "summary": summary}
    return result


def paired_comparison(model_rows, temporal_rows, cohort):
    paired = []
    keys = ("Q95_false_safe_rate", "safe_plan_rejection_rate", "unsafe_plan_acceptance_rate",
            "mean_within_state_spearman", "mean_within_state_pairwise_accuracy",
            "false_predicted_infeasibility_rate", "usable_state_fraction_among_feasible",
            "Q95_raw_coverage", "Q95_corrected_coverage", "four_quantile_pinball_N")
    for seed in SEEDS:
        deploy = next(r for r in model_rows if r["model"] == "M_DEPLOY" and r["training_seed"] == seed)
        priv = next(r for r in model_rows if r["model"] == "M_PRIV" and r["training_seed"] == seed)
        dsh = next(r for r in temporal_rows if r["model"] == "M_DEPLOY" and
                   r["training_seed"] == seed and r["temporal_category"] == "SAFE-HIGH")
        psh = next(r for r in temporal_rows if r["model"] == "M_PRIV" and
                   r["training_seed"] == seed and r["temporal_category"] == "SAFE-HIGH")
        paired.append({"training_seed": seed,
            **{f"priv_minus_deploy_{key}": float(priv[key]-deploy[key]) for key in keys},
            "deploy_SAFE_HIGH_miss_count": dsh["miss_count_if_true_unsafe"],
            "priv_SAFE_HIGH_miss_count": psh["miss_count_if_true_unsafe"],
            "priv_minus_deploy_SAFE_HIGH_miss_count": psh["miss_count_if_true_unsafe"]-dsh["miss_count_if_true_unsafe"]})
    return {"status": "G97_ALL_SIX_FROZEN_MODELS_IDENTICAL_PLANS", "cohort": cohort,
        "same_plans_per_model": model_rows[0]["candidate_rows"],
        "same_parent_states_per_model": model_rows[0]["planner_states"],
        "paired_seed_deltas": paired,
        "training_seed_robustness": robustness(model_rows, temporal_rows),
        "no_best_seed_selection": True,
        "classification": "See PHASE_G97_REPORT.md after untouched confirmation"}


def identifiability(arrays, raw_states):
    """Diagnostic nearest-neighbor target variance, no fit or model selection."""
    with np.load(OUT / "privileged_input_adapter.npz") as z:
        adapter = {key: z[key].copy() for key in z.files}
    state_data = {"raw": raw_states}
    x_deploy = matrix(arrays, state_data, adapter, "M_DEPLOY")
    x_priv = matrix(arrays, state_data, adapter, "M_PRIV")
    _, deploy_mean, deploy_scale, _ = load_replica("M_DEPLOY", SEEDS[0])
    _, priv_mean, priv_scale, _ = load_replica("M_PRIV", SEEDS[0])
    deploy = (x_deploy[:, :294]-deploy_mean[:294])/deploy_scale[:294]
    priv = (x_priv[:, :588]-priv_mean[:588])/priv_scale[:588]
    actions = arrays["action_pair"].reshape(len(deploy), 4).astype(np.float64)
    y = arrays["R2_true_N"].astype(np.float64)
    ids = arrays["state_global_id"]
    rows = []
    for i in range(len(y)):
        eligible = np.flatnonzero((ids != ids[i]) & (np.linalg.norm(actions-actions[i], axis=1) <= .75))
        if len(eligible) < 5:
            continue
        record = {"episode_seed": int(arrays["seed"][i]),
            "state_id": int(ids[i]), "candidate_id": int(arrays["candidate_id"][i]),
            "eligible_other_state_similar_action_plans": len(eligible),
            "query_R2_N": float(y[i])}
        for name, state_space in (("deploy", deploy), ("priv", priv)):
            distance = np.linalg.norm(state_space[eligible]-state_space[i], axis=1)
            nearest = eligible[np.argsort(distance, kind="stable")[:5]]
            local = np.r_[y[i], y[nearest]]
            record[f"{name}_local_R2_variance_N2"] = float(np.var(local))
            record[f"{name}_local_R2_mean_abs_query_difference_N"] = float(np.mean(np.abs(y[nearest]-y[i])))
            record[f"{name}_mean_state_distance"] = float(np.mean(distance[np.argsort(distance, kind="stable")[:5]]))
            record[f"{name}_neighbor_state_ids"] = json.dumps(ids[nearest].astype(int).tolist(), separators=(",", ":"))
        rows.append(record)
    if not rows:
        raise RuntimeError("G9.7 no similar-action held-out neighborhoods")
    summary = {"status": "DIAGNOSTIC_ONLY_NO_LABEL_FIT", "heldout_plans": len(y),
        "queries_with_five_other_state_similar_action_neighbors": len(rows),
        "action_pair_L2_max": .75, "neighbor_count": 5,
        "mean_deploy_local_R2_variance_N2": float(np.mean([r["deploy_local_R2_variance_N2"] for r in rows])),
        "mean_priv_local_R2_variance_N2": float(np.mean([r["priv_local_R2_variance_N2"] for r in rows])),
        "mean_deploy_abs_query_neighbor_R2_difference_N": float(np.mean([r["deploy_local_R2_mean_abs_query_difference_N"] for r in rows])),
        "mean_priv_abs_query_neighbor_R2_difference_N": float(np.mean([r["priv_local_R2_mean_abs_query_difference_N"] for r in rows])),
        "interpretation_limit": "nearest-neighbor variance is descriptive and depends on train-only scaling/PCA; not a causal proof of aliasing or Markovity"}
    return rows, summary


def development():
    planner, states = inputs()
    mask = planner["group_index"] == 2
    arrays = {key: value[mask] for key, value in planner.items()}
    rows, events, state_rows, plans = score_arrays(arrays, states["raw"], "development_test")
    csv_new(OUT / "development_metrics.csv", rows)
    csv_new(OUT / "temporal_category_metrics.csv", events)
    csv_new(OUT / "state_ranking_metrics.csv", state_rows)
    csv_new(OUT / "development_plan_results.csv", plans)
    alias_rows, alias_summary = identifiability(arrays, states["raw"])
    csv_new(OUT / "identifiability_analysis.csv", alias_rows)
    json_new(OUT / "identifiability_summary.json", alias_summary)
    json_new(OUT / "training_seed_robustness.json", robustness(rows, events))
    json_new(OUT / "development_deploy_vs_privileged_comparison.json",
             paired_comparison(rows, events, "development_test"))
    print("G9.7 development matched-plan evaluation and identifiability saved", flush=True)


if __name__ == "__main__":
    development()
