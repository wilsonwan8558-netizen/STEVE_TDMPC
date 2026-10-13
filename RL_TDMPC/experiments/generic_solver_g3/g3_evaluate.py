"""Validation freeze, then held-out G3 development diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from scipy.stats import pearsonr, rankdata, spearmanr

from prepare import OUT, G2, sha
from g3_model_train import model, behavior

MODELS = ("M0_behavior", "M1_counterfactual")
QUANTILE_KEYS = ("q90", "q95", "q97.5", "q99")
PAIR_TIE_TOLERANCE_N = 1e-6


def json_new(path: Path, obj):
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False)+"\n")


def csv_new(path: Path, rows):
    if path.exists():
        raise FileExistsError(path)
    if not rows:
        raise RuntimeError(f"No rows for {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_predictor(name, device):
    folder = OUT / "models" / name
    norm = json.loads((folder / "normalization.json").read_text())
    net = model().to(device)
    net.load_state_dict(torch.load(folder / "model.pt", map_location=device, weights_only=True))
    net.eval()
    return net, np.asarray(norm["input_mean"]), np.asarray(norm["input_std"])


def predict(predictor, x, device):
    net, mean, scale = predictor
    start = time.perf_counter()
    with torch.inference_mode():
        output = np.concatenate([net(torch.as_tensor((x[i:i+4096]-mean)/scale, dtype=torch.float32, device=device)).squeeze(-1).cpu().numpy()
                                 for i in range(0, len(x), 4096)]).astype(float)
    return output, time.perf_counter()-start


def corr_spearman(x, y):
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(spearmanr(x, y).statistic)


def classification(y, p, q):
    truth = y > q
    pred = p > q
    tp = int(np.sum(truth & pred))
    fn = int(np.sum(truth & ~pred))
    fp = int(np.sum(~truth & pred))
    tn = int(np.sum(~truth & ~pred))
    positive, negative = tp+fn, fp+tn
    ranks = rankdata(p, method="average")
    auroc = float((np.sum(ranks[truth])-positive*(positive+1)/2)/(positive*negative)) if positive and negative else None
    if positive:
        # Tie-aware average precision: evaluate precision at each distinct score.
        order = np.argsort(-p, kind="stable")
        scores, labels = p[order], truth[order]
        ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores)-1]
        cumulative = np.cumsum(labels)
        prior = 0
        ap = 0.0
        for end in ends:
            current = int(cumulative[end])
            ap += (current-prior)/positive * current/(end+1)
            prior = current
    else:
        ap = None
    severity = y[truth & ~pred]-q
    return {"candidate_count": len(y), "true_exceedances": positive, "predicted_exceedances": tp+fp,
            "TP": tp, "FN_false_safe": fn, "FP_false_unsafe": fp, "TN": tn,
            "unsafe_recall": tp/positive if positive else None,
            "precision": tp/(tp+fp) if tp+fp else None,
            "specificity": tn/negative if negative else None,
            "NPV": tn/(tn+fn) if tn+fn else None,
            "false_safe_rate_P_pred_safe_given_true_unsafe": fn/positive if positive else None,
            "false_unsafe_rate_P_pred_unsafe_given_true_safe": fp/negative if negative else None,
            "AUROC": auroc, "AUPRC": ap,
            "false_safe_severity_mean_N": float(np.mean(severity)) if len(severity) else None,
            "false_safe_severity_median_N": float(np.median(severity)) if len(severity) else None,
            "false_safe_severity_max_N": float(np.max(severity)) if len(severity) else None}


def regression(y, p):
    residual = p-y
    ss = float(np.sum((y-np.mean(y))**2))
    return {"count": len(y), "RMSE_N": float(np.sqrt(np.mean(residual**2))),
            "MAE_N": float(np.mean(np.abs(residual))), "R2": float(1-np.sum(residual**2)/ss) if ss > 0 else None,
            "Pearson": float(pearsonr(y, p).statistic) if np.ptp(y) and np.ptp(p) else None,
            "Spearman": corr_spearman(y, p)}


def states_for(group, split):
    paths = sorted((OUT / "development_candidates" / group).glob("state_*.json"))
    expected = 120
    if len(paths) != expected:
        raise RuntimeError(f"Expected {expected} {group} states, got {len(paths)}")
    states = [json.loads(path.read_text()) for path in paths]
    if any(state["group"] != group or state["seed"] not in split[group] or len(state["candidates"]) != 21 for state in states):
        raise RuntimeError(f"Invalid {group} state data")
    return states


def state_metrics(state, prediction):
    y = np.asarray([c["force_next_max_N"] for c in state["candidates"]], dtype=float)
    p = np.asarray(prediction, dtype=float)
    if y.shape != (21,) or p.shape != (21,):
        raise RuntimeError("Candidate profile length differs")
    true_order = np.argsort(y)
    predicted_order = np.argsort(p)
    pairs = np.triu_indices(21, 1)
    differences = y[pairs[0]]-y[pairs[1]]
    predicted_differences = p[pairs[0]]-p[pairs[1]]
    non_tied = np.abs(differences) > PAIR_TIE_TOLERANCE_N
    pair_accuracy = float(np.mean(np.sign(differences[non_tied]) == np.sign(predicted_differences[non_tied]))) if np.any(non_tied) else None
    selected = int(predicted_order[0])
    max_action = int(true_order[-1])
    min_action = int(true_order[0])
    ranks = np.empty(21, dtype=int)
    ranks[predicted_order] = np.arange(1, 22)
    return {"seed": state["seed"], "decision_index": state["decision_index"], "group": state["group"],
            "state_id": state["state_id"], "spearman": corr_spearman(y, p),
            "pairwise_accuracy": pair_accuracy, "profile_RMSE_N": float(np.sqrt(np.mean((p-y)**2))),
            "profile_MAE_N": float(np.mean(np.abs(p-y))),
            "predicted_min_N": float(np.min(p)), "predicted_max_N": float(np.max(p)), "predicted_range_N": float(np.ptp(p)),
            "true_min_N": float(np.min(y)), "true_max_N": float(np.max(y)), "true_range_N": float(np.ptp(y)),
            "true_max_action_predicted_rank": int(ranks[max_action]),
            "true_min_action_predicted_rank": int(ranks[min_action]),
            "top3_high_containment": len(set(np.argsort(y)[-3:]) & set(np.argsort(p)[-3:]))/3,
            "top3_low_containment": len(set(np.argsort(y)[:3]) & set(np.argsort(p)[:3]))/3,
            "safe_choice_regret_N": float(y[selected]-np.min(y)),
            "selected_lowest_predicted_candidate_id": selected,
            "true_min_candidate_id": min_action, "true_max_candidate_id": max_action}


def evaluate_candidates(states, predictors, device):
    x = np.asarray([np.r_[np.asarray(state["latent"], dtype=float), np.asarray(candidate["action"], dtype=float)]
                    for state in states for candidate in state["candidates"]], dtype=np.float64)
    y = np.asarray([candidate["force_next_max_N"] for state in states for candidate in state["candidates"]], dtype=float)
    outputs = {}
    for name, predictor in predictors.items():
        p, seconds = predict(predictor, x, device)
        metrics = [{"model": name, **state_metrics(state, p[i*21:(i+1)*21])} for i, state in enumerate(states)]
        outputs[name] = {"truth": y, "prediction": p, "state_metrics": metrics, "inference_seconds": seconds}
    return outputs


def aggregate_candidate(output):
    records = output["state_metrics"]
    def values(key):
        return np.asarray([record[key] for record in records if record[key] is not None], dtype=float)
    return {"candidate_RMSE_N": regression(output["truth"], output["prediction"])["RMSE_N"],
            "candidate_MAE_N": regression(output["truth"], output["prediction"])["MAE_N"],
            "state_count": len(records), "median_within_state_Spearman": float(np.median(values("spearman"))) if len(values("spearman")) else None,
            "mean_pairwise_accuracy": float(np.mean(values("pairwise_accuracy"))) if len(values("pairwise_accuracy")) else None,
            "median_pairwise_accuracy": float(np.median(values("pairwise_accuracy"))) if len(values("pairwise_accuracy")) else None,
            "mean_safe_choice_regret_N": float(np.mean(values("safe_choice_regret_N"))),
            "median_safe_choice_regret_N": float(np.median(values("safe_choice_regret_N"))),
            "mean_top3_high_containment": float(np.mean(values("top3_high_containment"))),
            "mean_top3_low_containment": float(np.mean(values("top3_low_containment"))),
            "inference_seconds_total": output["inference_seconds"]}


def quantiles():
    source = json.loads((G2 / "force_distribution_summary.json").read_text())["all_rl_steps"]
    return {key: float(source[key]) for key in QUANTILE_KEYS}


def validation_freeze(device):
    if (OUT / "frozen_model_manifest.json").exists():
        raise FileExistsError("Model already frozen")
    split = json.loads((OUT / "development_split.json").read_text())
    states = states_for("validation", split)
    predictors = {name: load_predictor(name, device) for name in MODELS}
    outputs = evaluate_candidates(states, predictors, device)
    x_val, y_val = behavior("validation", split)
    metrics = []
    for name in MODELS:
        p_val, _ = predict(predictors[name], x_val, device)
        metrics.append({"model": name, **aggregate_candidate(outputs[name]),
                        "validation_behavior_RMSE_N": regression(y_val, p_val)["RMSE_N"],
                        "validation_behavior_MAE_N": regression(y_val, p_val)["MAE_N"]})
    csv_new(OUT / "validation_metrics.csv", metrics)
    # Predeclared selection: candidate ranking first; RMSE only breaks ties.
    def key(row):
        rho = row["median_within_state_Spearman"]
        return (-rho if rho is not None else float("inf"), row["candidate_RMSE_N"])
    selected = min(metrics, key=key)["model"]
    manifest = {"primary_selected_model": selected, "selection_rule": "validation median within-state Spearman highest; validation candidate RMSE tie-break",
                "validation_only": True, "development_test_labels_opened_before_freeze": False,
                "confirmation_seeds_opened_before_freeze": False, "architecture": "258-64-ReLU-64-ReLU-1, raw force MSE",
                "model_selection_protocol_sha256": sha(OUT / "model_selection_protocol.json"),
                "confirmation_seed_protocol_sha256": sha(OUT / "confirmation_seed_protocol.json"),
                "development_split_sha256": sha(OUT / "development_split.json"),
                "candidate_action_protocol_sha256": sha(OUT / "candidate_action_protocol.json"),
                "checkpoint_sha256": {name: sha(OUT / "models" / name / "model.pt") for name in MODELS},
                "normalization_sha256": {name: sha(OUT / "models" / name / "normalization.json") for name in MODELS},
                "comparison_models_frozen": list(MODELS), "validation_candidate_metrics": metrics}
    json_new(OUT / "frozen_model_manifest.json", manifest)
    print(f"Frozen primary model: {selected}", flush=True)


def behavior_metrics(name, predictor, split, q, device):
    x, y = behavior("development_test", split)
    p, seconds = predict(predictor, x, device)
    metrics = {"model": name, **regression(y, p), "inference_seconds": seconds}
    tail = []
    for key, level in q.items():
        mask = y > level
        error = y[mask]-p[mask]
        tail.append({"model": name, "reference": key, "reference_N": level, "above_count": int(np.sum(mask)),
                     "MAE_above_N": float(np.mean(np.abs(error))) if len(error) else None,
                     "mean_underprediction_N": float(np.mean(error)) if len(error) else None,
                     "median_underprediction_N": float(np.median(error)) if len(error) else None,
                     "false_safe_count": int(np.sum(mask & (p <= level)))})
    return metrics, tail


def bootstrap(outputs, q95):
    rng = np.random.default_rng(20261008)
    m0, m1 = outputs[MODELS[0]], outputs[MODELS[1]]
    n = len(m0["state_metrics"])
    if n != 120:
        raise RuntimeError("Bootstrap requires 120 paired states")
    names = ("within_state_Spearman", "pairwise_accuracy", "safe_choice_regret_N", "Q95_false_safe_rate", "Q95_AUPRC")
    draws = {name: [] for name in names}
    point = {}
    for name, field in (("within_state_Spearman", "spearman"), ("pairwise_accuracy", "pairwise_accuracy"),
                        ("safe_choice_regret_N", "safe_choice_regret_N")):
        paired = np.asarray([b[field]-a[field] for a, b in zip(m0["state_metrics"], m1["state_metrics"])
                             if a[field] is not None and b[field] is not None], dtype=float)
        point[name] = float(np.median(paired) if name == "within_state_Spearman" else np.mean(paired))
    full0 = classification(m0["truth"], m0["prediction"], q95)
    full1 = classification(m1["truth"], m1["prediction"], q95)
    for name, field in (("Q95_false_safe_rate", "false_safe_rate_P_pred_safe_given_true_unsafe"), ("Q95_AUPRC", "AUPRC")):
        point[name] = float(full1[field]-full0[field]) if full0[field] is not None and full1[field] is not None else None
    for _ in range(1000):
        indices = rng.integers(0, n, size=n)
        for name, field in (("within_state_Spearman", "spearman"), ("pairwise_accuracy", "pairwise_accuracy"),
                            ("safe_choice_regret_N", "safe_choice_regret_N")):
            paired = np.asarray([m1["state_metrics"][i][field]-m0["state_metrics"][i][field]
                                 for i in indices if m0["state_metrics"][i][field] is not None
                                 and m1["state_metrics"][i][field] is not None], dtype=float)
            if len(paired):
                draws[name].append(float(np.median(paired) if name == "within_state_Spearman" else np.mean(paired)))
        rows = np.concatenate([np.arange(i*21, (i+1)*21) for i in indices])
        c0 = classification(m0["truth"][rows], m0["prediction"][rows], q95)
        c1 = classification(m1["truth"][rows], m1["prediction"][rows], q95)
        for name, field in (("Q95_false_safe_rate", "false_safe_rate_P_pred_safe_given_true_unsafe"), ("Q95_AUPRC", "AUPRC")):
            if c0[field] is not None and c1[field] is not None:
                draws[name].append(float(c1[field]-c0[field]))
    return {name: {"replicates": len(draws[name]), "point_estimate_M1_minus_M0": point[name],
                   "M1_minus_M0_mean": float(np.mean(draws[name])) if draws[name] else None,
                   "CI95_percentile": [float(np.quantile(draws[name], .025)), float(np.quantile(draws[name], .975))] if draws[name] else None}
            for name in names}


def development_test(device):
    frozen = json.loads((OUT / "frozen_model_manifest.json").read_text())
    if not frozen["validation_only"] or frozen["development_test_labels_opened_before_freeze"]:
        raise RuntimeError("Model freeze invalid")
    for name in MODELS:
        if sha(OUT / "models" / name / "model.pt") != frozen["checkpoint_sha256"][name]:
            raise RuntimeError("Frozen checkpoint changed")
    split = json.loads((OUT / "development_split.json").read_text())
    q = quantiles()
    predictors = {name: load_predictor(name, device) for name in MODELS}
    behavior_rows, tail_rows = [], []
    for name in MODELS:
        metrics, tails = behavior_metrics(name, predictors[name], split, q, device)
        behavior_rows.append(metrics)
        tail_rows.extend(tails)
    csv_new(OUT / "development_behavior_metrics.csv", behavior_rows)
    csv_new(OUT / "development_behavior_tail_metrics.csv", tail_rows)
    states = states_for("development_test", split)
    outputs = evaluate_candidates(states, predictors, device)
    candidate_rows = [row for name in MODELS for row in outputs[name]["state_metrics"]]
    csv_new(OUT / "development_candidate_metrics.csv", candidate_rows)
    prediction_rows = []
    for name in MODELS:
        truth, prediction = outputs[name]["truth"], outputs[name]["prediction"]
        for i, state in enumerate(states):
            for j, candidate in enumerate(state["candidates"]):
                k = i*21+j
                prediction_rows.append({"model": name, "seed": state["seed"], "decision_index": state["decision_index"],
                                        "state_id": state["state_id"], "candidate_id": j, "candidate_kind": candidate["kind"],
                                        "action_translation": candidate["action"][0], "action_rotation": candidate["action"][1],
                                        "force_next_max_N": truth[k], "predicted_force_next_max_N": prediction[k]})
    csv_new(OUT / "development_candidate_predictions.csv", prediction_rows)
    aggregate = [{"model": name, **aggregate_candidate(outputs[name])} for name in MODELS]
    csv_new(OUT / "development_candidate_summary.csv", aggregate)
    boundary = [{"model": name, "reference": key, "reference_N": level,
                 **classification(outputs[name]["truth"], outputs[name]["prediction"], level)}
                for name in MODELS for key, level in q.items()]
    csv_new(OUT / "development_boundary_metrics.csv", boundary)
    json_new(OUT / "bootstrap_summary.json", {"method": "1000 fixed-seed state-level paired bootstrap; resample 120 states with all 21 candidates",
                                              "seed": 20261008, "comparison": "M1 minus M0", "metrics": bootstrap(outputs, q["q95"])})
    print("Development-test diagnostics completed; confirmation still untouched", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("validation-freeze", "development-test"), required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Frozen CUDA device unavailable")
    status = json.loads((OUT / "development_collection_status.json").read_text())
    if status["status"] != "COMPLETE_ALL_800_STATES_AND_G2_PARENT_REPRODUCED":
        raise RuntimeError("Development labels incomplete")
    device = torch.device("cuda:0")
    if args.stage == "validation-freeze":
        validation_freeze(device)
    else:
        development_test(device)


if __name__ == "__main__":
    main()
