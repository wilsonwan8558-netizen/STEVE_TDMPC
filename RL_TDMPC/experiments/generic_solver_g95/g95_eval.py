"""Prospective planner-plan and per-state metrics, shared by development/confirmation."""

from __future__ import annotations

import argparse
import csv
import json

import numpy as np
import torch
from scipy.stats import rankdata

from g95_common import OUT, G9, SEEDS, F_LIM, TAUS, cuda_gate, sha, json_new
from g9_risk import QuantileNet, predict, load_model as load_g9


def csv_new(path, rows):
    if not rows:
        raise RuntimeError(f"No rows to write {path}")
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_replica(kind, seed):
    manifest = json.loads((OUT / "model_training_manifest.json").read_text())
    key = manifest["models"][kind][str(seed)]
    path = OUT / "models" / kind / f"seed_{seed}.pt"
    selection = OUT / "models" / kind / f"seed_{seed}_selection.json"
    if sha(path) != key["checkpoint_sha256"] or sha(selection) != key["selection_sha256"]:
        raise RuntimeError(f"G9.5 frozen replica changed {kind}/{seed}")
    norm = json.loads((G9 / "models/risk/E1/H2/normalization.json").read_text())
    mean, scale = np.asarray(norm["mean"]), np.asarray(norm["std"])
    net = QuantileNet(298).to("cuda:0")
    net.load_state_dict(torch.load(path, map_location="cuda:0", weights_only=True))
    net.eval()
    return net, mean, scale, float(key["validation_Q95_delta_N"])


def auc_ap(y_binary, score):
    y = np.asarray(y_binary, dtype=bool)
    s = np.asarray(score, dtype=np.float64)
    positives, negatives = int(np.sum(y)), int(np.sum(~y))
    if not positives or not negatives:
        return None, None
    ranks = rankdata(s, method="average")
    auc = (float(np.sum(ranks[y]))-positives*(positives+1)/2)/(positives*negatives)
    order = np.argsort(-s, kind="stable")
    tp = fp = 0
    ap = 0.0
    k = 0
    while k < len(order):
        end = k+1
        while end < len(order) and s[order[end]] == s[order[k]]:
            end += 1
        block_positive = int(np.sum(y[order[k:end]]))
        tp += block_positive
        fp += end-k-block_positive
        ap += block_positive/positives * tp/(tp+fp)
        k = end
    return float(auc), float(ap)


def per_state_metrics(state_id, seed, decision, candidate_id, y, risk, accepted, model, replica):
    n = len(y)
    safe = y <= F_LIM
    usable = safe & accepted
    pairs = correct = tied = 0
    for i in range(n):
        for j in range(i+1, n):
            if abs(float(y[i]-y[j])) <= 1e-8:
                continue
            pairs += 1
            relation = float((risk[i]-risk[j])*(y[i]-y[j]))
            if relation > 0:
                correct += 1
            elif relation == 0:
                tied += 1
    pairwise = (correct+0.5*tied)/pairs if pairs else None
    if n >= 3 and np.ptp(y) > 1e-8 and np.ptp(risk) > 1e-8:
        ry, rp = rankdata(y), rankdata(risk)
        spearman = float(np.corrcoef(ry, rp)[0, 1])
    else:
        spearman = None
    if n >= 3 and np.ptp(y) > 1e-8:
        k = min(3, n)
        true_high = set(np.lexsort((candidate_id, -y))[:k])
        pred_high = set(np.lexsort((candidate_id, -risk))[:k])
        true_low = set(np.lexsort((candidate_id, y))[:k])
        pred_low = set(np.lexsort((candidate_id, risk))[:k])
        high_containment = len(true_high & pred_high)/k
        low_containment = len(true_low & pred_low)/k
    else:
        high_containment = low_containment = None
    return {"model": model, "training_seed": replica,
        "state_global_id": int(state_id), "episode_seed": int(seed),
        "decision_index": int(decision), "complete_candidate_plans": n,
        "true_safe_plans": int(np.sum(safe)),
        "true_high_risk_plans": int(np.sum(~safe)),
        "predicted_safe_plans": int(np.sum(accepted)),
        "true_safe_predicted_safe_overlap": int(np.sum(usable)),
        "has_true_safe_plan": int(np.any(safe)),
        "leaves_at_least_one_usable_safe_plan": int(np.any(usable)),
        "false_predicted_infeasible": int(np.any(safe) and not np.any(accepted)),
        "within_state_spearman_Q95_vs_R2": spearman,
        "pairwise_risk_ranking_accuracy": pairwise,
        "top3_high_risk_containment": high_containment,
        "top3_low_risk_containment": low_containment,
        "ranking_evaluable_pairs": pairs}


def calculate(model, replica, arrays, pred, delta):
    y = arrays["R2_true_N"].astype(np.float64)
    p = np.asarray(pred, dtype=np.float64)
    if (p.shape != (len(y), 4) or np.any(~np.isfinite(p)) or np.any(np.diff(p, axis=1) < 0) or
            np.any(~np.isfinite(y)) or np.any(y < 0)):
        raise RuntimeError("G9.5 quantile prediction/true force invalid")
    risk = p[:, 2]+delta
    unsafe = y > F_LIM
    accepted = risk <= F_LIM
    safe_count, unsafe_count = int(np.sum(~unsafe)), int(np.sum(unsafe))
    if not safe_count or not unsafe_count:
        raise RuntimeError("G9.5 planner evaluation lacks safe or unsafe class")
    state_rows = []
    for state_id in np.unique(arrays["state_global_id"]):
        indices = np.flatnonzero(arrays["state_global_id"] == state_id)
        if len(set(arrays["seed"][indices])) != 1 or len(set(arrays["decision_index"][indices])) != 1:
            raise RuntimeError("G9.5 candidate rows crossed state boundary")
        state_rows.append(per_state_metrics(state_id, arrays["seed"][indices[0]],
            arrays["decision_index"][indices[0]], arrays["candidate_id"][indices],
            y[indices], risk[indices], accepted[indices], model, replica))
    feasible = [r for r in state_rows if r["has_true_safe_plan"]]
    finite = lambda key: [float(r[key]) for r in state_rows if r[key] is not None]
    auc, ap = auc_ap(unsafe, risk)
    row = {"model": model, "training_seed": replica,
        "candidate_rows": len(y), "planner_states": len(state_rows),
        "true_safe_plans": safe_count, "true_high_risk_plans": unsafe_count,
        "predicted_safe_plans": int(np.sum(accepted)),
        "safe_plan_accepted": int(np.sum((~unsafe) & accepted)),
        "safe_plan_rejected": int(np.sum((~unsafe) & (~accepted))),
        "safe_plan_acceptance_rate": float(np.mean(accepted[~unsafe])),
        "safe_plan_rejection_rate": float(np.mean((~accepted)[~unsafe])),
        "unsafe_plan_accepted": int(np.sum(unsafe & accepted)),
        "unsafe_plan_rejected": int(np.sum(unsafe & (~accepted))),
        "unsafe_plan_acceptance_rate": float(np.mean(accepted[unsafe])),
        "unsafe_plan_rejection_rate": float(np.mean((~accepted)[unsafe])),
        "Q95_false_safe_count": int(np.sum(unsafe & accepted)),
        "Q95_false_safe_rate": float(np.mean(accepted[unsafe])),
        "false_warning_count": int(np.sum((~unsafe) & (~accepted))),
        "false_warning_rate": float(np.mean((~accepted)[~unsafe])),
        "Q95_raw_coverage": float(np.mean(y <= p[:, 2])),
        "Q95_corrected_coverage": float(np.mean(y <= risk)),
        "Q95_validation_delta_N": float(delta),
        "four_quantile_pinball_N": float(np.mean([np.mean(np.maximum(t*(y-p[:, i]), (t-1)*(y-p[:, i]))) for i,t in enumerate(TAUS)])),
        "Q50_coverage": float(np.mean(y <= p[:, 0])),
        "Q90_coverage": float(np.mean(y <= p[:, 1])),
        "Q95_coverage": float(np.mean(y <= p[:, 2])),
        "Q99_coverage": float(np.mean(y <= p[:, 3])),
        "AUROC_high_risk": auc, "AUPRC_high_risk": ap,
        "states_with_true_safe_plan": len(feasible),
        "fraction_states_with_true_safe_plan": len(feasible)/len(state_rows),
        "states_with_usable_predicted_safe_true_safe_plan": sum(r["leaves_at_least_one_usable_safe_plan"] for r in feasible),
        "usable_state_fraction_among_feasible": sum(r["leaves_at_least_one_usable_safe_plan"] for r in feasible)/len(feasible) if feasible else None,
        "false_predicted_infeasible_states": sum(r["false_predicted_infeasible"] for r in feasible),
        "false_predicted_infeasibility_rate": sum(r["false_predicted_infeasible"] for r in feasible)/len(feasible) if feasible else None,
        "mean_within_state_spearman": float(np.mean(finite("within_state_spearman_Q95_vs_R2"))) if finite("within_state_spearman_Q95_vs_R2") else None,
        "mean_within_state_pairwise_accuracy": float(np.mean(finite("pairwise_risk_ranking_accuracy"))) if finite("pairwise_risk_ranking_accuracy") else None,
        "mean_top3_high_risk_containment": float(np.mean(finite("top3_high_risk_containment"))) if finite("top3_high_risk_containment") else None,
        "mean_top3_low_risk_containment": float(np.mean(finite("top3_low_risk_containment"))) if finite("top3_low_risk_containment") else None}
    return row, state_rows


def robustness(rows):
    result = {}
    keys = ("safe_plan_rejection_rate", "unsafe_plan_acceptance_rate", "Q95_false_safe_rate",
            "mean_within_state_spearman", "mean_within_state_pairwise_accuracy",
            "false_predicted_infeasibility_rate")
    for kind in ("M0_behavior", "M1_planner_conditioned"):
        group = [r for r in rows if r["model"] == kind]
        if len(group) != 3:
            raise RuntimeError(f"G9.5 missing three replicas {kind}")
        result[kind] = {"replicas": {str(r["training_seed"]): r for r in group}, "summary": {}}
        for key in keys:
            values = np.asarray([r[key] for r in group], dtype=np.float64)
            if not np.all(np.isfinite(values)):
                result[kind]["summary"][key] = None
            else:
                higher_bad = key not in ("mean_within_state_spearman", "mean_within_state_pairwise_accuracy")
                result[kind]["summary"][key] = {"mean": float(np.mean(values)),
                    "std_population": float(np.std(values)),
                    "worst_replica": float(np.max(values) if higher_bad else np.min(values))}
    return result


def development():
    cuda_gate()
    manifest = json.loads((OUT / "model_training_manifest.json").read_text())
    if manifest["status"] != "THREE_FIXED_REPLICAS_PER_MODEL_BEFORE_DEVELOPMENT_TEST":
        raise RuntimeError("G9.5 model training incomplete")
    with np.load(OUT / "development_planner_dataset.npz") as z:
        d = {key: z[key].copy() for key in z.files}
    mask = d["group_index"] == 2
    arrays = {key: value[mask] for key, value in d.items()}
    rows, state_rows = [], []
    for kind in ("M0_behavior", "M1_planner_conditioned"):
        for seed in SEEDS:
            net, mean, scale, delta = load_replica(kind, seed)
            p = predict(net, arrays["x"], mean, scale)
            row, by_state = calculate(kind, seed, arrays, p, delta)
            rows.append(row)
            state_rows.extend(by_state)
            print(f"G9.5 development {kind}/{seed} safe rejection={row['safe_plan_rejection_rate']:.3f} unsafe acceptance={row['unsafe_plan_acceptance_rate']:.3f} usable={row['usable_state_fraction_among_feasible']:.3f}", flush=True)
    old, mean, scale, delta = load_g9("E1", 2)
    old_p = predict(old, arrays["x"], mean, scale)
    old_row, old_states = calculate("G9_exact_old_E1_H2", SEEDS[0], arrays, old_p, delta)
    rows.append(old_row)
    state_rows.extend(old_states)
    csv_new(OUT / "development_plan_metrics.csv", rows)
    csv_new(OUT / "development_state_metrics.csv", state_rows)
    json_new(OUT / "training_seed_robustness.json", {"status": "ALL_THREE_REPLICAS_REPORTED_NO_BEST_SEED_SELECTION",
        "development_test_candidate_rows": len(arrays["x"]),
        "development_test_planner_states": len(set(arrays["state_global_id"])),
        "models": robustness(rows),
        "G9_exact_old_reference": old_row})
    print(f"G9.5 development evaluation complete: {len(rows)} model rows, {len(state_rows)} state rows", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("development",), required=True)
    if parser.parse_args().stage == "development":
        development()


if __name__ == "__main__":
    main()
