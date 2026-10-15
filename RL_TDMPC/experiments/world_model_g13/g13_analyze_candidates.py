"""Analyze paired G13/G12 predictions against exact same-parent SOFA forks."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
OUT = RL / "logs/world_model_force_aware_planning_g13"
F_LIM = 3.685331040756903


def pearson(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0 else None


def spearman(x, y):
    value = spearmanr(x, y).statistic
    return float(value) if np.isfinite(value) else None


def pairwise(truth, predicted, actions):
    i, j = np.triu_indices(len(truth), k=1)
    valid = ((np.max(np.abs(actions[i] - actions[j]), axis=1) > 1e-8)
             & (np.abs(truth[i] - truth[j]) > 1e-8))
    i, j = i[valid], j[valid]
    true_difference = truth[i] - truth[j]
    prediction_difference = predicted[i] - predicted[j]
    high_low = (truth[i] > F_LIM) != (truth[j] > F_LIM)
    predicted_tie = np.abs(prediction_difference) <= 1e-8
    correct_direction = (~predicted_tie) & (true_difference * prediction_difference > 0)
    comparable = len(i)
    correct = int(correct_direction.sum())
    ties = int(predicted_tie.sum())
    high_low_comparable = int(high_low.sum())
    high_low_correct = int(np.sum(high_low & correct_direction))
    high_low_ties = int(np.sum(high_low & predicted_tie))
    return {
        "comparable_pairs": comparable, "correct_pairs": correct,
        "predicted_tie_pairs": ties,
        "accuracy_with_half_credit_for_prediction_tie":
            (correct + 0.5 * ties) / comparable if comparable else None,
        "high_vs_low_comparable_pairs": high_low_comparable,
        "high_vs_low_correct_pairs": high_low_correct,
        "high_vs_low_predicted_tie_pairs": high_low_ties,
        "high_vs_low_accuracy_with_half_credit_for_prediction_tie":
            (high_low_correct + 0.5 * high_low_ties) / high_low_comparable
            if high_low_comparable else None,
    }


def classification(truth, prediction):
    high = truth > F_LIM
    safe = ~high
    false_safe = (prediction <= F_LIM) & high
    false_warning = (prediction > F_LIM) & safe
    return {
        "count": len(truth), "true_above_limit_count": int(high.sum()),
        "true_safe_count": int(safe.sum()),
        "MAE_N": float(np.mean(np.abs(prediction - truth))),
        "RMSE_N": float(np.sqrt(np.mean((prediction - truth)**2))),
        "Pearson": pearson(truth, prediction),
        "Spearman": spearman(truth, prediction),
        "false_safe_count": int(false_safe.sum()),
        "false_safe_rate_given_true_high": float(np.mean(prediction[high] <= F_LIM)) if high.any() else None,
        "above_limit_recall": float(np.mean(prediction[high] > F_LIM)) if high.any() else None,
        "false_warning_count": int(false_warning.sum()),
        "false_warning_rate_given_true_safe": float(np.mean(prediction[safe] > F_LIM)) if safe.any() else None,
    }


def main() -> None:
    source = OUT / "candidate_force_discrimination.csv"
    audit = json.loads((OUT / "candidate_fork_collection_audit.json").read_text())
    if audit["status"] != "PASS_NORMAL_G13_MPPI_AND_EXACT_SAME_PARENT_SOFA_FORKS":
        raise RuntimeError("Candidate collection audit did not pass")
    rows = list(csv.DictReader(source.open(newline="", encoding="utf-8")))
    grouped: dict[tuple[int, int], list[dict]] = {}
    for row in rows:
        grouped.setdefault((int(row["seed"]), int(row["decision_index"])), []).append(row)
    if len(grouped) != audit["states"] or len(rows) != audit["candidate_forks"]:
        raise RuntimeError("Candidate fork rows incomplete")
    if any(len(group) != 513 or group[0]["candidate_label"] != "baseline_weighted_mean"
           for group in grouped.values()):
        raise RuntimeError("Each parent requires one baseline and all 512 normal MPPI candidates")
    state_rows = []
    high_rows = []
    totals = {arm: {"correct_pairs": 0, "comparable_pairs": 0, "predicted_tie_pairs": 0,
                    "high_vs_low_correct_pairs": 0, "high_vs_low_comparable_pairs": 0,
                    "high_vs_low_predicted_tie_pairs": 0}
              for arm in ("G12", "G13")}
    all_truth = []
    all_pred = {"G12": [], "G13": []}
    per_state_spearman = {"G12": [], "G13": []}
    for (seed, decision), group in sorted(grouped.items()):
        truth = np.asarray([float(row["true_force_N"]) for row in group])
        actions = np.asarray([[float(row["action_translation"]), float(row["action_rotation"])]
                              for row in group])
        pred = {"G12": np.asarray([float(row["G12_predicted_force_N"]) for row in group]),
                "G13": np.asarray([float(row["G13_predicted_force_N"]) for row in group])}
        all_truth.extend(truth.tolist())
        for arm in pred:
            all_pred[arm].extend(pred[arm].tolist())
        has_high = bool(np.any(truth > F_LIM))
        has_safe = bool(np.any(truth <= F_LIM))
        baseline_high = bool(truth[0] > F_LIM)
        candidate_safe = bool(np.any(truth[1:] <= F_LIM))
        state = {"seed": seed, "decision_index": decision,
                 "true_min_force_N": float(np.min(truth)),
                 "true_max_force_N": float(np.max(truth)),
                 "baseline_true_force_N": float(truth[0]),
                 "true_above_limit_candidate_count": int(np.sum(truth > F_LIM)),
                 "mixed_high_and_safe_candidates": has_high and has_safe,
                 "baseline_above_limit_with_safe_sampled_alternative": baseline_high and candidate_safe}
        for arm in ("G12", "G13"):
            pair = pairwise(truth, pred[arm], actions)
            for key in totals[arm]:
                totals[arm][key] += pair[key]
            rho = spearman(truth, pred[arm])
            if rho is not None:
                per_state_spearman[arm].append(rho)
            state[f"{arm}_within_state_Spearman"] = rho if rho is not None else ""
            state[f"{arm}_pairwise_accuracy"] = pair["accuracy_with_half_credit_for_prediction_tie"]
            high_low_accuracy = pair["high_vs_low_accuracy_with_half_credit_for_prediction_tie"]
            state[f"{arm}_high_vs_low_pairwise_accuracy"] = high_low_accuracy if high_low_accuracy is not None else ""
            selected_index = int(np.argmin(pred[arm][1:])) + 1
            state[f"{arm}_minimum_predicted_candidate_true_force_N"] = float(truth[selected_index])
            state[f"{arm}_minimum_predicted_candidate_is_safe"] = bool(truth[selected_index] <= F_LIM)
            state[f"{arm}_minimum_predicted_candidate_reduces_baseline_force"] = bool(truth[selected_index] < truth[0] - 1e-8)
            state[f"{arm}_selected_candidate_force_minus_baseline_N"] = float(truth[selected_index] - truth[0])
        state_rows.append(state)
        if has_high:
            high_rows.append(state)
    truth_all = np.asarray(all_truth)
    analysis = {
        "status": "PAIRED_G12_G13_NORMAL_MPPI_CANDIDATE_FORCE_DISCRIMINATION",
        "F_lim_N": F_LIM, "H_force": 1,
        "states": len(grouped), "true_SOFA_candidate_forks": len(rows),
        "true_low_count_F_le_0p5F_lim": int(np.sum(truth_all <= 0.5 * F_LIM)),
        "true_near_safe_count_0p5F_lim_lt_F_le_F_lim": int(np.sum((truth_all > 0.5 * F_LIM) & (truth_all <= F_LIM))),
        "true_above_limit_count": int(np.sum(truth_all > F_LIM)),
        "states_with_any_true_above_limit_candidate": sum(row["true_above_limit_candidate_count"] > 0
                                                         for row in state_rows),
        "baseline_true_above_limit_states": sum(row["baseline_true_force_N"] > F_LIM
                                                for row in state_rows),
        "mixed_true_high_and_safe_states": sum(row["mixed_high_and_safe_candidates"] for row in state_rows),
        "baseline_above_limit_with_safe_sampled_alternative_states":
            sum(row["baseline_above_limit_with_safe_sampled_alternative"] for row in state_rows),
        "arms": {},
        "comparison_caveat": "Same G13 on-policy development parents and same first-action candidate set for both heads. G12 head is evaluated out of its original controller's on-policy distribution. The cohort is a fixed-decision-index diagnostic, not population prevalence.",
    }
    for arm in ("G12", "G13"):
        pred_all = np.asarray(all_pred[arm])
        arm_summary = classification(truth_all, pred_all)
        total = totals[arm]
        arm_summary.update({
            "within_state_Spearman_median": float(np.median(per_state_spearman[arm])) if per_state_spearman[arm] else None,
            "within_state_Spearman_mean": float(np.mean(per_state_spearman[arm])) if per_state_spearman[arm] else None,
            "within_state_Spearman_defined_states": len(per_state_spearman[arm]),
            "pairwise_comparable_count": total["comparable_pairs"],
            "pairwise_order_accuracy":
                (total["correct_pairs"] + 0.5 * total["predicted_tie_pairs"]) / total["comparable_pairs"]
                if total["comparable_pairs"] else None,
            "high_vs_safe_pairwise_comparable_count": total["high_vs_low_comparable_pairs"],
            "high_vs_safe_pairwise_order_accuracy":
                (total["high_vs_low_correct_pairs"] + 0.5 * total["high_vs_low_predicted_tie_pairs"]) /
                total["high_vs_low_comparable_pairs"] if total["high_vs_low_comparable_pairs"] else None,
            "mixed_high_safe_states_where_lowest_predicted_candidate_is_truly_safe":
                sum(row["mixed_high_and_safe_candidates"] and row[f"{arm}_minimum_predicted_candidate_is_safe"]
                    for row in state_rows),
            "baseline_high_with_safe_alternative_states_where_selected_candidate_is_truly_safe":
                sum(row["baseline_above_limit_with_safe_sampled_alternative"] and
                    row[f"{arm}_minimum_predicted_candidate_is_safe"] for row in state_rows),
            "baseline_high_with_safe_alternative_states_where_selected_candidate_reduces_true_force":
                sum(row["baseline_above_limit_with_safe_sampled_alternative"] and
                    row[f"{arm}_minimum_predicted_candidate_reduces_baseline_force"] for row in state_rows),
            "mean_selected_candidate_minus_baseline_true_force_N_on_mixed_states":
                float(np.mean([row[f"{arm}_selected_candidate_force_minus_baseline_N"]
                               for row in state_rows if row["mixed_high_and_safe_candidates"]]))
                if any(row["mixed_high_and_safe_candidates"] for row in state_rows) else None,
            "mean_selected_candidate_minus_baseline_true_force_N_when_baseline_high_with_safe_alternative":
                float(np.mean([row[f"{arm}_selected_candidate_force_minus_baseline_N"]
                               for row in state_rows if row["baseline_above_limit_with_safe_sampled_alternative"]]))
                if any(row["baseline_above_limit_with_safe_sampled_alternative"] for row in state_rows) else None,
        })
        analysis["arms"][arm] = arm_summary
    comparable_rho = [row for row in state_rows
                      if row["G12_within_state_Spearman"] != ""
                      and row["G13_within_state_Spearman"] != ""]
    comparable_pair = [row for row in state_rows
                       if row["G12_pairwise_accuracy"] is not None
                       and row["G13_pairwise_accuracy"] is not None]
    analysis["paired_state_improvement"] = {
        "within_state_Spearman_comparable_states": len(comparable_rho),
        "G13_higher_Spearman_states": sum(row["G13_within_state_Spearman"] > row["G12_within_state_Spearman"]
                                          for row in comparable_rho),
        "G13_lower_Spearman_states": sum(row["G13_within_state_Spearman"] < row["G12_within_state_Spearman"]
                                         for row in comparable_rho),
        "pairwise_accuracy_comparable_states": len(comparable_pair),
        "G13_higher_pairwise_accuracy_states": sum(row["G13_pairwise_accuracy"] > row["G12_pairwise_accuracy"]
                                                   for row in comparable_pair),
        "G13_lower_pairwise_accuracy_states": sum(row["G13_pairwise_accuracy"] < row["G12_pairwise_accuracy"]
                                                  for row in comparable_pair),
    }
    state_path = OUT / "candidate_state_metrics.csv"
    with state_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(state_rows[0]))
        writer.writeheader()
        writer.writerows(state_rows)
    high_path = OUT / "high_force_candidate_results.csv"
    with high_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(state_rows[0]))
        writer.writeheader()
        writer.writerows(high_rows)
    (OUT / "candidate_discrimination_summary.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print("G13_CANDIDATE_ANALYSIS", json.dumps(analysis, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
