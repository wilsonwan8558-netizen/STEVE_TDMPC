"""Analyze the frozen high-force G2-parent G13/G12 SOFA fork diagnostic."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from g13_analyze_candidates import F_LIM, pairwise, pearson, spearman

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
OUT = RL / "logs/world_model_force_aware_planning_g13"


def main() -> None:
    audit = json.loads((OUT / "archived_high_fork_collection_audit.json").read_text())
    if audit["status"] != "PASS_EXACT_G2_REUSED_SCENE_PARENT_AND_SAME_PARENT_FORKS":
        raise RuntimeError("Archived high-force fork audit did not pass")
    rows = list(csv.DictReader(
        (OUT / "archived_high_force_candidate_results.csv").open(newline="", encoding="utf-8")))
    groups = {}
    for row in rows:
        groups.setdefault((int(row["seed"]), int(row["decision_index"])), []).append(row)
    if len(groups) != 40 or any(len(group) != 34 for group in groups.values()):
        raise RuntimeError("Archived high-force candidate rows incomplete")
    state_rows = []
    for (seed, decision), group in sorted(groups.items()):
        if (group[0]["candidate_label"] != "archived_G2_behavior_anchor"
                or group[1]["candidate_label"] != "G13_weighted_mean"):
            raise RuntimeError("Archived anchor order changed")
        truth = np.asarray([float(row["true_force_N"]) for row in group])
        actions = np.asarray([[float(row["action_translation"]), float(row["action_rotation"])]
                              for row in group])
        stratum = group[0]["stratum"]
        if (stratum == "archived_true_high_anchor") != (truth[0] > F_LIM):
            raise RuntimeError(f"Frozen high/control parent force stratum failed {seed}/{decision}")
        state = {
            "seed": seed, "decision_index": decision, "stratum": stratum,
            "archived_anchor_true_force_N": float(truth[0]),
            "G13_weighted_mean_true_force_N": float(truth[1]),
            "sampled_true_min_force_N": float(np.min(truth[2:])),
            "sampled_true_max_force_N": float(np.max(truth[2:])),
            "sampled_true_safe_candidate_count": int(np.sum(truth[2:] <= F_LIM)),
            "sampled_true_above_candidate_count": int(np.sum(truth[2:] > F_LIM)),
            "has_sampled_true_safe_alternative": bool(np.any(truth[2:] <= F_LIM)),
            "has_sampled_true_lower_force_alternative": bool(np.any(truth[2:] < truth[0] - 1e-8)),
        }
        for arm in ("G12", "G13"):
            pred = np.asarray([float(row[f"{arm}_predicted_force_N"]) for row in group])
            pair = pairwise(truth, pred, actions)
            chosen = 2 + int(np.argmin(pred[2:]))
            state[f"{arm}_Pearson"] = pearson(truth, pred)
            state[f"{arm}_within_state_Spearman"] = spearman(truth, pred)
            state[f"{arm}_pairwise_accuracy"] = pair["accuracy_with_half_credit_for_prediction_tie"]
            state[f"{arm}_high_vs_safe_pairwise_accuracy"] = pair["high_vs_low_accuracy_with_half_credit_for_prediction_tie"]
            state[f"{arm}_chosen_sample_true_force_N"] = float(truth[chosen])
            state[f"{arm}_chosen_sample_truly_safe"] = bool(truth[chosen] <= F_LIM)
            state[f"{arm}_chosen_sample_reduces_archived_anchor_force"] = bool(truth[chosen] < truth[0] - 1e-8)
            state[f"{arm}_chosen_minus_archived_anchor_true_force_N"] = float(truth[chosen] - truth[0])
            state[f"{arm}_false_safe_count"] = int(np.sum((pred <= F_LIM) & (truth > F_LIM)))
        state_rows.append(state)
    output = OUT / "archived_high_state_metrics.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(state_rows[0]))
        writer.writeheader()
        writer.writerows(state_rows)
    high = [row for row in state_rows if row["stratum"] == "archived_true_high_anchor"]
    control = [row for row in state_rows if row["stratum"] == "matched_low_force_control"]
    if len(high) != 20 or len(control) != 20:
        raise RuntimeError("Archived high/control state count changed")
    eligible = [row for row in high if row["has_sampled_true_safe_alternative"]]
    lower = [row for row in high if row["has_sampled_true_lower_force_alternative"]]
    summary = {
        "status": "PAIRED_G12_G13_ARCHIVED_TRUE_HIGH_PARENT_DISCRIMINATION",
        "F_lim_N": F_LIM, "H_force": 1,
        "true_high_parent_states": len(high), "matched_low_force_control_states": len(control),
        "true_high_parents_with_sampled_safe_alternative": len(eligible),
        "true_high_parents_with_sampled_lower_force_alternative": len(lower),
        "true_high_sampled_candidate_outcomes": sum(row["sampled_true_above_candidate_count"] for row in high),
        "true_safe_sampled_candidate_outcomes_at_high_parents": sum(row["sampled_true_safe_candidate_count"] for row in high),
        "arms": {},
        "interpretation_caveat": "True-force-enriched archived G2 parents and G13 cold-start normal MPPI samples; this is an off-policy diagnostic, not on-policy prevalence or a closed-loop planner result.",
        "paired_lower_alternative_states": {
            "G13_chosen_sample_lower_true_force_than_G12":
                sum(row["G13_chosen_sample_true_force_N"] < row["G12_chosen_sample_true_force_N"] - 1e-8
                    for row in lower),
            "G12_chosen_sample_lower_true_force_than_G13":
                sum(row["G12_chosen_sample_true_force_N"] < row["G13_chosen_sample_true_force_N"] - 1e-8
                    for row in lower),
            "equal_chosen_sample_true_force":
                sum(abs(row["G13_chosen_sample_true_force_N"] - row["G12_chosen_sample_true_force_N"]) <= 1e-8
                    for row in lower),
            "median_G13_minus_G12_chosen_true_force_N":
                float(np.median([row["G13_chosen_sample_true_force_N"] - row["G12_chosen_sample_true_force_N"]
                                 for row in lower])) if lower else None,
            "mean_G13_minus_G12_chosen_true_force_N":
                float(np.mean([row["G13_chosen_sample_true_force_N"] - row["G12_chosen_sample_true_force_N"]
                               for row in lower])) if lower else None,
        },
    }
    for arm in ("G12", "G13"):
        summary["arms"][arm] = {
            "high_parent_mean_within_state_Spearman":
                float(np.mean([row[f"{arm}_within_state_Spearman"] for row in high
                               if row[f"{arm}_within_state_Spearman"] is not None]))
                if any(row[f"{arm}_within_state_Spearman"] is not None for row in high) else None,
            "high_parent_median_pairwise_accuracy":
                float(np.median([row[f"{arm}_pairwise_accuracy"] for row in high
                                 if row[f"{arm}_pairwise_accuracy"] is not None]))
                if any(row[f"{arm}_pairwise_accuracy"] is not None for row in high) else None,
            "high_parents_with_safe_alternative_where_min_predicted_sample_is_truly_safe":
                sum(row[f"{arm}_chosen_sample_truly_safe"] for row in eligible),
            "high_parents_with_lower_force_alternative_where_min_predicted_sample_reduces_anchor_force":
                sum(row[f"{arm}_chosen_sample_reduces_archived_anchor_force"] for row in lower),
            "mean_chosen_minus_archived_anchor_true_force_N_when_lower_alternative_exists":
                float(np.mean([row[f"{arm}_chosen_minus_archived_anchor_true_force_N"] for row in lower]))
                if lower else None,
            "median_chosen_minus_archived_anchor_true_force_N_when_lower_alternative_exists":
                float(np.median([row[f"{arm}_chosen_minus_archived_anchor_true_force_N"] for row in lower]))
                if lower else None,
            "mean_chosen_minus_archived_anchor_true_force_N_when_safe_alternative_exists":
                float(np.mean([row[f"{arm}_chosen_minus_archived_anchor_true_force_N"] for row in eligible]))
                if eligible else None,
            "false_safe_count_across_high_parent_forks_including_anchors":
                sum(row[f"{arm}_false_safe_count"] for row in high),
        }
    (OUT / "archived_high_candidate_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print("G13_ARCHIVED_HIGH_ANALYSIS", json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
