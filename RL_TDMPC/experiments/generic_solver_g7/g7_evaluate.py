"""Untouched G7/G6B paired confirmation metrics at both frozen decision rules."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
RL=HERE.parents[1]
sys.path.insert(0,str(RL/"experiments/generic_solver_g6a"))
sys.path.insert(0,str(RL/"experiments/generic_solver_g3"))
from g6_metrics import metrics,availability,contact_emergence
from g3_evaluate import classification
from g7_prepare_train import OUT,SEEDS,sha,new_json
from g7_calibrate import load,csv_new
from g7_metrics import quantile_rows,boundary_row
from g7_collect import verify_g7
from g7_freeze import verify


def summary(rows,field,larger_bad=True):
    values=np.asarray([float(r[field]) for r in rows],dtype=np.float64)
    return {"by_seed":{str(r["training_seed"]):float(r[field]) for r in rows},
        "mean":float(np.mean(values)),"sample_sd":float(np.std(values,ddof=1)),
        "worst":float(np.max(values) if larger_bad else np.min(values))}


def baseline_row(y,p,contact,seed,limit,q99,cutoff):
    decision=metrics(y,p,cutoff,limit)
    avail=availability(p,cutoff)
    emerging=contact_emergence(y,p,contact,cutoff,limit)
    misses=(~contact)&(y>limit)&(p<=cutoff)
    severity=y[misses]-limit
    return {"training_seed":seed,"method":"G6B_frozen_point_plus_global_margin","F_lim_N":limit,
        "point_cutoff_N":cutoff,"Q95_true_high_force_count":decision["true_unsafe_count"],
        "Q95_false_safe_count":decision["false_safe_count"],"Q95_false_safe_rate":decision["false_safe_rate"],
        "Q95_unsafe_recall":decision["unsafe_recall"],"Q95_precision":decision["precision"],
        "Q95_specificity":decision["specificity"],"Q95_NPV":decision["NPV"],
        "Q95_AUROC":classification(y,p,limit)["AUROC"],"Q95_AUPRC":classification(y,p,limit)["AUPRC"],
        "Q95_false_safe_severity_mean_N":decision["false_safe_severity"]["mean_N"],
        "Q95_false_safe_severity_median_N":decision["false_safe_severity"]["median_N"],
        "Q95_false_safe_severity_q90_N":decision["false_safe_severity"]["q90_N"],
        "Q95_false_safe_severity_max_N":decision["false_safe_severity"]["max_N"],
        "candidate_acceptance_rate":decision["acceptance_rate"],
        "fraction_states_with_accepted_action":avail["fraction_states_with_at_least_one_accepted"],
        "fraction_states_zero_accepted":avail["fraction_states_with_zero_accepted"],
        "zero_accepted_states":avail["states_with_zero_accepted"],
        "median_accepted_actions_per_state":avail["median_accepted_actions_per_state"],
        "no_contact_to_Q95_high_force_count":emerging["no_contact_to_Q95_high_force_count"],
        "no_contact_false_safe_count":emerging["false_safe_count"],
        "no_contact_false_safe_rate":emerging["false_safe_rate"],
        "no_contact_false_safe_severity_mean_N":float(np.mean(severity)) if len(severity) else None,
        "no_contact_false_safe_severity_max_N":float(np.max(severity)) if len(severity) else None,
        "Q99_true_extreme_count":int(np.sum(y>q99))}


def main():
    verify();q=verify_g7()
    selected=json.loads((OUT/"selected_residual_calibration.json").read_text())
    delta=float(selected["delta_cal_N"])
    g6b_selected=json.loads((RL/"logs/generic_solver_phase_g6b_full_contact_predictor/selected_cutoff.json").read_text())
    cutoff=float(g6b_selected["c_pred_N"])
    audit=json.loads((OUT/"confirmation_scoring_audit.json").read_text())
    if (audit["status"]!="PASS_THREE_G7_REPLICAS_SAME_LABELS" or not audit["G6B_same_rows_scored"] or
            audit["candidate_results_sha256"]!=sha(OUT/"confirmation_candidate_results.csv")):
        raise RuntimeError("G7/G6B paired confirmation scoring incomplete")
    rows,y,contact,p=load("confirmation")
    qrows=[];g7rows=[];rawrows=[];g6brows=[]
    for seed in SEEDS:
        qrows.extend(quantile_rows(y,p[seed],seed,"confirmation"))
        g7rows.append(boundary_row(y,p[seed],contact,seed,"confirmation_frozen",delta,q["q95"],q["q99"]))
        rawrows.append(boundary_row(y,p[seed],contact,seed,"confirmation_raw",0.,q["q95"],q["q99"]))
        baseline=np.asarray([float(r[f"G6B_point_seed_{seed}_N"]) for r in rows],dtype=np.float64)
        g6brows.append(baseline_row(y,baseline,contact,seed,q["q95"],q["q99"],cutoff))
    csv_new(OUT/"confirmation_quantile_metrics.csv",qrows)
    csv_new(OUT/"g7_confirmation_metrics.csv",g7rows)
    csv_new(OUT/"g7_confirmation_raw_metrics.csv",rawrows)
    csv_new(OUT/"g6b_same_data_baseline_metrics.csv",g6brows)
    fields=("Q95_false_safe_rate","candidate_acceptance_rate","zero_accepted_states","no_contact_false_safe_rate")
    comparison={"status":"UNTOUCHED_PAIRED_G7_VS_FROZEN_G6B_CONFIRMATION",
        "same_candidate_rows":True,"candidate_count":len(y),"state_count":120,
        "Q95_true_high_force_count":int(np.sum(y>q["q95"])),
        "no_contact_to_Q95_high_force_count":g7rows[0]["no_contact_to_Q95_high_force_count"],
        "F_lim_N":q["q95"],"G6B_point_cutoff_N":cutoff,"G6B_global_margin_N":g6b_selected["Delta_N"],
        "G7_delta_cal_N":delta,"G7_delta_smaller_than_G6B_margin":delta<g6b_selected["Delta_N"],
        "by_seed":{str(seed):{"G7":g7rows[i],"G6B":g6brows[i],
            "G7_minus_G6B_Q95_false_safe_rate":g7rows[i]["Q95_false_safe_rate"]-g6brows[i]["Q95_false_safe_rate"],
            "G7_minus_G6B_candidate_acceptance_rate":g7rows[i]["candidate_acceptance_rate"]-g6brows[i]["candidate_acceptance_rate"],
            "G7_minus_G6B_zero_action_states":g7rows[i]["zero_accepted_states"]-g6brows[i]["zero_accepted_states"]}
            for i,seed in enumerate(SEEDS)},
        "G7_robustness":{field:summary(g7rows,field,larger_bad=field!="candidate_acceptance_rate") for field in fields},
        "G6B_robustness":{field:summary(g6brows,field,larger_bad=field!="candidate_acceptance_rate") for field in fields},
        "quantile_coverage_file":"confirmation_quantile_metrics.csv",
        "G7_calibration_used_confirmation":False,"G6B_retrained_or_recalibrated":False}
    new_json(OUT/"g7_vs_g6b_comparison.json",comparison)
    print("G7 versus G6B same-data confirmation",[(seed,g7rows[i]["Q95_false_safe_rate"],g7rows[i]["candidate_acceptance_rate"],g7rows[i]["zero_accepted_states"],
        g6brows[i]["Q95_false_safe_rate"],g6brows[i]["candidate_acceptance_rate"],g6brows[i]["zero_accepted_states"]) for i,seed in enumerate(SEEDS)],flush=True)


if __name__=="__main__":main()
