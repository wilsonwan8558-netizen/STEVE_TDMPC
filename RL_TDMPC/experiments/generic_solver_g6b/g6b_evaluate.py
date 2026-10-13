"""Evaluate untouched G6B confirmation at frozen raw/shared cutoffs; no tuning."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
RL=HERE.parents[1]
for name in ("generic_solver_g6a","generic_solver_g5","generic_solver_g3"):
    sys.path.insert(0,str(RL/"experiments"/name))
import g6_metrics
from g3_evaluate import state_metrics,aggregate_candidate,classification
from g6b_prepare_train import OUT,SEEDS,sha,new_json
from g6b_collect import verify_g6b
from g6b_freeze import verify


def csv_new(path,rows):
    if path.exists():raise FileExistsError(path)
    if not rows:raise RuntimeError("Empty output")
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    verify();q=verify_g6b();limit=q["q95"]
    selected=json.loads((OUT/"selected_cutoff.json").read_text());cutoff=float(selected["c_pred_N"])
    if cutoff>limit or selected["F_lim_N"]!=limit:raise RuntimeError("Frozen cutoff changed")
    audit=json.loads((OUT/"confirmation_scoring_audit.json").read_text())
    if audit["candidate_results_sha256"]!=sha(OUT/"confirmation_candidate_results.csv") or audit["candidate_rows"]!=2520:
        raise RuntimeError("Confirmation scoring incomplete")
    g6_metrics.OUT=OUT;g6_metrics.SEEDS=SEEDS
    data=g6_metrics.load_rows("confirmation")
    truth=data["truth"];contact=data["current_contact"]
    states=[json.loads(path.read_text()) for path in sorted((OUT/"confirmation_candidates").glob("state_*.json"))]
    if len(states)!=120:raise RuntimeError("Confirmation states incomplete")
    metrics_rows=[];contact_rows=[]
    for seed in SEEDS:
        prediction=data["prediction"][seed]
        per_state=[state_metrics(state,prediction[i*21:(i+1)*21]) for i,state in enumerate(states)]
        aggregate=aggregate_candidate({"truth":truth,"prediction":prediction,"state_metrics":per_state,"inference_seconds":0.})
        for name,c in (("raw",limit),("frozen_shared",cutoff)):
            qm=g6_metrics.metrics(truth,prediction,c,limit)
            q99=g6_metrics.metrics(truth,prediction,c,q["q99"])
            rank=classification(truth,prediction,limit)
            avail=g6_metrics.availability(prediction,c)
            emerging=g6_metrics.contact_emergence(truth,prediction,contact,c,limit)
            emergence_mask=(~contact)&(truth>limit)&(prediction<=c)
            severity=truth[emergence_mask]-limit
            metrics_rows.append({"training_seed":seed,"cutoff_type":name,"c_pred_N":c,
                "candidate_RMSE_N":aggregate["candidate_RMSE_N"],"candidate_MAE_N":aggregate["candidate_MAE_N"],
                "median_within_state_Spearman":aggregate["median_within_state_Spearman"],
                "mean_pairwise_accuracy":aggregate["mean_pairwise_accuracy"],
                "mean_safe_choice_regret_N":aggregate["mean_safe_choice_regret_N"],
                "Q95_true_positive_count":qm["true_unsafe_count"],"Q95_false_safe_count":qm["false_safe_count"],
                "Q95_false_safe_rate":qm["false_safe_rate"],"Q95_false_safe_severity_mean_N":qm["false_safe_severity"]["mean_N"],
                "Q95_false_safe_severity_max_N":qm["false_safe_severity"]["max_N"],
                "Q95_unsafe_recall":qm["unsafe_recall"],"Q95_AUROC":rank["AUROC"],"Q95_AUPRC":rank["AUPRC"],
                "Q99_true_positive_count":q99["true_unsafe_count"],"Q99_false_safe_count":q99["false_safe_count"],
                "Q99_false_safe_rate":q99["false_safe_rate"],"candidate_acceptance_rate":qm["acceptance_rate"],
                "fraction_states_with_accepted_action":avail["fraction_states_with_at_least_one_accepted"],
                "fraction_states_zero_accepted":avail["fraction_states_with_zero_accepted"],
                "median_accepted_actions_per_state":avail["median_accepted_actions_per_state"]})
            contact_rows.append({"training_seed":seed,"cutoff_type":name,"c_pred_N":c,
                **emerging,"false_safe_severity_mean_N":float(np.mean(severity)) if len(severity) else None,
                "false_safe_severity_max_N":float(np.max(severity)) if len(severity) else None})
    csv_new(OUT/"confirmation_metrics.csv",metrics_rows)
    csv_new(OUT/"confirmation_contact_emergence.csv",contact_rows)
    print("G6B confirmation metrics",[(r["training_seed"],r["Q95_false_safe_rate"],r["candidate_acceptance_rate"])
        for r in metrics_rows if r["cutoff_type"]=="frozen_shared"],flush=True)


if __name__=="__main__":main()
