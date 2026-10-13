"""Evaluate all three frozen full-input replicas on the existing G3 development states."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
RL=HERE.parents[1]
for name in ("generic_solver_g6a","generic_solver_g5","generic_solver_g4","generic_solver_g3"):
    sys.path.insert(0,str(RL/"experiments"/name))
from g6b_prepare_train import OUT,SEEDS,sha,verify_source,matrices,candidate_matrix,cuda_gate,new_json
from g6b_features import FrozenVesselGeometry
from g5_data import source
from g4_data import candidate_states
from g4_train import model
from g4_evaluate import predict as predict_g4
from g3_evaluate import state_metrics, aggregate_candidate, classification, regression
from g6_metrics import metrics,contact_emergence


def predictor(seed):
    manifest=json.loads((OUT/"model_manifest.json").read_text())
    if sha(OUT/f"models/FULL/seed_{seed}/model.pt")!=manifest["checkpoint_sha256"][str(seed)]:
        raise RuntimeError("Frozen G6B model changed")
    norm=json.loads((OUT/"models/FULL/normalization.json").read_text())
    mean=np.asarray(norm["input_mean"]);scale=np.asarray(norm["input_std"])
    net=model(len(mean)).to(torch.device("cuda:0"))
    net.load_state_dict(torch.load(OUT/f"models/FULL/seed_{seed}/model.pt",map_location="cuda:0",weights_only=True))
    net.eval()
    return net,mean,scale


def prediction(seed,x):
    net,mean,scale=predictor(seed)
    return predict_g4((net,mean,scale),x,torch.device("cuda:0"))


def evaluate():
    cuda_gate(); q=verify_source();geometry=FrozenVesselGeometry()
    if (OUT/"development_metrics.csv").exists(): raise FileExistsError("Development metrics already saved")
    arrays,split,index,physical=source()
    states=candidate_states("development_test",split)
    x,y,contact=candidate_matrix(states,arrays,index,physical,geometry)
    if x.shape!=(2520,308): raise RuntimeError("Frozen development candidate count changed")
    rows=[]
    for seed in SEEDS:
        p=prediction(seed,x)
        records=[state_metrics(state,p[i*21:(i+1)*21]) for i,state in enumerate(states)]
        aggregate=aggregate_candidate({"truth":y,"prediction":p,"state_metrics":records,"inference_seconds":0.})
        q95=classification(y,p,q["q95"]);q99=classification(y,p,q["q99"])
        no_contact=contact_emergence(y,p,contact,q["q95"],q["q95"])
        rows.append({"model":"FULL","training_seed":seed,**aggregate,
            "Q95_true_positive_count":q95["true_exceedances"],"Q95_false_safe_count":q95["FN_false_safe"],
            "Q95_false_safe_rate":q95["false_safe_rate_P_pred_safe_given_true_unsafe"],
            "Q95_false_safe_severity_mean_N":q95["false_safe_severity_mean_N"],
            "Q95_false_safe_severity_max_N":q95["false_safe_severity_max_N"],
            "Q95_AUROC":q95["AUROC"],"Q95_AUPRC":q95["AUPRC"],
            "Q99_true_positive_count":q99["true_exceedances"],"Q99_false_safe_count":q99["FN_false_safe"],
            "Q99_false_safe_rate":q99["false_safe_rate_P_pred_safe_given_true_unsafe"],
            "no_contact_to_Q95_high_force_count":no_contact["no_contact_to_Q95_high_force_count"],
            "no_contact_false_safe_count":no_contact["false_safe_count"],
            "no_contact_false_safe_rate":no_contact["false_safe_rate"]})
    with (OUT/"development_metrics.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fields=("Q95_false_safe_rate","candidate_RMSE_N","mean_pairwise_accuracy","no_contact_false_safe_rate")
    robustness={field:{"values_by_seed":{str(row["training_seed"]):row[field] for row in rows},
        "mean":float(np.mean([r[field] for r in rows])),"sample_sd":float(np.std([r[field] for r in rows],ddof=1)),
        "worst_replica":float(max(r[field] for r in rows) if field!="mean_pairwise_accuracy" else min(r[field] for r in rows))} for field in fields}
    historical=RL/"logs/generic_solver_phase_g5_feature_attribution"
    with (historical/"development_candidate_metrics.csv").open(newline="") as stream:
        old=[r for r in csv.DictReader(stream) if r["model"]=="ZP_noC"]
    with (historical/"development_boundary_metrics.csv").open(newline="") as stream:
        old_bound=[r for r in csv.DictReader(stream) if r["model"]=="ZP_noC" and r["reference"]=="q95"]
    with (historical/"no_contact_emergence_metrics.csv").open(newline="") as stream:
        old_contact=[r for r in csv.DictReader(stream) if r["model"]=="ZP_noC"]
    new_json(OUT/"training_seed_robustness.json",{"full":robustness,"historical_Z_plus_K_plus_D_source":"G5 existing development results, no rerun",
        "historical_candidate_metrics":old,"historical_Q95_boundary_metrics":old_bound,"historical_no_contact_emergence":old_contact})
    print("G6B development",[(r["training_seed"],r["candidate_RMSE_N"],r["Q95_false_safe_rate"],r["no_contact_false_safe_rate"]) for r in rows],flush=True)


if __name__=="__main__": evaluate()
