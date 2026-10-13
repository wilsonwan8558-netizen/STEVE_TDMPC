"""Shared G7 quantile coverage and Q95 decision metrics, without fitted thresholds."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
RL=HERE.parents[1]
for name in ("generic_solver_g6a","generic_solver_g3"):
    sys.path.insert(0,str(RL/"experiments"/name))
from g6_metrics import metrics,availability,contact_emergence
from g3_evaluate import classification,regression
from g7_prepare_train import OUT,SEEDS,TAUS,QuantileNet,sha


def predictor(seed):
    manifest=json.loads((OUT/"model_manifest.json").read_text())
    if manifest["checkpoint_sha256"][str(seed)]!=sha(OUT/f"models/seed_{seed}/model.pt"):
        raise RuntimeError("G7 checkpoint changed")
    if manifest["normalization_sha256"]!=sha(OUT/"models/normalization.json"):
        raise RuntimeError("G7 normalization changed")
    norm=json.loads((OUT/"models/normalization.json").read_text())
    mean=np.asarray(norm["input_mean"],dtype=np.float64)
    scale=np.asarray(norm["input_std"],dtype=np.float64)
    net=QuantileNet(len(mean)).to(torch.device("cuda:0"))
    net.load_state_dict(torch.load(OUT/f"models/seed_{seed}/model.pt",map_location="cuda:0",weights_only=True))
    net.eval()
    return net,mean,scale


def predict(seed,x):
    net,mean,scale=predictor(seed)
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=308 or not np.all(np.isfinite(x)):
        raise RuntimeError("Invalid G7 exact-input matrix")
    with torch.inference_mode():
        output=np.concatenate([net(torch.as_tensor((x[i:i+4096]-mean)/scale,dtype=torch.float32,device="cuda:0")).cpu().numpy()
            for i in range(0,len(x),4096)],axis=0).astype(np.float64)
    if (output.shape!=(len(x),4) or not np.all(np.isfinite(output)) or
            np.any(output[:,0]<0) or np.any(np.diff(output,axis=1)<0)):
        raise RuntimeError("Quantile non-crossing/nonnegative invariant failed")
    return output


def quantile_rows(y,p,seed,group):
    y=np.asarray(y,dtype=np.float64);p=np.asarray(p,dtype=np.float64)
    if p.shape!=(len(y),4):raise RuntimeError("Quantile label alignment failed")
    rows=[]
    for j,tau in enumerate(TAUS):
        error=y-p[:,j]
        loss=np.maximum(tau*error,(tau-1)*error)
        coverage=float(np.mean(y<=p[:,j]))
        rows.append({"group":group,"training_seed":seed,"quantile":f"Q{int(tau*100)}","tau":tau,
            "candidate_count":len(y),"empirical_coverage":coverage,"coverage_error":coverage-tau,
            "mean_pinball_N":float(np.mean(loss)),"mean_predicted_quantile_N":float(np.mean(p[:,j])),
            "minimum_predicted_quantile_N":float(np.min(p[:,j])),"maximum_predicted_quantile_N":float(np.max(p[:,j])),
            "crossing_count":int(np.sum(np.diff(p,axis=1)<0))})
    return rows


def boundary_row(y,p,contact,seed,group,delta,limit,q99):
    y=np.asarray(y,dtype=np.float64);score=np.asarray(p[:,2],dtype=np.float64)
    cutoff=float(limit-delta)
    corrected=score+float(delta)
    decision=metrics(y,corrected,limit,limit)
    rank=classification(y,score,limit)
    avail=availability(corrected,limit)
    emerging=contact_emergence(y,corrected,contact,limit,limit)
    no_contact_miss=(~contact)&(y>limit)&(corrected<=limit)
    emergence_excess=y[no_contact_miss]-limit
    upper=metrics(y,p[:,3],q99,q99)
    return {"group":group,"training_seed":seed,"delta_cal_N":delta,"Q95_decision_cutoff_N":cutoff,
        "Q95_true_high_force_count":decision["true_unsafe_count"],"Q95_false_safe_count":decision["false_safe_count"],
        "Q95_false_safe_rate":decision["false_safe_rate"],"Q95_unsafe_recall":decision["unsafe_recall"],
        "Q95_precision":decision["precision"],"Q95_specificity":decision["specificity"],"Q95_NPV":decision["NPV"],
        "Q95_AUROC":rank["AUROC"],"Q95_AUPRC":rank["AUPRC"],
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
        "no_contact_false_safe_severity_mean_N":float(np.mean(emergence_excess)) if len(emergence_excess) else None,
        "no_contact_false_safe_severity_max_N":float(np.max(emergence_excess)) if len(emergence_excess) else None,
        "Q99_true_extreme_count":upper["true_unsafe_count"],"Q99_extreme_missed_by_Q99_count":upper["false_safe_count"],
        "Q99_extreme_recall_at_G2_Q99_reference":upper["unsafe_recall"],
        "Q99_AUROC_for_extreme":classification(y,p[:,3],q99)["AUROC"],
        "Q99_AUPRC_for_extreme":classification(y,p[:,3],q99)["AUPRC"],
        "quantile_crossing_count":int(np.sum(np.diff(p,axis=1)<0))}
