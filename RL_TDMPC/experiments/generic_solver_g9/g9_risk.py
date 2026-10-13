"""Direct, non-autoregressive G9 noncrossing quantile models for R_H."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
RL=HERE.parents[1]
OUT=RL/"logs/generic_solver_phase_g9_temporal_anticipatory_safety"
HORIZONS=(1,2,3,5)
ESTIMATORS=("E0","E1","E2")
TAUS=(.50,.90,.95,.99)
F_LIM=3.685331040756903
DEVICE=torch.device("cuda:0")


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def json_new(path,value):
    with path.open("x") as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write("\n")


def csv_new(path,rows):
    with path.open("x",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def cuda_gate():
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0" or not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError("G9 quantile models require CUDA_VISIBLE_DEVICES=0; no CPU fallback")


class QuantileNet(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.trunk=nn.Sequential(nn.Linear(dim,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU())
        self.head=nn.Linear(64,4)

    def forward(self,x):
        return torch.cumsum(F.softplus(self.head(self.trunk(x))),dim=-1)


def pinball(pred,target):
    levels=torch.as_tensor(TAUS,dtype=pred.dtype,device=pred.device)
    error=target[:,None]-pred
    return torch.maximum(levels*error,(levels-1)*error).mean()


def data():
    with np.load(OUT/"development_temporal_behavior.npz") as z:
        d={k:z[k].copy() for k in z.files}
    with np.load(OUT/"estimated_current_physical.npz") as z:
        estimates={k:z[k].copy() for k in z.files}
    split=json.loads((OUT/"dataset_split.json").read_text())
    return d,estimates,split


def matrix(d,estimated_p,H):
    x=np.concatenate((d["latent"],estimated_p,d["behavior_action_sequence"][:,:H].reshape(len(estimated_p),2*H)),axis=1)
    y=np.max(d["future_force_sequence"][:,:H],axis=1).astype(np.float64)
    if x.shape!=(len(y),294+2*H) or not np.all(np.isfinite(x)) or np.any(y<0):
        raise RuntimeError(f"Invalid G9 H{H} direct risk matrix")
    return x.astype(np.float64),y


def predict(net,x,mean,scale):
    net.eval()
    with torch.inference_mode():
        parts=[]
        for start in range(0,len(x),4096):
            block=torch.as_tensor((x[start:start+4096]-mean)/scale,dtype=torch.float32,device=DEVICE)
            parts.append(net(block).cpu().numpy())
    p=np.concatenate(parts).astype(np.float64)
    if p.shape!=(len(x),4) or np.any(~np.isfinite(p)) or np.any(np.diff(p,axis=1)<0) or np.any(p<0):
        raise RuntimeError("G9 noncrossing/nonnegative prediction invariant failed")
    return p


def train():
    cuda_gate();d,estimates,split=data()
    if (OUT/"model_manifest.json").exists():raise FileExistsError(OUT/"model_manifest.json")
    definitions={"status":"G9_DIRECT_RISK_PREDECLARED_BEFORE_MODEL_TRAINING",
        "estimators":ESTIMATORS,"horizons":HORIZONS,"quantiles":TAUS,
        "state_input":"frozen 256D TD-MPC2 current-observation latent + E0/E1/E2 estimated current 38D privileged-physical target",
        "action_input":"explicit Hx2 candidate sequence; development training uses realized frozen-policy behavior sequences",
        "input_dimension_by_horizon":{str(H):294+2*H for H in HORIZONS},
        "target":"R_H=max true G2 max-substep distal force over next H complete same-episode intervals",
        "model":"Linear(input,64)-ReLU-Linear(64,64)-ReLU-Linear(64,4), softplus increments cumulatively summed to Q50/Q90/Q95/Q99",
        "loss":"equal-weight pinball across four quantiles","optimizer":"Adam lr=0.001 batch=512 max_epochs=60 patience=8",
        "normalization":"TRAIN input mean/std only, std<1e-6 becomes 1; no target scaling",
        "checkpoint_selection":"minimum VALIDATION four-quantile pinball, strictly before development-test evaluation",
        "Q95_calibration":"nonnegative global additive correction from VALIDATION 95th percentile of residual y-Q95 with higher order statistic; applied only to decision score, not raw quantile output",
        "no_recursive_predicted_state_rollout":True,"no_future_state_or_force_input":True,
        "H5_limitation":"TD-MPC2 planner horizon is 3; H5 behavior-sequence diagnostics are not a deployable current-plan score"}
    json_new(OUT/"direct_quantile_model_definition.json",definitions)
    json_new(OUT/"action_sequence_protocol.json",{"frozen_controller":"TD-MPC2 step_280000.pt",
        "planning_horizon":3,"CEM_iterations":6,"CEM_samples":512,"CEM_elites":64,"policy_trajectories":24,
        "CEM_initial_std":2.0,"CEM_min_std":0.05,"temperature":0.5,
        "warm_start":"previous optimized mean shifted one step; final eval action=clipped first mean action",
        "development_training_sequence":"successive realized frozen-TD-MPC2 behavior actions, not falsely relabeled as one time-t plan",
        "prospective_H1_H2_H3":"must use plan mean captured at t and matched true-SOFA counterfactual fork labels",
        "H5":"no native 5-step TD-MPC2 plan; behavior sequence is retrospective diagnostic only",
        "controller_source_sha256":sha(RL/"tdmpc2/agent.py")})
    manifest={"training_seed_by_model":{},"checkpoint_sha256":{},"normalization_sha256":{},
        "calibration_delta_Q95_N":{},"validation_pinball":{},"dataset_sha256":sha(OUT/"development_temporal_behavior.npz"),
        "estimated_state_sha256":sha(OUT/"estimated_current_physical.npz"),
        "training_code_sha256":sha(__file__),"selection":"validation only; no development-test or confirmation labels opened"}
    for e_idx,name in enumerate(ESTIMATORS):
        for H in HORIZONS:
            x,y=matrix(d,estimates[name],H)
            tr=np.isin(d["seed"],split["train"]);va=np.isin(d["seed"],split["validation"])
            mean=np.mean(x[tr],axis=0);scale=np.std(x[tr],axis=0);scale[scale<1e-6]=1.
            folder=OUT/"models"/"risk"/name/f"H{H}";folder.mkdir(parents=True,exist_ok=False)
            json_new(folder/"normalization.json",{"mean":mean.tolist(),"std":scale.tolist(),"train_only":True})
            tx=torch.as_tensor((x[tr]-mean)/scale,dtype=torch.float32,device=DEVICE)
            ty=torch.as_tensor(y[tr],dtype=torch.float32,device=DEVICE)
            vx=torch.as_tensor((x[va]-mean)/scale,dtype=torch.float32,device=DEVICE)
            vy=torch.as_tensor(y[va],dtype=torch.float32,device=DEVICE)
            seed=260300+e_idx*10+H
            torch.manual_seed(seed);net=QuantileNet(x.shape[1]).to(DEVICE)
            optimizer=torch.optim.Adam(net.parameters(),lr=1e-3)
            generator=torch.Generator(device="cpu").manual_seed(seed)
            best=math.inf;patience=0;history=[]
            for epoch in range(60):
                net.train();order=torch.randperm(len(tx),generator=generator).to(DEVICE)
                for chunk in order.split(512):
                    loss=pinball(net(tx[chunk]),ty[chunk])
                    optimizer.zero_grad();loss.backward();optimizer.step()
                net.eval()
                with torch.inference_mode():value=float(pinball(net(vx),vy).cpu())
                history.append({"epoch":epoch,"validation_pinball_N":value})
                if value<best-1e-5:
                    best=value;patience=0;torch.save(net.cpu().state_dict(),folder/"model.pt");net.to(DEVICE)
                else:
                    patience+=1
                    if patience>=8:break
            net.load_state_dict(torch.load(folder/"model.pt",map_location=DEVICE,weights_only=True))
            val_pred=predict(net,x[va],mean,scale)
            residual=y[va]-val_pred[:,2]
            delta=max(0.,float(np.quantile(residual,.95,method="higher")))
            json_new(folder/"selection.json",{"training_seed":seed,"best_validation_pinball_N":best,
                "history":history,"validation_only_Q95_delta_N":delta,
                "validation_raw_Q95_coverage":float(np.mean(y[va]<=val_pred[:,2])),
                "validation_corrected_Q95_coverage":float(np.mean(y[va]<=val_pred[:,2]+delta))})
            key=f"{name}_H{H}"
            manifest["training_seed_by_model"][key]=seed
            manifest["checkpoint_sha256"][key]=sha(folder/"model.pt")
            manifest["normalization_sha256"][key]=sha(folder/"normalization.json")
            manifest["calibration_delta_Q95_N"][key]=delta
            manifest["validation_pinball"][key]=best
            print(f"G9 trained {key}: val pinball={best:.5f} Q95 delta={delta:.5f}",flush=True)
    json_new(OUT/"model_manifest.json",manifest)


def load_model(name,H):
    manifest=json.loads((OUT/"model_manifest.json").read_text())
    key=f"{name}_H{H}";folder=OUT/"models"/"risk"/name/f"H{H}"
    if sha(folder/"model.pt")!=manifest["checkpoint_sha256"][key] or sha(folder/"normalization.json")!=manifest["normalization_sha256"][key]:
        raise RuntimeError(f"G9 frozen risk model changed {key}")
    norm=json.loads((folder/"normalization.json").read_text())
    net=QuantileNet(len(norm["mean"])).to(DEVICE)
    net.load_state_dict(torch.load(folder/"model.pt",map_location=DEVICE,weights_only=True));net.eval()
    return net,np.asarray(norm["mean"]),np.asarray(norm["std"]),manifest["calibration_delta_Q95_N"][key]


def metrics(y,p,delta,name,H,group):
    score=p[:,2]+delta;unsafe=y>F_LIM;accepted=score<=F_LIM
    true_safe=~unsafe
    return {"estimator":name,"horizon":H,"group":group,"rows":len(y),
        "true_high_risk_count":int(np.sum(unsafe)),"accepted_count":int(np.sum(accepted)),
        "Q95_false_safe_count":int(np.sum(unsafe&accepted)),
        "Q95_false_safe_rate":float(np.mean(accepted[unsafe])) if np.any(unsafe) else None,
        "Q95_false_warning_count":int(np.sum(true_safe&~accepted)),
        "Q95_false_warning_rate":float(np.mean((~accepted)[true_safe])) if np.any(true_safe) else None,
        "Q95_warning_recall":float(np.mean((~accepted)[unsafe])) if np.any(unsafe) else None,
        "Q95_raw_coverage":float(np.mean(y<=p[:,2])),
        "Q95_corrected_coverage":float(np.mean(y<=score)),
        "Q95_calibration_delta_N":float(delta),
        "Q50_MAE_N":float(np.mean(np.abs(y-p[:,0]))),
        "mean_four_quantile_pinball_N":float(np.mean([np.mean(np.maximum(t*(y-p[:,i]),(t-1)*(y-p[:,i]))) for i,t in enumerate(TAUS)]))}


def evaluate():
    cuda_gate();d,estimates,split=data();manifest=json.loads((OUT/"model_manifest.json").read_text())
    if manifest["training_code_sha256"]!=sha(__file__) or manifest["dataset_sha256"]!=sha(OUT/"development_temporal_behavior.npz"):
        raise RuntimeError("G9 risk model/data changed after training")
    horizon_rows=[];coverage_rows=[];pred_arrays={}
    for name in ESTIMATORS:
        for H in HORIZONS:
            x,y=matrix(d,estimates[name],H)
            net,mean,scale,delta=load_model(name,H)
            p=predict(net,x,mean,scale)
            pred_arrays[f"{name}_H{H}"]=p.astype(np.float32)
            for group in ("train","validation","development_test"):
                mask=np.isin(d["seed"],split[group])
                horizon_rows.append(metrics(y[mask],p[mask],delta,name,H,group))
                for i,tau in enumerate(TAUS):
                    coverage_rows.append({"estimator":name,"horizon":H,"group":group,
                        "quantile":f"Q{int(tau*100)}","tau":tau,"rows":int(np.sum(mask)),
                        "empirical_coverage":float(np.mean(y[mask]<=p[mask,i])),
                        "coverage_error":float(np.mean(y[mask]<=p[mask,i])-tau),
                        "mean_pinball_N":float(np.mean(np.maximum(tau*(y[mask]-p[mask,i]),
                            (tau-1)*(y[mask]-p[mask,i])))),
                        "crossing_count":int(np.sum(np.diff(p[mask],axis=1)<0))})
    np.savez_compressed(OUT/"development_risk_predictions.npz",**pred_arrays)
    csv_new(OUT/"horizon_prediction_metrics.csv",horizon_rows)
    csv_new(OUT/"quantile_coverage_metrics.csv",coverage_rows)
    print("G9 direct quantile evaluation complete for 3 estimators x 4 horizons",flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=("train","evaluate"),required=True)
    stage=parser.parse_args().stage
    {"train":train,"evaluate":evaluate}[stage]()


if __name__=="__main__":main()
