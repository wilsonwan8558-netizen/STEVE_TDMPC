"""Audit frozen G6B inputs and fit one four-output non-crossing quantile formulation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
RL=HERE.parents[1]
for name in ("generic_solver_g6b","generic_solver_g6a","generic_solver_g5","generic_solver_g4"):
    sys.path.insert(0,str(RL/"experiments"/name))
from g6b_freeze import verify as verify_g6b
from g6b_prepare_train import OUT as G6B, matrices, candidate_matrix, SEEDS, sha
from g6b_features import FrozenVesselGeometry, EXTRA_NAMES, BASE_INDICES
from g5_data import source
from g6_prepare import verify_source

OUT=RL/"logs/generic_solver_phase_g7_quantile_force_prediction"
CALIBRATION=list(range(99000,99030))
CONFIRMATION=list(range(100000,100030))
TAUS=(.50,.90,.95,.99)


def new_json(path,value):
    if path.exists():raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")


def cuda_gate():
    if (os.environ.get("CUDA_VISIBLE_DEVICES")!="0" or not torch.cuda.is_available() or
            torch.cuda.device_count()!=1 or "RTX 5090" not in torch.cuda.get_device_name(0)):
        raise RuntimeError("G7 requires CUDA_VISIBLE_DEVICES=0 RTX 5090; no CPU fallback")


def array_sha(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str(a.shape).encode()+str(a.dtype).encode()+a.tobytes()).hexdigest()


def audit():
    freeze=verify_g6b();q=verify_source()
    definition=json.loads((G6B/"full_contact_feature_definition.json").read_text())
    old_manifest=json.loads((G6B/"model_manifest.json").read_text())
    if (definition["total_state_dimension"]!=306 or definition["input_dimension_with_candidate_action"]!=308 or
            definition["G5_K_plus_D_P_indices"]!=BASE_INDICES or definition["extras_order"]!=list(EXTRA_NAMES) or
            definition["feature_extractor_sha256"]!=sha(RL/"experiments/generic_solver_g6b/g6b_features.py") or
            definition["no_future_post_action_or_candidate_outcome_input"] is not True):
        raise RuntimeError("Exact G6B input unavailable")
    geometry=FrozenVesselGeometry()
    if geometry.canonical_sha256!=definition["mesh_points_faces_canonical_sha256"]:
        raise RuntimeError("G6B vessel geometry changed")
    arrays,split,index,physical=source()
    xtr,ytr,ktr=matrices(arrays,split,index,physical,geometry,"train")
    xva,yva,kva=matrices(arrays,split,index,physical,geometry,"validation")
    if xtr.shape!=(10903,308) or xva.shape!=(2286,308) or np.any(ytr<0) or np.any(yva<0):
        raise RuntimeError("G6B training rows/targets changed")
    norm=json.loads((G6B/"models/FULL/normalization.json").read_text())
    mean=np.mean(xtr,axis=0);scale=np.std(xtr,axis=0);scale[scale<1e-8]=1.
    if (old_manifest["normalization_sha256"]!=sha(G6B/"models/FULL/normalization.json") or
            not np.array_equal(mean,np.asarray(norm["input_mean"])) or
            not np.array_equal(scale,np.asarray(norm["input_std"])) or norm["target_normalization"]!="none"):
        raise RuntimeError("G6B exact train-only normalization not reproduced")
    settings=json.loads((RL/"logs/generic_solver_phase_g5_feature_attribution/training_seed_manifest.json").read_text())
    if settings["training_seeds"]!=list(SEEDS) or old_manifest["train_behavior_rows"]!=len(ytr) or old_manifest["validation_behavior_rows"]!=len(yva):
        raise RuntimeError("G6B behavior-only supervision/training convention changed")
    return q,definition,old_manifest,freeze,settings,(xtr,ytr,ktr,xva,yva,kva),norm


class QuantileNet(nn.Module):
    def __init__(self,input_dim=308):
        super().__init__()
        self.trunk=nn.Sequential(nn.Linear(input_dim,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU())
        self.head=nn.Linear(64,4)

    def forward(self,x):
        raw=self.head(self.trunk(x))
        positive=F.softplus(raw)
        return torch.cumsum(positive,dim=-1)


def pinball(pred,target):
    levels=torch.as_tensor(TAUS,dtype=pred.dtype,device=pred.device)
    error=target[:,None]-pred
    return torch.maximum(levels*error,(levels-1)*error).mean()


def predeclare():
    cuda_gate()
    if OUT.exists():raise FileExistsError(OUT)
    q,definition,old_manifest,freeze,settings,data,norm=audit()
    xtr,ytr,ktr,xva,yva,kva=data
    OUT.mkdir(parents=True)
    new_json(OUT/"metadata.json",{"phase":"G7","branch":"feature/generic-contact-force-safety",
        "python":sys.executable,"torch":torch.__version__,"torch_cuda":torch.version.cuda,
        "cuda_available":torch.cuda.is_available(),"CUDA_VISIBLE_DEVICES":os.environ["CUDA_VISIBLE_DEVICES"],
        "visible_cuda_device_count":torch.cuda.device_count(),"model_device_index":0,
        "device_name":torch.cuda.get_device_name(0),"F_lim_N":q["q95"],"Q99_reference_N":q["q99"],
        "G6B_pre_confirmation_freeze_sha256":sha(G6B/"pre_confirmation_freeze.json")})
    new_json(OUT/"g6b_input_audit.json",{"status":"EXACT_FROZEN_G6B_INPUT_REPRODUCED",
        "G6B_feature_definition_sha256":sha(G6B/"full_contact_feature_definition.json"),
        "G6B_feature_extractor_sha256":definition["feature_extractor_sha256"],
        "G6B_freeze_file_count":len(freeze["file_sha256"]),"G6B_normalization_sha256":old_manifest["normalization_sha256"],
        "total_state_dimension":306,"input_dimension":308,"input_order":definition["order"],
        "physical_indices":BASE_INDICES,"extras_order":list(EXTRA_NAMES),
        "coordinate_frames_and_units":definition["geometry"],"timing":definition["timing"],
        "no_future_or_post_action_input":True,"simulation_only_current_force_and_contact_count":True})
    new_json(OUT/"training_data_audit.json",{"status":"EXACT_G6B_BEHAVIOR_ONLY_TRAIN_VALIDATION_RECOVERED",
        "source":"frozen G3 behavior NPZ and G4 P keyed by (episode seed,decision index), same G6B matrices() function",
        "G6B_matrix_builder_sha256":sha(RL/"experiments/generic_solver_g6b/g6b_prepare_train.py"),
        "train_rows":len(ytr),"validation_rows":len(yva),"train_episodes":140,"validation_episodes":30,
        "behavior_only":True,"counterfactual_training_rows":0,"candidate_action_last_two_coordinates":True,
        "target":"maximum distal contact-force magnitude over next 22 SOFA substeps, nominal N",
        "train_x_sha256":array_sha(xtr),"train_y_sha256":array_sha(ytr),
        "validation_x_sha256":array_sha(xva),"validation_y_sha256":array_sha(yva),
        "train_keys_sha256":hashlib.sha256(json.dumps(ktr,separators=(",",":")).encode()).hexdigest(),
        "validation_keys_sha256":hashlib.sha256(json.dumps(kva,separators=(",",":")).encode()).hexdigest(),
        "normalization_source":"TRAIN behavior rows only, x mean/std with <1e-8 fallback 1; target unnormalized",
        "G6B_normalization_exact_array_equal":True,"G6B_normalization_sha256":old_manifest["normalization_sha256"]})
    new_json(OUT/"models/normalization.json",norm)
    new_json(OUT/"quantile_model_definition.json",{"status":"PREDECLARED_BEFORE_MODEL_EVALUATION",
        "input_dimension":308,"trunk":"Linear(308,64)-ReLU-Linear(64,64)-ReLU",
        "raw_head":"Linear(64,4)","quantile_levels":list(TAUS),
        "noncrossing":"Q50=softplus(raw0); Q90=Q50+softplus(raw1); Q95=Q90+softplus(raw2); Q99=Q95+softplus(raw3)",
        "nonnegative_Q50":True,"loss":"equal-weight mean of standard pinball loss over four quantiles and batch",
        "output_units":"nominal N","checkpoint_selection":"minimum validation behavior mean four-quantile pinball only",
        "architecture_code_sha256":sha(HERE/"g7_prepare_train.py")})
    new_json(OUT/"training_seed_manifest.json",{"training_seeds":list(SEEDS),
        "training_settings":settings,"train_rows_per_validation_round":len(ytr),
        "optimizer":"Adam","validation_objective":"mean Q50/Q90/Q95/Q99 pinball",
        "same_G6B_budget_batch_optimizer_early_stopping":True,
        "development_test_or_new_calibration_or_confirmation_opened_before_training":False})
    for group,seeds in (("calibration",CALIBRATION),("confirmation",CONFIRMATION)):
        new_json(OUT/f"{group}_seed_protocol.json",{"status":"PREDECLARED_BEFORE_NEW_LABELS",
            "seeds":seeds,"episodes":30,"states_per_episode":4,"candidates_per_state":21,"expected_labels":2520,
            "behavior_lifecycle":"one reused Generic scene ordered resets",
            "candidate_lifecycle":"fresh reused Generic scene ordered replay and 21 restore/forks at selected states",
            "state_selection":"exact G3/G5 four-state progress/prior-force rule, group=confirmation",
            "candidate_protocol":"exact G3/G5 21 actions, group=confirmation, RNG offset 400000",
            "confirmation_only_after_freeze":group=="confirmation"})
    new_json(OUT/"residual_calibration_protocol.json",{"F_lim_N":q["q95"],
        "raw_first":"if all three empirical Q95 FSR <=5%, delta_cal=0",
        "otherwise":"smallest shared nonnegative additive Q95 correction yielding max across replicas empirical FSR <=5%",
        "exact_event_breakpoints":True,"no_negative_correction":True,"calibration_seeds":CALIBRATION,
        "untouched_confirmation_seeds":CONFIRMATION})
    print("G7 exact 308-D G6B input and 10903/2286 behavior rows audited and predeclared",flush=True)


def train():
    cuda_gate()
    q,definition,old_manifest,freeze,settings,data,norm=audit()
    xtr,ytr,ktr,xva,yva,kva=data
    if sha(HERE/"g7_prepare_train.py")!=json.loads((OUT/"quantile_model_definition.json").read_text())["architecture_code_sha256"]:
        raise RuntimeError("G7 quantile architecture changed after predeclaration")
    if sha(G6B/"models/FULL/normalization.json")!=json.loads((OUT/"training_data_audit.json").read_text())["G6B_normalization_sha256"]:
        raise RuntimeError("G6B normalization changed")
    mean=np.asarray(norm["input_mean"]);scale=np.asarray(norm["input_std"])
    device=torch.device("cuda:0")
    tr_x=torch.as_tensor((xtr-mean)/scale,dtype=torch.float32,device=device)
    tr_y=torch.as_tensor(ytr,dtype=torch.float32,device=device)
    va_x=torch.as_tensor((xva-mean)/scale,dtype=torch.float32,device=device)
    va_y=torch.as_tensor(yva,dtype=torch.float32,device=device)
    for seed in SEEDS:
        folder=OUT/f"models/seed_{seed}"
        folder.mkdir(parents=True,exist_ok=False)
        torch.manual_seed(seed)
        net=QuantileNet().to(device)
        optimizer=torch.optim.Adam(net.parameters(),lr=settings["learning_rate"],weight_decay=settings["weight_decay"])
        generator=torch.Generator(device="cpu").manual_seed(seed)
        best,best_epoch,stale,best_state=math.inf,None,0,None
        history=[];start=time.monotonic()
        for epoch in range(1,settings["max_validation_rounds"]+1):
            net.train()
            permutation=torch.randperm(len(tr_x),generator=generator)[:len(tr_x)]
            for offset in range(0,len(tr_x),settings["batch_size"]):
                indices=permutation[offset:offset+settings["batch_size"]].to(device)
                loss=pinball(net(tr_x[indices]),tr_y[indices])
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            net.eval()
            with torch.inference_mode():value=float(pinball(net(va_x),va_y).item())
            history.append({"epoch":epoch,"validation_mean_four_quantile_pinball_N":value})
            if value<best-settings["minimum_validation_MSE_improvement"]:
                best,best_epoch,stale=value,epoch,0
                best_state={key:val.detach().cpu().clone() for key,val in net.state_dict().items()}
            else:
                stale+=1
                if stale>=settings["early_stopping_patience"]:break
        torch.save(best_state,folder/"model.pt")
        new_json(folder/"training_history.json",history)
        new_json(folder/"selection.json",{"training_seed":seed,"selected_epoch":best_epoch,
            "selected_validation_mean_pinball_N":best,"train_rows":len(ytr),"validation_rows":len(yva),
            "samples_per_validation_round":len(ytr),"optimizer_steps_per_round":math.ceil(len(ytr)/settings["batch_size"]),
            "train_wall_seconds":time.monotonic()-start})
        print(f"G7 seed={seed} epoch={best_epoch} validation_mean_pinball={best:.6f}",flush=True)
    new_json(OUT/"model_manifest.json",{"status":"THREE_INDEPENDENT_VALIDATION_SELECTED_QUANTILE_REPLICAS",
        "training_seeds":list(SEEDS),"quantile_levels":list(TAUS),"normalization_sha256":sha(OUT/"models/normalization.json"),
        "model_definition_sha256":sha(OUT/"quantile_model_definition.json"),
        "checkpoint_sha256":{str(seed):sha(OUT/f"models/seed_{seed}/model.pt") for seed in SEEDS},
        "selected_epoch":{str(seed):json.loads((OUT/f"models/seed_{seed}/selection.json").read_text())["selected_epoch"] for seed in SEEDS},
        "development_or_new_seed_labels_opened_before_training":False})


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=("predeclare","train"));a=parser.parse_args()
    {"predeclare":predeclare,"train":train}[a.stage]()
