"""Freeze one full pre-action representation, then fit three G4-identical regressors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
sys.path.insert(0, str(RL / "experiments/generic_solver_g6a"))
sys.path.insert(0, str(RL / "experiments/generic_solver_g5"))
sys.path.insert(0, str(RL / "experiments/generic_solver_g4"))
from g6_prepare import verify_source, sha, G5, G2, G1B, EXPECTED_CHECKPOINT_SHA
from g5_data import source
from g4_data import candidate_states
import g4_train
from g6b_features import FrozenVesselGeometry, EXTRA_NAMES, BASE_INDICES, full_state

OUT = RL / "logs/generic_solver_phase_g6b_full_contact_predictor"
SEEDS = (260206, 260207, 260208)
CALIBRATION = list(range(97000, 97030))
CONFIRMATION = list(range(98000, 98030))


def new_json(path, value):
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def cuda_gate():
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available() or
            torch.cuda.device_count() != 1 or "RTX 5090" not in torch.cuda.get_device_name(0)):
        raise RuntimeError("G6B requires CUDA_VISIBLE_DEVICES=0 RTX 5090; no CPU fallback")


def metadata():
    return {"phase":"G6B", "branch":"feature/generic-contact-force-safety", "python":sys.executable,
            "torch":torch.__version__, "torch_cuda":torch.version.cuda,
            "cuda_available":torch.cuda.is_available(), "CUDA_VISIBLE_DEVICES":os.environ["CUDA_VISIBLE_DEVICES"],
            "visible_cuda_device_count":torch.cuda.device_count(), "model_device_index":0,
            "device_name":torch.cuda.get_device_name(0), "training_seeds":list(SEEDS)}


def predeclare():
    cuda_gate()
    if OUT.exists():
        raise FileExistsError(OUT)
    q=verify_source()
    geometry=FrozenVesselGeometry()
    arrays, split, index, physical=source()
    extras={key:geometry.extra(p) for key,p in physical.items()}
    if len(extras)!=15300 or any(not np.all(np.isfinite(v)) for v in extras.values()):
        raise RuntimeError("Full pre-action geometry audit incomplete")
    extra_array=np.asarray(list(extras.values()))
    OUT.mkdir(parents=True)
    new_json(OUT/"metadata.json", {**metadata(), "G2_reference_quantiles_nominal_N":q,
        "navigation_checkpoint_sha256":EXPECTED_CHECKPOINT_SHA,
        "G5_pre_confirmation_freeze_sha256":sha(G5/"pre_confirmation_freeze.json"),
        "G6A_pre_confirmation_freeze_sha256":sha(RL/"logs/generic_solver_phase_g6a_boundary_calibration/pre_confirmation_freeze.json"),
        "G2_step_force_sha256":sha(G2/"step_level_force.csv")})
    new_json(OUT/"full_contact_feature_definition.json", {
        "status":"FROZEN_BEFORE_MODEL_EVALUATION", "total_state_dimension":256+len(BASE_INDICES)+len(EXTRA_NAMES),
        "input_dimension_with_candidate_action":256+len(BASE_INDICES)+len(EXTRA_NAMES)+2,
        "order":["frozen_navigation_latent_z_256", "G5_sorted_K_plus_D_P_indices_33", "pre_action_extras_17", "candidate_action_translation_rotation_2"],
        "G5_K_plus_D_P_indices":BASE_INDICES, "extras_order":list(EXTRA_NAMES),
        "timing":"d>=1 after preceding interval and before selected/requested candidate action d",
        "force_fields":"exact current/previous-final-substep G4 solver-reconstructed distal contact force; never F_next",
        "contact_fields":"exact current G4 solver contact group count; simulation-only, no physical-sensor equivalence established",
        "geometry":"frozen AorticArch seed=30 scaling=[1,1,1,0.75]; VTK implicit signed distance positive inside, outward normal=-gradient; five current distal nodes from G4 P",
        "wall_distance_units":"mm", "force_units":"nominal N", "velocity_units":"mm/s",
        "mesh_points_faces_canonical_sha256":geometry.canonical_sha256,
        "mesh_points":geometry.mesh.n_points, "mesh_triangles":geometry.mesh.n_cells,
        "feature_extractor_sha256":sha(HERE/"g6b_features.py"),
        "G4_P_source_definition_sha256":sha(RL/"logs/generic_solver_phase_g4_state_observability/privileged_feature_definition.json"),
        "development_preaction_P_rows_audited":len(extras), "extra_min":extra_array.min(axis=0).tolist(),
        "extra_max":extra_array.max(axis=0).tolist(), "all_finite":True,
        "no_future_post_action_or_candidate_outcome_input":True,
        "physical_deployability_caveat":"Current exact contact force and count are Generic solver-derived and cannot be assumed measurable on hardware; K/D and wall geometry require validated tracking/segmentation."})
    for group,seeds in (("calibration",CALIBRATION),("confirmation",CONFIRMATION)):
        new_json(OUT/f"{group}_seed_protocol.json", {"status":"PREDECLARED_BEFORE_NEW_LABELS", "seeds":seeds,
            "episodes":30,"states_per_episode":4,"candidates_per_state":21,"expected_labels":2520,
            "behavior_lifecycle":"one reused Generic scene, ordered resets",
            "candidate_lifecycle":"fresh reused Generic scene, ordered full replay and restore/fork at selected states",
            "state_selection":"exact G3/G5 four-state progress/prior-force rule, group=confirmation",
            "candidate_protocol":"exact G3/G5 21 actions, group=confirmation, RNG offset 400000",
            "confirmation_only_after_freeze":group=="confirmation"})
    new_json(OUT/"calibration_selection_protocol.json", {"F_lim_N":q["q95"],
        "raw_first":"if every replica empirical Q95 false-safe rate <=5%, c_star=F_lim",
        "otherwise":"largest float64 cutoff <=F_lim with worst of three calibration FSR <=5%; exact-event sweep",
        "selection_dataset":"97000-97029 only", "no_ensemble_or_best_replica":True})
    print("G6B pre-action 15300-row feature audit PASS; full 306-D state frozen",flush=True)


def matrices(arrays,split,index,physical,geometry,group):
    allowed=set(split[group])
    keys=[(int(s),int(d)) for s,d in zip(arrays["seed"],arrays["decision_index"]) if int(s) in allowed and int(d)>=1]
    x=np.asarray([np.r_[full_state(arrays["latent"][index[key]],physical[key],geometry),arrays["action"][index[key]]] for key in keys],dtype=np.float64)
    y=np.asarray([arrays["force_next_max_N"][index[key]] for key in keys],dtype=np.float64)
    return x,y,keys


def candidate_matrix(states,arrays,index,physical,geometry):
    x=[]; y=[]; contact=[]
    for state in states:
        key=(int(state["seed"]),int(state["decision_index"]))
        if not np.array_equal(np.asarray(state["observation"],np.float32),arrays["observation"][index[key]]):
            raise RuntimeError("Candidate state/behavior alignment failed")
        v=full_state(arrays["latent"][index[key]],physical[key],geometry)
        for candidate in state["candidates"]:
            x.append(np.r_[v,candidate["action"]]); y.append(candidate["force_next_max_N"])
            contact.append(float(physical[key][34])>0)
    return np.asarray(x),np.asarray(y),np.asarray(contact,dtype=bool)


def train():
    cuda_gate(); verify_source()
    definition=json.loads((OUT/"full_contact_feature_definition.json").read_text())
    if definition["feature_extractor_sha256"]!=sha(HERE/"g6b_features.py"):
        raise RuntimeError("Full feature vector changed after predeclaration")
    geometry=FrozenVesselGeometry()
    arrays,split,index,physical=source()
    xtr,ytr,_=matrices(arrays,split,index,physical,geometry,"train")
    xva,yva,_=matrices(arrays,split,index,physical,geometry,"validation")
    if xtr.shape!=(10903,308) or xva.shape!=(2286,308):
        raise RuntimeError("G5-matched training/validation rows changed")
    mean=np.mean(xtr,axis=0); scale=np.std(xtr,axis=0); scale[scale<1e-8]=1.
    new_json(OUT/"models/FULL/normalization.json",{"source":"TRAIN behavior rows d>=1 only, shared across three replicas", "input_mean":mean.tolist(),"input_std":scale.tolist(),"target_normalization":"none"})
    settings=json.loads((G5/"training_seed_manifest.json").read_text())
    if settings["training_seeds"]!=list(SEEDS): raise RuntimeError("G5 seed convention changed")
    g4_train.OUT=OUT
    for seed in SEEDS:
        g4_train.fit("FULL",seed,xtr,ytr,xva,yva,mean,scale,len(ytr),torch.device("cuda:0"),settings)
    new_json(OUT/"model_manifest.json",{"status":"THREE_BEHAVIOR_ONLY_REPLICAS_VALIDATION_SELECTED",
        "model_family":"308-64-ReLU-64-ReLU-1", "loss":"ordinary raw force MSE", "training_protocol":"exact G4/G5 g4_train.fit behavior-only",
        "training_seeds":list(SEEDS),"train_behavior_rows":len(ytr),"validation_behavior_rows":len(yva),
        "normalization_sha256":sha(OUT/"models/FULL/normalization.json"),
        "feature_definition_sha256":sha(OUT/"full_contact_feature_definition.json"),
        "checkpoint_sha256":{str(seed):sha(OUT/f"models/FULL/seed_{seed}/model.pt") for seed in SEEDS},
        "selected_epoch":{str(seed):json.loads((OUT/f"models/FULL/seed_{seed}/selection.json").read_text())["selected_epoch"] for seed in SEEDS},
        "development_test_opened_before_training":False,"new_calibration_or_confirmation_opened_before_training":False})


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=("predeclare","train"));a=parser.parse_args()
    {"predeclare":predeclare,"train":train}[a.stage]()
