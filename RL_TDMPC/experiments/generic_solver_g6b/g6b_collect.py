"""Reuse the audited G6A Generic scene/replay/fork collector with G6B seeds and scoring."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
RL=HERE.parents[1]
sys.path.insert(0,str(RL/"experiments/generic_solver_g6a"))
sys.path.insert(0,str(RL/"experiments/generic_solver_g5"))
import g6_collect as inherited
from g6b_prepare_train import OUT,SEEDS,CALIBRATION,CONFIRMATION,sha,verify_source,cuda_gate,new_json
from g6b_features import FrozenVesselGeometry,full_state
from g6b_development import prediction


def verify_g6b():
    q=verify_source()
    manifest=json.loads((OUT/"model_manifest.json").read_text())
    if manifest["feature_definition_sha256"]!=sha(OUT/"full_contact_feature_definition.json"):
        raise RuntimeError("Feature definition changed")
    if manifest["normalization_sha256"]!=sha(OUT/"models/FULL/normalization.json"):
        raise RuntimeError("Normalization changed")
    if any(manifest["checkpoint_sha256"][str(seed)]!=sha(OUT/f"models/FULL/seed_{seed}/model.pt") for seed in SEEDS):
        raise RuntimeError("G6B model changed")
    definition=json.loads((OUT/"full_contact_feature_definition.json").read_text())
    if definition["feature_extractor_sha256"]!=sha(HERE/"g6b_features.py"):
        raise RuntimeError("Feature implementation changed")
    return q


def protocol(group):
    verify_g6b()
    if group=="confirmation":
        from g6b_freeze import verify
        verify()
    elif group!="calibration":
        raise ValueError(group)
    seeds=CALIBRATION if group=="calibration" else CONFIRMATION
    return seeds,OUT/f"{group}_behavior_episodes.csv",OUT/f"{group}_behavior_steps.csv",OUT/f"{group}_candidates"


def prepare_inherited():
    inherited.OUT=OUT
    inherited.SEEDS=SEEDS
    inherited.CALIBRATION=CALIBRATION
    inherited.CONFIRMATION=CONFIRMATION
    inherited.verify_source=verify_g6b
    inherited.protocol=protocol


def score(group):
    seeds,_,_,folder=protocol(group)
    audit=json.loads((OUT/f"{group}_replay_audit.json").read_text())
    if audit["status"]!="PASS" or audit["candidate_evaluations"]!=2520:
        raise RuntimeError("Replay/fork validation failed")
    states=[json.loads(path.read_text()) for path in sorted(folder.glob("state_*.json"))]
    if len(states)!=120 or [s["state_id"] for s in states]!=list(range(120)) or sorted({s["seed"] for s in states})!=seeds:
        raise RuntimeError("Candidate state set incomplete")
    geometry=FrozenVesselGeometry()
    x=np.asarray([np.r_[full_state(s["latent"],s["physical"],geometry),c["action"]]
                  for s in states for c in s["candidates"]],dtype=np.float64)
    y=np.asarray([c["force_next_max_N"] for s in states for c in s["candidates"]],dtype=np.float64)
    if x.shape!=(2520,308) or len(y)!=2520 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise RuntimeError("G6B candidate input/labels invalid")
    q=verify_g6b()
    predictions={seed:prediction(seed,x) for seed in SEEDS}
    rows=[]
    for i,state in enumerate(states):
        physical=np.asarray(state["physical"],dtype=float)
        for j,candidate in enumerate(state["candidates"]):
            k=i*21+j
            rows.append({"episode_seed":state["seed"],"decision_index":state["decision_index"],"state_id":state["state_id"],
                "candidate_id":j,"candidate_kind":candidate["kind"],"current_contact":int(physical[34]>0),
                "current_distal_force_N":float(physical[33]),"action_translation":float(candidate["action"][0]),
                "action_rotation":float(candidate["action"][1]),"predictor_input_json":json.dumps(x[k].tolist(),separators=(",",":")),
                "F_true_N":float(y[k]),"true_Q95_high_force":int(y[k]>q["q95"]),
                **{f"prediction_seed_{seed}_N":float(predictions[seed][k]) for seed in SEEDS}})
    path=OUT/f"{group}_candidate_results.csv"
    if path.exists():raise FileExistsError(path)
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    new_json(OUT/f"{group}_scoring_audit.json",{"status":"PASS_THREE_FROZEN_REPLICA_PREDICTIONS",
        "seeds":seeds,"states":120,"candidate_rows":2520,"model_training_seeds":list(SEEDS),
        "checkpoint_hashes":{str(seed):sha(OUT/f"models/FULL/seed_{seed}/model.pt") for seed in SEEDS},
        "input_dimension":308,"Q95_true_high_force_count":int(np.sum(y>q["q95"])),
        "candidate_results_sha256":sha(path)})
    print(f"G6B {group}: scored 2520 labels with three replicas",flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--group",choices=("calibration","confirmation"),required=True)
    parser.add_argument("--stage",choices=("behavior","candidates","score"),required=True);a=parser.parse_args()
    cuda_gate();prepare_inherited()
    {"behavior":inherited.behavior,"candidates":inherited.candidates,"score":score}[a.stage](a.group)


if __name__=="__main__":main()
