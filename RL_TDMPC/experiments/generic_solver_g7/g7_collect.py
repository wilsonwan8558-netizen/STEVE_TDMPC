"""Exact Generic replay/fork collection; independent G7/G6B scoring on shared labels."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
RL=HERE.parents[1]
for name in ("generic_solver_g6a","generic_solver_g6b","generic_solver_g5"):
    sys.path.insert(0,str(RL/"experiments"/name))
import g6_collect as inherited
from g6b_freeze import verify as verify_g6b
from g6b_prepare_train import OUT as G6B
from g6b_development import prediction as g6b_prediction
from g6b_features import FrozenVesselGeometry,full_state
from g7_prepare_train import OUT,SEEDS,CALIBRATION,CONFIRMATION,sha,cuda_gate,new_json
from g7_metrics import predict
from g6_prepare import verify_source


def verify_g7():
    verify_g6b();q=verify_source()
    definition=json.loads((OUT/"quantile_model_definition.json").read_text())
    manifest=json.loads((OUT/"model_manifest.json").read_text())
    if (definition["architecture_code_sha256"]!=sha(HERE/"g7_prepare_train.py") or
            manifest["model_definition_sha256"]!=sha(OUT/"quantile_model_definition.json") or
            manifest["normalization_sha256"]!=sha(OUT/"models/normalization.json") or
            sha(OUT/"models/normalization.json")!=sha(G6B/"models/FULL/normalization.json")):
        raise RuntimeError("G7 exact input/model/normalization changed")
    if any(manifest["checkpoint_sha256"][str(seed)]!=sha(OUT/f"models/seed_{seed}/model.pt") for seed in SEEDS):
        raise RuntimeError("G7 checkpoint changed")
    return q


def protocol(group):
    verify_g7()
    if group=="confirmation":
        from g7_freeze import verify
        verify()
    elif group!="calibration":raise ValueError(group)
    seeds=CALIBRATION if group=="calibration" else CONFIRMATION
    return seeds,OUT/f"{group}_behavior_episodes.csv",OUT/f"{group}_behavior_steps.csv",OUT/f"{group}_candidates"


def prepare_inherited():
    inherited.OUT=OUT;inherited.SEEDS=SEEDS
    inherited.CALIBRATION=CALIBRATION;inherited.CONFIRMATION=CONFIRMATION
    inherited.verify_source=verify_g7;inherited.protocol=protocol


def score(group):
    seeds,_,_,folder=protocol(group)
    audit=json.loads((OUT/f"{group}_replay_audit.json").read_text())
    if audit["status"]!="PASS" or audit["candidate_evaluations"]!=2520:
        raise RuntimeError("G7 replay/fork audit incomplete")
    states=[json.loads(path.read_text()) for path in sorted(folder.glob("state_*.json"))]
    if len(states)!=120 or [s["state_id"] for s in states]!=list(range(120)) or sorted({s["seed"] for s in states})!=seeds:
        raise RuntimeError("G7 candidate states incomplete")
    geometry=FrozenVesselGeometry()
    x=np.asarray([np.r_[full_state(s["latent"],s["physical"],geometry),c["action"]]
        for s in states for c in s["candidates"]],dtype=np.float64)
    y=np.asarray([c["force_next_max_N"] for s in states for c in s["candidates"]],dtype=np.float64)
    if x.shape!=(2520,308) or not np.all(np.isfinite(x)) or len(y)!=2520 or np.any(y<0):
        raise RuntimeError("G7 exact candidate input/target changed")
    q=verify_g7()
    predictions={seed:predict(seed,x) for seed in SEEDS}
    g6b={seed:g6b_prediction(seed,x) for seed in SEEDS} if group=="confirmation" else None
    rows=[]
    for i,state in enumerate(states):
        physical=np.asarray(state["physical"],dtype=float)
        for j,candidate in enumerate(state["candidates"]):
            k=i*21+j
            row={"episode_seed":state["seed"],"decision_index":state["decision_index"],"state_id":state["state_id"],
                "candidate_id":j,"candidate_kind":candidate["kind"],"current_contact":int(physical[34]>0),
                "current_distal_force_N":float(physical[33]),"action_translation":float(candidate["action"][0]),
                "action_rotation":float(candidate["action"][1]),"predictor_input_json":json.dumps(x[k].tolist(),separators=(",",":")),
                "F_true_N":float(y[k]),"true_Q95_high_force":int(y[k]>q["q95"])}
            for seed in SEEDS:
                for m,name in enumerate(("Q50","Q90","Q95","Q99")):
                    row[f"{name}_seed_{seed}_N"]=float(predictions[seed][k,m])
                if g6b is not None:row[f"G6B_point_seed_{seed}_N"]=float(g6b[seed][k])
            rows.append(row)
    path=OUT/f"{group}_candidate_results.csv"
    if path.exists():raise FileExistsError(path)
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    new_json(OUT/f"{group}_scoring_audit.json",{"status":"PASS_THREE_G7_REPLICAS_SAME_LABELS",
        "seeds":seeds,"states":120,"candidate_rows":2520,"input_dimension":308,
        "Q95_true_high_force_count":int(np.sum(y>q["q95"])),
        "Q99_true_extreme_count":int(np.sum(y>q["q99"])),
        "G7_checkpoint_sha256":{str(seed):sha(OUT/f"models/seed_{seed}/model.pt") for seed in SEEDS},
        "G6B_same_rows_scored":g6b is not None,
        "G6B_checkpoint_sha256":{str(seed):sha(G6B/f"models/FULL/seed_{seed}/model.pt") for seed in SEEDS} if g6b is not None else None,
        "candidate_results_sha256":sha(path)})
    print(f"G7 {group}: scored 2520 shared labels; G6B same-data baseline={g6b is not None}",flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--group",choices=("calibration","confirmation"),required=True)
    parser.add_argument("--stage",choices=("behavior","candidates","score"),required=True);a=parser.parse_args()
    cuda_gate();prepare_inherited()
    {"behavior":inherited.behavior,"candidates":inherited.candidates,"score":score}[a.stage](a.group)


if __name__=="__main__":main()
