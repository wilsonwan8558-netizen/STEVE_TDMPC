"""One-time G8 Generic behavior/replay/fork labels and frozen G7/G6B scoring."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
RL=HERE.parents[1]
for name in ("generic_solver_g6a","generic_solver_g7","generic_solver_g6b","generic_solver_g5"):
    sys.path.insert(0,str(RL/"experiments"/name))
import g6_collect as inherited
from g8_prepare import OUT,TEST_SEEDS,verify,cuda_gate,sha,new_json,G7,G6B,MODEL_SEEDS
from g8_repair import validate_candidates
from g6b_features import FrozenVesselGeometry,full_state
from g7_metrics import predict as g7_predict
from g6b_development import prediction as g6b_predict
from g6_prepare import verify_source


def protocol(group):
    if group!="confirmation":raise ValueError("G8 has one previously unopened test group")
    verify()
    return TEST_SEEDS,OUT/"confirmation_behavior_episodes.csv",OUT/"confirmation_behavior_steps.csv",OUT/"confirmation_candidates"


def prepare_inherited():
    inherited.OUT=OUT
    inherited.SEEDS=MODEL_SEEDS
    inherited.CALIBRATION=[]
    inherited.CONFIRMATION=TEST_SEEDS
    inherited.verify_source=verify_source
    inherited.protocol=protocol


def score():
    seeds,_,_,folder=protocol("confirmation")
    audit=json.loads((OUT/"confirmation_replay_audit.json").read_text())
    if audit["status"]!="PASS" or audit["candidate_evaluations"]!=2520:
        raise RuntimeError("G8 replay/fork integrity gate failed")
    states=[json.loads(path.read_text()) for path in sorted(folder.glob("state_*.json"))]
    if len(states)!=120 or [s["state_id"] for s in states]!=list(range(120)) or sorted({s["seed"] for s in states})!=seeds:
        raise RuntimeError("G8 selected states incomplete")
    for state in states:validate_candidates(state["candidates"],state["behavior_action"])
    geometry=FrozenVesselGeometry()
    x=np.asarray([np.r_[full_state(s["latent"],s["physical"],geometry),c["action"]]
        for s in states for c in s["candidates"]],dtype=np.float64)
    y=np.asarray([c["force_next_max_N"] for s in states for c in s["candidates"]],dtype=np.float64)
    if x.shape!=(2520,308) or y.shape!=(2520,) or np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)) or np.any(y<0):
        raise RuntimeError("G8 candidate input or target changed")
    q=verify_source()
    g7={seed:g7_predict(seed,x) for seed in MODEL_SEEDS}
    g6b={seed:g6b_predict(seed,x) for seed in MODEL_SEEDS}
    rows=[]
    for i,state in enumerate(states):
        physical=np.asarray(state["physical"],dtype=float)
        for j,candidate in enumerate(state["candidates"]):
            k=i*21+j
            row={"episode_seed":state["seed"],"decision_index":state["decision_index"],
                "state_id":state["state_id"],"candidate_id":j,"candidate_kind":candidate["kind"],
                "current_contact":int(physical[34]>0),"current_distal_force_N":float(physical[33]),
                "action_translation":float(candidate["action"][0]),"action_rotation":float(candidate["action"][1]),
                "predictor_input_json":json.dumps(x[k].tolist(),separators=(",",":")),
                "F_true_N":float(y[k]),"true_Q95_high_force":int(y[k]>q["q95"]),
                "true_Q99_extreme":int(y[k]>q["q99"])}
            for seed in MODEL_SEEDS:
                for m,name in enumerate(("Q50","Q90","Q95","Q99")):
                    row[f"G7_{name}_seed_{seed}_N"]=float(g7[seed][k,m])
                row[f"G6B_point_seed_{seed}_N"]=float(g6b[seed][k])
            rows.append(row)
    path=OUT/"candidate_results.csv"
    if path.exists():raise FileExistsError(path)
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    new_json(OUT/"candidate_scoring_audit.json",{"status":"PASS_G8_SAME_2520_LABELS_G7_G6B_ORACLE",
        "states":120,"candidate_rows":2520,"candidate_zero_exact_nominal_count":120,
        "G7_checkpoint_sha256":{str(seed):sha(G7/f"models/seed_{seed}/model.pt") for seed in MODEL_SEEDS},
        "G6B_checkpoint_sha256":{str(seed):sha(G6B/f"models/FULL/seed_{seed}/model.pt") for seed in MODEL_SEEDS},
        "Q95_true_high_force_candidates":int(np.sum(y>q["q95"])),
        "Q99_true_extreme_candidates":int(np.sum(y>q["q99"])),
        "candidate_results_sha256":sha(path)})
    print("G8 scored 120 states/2520 one-time labels for three G7 and three G6B frozen replicas",flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=("behavior","candidates","score"),required=True)
    args=parser.parse_args();cuda_gate();prepare_inherited()
    {"behavior":lambda:inherited.behavior("confirmation"),
     "candidates":lambda:inherited.candidates("confirmation"),"score":score}[args.stage]()


if __name__=="__main__":main()
