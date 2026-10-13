"""Select the smallest shared additive Q95 residual on new calibration labels only."""

from __future__ import annotations

import csv
import json
import struct
import sys
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
RL=HERE.parents[1]
sys.path.insert(0,str(RL/"experiments/generic_solver_g6a"))
from g6_metrics import metrics,availability,contact_emergence
from g7_prepare_train import OUT,SEEDS,sha,new_json
from g7_collect import verify_g7
from g7_metrics import quantile_rows,boundary_row


def csv_new(path,rows):
    if path.exists():raise FileExistsError(path)
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def load(group):
    path=OUT/f"{group}_candidate_results.csv"
    with path.open(newline="") as stream:rows=list(csv.DictReader(stream))
    if len(rows)!=2520:raise RuntimeError("Candidate count changed")
    state=np.asarray([int(r["state_id"]) for r in rows]);candidate=np.asarray([int(r["candidate_id"]) for r in rows])
    if not np.array_equal(state,np.repeat(np.arange(120),21)) or not np.array_equal(candidate,np.tile(np.arange(21),120)):
        raise RuntimeError("Candidate row order changed")
    y=np.asarray([float(r["F_true_N"]) for r in rows],dtype=np.float64)
    contact=np.asarray([bool(int(r["current_contact"])) for r in rows],dtype=bool)
    p={seed:np.asarray([[float(r[f"{name}_seed_{seed}_N"]) for name in ("Q50","Q90","Q95","Q99")]
        for r in rows],dtype=np.float64) for seed in SEEDS}
    if np.any(~np.isfinite(y)) or any(np.any(~np.isfinite(v)) or np.any(np.diff(v,axis=1)<0) or np.any(v[:,0]<0) for v in p.values()):
        raise RuntimeError("Invalid/crossing quantile predictions")
    return rows,y,contact,p


def float_bits(value):
    return struct.unpack("<Q",struct.pack("<d",float(value)))[0]


def bits_float(bits):
    return struct.unpack("<d",struct.pack("<Q",int(bits)))[0]


def select_delta(y,p,limit):
    unsafe=y>limit
    positive=int(np.sum(unsafe))
    if positive==0:raise RuntimeError("No calibration Q95 positives")
    scores=np.asarray([p[seed][:,2] for seed in SEEDS])
    def evaluate(delta):
        misses=np.sum(unsafe[None,:]&(scores+delta<=limit),axis=1)
        return misses, float(np.max(misses/positive))
    raw_misses,raw_worst=evaluate(0.)
    if raw_worst<=.05+1e-15:
        return 0.,{"method":"raw passed; exact delta=0","raw_misses_by_replica":raw_misses.tolist(),
            "raw_worst_FSR":raw_worst,"binary_search_steps":0,"predecessor_delta_N":None}
    upper=max(1.,float(limit-np.min(scores))+1.)
    if not np.isfinite(upper) or evaluate(upper)[1]>.05:
        raise RuntimeError("Unable to bracket a conservative correction")
    low,high=float_bits(0.),float_bits(upper)
    steps=0
    while high-low>1:
        middle=(low+high)//2
        value=bits_float(middle)
        _,worst=evaluate(value)
        if worst<=.05+1e-15:high=middle
        else:low=middle
        steps+=1
    chosen=bits_float(high);previous=bits_float(high-1)
    misses,worst=evaluate(chosen)
    predecessor_misses,predecessor_worst=evaluate(previous)
    if worst>.05+1e-15 or predecessor_worst<=.05+1e-15:
        raise RuntimeError("Smallest representable nonnegative correction gate failed")
    return chosen,{"method":"monotone binary search over positive IEEE-754 float64 bit patterns; actual Q95+delta<=F_lim arithmetic",
        "raw_misses_by_replica":raw_misses.tolist(),"raw_worst_FSR":raw_worst,
        "binary_search_steps":steps,"bracket_upper_N":upper,"predecessor_delta_N":previous,
        "predecessor_misses_by_replica":predecessor_misses.tolist(),"predecessor_worst_FSR":predecessor_worst,
        "selected_misses_by_replica":misses.tolist(),"selected_worst_FSR":worst}


def main():
    if (OUT/"selected_residual_calibration.json").exists():raise FileExistsError("Already calibrated")
    q=verify_g7();audit=json.loads((OUT/"calibration_scoring_audit.json").read_text())
    if audit["status"]!="PASS_THREE_G7_REPLICAS_SAME_LABELS" or audit["candidate_results_sha256"]!=sha(OUT/"calibration_candidate_results.csv"):
        raise RuntimeError("G7 calibration scoring incomplete")
    rows,y,contact,p=load("calibration")
    if any(int(r["true_Q95_high_force"])!=int(v>q["q95"]) for r,v in zip(rows,y)):
        raise RuntimeError("G2 Q95 truth changed")
    delta,search=select_delta(y,p,q["q95"])
    raw=[boundary_row(y,p[seed],contact,seed,"calibration_raw",0.,q["q95"],q["q99"]) for seed in SEEDS]
    selected=[boundary_row(y,p[seed],contact,seed,"calibration_selected",delta,q["q95"],q["q99"]) for seed in SEEDS]
    if max(row["Q95_false_safe_rate"] for row in selected)>.05+1e-15:
        raise RuntimeError("Selected residual fails empirical worst-replica target")
    csv_new(OUT/"calibration_quantile_metrics.csv",[r for seed in SEEDS for r in quantile_rows(y,p[seed],seed,"calibration")])
    csv_new(OUT/"calibration_boundary_metrics.csv",raw+selected)
    selected_obj={"status":"SMALLEST_SHARED_NONNEGATIVE_RESIDUAL_SELECTED_ON_CALIBRATION_ONLY",
        "F_lim_N":q["q95"],"delta_cal_N":delta,"G6B_global_margin_N":float(json.loads((RL/"logs/generic_solver_phase_g6b_full_contact_predictor/selected_cutoff.json").read_text())["Delta_N"]),
        "decision_rule":"Q95(x,a)+delta_cal <= F_lim, evaluated in float64",
        "selection_rule":"smallest shared nonnegative float64 residual with max of three empirical Q95 FSR <=5%",
        "search_audit":search,"raw_by_replica":{str(r["training_seed"]):r for r in raw},
        "selected_by_replica":{str(r["training_seed"]):r for r in selected},
        "all_replicas_meet_empirical_5pct":True,"confirmation_seeds_opened_before_selection":False,
        "calibration_candidate_results_sha256":sha(OUT/"calibration_candidate_results.csv")}
    new_json(OUT/"selected_residual_calibration.json",selected_obj)
    new_json(OUT/"quantile_calibration_summary.json",{"F_lim_N":q["q95"],"true_Q95_high_force_count":int(np.sum(y>q["q95"])),
        "true_Q99_extreme_count":int(np.sum(y>q["q99"])),"raw_by_replica":selected_obj["raw_by_replica"],
        "selected_by_replica":selected_obj["selected_by_replica"],"quantile_coverage_file":"calibration_quantile_metrics.csv",
        "selected_delta_cal_N":delta,"calibration_label_count":len(y),"state_count":120})
    print(f"G7 calibration delta={delta:.17g}; raw worst FSR={search['raw_worst_FSR']:.4f}; selected worst FSR={max(r['Q95_false_safe_rate'] for r in selected):.4f}",flush=True)


if __name__=="__main__":main()
