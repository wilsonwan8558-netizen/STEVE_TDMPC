"""Three-replica quantile and raw Q95 diagnostics on frozen development candidates."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
RL=HERE.parents[1]
for name in ("generic_solver_g6b","generic_solver_g5","generic_solver_g4"):
    sys.path.insert(0,str(RL/"experiments"/name))
from g7_prepare_train import OUT,SEEDS,audit,cuda_gate
from g7_metrics import predict,quantile_rows,boundary_row
from g6b_prepare_train import candidate_matrix
from g6b_features import FrozenVesselGeometry
from g5_data import source
from g4_data import candidate_states


def csv_new(path,rows):
    if path.exists():raise FileExistsError(path)
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    cuda_gate();q,definition,old_manifest,freeze,settings,data,norm=audit()
    if not (OUT/"model_manifest.json").exists():raise RuntimeError("Model training incomplete")
    arrays,split,index,physical=source()
    states=candidate_states("development_test",split)
    x,y,contact=candidate_matrix(states,arrays,index,physical,FrozenVesselGeometry())
    if x.shape!=(2520,308):raise RuntimeError("G6B frozen development candidates changed")
    quantiles=[];boundaries=[]
    for seed in SEEDS:
        p=predict(seed,x)
        quantiles.extend(quantile_rows(y,p,seed,"development_test"))
        boundaries.append(boundary_row(y,p,contact,seed,"development_test",0.,q["q95"],q["q99"]))
    csv_new(OUT/"development_quantile_metrics.csv",quantiles)
    csv_new(OUT/"development_boundary_metrics.csv",boundaries)
    print("G7 development",[(r["training_seed"],r["Q95_false_safe_rate"],r["candidate_acceptance_rate"],r["zero_accepted_states"])
        for r in boundaries],flush=True)


if __name__=="__main__":main()
