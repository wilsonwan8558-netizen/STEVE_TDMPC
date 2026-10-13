"""Apply unchanged G6A exact-event worst-replica cutoff rule to new G6B labels."""

from __future__ import annotations

import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/"experiments/generic_solver_g6a"))
import g6_metrics
import g6_calibrate
from g6b_prepare_train import OUT,SEEDS
from g6b_collect import verify_g6b


def main():
    g6_metrics.OUT=OUT
    g6_metrics.SEEDS=SEEDS
    g6_calibrate.OUT=OUT
    g6_calibrate.SEEDS=SEEDS
    g6_calibrate.verify_source=verify_g6b
    g6_calibrate.main()


if __name__=="__main__":main()
