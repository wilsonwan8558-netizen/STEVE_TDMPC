"""Check whether a passive SOFA Controller can sample each 0.006 s substep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import Sofa

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from envs.steve_env import StEVEEnv  # noqa: E402


class Counter(Sofa.Core.Controller):
    def __init__(self):
        super().__init__(name="g05_passive_substep_counter")
        self.count = 0

    def onAnimateEndEvent(self, _event):
        self.count += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    env = StEVEEnv()
    try:
        env.reset(seed=90008)
        counter = Counter()
        env._simulation.root.addObject(counter)
        env.step(np.asarray([0.5, 0.5], dtype=np.float32))
        data = {"sofa_substeps_observed_for_one_RL_action": counter.count,
                "expected_from_simulation_step": 22,
                "sofa_dt_s": float(env._simulation.root.dt.value),
                "pass": counter.count == 22}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(data, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
