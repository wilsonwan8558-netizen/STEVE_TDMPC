"""Synthetic pre-freeze smoke test of confirmation aggregation; no new seeds."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

import g102_confirm_analysis as analysis
from g102_eval_common import search_points, key
from g102_prepare import G101


def run():
    source = json.loads((G101 / "heldout_state_curves/state_0003.json").read_text())
    v_nom = source["v_nom"]
    vs = search_points(v_nom)
    by_v = {key(r["v"]): r for r in source["true_curve"]}
    if not all(key(v) in by_v for v in vs):
        raise RuntimeError("G10.2 synthetic confirmation fixture missing G10.1 truth")
    root = Path(tempfile.mkdtemp(prefix="g102_confirmation_analysis_smoke_"))
    (root / "confirmation_state_curves").mkdir()
    (root / "pre_confirmation_freeze.json").write_text("{}\n")
    selected = []
    for seed in range(108000, 108030):
        for decision in (10, 20):
            source_hash = f"SYNTHETIC_{seed}_{decision}"
            selected.append({"seed": seed, "decision_index": decision,
                "episode_sha256": source_hash, "selection_slot":
                    "quarter_progress" if decision == 10 else "maximum_prior_executed_force"})
            q = np.zeros((len(vs), 3, 3), dtype=np.float64)
            item = {"seed": seed, "decision_index": decision,
                "source_episode_sha256": source_hash,
                "selection_slot": selected[-1]["selection_slot"],
                "v_nom": v_nom, "w_nom_fixed": source["w_nom_fixed"],
                "tested_v": vs.tolist(), "predicted_delta_quantiles_replicas_N": q.tolist(),
                "nominal_conservative_Q95_N": 3.,
                "predicted_absolute_R2_N": [3.]*len(vs),
                "predicted_uncertainty_constraint_pass": [True]*len(vs),
                "predicted_safe": [True]*len(vs),
                "true_curve": [by_v[key(v)] for v in vs]}
            path = root / "confirmation_state_curves" / f"seed_{seed}_decision_{decision:04d}.json"
            path.write_text(json.dumps(item))
    (root / "confirmation_state_selection.json").write_text(json.dumps({
        "ordered_seeds": list(range(108000, 108030)), "selected_states": selected}))
    (root / "confirmation_parent_replay_audit.json").write_text(json.dumps({
        "status": "PASS_EXACT_PASSIVE_PARENT_AND_FORK_REPLAY", "states": 60,
        "max_abs_parent_difference": {"observation": 0.},
        "pre_confirmation_freeze_sha256": analysis.sha(root / "pre_confirmation_freeze.json"),
        "true_SOFA_two_action_forks": 60*len(vs), "one_interval_censored_forks": 0}))
    analysis.OUT = root
    analysis.freeze_gate = lambda: {}
    analysis.analyze()
    result = json.loads((root / "confirmation_summary.json").read_text())
    if result["states"] != 60 or len(list(root.glob("confirmation_*_metrics.csv"))) != 2:
        raise RuntimeError("G10.2 confirmation aggregation smoke test failed")
    print("PASS synthetic confirmation aggregation; no 108000 seed or new SOFA scene opened")


if __name__ == "__main__":
    run()
