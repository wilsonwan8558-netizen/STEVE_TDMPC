"""Exact CUDA archived-state audit of frozen final-elite MPPI weight masking."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
sys.path.insert(0, str(RL))
sys.path.insert(0, str(RL / "experiments/world_model_g12"))
from g12_train import OUT as G12, F_LIM  # noqa: E402
from tdmpc2.common import set_seed  # noqa: E402
from g123_planner import OUT, G122, load_agent  # noqa: E402


def save_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(name)
    with (OUT / name).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    with np.load(G12 / "force_transitions.npz", allow_pickle=False) as source:
        data = {name: source[name] for name in source.files}
    selected = {}
    with (G12 / "planner_candidate_diagnostics_expanded_final.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (int(row["seed"]), int(row["decision_index"]))
            selected.setdefault(key, set()).add("G12_expanded_final")
    with (G122 / "candidate_availability_boundary_validation.csv").open(newline="", encoding="utf-8") as stream:
        boundary = {(int(r["seed"]), int(r["decision_index"])): r for r in csv.DictReader(stream)}
    for key in boundary:
        selected.setdefault(key, set()).add("G12.2_boundary_validation")
    if len(selected) < 40:
        raise RuntimeError("Archived-state audit lacks representative states")
    agent = load_agent()
    rows = []
    for seed, target in sorted(selected):
        indices = np.flatnonzero(data["seed"] == seed)
        set_seed(seed)
        agent.mask_enabled = False
        agent.trace_enabled = False
        with torch.inference_mode():
            for step in range(target):
                index = int(indices[step])
                action = agent.act(data["observation"][index], first_step=step == 0,
                                   eval_mode=True)
                if not np.array_equal(action, data["action"][index]):
                    raise RuntimeError(f"G12.3 archived CUDA replay mismatch {seed}/{step}")
            index = int(indices[target])
            agent.mask_enabled = True
            action = agent.act(data["observation"][index], first_step=target == 0,
                               eval_mode=True)
        gate = agent.last_gate
        baseline = np.asarray(gate["baseline_action"], dtype=np.float32)
        if not np.array_equal(baseline, data["action"][index]):
            raise RuntimeError(f"G12.3 target baseline action mismatch {seed}/{target}")
        if not np.array_equal(action, np.asarray(gate["gated_action"], dtype=np.float32)):
            raise RuntimeError(f"G12.3 agent return/gate log mismatch {seed}/{target}")
        if (seed, target) in boundary:
            old = boundary[(seed, target)]
            if (int(old["predicted_safe_candidate_count"]) != gate["safe_candidate_count"] or
                    old["gate_class"] == "GATE-EMPTY" and gate["safe_candidate_count"] != 0):
                raise RuntimeError(f"G12.2/G12.3 frozen candidate mismatch {seed}/{target}")
        difference = action - baseline
        all_safe = gate["safe_candidate_count"] == gate["total_candidate_count"]
        if all_safe and not np.array_equal(action, baseline):
            raise RuntimeError(f"G12.3 FORCE-INACTIVE invariance failed {seed}/{target}")
        if gate["safe_candidate_count"] == 0 and not np.array_equal(action, baseline):
            raise RuntimeError(f"G12.3 GATE-EMPTY baseline fallback failed {seed}/{target}")
        record = {
            "state_id": f"{seed}_{target}", "seed": seed, "decision_index": target,
            "source": "+".join(sorted(selected[(seed, target)])),
            "gate_class": gate["gate_class"],
            "safe_candidate_count": gate["safe_candidate_count"],
            "safe_candidate_fraction": gate["safe_candidate_fraction"],
            "retained_weight_mass": gate["retained_weight_mass"],
            "baseline_ESS": gate["baseline_ESS"], "gated_ESS": gate["gated_ESS"],
            "baseline_action_json": json.dumps(baseline.tolist()),
            "gated_action_json": json.dumps(action.tolist()),
            "action_L2_difference": float(np.linalg.norm(difference)),
            "translation_difference": float(difference[0]),
            "rotation_difference": float(difference[1]),
            "baseline_predicted_calibrated_force_N": gate["baseline_predicted_calibrated_force_N"],
            "gated_predicted_calibrated_force_N": gate["gated_predicted_calibrated_force_N"],
            "baseline_archived_true_next_force_N": float(data["force_next_max_N"][index]),
            "baseline_true_above_F_lim": bool(data["force_next_max_N"][index] > F_LIM),
            "masked_elite_count": gate["masked_elite_count"],
            "masked_weighting_applied": gate["masked_weighting_applied"],
            "baseline_fallback": gate["baseline_fallback"],
            "all_candidates_safe": all_safe,
            "baseline_action_exact_archived": True,
        }
        rows.append(record)
        print("G123_ARCHIVED_STATE", record["state_id"], gate["gate_class"],
              round(gate["retained_weight_mass"], 5),
              round(record["action_L2_difference"], 5), flush=True)
    save_csv("archived_state_masking_diagnostics.csv", rows)
    save_csv("retained_weight_mass_metrics.csv", [{
        "state_id": r["state_id"], "gate_class": r["gate_class"],
        "safe_candidate_fraction": r["safe_candidate_fraction"],
        "retained_weight_mass": r["retained_weight_mass"],
        "masked_elite_count": r["masked_elite_count"],
    } for r in rows])
    save_csv("effective_sample_size_metrics.csv", [{
        "state_id": r["state_id"], "gate_class": r["gate_class"],
        "retained_weight_mass": r["retained_weight_mass"],
        "baseline_ESS": r["baseline_ESS"], "gated_ESS": r["gated_ESS"],
        "ESS_ratio": r["gated_ESS"] / r["baseline_ESS"],
    } for r in rows])
    save_csv("gate_inactive_invariance.csv", [{
        "state_id": r["state_id"], "all_candidates_safe": r["all_candidates_safe"],
        "baseline_action_json": r["baseline_action_json"],
        "gated_action_json": r["gated_action_json"],
        "action_L2_difference": r["action_L2_difference"],
        "exact_match": r["action_L2_difference"] == 0,
    } for r in rows if r["all_candidates_safe"]])
    print("G123_ARCHIVED_AUDIT_COMPLETE", len(rows), flush=True)


if __name__ == "__main__":
    main()
