"""Freeze class-balanced validation parent states and exact candidate actions.

Selection is first three seed-ordered states in each class from the already
saved boundary audit. This script does not create a SOFA environment.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
sys.path.insert(0, str(RL / "experiments/world_model_g12"))
sys.path.insert(0, str(RL / "experiments/world_model_g121"))
from g12_train import OUT as G12, F_LIM  # noqa: E402
from g121_planner import load_agent  # noqa: E402
from tdmpc2.common import set_seed  # noqa: E402

OUT = RL / "logs/world_model_force_aware_planning_g122"
CLASSES = ("GATE-INACTIVE", "GATE-ACTIVE", "GATE-EMPTY")


def main() -> None:
    selected = json.loads((OUT / "selected_gate_calibration.json").read_text())
    delta = float(selected["delta_N"])
    with (OUT / "candidate_availability_boundary_validation.csv").open(newline="", encoding="utf-8") as stream:
        boundary = list(csv.DictReader(stream))
    chosen = []
    for label in CLASSES:
        group = sorted((r for r in boundary if r["gate_class"] == label),
                       key=lambda r: (int(r["seed"]), int(r["decision_index"])))
        if len(group) < 3:
            raise RuntimeError(f"Not enough representative states for {label}")
        chosen.extend(group[:3])
    chosen.sort(key=lambda r: (int(r["seed"]), int(r["decision_index"])))
    with np.load(G12 / "force_transitions.npz", allow_pickle=False) as source:
        data = {name: source[name] for name in source.files}
    agent = load_agent(0.0)
    records = []
    for frozen in chosen:
        seed, target = int(frozen["seed"]), int(frozen["decision_index"])
        indices = np.flatnonzero(data["seed"] == seed)
        set_seed(seed)
        agent.lambda_F = 0.0
        agent.trace_enabled = False
        with torch.inference_mode():
            for step in range(target):
                index = int(indices[step])
                action = agent.act(data["observation"][index], first_step=step == 0,
                                   eval_mode=True)
                if not np.array_equal(action, data["action"][index]):
                    raise RuntimeError(f"Archived baseline action mismatch {seed}/{step}")
            index = int(indices[target])
            agent.trace_enabled = True
            action = agent.act(data["observation"][index], first_step=target == 0,
                               eval_mode=True)
            if not np.array_equal(action, data["action"][index]):
                raise RuntimeError(f"Target archived baseline action mismatch {seed}/{target}")
            trace = agent.last_scored
            scores = trace["J_task"].squeeze(-1).cpu().numpy()
            raw_force = trace["F_hat_cal_N"].squeeze(-1).cpu().numpy() - agent.delta_F
            first_actions = trace["actions"][0].cpu().numpy()
        order = np.argsort(-scores, kind="stable")
        top = int(order[0])
        safe = raw_force + delta <= F_LIM
        safe_top = int(order[np.flatnonzero(safe[order])[0]]) if safe.any() else None
        min_force = int(np.argmin(raw_force))
        label = "GATE-INACTIVE" if safe[top] else "GATE-ACTIVE" if safe.any() else "GATE-EMPTY"
        if (label != frozen["gate_class"] or int(safe.sum()) != int(frozen["predicted_safe_candidate_count"]) or
                top != int(frozen["baseline_best_candidate_index"]) or
                abs(float(scores[top]) - float(frozen["baseline_best_candidate_J_task"])) > 1e-6):
            raise RuntimeError(f"Candidate identity changed at {seed}/{target}")
        record = {
            "state_id": frozen["state_id"], "seed": seed, "decision_index": target,
            "gate_class": label, "safe_candidate_count": int(safe.sum()),
            "baseline_applied_mean_action": action.tolist(),
            "baseline_best_candidate_index": top,
            "baseline_best_candidate_action": first_actions[top].tolist(),
            "baseline_best_candidate_J_task": float(scores[top]),
            "baseline_best_candidate_raw_force_N": float(raw_force[top]),
            "best_safe_candidate_index": safe_top,
            "best_safe_candidate_action": first_actions[safe_top].tolist() if safe_top is not None else None,
            "best_safe_candidate_J_task": float(scores[safe_top]) if safe_top is not None else None,
            "best_safe_candidate_raw_force_N": float(raw_force[safe_top]) if safe_top is not None else None,
            "min_force_candidate_index": min_force,
            "min_force_candidate_action": first_actions[min_force].tolist(),
            "min_force_candidate_J_task": float(scores[min_force]),
            "min_force_candidate_raw_force_N": float(raw_force[min_force]),
            "best_safe_task_loss": float(scores[top] - scores[safe_top]) if safe_top is not None else None,
            "candidate_vs_applied_L2": float(np.linalg.norm(first_actions[top] - action)),
        }
        records.append(record)
        print("G122_FORK_FROZEN", record["state_id"], label, flush=True)
    with (OUT / "fork_selection_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump({"selection_rule": "first three seed-ordered validation states in each gate class from candidate_availability_boundary_validation.csv",
                   "selected_variant": selected["selected_variant"], "delta_N": delta,
                   "F_gate_N": F_LIM, "H_force": 1,
                   "baseline_action_semantics": "archived TD-MPC2 final weighted MPPI mean",
                   "gated_action_semantics": "argmax J_task among predicted-safe final-iteration candidates; top candidate if inactive",
                   "F0_empty_semantics": "unconstrained top-scoring sampled candidate",
                   "F1_empty_semantics": "sampled candidate with minimum one-step predicted force",
                   "states": records}, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
