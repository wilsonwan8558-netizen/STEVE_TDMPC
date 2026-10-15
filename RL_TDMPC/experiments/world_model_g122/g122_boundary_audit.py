"""Validation-only expansion of candidate availability near the selected gate.

No SOFA scene is created. Each frozen archived trajectory is replayed through
the exact CUDA baseline planner, including prior MPPI means and RNG draws.
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


def main() -> None:
    selected = json.loads((OUT / "selected_gate_calibration.json").read_text())
    delta = float(selected["delta_N"])
    split = json.loads((G12 / "force_dataset_split.json").read_text())
    with np.load(G12 / "force_transitions.npz", allow_pickle=False) as source:
        data = {name: source[name] for name in source.files}
    agent = load_agent(0.0)
    with torch.inference_mode():
        raw = agent.force_head(torch.from_numpy(data["latent"]).to("cuda:0"),
                               torch.from_numpy(data["action"]).to("cuda:0"))
        raw = raw.squeeze(-1).cpu().numpy()
    target_raw = F_LIM - delta
    chosen = []
    for seed in split["validation_seeds"]:
        indices = np.flatnonzero(data["seed"] == seed)
        positions = np.argsort(np.abs(raw[indices] - target_raw), kind="stable")
        position = int(positions[0])
        chosen.append((abs(float(raw[indices[position]] - target_raw)), int(seed), position,
                       float(raw[indices[position]])))
    chosen.sort()
    rows = []
    for distance, seed, target, archived_raw in chosen:
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
            forces = (trace["F_hat_cal_N"].squeeze(-1).cpu().numpy() - agent.delta_F)
            candidates = trace["actions"][0].cpu().numpy()
        order = np.argsort(-scores, kind="stable")
        best = int(order[0])
        safe = forces + delta <= F_LIM
        if safe[best]:
            label = "GATE-INACTIVE"
            chosen_index = best
        elif bool(np.any(safe)):
            label = "GATE-ACTIVE"
            chosen_index = int(order[np.flatnonzero(safe[order])[0]])
        else:
            label = "GATE-EMPTY"
            chosen_index = None
        row = {
            "state_id": f"{seed}_{target}", "seed": seed, "decision_index": target,
            "archived_action_predicted_raw_force_N": archived_raw,
            "distance_to_gate_raw_N": distance,
            "variant": selected["selected_variant"], "delta_N": delta,
            "total_candidate_count": len(scores),
            "predicted_safe_candidate_count": int(safe.sum()),
            "predicted_safe_candidate_fraction": float(safe.mean()),
            "gate_class": label, "baseline_best_candidate_index": best,
            "baseline_best_candidate_J_task": float(scores[best]),
            "baseline_best_candidate_force_raw_N": float(forces[best]),
            "best_safe_candidate_index": chosen_index,
            "best_safe_candidate_J_task": float(scores[chosen_index]) if chosen_index is not None else None,
            "best_safe_candidate_force_raw_N": float(forces[chosen_index]) if chosen_index is not None else None,
            "best_safe_candidate_rank": int(np.flatnonzero(order == chosen_index)[0]) + 1 if chosen_index is not None else None,
            "best_safe_candidate_task_loss": float(scores[best] - scores[chosen_index]) if chosen_index is not None else None,
            "baseline_applied_action_json": json.dumps(action.tolist()),
            "best_candidate_first_action_json": json.dumps(candidates[best].tolist()),
            "best_safe_first_action_json": json.dumps(candidates[chosen_index].tolist()) if chosen_index is not None else None,
            "best_candidate_vs_applied_L2": float(np.linalg.norm(candidates[best] - action)),
        }
        rows.append(row)
        print("G122_BOUNDARY_STATE", row["state_id"], label, int(safe.sum()), flush=True)
    with (OUT / "candidate_availability_boundary_validation.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
