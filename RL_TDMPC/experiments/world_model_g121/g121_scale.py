"""Scale a one-step calibrated force cost on frozen development planner candidates."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
G12 = RL / "logs/world_model_force_aware_planning_g12"
OUT = RL / "logs/world_model_force_aware_planning_g121"


def save_json(name: str, data: dict) -> None:
    with (OUT / name).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def load_candidates(path: Path, delta: float, f_lim: float) -> dict[str, dict]:
    groups = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = row["state_id"]
            group = groups.setdefault(key, {"force_state": row["force_state"],
                                            "seed": int(row["seed"]), "decision_index": int(row["decision_index"]),
                                            "task": [], "raw": [], "candidate_index": [], "first_action": []})
            group["task"].append(float(row["J_TDMPC2"]))
            group["raw"].append(float(row["F_hat_1_N"]))
            group["candidate_index"].append(int(row["candidate_index"]))
            group["first_action"].append(json.loads(row["action_sequence_json"])[0])
    for group in groups.values():
        group["task"] = np.asarray(group["task"], dtype=np.float64)
        group["raw"] = np.asarray(group["raw"], dtype=np.float64)
        group["calibrated"] = group["raw"] + delta
        group["cost"] = np.square(np.maximum(group["calibrated"] - f_lim, 0.0))
        group["first_action"] = np.asarray(group["first_action"], dtype=np.float64)
        if len(group["task"]) != 512 or group["candidate_index"] != list(range(512)):
            raise RuntimeError("G12 frozen planner candidate count/order changed")
    return groups


def spread(x: np.ndarray) -> float:
    return float(np.quantile(x, .90) - np.quantile(x, .10))


def main() -> None:
    calibration = json.loads((OUT / "force_calibration.json").read_text())
    inherited = json.loads((OUT / "inherited_g12_audit.json").read_text())
    delta = float(calibration["delta_F_N_nominal"])
    f_lim = float(calibration["F_lim_N_nominal"])
    if delta < 0 or inherited["force_head_checkpoint_sha256"] != calibration["force_head_checkpoint_sha256"]:
        raise RuntimeError("Frozen G12.1 calibration identity changed")
    source = G12 / "planner_candidate_diagnostics_expanded_final.csv"
    groups = load_candidates(source, delta, f_lim)
    if len(groups) != 16 or len({g["seed"] for g in groups.values()}) < 4:
        raise RuntimeError("Frozen G12 expanded validation planner-state cohort changed")
    task_spreads = [spread(g["task"]) for g in groups.values()]
    active = [g for g in groups.values() if np.any(g["cost"] > 0)]
    force_spreads = [spread(g["cost"]) for g in active]
    s_task = float(np.median(task_spreads))
    s_force = float(np.median(force_spreads)) if force_spreads else 0.0
    # Float64 analysis of float32 model outputs: a spread smaller than the
    # corresponding float32 rounding scale is not a reliable denominator.
    rounding_floor = float(np.finfo(np.float32).eps * max(1.0, np.median([
        np.median(g["cost"]) for g in active]) if active else 1.0))
    stable = bool(np.isfinite(s_task) and np.isfinite(s_force) and s_task > 0 and s_force > rounding_floor)
    lam_base = s_task / s_force if stable else None
    weights = [0.0, .1*lam_base, .3*lam_base, lam_base] if stable else []
    save_json("lambda_scale_definition.json", {
        "status": "DEFINED_FROM_DEVELOPMENT_PLANNER_STATES" if stable else "STOP_NUMERICALLY_DEGENERATE",
        "state_source": str(source),
        "state_source_semantics": "16 G12 expanded validation states, used as development-only planner states; no held-out/test/confirmation state or label used for lambda scale",
        "candidate_set": "512 original frozen TD-MPC2 candidates per state at final of six CUDA MPPI iterations; no force-aware planner run",
        "J_task": "exact original _estimate_value scalar; larger is better; reward plus terminal Q, verified by G12 audit",
        "H_force": 1,
        "cost": "c_force=max(F_hat_raw_first+delta_F-F_lim,0)^2, nominal N^2",
        "delta_F_N_nominal": delta,
        "F_lim_N_nominal": f_lim,
        "S_task_definition": "median across all development states of Q90(J_task)-Q10(J_task)",
        "S_force_definition": "median across force-active development states of Q90(c_force)-Q10(c_force)",
        "S_task": s_task,
        "S_force": s_force,
        "rounding_floor_for_S_force": rounding_floor,
        "force_active_state_count": len(active),
        "force_active_state_spreads": force_spreads,
        "lambda_base": lam_base,
        "fixed_lambda_set": weights,
        "lambda_set_status": "PROPOSED_UNRUN" if stable else "NOT_DEFINED",
    })
    print("G121_SCALE", "STABLE" if stable else "STOP", "S_task", s_task,
          "S_force", s_force, "floor", rounding_floor, "active", len(active),
          "lambda_base", lam_base, flush=True)
    if not stable:
        return
    summaries = []
    classifications = []
    selections = []
    for state_id, g in groups.items():
        task = g["task"]
        cost = g["cost"]
        cspread = spread(cost)
        is_active = bool(np.any(cost > 0))
        # Nonzero Q90-Q10 is a meaningful candidate-set variation at the
        # scale used to define lambda_base, not merely a single outlier.
        is_discriminative = bool(cspread > rounding_floor)
        classification = "FORCE_DISCRIMINATIVE" if is_discriminative else ("FORCE_ACTIVE" if is_active else "FORCE_INACTIVE")
        classifications.append({"state_id": state_id, "seed": g["seed"],
                                "decision_index": g["decision_index"], "previous_force_category": g["force_state"],
                                "classification": classification, "nonzero_candidate_fraction": float(np.mean(cost > 0)),
                                "J_task_Q90_minus_Q10": spread(task), "c_force_Q90_minus_Q10": cspread})
        original_index = int(np.argmax(task))
        for weight in weights:
            total = task - weight * cost
            selected = int(np.argmax(total))
            selections.append({"state_id": state_id, "lambda_F": weight,
                               "baseline_candidate_index": original_index, "selected_candidate_index": selected,
                               "candidate_argmax_changed": selected != original_index,
                               "baseline_J_task": float(task[original_index]), "selected_J_task": float(task[selected]),
                               "J_task_difference_selected_minus_baseline": float(task[selected]-task[original_index]),
                               "baseline_F_hat_cal_N": float(g["calibrated"][original_index]),
                               "selected_F_hat_cal_N": float(g["calibrated"][selected]),
                               "F_hat_cal_difference_selected_minus_baseline_N": float(g["calibrated"][selected]-g["calibrated"][original_index]),
                               "baseline_first_action": g["first_action"][original_index].tolist(),
                               "selected_first_action": g["first_action"][selected].tolist(),
                               "first_action_L2_difference": float(np.linalg.norm(g["first_action"][selected]-g["first_action"][original_index]))})
        summaries.append({"state_id": state_id, "classification": classification,
                          "J_task_Q90_minus_Q10": spread(task),
                          "c_force_Q90_minus_Q10": cspread,
                          "nonzero_candidate_fraction": float(np.mean(cost > 0)),
                          "weighted_cost_spread_by_lambda": {str(w): w*cspread for w in weights}})
    for name, rows in (("planner_state_classification.csv", classifications),
                       ("planner_candidate_argmax_offline.csv", selections)):
        with (OUT / name).open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    save_json("objective_scale_reaudit.json", {
        "status": "OFFLINE_ORIGINAL_CANDIDATE_REAUDIT_NOT_FORCE_AWARE_MPPI",
        "S_task": s_task, "S_force": s_force, "lambda_base": lam_base,
        "lambda_F_values": weights,
        "states": len(groups), "candidates_per_state": 512,
        "force_inactive_states": sum(c["classification"] == "FORCE_INACTIVE" for c in classifications),
        "force_active_states": sum(c["classification"] != "FORCE_INACTIVE" for c in classifications),
        "force_discriminative_states": sum(c["classification"] == "FORCE_DISCRIMINATIVE" for c in classifications),
        "state_summaries": summaries,
        "selection_semantics": "same-candidate argmax only, not a full MPPI rerun nor the final optimized-mean first action",
    })


if __name__ == "__main__":
    main()
