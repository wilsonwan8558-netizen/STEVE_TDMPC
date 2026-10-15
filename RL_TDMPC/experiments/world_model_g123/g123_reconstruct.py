"""Audit and independently reproduce the frozen final MPPI weighting path.

No planner implementation is changed or SOFA scene created here.
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
from g12_train import CHECKPOINT, EXPECTED_CHECKPOINT_SHA, F_LIM, OUT as G12, sha  # noqa: E402

G121 = RL / "logs/world_model_force_aware_planning_g121"
G122 = RL / "logs/world_model_force_aware_planning_g122"
OUT = RL / "logs/world_model_force_aware_planning_g123"
ATOL = 1e-6


def save_json(name: str, data: dict) -> None:
    with (OUT / name).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main() -> None:
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1 or
            "RTX 5090" not in torch.cuda.get_device_name(0)):
        raise RuntimeError("G12.3 reconstruction requires CUDA_VISIBLE_DEVICES=0 RTX 5090")
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite G12.3 output: {OUT}")
    definition = json.loads((G12 / "force_head_definition.json").read_text())
    alignment = json.loads((G12 / "force_alignment_definition.json").read_text())
    g122 = json.loads((G122 / "selected_gate_calibration.json").read_text())
    if (sha(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA or
            sha(G12 / "force_head.pt") != definition["force_head_checkpoint_sha256"] or
            g122["selected_variant"] != "G1" or
            g122["delta_N"] != json.loads((G121 / "force_calibration.json").read_text())["delta_F_N_nominal"] or
            g122["F_gate_N"] != F_LIM or g122["H_force"] != 1 or
            alignment["F_lim_N_nominal"] != F_LIM):
        raise RuntimeError("Frozen G12/G12.1/G12.2 identity or G1 calibration changed")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    planning = checkpoint["config"]["planning"]
    if (int(planning["num_samples"]) != 512 or int(planning["num_elites"]) != 64 or
            int(planning["iterations"]) != 6 or float(planning["temperature"]) != 0.5):
        raise RuntimeError("Frozen MPPI configuration changed")
    OUT.mkdir(parents=True, exist_ok=True)
    save_json("inherited_g122_audit.json", {
        "status": "FROZEN_IDENTITIES_VERIFIED", "tdmpc2_checkpoint": str(CHECKPOINT),
        "tdmpc2_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "force_head_checkpoint": str(G12 / "force_head.pt"),
        "force_head_checkpoint_sha256": definition["force_head_checkpoint_sha256"],
        "g122_selected_gate_calibration_sha256": sha(G122 / "selected_gate_calibration.json"),
        "g122_fork_gate_decision_sha256": sha(G122 / "plan_fork_gate_decision.json"),
        "G1_delta_N": g122["delta_N"], "F_lim_N": F_LIM, "H_force": 1,
    })
    save_json("frozen_force_head.json", {
        "definition": definition, "alignment": alignment,
        "checkpoint_sha256": definition["force_head_checkpoint_sha256"],
        "input": "pre-transition z_t[256] and candidate first action a_t[2]",
        "normalization": "G12 train-only checkpoint mean/std buffers, floor 1e-4",
        "output": "next-RL-interval maximum distal force, nominal simulation N",
        "G1_delta_N": g122["delta_N"], "F_lim_N": F_LIM, "H_force": 1,
    })
    save_json("metadata.json", {
        "phase": "G12.3", "branch": "feature/world-model-force-aware-planning",
        "status": "ORIGINAL_MPPI_WEIGHTING_AUDITED_BASELINE_REPRODUCTION_PENDING",
        "CUDA_VISIBLE_DEVICES": "0", "CUDA_device_index": 0,
        "CUDA_device_name": torch.cuda.get_device_name(0),
        "python": sys.executable, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "baseline_reproduction_tolerance": {"rtol": 0, "atol": ATOL},
    })
    save_json("mppi_weighting_audit.json", {
        "status": "SOURCE_AND_CHECKPOINT_AUDITED_BEFORE_FORCE_MASK_IMPLEMENTATION",
        "source": "RL_TDMPC/tdmpc2/agent.py TDMPC2Agent._plan, lines 158-241 in frozen working tree",
        "source_sha256": sha(RL / "tdmpc2/agent.py"),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "candidate_sequences_per_iteration": 512,
        "candidate_generation": "24 policy-prior sequences plus 488 Gaussian-sampled/clipped sequences; six MPPI iterations; horizon three",
        "candidate_task_score": "nan_to_num(_estimate_value(z_t repeated 512 times, candidate action sequence), nan=0.0); original reward/termination/terminal-Q objective, unchanged",
        "elite_selection": "torch.topk(values.squeeze(-1), 64) at every iteration; only these 64 have nonzero original MPPI weight",
        "temperature": 0.5,
        "unnormalized_elite_weight": "u_i=exp(0.5*(J_task_i-max_elite_J_task))",
        "normalized_elite_weight": "w_i=u_i/(sum_j u_j + 1e-9); nonelite weight zero",
        "weighted_mean_action_sequence": "mean=(sum_i w_i*a_i)/(sum_i w_i+1e-9), separately over each horizon step and action dimension",
        "variance_update": "std=clamp(sqrt(sum_i w_i*(a_i-mean)^2/(sum_i w_i+1e-9)), min_std=0.05, max_std=2.0)",
        "warm_start": "mean[0:H-1] initialized from previous_mean[1:H] except at first step; previous_mean becomes final weighted mean",
        "final_first_action": "eval_mode uses clamp(final mean[0], -1, 1), then act returns float32 numpy; no best-candidate selection",
        "force_mask_scope_G123": "final-iteration elite weight masking only, after frozen candidate generation/task scoring and MPPI updates; previous_mean remains original task-weighted final mean so candidate sampling and warm-start are unchanged",
        "baseline_action_reproduction_source": "G12 archived final-iteration candidate actions and task scores versus G12.1 lambda=0 exact archived applied action",
    })
    by_state = {}
    with (G12 / "planner_candidate_diagnostics_expanded_final.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            by_state.setdefault(row["state_id"], []).append(row)
    archived = {}
    with (G121 / "planner_selection_diagnostics.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if float(row["lambda_F"]) == 0:
                archived[row["state_id"]] = row
    if len(by_state) != 16 or set(by_state) != set(archived):
        raise RuntimeError("Archived 16-state MPPI reconstruction source changed")
    records = []
    for state_id, rows in by_state.items():
        if [int(row["candidate_index"]) for row in rows] != list(range(512)):
            raise RuntimeError(f"Candidate identity/order ambiguous: {state_id}")
        values = torch.tensor([float(row["J_TDMPC2"]) for row in rows],
                              dtype=torch.float32, device="cuda:0")
        actions = torch.tensor([json.loads(row["action_sequence_json"]) for row in rows],
                               dtype=torch.float32, device="cuda:0").permute(1, 0, 2)
        elite_indices = torch.topk(values, 64).indices
        elite_values = values[elite_indices].unsqueeze(-1)
        elite_actions = actions[:, elite_indices]
        u = torch.exp(0.5 * (elite_values - elite_values.max(dim=0).values))
        w = u / (u.sum(dim=0, keepdim=True) + 1e-9)
        mean = (w.unsqueeze(0) * elite_actions).sum(dim=1) / (w.sum(dim=0) + 1e-9)
        reproduced = mean[0].clamp(-1, 1).cpu().numpy().astype(np.float32)
        expected = np.asarray(json.loads(archived[state_id]["selected_first_action"]), dtype=np.float32)
        error = np.abs(reproduced - expected)
        records.append({
            "state_id": state_id, "seed": int(rows[0]["seed"]),
            "decision_index": int(rows[0]["decision_index"]),
            "reconstructed_action_json": json.dumps(reproduced.tolist()),
            "archived_applied_action_json": json.dumps(expected.tolist()),
            "translation_abs_error": float(error[0]),
            "rotation_abs_error": float(error[1]),
            "max_abs_error": float(error.max()),
            "passes_atol_1e_6_rtol_0": bool(np.all(error <= ATOL)),
            "top64_weight_sum": float(w.sum().item()),
            "top64_baseline_ESS": float(1.0 / w.square().sum().item()),
        })
    with (OUT / "baseline_action_reproduction.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    max_error = max(r["max_abs_error"] for r in records)
    print("G123_BASELINE_REPRODUCTION", len(records), max_error,
          sum(r["passes_atol_1e_6_rtol_0"] for r in records), flush=True)
    if not all(r["passes_atol_1e_6_rtol_0"] for r in records):
        raise RuntimeError("G12.3 STOP: archived MPPI action reconstruction failed")


if __name__ == "__main__":
    main()
