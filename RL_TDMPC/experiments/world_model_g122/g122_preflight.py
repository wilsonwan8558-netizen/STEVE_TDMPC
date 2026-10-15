"""Validation-only G12.2 gate calibration and frozen candidate availability audit.

This script does not create or step a SOFA environment. Its intermediate rule
and selection rule are fixed in source before validation metrics are computed.
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
from g12_force_head import ForceHead  # noqa: E402
from g12_train import CHECKPOINT, EXPECTED_CHECKPOINT_SHA, F_LIM, OUT as G12, sha  # noqa: E402

G121 = RL / "logs/world_model_force_aware_planning_g121"
OUT = RL / "logs/world_model_force_aware_planning_g122"
INTERMEDIATE_COVERAGE = 0.80
MAX_G2_FALSE_WARNING_FRACTION_OF_G1 = 0.5
MAX_SELECTED_FALSE_SAFE_FRACTION_OF_G0 = 0.5


def save_json(name: str, value: dict) -> None:
    with (OUT / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def save_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(name)
    with (OUT / name).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(variant: str, truth: np.ndarray, raw: np.ndarray, delta: float) -> dict:
    high = truth > F_LIM
    predicted_safe = raw + delta <= F_LIM
    false_safe = int(np.sum(high & predicted_safe))
    false_warning = int(np.sum(~high & ~predicted_safe))
    return {
        "variant": variant, "delta_N": delta, "rows": len(truth),
        "true_high_rows": int(np.sum(high)), "true_safe_rows": int(np.sum(~high)),
        "false_safe_count": false_safe,
        "false_safe_rate": false_safe / int(np.sum(high)),
        "false_warning_count": false_warning,
        "false_warning_rate": false_warning / int(np.sum(~high)),
        "safe_acceptance_rate": float(np.mean(predicted_safe)),
        "above_limit_recall": 1 - false_safe / int(np.sum(high)),
        "below_limit_specificity": 1 - false_warning / int(np.sum(~high)),
    }


def main() -> None:
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1 or
            "RTX 5090" not in torch.cuda.get_device_name(0)):
        raise RuntimeError("G12.2 requires CUDA_VISIBLE_DEVICES=0 RTX 5090")
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite G12.2 output: {OUT}")
    definition = json.loads((G12 / "force_head_definition.json").read_text())
    alignment = json.loads((G12 / "force_alignment_definition.json").read_text())
    manifest = json.loads((G12 / "force_dataset_manifest.json").read_text())
    split = json.loads((G12 / "force_dataset_split.json").read_text())
    g121 = json.loads((G121 / "force_calibration.json").read_text())
    if (sha(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA or
            sha(G12 / "force_head.pt") != definition["force_head_checkpoint_sha256"] or
            sha(G12 / "force_transitions.npz") != manifest["joined_dataset_sha256"] or
            alignment["F_lim_N_nominal"] != F_LIM or
            g121["force_head_checkpoint_sha256"] != definition["force_head_checkpoint_sha256"] or
            g121["F_lim_N_nominal"] != F_LIM):
        raise RuntimeError("Frozen checkpoint, head, dataset, or force reference changed")
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(G12 / "force_transitions.npz", allow_pickle=False) as source:
        mask = np.isin(source["seed"], split["validation_seeds"])
        z = torch.from_numpy(source["latent"][mask]).to("cuda:0")
        a = torch.from_numpy(source["action"][mask]).to("cuda:0")
        truth = source["force_next_max_N"][mask].astype(np.float64)
    head_checkpoint = torch.load(G12 / "force_head.pt", map_location="cpu", weights_only=True)
    state = head_checkpoint["state_dict"]
    head = ForceHead(state["input_mean"], state["input_std"]).to("cuda:0")
    head.load_state_dict(state)
    head.eval().requires_grad_(False)
    with torch.inference_mode():
        raw = head(z, a).squeeze(-1).cpu().numpy().astype(np.float64)
    high = truth > F_LIM
    deficits = np.maximum(F_LIM - raw[high], 0)
    intermediate = float(np.nextafter(np.quantile(deficits, INTERMEDIATE_COVERAGE,
                                                   method="higher"), np.inf))
    full = float(g121["delta_F_N_nominal"])
    if not 0 < intermediate < full:
        raise RuntimeError("Intermediate calibration is not strictly between raw and full")
    deltas = {"G0": 0.0, "G1": full, "G2": intermediate}
    metric_rows = [metrics(k, truth, raw, d) for k, d in deltas.items()]
    by_variant = {r["variant"]: r for r in metric_rows}
    if (by_variant["G2"]["false_safe_count"] <=
            MAX_SELECTED_FALSE_SAFE_FRACTION_OF_G0 * by_variant["G0"]["false_safe_count"] and
            by_variant["G2"]["false_warning_count"] <=
            MAX_G2_FALSE_WARNING_FRACTION_OF_G1 * by_variant["G1"]["false_warning_count"]):
        selected = "G2"
    elif (by_variant["G1"]["false_safe_count"] <=
          MAX_SELECTED_FALSE_SAFE_FRACTION_OF_G0 * by_variant["G0"]["false_safe_count"]):
        selected = "G1"
    else:
        raise RuntimeError("No allowed gate has at least 50% lower validation false-safe count")
    save_json("inherited_g121_audit.json", {
        "status": "FROZEN_IDENTITIES_VERIFIED", "tdmpc2_checkpoint": str(CHECKPOINT),
        "tdmpc2_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "force_head_checkpoint": str(G12 / "force_head.pt"),
        "force_head_checkpoint_sha256": definition["force_head_checkpoint_sha256"],
        "force_dataset_sha256": manifest["joined_dataset_sha256"],
        "g121_calibration_sha256": sha(G121 / "force_calibration.json"),
        "g12_final_candidate_audit_sha256": sha(G12 / "planner_candidate_diagnostics_expanded_final.csv"),
    })
    save_json("frozen_force_head.json", {
        "definition": definition, "alignment": alignment,
        "checkpoint_sha256": definition["force_head_checkpoint_sha256"],
        "input": "same pre-transition z_t[256] and candidate first action a_t[2] as TD-MPC2 reward",
        "normalization": "frozen G12 train-only mean/std buffers, floor 1e-4",
        "output": "next-RL-interval maximum distal force in nominal simulation N",
        "H_force": 1,
    })
    save_json("metadata.json", {
        "phase": "G12.2", "status": "VALIDATION_CALIBRATION_AND_CANDIDATE_PREFLIGHT_ONLY",
        "branch": "feature/world-model-force-aware-planning", "F_gate_N": F_LIM,
        "H_force": 1, "cuda_visible_devices": "0", "cuda_index": 0,
        "cuda_name": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "python": sys.executable,
        "no_SOFA_rollout_in_this_script": True,
    })
    save_json("gate_calibration_candidates.json", {
        "validation_only": True,
        "G0": {"definition": "raw", "delta_N": 0},
        "G1": {"definition": "raw plus frozen G12.1 full calibration", "delta_N": full},
        "G2": {"definition": "raw plus 80th percentile (higher interpolation) of max(F_lim - raw, 0) among validation true-high rows, nextafter upward", "delta_N": intermediate},
        "predeclared_selection_rule": "Select G2 if validation false-safe count is <= 50% of G0 and false-warning count <= 50% of G1; else G1 if its false-safe count is <= 50% of G0; otherwise stop. Navigation outcomes excluded.",
    })
    save_csv("gate_calibration_metrics.csv", metric_rows)
    save_json("selected_gate_calibration.json", {
        "selected_variant": selected, "delta_N": deltas[selected],
        "F_gate_N": F_LIM, "H_force": 1,
        "selection_source": "G12 validation transitions only",
        "selection_rule": "predeclared in g122_preflight.py before metrics computation",
        "selected_validation_metrics": by_variant[selected],
    })
    candidates: dict[str, list[dict]] = {}
    with (G12 / "planner_candidate_diagnostics_expanded_final.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            candidates.setdefault(row["state_id"], []).append(row)
    if len(candidates) != 16 or any(len(rows) != 512 for rows in candidates.values()):
        raise RuntimeError("Frozen final-iteration representative candidate sets changed")
    audit_rows = []
    class_rows = []
    for state_id, rows in candidates.items():
        ids = [int(r["candidate_index"]) for r in rows]
        if ids != list(range(512)):
            raise RuntimeError(f"Candidate identity/order ambiguous for {state_id}")
        scores = np.asarray([float(r["J_TDMPC2"]) for r in rows])
        forces = np.asarray([float(r["F_hat_1_N"]) for r in rows])
        order = np.argsort(-scores, kind="stable")
        best = int(order[0])
        for variant, delta in deltas.items():
            safe = forces + delta <= F_LIM
            count = int(safe.sum())
            if safe[best]:
                label = "GATE-INACTIVE"
                chosen = best
            elif count:
                label = "GATE-ACTIVE"
                chosen = int(order[np.flatnonzero(safe[order])[0]])
            else:
                label = "GATE-EMPTY"
                chosen = None
            rank = int(np.flatnonzero(order == chosen)[0]) + 1 if chosen is not None else None
            loss = float(scores[best] - scores[chosen]) if chosen is not None else None
            row = {
                "state_id": state_id, "seed": int(rows[0]["seed"]),
                "decision_index": int(rows[0]["decision_index"]),
                "force_state": rows[0]["force_state"], "variant": variant,
                "delta_N": delta, "total_candidate_count": 512,
                "predicted_safe_candidate_count": count,
                "predicted_safe_candidate_fraction": count / 512,
                "at_least_one_survives": bool(count), "gate_class": label,
                "unconstrained_best_candidate_index": best,
                "unconstrained_best_J_task": float(scores[best]),
                "unconstrained_best_F_hat_raw_N": float(forces[best]),
                "best_safe_candidate_index": chosen,
                "best_safe_candidate_J_task": float(scores[chosen]) if chosen is not None else None,
                "best_safe_candidate_F_hat_raw_N": float(forces[chosen]) if chosen is not None else None,
                "best_safe_task_score_rank": rank,
                "best_safe_task_score_loss": loss,
            }
            audit_rows.append(row)
            class_rows.append({k: row[k] for k in (
                "state_id", "variant", "gate_class", "predicted_safe_candidate_fraction",
                "unconstrained_best_candidate_index", "best_safe_candidate_index",
                "best_safe_task_score_rank", "best_safe_task_score_loss")})
    save_csv("candidate_availability_audit.csv", audit_rows)
    save_csv("planner_state_gate_classes.csv", class_rows)
    summary = {variant: {label: sum(r["gate_class"] == label for r in audit_rows
                                    if r["variant"] == variant)
                         for label in ("GATE-INACTIVE", "GATE-ACTIVE", "GATE-EMPTY")}
               for variant in deltas}
    print("G122_PREFLIGHT", json.dumps({"selected": selected, "metrics": metric_rows,
                                          "gate_classes": summary}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
