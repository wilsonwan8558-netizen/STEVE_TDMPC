"""Freeze a validation-only additive calibration for the inherited G12 head."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "world_model_g12"))
from g12_force_head import ForceHead  # noqa: E402
from g12_train import CHECKPOINT, EXPECTED_CHECKPOINT_SHA, F_LIM, OUT as G12, sha  # noqa: E402

OUT = HERE.parents[1] / "logs/world_model_force_aware_planning_g121"
TARGET_COVERAGE = 0.95


def save_json(name: str, data: dict) -> None:
    path = OUT / name
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def metrics(split: str, label: str, truth: np.ndarray, prediction: np.ndarray) -> dict:
    high = truth > F_LIM
    safe = ~high
    missed = (prediction <= F_LIM) & high
    warned = (prediction > F_LIM) & safe
    return {
        "split": split, "prediction": label, "rows": len(truth),
        "above_limit_rows": int(high.sum()), "safe_rows": int(safe.sum()),
        "MAE_N": float(np.mean(np.abs(prediction - truth))),
        "RMSE_N": float(np.sqrt(np.mean(np.square(prediction - truth)))),
        "false_safe_count": int(missed.sum()),
        "false_safe_rate_given_true_high": float(missed.sum() / high.sum()) if high.any() else None,
        "false_warning_count": int(warned.sum()),
        "false_warning_rate_given_true_safe": float(warned.sum() / safe.sum()) if safe.any() else None,
        "coverage_above_F_lim": float(1 - missed.sum() / high.sum()) if high.any() else None,
    }


def main() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or "RTX 5090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("G12.1 calibration requires CUDA_VISIBLE_DEVICES=0 RTX 5090")
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError("G12.1 output directory is not empty")
    OUT.mkdir(parents=True, exist_ok=True)
    definition = json.loads((G12 / "force_head_definition.json").read_text())
    alignment = json.loads((G12 / "force_alignment_definition.json").read_text())
    manifest = json.loads((G12 / "force_dataset_manifest.json").read_text())
    split = json.loads((G12 / "force_dataset_split.json").read_text())
    if (sha(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA or
            sha(G12 / "force_head.pt") != definition["force_head_checkpoint_sha256"] or
            sha(G12 / "force_transitions.npz") != manifest["joined_dataset_sha256"] or
            alignment["F_lim_N_nominal"] != F_LIM or
            alignment["head_call"] != "F_hat_(t+k+1)=ForceHead(z_k,a_k), for the same z_k,a_k used by WorldModel.reward"):
        raise RuntimeError("Inherited G12 identities or force alignment changed")
    with np.load(G12 / "force_transitions.npz", allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    validation = np.isin(data["seed"], split["validation_seeds"])
    heldout = np.isin(data["seed"], split["heldout_test_seeds"])
    if validation.sum() == 0 or heldout.sum() == 0 or np.any(validation & heldout):
        raise RuntimeError("Validation and held-out partitions missing or overlapping")
    checkpoint = torch.load(G12 / "force_head.pt", map_location="cpu", weights_only=True)
    state = checkpoint["state_dict"]
    head = ForceHead(state["input_mean"], state["input_std"]).to("cuda:0")
    head.load_state_dict(state)
    head.eval().requires_grad_(False)
    output = {}
    with torch.inference_mode():
        for label, mask in (("validation", validation), ("heldout_test", heldout)):
            latent = torch.from_numpy(data["latent"][mask]).to("cuda:0")
            action = torch.from_numpy(data["action"][mask]).to("cuda:0")
            output[label] = {
                "truth": data["force_next_max_N"][mask].astype(np.float64),
                "raw": head(latent, action).squeeze(-1).cpu().numpy().astype(np.float64),
            }
    val = output["validation"]
    high = val["truth"] > F_LIM
    if high.sum() < 20:
        raise RuntimeError("Insufficient validation above-limit examples for additive calibration")
    deficits = np.maximum(F_LIM - val["raw"][high], 0.0)
    threshold = float(np.quantile(deficits, TARGET_COVERAGE, method="higher"))
    delta = float(np.nextafter(threshold, np.inf)) if threshold else 0.0
    rows = []
    for label, item in output.items():
        rows.append(metrics(label, "raw", item["truth"], item["raw"]))
        rows.append(metrics(label, "calibrated", item["truth"], item["raw"] + delta))
    with (OUT / "force_calibration_metrics.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json("inherited_g12_audit.json", {
        "status": "FROZEN_G12_IDENTITIES_VERIFIED",
        "tdmpc2_checkpoint": str(CHECKPOINT),
        "tdmpc2_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "force_head_checkpoint": str(G12 / "force_head.pt"),
        "force_head_checkpoint_sha256": definition["force_head_checkpoint_sha256"],
        "force_dataset_sha256": manifest["joined_dataset_sha256"],
        "force_alignment": alignment,
        "planner_audit": str(G12 / "tdmpc2_planner_audit.json"),
        "heldout_G12_one_step_MAE_N": 0.454824273763933,
        "heldout_G12_above_limit_false_safe": "45/105",
    })
    save_json("frozen_force_head.json", {
        "status": "FROZEN_NO_RETRAINING",
        "definition": definition,
        "checkpoint_sha256": definition["force_head_checkpoint_sha256"],
        "checkpoint_best_epoch": int(checkpoint["best_epoch"]),
        "input_alignment": "pre-transition planner z_t[256], first candidate action a_t[2]",
        "input_normalization": "frozen training-set featurewise mean/std checkpoint buffers, floor 1e-4",
        "raw_output": "nonnegative next-interval max distal force, nominal simulation N",
        "calibrated_output": "raw force + validation-only delta_F, nominal simulation N",
    })
    save_json("force_calibration.json", {
        "status": "FROZEN_VALIDATION_ONLY_BEFORE_G121_PLANNING",
        "selection_split": "G12 validation episodes only; G12 heldout_test used only for post-freeze evaluation",
        "selection_rule": "95th percentile (higher interpolation) of max(F_lim - raw_prediction,0) over validation true-above-limit rows; next representable float upward for strict >F_lim comparison",
        "target_validation_coverage": TARGET_COVERAGE,
        "F_lim_N_nominal": F_LIM,
        "delta_F_N_nominal": delta,
        "validation_above_limit_rows": int(high.sum()),
        "raw_validation_false_safe": rows[0]["false_safe_count"],
        "calibrated_validation_false_safe": rows[1]["false_safe_count"],
        "force_head_checkpoint_sha256": definition["force_head_checkpoint_sha256"],
        "metrics_file": "force_calibration_metrics.csv",
    })
    save_json("metadata.json", {
        "phase": "G12.1 calibrated one-step force-aware world-model planning",
        "branch": "feature/world-model-force-aware-planning",
        "status": "CALIBRATION_FROZEN_OBJECTIVE_SCALE_PENDING",
        "frozen_simulation": "GenericConstraintSolver, projected Gauss-Seidel, tolerance 1e-4, maxIterations 2000, computeConstraintForces True, mu 0.001, dt 0.006 s",
        "F_lim_N_nominal": F_LIM,
        "F_lim_interpretation": "simulation statistical reference, not clinical injury threshold",
        "H_force": 1,
        "task_planner_horizon": 3,
        "tdmpc2_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "force_head_checkpoint_sha256": definition["force_head_checkpoint_sha256"],
        "CUDA_VISIBLE_DEVICES": "0",
        "CUDA_device_index": 0,
        "CUDA_device_name": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": sys.executable,
    })
    print("G121_VALIDATION_CALIBRATION_FROZEN", delta, rows, flush=True)


if __name__ == "__main__":
    main()
