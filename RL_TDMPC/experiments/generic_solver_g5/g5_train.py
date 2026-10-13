"""Matched-budget G5 training; G4 optimizer and checkpoint rule are reused verbatim."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

from g5_data import OUT, PRIMARY, behavior, candidates, minimal_groups, source
from g5_prepare import G4, SEEDS
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generic_solver_g4"))
import g4_train  # noqa: E402
from g4_data import candidate_states  # noqa: E402


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    g4_train.new_json(path, value)


def norm_for(name, x):
    mean, scale = np.mean(x, axis=0), np.std(x, axis=0)
    scale[scale < 1e-8] = 1.
    norm = {"source": "TRAIN behavior rows d>=1 only, shared exactly between suffix0/1 and three seeds",
            "input_mean": mean.tolist(), "input_std": scale.tolist(), "target_normalization": "none"}
    path = OUT / "models" / name / "normalization.json"
    if path.exists():
        if json.loads(path.read_text()) != norm:
            raise RuntimeError(f"Existing normalization differs: {name}")
    else:
        save_json(path, norm)
    return mean, scale


def check_reference(name):
    reference = {"Z": "Z0", "P": "P0"}[name]
    actual_norm = json.loads((OUT / "models" / name / "normalization.json").read_text())
    frozen_norm = json.loads((G4 / "models" / name / "normalization.json").read_text())
    if actual_norm != frozen_norm:
        raise RuntimeError(f"G4 {name} normalization not reproduced")
    details = []
    for seed in SEEDS:
        p = OUT / "models" / name / f"seed_{seed}"
        q = G4 / "models" / reference / f"seed_{seed}"
        ours, theirs = json.loads((p / "selection.json").read_text()), json.loads((q / "selection.json").read_text())
        if ours["selected_epoch"] != theirs["selected_epoch"] or ours["selected_validation_behavior_MSE_N2"] != theirs["selected_validation_behavior_MSE_N2"]:
            raise RuntimeError(f"G4 {name}/{seed} validation checkpoint not exactly reproduced: {ours} vs {theirs}")
        a, b = torch.load(p / "model.pt", map_location="cpu", weights_only=True), torch.load(q / "model.pt", map_location="cpu", weights_only=True)
        max_diff = max(float(torch.max(torch.abs(a[k] - b[k]))) for k in a)
        if max_diff != 0:
            raise RuntimeError(f"G4 {name}/{seed} weights differ by {max_diff}")
        details.append({"seed": seed, "selected_epoch": ours["selected_epoch"],
                        "validation_MSE_N2": ours["selected_validation_behavior_MSE_N2"], "max_weight_abs_difference": max_diff})
    return details


def train(name, settings, arrays, split, index, physical, device, secondary=False, groups=None):
    view = "Pmin" if name == "Pmin1" else name.rstrip("1") if secondary else name
    xtr, ytr, _ = behavior(view, "train", arrays, split, index, physical, groups)
    xva, yva, _ = behavior(view, "validation", arrays, split, index, physical, groups)
    if (len(xtr), len(xva)) != (10903, 2286):
        raise RuntimeError("G4 behavior counts changed")
    mean, scale = norm_for(view, xtr)
    if secondary:
        states = candidate_states("train", split)
        xcf, ycf, _ = candidates(view, states, arrays, index, physical, groups)
        if len(xcf) != 11760:
            raise RuntimeError("G4 candidate count changed")
        train_x, train_y = np.concatenate((xtr, xcf)), np.concatenate((ytr, ycf))
    else:
        train_x, train_y = xtr, ytr
    for seed in SEEDS:
        folder = OUT / "models" / name / f"seed_{seed}"
        if folder.exists():
            if not (folder / "model.pt").exists() or not (folder / "selection.json").exists():
                raise RuntimeError(f"Incomplete model output: {folder}")
            print(f"{name}/{seed}: already complete", flush=True)
            continue
        g4_train.fit(name, seed, train_x, train_y, xva, yva, mean, scale, len(ytr), device, settings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("references", "primary", "secondary"), required=True)
    stage = parser.parse_args().stage
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available() or torch.cuda.device_count() != 1 or "RTX 5090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Frozen CUDA_VISIBLE_DEVICES=0 RTX 5090 required")
    settings = json.loads((OUT / "training_seed_manifest.json").read_text())
    if settings["training_seeds"] != list(SEEDS) or tuple(settings["primary_model_names"]) != PRIMARY:
        raise RuntimeError("G5 training protocol changed")
    runtime_path = OUT / "training_runtime.json"
    runtime = {"sys_executable": sys.executable, "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
               "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"], "cuda_available": torch.cuda.is_available(),
               "visible_cuda_device_count": torch.cuda.device_count(), "model_device_index": 0, "device_name": torch.cuda.get_device_name(0)}
    if runtime_path.exists():
        if json.loads(runtime_path.read_text()) != runtime:
            raise RuntimeError("CUDA runtime differs from initial G5 run")
    else:
        save_json(runtime_path, runtime)
    arrays, split, index, physical = source()
    g4_train.OUT = OUT
    device = torch.device("cuda:0")
    if stage == "references":
        for name in ("Z", "P"):
            train(name, settings, arrays, split, index, physical, device)
        save_json(OUT / "reference_model_reproduction.json", {"status": "PASS_EXACT_G4_Z0_P0_REPRODUCTION",
            "Z": check_reference("Z"), "P": check_reference("P")})
    elif stage == "primary":
        if not (OUT / "reference_model_reproduction.json").exists():
            raise RuntimeError("Reference model reproduction gate has not passed")
        for name in PRIMARY[2:]:
            train(name, settings, arrays, split, index, physical, device)
        save_json(OUT / "primary_training_complete.json", {"status": "COMPLETE_12_PRIMARY_CELLS_THREE_SEEDS",
            "names": list(PRIMARY), "training_seeds": list(SEEDS), "supervision": "behavior-only"})
    else:
        if not (OUT / "minimal_subset_selection.json").exists():
            raise RuntimeError("Validation-only P_min must be frozen first")
        groups = minimal_groups()
        for name in ("Z1", "P1", "Pmin1"):
            train(name, settings, arrays, split, index, physical, device, secondary=True, groups=groups)
        save_json(OUT / "secondary_training_complete.json", {"status": "COMPLETE_3_SECONDARY_CELLS_THREE_SEEDS",
            "names": ["Z1", "P1", "Pmin1"], "training_seeds": list(SEEDS), "P_min_groups": list(groups)})


if __name__ == "__main__":
    main()
