"""Train eight G4 ablation cells, three fixed seeds each, after P audit pass."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

from g4_data import OUT, REPRESENTATIONS, STATE_DIMS, source, physical_lookup, behavior_matrix, candidate_states, candidate_matrix

SEEDS = (260206, 260207, 260208)


def model(input_dim):
    return nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))


def new_json(path, data):
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def fit(name, seed, xtr, ytr, xva, yva, mean, scale, samples_per_round, device, settings):
    folder = OUT / "models" / name / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=False)
    train_x = torch.as_tensor((xtr-mean)/scale, dtype=torch.float32, device=device)
    train_y = torch.as_tensor(ytr[:, None], dtype=torch.float32, device=device)
    val_x = torch.as_tensor((xva-mean)/scale, dtype=torch.float32, device=device)
    val_y = torch.as_tensor(yva[:, None], dtype=torch.float32, device=device)
    torch.manual_seed(seed)
    net = model(xtr.shape[1]).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best_mse, best_epoch, stale, best_state = math.inf, None, 0, None
    history = []
    start = time.monotonic()
    for epoch in range(1, settings["max_validation_rounds"]+1):
        net.train()
        permutation = torch.randperm(len(train_x), generator=generator)[:samples_per_round]
        for offset in range(0, samples_per_round, settings["batch_size"]):
            indices = permutation[offset:offset+settings["batch_size"]].to(device)
            loss = torch.mean((net(train_x[indices])-train_y[indices])**2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        net.eval()
        with torch.inference_mode():
            val_mse = float(torch.mean((net(val_x)-val_y)**2).item())
        history.append({"epoch": epoch, "validation_behavior_MSE_N2": val_mse})
        if val_mse < best_mse - settings["minimum_validation_MSE_improvement"]:
            best_mse, best_epoch, stale = val_mse, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
        else:
            stale += 1
            if stale >= settings["early_stopping_patience"]:
                break
    torch.save(best_state, folder / "model.pt")
    new_json(folder / "training_history.json", history)
    new_json(folder / "selection.json", {"name": name, "training_seed": seed, "selected_epoch": best_epoch,
                                         "selected_validation_behavior_MSE_N2": best_mse,
                                         "train_rows": len(xtr), "validation_rows": len(xva),
                                         "samples_per_validation_round": samples_per_round,
                                         "optimizer_steps_per_round": math.ceil(samples_per_round/settings["batch_size"]),
                                         "train_wall_seconds": time.monotonic()-start})
    print(f"{name}/{seed}: epoch={best_epoch} val_MSE={best_mse:.6f}", flush=True)


def main():
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available() or
            torch.cuda.device_count() != 1 or "RTX 5090" not in torch.cuda.get_device_name(0)):
        raise RuntimeError("Frozen CUDA_VISIBLE_DEVICES=0 required")
    settings = json.loads((OUT / "training_seed_manifest.json").read_text())
    if settings["training_seeds"] != list(SEEDS):
        raise RuntimeError("Training seed manifest changed")
    arrays, split, index = source()
    physical = physical_lookup()
    new_json(OUT / "training_runtime.json", {"sys_executable": sys.executable, "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda, "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
        "cuda_available": torch.cuda.is_available(), "visible_cuda_device_count": torch.cuda.device_count(),
        "model_device_index": 0, "device_name": torch.cuda.get_device_name(0)})
    states = candidate_states("train", split)
    device = torch.device("cuda:0")
    count_manifest = {}
    for representation in REPRESENTATIONS:
        xtr, ytr, _ = behavior_matrix(representation, "train", arrays, split, index, physical)
        xva, yva, _ = behavior_matrix(representation, "validation", arrays, split, index, physical)
        xcf, ycf = candidate_matrix(representation, states, arrays, index, physical)
        mean, scale = np.mean(xtr, axis=0), np.std(xtr, axis=0)
        scale[scale < 1e-8] = 1.
        if len(xtr) != 10903 or len(xva) != 2286 or len(xcf) != 11760:
            raise RuntimeError(f"Matched row counts unexpected {representation}: {len(xtr)}/{len(xva)}/{len(xcf)}")
        norm_path = OUT / "models" / representation / "normalization.json"
        norm_path.parent.mkdir(parents=True, exist_ok=True)
        new_json(norm_path, {"source": "TRAIN behavior rows d>=1 only, shared exactly between suffix0/1 and three seeds",
                             "input_mean": mean.tolist(), "input_std": scale.tolist(), "target_normalization": "none"})
        count_manifest[representation] = {"train_behavior": len(xtr), "validation_behavior": len(xva),
                                          "train_candidates": len(xcf), "input_dim": STATE_DIMS[representation]+2}
        for supervision in (0, 1):
            name = f"{representation}{supervision}"
            train_x, train_y = (xtr, ytr) if supervision == 0 else (np.concatenate((xtr, xcf)), np.concatenate((ytr, ycf)))
            for seed in SEEDS:
                fit(name, seed, train_x, train_y, xva, yva, mean, scale, len(ytr), device, settings)
    new_json(OUT / "training_complete.json", {"status": "COMPLETE_ALL_24_VALIDATION_SELECTED_CHECKPOINTS",
                                              "cells": [f"{r}{s}" for r in REPRESENTATIONS for s in (0, 1)],
                                              "training_seeds": SEEDS, "counts": count_manifest,
                                              "development_test_opened_before_training": False,
                                              "confirmation_opened_before_training": False})


if __name__ == "__main__":
    main()
