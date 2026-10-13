"""Train matched G3 M0/M1 scalar regressors, selecting with validation only."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

from prepare import OUT, sha

MODEL_SEED = 260206
MAX_EPOCHS = 60
BATCH_SIZE = 512
PATIENCE = 8
MIN_DELTA = 1e-4
LEARNING_RATE = 1e-3


def model() -> nn.Module:
    return nn.Sequential(nn.Linear(258, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))


def write_new(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")


def behavior(split_name: str, split: dict):
    with np.load(OUT / "behavior_features.npz") as features:
        mask = np.isin(features["seed"], split[split_name])
        x = np.concatenate([features["latent"][mask], features["action"][mask]], axis=1).astype(np.float64)
        y = features["force_next_max_N"][mask].astype(np.float64)
        seed = features["seed"][mask]
    if len(x) == 0 or x.shape[1] != 258 or not set(seed.tolist()) == set(split[split_name]):
        raise RuntimeError(f"Invalid {split_name} behavior selection")
    return x, y


def candidate_train(split):
    x, y = [], []
    paths = sorted((OUT / "development_candidates/train").glob("state_*.json"))
    if len(paths) != 560:
        raise RuntimeError(f"Expected 560 train states, got {len(paths)}")
    for path in paths:
        state = json.loads(path.read_text())
        if state["group"] != "train" or state["seed"] not in split["train"] or len(state["candidates"]) != 21:
            raise RuntimeError(f"Counterfactual train state invalid {path}")
        latent = np.asarray(state["latent"], dtype=np.float64)
        for candidate in state["candidates"]:
            x.append(np.concatenate([latent, candidate["action"]]))
            y.append(candidate["force_next_max_N"])
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.shape != (11760, 258) or y.shape != (11760,) or not np.all(np.isfinite(y)) or np.any(y < 0):
        raise RuntimeError("Incomplete training candidate array")
    return x, y


def fit(name, x_train, y_train, x_val, y_val, mean, scale, samples_per_round, device):
    folder = OUT / "models" / name
    folder.mkdir(parents=True, exist_ok=False)
    xtr = torch.as_tensor((x_train-mean)/scale, dtype=torch.float32, device=device)
    ytr = torch.as_tensor(y_train[:, None], dtype=torch.float32, device=device)
    xva = torch.as_tensor((x_val-mean)/scale, dtype=torch.float32, device=device)
    yva = torch.as_tensor(y_val[:, None], dtype=torch.float32, device=device)
    torch.manual_seed(MODEL_SEED)
    network = model().to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    generator = torch.Generator(device="cpu").manual_seed(MODEL_SEED)
    best_loss, best_epoch, stale = math.inf, None, 0
    best_state = None
    history = []
    start = time.monotonic()
    for epoch in range(1, MAX_EPOCHS+1):
        network.train()
        # Matched compute/data-exposure budget: both models see exactly the
        # M0 behavior-training row count per validation round. M1 draws a
        # fresh without-replacement subset from its larger joint pool.
        permutation = torch.randperm(len(xtr), generator=generator)[:samples_per_round]
        for offset in range(0, len(permutation), BATCH_SIZE):
            index = permutation[offset:offset+BATCH_SIZE].to(device)
            loss = torch.mean((network(xtr[index])-ytr[index])**2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        network.eval()
        with torch.inference_mode():
            train_mse = float(torch.mean((network(xtr)-ytr)**2).item())
            val_mse = float(torch.mean((network(xva)-yva)**2).item())
        history.append({"epoch": epoch, "train_MSE_N2": train_mse, "validation_behavior_MSE_N2": val_mse})
        if val_mse < best_loss-MIN_DELTA:
            best_loss, best_epoch, stale = val_mse, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if best_state is None:
        raise RuntimeError("No validation-selected checkpoint")
    torch.save(best_state, folder / "model.pt")
    write_new(folder / "normalization.json", {"source": "train behavior only; shared exactly between M0 and M1",
                                               "input_mean": mean.tolist(), "input_std": scale.tolist(), "target_normalization": "none"})
    write_new(folder / "configuration.json", {"architecture": "258-64-ReLU-64-ReLU-1", "loss": "MSE on raw nominal N",
                                               "optimizer": "Adam", "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE,
                                               "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "minimum_improvement": MIN_DELTA,
                                               "samples_per_validation_round": samples_per_round,
                                               "optimizer_steps_per_validation_round": math.ceil(samples_per_round/BATCH_SIZE),
                                               "maximum_optimizer_steps": MAX_EPOCHS*math.ceil(samples_per_round/BATCH_SIZE),
                                               "validation_selection": "lowest validation behavior MSE only", "model_seed": MODEL_SEED,
                                               "selected_epoch": best_epoch, "selected_validation_MSE_N2": best_loss,
                                               "train_samples": len(y_train), "validation_behavior_samples": len(y_val),
                                               "train_wall_seconds": time.monotonic()-start, "checkpoint_sha256": sha(folder / "model.pt")})
    write_new(folder / "training_history.json", history)
    print(f"{name}: selected epoch {best_epoch} val MSE {best_loss:.6f}", flush=True)


def main():
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or "RTX 5090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Frozen CUDA device unavailable")
    status = json.loads((OUT / "development_collection_status.json").read_text())
    if status["status"] != "COMPLETE_ALL_800_STATES_AND_G2_PARENT_REPRODUCED":
        raise RuntimeError("G3 development counterfactual collection incomplete")
    split = json.loads((OUT / "development_split.json").read_text())
    xtr, ytr = behavior("train", split)
    xva, yva = behavior("validation", split)
    xcf, ycf = candidate_train(split)
    mean = np.mean(xtr, axis=0)
    scale = np.std(xtr, axis=0)
    scale[scale < 1e-8] = 1.0
    device = torch.device("cuda:0")
    fit("M0_behavior", xtr, ytr, xva, yva, mean, scale, len(ytr), device)
    fit("M1_counterfactual", np.concatenate([xtr, xcf]), np.concatenate([ytr, ycf]), xva, yva, mean, scale, len(ytr), device)
    write_new(OUT / "training_manifest.json", {"behavior_train_samples": len(ytr), "behavior_validation_samples": len(yva),
                                                   "counterfactual_train_states": 560, "counterfactual_train_labels": len(ycf),
                                                   "normalization": "train behavior only; identical M0/M1", "validation_metric": "behavior MSE",
                                                   "matched_budget": "same behavior-train-sized samples and same optimizer-step count per validation round",
                                                   "development_test_opened_before_training": False})


if __name__ == "__main__":
    main()
