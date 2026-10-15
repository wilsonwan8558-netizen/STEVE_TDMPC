"""Fit only the G12 force head on frozen Generic-SOFA latent transitions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

from g12_force_head import ForceHead

RL = Path(__file__).resolve().parents[2]
OUT = RL / "logs/world_model_force_aware_planning_g12"
CHECKPOINT = RL / "runs/generic_solver_tdmpc2_baseline_300k_20261006/checkpoints/step_280000.pt"
EXPECTED_CHECKPOINT_SHA = "476564f8d301ae0c76c418d8e61a4129037b30de7c6872377b563c70e84acd99"
F_LIM = 3.685331040756903
TRAIN_SEED = 20261014
MAX_EPOCHS = 120
PATIENCE = 15
BATCH_SIZE = 512
LR = 0.001


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def new_json(path: Path, data: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main() -> None:
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1 or
            "RTX 5090" not in torch.cuda.get_device_name(0)):
        raise RuntimeError("G12 training requires CUDA_VISIBLE_DEVICES=0 RTX 5090")
    if sha(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("Frozen TD-MPC2 checkpoint changed")
    target = OUT / "force_head.pt"
    if target.exists():
        raise FileExistsError(target)
    manifest = json.loads((OUT / "force_dataset_manifest.json").read_text())
    split = json.loads((OUT / "force_dataset_split.json").read_text())
    if (manifest["status"] != "VALIDATED_EXISTING_GENERIC_SOFA_TRANSITIONS" or
            sha(OUT / "force_transitions.npz") != manifest["joined_dataset_sha256"] or
            split["status"] != "FROZEN_BEFORE_G12_FORCE_HEAD_TRAINING"):
        raise RuntimeError("G12 dataset/split not frozen")
    with np.load(OUT / "force_transitions.npz", allow_pickle=False) as data:
        seeds = data["seed"]
        x = np.concatenate((data["latent"], data["action"]), axis=1).astype(np.float32)
        y = data["force_next_max_N"].astype(np.float32).reshape(-1, 1)
    train = np.isin(seeds, split["train_seeds"])
    validation = np.isin(seeds, split["validation_seeds"])
    heldout = np.isin(seeds, split["heldout_test_seeds"])
    if (not np.all(train | validation | heldout) or np.any(train & validation) or
            np.any(train & heldout) or np.any(validation & heldout)):
        raise RuntimeError("G12 episode split overlaps or misses rows")
    mean = x[train].mean(axis=0).astype(np.float32)
    std = np.maximum(x[train].std(axis=0), 1e-4).astype(np.float32)
    random.seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)
    torch.manual_seed(TRAIN_SEED)
    torch.cuda.manual_seed_all(TRAIN_SEED)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda:0")
    head = ForceHead(torch.from_numpy(mean), torch.from_numpy(std)).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=LR)
    x_train = torch.from_numpy(x[train]).to(device)
    y_train = torch.from_numpy(y[train]).to(device)
    x_val = torch.from_numpy(x[validation]).to(device)
    y_val = torch.from_numpy(y[validation]).to(device)
    history = []
    best_loss = float("inf")
    best_epoch = -1
    best_state = None
    for epoch in range(1, MAX_EPOCHS + 1):
        head.train()
        order = torch.randperm(len(x_train), device=device)
        batch_losses = []
        for indices in order.split(BATCH_SIZE):
            prediction = head(x_train[indices, :256], x_train[indices, 256:])
            loss = F.mse_loss(prediction, y_train[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()) * len(indices))
        head.eval()
        with torch.inference_mode():
            val_loss = float(F.mse_loss(head(x_val[:, :256], x_val[:, 256:]), y_val).cpu())
        train_loss = sum(batch_losses) / len(x_train)
        history.append({"epoch": epoch, "train_mse_N2": train_loss, "validation_mse_N2": val_loss})
        if val_loss < best_loss - 1e-8:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
        if epoch - best_epoch >= PATIENCE:
            break
    if best_state is None:
        raise RuntimeError("G12 force-head training produced no checkpoint")
    torch.save({
        "format_version": 1, "phase": "G12_world_model_force_aware_planning",
        "architecture": "258-64-64-1_ReLU_ReLU_Softplus",
        "target": "true next-interval max-substep distal contact force in nominal simulation N",
        "input": "f_force(z_t,a_t), frozen TD-MPC2 pre-transition planner latent and normalized action",
        "state_dict": best_state, "best_epoch": best_epoch,
        "validation_mse_N2": best_loss,
        "training_seed": TRAIN_SEED,
        "tdmpc2_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "force_dataset_sha256": manifest["joined_dataset_sha256"],
        "split_sha256": sha(OUT / "force_dataset_split.json"),
    }, target)
    new_json(OUT / "force_head_definition.json", {
        "status": "TRAINED_HEAD_ONLY_ON_FROZEN_WORLD_MODEL_LATENTS",
        "architecture": "258-64-64-1, ReLU hidden layers, Softplus scalar output",
        "input": "concat(pre-transition TD-MPC2 planner latent z_t[256], normalized action a_t[2])",
        "output": "nonnegative predicted next-interval maximum distal contact force, nominal simulation N",
        "input_normalization": "featurewise train-episode-only mean/std; std floor 1e-4; stored as checkpoint buffers",
        "target_normalization": "none; direct nominal-N MSE",
        "encoder_and_dynamics": "frozen; no TD-MPC2 parameters updated",
        "optimizer": "Adam lr=0.001, batch=512, maximum 120 epochs, validation MSE early stop patience 15",
        "model_selection": "minimum validation MSE only; held-out test rows unseen during fitting/selection",
        "training_seed": TRAIN_SEED,
        "best_epoch": best_epoch,
        "validation_mse_N2": best_loss,
        "force_head_checkpoint": str(target),
        "force_head_checkpoint_sha256": sha(target),
    })
    new_json(OUT / "force_alignment_definition.json", {
        "status": "TEMPORAL_ALIGNMENT_VERIFIED_FROM_G2_G3_ROWS",
        "planner_rollout": "z_0=encode(observation_t); z_(k+1)=WorldModel.next(z_k,a_k)",
        "head_call": "F_hat_(t+k+1)=ForceHead(z_k,a_k), for the same z_k,a_k used by WorldModel.reward",
        "training_label": "true distal_contact_force_max_N measured during env.step(a_t), maximum across 22 SOFA substeps",
        "next_observation": "the observation returned after that same env.step(a_t)",
        "units": "nominal simulation N",
        "F_lim_N_nominal": F_LIM,
        "not_a_separate_latent_rollout": True,
    })
    new_json(OUT / "force_head_training_history.json", {
        "epochs_run": len(history), "selected_epoch": best_epoch,
        "history": history,
    })
    print("G12_FORCE_HEAD_TRAINED", best_epoch, best_loss, sha(target), flush=True)


if __name__ == "__main__":
    main()
