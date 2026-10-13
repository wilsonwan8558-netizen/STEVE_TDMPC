"""Three fixed M0/M1 G9-family E1/H2 quantile replicas; validation only."""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
import torch

from g95_common import OUT, G9, SEEDS, H, TAUS, cuda_gate, json_new, sha
from g9_risk import QuantileNet, pinball, predict, data as g9_data, matrix as g9_matrix


def load_training():
    d, estimates, split = g9_data()
    bx, by = g9_matrix(d, estimates["E1"], 2)
    with np.load(OUT / "development_planner_dataset.npz") as z:
        p = {key: z[key].copy() for key in z.files}
    if p["x"].shape[1] != 298 or bx.shape[1] != 298:
        raise RuntimeError("G9.5 E1/H2 risk representation changed")
    masks = {group: np.isin(d["seed"], split[group]) for group in ("train", "validation", "development_test")}
    p_masks = {group: p["group_index"] == index for index, group in enumerate(("train", "validation", "development_test"))}
    if any(not np.any(masks[group]) or not np.any(p_masks[group]) for group in masks):
        raise RuntimeError("G9.5 behavior/planner development split empty")
    norm = json.loads((G9 / "models/risk/E1/H2/normalization.json").read_text())
    mean, scale = np.asarray(norm["mean"], dtype=np.float64), np.asarray(norm["std"], dtype=np.float64)
    if mean.shape != (298,) or scale.shape != (298,) or np.any(scale <= 0):
        raise RuntimeError("G9.5 G9 input normalization changed")
    return bx, by, masks, p, p_masks, mean, scale


def train_one(kind, seed, bx, by, bm, p, pm, mean, scale):
    folder = OUT / "models" / kind
    checkpoint_path = folder / f"seed_{seed}.pt"
    selection_path = folder / f"seed_{seed}_selection.json"
    if checkpoint_path.exists() or selection_path.exists():
        raise FileExistsError(f"G9.5 model seed already trained: {kind}/{seed}")
    if kind == "M0_behavior":
        x_train, y_train = bx[bm["train"]], by[bm["train"]]
        x_val, y_val = bx[bm["validation"]], by[bm["validation"]]
    elif kind == "M1_planner_conditioned":
        x_train = np.concatenate((bx[bm["train"]], p["x"][pm["train"]].astype(np.float64)))
        y_train = np.concatenate((by[bm["train"]], p["R2_true_N"][pm["train"]].astype(np.float64)))
        x_val = np.concatenate((bx[bm["validation"]], p["x"][pm["validation"]].astype(np.float64)))
        y_val = np.concatenate((by[bm["validation"]], p["R2_true_N"][pm["validation"]].astype(np.float64)))
    else:
        raise ValueError(kind)
    if (not np.all(np.isfinite(x_train)) or not np.all(np.isfinite(y_train)) or
            not np.all(np.isfinite(x_val)) or not np.all(np.isfinite(y_val)) or
            np.any(y_train < 0) or np.any(y_val < 0)):
        raise RuntimeError("G9.5 nonfinite/negative training or validation data")
    tx = torch.as_tensor((x_train-mean)/scale, dtype=torch.float32, device="cuda:0")
    ty = torch.as_tensor(y_train, dtype=torch.float32, device="cuda:0")
    vx = torch.as_tensor((x_val-mean)/scale, dtype=torch.float32, device="cuda:0")
    vy = torch.as_tensor(y_val, dtype=torch.float32, device="cuda:0")
    torch.manual_seed(seed)
    net = QuantileNet(298).to("cuda:0")
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best = math.inf
    patience = 0
    history = []
    for epoch in range(60):
        net.train()
        order = torch.randperm(len(tx), generator=generator).to("cuda:0")
        for batch in order.split(512):
            loss = pinball(net(tx[batch]), ty[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        net.eval()
        with torch.inference_mode():
            value = float(pinball(net(vx), vy).cpu())
        history.append({"epoch": epoch, "validation_pinball_N": value})
        if value < best-1e-5:
            best = value
            patience = 0
            torch.save(net.cpu().state_dict(), checkpoint_path)
            net.to("cuda:0")
        else:
            patience += 1
            if patience >= 8:
                break
    net.load_state_dict(torch.load(checkpoint_path, map_location="cuda:0", weights_only=True))
    val_pred = predict(net, x_val, mean, scale)
    residual = y_val-val_pred[:, 2]
    delta = max(0.0, float(np.quantile(residual, .95, method="higher")))
    json_new(selection_path, {"model": kind, "training_seed": seed,
        "train_rows": len(x_train), "validation_rows": len(x_val),
        "behavior_train_rows": int(np.sum(bm["train"])),
        "planner_train_rows": 0 if kind == "M0_behavior" else int(np.sum(pm["train"])),
        "behavior_validation_rows": int(np.sum(bm["validation"])),
        "planner_validation_rows": 0 if kind == "M0_behavior" else int(np.sum(pm["validation"])),
        "best_validation_pinball_N": best, "history": history,
        "validation_only_Q95_delta_N": delta,
        "validation_raw_Q95_coverage": float(np.mean(y_val <= val_pred[:, 2])),
        "validation_corrected_Q95_coverage": float(np.mean(y_val <= val_pred[:, 2]+delta)),
        "normalization": "exact frozen G9 E1/H2 behavior-TRAIN input mean/std",
        "checkpoint_sha256": sha(checkpoint_path)})
    print(f"G9.5 trained {kind}/{seed}: train={len(x_train)} val={len(x_val)} pinball={best:.5f} delta={delta:.5f}", flush=True)


def train():
    cuda_gate()
    audit = json.loads((OUT / "development_parent_replay_audit.json").read_text())
    if audit["status"] != "PASS" or audit["planner_candidate_forks"] != 7200:
        raise RuntimeError("G9.5 planner development labels not validated")
    if (OUT / "model_training_manifest.json").exists():
        raise FileExistsError(OUT / "model_training_manifest.json")
    bx, by, bm, p, pm, mean, scale = load_training()
    for kind in ("M0_behavior", "M1_planner_conditioned"):
        for seed in SEEDS:
            train_one(kind, seed, bx, by, bm, p, pm, mean, scale)
    models = {}
    for kind in ("M0_behavior", "M1_planner_conditioned"):
        models[kind] = {}
        for seed in SEEDS:
            path = OUT / "models" / kind / f"seed_{seed}.pt"
            item = json.loads((OUT / "models" / kind / f"seed_{seed}_selection.json").read_text())
            models[kind][str(seed)] = {"checkpoint_sha256": sha(path),
                "selection_sha256": sha(OUT / "models" / kind / f"seed_{seed}_selection.json"),
                "validation_Q95_delta_N": item["validation_only_Q95_delta_N"],
                "best_validation_pinball_N": item["best_validation_pinball_N"]}
    old = torch.load(G9 / "models/risk/E1/H2/model.pt", map_location="cpu", weights_only=True)
    first = torch.load(OUT / "models/M0_behavior" / f"seed_{SEEDS[0]}.pt", map_location="cpu", weights_only=True)
    max_abs = max(float(torch.max(torch.abs(old[name]-first[name]))) for name in old)
    json_new(OUT / "model_training_manifest.json", {"status": "THREE_FIXED_REPLICAS_PER_MODEL_BEFORE_DEVELOPMENT_TEST",
        "training_seeds": SEEDS, "models": models,
        "G9_exact_old_model_vs_M0_first_replica_max_abs_weight_difference": max_abs,
        "G9_original_model_sha256": sha(G9 / "models/risk/E1/H2/model.pt"),
        "G9_risk_normalization_sha256": sha(G9 / "models/risk/E1/H2/normalization.json"),
        "G9_behavior_temporal_sha256": sha(G9 / "development_temporal_behavior.npz"),
        "G95_planner_dataset_sha256": sha(OUT / "development_planner_dataset.npz"),
        "training_code_sha256": sha(__file__),
        "architecture_loss_optimizer_budget_identical_to_G9_E1_H2": True,
        "only_supervision_distribution_differs": True,
        "M1_validation_distribution": "behavior validation plus planner-state validation; same pinball selection and higher-order Q95 residual rule",
        "development_test_or_confirmation_used_for_training_selection": False})
    print(f"G9.5 six model replicas trained; old G9/M0 first max weight difference {max_abs:.6g}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("train",), required=True)
    if parser.parse_args().stage == "train":
        train()


if __name__ == "__main__":
    main()
