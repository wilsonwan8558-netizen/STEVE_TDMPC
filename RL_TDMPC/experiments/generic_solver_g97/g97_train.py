"""Matched scalar-H2 quantile replicas differing only in predictor state information."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
for folder in ("generic_solver_g97", "generic_solver_g96", "generic_solver_g95", "generic_solver_g9"):
    sys.path.insert(0, str(RL / "experiments" / folder))

from g97_prepare import (OUT, G95, GROUPS, SEEDS, PCA_DIM, RAW_DIM,
                         MODEL_INPUT_DIM, F_LIM, source_gate)  # noqa: E402
from g95_common import json_new, sha  # noqa: E402
from g9_risk import QuantileNet, pinball, predict  # noqa: E402

FAMILIES = ("M_DEPLOY", "M_PRIV")


def inputs():
    source_gate()
    audit = json.loads((OUT / "dataset_alignment_audit.json").read_text())
    if (audit["status"] != "PASS_ORIGINAL_7193_G95_LABELS_ACTIONS_AND_800_PRIVILEGED_STATES_ALIGNED" or
            audit["privileged_state_matrix_sha256"] != sha(OUT / "development_privileged_states.npz") or
            audit["original_G95_planner_dataset_sha256"] != sha(G95 / "development_planner_dataset.npz")):
        raise RuntimeError("G9.7 exact development data alignment not frozen")
    with np.load(G95 / "development_planner_dataset.npz") as z:
        planner = {key: z[key].copy() for key in z.files}
    with np.load(OUT / "development_privileged_states.npz") as z:
        states = {key: z[key].copy() for key in z.files}
    if (planner["x"].shape != (7193, 298) or states["raw"].shape != (800, RAW_DIM) or
            not np.array_equal(planner["R2_true_N"],
                               np.maximum(planner["F1_true_N"], planner["F2_true_N"]))):
        raise RuntimeError("G9.7 H2 scalar planner target changed")
    return planner, states


def fit_adapter(states):
    """Unsupervised, label-free train-state PCA; one row per parent state."""
    path = OUT / "privileged_input_adapter.npz"
    manifest_path = OUT / "privileged_input_adapter_manifest.json"
    if path.exists() or manifest_path.exists():
        if not path.exists() or not manifest_path.exists():
            raise RuntimeError("G9.7 partial privileged adapter")
        manifest = json.loads(manifest_path.read_text())
        if (manifest["status"] != "TRAIN_STATES_ONLY_256D_PRIVILEGED_ADAPTER" or
                manifest["adapter_sha256"] != sha(path) or
                manifest["development_privileged_states_sha256"] != sha(OUT / "development_privileged_states.npz")):
            raise RuntimeError("G9.7 privileged adapter source changed")
        with np.load(path) as z:
            return {key: z[key].copy() for key in z.files}
    train = states["raw"][states["group_index"] == 0].astype(np.float64)
    if train.shape != (560, RAW_DIM):
        raise RuntimeError("G9.7 train-state-only adapter input changed")
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    zero_variance = int(np.sum(scale < 1e-6))
    scale[scale < 1e-6] = 1.0
    standardized = ((train-mean)/scale).astype(np.float32)
    tensor = torch.as_tensor(standardized, device="cuda:0")
    # Full thin SVD is deterministic for the fixed input and has no labels.
    _, singular, right = torch.linalg.svd(tensor, full_matrices=False)
    components = right[:PCA_DIM].cpu().numpy().copy()
    singular = singular.cpu().numpy().copy()
    for row in components:
        peak = int(np.argmax(np.abs(row)))
        if row[peak] < 0:
            row *= -1
    variance = singular.astype(np.float64)**2
    explained = float(np.sum(variance[:PCA_DIM])/np.sum(variance))
    np.savez_compressed(path, mean=mean, scale=scale, components=components.astype(np.float32),
                        singular_values=singular.astype(np.float32))
    json_new(manifest_path, {"status": "TRAIN_STATES_ONLY_256D_PRIVILEGED_ADAPTER",
        "development_privileged_states_sha256": sha(OUT / "development_privileged_states.npz"),
        "adapter_sha256": sha(path), "raw_dimension": RAW_DIM,
        "train_parent_states": 560, "PCA_components": PCA_DIM,
        "train_explained_variance_fraction": explained,
        "zero_variance_coordinates_std_fallback_1": zero_variance,
        "training_labels_used": False, "validation_or_development_test_states_used_to_fit": False,
        "sign_convention": "largest-absolute loading in each component is nonnegative"})
    print(f"G9.7 train-only PCA retained {explained:.6f} training variance", flush=True)
    return {"mean": mean, "scale": scale, "components": components.astype(np.float32),
            "singular_values": singular.astype(np.float32)}


def matrix(planner, states, adapter, family):
    raw = states["raw"].astype(np.float64)
    mean, scale = adapter["mean"], adapter["scale"]
    comp = adapter["components"].astype(np.float64)
    if mean.shape != (RAW_DIM,) or scale.shape != (RAW_DIM,) or comp.shape != (PCA_DIM, RAW_DIM):
        raise RuntimeError("G9.7 input adapter dimension changed")
    pca = ((raw-mean)/scale) @ comp.T
    if not np.all(np.isfinite(pca)):
        raise RuntimeError("G9.7 privileged PCA projection nonfinite")
    state_index = planner["state_global_id"]
    exact_deploy = planner["x"][:, :294].astype(np.float64)
    action = planner["x"][:, 294:].astype(np.float64)
    if family == "M_DEPLOY":
        extra = np.zeros((len(action), 38 + PCA_DIM), dtype=np.float64)
    elif family == "M_PRIV":
        extra = np.c_[raw[state_index, :38], pca[state_index]]
    else:
        raise ValueError(family)
    x = np.c_[exact_deploy, extra, action]
    if x.shape != (len(action), MODEL_INPUT_DIM) or not np.all(np.isfinite(x)):
        raise RuntimeError(f"G9.7 {family} model input invalid")
    return x


def normalize(x, train_mask):
    mean = x[train_mask].mean(axis=0)
    scale = x[train_mask].std(axis=0)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def train_one(family, seed, x, y, train_mask, val_mask, mean, scale):
    folder = OUT / "models" / family / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    model_path, selection_path, norm_path = (folder / name for name in
        ("model.pt", "selection.json", "normalization.json"))
    if any(path.exists() for path in (model_path, selection_path, norm_path)):
        if not all(path.exists() for path in (model_path, selection_path, norm_path)):
            raise RuntimeError(f"G9.7 partial existing checkpoint {family}/{seed}")
        previous = json.loads(selection_path.read_text())
        if previous["checkpoint_sha256"] != sha(model_path) or previous["normalization_sha256"] != sha(norm_path):
            raise RuntimeError(f"G9.7 existing checkpoint changed {family}/{seed}")
        return previous
    json_new(norm_path, {"input_mean": mean.tolist(), "input_std": scale.tolist(),
        "train_rows": int(np.sum(train_mask)), "train_only": True,
        "zero_variance_std_fallback": 1.0})
    tx = torch.as_tensor((x[train_mask]-mean)/scale, dtype=torch.float32, device="cuda:0")
    ty = torch.as_tensor(y[train_mask], dtype=torch.float32, device="cuda:0")
    vx = torch.as_tensor((x[val_mask]-mean)/scale, dtype=torch.float32, device="cuda:0")
    vy = torch.as_tensor(y[val_mask], dtype=torch.float32, device="cuda:0")
    torch.manual_seed(seed)
    net = QuantileNet(MODEL_INPUT_DIM).to("cuda:0")
    parameter_count = sum(parameter.numel() for parameter in net.parameters())
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best = math.inf
    patience = 0
    history = []
    for epoch in range(60):
        net.train()
        order = torch.randperm(len(tx), generator=generator).to("cuda:0")
        for chunk in order.split(512):
            loss = pinball(net(tx[chunk]), ty[chunk])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        net.eval()
        with torch.inference_mode():
            value = float(pinball(net(vx), vy).cpu())
        history.append({"epoch": epoch, "validation_pinball_N": value})
        if value < best-1e-5:
            best, patience = value, 0
            torch.save(net.cpu().state_dict(), model_path)
            net.to("cuda:0")
        else:
            patience += 1
            if patience >= 8:
                break
    net.load_state_dict(torch.load(model_path, map_location="cuda:0", weights_only=True))
    prediction = predict(net, x[val_mask], mean, scale)
    delta = max(0.0, float(np.quantile(y[val_mask]-prediction[:, 2], .95, method="higher")))
    item = {"family": family, "training_seed": seed,
        "train_planner_rows": int(np.sum(train_mask)),
        "validation_planner_rows": int(np.sum(val_mask)),
        "best_validation_pinball_N": best,
        "validation_only_Q95_delta_N": delta,
        "validation_raw_Q95_coverage": float(np.mean(y[val_mask] <= prediction[:, 2])),
        "validation_corrected_Q95_coverage": float(np.mean(y[val_mask] <= prediction[:, 2]+delta)),
        "hidden_width": 64, "hidden_depth": 2,
        "parameter_count": parameter_count,
        "history": history,
        "checkpoint_sha256": sha(model_path),
        "normalization_sha256": sha(norm_path)}
    json_new(selection_path, item)
    print(f"G9.7 {family}/{seed} val pinball={best:.6f} delta={delta:.6f}", flush=True)
    return item


def train():
    planner, states = inputs()
    if (OUT / "training_manifest.json").exists():
        raise FileExistsError(OUT / "training_manifest.json")
    adapter = fit_adapter(states)
    y = planner["R2_true_N"].astype(np.float64)
    train_mask, val_mask = (planner["group_index"] == i for i in (0, 1))
    if int(np.sum(train_mask)) != 5038 or int(np.sum(val_mask)) != 1080:
        raise RuntimeError("G9.7 frozen G9.5 planner train/validation rows changed")
    summaries = {}
    shared_base_norm = None
    for family in FAMILIES:
        x = matrix(planner, states, adapter, family)
        mean, scale = normalize(x, train_mask)
        if shared_base_norm is None:
            shared_base_norm = (mean[:294].copy(), scale[:294].copy(), mean[-4:].copy(), scale[-4:].copy())
        elif (not np.array_equal(shared_base_norm[0], mean[:294]) or
              not np.array_equal(shared_base_norm[1], scale[:294]) or
              not np.array_equal(shared_base_norm[2], mean[-4:]) or
              not np.array_equal(shared_base_norm[3], scale[-4:])):
            raise RuntimeError("G9.7 deployable/action train normalization differs across families")
        summaries[family] = {str(seed): train_one(family, seed, x, y, train_mask,
                                                 val_mask, mean, scale)
                             for seed in SEEDS}
    if len({item["parameter_count"] for group in summaries.values() for item in group.values()}) != 1:
        raise RuntimeError("G9.7 model parameter capacities differ")
    json_new(OUT / "training_manifest.json", {
        "status": "SIX_FIXED_MATCHED_CAPACITY_REPLICAS_SELECTED_ON_VALIDATION_ONLY",
        "families": list(FAMILIES), "training_seeds": list(SEEDS),
        "models": summaries,
        "same_parameter_count": next(iter(summaries["M_DEPLOY"].values()))["parameter_count"],
        "same_592D_architecture_optimizer_budget_pinball_and_planner_rows": True,
        "development_test_or_confirmation_used_for_fitting_or_selection": False,
        "G95_dataset_sha256": sha(G95 / "development_planner_dataset.npz"),
        "privileged_state_matrix_sha256": sha(OUT / "development_privileged_states.npz"),
        "privileged_adapter_sha256": sha(OUT / "privileged_input_adapter.npz"),
        "model_protocol_sha256": sha(OUT / "model_protocol.json"),
        "training_source_sha256": sha(__file__),
        "calibration": "per-replica validation-only nonnegative higher-order Q95 residual",
        "F_lim_N": F_LIM})
    print("G9.7 six matched deploy/privileged scalar-H2 models frozen", flush=True)


def load_replica(family, seed):
    manifest = json.loads((OUT / "training_manifest.json").read_text())
    if (manifest["status"] != "SIX_FIXED_MATCHED_CAPACITY_REPLICAS_SELECTED_ON_VALIDATION_ONLY" or
            manifest["training_source_sha256"] != sha(__file__) or
            manifest["privileged_adapter_sha256"] != sha(OUT / "privileged_input_adapter.npz")):
        raise RuntimeError("G9.7 frozen model/source changed")
    item = manifest["models"][family][str(seed)]
    folder = OUT / "models" / family / f"seed_{seed}"
    model_path, norm_path = folder / "model.pt", folder / "normalization.json"
    if sha(model_path) != item["checkpoint_sha256"] or sha(norm_path) != item["normalization_sha256"]:
        raise RuntimeError(f"G9.7 model/checkpoint hash changed {family}/{seed}")
    norm = json.loads(norm_path.read_text())
    net = QuantileNet(MODEL_INPUT_DIM).to("cuda:0")
    net.load_state_dict(torch.load(model_path, map_location="cuda:0", weights_only=True))
    net.eval()
    return net, np.asarray(norm["input_mean"], dtype=np.float64), np.asarray(norm["input_std"], dtype=np.float64), item["validation_only_Q95_delta_N"]


if __name__ == "__main__":
    train()
