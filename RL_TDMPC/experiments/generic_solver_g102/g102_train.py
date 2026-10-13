"""Train three frozen compact distributional local delta-force replicas."""

from __future__ import annotations

import json
import math
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = __import__("pathlib").Path(__file__).resolve().parent
RL = HERE.parents[1]
for name in ("generic_solver_g102", "generic_solver_g101", "generic_solver_g10", "generic_solver_g95"):
    sys.path.insert(0, str(RL / "experiments" / name))

from g102_prepare import OUT, SEEDS, F_LIM, verify_source  # noqa: E402
from g95_common import json_new, sha  # noqa: E402

TAUS = torch.as_tensor((.5, .9, .95), dtype=torch.float32, device="cuda:0")


class DeltaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(296, 64), nn.ReLU(), nn.Linear(64, 64),
                                    nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        raw = self.layers(x)
        return torch.stack((raw[:, 0], raw[:, 0]+F.softplus(raw[:, 1]),
                            raw[:, 0]+F.softplus(raw[:, 1])+F.softplus(raw[:, 2])), dim=1)


def pinball(prediction, target, weight):
    diff = target[:, None]-prediction
    loss = torch.maximum(TAUS*diff, (TAUS-1)*diff).mean(dim=1)
    return torch.sum(loss*weight)/torch.sum(weight)


def assemble():
    verify_source()
    audit = json.loads((OUT / "expansion_parent_replay_audit.json").read_text())
    if audit["status"] != "PASS_EXACT_REUSED_SCENE_PARENT" or audit["train_validation_states"] != 680:
        raise RuntimeError("G10.2 exact train/validation expansion audit required")
    manifest = json.loads((OUT / "local_dataset_manifest.json").read_text())
    features, targets, weights, groups, state_ids, seeds, translations = [], [], [], [], [], [], []
    category = {g: {"abs_delta_le_0_1": 0, "abs_delta_le_0_2": 0,
                    "abs_delta_le_0_5": 0, "abs_delta_gt_0_5": 0,
                    "nonzero_complete_rows": 0, "censored_rows": 0}
                for g in ("train", "validation")}
    prior_force_coverage = {g: {"previous_force_zero": 0, "previous_force_positive_low": 0,
        "previous_force_near_0_8_to_1_2_F_lim": 0,
        "previous_force_above_1_2_F_lim": 0,
        "selection_slots": {"early": 0, "middle": 0, "late": 0, "prior_force_Q95": 0}}
        for g in ("train", "validation")}
    for row in manifest["train_validation_states"]:
        group = row["group"]
        path = OUT / f"{group}_local_curves" / f"state_{int(row['state_id']):04d}.json"
        item = json.loads(path.read_text())
        if (item["seed"], item["decision_index"], item["source_sha256"], item["nominal_pair"]) != (
                row["seed"], row["decision_index"], row["source_sha256"], row["nominal_pair"]):
            raise RuntimeError("G10.2 local force curve/source alignment changed")
        source = __import__("pathlib").Path(row["source_path"])
        if sha(source) != row["source_sha256"]:
            raise RuntimeError("G10.2 archived feature source changed")
        s = json.loads(source.read_text())
        previous_force = float(s["previous_executed_force_N"])
        if previous_force == 0:
            prior_force_coverage[group]["previous_force_zero"] += 1
        elif previous_force < .8*F_LIM:
            prior_force_coverage[group]["previous_force_positive_low"] += 1
        elif previous_force <= 1.2*F_LIM:
            prior_force_coverage[group]["previous_force_near_0_8_to_1_2_F_lim"] += 1
        else:
            prior_force_coverage[group]["previous_force_above_1_2_F_lim"] += 1
        prior_force_coverage[group]["selection_slots"][s["slot"]] += 1
        x = np.r_[s["current_latent"], s["E1_current_physical_estimate"]].astype(np.float32)
        if x.shape != (294,) or item["R2_nom_true_N"] != row["nominal_true_R2_N"]:
            raise RuntimeError("G10.2 local target/feature alignment changed")
        for probe in item["true_curve"]:
            if not probe["complete_two_intervals"]:
                category[group]["censored_rows"] += 1
                continue
            delta = float(np.float32(probe["v"]-item["v_nom"]))
            if probe["delta_v"] != delta or not np.isclose(
                    probe["delta_R2_true_N"], probe["R2_true_N"]-item["R2_nom_true_N"],
                    rtol=0, atol=1e-10):
                raise RuntimeError("G10.2 local delta target misaligned")
            ad = abs(delta)
            category[group]["abs_delta_le_0_1"] += ad <= .1+1e-7
            category[group]["abs_delta_le_0_2"] += ad <= .2+1e-7
            category[group]["abs_delta_le_0_5"] += ad <= .5+1e-7
            category[group]["abs_delta_gt_0_5"] += ad > .5+1e-7
            category[group]["nonzero_complete_rows"] += ad > 0
            features.append(np.r_[x, np.float32(item["v_nom"]), np.float32(delta)])
            targets.append(probe["delta_R2_true_N"])
            weights.append(0. if ad == 0 else (1. if ad <= .5+1e-7 else .25))
            groups.append(0 if group == "train" else 1)
            state_ids.append(row["state_id"])
            seeds.append(row["seed"])
            translations.append(probe["v"])
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32)
    g = np.asarray(groups, dtype=np.int8)
    if (x.shape[1:] != (296,) or len(x) != len(y) or not np.all(np.isfinite(x)) or
            not np.all(np.isfinite(y)) or sum(g == 0) == 0 or sum(g == 1) == 0):
        raise RuntimeError("G10.2 local training arrays invalid")
    if (OUT / "local_dataset_arrays.npz").exists():
        raise FileExistsError("G10.2 local dataset already materialized")
    np.savez_compressed(OUT / "local_dataset_arrays.npz", x=x, delta_R2_N=y,
        weight=w, group_index=g, state_id=np.asarray(state_ids, dtype=np.int32),
        episode_seed=np.asarray(seeds, dtype=np.int32), v=np.asarray(translations, dtype=np.float32))
    json_new(OUT / "local_dataset_summary.json", {"state_counts": {"train": 560, "validation": 120},
        "complete_row_count": len(x), "category_counts_overlapping_cumulative_bands": category,
        "descriptive_previous_executed_force_coverage_not_selection_by_future_repair_label": prior_force_coverage,
        "all_rows_episode_split_inherited_G95": True,
        "no_G101_dev_or_108000_confirmation_labels_in_training": True,
        "dataset_arrays_sha256": sha(OUT / "local_dataset_arrays.npz")})
    return x, y, w, g


def load_arrays():
    with np.load(OUT / "local_dataset_arrays.npz") as z:
        return z["x"].copy(), z["delta_R2_N"].copy(), z["weight"].copy(), z["group_index"].copy()


def predict(net, x, mean, std):
    z = torch.as_tensor((x-mean)/std, dtype=torch.float32, device="cuda:0")
    with torch.inference_mode():
        return net(z).cpu().numpy()


def train():
    verify_source()
    if (OUT / "model_training_manifest.json").exists():
        raise FileExistsError("G10.2 local models already frozen")
    if (OUT / "local_dataset_arrays.npz").exists():
        x, y, w, g = load_arrays()
    else:
        x, y, w, g = assemble()
    train_mask, val_mask = (g == 0) & (w > 0), (g == 1) & (w > 0)
    mean = np.mean(x[train_mask], axis=0, dtype=np.float64)
    std = np.maximum(np.std(x[train_mask], axis=0, dtype=np.float64), 1e-6)
    if (OUT / "normalization.json").exists():
        old = json.loads((OUT / "normalization.json").read_text())
        if not np.array_equal(mean, np.asarray(old["mean"])) or not np.array_equal(std, np.asarray(old["std"])):
            raise RuntimeError("G10.2 train-only normalization changed")
    else:
        json_new(OUT / "normalization.json", {"mean": mean.tolist(), "std": std.tolist(),
            "source": "train-only nonnominal complete local rows; validation/development/confirmation excluded",
            "std_floor": 1e-6})
    tx = torch.as_tensor((x[train_mask]-mean)/std, dtype=torch.float32, device="cuda:0")
    ty = torch.as_tensor(y[train_mask], dtype=torch.float32, device="cuda:0")
    tw = torch.as_tensor(w[train_mask], dtype=torch.float32, device="cuda:0")
    vx = torch.as_tensor((x[val_mask]-mean)/std, dtype=torch.float32, device="cuda:0")
    vy = torch.as_tensor(y[val_mask], dtype=torch.float32, device="cuda:0")
    vw = torch.as_tensor(w[val_mask], dtype=torch.float32, device="cuda:0")
    models = {}
    validation_predictions = []
    for replica_index, seed in enumerate(SEEDS, start=1):
        folder = OUT / "models" / f"delta_seed_{replica_index}"
        folder.mkdir(exist_ok=True)
        checkpoint = folder / "model.pt"
        selection = folder / "selection.json"
        if checkpoint.exists() or selection.exists():
            raise FileExistsError(f"G10.2 model seed already trained {seed}")
        torch.manual_seed(seed)
        net = DeltaNet().to("cuda:0")
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        best = math.inf
        patience = 0
        history = []
        for epoch in range(80):
            net.train()
            order = torch.randperm(len(tx), generator=generator).to("cuda:0")
            for batch in order.split(512):
                loss = pinball(net(tx[batch]), ty[batch], tw[batch])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            net.eval()
            with torch.inference_mode():
                value = float(pinball(net(vx), vy, vw).cpu())
            history.append({"epoch": epoch, "validation_weighted_pinball_N": value})
            if value < best-1e-5:
                best, patience = value, 0
                torch.save(net.cpu().state_dict(), checkpoint)
                net.to("cuda:0")
            else:
                patience += 1
                if patience >= 10:
                    break
        net.load_state_dict(torch.load(checkpoint, map_location="cuda:0", weights_only=True))
        net.eval()
        pred = predict(net, x[val_mask], mean, std)
        residual = y[val_mask]-pred[:, 2]
        correction = max(0., float(np.quantile(residual, .95, method="higher")))
        validation_predictions.append(pred[:, 2]+correction)
        json_new(selection, {"seed": seed, "best_validation_weighted_pinball_N": best,
            "epochs_run": len(history), "history": history,
            "train_rows_nonzero_delta": int(sum(train_mask)),
            "validation_rows_nonzero_delta": int(sum(val_mask)),
            "validation_raw_Q95_coverage": float(np.mean(y[val_mask] <= pred[:, 2])),
            "validation_corrected_Q95_coverage": float(np.mean(y[val_mask] <= pred[:, 2]+correction)),
            "validation_only_Q95_delta_correction_N": correction,
            "checkpoint_sha256": sha(checkpoint),
            "no_development_or_confirmation_labels_used": True})
        models[str(seed)] = {"checkpoint_sha256": sha(checkpoint),
            "selection_sha256": sha(selection),
            "validation_only_Q95_delta_correction_N": correction,
            "best_validation_weighted_pinball_N": best}
        print(f"G10.2 M_DELTA seed={seed} pinball={best:.5f}, correction={correction:.5f}", flush=True)
    corrected = np.stack(validation_predictions, axis=1)
    spread = np.max(corrected, axis=1)-np.min(corrected, axis=1)
    threshold = float(np.quantile(spread, .95, method="higher"))
    json_new(OUT / "validation_ensemble_rule.json", {
        "status": "FROZEN_FROM_VALIDATION_BEFORE_DEVELOPMENT_TEST",
        "Q95_corrected_replica_spread_threshold_N": threshold,
        "validation_nonzero_delta_rows": int(sum(val_mask)),
        "validation_spread_Q50_N": float(np.median(spread)),
        "validation_spread_Q95_N": threshold,
        "validation_spread_max_N": float(np.max(spread)),
        "all_replica_corrected_Q95_delta_conservative_max": True,
        "uncertainty_acceptance": "spread<=validation Q95 threshold",
        "nominal_delta_zero_override": True})
    json_new(OUT / "model_training_manifest.json", {
        "status": "THREE_FIXED_M_DELTA_REPLICAS_TRAINED_BEFORE_DEVELOPMENT_TEST",
        "training_seeds": list(SEEDS), "models": models,
        "normalization_sha256": sha(OUT / "normalization.json"),
        "validation_ensemble_rule_sha256": sha(OUT / "validation_ensemble_rule.json"),
        "dataset_arrays_sha256": sha(OUT / "local_dataset_arrays.npz"),
        "training_code_sha256": sha(__file__),
        "M_LOCAL_ABS_trained": False,
        "best_seed_selection": False,
        "development_test_or_confirmation_used_for_training_selection": False})
    print("G10.2 three fixed M_DELTA replicas and validation-only abstention rule frozen", flush=True)


if __name__ == "__main__":
    train()
