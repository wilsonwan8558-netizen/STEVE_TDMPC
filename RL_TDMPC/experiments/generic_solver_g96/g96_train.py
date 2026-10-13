"""Three fixed profile replicas; one shared validation-only correction per step."""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
import torch

from g96_common import (OUT, SEEDS, StepwiseNet, profile_pinball, predict_profile,
                        group_data, assert_prepared, json_new, sha)


def train_one(seed, index, tx, ty, vx, vy, mean, scale, behavior_train, planner_train,
              behavior_val, planner_val):
    folder = OUT / "models" / f"stepwise_seed_{index}"
    path = folder / "model.pt"
    selection = folder / "selection.json"
    if path.exists() or selection.exists():
        raise FileExistsError(f"G9.6 stepwise replica already trained {seed}")
    torch.manual_seed(seed)
    net = StepwiseNet().to("cuda:0")
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best = math.inf
    patience = 0
    history = []
    for epoch in range(60):
        net.train()
        order = torch.randperm(len(tx), generator=generator).to("cuda:0")
        for batch in order.split(512):
            loss = profile_pinball(net(tx[batch]), ty[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        net.eval()
        with torch.inference_mode():
            value = float(profile_pinball(net(vx), vy).cpu())
        history.append({"epoch": epoch, "validation_sum_step_pinball_N": value})
        if value < best-1e-5:
            best = value
            patience = 0
            torch.save(net.cpu().state_dict(), path)
            net.to("cuda:0")
        else:
            patience += 1
            if patience >= 8:
                break
    json_new(selection, {"training_seed": seed, "replica_index": index,
        "train_rows": len(tx), "validation_rows": len(vx),
        "behavior_train_rows": behavior_train, "planner_train_rows": planner_train,
        "behavior_validation_rows": behavior_val, "planner_validation_rows": planner_val,
        "best_validation_sum_step_pinball_N": best, "history": history,
        "checkpoint_sha256": sha(path), "normalization": "exact frozen G9 E1/H2 input mean/std",
        "checkpoint_selected_on": "minimum VALIDATION sum of F1+F2 four-quantile pinball"})
    print(f"G9.6 stepwise seed={seed} best val loss={best:.6f}, epochs={len(history)}", flush=True)
    return {"checkpoint_sha256": sha(path), "selection_sha256": sha(selection),
            "best_validation_sum_step_pinball_N": best}


def train():
    source = assert_prepared()
    if (OUT / "training_manifest.json").exists() or (OUT / "calibration_result.json").exists():
        raise FileExistsError("G9.6 training/calibration already completed")
    xtrain, ytrain, mean, scale, bt, pt = group_data("train")
    xval, yval, _, _, bv, pv = group_data("validation")
    if (len(xtrain), len(xval), bt, pt, bv, pv) != (15241, 3216, 10203, 5038, 2136, 1080):
        raise RuntimeError("G9.6 row counts differ from G9.5 M1")
    tx = torch.as_tensor((xtrain-mean)/scale, dtype=torch.float32, device="cuda:0")
    ty = torch.as_tensor(ytrain, dtype=torch.float32, device="cuda:0")
    vx = torch.as_tensor((xval-mean)/scale, dtype=torch.float32, device="cuda:0")
    vy = torch.as_tensor(yval, dtype=torch.float32, device="cuda:0")
    models = {}
    for index, seed in enumerate(SEEDS, 1):
        models[str(seed)] = train_one(seed, index, tx, ty, vx, vy, mean, scale,
                                      bt, pt, bv, pv)
    json_new(OUT / "training_manifest.json", {"status": "THREE_FIXED_G96_REPLICAS_VALIDATION_SELECTED",
        "seeds": SEEDS, "models": models, "source_G95_fingerprint": source,
        "training_code_sha256": sha(__file__),
        "train_rows": len(xtrain), "validation_rows": len(xval),
        "development_test_or_confirmation_used_for_selection": False,
        "no_best_seed_selection": True})
    print("G9.6 all three stepwise models trained; no development-test or confirmation labels selected them", flush=True)


def load_replica(seed):
    manifest = json.loads((OUT / "training_manifest.json").read_text())
    index = SEEDS.index(seed)+1
    path = OUT / "models" / f"stepwise_seed_{index}" / "model.pt"
    item = manifest["models"][str(seed)]
    if sha(path) != item["checkpoint_sha256"]:
        raise RuntimeError(f"G9.6 checkpoint changed {seed}")
    net = StepwiseNet().to("cuda:0")
    net.load_state_dict(torch.load(path, map_location="cuda:0", weights_only=True))
    net.eval()
    return net


def calibrate():
    assert_prepared()
    if (OUT / "calibration_result.json").exists():
        raise FileExistsError(OUT / "calibration_result.json")
    definition = json.loads((OUT / "calibration_definition.json").read_text())
    if definition["status"] != "PREDECLARED_BEFORE_G96_TRAINING_AND_DEVELOPMENT_TEST":
        raise RuntimeError("G9.6 predeclared calibration rule changed")
    xval, yval, mean, scale, bv, pv = group_data("validation")
    if len(xval) != 3216:
        raise RuntimeError("G9.6 validation cohort changed")
    predictions = {}
    coverage = {}
    residuals = []
    for seed in SEEDS:
        p = predict_profile(load_replica(seed), xval, mean, scale)
        predictions[str(seed)] = p
        coverage[str(seed)] = [float(np.mean(yval[:, step] <= p[:, step, 2]))
                               for step in range(2)]
        residuals.append(yval-p[:, :, 2])
    delta = []
    applied = []
    for step in range(2):
        mean_coverage = float(np.mean([coverage[str(seed)][step] for seed in SEEDS]))
        under = mean_coverage < .95
        pooled = np.concatenate([residuals[index][:, step] for index in range(3)])
        value = max(0.0, float(np.quantile(pooled, .95, method="higher"))) if under else 0.0
        delta.append(value)
        applied.append(under)
    corrected = {str(seed): [float(np.mean(yval[:, step] <=
        predictions[str(seed)][:, step, 2]+delta[step])) for step in range(2)] for seed in SEEDS}
    json_new(OUT / "calibration_result.json", {"status": "VALIDATION_ONLY_SHARED_STEP_CORRECTION_FROZEN_BEFORE_DEVELOPMENT_TEST",
        "seeds": SEEDS, "validation_behavior_rows": bv, "validation_planner_rows": pv,
        "raw_Q95_coverage_by_seed_F1_F2": coverage,
        "mean_raw_Q95_coverage_F1_F2": [float(np.mean([coverage[str(seed)][step]
                                                    for seed in SEEDS])) for step in range(2)],
        "correction_applied_F1_F2": applied, "shared_delta1_N": delta[0],
        "shared_delta2_N": delta[1],
        "corrected_Q95_coverage_by_seed_F1_F2": corrected,
        "calibration_definition_sha256": sha(OUT / "calibration_definition.json"),
        "training_manifest_sha256": sha(OUT / "training_manifest.json"),
        "no_development_test_or_confirmation_labels_used": True})
    print(f"G9.6 shared VALIDATION deltas F1={delta[0]:.6f} F2={delta[1]:.6f}; raw coverage={coverage}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("train", "calibrate"), required=True)
    {"train": train, "calibrate": calibrate}[parser.parse_args().stage]()


if __name__ == "__main__":
    main()
