"""Frozen-model G4 development diagnostics and paired state bootstrap."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
sys.path.insert(0, str(RL / "experiments/generic_solver_g3"))
from g3_evaluate import regression, classification, state_metrics, aggregate_candidate, quantiles  # noqa: E402
from prepare import sha  # noqa: E402
from g4_data import OUT, REPRESENTATIONS, source, physical_lookup, behavior_matrix, candidate_states, candidate_matrix  # noqa: E402
from g4_train import model, SEEDS  # noqa: E402

CELLS = tuple(f"{r}{s}" for r in REPRESENTATIONS for s in (0, 1))
COMPARISONS = (("Z0", "O0"), ("O0", "H0"), ("H0", "P0"), ("Z0", "P0"),
               ("Z1", "O1"), ("O1", "H1"), ("H1", "P1"), ("Z1", "P1"),
               ("Z0", "Z1"), ("O0", "O1"), ("H0", "H1"), ("P0", "P1"))


def write_csv(path, rows):
    if path.exists():
        raise FileExistsError(path)
    if not rows:
        raise RuntimeError(f"No rows for {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def three_seed_summary(rows, group_fields=("model",)):
    grouped = {}
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        grouped.setdefault(key, []).append(row)
    result = {}
    for key, members in grouped.items():
        if len(members) != 3 or sorted(row["training_seed"] for row in members) != list(SEEDS):
            raise RuntimeError(f"Incomplete three-seed metric group {key}")
        fields = [field for field in members[0] if field not in (*group_fields, "training_seed")]
        metrics = {}
        for field in fields:
            values = [row[field] for row in members]
            if all(value is not None and isinstance(value, (int, float)) and math.isfinite(value) for value in values):
                metrics[field] = {"individual": {str(row["training_seed"]): float(row[field]) for row in members},
                                  "mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1))}
        result["/".join(map(str, key))] = metrics
    return result


def load_predictor(cell, seed, device):
    representation = cell[0]
    norm = json.loads((OUT / "models" / representation / "normalization.json").read_text())
    mean = np.asarray(norm["input_mean"], dtype=np.float64)
    scale = np.asarray(norm["input_std"], dtype=np.float64)
    path = OUT / "models" / cell / f"seed_{seed}" / "model.pt"
    network = model(len(mean)).to(device)
    network.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    network.eval()
    return network, mean, scale


def predict(predictor, x, device):
    network, mean, scale = predictor
    with torch.inference_mode():
        return np.concatenate([network(torch.as_tensor((x[i:i+4096]-mean)/scale, dtype=torch.float32, device=device)).squeeze(-1).cpu().numpy()
                               for i in range(0, len(x), 4096)]).astype(np.float64)


def candidate_metrics(states, truth, prediction, cell, seed):
    individual = [{"model": cell, "training_seed": seed, **state_metrics(state, prediction[i*21:(i+1)*21])}
                  for i, state in enumerate(states)]
    aggregate = aggregate_candidate({"truth": truth, "prediction": prediction,
                                     "state_metrics": individual, "inference_seconds": 0.})
    return individual, aggregate


def bootstrap(outputs, states, q95):
    """1000 paired state draws; every draw evaluates all three seeds separately."""
    if len(states) != 120:
        raise RuntimeError("Expected 120 held-out states")
    rng = np.random.default_rng(20261008)
    metrics = ("median_within_state_Spearman", "mean_pairwise_accuracy", "mean_safe_choice_regret_N", "Q95_false_safe_rate")
    report = {}

    def values(cell, seed, indices):
        output = outputs[cell][seed]
        rows = output["states"]
        rho = [rows[i]["spearman"] for i in indices if rows[i]["spearman"] is not None]
        pairs = [rows[i]["pairwise_accuracy"] for i in indices if rows[i]["pairwise_accuracy"] is not None]
        regret = [rows[i]["safe_choice_regret_N"] for i in indices]
        candidate_indices = np.concatenate([np.arange(i*21, (i+1)*21) for i in indices])
        boundary = classification(output["truth"][candidate_indices], output["prediction"][candidate_indices], q95)
        return {"median_within_state_Spearman": float(np.median(rho)) if rho else np.nan,
                "mean_pairwise_accuracy": float(np.mean(pairs)) if pairs else np.nan,
                "mean_safe_choice_regret_N": float(np.mean(regret)),
                "Q95_false_safe_rate": boundary["false_safe_rate_P_pred_safe_given_true_unsafe"]}

    all_indices = np.arange(120)
    point = {cell: {seed: values(cell, seed, all_indices) for seed in SEEDS} for cell in CELLS}
    draw_store = {f"{right}_minus_{left}": {metric: [] for metric in metrics} for left, right in COMPARISONS}
    for _ in range(1000):
        indices = rng.integers(0, 120, size=120)
        sampled = {cell: {seed: values(cell, seed, indices) for seed in SEEDS} for cell in CELLS}
        for left, right in COMPARISONS:
            for metric in metrics:
                differences = []
                for seed in SEEDS:
                    a, b = sampled[left][seed][metric], sampled[right][seed][metric]
                    if a is not None and b is not None and np.isfinite(a) and np.isfinite(b):
                        differences.append(b-a)
                if differences:
                    draw_store[f"{right}_minus_{left}"][metric].append(float(np.mean(differences)))
    for left, right in COMPARISONS:
        key = f"{right}_minus_{left}"
        point_by_seed = {seed: {metric: point[right][seed][metric]-point[left][seed][metric]
                                for metric in metrics} for seed in SEEDS}
        draws = draw_store[key]
        report[key] = {metric: {"individual_seed_differences": {str(seed): point_by_seed[seed][metric] for seed in SEEDS},
                                "training_seed_mean_difference": float(np.nanmean([point_by_seed[seed][metric] for seed in SEEDS])),
                                "training_seed_sd_difference": float(np.nanstd([point_by_seed[seed][metric] for seed in SEEDS], ddof=1)),
                                "between_state_CI95_percentile": [float(np.quantile(draws[metric], .025)), float(np.quantile(draws[metric], .975))] if draws[metric] else None,
                                "bootstrap_replicates": len(draws[metric])} for metric in metrics}
    return {"method": "1000 fixed-seed paired bootstrap draws of 120 states; all 21 actions within state retained; metrics averaged over three independently trained models, no prediction ensembling",
            "seed": 20261008, "comparisons": report}


def main():
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available() or
            torch.cuda.device_count() != 1 or "RTX 5090" not in torch.cuda.get_device_name(0)):
        raise RuntimeError("Frozen CUDA device required")
    complete = json.loads((OUT / "training_complete.json").read_text())
    if complete["status"] != "COMPLETE_ALL_24_VALIDATION_SELECTED_CHECKPOINTS":
        raise RuntimeError("Training incomplete")
    arrays, split, index = source()
    physical = physical_lookup()
    states = candidate_states("development_test", split)
    q = quantiles()
    device = torch.device("cuda:0")
    behavior_rows, candidate_rows, boundary_rows = [], [], []
    state_rows = []
    outputs = {cell: {} for cell in CELLS}
    counts = {}
    for representation in REPRESENTATIONS:
        xbeh, ybeh, _ = behavior_matrix(representation, "development_test", arrays, split, index, physical)
        xcan, ycan = candidate_matrix(representation, states, arrays, index, physical)
        if len(xbeh) != 2111 or len(ycan) != 2520:
            raise RuntimeError(f"G4 development-test counts changed {representation}")
        counts[representation] = {"behavior": len(ybeh), "states": len(states), "candidates": len(ycan)}
        for supervision in (0, 1):
            cell = f"{representation}{supervision}"
            for seed in SEEDS:
                predictor = load_predictor(cell, seed, device)
                pbeh = predict(predictor, xbeh, device)
                pcan = predict(predictor, xcan, device)
                behavior_q95 = classification(ybeh, pbeh, q["q95"])
                behavior_q99 = classification(ybeh, pbeh, q["q99"])
                behavior_rows.append({"model": cell, "training_seed": seed, **regression(ybeh, pbeh),
                                      "Q95_true_unsafe": behavior_q95["true_exceedances"],
                                      "Q95_false_safe_count": behavior_q95["FN_false_safe"],
                                      "Q95_false_safe_rate": behavior_q95["false_safe_rate_P_pred_safe_given_true_unsafe"],
                                      "Q99_true_unsafe": behavior_q99["true_exceedances"],
                                      "Q99_false_safe_count": behavior_q99["FN_false_safe"],
                                      "Q99_false_safe_rate": behavior_q99["false_safe_rate_P_pred_safe_given_true_unsafe"]})
                individual, aggregate = candidate_metrics(states, ycan, pcan, cell, seed)
                state_rows.extend(individual)
                candidate_rows.append({"model": cell, "training_seed": seed, **aggregate})
                for reference, level in q.items():
                    boundary_rows.append({"model": cell, "training_seed": seed, "reference": reference,
                                          "reference_N": level, **classification(ycan, pcan, level)})
                outputs[cell][seed] = {"truth": ycan, "prediction": pcan, "states": individual}
    write_csv(OUT / "development_behavior_metrics.csv", behavior_rows)
    write_csv(OUT / "development_candidate_metrics.csv", candidate_rows)
    write_csv(OUT / "development_boundary_metrics.csv", boundary_rows)
    write_csv(OUT / "development_state_metrics.csv", state_rows)
    write_json(OUT / "development_three_seed_summary.json", {
        "behavior": three_seed_summary(behavior_rows),
        "candidate": three_seed_summary(candidate_rows),
        "boundary": three_seed_summary(boundary_rows, ("model", "reference")),
        "standard_deviation": "sample SD across three independent training seeds, ddof=1"})
    write_json(OUT / "bootstrap_summary.json", bootstrap(outputs, states, q["q95"]))
    write_json(OUT / "development_evaluation_summary.json", {"status": "COMPLETE_ALL_EIGHT_CELLS_THREE_SEEDS", "counts": counts,
                                                               "comparison_rule": "descriptive paired representations and supervision modes; no post-hoc threshold tuning",
                                                               "behavior_metrics_sha256": sha(OUT / "development_behavior_metrics.csv"),
                                                               "candidate_metrics_sha256": sha(OUT / "development_candidate_metrics.csv"),
                                                               "boundary_metrics_sha256": sha(OUT / "development_boundary_metrics.csv")})
    print("G4 development evaluation complete; confirmation still untouched", flush=True)


if __name__ == "__main__":
    main()
