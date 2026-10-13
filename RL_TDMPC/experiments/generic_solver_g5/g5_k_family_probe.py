"""Post-freeze, development-only mean-masking sensitivity probe of heterogeneous K.

Not a retrained leave-family-out comparison and not part of frozen model selection.
"""

from __future__ import annotations

import csv
import json
import os

import numpy as np
import torch

from g5_data import OUT, candidates, source, spec
from g5_prepare import SEEDS
from g5_metrics import load_predictor, predict, evaluate_candidate, quantiles
from g5_freeze import verify
from g4_data import candidate_states

FAMILIES = {"tip_xyz": list(range(0,3)), "distal_velocity_xyz": list(range(15,30)),
            "insertion_length": [35], "rotation_sin_cos": [36,37]}
MODELS = ("PKD", "ZP_noC")


def main():
    verify()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Frozen CUDA device required")
    arrays, split, index, physical = source()
    states = candidate_states("development_test", split)
    q = quantiles()
    rows = []
    device = torch.device("cuda:0")
    for name in MODELS:
        include_z, p_indices = spec(name)
        x, y, _ = candidates(name, states, arrays, index, physical)
        offset = 256 if include_z else 0
        for seed in SEEDS:
            predictor = load_predictor(name, seed, device)
            baseline = predict(predictor, x, device)
            _, base_aggregate, base_boundaries = evaluate_candidate(states, y, baseline, q)
            for family, members in FAMILIES.items():
                masked = x.copy()
                columns = [offset+p_indices.index(i) for i in members]
                for col in columns:
                    masked[:,col] = predictor[1][col]
                prediction = predict(predictor, masked, device)
                _, aggregate, boundaries = evaluate_candidate(states, y, prediction, q)
                rows.append({"model": name, "training_seed": seed, "family_mean_masked": family,
                    "masked_P_indices": "+".join(map(str,members)), "dimension_masked": len(members),
                    "baseline_RMSE_N": base_aggregate["candidate_RMSE_N"], "masked_RMSE_N": aggregate["candidate_RMSE_N"],
                    "RMSE_change_masked_minus_baseline_N": aggregate["candidate_RMSE_N"]-base_aggregate["candidate_RMSE_N"],
                    "baseline_Q95_false_safe_rate": base_boundaries["q95"]["false_safe_rate_P_pred_safe_given_true_unsafe"],
                    "masked_Q95_false_safe_rate": boundaries["q95"]["false_safe_rate_P_pred_safe_given_true_unsafe"],
                    "Q95_false_safe_change_masked_minus_baseline": boundaries["q95"]["false_safe_rate_P_pred_safe_given_true_unsafe"]-base_boundaries["q95"]["false_safe_rate_P_pred_safe_given_true_unsafe"],
                    "baseline_pairwise": base_aggregate["mean_pairwise_accuracy"], "masked_pairwise": aggregate["mean_pairwise_accuracy"],
                    "baseline_regret_N": base_aggregate["mean_safe_choice_regret_N"], "masked_regret_N": aggregate["mean_safe_choice_regret_N"]})
    path = OUT / "exploratory_postfreeze_K_family_mean_masking.csv"
    if path.exists():
        raise FileExistsError(path)
    with path.open("w",newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    note = OUT / "exploratory_postfreeze_K_family_mean_masking.json"
    note.write_text(json.dumps({"status":"EXPLORATORY_POSTFREEZE_DEVELOPMENT_ONLY_NO_RETRAINING_OR_SELECTION",
        "method":"Replace one K family at a time by its train-behavior input mean in the frozen model; candidate actions, other features and true labels unchanged",
        "limitation":"Mean masking breaks feature correlations and can be out-of-distribution; it is not a controlled retrained leave-family-out model, causal feature importance, or a deployability result",
        "development_test_only":True,"confirmation_predictions_unchanged":True,"P_min_unchanged":True,
        "models":list(MODELS),"families":FAMILIES,"training_seeds":list(SEEDS)},indent=2,sort_keys=True)+"\n")
    print("Post-freeze K-family sensitivity probe complete on development only",flush=True)


if __name__ == "__main__":
    main()
