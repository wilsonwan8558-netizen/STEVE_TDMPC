"""Freeze minimal K/D subset using validation candidate labels only."""

from __future__ import annotations

import csv
import json
import os

import numpy as np
import torch

from g5_data import OUT, GROUP_FOR_SUBSET, candidates, source, selected_indices
from g5_prepare import SEEDS
from g5_metrics import load_predictor, predict, evaluate_candidate, quantiles
from g4_data import candidate_states
from g4_train import new_json


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Frozen CUDA device required")
    if not (OUT / "primary_training_complete.json").exists():
        raise RuntimeError("Primary training incomplete")
    arrays, split, index, physical = source()
    states = candidate_states("validation", split)
    if len(states) != 120:
        raise RuntimeError("Validation state count differs")
    q = quantiles()
    rows = []
    device = torch.device("cuda:0")
    for subset, (z_name, p_name) in GROUP_FOR_SUBSET.items():
        groups = tuple(subset) if subset != "KD" else ("K", "D")
        for name in (z_name, p_name):
            x, y, _ = candidates(name, states, arrays, index, physical)
            for seed in SEEDS:
                prediction = predict(load_predictor(name, seed, device), x, device)
                _, aggregate, boundaries = evaluate_candidate(states, y, prediction, q)
                rows.append({"subset": subset, "model": name, "training_seed": seed, "groups": "+".join(groups),
                             "P_dimension": len(selected_indices(groups)), "total_input_dimension": x.shape[1],
                             "Q95_false_safe_rate": boundaries["q95"]["false_safe_rate_P_pred_safe_given_true_unsafe"],
                             "candidate_RMSE_N": aggregate["candidate_RMSE_N"],
                             "mean_pairwise_accuracy": aggregate["mean_pairwise_accuracy"],
                             "mean_safe_choice_regret_N": aggregate["mean_safe_choice_regret_N"],
                             "Q95_true_exceedances": boundaries["q95"]["true_exceedances"]})
    path = OUT / "deployable_subset_metrics.csv"
    if path.exists():
        raise FileExistsError(path)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    means = {}
    for subset, (z_name, _) in GROUP_FOR_SUBSET.items():
        members = [row for row in rows if row["model"] == z_name]
        if len(members) != 3:
            raise RuntimeError("Subset training seeds incomplete")
        means[subset] = {field: float(np.mean([row[field] for row in members])) for field in
                         ("Q95_false_safe_rate", "candidate_RMSE_N", "mean_pairwise_accuracy", "mean_safe_choice_regret_N")}
    keys = lambda subset: (means[subset]["Q95_false_safe_rate"], means[subset]["candidate_RMSE_N"],
                            -means[subset]["mean_pairwise_accuracy"], means[subset]["mean_safe_choice_regret_N"],
                            len(subset), len(selected_indices(tuple(subset))))
    chosen = min(GROUP_FOR_SUBSET, key=keys)
    groups = tuple(chosen)
    new_json(OUT / "minimal_subset_selection.json", {"status": "FROZEN_VALIDATION_ONLY_P_MIN_SELECTED",
        "included_groups": list(groups), "P_min_indices": selected_indices(groups), "P_min_dimension": len(selected_indices(groups)),
        "selection_model": GROUP_FOR_SUBSET[chosen][0], "P_min_behavior_only_model": GROUP_FOR_SUBSET[chosen][1],
        "selection_metric_order": ["mean_validation_Q95_false_safe_rate", "mean_validation_candidate_RMSE_N",
                                   "negative_mean_validation_pairwise_accuracy", "mean_validation_safe_choice_regret_N", "group_count", "dimension"],
        "validation_three_seed_means": means, "development_test_opened_before_selection": False,
        "confirmation_opened_before_selection": False,
        "limitation": "Class-B shape and velocity features remain unmeasured in the deployed controller even if selected"})
    print(f"G5 validation-only P_min selected {chosen}; test and confirmation unopened", flush=True)


if __name__ == "__main__":
    main()
