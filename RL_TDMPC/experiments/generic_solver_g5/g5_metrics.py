"""Common frozen-model inference and matched G3/G4 metrics for G5."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

from g5_data import OUT
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generic_solver_g4"))
from g4_train import model  # noqa: E402
from g4_evaluate import predict  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generic_solver_g3"))
from g3_evaluate import regression, classification, state_metrics, aggregate_candidate, quantiles  # noqa: E402


def canonical(name, groups=None):
    if name == "Pmin":
        return {("K",): "PK", ("D",): "PD", ("K", "D"): "PKD"}[tuple(groups)]
    if name == "ZPmin":
        return {("K",): "ZK", ("D",): "ZD", ("K", "D"): "ZP_noC"}[tuple(groups)]
    return name


def load_predictor(name, seed, device, groups=None):
    actual = canonical(name, groups)
    norm_name = "Pmin" if actual == "Pmin1" else actual[:-1] if actual in ("Z1", "P1") else actual
    norm = json.loads((OUT / "models" / norm_name / "normalization.json").read_text())
    mean = np.asarray(norm["input_mean"], dtype=np.float64)
    scale = np.asarray(norm["input_std"], dtype=np.float64)
    network = model(len(mean)).to(device)
    network.load_state_dict(torch.load(OUT / "models" / actual / f"seed_{seed}" / "model.pt", map_location=device, weights_only=True))
    network.eval()
    return network, mean, scale


def evaluate_candidate(states, truth, prediction, q):
    individual = [state_metrics(state, prediction[i*21:(i+1)*21]) for i, state in enumerate(states)]
    aggregate = aggregate_candidate({"truth": truth, "prediction": prediction, "state_metrics": individual, "inference_seconds": 0.})
    boundaries = {reference: classification(truth, prediction, level) for reference, level in q.items()}
    return individual, aggregate, boundaries
