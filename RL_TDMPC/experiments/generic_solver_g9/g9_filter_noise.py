"""Validation-only UKF measurement-noise and per-episode reset diagnostics."""

from __future__ import annotations

import json

import numpy as np
import torch

from g9_estimate import OUT, G3, Transition, ukf_episode, data, split_mask, cuda_gate
from g9_events import map_physical
from g9_risk import json_new


def main():
    cuda_gate()
    d = data()
    val = split_mask(d, "validation")
    selected = {(int(s), int(k)): i for i, (s, k) in enumerate(zip(d["seed"], d["decision_index"])) if val[i]}
    config = json.loads((OUT / "filter_definition.json").read_text())
    Q = np.asarray(config["process_covariance_diagonal"])
    R = float(config["measurement_covariance_diagonal"][0])
    model = Transition()
    model.load_state_dict(torch.load(OUT / "models/transition.pt", map_location="cpu", weights_only=True))
    model.eval()
    with np.load(G3 / "behavior_features.npz") as z:
        seeds, decisions, observations, actions = [z[k].copy() for k in ("seed", "decision_index", "observation", "action")]
    clean = np.load(OUT / "ukf_current_features.npz")["features"]
    result = {"source": "validation episodes only", "noise_model": "independent Gaussian noise on ten normalized tracking-position measurements at every decision; fixed seed 260901+episode_seed",
              "sigma_normalized_tracking": (0, .005, .01), "rows": int(np.sum(val)), "per_episode_reset": True, "results": []}
    for sigma in result["sigma_normalized_tracking"]:
        features = np.empty((len(selected), 34), dtype=np.float32)
        placement = {}
        innovations = []
        initial_velocity_norm = []
        for seed in sorted(set(int(s) for s in d["seed"][val])):
            indices = np.flatnonzero(seeds == seed)
            if not np.array_equal(decisions[indices], np.arange(len(indices))):
                raise RuntimeError(f"UKF validation episode order changed {seed}")
            obs = observations[indices].copy()
            if sigma:
                rng = np.random.default_rng(260901+seed)
                obs[:, :10] += rng.normal(0, sigma, size=(len(obs), 10)).astype(np.float32)
            ft, diag = ukf_episode(model, obs, actions[indices], Q, R)
            initial_velocity_norm.append(float(np.linalg.norm(ft[0, 14:24])))
            innovations.extend(float(row[0]) for row in diag)
            for decision in range(len(indices)):
                key = (seed, decision)
                if key in selected:
                    placement[key] = ft[decision]
        ordered = [(int(s), int(k)) for s, k in zip(d["seed"][val], d["decision_index"][val])]
        features = np.asarray([placement[key] for key in ordered], dtype=np.float32)
        if sigma == 0 and not np.array_equal(features, clean[val]):
            raise RuntimeError("Validation UKF clean rerun differs from previously saved features")
        estimate = map_physical("E2", features)
        truth = d["privileged_physical"][val]
        errors = estimate-truth
        result["results"].append({"sigma": sigma,
            "contact_magnitude_RMSE_N": float(np.sqrt(np.mean(errors[:, 33]**2))),
            "distal_velocity_RMSE_mm_per_s": float(np.sqrt(np.mean(errors[:, 15:30]**2))),
            "tip_position_RMSE_mm": float(np.sqrt(np.mean(errors[:, :3]**2))),
            "mean_innovation_norm": float(np.mean(innovations)),
            "max_initial_velocity_norm_after_reset": max(initial_velocity_norm),
            "nonfinite_feature_count": int(np.sum(~np.isfinite(features)))})
    json_new(OUT / "filter_noise_sensitivity.json", result)
    print(json.dumps(result["results"], indent=2))


if __name__ == "__main__":
    main()
