"""Report all frozen quantile coverages on untouched confirmation behavior."""

from __future__ import annotations

import csv
import json

import numpy as np

from g9_risk import OUT, ESTIMATORS, HORIZONS, TAUS, cuda_gate, load_model, predict


def main():
    cuda_gate()
    frozen = json.loads((OUT / "pre_confirmation_freeze.json").read_text())
    if frozen["status"] != "G9_ALL_MODEL_AND_HORIZON_CHOICES_FROZEN_BEFORE_102000_CONFIRMATION":
        raise RuntimeError("G9 model freeze missing")
    with np.load(OUT / "confirmation_temporal_behavior.npz") as z:
        d = {key: z[key].copy() for key in z.files}
    rows = []
    for name in ESTIMATORS:
        for horizon in HORIZONS:
            x = np.concatenate((d["latent"], d[name], d["action_sequence"][:, :horizon].reshape(len(d["seed"]), -1)), axis=1)
            y = np.max(d["force_sequence"][:, :horizon], axis=1)
            net, mean, scale, delta = load_model(name, horizon)
            p = predict(net, x, mean, scale)
            for index, tau in enumerate(TAUS):
                rows.append({"estimator": name, "horizon": horizon, "quantile": f"Q{int(tau*100)}",
                    "nominal_tau": tau, "rows": len(y),
                    "empirical_coverage": float(np.mean(y <= p[:, index])),
                    "coverage_error": float(np.mean(y <= p[:, index])-tau),
                    "mean_pinball_N": float(np.mean(np.maximum(tau*(y-p[:, index]), (tau-1)*(y-p[:, index])))),
                    "crossing_count": int(np.sum(np.diff(p, axis=1) < 0)),
                    "Q95_validation_delta_N": delta if index == 2 else None})
    with (OUT / "confirmation_quantile_coverage.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"G9 confirmation all-quantile coverage rows: {len(rows)}")


if __name__ == "__main__":
    main()
