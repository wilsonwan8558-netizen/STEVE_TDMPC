"""Apply the frozen validation onset-warning rule to untouched confirmation."""

from __future__ import annotations

import csv
import json

import numpy as np

from g9_risk import OUT, ESTIMATORS, HORIZONS, F_LIM, cuda_gate, load_model, predict


def write_csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    cuda_gate()
    frozen = json.loads((OUT / "pre_confirmation_freeze.json").read_text())
    if frozen["status"] != "G9_ALL_MODEL_AND_HORIZON_CHOICES_FROZEN_BEFORE_102000_CONFIRMATION":
        raise RuntimeError("G9 pre-confirmation freeze missing")
    with np.load(OUT / "confirmation_temporal_behavior.npz") as z:
        d = {key: z[key].copy() for key in z.files}
    keys = {(int(s), int(k)): i for i, (s, k) in enumerate(zip(d["seed"], d["decision_index"]))}
    onsets = []
    for (seed, decision), i in sorted(keys.items()):
        prev = keys.get((seed, decision-1))
        if prev is not None and d["force_sequence"][prev, 0] <= F_LIM < d["force_sequence"][i, 0]:
            onsets.append((seed, decision))
    if not onsets:
        raise RuntimeError("No confirmation high-force onsets in complete H5 cohort")
    summary, detail = [], []
    for name in ESTIMATORS:
        for horizon in HORIZONS:
            x = np.concatenate((d["latent"], d[name], d["action_sequence"][:, :horizon].reshape(len(d["seed"]), -1)), axis=1)
            y = np.max(d["force_sequence"][:, :horizon], axis=1)
            net, mean, scale, delta = load_model(name, horizon)
            p = predict(net, x, mean, scale)
            leads, clean_leads = [], []
            for seed, onset in onsets:
                eligible = [keys[(seed, k)] for k in range(onset-5, onset)
                            if (seed, k) in keys and k+horizon-1 >= onset]
                warning = [i for i in eligible if y[i] > F_LIM and p[i, 2]+delta > F_LIM]
                lead = max((onset-int(d["decision_index"][i]) for i in warning), default=None)
                clean = [i for i in warning if (seed, int(d["decision_index"][i])-1) in keys and
                         d["force_sequence"][keys[(seed, int(d["decision_index"][i])-1)], 0] <= F_LIM]
                clean_lead = max((onset-int(d["decision_index"][i]) for i in clean), default=None)
                leads.append(lead)
                clean_leads.append(clean_lead)
                detail.append({"estimator": name, "horizon": horizon, "seed": seed,
                    "onset_decision_index": onset, "warning_lead_steps": lead,
                    "warning_lead_s": None if lead is None else lead*.132,
                    "clean_preexisting_force_warning_lead_steps": clean_lead,
                    "warning_requires_future_realized_behavior_actions": True})
            valid = [value for value in leads if value is not None]
            valid_clean = [value for value in clean_leads if value is not None]
            summary.append({"estimator": name, "horizon": horizon, "onset_events": len(onsets),
                "early_warning_count": len(valid), "early_warning_recall": len(valid)/len(onsets),
                "no_warning_count": len(onsets)-len(valid),
                "median_warning_lead_steps_when_warned": None if not valid else float(np.median(valid)),
                "max_warning_lead_steps": None if not valid else max(valid),
                "clean_preexisting_force_warning_count": len(valid_clean),
                "clean_preexisting_force_warning_recall": len(valid_clean)/len(onsets),
                "protocol": "validation-frozen first above-limit behavior interval after a below-limit interval; event force must lie within H; realized future behavior actions are hindsight, not deployable"})
    write_csv(OUT / "confirmation_early_warning.csv", summary)
    write_csv(OUT / "confirmation_event_warning_details.csv", detail)
    print(f"G9 confirmation onset warning analysis: {len(onsets)} onsets, 12 estimator-horizon rows")


if __name__ == "__main__":
    main()
