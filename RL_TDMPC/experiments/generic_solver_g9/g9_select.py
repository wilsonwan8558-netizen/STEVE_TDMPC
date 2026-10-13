"""Validation-only selection and high-force-onset warning proxy for G9."""

from __future__ import annotations

import json

import numpy as np

from g9_risk import OUT, ESTIMATORS, HORIZONS, F_LIM, cuda_gate, data, json_new, load_model, matrix, metrics, predict


def main():
    cuda_gate()
    d, estimates, split = data()
    mask = np.isin(d["seed"], split["validation"])
    keys = {(int(s), int(k)): i for i, (s, k) in enumerate(zip(d["seed"], d["decision_index"])) if mask[i]}
    events = []
    # A proxy for the later G8.5 event analysis: a first above-limit behavior
    # interval following a below-limit interval, with a complete H5 cohort.
    for (seed, decision), i in sorted(keys.items()):
        preceding = keys.get((seed, decision - 1))
        if preceding is None:
            continue
        if (d["future_force_sequence"][i, 0] > F_LIM and
                d["future_force_sequence"][preceding, 0] <= F_LIM):
            events.append((seed, decision))
    if not events:
        raise RuntimeError("No validation high-force onsets for horizon selection")

    rows = []
    for name in ESTIMATORS:
        for horizon in HORIZONS:
            x, y = matrix(d, estimates[name], horizon)
            net, mean, scale, delta = load_model(name, horizon)
            p = predict(net, x, mean, scale)
            row = metrics(y[mask], p[mask], delta, name, horizon, "validation")
            detected, lead = 0, []
            for seed, onset in events:
                eligible = [keys[(seed, k)] for k in range(onset - 5, onset)
                            if (seed, k) in keys and k + horizon - 1 >= onset]
                warned = [onset - int(d["decision_index"][i]) for i in eligible
                          if y[i] > F_LIM and p[i, 2] + delta > F_LIM]
                if warned:
                    detected += 1
                    lead.append(max(warned))
            rows.append({"estimator": name, "horizon": horizon,
                         "eligible_high_force_onsets": len(events),
                         "early_warning_event_count": detected,
                         "early_warning_event_recall": detected / len(events),
                         "median_warning_lead_steps_when_warned": float(np.median(lead)) if lead else None,
                         "max_warning_lead_steps": max(lead) if lead else None,
                         "validation_false_warning_rate": row["Q95_false_warning_rate"],
                         "validation_false_safe_rate": row["Q95_false_safe_rate"],
                         "validation_pinball_N": row["mean_four_quantile_pinball_N"],
                         "validation_raw_Q95_coverage": row["Q95_raw_coverage"],
                         "validation_corrected_Q95_coverage": row["Q95_corrected_coverage"]})
    # Selection is restricted to the native TD-MPC2 plan horizon. H5 stays a
    # retrospective behavior-sequence diagnostic, never a claimed plan score.
    # A horizon earns focus if it gives >=10 percentage points more onset recall
    # than H1 and has <=20% false warnings on below-limit behavior windows.
    focus = {}
    for name in ESTIMATORS:
        base = next(r for r in rows if r["estimator"] == name and r["horizon"] == 1)
        eligible = [r for r in rows if r["estimator"] == name and r["horizon"] in (2, 3)
                    and r["early_warning_event_recall"] >= base["early_warning_event_recall"] + .10
                    and r["validation_false_warning_rate"] is not None
                    and r["validation_false_warning_rate"] <= .20]
        focus[name] = min((r["horizon"] for r in eligible), default=None)
    result = {"status": "VALIDATION_ONLY_SELECTION_BEFORE_DEVELOPMENT_TEST_AND_CONFIRMATION",
              "high_force_onset_definition": "first behavior decision with true next-interval max force above F_lim after a below-limit interval, among common H5-complete rows",
              "warning_definition": "earlier decision within 5 steps; true R_H and calibrated predicted Q95 both above F_lim; onset force must lie within H",
              "action_sequence_caveat": "validation warning uses successive realized behavior actions; these are known in hindsight and cannot be used as a prospective deployable warning at the earlier time",
              "selection_rule": "per estimator, shortest H in {2,3} with >=0.10 absolute increase in onset recall over H1 and <=0.20 false-warning rate; H5 excluded because no native plan",
              "focus_horizon_by_estimator": focus, "rows": rows,
              "event_count": len(events), "event_keys": [{"seed": s, "decision_index": k} for s, k in events]}
    json_new(OUT / "validation_horizon_selection.json", result)
    print(json.dumps({"events": len(events), "focus": focus,
                      "rows": [{k: r[k] for k in ("estimator", "horizon", "early_warning_event_recall", "validation_false_warning_rate")}
                               for r in rows]}, indent=2))


if __name__ == "__main__":
    main()
