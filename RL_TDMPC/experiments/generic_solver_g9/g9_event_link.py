"""Separate event-linked warnings from earlier, unrelated high-force windows."""

from __future__ import annotations

import csv
import json

from g9_risk import OUT, F_LIM


def main():
    with (OUT / "lead_time_analysis.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 132:
        raise RuntimeError("G8.5 difficult-event warning table incomplete")
    output = []
    for row in rows:
        onset, horizon = int(row["event_decision_index"]), int(row["horizon"])
        prior = json.loads(row["prior_detail_json"])
        linked = [entry for entry in prior if entry["decision_index"] + horizon - 1 >= onset]
        warnings = [entry for entry in linked if entry["warned_correctly"]]
        lead = max((int(entry["lookback_steps"]) for entry in warnings), default=None)
        output.append({"state_id": int(row["state_id"]), "episode_seed": int(row["episode_seed"]),
                       "event_decision_index": onset, "estimator": row["estimator"], "horizon": horizon,
                       "event_force_N": float(row["event_nominal_force_N"]),
                       "event_force_above_limit": int(float(row["event_nominal_force_N"]) > F_LIM),
                       "event_reachable_prior_decisions": len(linked),
                       "event_linked_correct_warning_count": len(warnings),
                       "event_linked_earliest_lead_steps": lead,
                       "event_linked_earliest_lead_s": None if lead is None else lead * .132,
                       "event_linked_no_warning": int(lead is None),
                       "protocol": "retrospective realized behavior actions; requires event force interval inside H"})
    with (OUT / "lead_time_event_linked.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print("G9 event-linked retrospective warning rows:", len(output))


if __name__ == "__main__":
    main()
