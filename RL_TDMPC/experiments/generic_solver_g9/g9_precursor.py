"""Audit whether G8.5 warning lead predates an already-above-limit interval."""

from __future__ import annotations

import csv
import json

from g9_events import archived
from g9_risk import OUT, F_LIM


def main():
    by_seed, events = archived()
    with (OUT / "lead_time_analysis.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    output = []
    for row in rows:
        seed, onset, horizon = int(row["episode_seed"]), int(row["event_decision_index"]), int(row["horizon"])
        previous_action_force = float(by_seed[seed][onset-1]["distal_contact_force_max_N"])
        detail = json.loads(row["prior_detail_json"])
        linked = [r for r in detail if r["decision_index"] + horizon - 1 >= onset]
        clean = [r for r in linked if r["decision_index"] >= 1 and
                 float(by_seed[seed][r["decision_index"]-1]["distal_contact_force_max_N"]) <= F_LIM]
        warning = [r for r in clean if r["warned_correctly"]]
        lead = max((int(r["lookback_steps"]) for r in warning), default=None)
        output.append({"state_id": int(row["state_id"]), "episode_seed": seed,
                       "event_decision_index": onset, "estimator": row["estimator"],
                       "horizon": horizon, "force_at_tstar_minus_1_action_N": previous_action_force,
                       "already_above_limit_at_tstar_minus_1_action": int(previous_action_force > F_LIM),
                       "event_linked_prior_decisions_with_preexisting_force_below_limit": len(clean),
                       "correct_warning_before_preexisting_force_above_limit": len(warning),
                       "earliest_clean_event_linked_warning_lead_steps": lead,
                       "no_clean_event_linked_warning": int(lead is None),
                       "caveat": "uses realized future behavior actions in hindsight, not an available time-t plan"})
    with (OUT / "lead_time_preexisting_force_safe.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print("G9 precursor audit saved; all event-minus-one action forces above limit:",
          all(float(by_seed[int(e["seed"])][int(e["decision_index"])-1]["distal_contact_force_max_N"]) > F_LIM for e in events))


if __name__ == "__main__":
    main()
