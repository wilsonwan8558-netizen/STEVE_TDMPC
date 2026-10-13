"""Summarize the predeclared offline NO-GO without opening G10 online seeds."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
for name in ("generic_solver_g10", "generic_solver_g95"):
    sys.path.insert(0, str(RL / "experiments" / name))

from g10_prepare import OUT, F_LIM, gate  # noqa: E402
from g10_offline import manifest  # noqa: E402
from g95_common import json_new, sha  # noqa: E402
from g95_eval import csv_new  # noqa: E402


def output_row(item):
    proposal = item["proposal"]
    nominal, repaired, fallback = (item[key] for key in
                                  ("nominal_true", "repaired_true", "fallback_zero_true"))
    return {"cohort": item["cohort"], "state_id": item["state_id"],
        "seed": item["seed"], "decision_index": item["decision_index"],
        "event_decision_index": item["event_decision_index"],
        "lookback_steps": item["lookback_steps"],
        "nominal_action": json.dumps(proposal["nominal_action"]),
        "fixed_nominal_second_action": json.dumps(proposal["second_action_fixed"]),
        "repair_status": proposal["status"],
        "repaired_action": json.dumps(proposal["action"]),
        "delta_translation": proposal["delta_translation"],
        "delta_rotation": proposal["delta_rotation"],
        "action_deviation": proposal["distance"],
        "nominal_max_corrected_Q95_N": max(proposal["nominal_corrected_Q95_N"]),
        "repaired_max_corrected_Q95_N": proposal["max_corrected_Q95_N"],
        "all_replica_Q95_nominal_N": json.dumps(proposal["nominal_corrected_Q95_N"]),
        "all_replica_Q95_repaired_N": json.dumps(proposal["repaired_corrected_Q95_N"]),
        "nominal_F1_N": nominal["F1_N"], "nominal_F2_N": nominal["F2_N"],
        "nominal_true_R2_N": nominal["R2_N"],
        "repaired_F1_N": repaired["F1_N"] if repaired else None,
        "repaired_F2_N": repaired["F2_N"] if repaired else None,
        "repaired_true_R2_N": repaired["R2_N"] if repaired else None,
        "repaired_true_safe": repaired["true_safe"] if repaired else None,
        "zero_fallback_F1_N": fallback["F1_N"],
        "zero_fallback_F2_N": fallback["F2_N"],
        "zero_fallback_true_R2_N": fallback["R2_N"],
        "zero_fallback_true_safe": fallback["true_safe"],
        "predictor_evaluations": proposal["predictor_evaluations"],
        "grid_predicted_safe_count": proposal["grid_predicted_safe_count"],
        "refinement_predictor_evaluations": proposal["refinement_predictor_evaluations"],
        "oracle_recoverable_zero_action_witness": bool(fallback["true_safe"]),
        "no_global_physical_infeasibility_claim": True}


def aggregate(items):
    statuses = [item["proposal"]["status"] for item in items]
    repairs = [item for item in items if item["proposal"]["status"] == "PREDICTED_SAFE_REPAIR"]
    distances = np.asarray([item["proposal"]["distance"] for item in repairs], dtype=np.float64)
    false_safe_repairs = [item for item in repairs if not item["repaired_true"]["true_safe"]]
    no_action = [item for item in items if item["proposal"]["status"] == "INFEASIBLE_ON_PREDECLARED_SEARCH"]
    unsafe_pass = [item for item in items if item["proposal"]["status"] == "NOMINAL_PREDICTED_SAFE_NO_REPAIR"
                   and not item["nominal_true"]["true_safe"]]
    if any(item["nominal_true"]["true_safe"] for item in items):
        raise RuntimeError("G10 predeclared offline state unexpectedly nominal-true-safe")
    return {"nominal_true_unsafe_states": len(items),
        "predicted_safe_repair_found": len(repairs),
        "true_safe_repair_found": len(repairs)-len(false_safe_repairs),
        "false_safe_repair": len(false_safe_repairs),
        "unsafe_nominal_predicted_safe_pass_through": len(unsafe_pass),
        "no_predicted_safe_action_found_on_predeclared_search": len(no_action),
        "zero_action_true_safe_witness_count": sum(i["fallback_zero_true"]["true_safe"] for i in items),
        "oracle_recoverable_but_model_declared_infeasible_lower_bound":
            sum(i["fallback_zero_true"]["true_safe"] for i in no_action),
        "zero_action_true_unsafe_or_censored_count":
            sum(not i["fallback_zero_true"]["true_safe"] for i in items),
        "repair_deviation_mean": float(np.mean(distances)) if len(distances) else None,
        "repair_deviation_median": float(np.median(distances)) if len(distances) else None,
        "repair_deviation_Q90": float(np.quantile(distances, .9)) if len(distances) else None,
        "repair_deviation_max": float(np.max(distances)) if len(distances) else None,
        "repeatable_true_SOFA_parent_audit_required": True,
        "finite_grid_no_safe_not_physical_infeasibility": True}


def analyze():
    gate()
    states = manifest()
    audits = [OUT / "heldout_parent_replay_audit.json",
              OUT / "difficult_event_parent_replay_audit.json"]
    if any(not path.is_file() for path in audits):
        raise RuntimeError("G10 both offline true-SOFA audits required")
    if any(json.loads(path.read_text())["status"][:4] != "PASS" for path in audits):
        raise RuntimeError("G10 offline true-SOFA replay audit failed")
    heldout, events = [], []
    for record in states:
        if record["cohort"] == "heldout_development_unsafe_nominal":
            path = OUT / "heldout_state_results" / f"state_{record['state_id']:04d}.json"
            group = heldout
        else:
            path = (OUT / "difficult_event_state_results" /
                    f"state_{record['state_id']:04d}_minus_{record['lookback_steps']}.json")
            group = events
        item = json.loads(path.read_text())
        if (item["seed"], item["decision_index"], item["state_id"]) != (
                record["seed"], record["decision_index"], record["state_id"]):
            raise RuntimeError("G10 offline true-SOFA result/state manifest misalignment")
        group.append(item)
    if len(heldout) != 18 or len(events) != 22:
        raise RuntimeError("G10 offline repair cohorts incomplete")
    csv_new(OUT / "offline_repair_results.csv", [output_row(i) for i in heldout])
    csv_new(OUT / "difficult_event_repair_results.csv", [output_row(i) for i in events])
    # The explicit NO-GO stop precludes spending further SOFA forks on a
    # closest-action oracle. Zero-action forks give fixed-second-action true-safe
    # *witnesses* and hence lower bounds on model-missed recoverability only.
    oracle_rows = []
    for item in heldout+events:
        fallback = item["fallback_zero_true"]
        oracle_rows.append({"cohort": item["cohort"], "state_id": item["state_id"],
            "seed": item["seed"], "decision_index": item["decision_index"],
            "lookback_steps": item["lookback_steps"],
            "fixed_second_action": json.dumps(item["nominal_pair"][1]),
            "zero_action_true_safe_witness": fallback["true_safe"],
            "zero_action_R2_N": fallback["R2_N"],
            "zero_action_deviation": float(np.linalg.norm(item["nominal_pair"][0])),
            "closest_safe_oracle_action": None, "closest_safe_oracle_deviation": None,
            "repair_excess_deviation_vs_closest_oracle": None,
            "closest_oracle_status": "NOT_RUN_AFTER_CLEAR_OFFLINE_NO_GO_STOP",
            "physical_unrecoverability_claim": False})
    csv_new(OUT / "oracle_repair_results.csv", oracle_rows)
    heldout_summary = aggregate(heldout)
    event_1 = aggregate([i for i in events if i["lookback_steps"] == 1])
    event_2 = aggregate([i for i in events if i["lookback_steps"] == 2])
    all_summary = aggregate(heldout+events)
    summary = {"status": "OFFLINE_TRUE_SOFA_REPAIR_EVALUATED_NO_GO",
        "heldout_unsafe_nominal": heldout_summary,
        "difficult_events_t_star_minus_1": event_1,
        "difficult_events_t_star_minus_2": event_2,
        "combined_descriptive_not_exchangeable_cohorts": all_summary,
        "F_lim_N": F_LIM,
        "F_lim_interpretation": "Generic simulation-derived statistical high-force reference, not a clinical injury threshold",
        "oracle_recoverability_lower_bound_method": "true-SOFA zero current action with unchanged frozen nominal second action; no closest-safe oracle or physical impossibility proof",
        "closest_safe_oracle_suboptimality": "NOT_COMPUTED_AFTER_CLEAR_NO_GO_STOP",
        "training_or_predictor_changes": False,
        "107000_series_accessed": False,
        "parent_replay_audit_sha256": {p.name: sha(p) for p in audits}}
    json_new(OUT / "offline_summary.json", summary)
    no_go = {"decision": "NO_GO_OFFLINE_STOP", "final_classification": "C_FORCE_CONSTRAINED_REPAIR_FAILS",
        "online_closed_loop_justified": False,
        "online_confirmation_107000_107199_opened": False,
        "reasons": [
            "Only one of 18 held-out truly unsafe nominals received a predicted-safe repair; 15 had no predicted-safe action on the predeclared 41x41 grid and 2 unsafe nominals were predicted safe without intervention.",
            "Five of those 15 held-out model-infeasible cases had a true-safe zero-action witness with the same fixed second action.",
            "At t*-2 only one of 11 difficult events received a true-safe repair; at t*-1 the sole proposed repair was false-safe. Five model-infeasible difficult-event states had a true-safe zero-action witness.",
            "Zero action itself was not universally true-safe; no automatic fallback or online deployment is justified.",
            "No arbitrary numeric pass threshold is used: the combined failure modes directly undermine reliable minimal force-constrained repair."],
        "closest_action_oracle_not_run_reason": "Explicit stop on clear offline true-SOFA failure; only fixed-second zero-action true-safe witnesses are reported as recoverability lower bounds, not closest oracle solutions.",
        "future_online_behavior_metrics": "NOT_MEASURED; online seeds unopened",
        "physical_infeasibility_not_proven": True,
        "no_G10_online_or_G11_automatic_launch": True}
    json_new(OUT / "offline_gate_decision.json", no_go)
    print("G10 offline NO-GO: heldout", heldout_summary,
          "t*-1", event_1, "t*-2", event_2, flush=True)


if __name__ == "__main__":
    analyze()
