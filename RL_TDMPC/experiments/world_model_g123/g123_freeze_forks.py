"""Freeze G12.3 paired-fork states without inspecting gated SOFA outcomes."""

from __future__ import annotations

import csv
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "logs/world_model_force_aware_planning_g123"
ACTIVE = {"LIGHT-GATE", "MODERATE-GATE", "SEVERE-GATE"}


def main() -> None:
    with (OUT / "archived_state_masking_diagnostics.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    active = [r for r in rows if r["gate_class"] in ACTIVE]
    inactive = [r for r in rows if r["gate_class"] == "FORCE-INACTIVE"]
    empty = [r for r in rows if r["gate_class"] == "GATE-EMPTY"]
    if not (len(active) >= 12 and len(inactive) >= 8 and len(empty) >= 2):
        raise RuntimeError("Insufficient class-balanced archived states")
    # Prefer baseline-elevated TRUE force in the unchanged-action controls.
    # Active states are included exhaustively; no gated SOFA outcome is known.
    key = lambda r: (-float(r["baseline_archived_true_next_force_N"]), int(r["seed"]), int(r["decision_index"]))
    selected = active + sorted(inactive, key=key)[:8] + sorted(empty, key=key)[:2]
    selected.sort(key=lambda r: (int(r["seed"]), int(r["decision_index"])))
    if len(selected) != 22 or len({r["state_id"] for r in selected}) != 22:
        raise RuntimeError("Frozen G12.3 fork state count/identity changed")
    records = [{
        "state_id": r["state_id"], "seed": int(r["seed"]),
        "decision_index": int(r["decision_index"]), "gate_class": r["gate_class"],
        "baseline_action": json.loads(r["baseline_action_json"]),
        "masked_action": json.loads(r["gated_action_json"]),
        "baseline_archived_true_next_force_N": float(r["baseline_archived_true_next_force_N"]),
        "baseline_predicted_calibrated_force_N": float(r["baseline_predicted_calibrated_force_N"]),
        "masked_predicted_calibrated_force_N": float(r["gated_predicted_calibrated_force_N"]),
        "retained_weight_mass": float(r["retained_weight_mass"]),
        "baseline_ESS": float(r["baseline_ESS"]),
        "gated_ESS": float(r["gated_ESS"]),
        "safe_candidate_fraction": float(r["safe_candidate_fraction"]),
        "action_L2_difference": float(r["action_L2_difference"]),
    } for r in selected]
    with (OUT / "fork_selection_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump({
            "status": "FROZEN_BEFORE_ANY_G123_SOFA_FORK",
            "selection_rule": "all 12 archived positive-gate states, plus eight force-inactive and two weight-empty controls ranked by descending archived baseline true next force; no gated SOFA outcome used",
            "state_count": len(records),
            "class_counts": {label: sum(r["gate_class"] == label for r in records)
                             for label in ("FORCE-INACTIVE", "LIGHT-GATE", "MODERATE-GATE", "SEVERE-GATE", "GATE-EMPTY")},
            "states": records,
        }, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print("G123_FORK_SELECTION_FROZEN", len(records),
          [(r["state_id"], r["gate_class"]) for r in records], flush=True)


if __name__ == "__main__":
    main()
