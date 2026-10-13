"""Seal untouched confirmation label accounting and hashes after evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import os

import numpy as np

from prepare import OUT, sha


def main():
    replay = json.loads((OUT / "confirmation_replay_audit.json").read_text())
    summary = json.loads((OUT / "confirmation_summary.json").read_text())
    seed_manifest = json.loads((OUT / "confirmation_seed_manifest.json").read_text())
    frozen = json.loads((OUT / "frozen_model_manifest.json").read_text())
    if replay["status"] != "PASS" or replay["candidate_states"] != 120 or replay["candidate_evaluations"] != 2520:
        raise RuntimeError("Untouched confirmation replay incomplete")
    if not summary["models_frozen_before_confirmation"] or seed_manifest["frozen_model_manifest_sha256"] != sha(OUT / "frozen_model_manifest.json"):
        raise RuntimeError("Models were not frozen before confirmation")
    if seed_manifest["seeds"] != list(range(92000, 92030)) or frozen["primary_selected_model"] != summary["primary_selected_model"]:
        raise RuntimeError("Confirmation seeds/model selection changed")
    paths = sorted((OUT / "confirmation_candidates").glob("state_*.json"))
    if len(paths) != 120:
        raise RuntimeError("Expected 120 confirmation state files")
    digest = hashlib.sha256()
    seen = set()
    for path in paths:
        state = json.loads(path.read_text())
        key = state["seed"], state["decision_index"]
        if key in seen or state["seed"] not in seed_manifest["seeds"] or len(state["candidates"]) != 21:
            raise RuntimeError(f"Invalid confirmation state {path}")
        seen.add(key)
        if any(not np.isfinite(candidate["force_next_max_N"]) or candidate["force_next_max_N"] < 0 for candidate in state["candidates"]):
            raise RuntimeError(f"Invalid confirmation force {path}")
        digest.update(str(path.relative_to(OUT)).encode())
        digest.update(sha(path).encode())
    with (OUT / "confirmation_candidate_results.csv").open(newline="") as stream:
        prediction_rows = sum(1 for _ in csv.DictReader(stream))
    with (OUT / "confirmation_state_metrics.csv").open(newline="") as stream:
        state_metric_rows = sum(1 for _ in csv.DictReader(stream))
    if prediction_rows != 5040 or state_metric_rows != 240:
        raise RuntimeError("Confirmation model prediction/metric row count mismatch")
    path = OUT / "data_manifest.json"
    manifest = json.loads(path.read_text())
    if "confirmation_candidate_counts" in manifest:
        raise RuntimeError("Confirmation already sealed")
    manifest.update({"confirmation_candidate_counts": {"episodes": 30, "states": 120, "labels": 2520},
                     "confirmation_candidate_all_files_sha256": digest.hexdigest(),
                     "confirmation_seed_manifest_sha256": sha(OUT / "confirmation_seed_manifest.json"),
                     "confirmation_behavior_steps_sha256": sha(OUT / "confirmation_behavior_steps.csv"),
                     "confirmation_replay_audit_sha256": sha(OUT / "confirmation_replay_audit.json"),
                     "confirmation_candidate_results_sha256": sha(OUT / "confirmation_candidate_results.csv"),
                     "confirmation_summary_sha256": sha(OUT / "confirmation_summary.json"),
                     "frozen_model_manifest_sha256": sha(OUT / "frozen_model_manifest.json"),
                     "confirmation_labels_used_for_training_or_selection": False,
                     "total_counterfactual_one_step_evaluations": 16800+2520})
    partial = path.with_suffix(".json.partial")
    partial.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)+"\n")
    os.replace(partial, path)
    print("Sealed untouched confirmation: 30 episodes, 120 states, 2,520 labels", flush=True)


if __name__ == "__main__":
    main()
