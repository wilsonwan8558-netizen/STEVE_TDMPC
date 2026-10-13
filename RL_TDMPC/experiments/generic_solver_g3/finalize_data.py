"""Audit all completed development state files and seal the data manifest."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from prepare import OUT, sha


def main():
    status = json.loads((OUT / "development_collection_status.json").read_text())
    if status["status"] != "COMPLETE_ALL_800_STATES_AND_G2_PARENT_REPRODUCED" or status["parent_steps_compared"] != 15500:
        raise RuntimeError("Development collection not sealed")
    split = json.loads((OUT / "development_split.json").read_text())
    selection = json.loads((OUT / "development_state_selection.json").read_text())["states"]
    digest = hashlib.sha256()
    group_counts = {}
    label_count = 0
    for group, expected in (("train", 560), ("validation", 120), ("development_test", 120)):
        paths = sorted((OUT / "development_candidates" / group).glob("state_*.json"))
        if len(paths) != expected:
            raise RuntimeError(f"{group}: expected {expected} states, found {len(paths)}")
        group_counts[group] = {"states": len(paths), "labels": 0}
        for path, selected in zip(paths, selection[group]):
            state = json.loads(path.read_text())
            if state["state_id"] != selected["state_id"] or state["seed"] != selected["seed"] or state["decision_index"] != selected["decision_index"]:
                raise RuntimeError(f"Selected state mismatches saved state {path}")
            if state["seed"] not in split[group] or len(state["candidates"]) != 21:
                raise RuntimeError(f"Candidate state invalid {path}")
            if any(not np.isfinite(candidate["force_next_max_N"]) or candidate["force_next_max_N"] < 0 for candidate in state["candidates"]):
                raise RuntimeError(f"Nonfinite or negative label {path}")
            digest.update(str(path.relative_to(OUT)).encode())
            digest.update(sha(path).encode())
            group_counts[group]["labels"] += 21
            label_count += 21
    if label_count != 16800:
        raise RuntimeError("Expected 16,800 candidate labels")
    path = OUT / "data_manifest.json"
    manifest = json.loads(path.read_text())
    if "development_candidate_counts" in manifest:
        raise RuntimeError("Development candidate manifest already finalized")
    manifest.update({"development_candidate_counts": group_counts,
                     "development_candidate_all_files_sha256": digest.hexdigest(),
                     "development_state_selection_sha256": sha(OUT / "development_state_selection.json"),
                     "candidate_action_protocol_sha256": sha(OUT / "candidate_action_protocol.json"),
                     "counterfactual_restore_audit_sha256": sha(OUT / "counterfactual_restore_audit.json"),
                     "development_collection_status_sha256": sha(OUT / "development_collection_status.json"),
                     "parent_steps_compared_to_G2": 15500, "development_one_step_candidate_evaluations": 16800,
                     "test_or_confirmation_labels_used_for_training": False})
    partial = path.with_suffix(".json.partial")
    partial.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)+"\n")
    os.replace(partial, path)
    print(f"Sealed G3 data manifest: {group_counts}", flush=True)


if __name__ == "__main__":
    main()
