"""G5 immutable pre-confirmation identity manifest and verifier."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from g5_data import OUT, PRIMARY, minimal_groups
from g5_prepare import G4, G3, G2, G1B, RL, SEEDS, CONFIRM, EXPECTED_CHECKPOINT_SHA, EXPECTED_G2_SHA, sha
sys.path.insert(0, str(RL / "experiments/generic_solver_g4"))
from g4_freeze import verify as verify_g4  # noqa: E402

HERE = Path(__file__).resolve().parent
MANIFEST = OUT / "pre_confirmation_freeze.json"
MODEL_NAMES = PRIMARY + ("Z1", "P1", "Pmin1")


def files_to_hash():
    files = [HERE / name for name in ("g5_prepare.py", "g5_data.py", "g5_train.py", "g5_select.py",
                                    "g5_metrics.py", "g5_development.py", "g5_freeze.py", "g5_confirm.py")]
    files += [OUT / name for name in (
        "metadata.json", "g4_reproduction_audit.json", "reference_model_reproduction.json",
        "privileged_feature_inventory.json", "semantic_feature_groups.json", "deployability_classification.json",
        "data_manifest.json", "training_seed_manifest.json", "training_runtime.json", "confirmation_seed_protocol.json",
        "primary_training_complete.json", "secondary_training_complete.json", "deployable_subset_metrics.csv",
        "minimal_subset_selection.json", "development_behavior_metrics.csv", "development_candidate_metrics.csv",
        "development_boundary_metrics.csv", "development_state_metrics.csv", "development_evaluation_summary.json",
        "single_group_addition_metrics.csv", "leave_one_group_out_metrics.csv", "contact_conditioned_metrics.csv",
        "no_contact_emergence_metrics.csv", "feature_group_bootstrap.json", "counterfactual_secondary_metrics.csv")]
    files += [G4 / "pre_confirmation_freeze.json", G3 / "development_split.json", G3 / "behavior_features.npz",
              G2 / "step_level_force.csv", G2 / "force_distribution_summary.json"]
    checkpoint = Path(json.loads((G1B / "selected_checkpoint.json").read_text())["path"])
    files.append(checkpoint)
    for name in MODEL_NAMES:
        for seed in SEEDS:
            folder = OUT / "models" / name / f"seed_{seed}"
            files.extend(folder / leaf for leaf in ("model.pt", "selection.json", "training_history.json"))
    for name in PRIMARY + ("Pmin",):
        if name in ("Z1", "P1"):
            continue
        files.append(OUT / "models" / name / "normalization.json")
    return files


def expected_identity():
    files = files_to_hash()
    if len(set(files)) != len(files):
        raise RuntimeError("Duplicate freeze paths")
    return {str(path.relative_to(RL)) if path.is_relative_to(RL) else str(path): sha(path) for path in files}


def verify():
    verify_g4()
    data = json.loads(MANIFEST.read_text())
    if data["status"] != "ALL_G5_MODELS_FEATURES_AND_DEVELOPMENT_METRICS_FROZEN_BEFORE_CONFIRMATION":
        raise RuntimeError("G5 pre-confirmation status invalid")
    if data["confirmation_seeds"] != CONFIRM or data["training_seeds"] != SEEDS or data["file_sha256"] != expected_identity():
        raise RuntimeError("G5 frozen code/model/data/metric identity changed")
    return data


def main():
    if MANIFEST.exists():
        raise FileExistsError(MANIFEST)
    verify_g4()
    if json.loads((OUT / "development_evaluation_summary.json").read_text())["status"] != "COMPLETE_G5_ALL_PRIMARY_SECONDARY_AND_ALIASES":
        raise RuntimeError("G5 development diagnostics incomplete")
    groups = minimal_groups()
    protocol = json.loads((OUT / "confirmation_seed_protocol.json").read_text())
    if protocol["seeds"] != CONFIRM or protocol["expected_labels"] != 2520:
        raise RuntimeError("G5 untouched confirmation protocol changed")
    if any(OUT.glob("confirmation_behavior_*")) or (OUT / "confirmation_candidates").exists():
        raise RuntimeError("G5 confirmation already opened")
    hashes = expected_identity()
    if hashes[str((G2 / "step_level_force.csv").relative_to(RL))] != EXPECTED_G2_SHA:
        raise RuntimeError("Force source changed")
    checkpoint = Path(json.loads((G1B / "selected_checkpoint.json").read_text())["path"])
    checkpoint_key = str(checkpoint.relative_to(RL)) if checkpoint.is_relative_to(RL) else str(checkpoint)
    if hashes[checkpoint_key] != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("Navigation checkpoint changed")
    MANIFEST.write_text(json.dumps({"status": "ALL_G5_MODELS_FEATURES_AND_DEVELOPMENT_METRICS_FROZEN_BEFORE_CONFIRMATION",
        "confirmation_seeds": CONFIRM, "training_seeds": SEEDS, "frozen_model_count": len(MODEL_NAMES)*len(SEEDS),
        "P_min_groups": groups, "no_confirmation_data_opened": True, "file_sha256": hashes}, indent=2, sort_keys=True, allow_nan=False)+"\n")
    verify()
    print(f"G5 pre-confirmation freeze PASS: {len(hashes)} hashes", flush=True)


if __name__ == "__main__":
    main()
