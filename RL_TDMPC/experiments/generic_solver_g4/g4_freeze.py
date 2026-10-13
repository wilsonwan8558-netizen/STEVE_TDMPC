"""Immutable G4 pre-confirmation identity manifest and verifier."""

from __future__ import annotations

import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
sys.path.insert(0, str(RL / "experiments/generic_solver_g3"))
from prepare import sha, G2, OUT as G3, G1B, CONFIG, EXPECTED_CHECKPOINT_SHA, EXPECTED_G2_SHA  # noqa: E402
from g4_data import OUT, REPRESENTATIONS  # noqa: E402
from g4_train import SEEDS  # noqa: E402

SEEDS_CONFIRM = list(range(93000, 93030))
MANIFEST = OUT / "pre_confirmation_freeze.json"


def files_to_hash():
    files = [HERE / name for name in (
        "physical_features.py", "prepare_protocol.py", "align_physical.py", "g4_data.py",
        "g4_train.py", "g4_evaluate.py", "g4_aliasing.py", "g4_freeze.py", "g4_confirm.py")]
    files += [OUT / name for name in (
        "g3_data_audit.json", "representation_definitions.json", "privileged_feature_definition.json",
        "physical_source_probe.json", "physical_state_alignment_audit.json", "training_seed_manifest.json", "training_runtime.json",
        "confirmation_seed_protocol.json", "metadata.json", "training_complete.json",
        "development_behavior_metrics.csv", "development_candidate_metrics.csv", "development_boundary_metrics.csv",
        "development_state_metrics.csv", "development_evaluation_summary.json", "nearest_neighbor_aliasing.csv",
        "aliasing_summary.json", "bootstrap_summary.json", "development_three_seed_summary.json")]
    files += [G3 / name for name in ("behavior_features.npz", "development_split.json", "development_state_selection.json",
                                       "candidate_action_protocol.json", "data_manifest.json")]
    files += [path for group in ("train", "validation", "development_test")
              for path in sorted((G3 / "development_candidates" / group).glob("state_*.json"))]
    if len([path for path in files if "development_candidates" in str(path)]) != 800:
        raise RuntimeError("Frozen G3 development candidate files incomplete")
    files += [G2 / "step_level_force.csv", G2 / "force_distribution_summary.json"]
    files += [RL / "experiments/generic_solver_g2" / name for name in
              ("force_reconstruction.py", "substep_monitor.py", "collect.py")]
    files += [RL / "experiments/generic_solver_g3" / name for name in
              ("protocol.py", "restore_audit.py", "g3_evaluate.py", "prepare.py")]
    files += [RL.parent / "eve/intervention/simulation/sofabeamadapter.py", CONFIG]
    for representation in REPRESENTATIONS:
        files.append(OUT / "models" / representation / "normalization.json")
        for suffix in (0, 1):
            for seed in SEEDS:
                folder = OUT / "models" / f"{representation}{suffix}" / f"seed_{seed}"
                files.extend([folder / "model.pt", folder / "selection.json", folder / "training_history.json"])
    files += sorted((OUT / "development_physical_by_episode").glob("seed_*.npz"))
    if len(list((OUT / "development_physical_by_episode").glob("seed_*.npz"))) != 200:
        raise RuntimeError("Physical episode files incomplete")
    checkpoint = Path(json.loads((G1B / "selected_checkpoint.json").read_text())["path"])
    files.append(checkpoint)
    return files


def expected_identity():
    files = files_to_hash()
    return {str(path.relative_to(RL)) if path.is_relative_to(RL) else str(path): sha(path) for path in files}


def verify():
    manifest = json.loads(MANIFEST.read_text())
    if manifest["status"] != "ALL_G4_DEVELOPMENT_MODELS_AND_METRICS_FROZEN_BEFORE_CONFIRMATION":
        raise RuntimeError("Pre-confirmation freeze incomplete")
    if manifest["confirmation_seeds"] != SEEDS_CONFIRM or manifest["training_seeds"] != list(SEEDS):
        raise RuntimeError("Frozen seeds changed")
    if manifest["file_sha256"] != expected_identity():
        raise RuntimeError("Frozen G4 code, model, feature, data or metric identity changed")
    return manifest


def main():
    if MANIFEST.exists():
        raise FileExistsError(MANIFEST)
    for name, expected in (("physical_state_alignment_audit.json", "PASS_G3_G2_PARENT_CANDIDATE_AND_PREACTION_PHYSICAL_ALIGNMENT"),
                           ("training_complete.json", "COMPLETE_ALL_24_VALIDATION_SELECTED_CHECKPOINTS"),
                           ("development_evaluation_summary.json", "COMPLETE_ALL_EIGHT_CELLS_THREE_SEEDS")):
        if json.loads((OUT / name).read_text())["status"] != expected:
            raise RuntimeError(f"G4 freeze prerequisite failed: {name}")
    if any(OUT.glob("confirmation_behavior_*")) or (OUT / "confirmation_candidate_results.csv").exists():
        raise RuntimeError("Confirmation already opened before freeze")
    protocol = json.loads((OUT / "confirmation_seed_protocol.json").read_text())
    if protocol["seeds"] != SEEDS_CONFIRM or protocol["expected_labels"] != 2520:
        raise RuntimeError("Confirmation seed/action protocol changed")
    hashes = expected_identity()
    if hashes[str((G2 / "step_level_force.csv").relative_to(RL))] != EXPECTED_G2_SHA:
        raise RuntimeError("G2 force identity changed")
    checkpoint = Path(json.loads((G1B / "selected_checkpoint.json").read_text())["path"])
    checkpoint_key = str(checkpoint.relative_to(RL)) if checkpoint.is_relative_to(RL) else str(checkpoint)
    if hashes[checkpoint_key] != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("Frozen TD-MPC2 checkpoint changed")
    manifest = {"status": "ALL_G4_DEVELOPMENT_MODELS_AND_METRICS_FROZEN_BEFORE_CONFIRMATION",
                "confirmation_seeds": SEEDS_CONFIRM, "training_seeds": list(SEEDS),
                "frozen_model_count": 24, "frozen_representations": list(REPRESENTATIONS),
                "all_metrics_predefined_and_completed_before_confirmation": True,
                "no_confirmation_data_opened": True, "file_sha256": hashes}
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    verify()
    print(f"G4 pre-confirmation freeze PASS: {len(hashes)} identity hashes", flush=True)


if __name__ == "__main__":
    main()
