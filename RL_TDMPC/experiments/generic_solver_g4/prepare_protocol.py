"""Freeze G4 representations, seeds, and G3 provenance before any G4 fit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
sys.path.insert(0, str(RL / "experiments/generic_solver_g3"))
sys.path.insert(0, str(RL / "experiments/generic_solver_g2"))
from prepare import OUT as G3, G2, G1B, EXPECTED_CHECKPOINT_SHA, EXPECTED_G2_SHA, check_force_source, sha  # noqa: E402
from physical_features import DIMENSION  # noqa: E402

OUT = RL / "logs/generic_solver_phase_g4_state_observability"
SEEDS = [260206, 260207, 260208]
CONFIRMATION_SEEDS = list(range(93000, 93030))


def new_json(name, value):
    path = OUT / name
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")


def main():
    check_force_source()
    probe = json.loads((OUT / "physical_source_probe.json").read_text())
    if probe["status"] != "PASS_PREACTION_PHYSICAL_SOURCE_AND_G2_ALIGNMENT" or len(probe["snapshots"]) != 12:
        raise RuntimeError("Physical source probe did not pass")
    source_manifest = json.loads((G3 / "data_manifest.json").read_text())
    expected_files = {
        "behavior_features.npz": source_manifest["behavior_features_sha256"],
        "development_split.json": None,
        "candidate_action_protocol.json": source_manifest["candidate_action_protocol_sha256"],
        "development_state_selection.json": source_manifest["development_state_selection_sha256"],
        "confirmation_summary.json": source_manifest["confirmation_summary_sha256"],
    }
    for name, digest in expected_files.items():
        if digest is not None and sha(G3 / name) != digest:
            raise RuntimeError(f"G3 source file changed: {name}")
    split = json.loads((G3 / "development_split.json").read_text())
    if [len(split[g]) for g in ("train", "validation", "development_test")] != [140, 30, 30]:
        raise RuntimeError("G3 split changed")
    if sha(G2 / "step_level_force.csv") != EXPECTED_G2_SHA:
        raise RuntimeError("G2 force source changed")
    checkpoint = Path(json.loads((G1B / "selected_checkpoint.json").read_text())["path"])
    if sha(checkpoint) != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("Frozen navigation checkpoint changed")
    if DIMENSION != 38:
        raise RuntimeError("Privileged feature dimension changed")
    g3_status = json.loads((G3 / "development_collection_status.json").read_text())
    if g3_status["status"] != "COMPLETE_ALL_800_STATES_AND_G2_PARENT_REPRODUCED" or g3_status["parent_steps_compared"] != 15500:
        raise RuntimeError("G3 development collection is not complete")
    OUT.mkdir(parents=True, exist_ok=True)
    new_json("g3_data_audit.json", {
        "status": "FROZEN_G3_DEVELOPMENT_SOURCE_VERIFIED", "g3_data_manifest_sha256": sha(G3 / "data_manifest.json"),
        "g3_behavior_features_sha256": sha(G3 / "behavior_features.npz"),
        "g3_split_sha256": sha(G3 / "development_split.json"),
        "g3_state_selection_sha256": sha(G3 / "development_state_selection.json"),
        "g3_candidate_protocol_sha256": sha(G3 / "candidate_action_protocol.json"),
        "g2_step_force_sha256": EXPECTED_G2_SHA, "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "source_behavior_rows": 15500, "source_selected_states": 800, "source_candidate_labels": 16800,
        "g3_confirmation_is_historical_not_untouched_for_g4": True})
    new_json("representation_definitions.json", {
        "status": "FROZEN_BEFORE_G4_TRAINING", "common_decision_filter": "d>=1 for all representations and both supervision modes",
        "target": "force_next_max_N: G2/G3 max 3mm distal force across 22 SOFA substeps immediately after executing candidate a_d",
        "action": "unchanged 2-D normalized G3 candidate action",
        "Z": {"state_dim": 256, "input_dim": 258, "definition": "exact G3 frozen TD-MPC2 latent z_d"},
        "O": {"state_dim": 14, "input_dim": 16, "definition": "exact G2/G3 pre-action policy observation o_d"},
        "H": {"state_dim": 30, "input_dim": 32, "definition": "[o_(d-1), o_d, a_(d-1)] from same episode; exactly two observations; excludes d=0"},
        "P": {"state_dim": 38, "input_dim": 40, "definition": "diagnostic-only pre-action SOFA physical_features.extract()"},
        "no_future_inputs": True, "same_candidate_states_actions_labels_for_all_representations": True})
    new_json("privileged_feature_definition.json", {
        "status": "FROZEN_BEFORE_G4_TRAINING_AND_CONFIRMATION", "dimension": 38,
        "extractor_sha256": sha(HERE / "physical_features.py"), "observation_instant": "after previous control interval, before agent.act or candidate env.step(a_d)",
        "source": "Generic CollisionDOFs and current Generic constraintForces at decision instant; no fork outcome", "dtype": "float32 after finite double-precision source checks",
        "coordinate_frame": "global SOFA scene for xyz vectors; relative distal-node offsets to tip; SOFA scene length in mm and velocity in scene length per second",
        "indices": {"0:3": "absolute distal collision-tip xyz position (mm)",
                    "3:15": "four preceding collision-node xyz positions minus tip position, ordered proximal-to-distal, flattened (mm)",
                    "15:30": "last five collision-node xyz velocities, same node order, flattened (scene mm/s)",
                    "30:33": "current distal contact-force vector from prior interval final substep (nominal N)",
                    "33": "current distal contact-force magnitude from prior interval final substep (nominal N)",
                    "34": "current distal contact-group count from prior interval final substep",
                    "35": "current guidewire insertion length (mm)",
                    "36": "sin(current guidewire rotation)", "37": "cos(current guidewire rotation)"},
        "force_formula": "G2 reconstruct_distal_contact_force using current Generic multiplier/Jacobian/collision geometry, 3mm tip region, vector sum before norm, divided by 0.006 s",
        "alignment_assertion": "P[33] equals G2 prior row distal_contact_force_final_N at d>=1; P never uses current target force_next_max_N",
        "omitted_features": {"vessel_wall_proximity": "No robust, audited fixed-dimensional triangle-distance extractor established in this phase",
                             "contact_normal_set": "Variable cardinality and topology; no deterministic compact summary audited",
                             "all_SOFA_state": "Not compact/interpretable and risks leakage or overfitting"},
        "normalization": "per-representation input mean/std fit on TRAIN behavior rows only; shared exactly between suffix0 and suffix1; zero-std fallback 1",
        "diagnostic_only_not_assumed_deployable": True})
    new_json("training_seed_manifest.json", {
        "status": "FROZEN_BEFORE_G4_TRAINING", "training_seeds": SEEDS, "same_seeds_for_all_eight_configurations": True,
        "architecture": "input_dim-64-ReLU-64-ReLU-1", "loss": "raw nominal-N MSE", "optimizer": "Adam", "learning_rate": .001,
        "weight_decay": 0.0, "batch_size": 512, "max_validation_rounds": 60, "early_stopping_patience": 8,
        "minimum_validation_MSE_improvement": .0001, "checkpoint_selection": "lowest VALIDATION behavior MSE per training seed/configuration",
        "matched_budget": "each model draws the common d>=1 TRAIN behavior row count per validation round without replacement; same optimizer steps per round",
        "normalization": "TRAIN behavior only, same suffix0/1 for each representation; no target normalization",
        "no_test_or_confirmation_labels_for_selection": True})
    new_json("confirmation_seed_protocol.json", {
        "status": "PREDECLARED_BEFORE_G4_TRAINING", "seeds": CONFIRMATION_SEEDS,
        "scene_lifecycle": "one reused Generic scene for ordered 93000-93029 resets, then deterministic full replay for candidate forks",
        "state_selection": "same frozen G3 four-state progress/prior-force rule", "candidate_action_protocol": "exact frozen G3 21-action rule and confirmation RNG group offset 400000, applied to new G4 seeds",
        "states_per_episode": 4, "candidates_per_state": 21, "expected_states": 120, "expected_labels": 2520,
        "not_open_until": "all G4 representations, features, seeds, model settings, selected checkpoints, development comparison metrics, and pre_confirmation_freeze.json frozen"})
    new_json("metadata.json", {"phase": "G4", "status": "REPRESENTATIONS_PREDECLARED_PHYSICAL_ALIGNMENT_PENDING",
                               "branch": "feature/generic-contact-force-safety", "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
                               "g3_source": str(G3), "python": sys.executable, "torch": torch.__version__,
                               "torch_cuda": torch.version.cuda, "confirmation_seed_range": [93000, 93029],
                               "generic_reference_nominal_N": json.loads((G2 / "force_distribution_summary.json").read_text())["all_rl_steps"]})
    print(f"G4 protocol frozen in {OUT}", flush=True)


if __name__ == "__main__":
    main()
