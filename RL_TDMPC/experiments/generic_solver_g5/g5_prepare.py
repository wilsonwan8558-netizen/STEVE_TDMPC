"""Freeze exact G4 provenance and G5 feature semantics before fitting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
sys.path.insert(0, str(RL / "experiments/generic_solver_g4"))
sys.path.insert(0, str(RL / "experiments/generic_solver_g3"))
from g4_freeze import verify as verify_g4  # noqa: E402
from g4_data import source, physical_lookup, candidate_states  # noqa: E402
from prepare import sha, G2, G1B, EXPECTED_CHECKPOINT_SHA, EXPECTED_G2_SHA, check_force_source  # noqa: E402

OUT = RL / "logs/generic_solver_phase_g5_feature_attribution"
G4 = RL / "logs/generic_solver_phase_g4_state_observability"
G3 = RL / "logs/generic_solver_phase_g3_force_predictor"
SEEDS = [260206, 260207, 260208]
CONFIRM = list(range(94000, 94030))
GROUPS = {"K": list(range(0, 3)) + list(range(15, 30)) + list(range(35, 38)),
          "D": list(range(3, 15)), "C": list(range(30, 35))}
PRIMARY = ["Z", "P", "ZP", "ZK", "ZD", "ZC", "ZP_noK", "ZP_noD", "ZP_noC", "PK", "PD", "PKD"]


def new_json(name, data):
    path = OUT / name
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    check_force_source()
    g4_freeze = verify_g4()
    if sha(G2 / "step_level_force.csv") != EXPECTED_G2_SHA:
        raise RuntimeError("G2 force target changed")
    checkpoint = Path(json.loads((G1B / "selected_checkpoint.json").read_text())["path"])
    if sha(checkpoint) != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("Frozen navigation checkpoint changed")
    definition = json.loads((G4 / "privileged_feature_definition.json").read_text())
    if definition["dimension"] != 38 or sha(RL / "experiments/generic_solver_g4/physical_features.py") != definition["extractor_sha256"]:
        raise RuntimeError("Exact G4 P extractor unavailable")
    alignment = json.loads((G4 / "physical_state_alignment_audit.json").read_text())
    if alignment["status"] != "PASS_G3_G2_PARENT_CANDIDATE_AND_PREACTION_PHYSICAL_ALIGNMENT" or alignment["candidate_forks"] != 16800:
        raise RuntimeError("G4 P alignment not valid")
    if sorted(np.concatenate([GROUPS[g] for g in GROUPS]).tolist()) != list(range(38)):
        raise RuntimeError("P semantic grouping is not a partition of 38 features")
    arrays, split, index = source()
    physical = physical_lookup()
    if len(index) != 15500 or len(physical) != 15300 or [len(split[g]) for g in ("train", "validation", "development_test")] != [140, 30, 30]:
        raise RuntimeError("G4/G3 source counts changed")
    if [len(candidate_states(g, split)) for g in ("train", "validation", "development_test")] != [560, 120, 120]:
        raise RuntimeError("G3 candidate states changed")
    norm = json.loads((G4 / "models/P/normalization.json").read_text())
    mean, std = np.asarray(norm["input_mean"], dtype=float), np.asarray(norm["input_std"], dtype=float)
    if mean.shape != (40,) or std.shape != (40,) or np.any(std <= 0):
        raise RuntimeError("G4 P normalization missing")
    # Every member is a contiguous vector/scalar from the *existing* G4 P.
    specs = [
        ("distal_tip_xyz", 0, 3, "K", "B", "CollisionModel.CollisionDOFs.position.value[-1]", "Global SOFA mm", "3D tip position; not present in 2D policy tracking, plausibly image-trackable in principle"),
        ("distal_relative_node_xyz", 3, 15, "D", "B", "CollisionModel.CollisionDOFs.position.value[-5:-1] minus tip", "Tip-relative SOFA mm", "Four proximal-to-distal collision-node offsets; a 3D shape estimate would require sensing/imaging"),
        ("distal_node_velocity_xyz", 15, 30, "K", "B", "CollisionModel.CollisionDOFs.velocity.value[-5:]", "Global scene mm/s", "Five distal velocities; plausibly estimable from tracked motion, not currently measured"),
        ("current_distal_force_vector", 30, 33, "C", "C", "Generic.constraintForces plus CollisionDOFs.constraint, position and edge geometry", "Global SOFA nominal N", "Exact G4 solver-reconstructed pre-action contact reaction; a physical force sensor would be a different, unvalidated surrogate"),
        ("current_distal_force_magnitude", 33, 34, "C", "C", "Norm of exact current distal force vector", "Nominal N", "Exact preceding-interval final-substep solver contact force; not the future target"),
        ("current_distal_contact_group_count", 34, 35, "C", "C", "Generic contact reconstruction / current constraint manifold", "Count", "Constraint-manifold-derived contact count; no direct current controller measurement"),
        ("insertion_length", 35, 36, "K", "A", "simulation.inserted_lengths[0] / intervention.device_lengths_inserted[0]", "SOFA mm", "Current controller/device insertion state, before requested action"),
        ("rotation_sin_cos", 36, 38, "K", "A", "sin/cos(simulation.rotations[0]) / existing Rotations observation", "Dimensionless, global device rotation", "Already included in the 14-D policy observation; this is an exact duplicate channel"),
    ]
    inventory = []
    for name, start, end, group, deploy, source_field, frame, rationale in specs:
        inventory.append({"feature_name": name, "indices": list(range(start, end)), "dimension": end-start,
                          "semantic_group": group, "deployability_class": deploy, "semantic_meaning_and_rationale": rationale,
                          "source_object_or_field": source_field, "coordinate_frame": frame,
                          "timing": "after prior control interval and before a_d, d>=1", "candidate_action_dependent": False,
                          "available_before_a_d": True, "future_or_post_action_information": False,
                          "g4_train_behavior_input_mean": mean[start:end].tolist(),
                          "g4_train_behavior_input_std": std[start:end].tolist()})
    OUT.mkdir(parents=True, exist_ok=True)
    new_json("privileged_feature_inventory.json", {"status": "EXACT_G4_P_RECOVERED_BEFORE_G5_FITTING", "source_definition_sha256": sha(G4 / "privileged_feature_definition.json"),
        "source_extractor_sha256": definition["extractor_sha256"], "dimension": 38, "dtype": "float32", "normalization": "train behavior mean/std per input coordinate, zero-std fallback 1; candidate action appended last", "features": inventory,
        "no_future_target_or_candidate_outcome": True})
    new_json("semantic_feature_groups.json", {"status": "FROZEN_BEFORE_MODEL_PERFORMANCE", "groups": {
        "K": {"name": "kinematics_pose_and_device_state", "indices": GROUPS["K"], "dimension": 21},
        "D": {"name": "distal_deformation_shape", "indices": GROUPS["D"], "dimension": 12},
        "C": {"name": "current_solver_contact_state", "indices": GROUPS["C"], "dimension": 5}},
        "omitted_wall_relation_group": "G4 P had no wall proximity or vessel-relative geometry; G5 cannot attribute it without adding information", "complete_nonoverlapping_partition": True})
    new_json("deployability_classification.json", {"status": "FROZEN_BEFORE_MODEL_PERFORMANCE", "classes": {
        "A": "already available from controller/tracking/command state, no Generic solver internals",
        "B": "physically measurable/estimable in principle, not currently measured or available in the policy input",
        "C": "exact G4 feature depends on Generic constraint multipliers/manifold internals; not directly deployable"},
        "per_feature": [{"feature_name": item["feature_name"], "indices": item["indices"], "class": item["deployability_class"],
                          "rationale": item["semantic_meaning_and_rationale"]} for item in inventory],
        "P_deployable_indices": sorted(GROUPS["K"] + GROUPS["D"]), "P_deployable_dimension": 33,
        "P_simulator_internal_indices": GROUPS["C"], "P_simulator_internal_dimension": 5,
        "B_is_not_currently_measured_experimentally": True})
    new_json("g4_reproduction_audit.json", {"status": "G4_SOURCE_AND_ALIGNMENT_IDENTITIES_PASS_MODEL_REPRODUCTION_PENDING",
        "g4_pre_confirmation_freeze_sha256": sha(G4 / "pre_confirmation_freeze.json"),
        "g4_frozen_file_count": len(g4_freeze["file_sha256"]), "g4_P_definition_sha256": sha(G4 / "privileged_feature_definition.json"),
        "g4_P_normalization_sha256": sha(G4 / "models/P/normalization.json"),
        "g4_alignment_sha256": sha(G4 / "physical_state_alignment_audit.json"),
        "source_parent_steps": 15500, "source_P_rows": 15300, "source_selected_states": 800, "source_candidate_labels": 16800,
        "all_g4_alignment_max_abs_differences": alignment["max_abs_difference"], "reference_models_to_retrain_and_compare": ["Z", "P"]})
    new_json("data_manifest.json", {"status": "FROZEN_G4_DEVELOPMENT_DATA", "G3_split_sha256": sha(G3 / "development_split.json"),
        "G3_behavior_features_sha256": sha(G3 / "behavior_features.npz"),
        "G3_candidate_all_files_sha256": json.loads((G3 / "data_manifest.json").read_text())["development_candidate_all_files_sha256"],
        "G4_physical_alignment_sha256": sha(G4 / "physical_state_alignment_audit.json"),
        "G4_pre_confirmation_freeze_sha256": sha(G4 / "pre_confirmation_freeze.json"),
        "split_episode_counts": {g: len(split[g]) for g in ("train", "validation", "development_test")},
        "behavior_rows_d_ge_1": {g: int(sum(int(seed) in split[g] and int(d)>=1 for seed,d in zip(arrays["seed"],arrays["decision_index"]))) for g in ("train", "validation", "development_test")},
        "candidate_states": {"train": 560, "validation": 120, "development_test": 120}, "candidate_labels": {"train": 11760, "validation": 2520, "development_test": 2520},
        "G4_confirmation_93000_93029_is_historical_for_G5": True})
    settings = json.loads((G4 / "training_seed_manifest.json").read_text())
    if settings["training_seeds"] != SEEDS:
        raise RuntimeError("G4 training seeds changed")
    new_json("training_seed_manifest.json", {**settings, "phase": "G5", "primary_supervision": "behavior-only",
        "primary_model_names": PRIMARY, "secondary_model_names": ["Z1", "P1", "Pmin1"],
        "validation_subset_candidates": ["K", "D", "KD"],
        "validation_subset_selection_model": "Z+subset, three-seed mean metrics; lexicographic Q95 false-safe rate, candidate RMSE, negative pairwise accuracy, safe-choice regret, group count, dimension",
        "selection_uses_development_test": False, "selection_uses_confirmation": False,
        "reference_reproduction": "retrain Z and P with exact G4 behavior-only fit code and compare validation selection to frozen G4 Z0/P0"})
    new_json("confirmation_seed_protocol.json", {"status": "PREDECLARED_BEFORE_G5_MODEL_PERFORMANCE", "seeds": CONFIRM,
        "scene_lifecycle": "one reused Generic scene for ordered 94000-94029 behavior; fresh scene ordered full replay for candidate forks",
        "state_selection": "exact G3/G4 four-state progress/prior-force rule", "candidate_action_protocol": "exact G3/G4 21-action rule, confirmation RNG group offset 400000 with state IDs 0..119",
        "states_per_episode": 4, "candidates_per_state": 21, "expected_states": 120, "expected_labels": 2520,
        "not_open_until": "P_min, all model checkpoints and development metrics frozen in pre_confirmation_freeze.json"})
    new_json("metadata.json", {"phase": "G5", "status": "FEATURE_GROUPS_AND_DEPLOYABILITY_PREDECLARED",
        "branch": "feature/generic-contact-force-safety", "python": sys.executable, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "navigation_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA, "force_source_G2_sha256": EXPECTED_G2_SHA,
        "confirmation_seed_range": [94000, 94029]})
    print(f"G5 exact P inventory, grouping, deployability and seeds frozen in {OUT}", flush=True)


if __name__ == "__main__":
    main()
