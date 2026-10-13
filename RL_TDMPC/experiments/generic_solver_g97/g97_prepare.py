"""Predeclare the privileged-state ceiling experiment without opening 106000 seeds."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
for folder in ("generic_solver_g97", "generic_solver_g96", "generic_solver_g95",
               "generic_solver_g9", "generic_solver_g4"):
    sys.path.insert(0, str(RL / "experiments" / folder))

from g96_common import assert_prepared, G95, G9, F_LIM, json_new, sha  # noqa: E402
from g97_state import definition as raw_definition, RAW_DIM  # noqa: E402

OUT = RL / "logs/generic_solver_phase_g97_privileged_state_ceiling"
CONFIRM = tuple(range(106000, 106030))
SEEDS = (260322, 260323, 260324)
GROUPS = ("train", "validation", "development_test")
PCA_DIM = 256
DEPLOY_STATE_DIM = 294
EXACT_PHYSICAL_DIM = 38
MODEL_INPUT_DIM = DEPLOY_STATE_DIM + EXACT_PHYSICAL_DIM + PCA_DIM + 4


def cuda_gate():
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or
            not torch.cuda.is_available() or torch.cuda.device_count() != 1 or
            "RTX 5090" not in torch.cuda.get_device_name(0)):
        raise RuntimeError("G9.7 requires frozen CUDA_VISIBLE_DEVICES=0 RTX 5090; no CPU fallback")


def source_gate():
    cuda_gate()
    source = assert_prepared()
    previous_freeze = json.loads((RL / "logs/generic_solver_phase_g96_stepwise_h2_force_profile" /
                                  "pre_confirmation_freeze_v2.json").read_text())
    if (previous_freeze["status"] != "G96_PROFILE_SCALAR_AND_EVALUATION_FROZEN_BEFORE_105000_CONFIRMATION" or
            previous_freeze["G95_dataset_sha256"] != sha(G95 / "development_planner_dataset.npz") or
            previous_freeze["controller_checkpoint_sha256"] != source["controller_checkpoint_sha256"]):
        raise RuntimeError("G9.6 final frozen source does not match G9.5 data")
    split = json.loads((G95 / "dataset_split.json").read_text())
    with np.load(G95 / "development_planner_dataset.npz") as z:
        data = {key: z[key].copy() for key in z.files}
    if (len(data["x"]) != 7193 or data["x"].shape[1] != 298 or
            len(set(data["state_global_id"])) != 800 or
            [len(split["selected_development_states"][group]) for group in GROUPS] != [560, 120, 120] or
            not np.array_equal(data["R2_true_N"], np.maximum(data["F1_true_N"], data["F2_true_N"]))):
        raise RuntimeError("G9.7 frozen G9.5 planner state/action/target contract changed")
    return source, split, data


def repeatability_cases(data):
    y = data["R2_true_N"].astype(np.float64)
    rankings = (("low_force", np.argsort(y, kind="stable")),
                ("near_boundary", np.argsort(np.abs(y-F_LIM), kind="stable")),
                ("high_contact", np.argsort(-y, kind="stable")))
    used_states = set()
    result = []
    for category, ordered in rankings:
        index = next(int(i) for i in ordered if int(data["state_global_id"][i]) not in used_states)
        state = int(data["state_global_id"][index])
        used_states.add(state)
        result.append({"category": category, "development_row": index,
            "state_global_id": state, "seed": int(data["seed"][index]),
            "decision_index": int(data["decision_index"][index]),
            "candidate_id": int(data["candidate_id"][index]),
            "action_pair": data["action_pair"][index].tolist(),
            "archived_F1_N": float(data["F1_true_N"][index]),
            "archived_F2_N": float(data["F2_true_N"][index]),
            "archived_R2_N": float(y[index])})
    return result


def main():
    source, split, data = source_gate()
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    json_new(OUT / "metadata.json", {"phase": "G9.7", "branch": "feature/generic-contact-force-safety",
        "status": "PREDECLARED_PRIVILEGED_STATE_CEILING_BEFORE_DEVELOPMENT_REPLAY",
        "python": sys.executable, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "cuda_device_index": 0, "cuda_device_name": torch.cuda.get_device_name(0),
        "Generic_solver_unchanged": True, "controller_checkpoint_sha256": source["controller_checkpoint_sha256"],
        "planner_source_sha256": source["planner_source_sha256"],
        "H": 2, "F_lim_N": F_LIM,
        "force_target": "R2=max(next two complete RL intervals' max-substep ~3 mm distal J_contact^T lambda / 0.006 s nominal N)",
        "confirmation_seeds_declared_unopened": list(CONFIRM),
        "prior_103000_104000_105000_cohorts_excluded_from_G97_final_confirmation": True})
    json_new(OUT / "deployable_state_definition.json", {
        "name": "X_DEPLOY", "state_dimension": DEPLOY_STATE_DIM,
        "features": [
            {"name": "frozen_TD_MPC2_current_observation_latent", "dimension": 256,
             "indices": [0, 256], "source": "G9.5 development_states.current_latent; frozen step_280000 TD-MPC2 encoder of 14D current observation",
             "pre_action": True, "deployable": True},
            {"name": "frozen_E1_estimated_current_physical", "dimension": 38,
             "indices": [256, 294], "source": "G9.5 development_states.E1_current_physical_estimate; G9 E1 mapper using three 14D observations and two preceding 2D actions",
             "pre_action": True, "deployable": True}],
        "candidate_action": {"name": "frozen_planner_action_pair", "dimension": 4,
             "source": "G9.5 exact first two frozen planner candidate actions"},
        "original_G95_G96_model_input_dimension": 298,
        "capacity_matching_only": "M_DEPLOY appends 294 all-zero non-informative channels before the identical final 4 action components; its nonzero information is exactly the frozen 298D G9.5 input",
        "same_model_input_dimension_as_M_PRIV": MODEL_INPUT_DIM,
        "future_or_post_action_information": False})
    definition = raw_definition()
    definition.update({"name": "X_PRIV", "includes_exact_X_DEPLOY_state": True,
        "additional_exact_G4_physical_dimension": EXACT_PHYSICAL_DIM,
        "raw_feature_source_sha256": sha(HERE / "g97_state.py"),
        "raw_feature_dimension": RAW_DIM,
        "raw_extra_adapter": "train-state-only coordinate standardization, then first 256 right singular vectors of the 560 train states; deterministic sign convention; no labels, validation, development-test, or confirmation in fitting",
        "compressed_extra_dimension": PCA_DIM,
        "model_state_dimension": DEPLOY_STATE_DIM + EXACT_PHYSICAL_DIM + PCA_DIM,
        "model_input_dimension_with_unchanged_two_planner_actions": MODEL_INPUT_DIM,
        "both_model_families_same_592D_64x64_network": True,
        "not_a_full_Markov_state": True})
    json_new(OUT / "privileged_state_definition.json", definition)
    json_new(OUT / "privileged_state_audit.json", {
        "status": "PRECOLLECTION_COVERAGE_AUDIT_NOT_A_FULL_MARKOV_STATE",
        "included": [row["name"] for row in definition["features"]],
        "exposed_preaction_solver_fields": ["Generic.constraintForces", "currentNumConstraints",
            "currentNumConstraintGroups", "currentIterations", "currentError",
            "CollisionDOFs.constraint sparse Jacobian", "DOFs.free_position", "DOFs.free_velocity"],
        "not_accessed_or_not_demonstrated": [
            "projected-Gauss-Seidel warm-start vector/history internal to GenericConstraintSolver",
            "contact manifold cache and collision-pipeline/BVH temporal cache",
            "constraint ordering and internal resolution objects beyond exposed Jacobian row IDs",
            "implicit-integrator scratch buffers or any unexposed beam/material internal history",
            "exact triangle-surface signed wall distance (nearest-vertex proxy only)"],
        "static_scene_parameters": "frozen solver, vessel mesh, wire material/rest shape shared across observations; not per-state variable inputs",
        "coverage_claim": "privileged observable simulator state; NOT proven complete transition/Markov state",
        "probe_development_seeds": [90000, 90001],
        "probe_parent_steps_exact_G2": 259,
        "probe_fixed_parent_DOF_shape": [63, 7],
        "probe_fixed_collision_DOF_shape": [76, 3],
        "probe_fixed_vessel_vertex_shape": [636, 3],
        "no_future_force_or_post_candidate_solver_field": True})
    json_new(OUT / "dataset_split.json", {"source_G95_dataset_split_sha256": sha(G95 / "dataset_split.json"),
        "source_G95_planner_dataset_sha256": sha(G95 / "development_planner_dataset.npz"),
        "groups": split["development_groups"],
        "state_counts": {g: len(split["selected_development_states"][g]) for g in GROUPS},
        "candidate_rows_from_same_parent_state_never_cross_splits": True,
        "confirmation_seeds_unopened": list(CONFIRM)})
    json_new(OUT / "dataset_manifest.json", {"status": "EXACT_G95_LABELS_REUSED_PRIVILEGED_PREACTION_REPLAY_PENDING",
        "source_G95_dataset_sha256": sha(G95 / "development_planner_dataset.npz"),
        "source_G95_state_split_sha256": sha(G95 / "dataset_split.json"),
        "source_G95_candidate_protocol_sha256": sha(G95 / "planner_candidate_protocol.json"),
        "source_G2_behavior_sha256": sha(RL / "logs/generic_solver_phase_g2_force_baseline/step_level_force.csv"),
        "source_G9_E1_model_sha256": sha(G9 / "models/E1/model.pt"),
        "state_replay_required": "raw 3582D pre-action physical/solver state was not stored in G9.5; reuse all original planner actions and F1/F2/R2 fork labels without reforking",
        "development_states": 800, "complete_planner_rows": len(data["x"]),
        "censored_planner_forks": 7200-len(data["x"]),
        "confirmation_106000_series_accessed": False})
    json_new(OUT / "repeatability_protocol.json", {
        "status": "THREE_DEVELOPMENT_STATE_ACTION_CASES_FROZEN_BEFORE_REPLAY",
        "method": "Linux fork copy-on-write of exact live parent Generic scene at pre-action decision; child executes exact archived two-action plan and returns F1/F2/R2 plus post-step privileged observable state; repeat independently from same parent memory snapshot",
        "repetitions_per_case": 10,
        "cases": repeatability_cases(data),
        "not_training_data": True,
        "numeric_summary": "per case max and mean absolute deviation and population std of F1/F2/R2 and next-state vectors across 10 repeats",
        "claim_limit": "three representative development cases cannot prove global determinism or complete state restoration"})
    json_new(OUT / "model_protocol.json", {
        "status": "PREDECLARED_BEFORE_PRIVILEGED_DATA_AND_MODEL_FITTING",
        "only_independent_variable": "predictor state information: exact X_DEPLOY vs X_DEPLOY plus current G4 physical and train-only PCA of raw privileged observable simulator state",
        "target": "scalar R2_true=max(F1_true,F2_true)", "H": 2, "quantiles": [0.5, 0.9, 0.95, 0.99],
        "model": "same 592D Linear(592,64)-ReLU-Linear(64,64)-ReLU-Linear(64,4), cumulative softplus noncrossing quantiles",
        "deploy_extra_channels": "294 fixed zeros, no added state information",
        "priv_extra_channels": "38 exact current G4 physical values plus 256 train-state-only PCA components of raw 3582D simulator state",
        "optimizer": "Adam lr=0.001, batch=512, max 60 epochs, patience=8, equal-weight four-quantile pinball",
        "training_seeds": list(SEEDS), "train_rows": 5038, "validation_rows": 1080,
        "training_distribution": "same complete G9.5 planner-conditioned forks for both model families; no additional behavior rows in either family",
        "checkpoint_selection": "minimum validation pinball only",
        "Q95_calibration": "per-replica nonnegative validation 95th percentile residual y-Q95 using higher order statistic, fixed before development-test",
        "decision": "corrected scalar Q95(R2)<=frozen F_lim; no threshold tuning",
        "identifiability_diagnostic": "development-test only; each plan seeks five nearest other-state plans in its own standardized state space among action pairs with Euclidean normalized-action distance <=0.75; compare local true R2 variance and exact-neighbor composition; no fit/selection",
        "confirmation_seeds": list(CONFIRM),
        "confirmation_protocol": "one reused scene for ordered behavior and a second reused scene for ordered parent replay/forks; four frozen G3 selected states per episode; exact validated G9.6 v2 nine-plan canonical executable-action rule; 106000-106029 unopened until all six checkpoints/calibrations/metrics/source hashes are frozen",
        "no_future_or_post_action_predictor_state": True,
        "no_G10_or_online_filter": True})
    print(f"G9.7 predeclared 3582D raw privileged state, 592D capacity-matched model input; 106000-106029 unopened", flush=True)


if __name__ == "__main__":
    main()
