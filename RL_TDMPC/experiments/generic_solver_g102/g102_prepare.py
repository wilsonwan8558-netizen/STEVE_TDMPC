"""Freeze the G10.2 local-translation experiment before new force labels."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
for name in ("generic_solver_g101", "generic_solver_g10", "generic_solver_g95"):
    sys.path.insert(0, str(RL / "experiments" / name))
from g101_prepare import OUT as G101, G10, G95, F_LIM, gate as source_gate  # noqa: E402
from g95_common import json_new, sha  # noqa: E402

OUT = RL / "logs/generic_solver_phase_g102_local_translation_force"
SEEDS = (261021, 261022, 261023)
LOCAL_OFFSETS = np.asarray((-.5, -.2, -.1, -.05, 0., .05, .1, .2, .5), dtype=np.float32)
WITHDRAWAL_VALUES = np.asarray((-1., -.5, 0.), dtype=np.float32)
SEARCH_AXIS = np.linspace(-1., 1., 101, dtype=np.float32)
CONFIRM_SEEDS = tuple(range(108000, 108030))


def verify_source():
    source_gate()
    decision = json.loads((G101 / "offline_gate_decision.json").read_text())
    if (decision["final_classification"] !=
            "B_TRANSLATION_ONLY_PHYSICS_PROMISING_CURRENT_PREDICTOR_NOT_USABLE" or
            decision["online_107000_107199_opened"]):
        raise RuntimeError("G10.1 offline B decision / unopened online seeds changed")
    for name in ("heldout", "events"):
        audit = json.loads((G101 / f"{name}_parent_replay_audit.json").read_text())
        if audit["status"] != "PASS_EXACT_FROZEN_PARENT_AND_G10_NOMINAL" or any(
                x != 0 for x in audit["max_abs_parent_difference"].values()):
            raise RuntimeError("G10.1 reused-scene parent replay not exact")
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available() or
            torch.cuda.device_count() != 1 or "RTX 5090" not in torch.cuda.get_device_name(0)):
        raise RuntimeError("G10.2 requires archived CUDA_VISIBLE_DEVICES=0 RTX 5090; no CPU fallback")


def points(v_nom):
    """Frozen local probes; exact nominal, stop, and withdrawal witnesses."""
    nominal = np.float32(v_nom)
    values = {float(np.float32(np.clip(nominal+d, -1, 1))) for d in LOCAL_OFFSETS}
    values.update(float(v) for v in WITHDRAWAL_VALUES)
    values.add(float(nominal))
    return np.asarray(sorted(values), dtype=np.float32)


def preparation():
    verify_source()
    if OUT.exists():
        raise FileExistsError(OUT)
    source = json.loads((G95 / "dataset_split.json").read_text())
    selected = source["selected_development_states"]
    if {k: len(selected[k]) for k in ("train", "validation", "development_test")} != {
            "train": 560, "validation": 120, "development_test": 120}:
        raise RuntimeError("G9.5 exact state split changed")
    train_seeds = {int(s["seed"]) for s in selected["train"]}
    val_seeds = {int(s["seed"]) for s in selected["validation"]}
    test_seeds = {int(s["seed"]) for s in selected["development_test"]}
    if (len(train_seeds), len(val_seeds), len(test_seeds)) != (140, 30, 30) or (
            train_seeds & val_seeds or train_seeds & test_seeds or val_seeds & test_seeds):
        raise RuntimeError("G10.2 split must be episode-disjoint")
    states = []
    for group in ("train", "validation"):
        for s in selected[group]:
            path = G95 / "development_states" / f"state_{group}_{int(s['state_id']):04d}.json"
            row = json.loads(path.read_text())
            pair = np.asarray(row["plans"][8]["actions"], dtype=np.float32)
            fork = row["plans"][8]["fork"]
            if ((row["group"], row["seed"], row["decision_index"]) !=
                    (group, s["seed"], s["decision_index"]) or
                    not fork["complete_two_intervals"] or pair.shape != (2, 2) or
                    not np.array_equal(pair[0], np.asarray(row["planner_mean_action"], dtype=np.float32)) or
                    len(row["current_latent"]) != 256 or len(row["E1_current_physical_estimate"]) != 38):
                raise RuntimeError(f"G10.2 G9.5 nominal state alignment invalid {path}")
            states.append({"group": group, "state_id": s["state_id"], "seed": s["seed"],
                "decision_index": s["decision_index"], "slot": s["slot"],
                "source_path": str(path), "source_sha256": sha(path),
                "nominal_pair": pair.tolist(), "nominal_true_R2_N": fork["R2_true_N"],
                "v_probe_count": len(points(pair[0, 0])),
                "selection_uses_future_repair_success": False})
    if len(states) != 680:
        raise RuntimeError("G10.2 680 train/validation states required")
    g101_manifest = json.loads((G101 / "state_manifest.json").read_text())
    if g101_manifest["count"] != 40:
        raise RuntimeError("G10.1 exact 40-state development set required")
    dev = []
    for s in g101_manifest["states"]:
        name = (f"state_{s['state_id']:04d}.json" if s["cohort"] == "heldout_development_unsafe_nominal"
                else f"state_{s['state_id']:04d}_minus_{s['lookback_steps']}.json")
        group = "heldout" if s["cohort"] == "heldout_development_unsafe_nominal" else "events"
        path = G101 / f"{group}_state_curves" / name
        item = json.loads(path.read_text())
        if (item["seed"], item["decision_index"], item["G10_result_sha256"]) != (
                s["seed"], s["decision_index"], s["G10_result_sha256"]):
            raise RuntimeError("G10.1 dev state/curve alignment invalid")
        if group == "heldout" and s["seed"] not in test_seeds:
            raise RuntimeError("G10.1 heldout state outside G9.5 devtest episodes")
        if s["seed"] in train_seeds | val_seeds:
            raise RuntimeError("G10.1 development leakage into train/validation")
        dev.append({"cohort": s["cohort"], "state_id": s["state_id"],
            "seed": s["seed"], "decision_index": s["decision_index"],
            "lookback_steps": s.get("lookback_steps"), "curve_path": str(path),
            "curve_sha256": sha(path), "nominal_pair": item["nominal_pair"]})
    OUT.mkdir(parents=True)
    for name in ("train", "validation"):
        (OUT / f"{name}_local_curves").mkdir()
    (OUT / "development_event_features").mkdir()
    (OUT / "models").mkdir()
    json_new(OUT / "metadata.json", {"phase": "G10.2", "branch": "feature/generic-contact-force-safety",
        "status": "LOCAL_PROTOCOL_PREDECLARED_BEFORE_NEW_LABELS", "python": sys.executable,
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
        "cuda_device": torch.cuda.get_device_name(0), "F_lim_N": F_LIM,
        "F_lim_not_clinical_injury_threshold": True,
        "G101_report_sha256": sha(G101 / "PHASE_G101_REPORT.md"),
        "G101_comparison_sha256": sha(G101 / "G10_vs_G101_comparison.json"),
        "G95_split_sha256": sha(G95 / "dataset_split.json"),
        "controller_checkpoint_sha256": json.loads((G101 / "metadata.json").read_text())["controller_checkpoint_sha256"],
        "confirmation_108000_108029_unopened": True,
        "online_107000_107199_unopened": True})
    json_new(OUT / "local_dataset_manifest.json", {"status": "FROZEN_BEFORE_EXPANSION",
        "source": "G9.5 exact optimized-mean plan states for TRAIN/VALIDATION; G10.1 exact 40 curves for DEVELOPMENT-TEST",
        "train_validation_states": states, "development_test_states": dev,
        "train_state_count": 560, "validation_state_count": 120,
        "development_test_state_count": 40,
        "G101_state_manifest_sha256": sha(G101 / "state_manifest.json"),
        "G95_dataset_manifest_sha256": sha(G95 / "dataset_manifest.json"),
        "no_future_repair_success_selection": True})
    json_new(OUT / "local_dataset_split.json", {"split_unit": "parent episode; all probes from a state and all states from an episode stay in one split",
        "train_episode_seeds": sorted(train_seeds), "validation_episode_seeds": sorted(val_seeds),
        "development_test_90000_episode_seeds": sorted(test_seeds),
        "development_test_101000_event_episode_seeds": sorted({int(s["seed"]) for s in dev if s["cohort"] != "heldout_development_unsafe_nominal"}),
        "state_counts": {"train": 560, "validation": 120, "development_test": 40},
        "no_split_by_perturbation_row": True,
        "G101_development_was_previously_inspected_not_untouched": True,
        "confirmation_108000_108029_still_sealed": True})
    json_new(OUT / "local_training_range.json", {"primary_abs_delta_v_max": .5,
        "local_offsets_before_float32_clipping": LOCAL_OFFSETS.tolist(),
        "additional_stopping_withdrawal_witness_v": WITHDRAWAL_VALUES.tolist(),
        "train_use": "all complete probes; primary local subset |realized delta_v|<=0.5, larger stopping/withdrawal witnesses retained for coverage with one row per distinct float32 v",
        "category_report": ["|delta_v|<=0.1", "|delta_v|<=0.2", "|delta_v|<=0.5", "|delta_v|>0.5"],
        "primary_training_loss_weight": "1 for |realized delta_v|<=0.5; 0.25 for larger witness probes; 0 for exact nominal because output is overridden to 0",
        "search_uniform_grid": SEARCH_AXIS.tolist(), "search_uniform_grid_points": len(SEARCH_AXIS),
        "search_extra_points": "exact nominal and clipped nominal +/-0.05,0.10,0.20; all are present in G10.1 true curves",
        "closest_safe_tie_break": "minimum absolute v-v_nom, then lower v",
        "true_safe": "two complete intervals and R2<=F_lim", "H": 2,
        "rotation": "fixed exact nominal optimized-mean w_nom",
        "second_action": "fixed exact nominal optimized-mean second action, no replanning",
        "larger_withdrawal_coverage_not_claimed_local": True})
    json_new(OUT / "model_definition.json", {"primary_model": "M_DELTA",
        "alternative_model": "NOT_TRAINED; avoid model-comparison suite",
        "input": "frozen deployable 256D TD-MPC2 current latent + 38D E1 current physical estimate + v_nom + realized delta_v; 296D total",
        "excluded": "true future R2_nom, privileged SOFA state, current rotation as separate feature (already encoded in fixed nominal plan semantics)",
        "target": "delta_R2=R2_true([v,w_nom], fixed second action)-R2_true(nominal pair)",
        "output": "Q50,Q90,Q95 of delta_R2 in nominal N, noncrossing",
        "architecture": "296->64->64->3 ReLU MLP; unrestricted Q50 plus cumulative softplus increments to Q90/Q95",
        "loss": "weighted pinball at quantiles 0.50,0.90,0.95",
        "training_seeds": list(SEEDS), "optimizer": "Adam lr=1e-3", "max_epochs": 80,
        "batch_size": 512, "validation_patience": 10,
        "input_normalization": "train-only mean/std, std floored at 1e-6; frozen after training",
        "nominal_delta_override": "exact delta_v=0 predicts delta_R2=0 by definition",
        "nominal_force_deployment_baseline": "max over all three frozen G10 single-action validation-corrected Q95(R2) outputs at the exact nominal optimized-mean pair; never true future R2_nom",
        "absolute_reconstruction": "nominal G10 conservative Q95 + maximum over three M_DELTA validation-corrected Q95(delta_R2)",
        "validation_correction": "per replica max(0, validation 95th percentile of true delta_R2 minus predicted Q95 with higher interpolation); validation only",
        "F_lim_N": F_LIM, "H": 2})
    json_new(OUT / "abstention_rule.json", {"status": "RULE_FROZEN_BEFORE_MODEL_TRAINING",
        "predicted_safe_candidate": "reconstructed conservative R2<=F_lim AND inter-replica corrected Q95(delta_R2) range<=validation-only Q95 range threshold",
        "uncertainty_threshold": "validation 95th percentile (higher interpolation) of inter-replica corrected Q95(delta_R2) range; computed once and frozen before development-test",
        "abstain": "no candidate satisfying both force and uncertainty rules on frozen 101-point grid plus exact local probes",
        "no_arbitrary_posttest_threshold_tuning": True,
        "no_forced_withdrawal_fallback": True})
    json_new(OUT / "confirmation_seed_manifest.json", {"status": "SEALED_NOT_OPENED_BEFORE_PRE_CONFIRMATION_FREEZE",
        "ordered_seeds": list(CONFIRM_SEEDS), "seed_count": len(CONFIRM_SEEDS),
        "state_selection": "two per episode after a passive first replay: nearest decision to 25% episode progress and maximum prior executed force among noninitial decisions, deduplicated; no candidate force/repair success labels used",
        "parent_replay": "one reused scene through 108000-108029; repeat fresh ordered replay for selected-state forks after passive selection",
        "confirmation_translation_grid": "same 101 uniform float32 points plus exact nominal and nominal +/-0.05,0.10,0.20; two-step true-SOFA fork",
        "no_model_training_or_recalibration_after_opening": True})
    print("G10.2 predeclared 680 expansion states, 40 G10.1 development states; confirmation sealed", flush=True)


if __name__ == "__main__":
    preparation()
