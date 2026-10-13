"""Predeclare translation-only offline rescue diagnostic before force sweeps."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
sys.path.insert(0, str(RL / "experiments/generic_solver_g10"))
sys.path.insert(0, str(RL / "experiments/generic_solver_g95"))

from g10_prepare import OUT as G10, G95, F_LIM, gate as g10_gate  # noqa: E402
from g95_common import json_new, sha  # noqa: E402

OUT = RL / "logs/generic_solver_phase_g101_translation_only_repair"
PRED_AXIS = np.linspace(-1, 1, 201, dtype=np.float32)
TRUE_AXIS = np.linspace(-1, 1, 101, dtype=np.float32)
LOCAL_DELTAS = (0.05, 0.10, 0.20)


def gate():
    source, models = g10_gate()
    decision = json.loads((G10 / "offline_gate_decision.json").read_text())
    if (decision["decision"] != "NO_GO_OFFLINE_STOP" or
            decision["final_classification"] != "C_FORCE_CONSTRAINED_REPAIR_FAILS" or
            decision["online_confirmation_107000_107199_opened"]):
        raise RuntimeError("G10 no-go / unopened online cohort contract changed")
    definition = json.loads((G10 / "frozen_predictor_definition.json").read_text())
    old = models["models"]["M1_planner_conditioned"]
    if (definition["replica_checkpoint_sha256"] !=
            {str(s): old[str(s)]["checkpoint_sha256"] for s in models["training_seeds"]} or
            definition["normalization_sha256"] !=
            sha(G10.parent / "generic_solver_phase_g9_temporal_anticipatory_safety" /
                "models/risk/E1/H2/normalization.json")):
        raise RuntimeError("G10.1 frozen G10 predictor definition changed")
    return source, models, definition


def states():
    source = json.loads((G10 / "offline_state_manifest.json").read_text())
    if len(source["states"]) != 40:
        raise RuntimeError("G10.1 requires exact 40 G10 offline parent states")
    result = []
    for state in source["states"]:
        if state["cohort"] == "heldout_development_unsafe_nominal":
            path = G10 / "heldout_state_results" / f"state_{state['state_id']:04d}.json"
        else:
            path = (G10 / "difficult_event_state_results" /
                    f"state_{state['state_id']:04d}_minus_{state['lookback_steps']}.json")
        prior = json.loads(path.read_text())
        if (prior["seed"], prior["decision_index"], prior["state_id"]) != (
                state["seed"], state["decision_index"], state["state_id"]):
            raise RuntimeError("G10.1 G10 state/result identity changed")
        result.append({**state, "G10_result_sha256": sha(path),
                       "G10_nominal_pair": prior["nominal_pair"],
                       "G10_status": prior["proposal"]["status"],
                       "G10_zero_rotation_zero_translation_true_safe":
                       prior["fallback_zero_true"]["true_safe"]})
    return result


def prepare():
    source, _, definition = gate()
    if OUT.exists():
        raise FileExistsError(OUT)
    rows = states()
    OUT.mkdir(parents=True)
    json_new(OUT / "metadata.json", {"phase": "G10.1",
        "branch": "feature/generic-contact-force-safety",
        "status": "OFFLINE_TRANSLATION_ONLY_PROTOCOL_PREDECLARED",
        "python": sys.executable, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
        "cuda_device_index": 0, "cuda_device_name": torch.cuda.get_device_name(0),
        "G10_no_go_decision_sha256": sha(G10 / "offline_gate_decision.json"),
        "G10_state_manifest_sha256": sha(G10 / "offline_state_manifest.json"),
        "controller_checkpoint_sha256": source["checkpoint_sha256"],
        "F_lim_N": F_LIM,
        "F_lim_interpretation": "Generic-simulation statistical high-force reference, not a clinical injury threshold",
        "107000_107199_online_seeds_unopened": True})
    json_new(OUT / "frozen_predictor_definition.json", {**definition,
        "G10_definition_sha256": sha(G10 / "frozen_predictor_definition.json"),
        "G10_predictor_source_sha256": sha(RL / "experiments/generic_solver_g10/g10_repair.py"),
        "frozen_G10_predictor_reused_without_retraining_or_recalibration": True})
    json_new(OUT / "translation_only_protocol.json", {
        "status": "FROZEN_BEFORE_TRUE_TRANSLATION_SWEEPS",
        "objective": "minimize |v-v_nom|^2 over float32 v in [-1,1] subject to all three G10 corrected Q95(R2)<=F_lim",
        "current_rotation": "exact nominal TD-MPC2 optimized-mean rotation w_nom; never varied",
        "second_action": "exact frozen nominal optimized-mean second action from G9.6-v2; never varied/replanned in fork",
        "H": 2, "F_lim_N": F_LIM,
        "predicted_uniform_axis": PRED_AXIS.tolist(), "predicted_uniform_points": len(PRED_AXIS),
        "true_uniform_axis": TRUE_AXIS.tolist(), "true_uniform_points": len(TRUE_AXIS),
        "extra_predicted_and_true_points": "exact float32 v_nom and clipped v_nom+/-{0.05,0.10,0.20}; exact predicted closest-safe v added to true SOFA sweep if not already present; deduplicate exact float32",
        "local_perturbation_abs_deltas": list(LOCAL_DELTAS),
        "predicted_safe": "all three frozen validation-corrected Q95 values <= F_lim",
        "true_safe": "two complete true-SOFA intervals and R2=max(F1,F2)<=F_lim",
        "closest_safe_tie_break": "minimum absolute v-v_nom, then lower v",
        "finite_sweep_not_global_optimization_or_impossibility_proof": True,
        "direction_sign_zero_tolerance_N": 1e-6,
        "directional_agreement": "per valid exact nominal +/-delta probe compare sign of true R2 change and sign of maximum three-replica corrected Q95 change; report each sign including zero",
        "slope": "central difference at +/-0.05 if both in bounds and complete; otherwise one-sided 0.05 if possible; report sign and across-state correlation descriptively",
        "monotonicity": "report Spearman(v,R2) over complete 101-point true uniform grid and fraction adjacent decreases greater than 1e-6 N; descriptive approximate nondecreasing if <=5% such decreases; local analogous on uniform points within +/-0.20 of v_nom; no tuned threshold",
        "zero_translation": "[0,w_nom], distinct from G10 [0,0]",
        "withdrawal_witnesses": [-1.,-.5,0.],
        "cohorts": "exact G10 18 held-out unsafe optimized-mean states plus 11 difficult G8.5 events at t*-1 and t*-2",
        "no_online_107000_series": True,
        "no_training_or_threshold_change": True})
    json_new(OUT / "state_manifest.json", {"states": rows, "count": 40,
        "heldout_unsafe_nominal_count": 18, "difficult_event_prior_state_count": 22,
        "G10_state_manifest_sha256": sha(G10 / "offline_state_manifest.json"),
        "outcome_selection_after_G10": False,
        "107000_107199_accessed": False})
    print("G10.1 predeclared exact 40-state translation-only sweep; 107000-series unopened", flush=True)


if __name__ == "__main__":
    prepare()
