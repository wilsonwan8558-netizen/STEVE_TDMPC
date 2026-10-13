"""Predeclare G6A three-replica boundary protocol and audit frozen G5 inference."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
sys.path.insert(0, str(RL / "experiments/generic_solver_g5"))
sys.path.insert(0, str(RL / "experiments/generic_solver_g3"))
from g5_freeze import verify as verify_g5  # noqa: E402
from g5_prepare import OUT as G5, G2, G1B, EXPECTED_CHECKPOINT_SHA, EXPECTED_G2_SHA, sha  # noqa: E402
from g5_data import spec  # noqa: E402
from g5_metrics import load_predictor, predict, quantiles  # noqa: E402

OUT = RL / "logs/generic_solver_phase_g6a_boundary_calibration"
MODEL = "ZP_noC"
SEEDS = (260206, 260207, 260208)
HASHES = {
    260206: "05efea349196c83e088fc0a8337539977c883ad89ef4899e07eb72cafac596a5",
    260207: "40f54351b98c19c2ee7a4662eb61156c0b8a2bb6c7b3b17967c3a41444ecb148",
    260208: "77f75420b6b31339fdb462b7d14fe5e8d306a567e5f4db6d657510ea0ee9f4aa",
}
NORM_SHA = "1cd651d19d6d93c48a638c2824586a7a0313872b4b0edb5d931f1c16fb09f399"
CALIBRATION = list(range(95000, 95030))
CONFIRMATION = list(range(96000, 96030))


def new_json(path, value):
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")


def verify_source():
    verify_g5()
    selection = json.loads((G5 / "minimal_subset_selection.json").read_text())
    if selection["included_groups"] != ["K", "D"] or selection["P_min_dimension"] != 33 or selection["selection_model"] != MODEL:
        raise RuntimeError("G5 Z+P_min definition changed")
    include_z, indices = spec(MODEL)
    if not include_z or indices != selection["P_min_indices"] or len(indices) != 33:
        raise RuntimeError("G5 input order/indices changed")
    if sha(G5 / "models" / MODEL / "normalization.json") != NORM_SHA:
        raise RuntimeError("G5 train normalization changed")
    for seed, expected in HASHES.items():
        if sha(G5 / "models" / MODEL / f"seed_{seed}" / "model.pt") != expected:
            raise RuntimeError(f"G5 frozen model changed: {seed}")
    checkpoint = Path(json.loads((G1B / "selected_checkpoint.json").read_text())["path"])
    if sha(checkpoint) != EXPECTED_CHECKPOINT_SHA or sha(G2 / "step_level_force.csv") != EXPECTED_G2_SHA:
        raise RuntimeError("Frozen navigation or G2 force source changed")
    q = quantiles()
    if abs(q["q95"] - 3.685331040756903) > 1e-12:
        raise RuntimeError("G2 Q95 changed")
    return q


def audit_g5(q):
    paths = sorted((G5 / "confirmation_candidates").glob("state_*.json"))[:3]
    if len(paths) != 3:
        raise RuntimeError("G5 existing confirmation states unavailable for semantic audit")
    states = [json.loads(path.read_text()) for path in paths]
    include_z, indices = spec(MODEL)
    if not include_z:
        raise RuntimeError("Z+P_min latent channel missing")
    x = np.asarray([np.r_[np.asarray(state["latent"], dtype=np.float64),
                          np.asarray(state["physical"], dtype=np.float64)[indices], candidate["action"]]
                    for state in states for candidate in state["candidates"]], dtype=np.float64)
    truth = np.asarray([candidate["force_next_max_N"] for state in states for candidate in state["candidates"]], dtype=np.float64)
    if x.shape != (63,291):
        raise RuntimeError("G5 Z+P_min+action shape changed")
    frozen_rows = {}
    with (G5 / "confirmation_candidate_results.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["model"] == MODEL and int(row["state_id"]) < 3:
                frozen_rows[(int(row["training_seed"]), int(row["state_id"]), int(row["candidate_id"]))] = row
    if len(frozen_rows) != 3*63:
        raise RuntimeError("G5 reference prediction count changed")
    differences = {}
    for seed in SEEDS:
        predictor = load_predictor(MODEL, seed, torch.device("cuda:0"))
        p = predict(predictor, x, torch.device("cuda:0"))
        model_error, label_error, classification_mismatch = 0., 0., 0
        for i, state in enumerate(states):
            for j in range(21):
                row = frozen_rows[(seed, int(state["state_id"]), j)]
                k = i*21+j
                model_error = max(model_error, abs(p[k]-float(row["predicted_force_next_max_N"])))
                label_error = max(label_error, abs(truth[k]-float(row["force_next_max_N"])))
                classification_mismatch += int((truth[k]>q["q95"]) != (float(row["force_next_max_N"])>q["q95"]))
        if model_error > 1e-8 or label_error != 0 or classification_mismatch:
            raise RuntimeError(f"G5 prediction/target semantic reproduction failed {seed}")
        differences[str(seed)] = {"max_prediction_abs_difference_N": model_error, "max_label_abs_difference_N": label_error,
                                  "Q95_truth_mismatch_count": classification_mismatch}
    return {"status":"PASS_EXACT_G5_THREE_REPLICA_SEMANTICS", "source":"G5 untouched confirmation read-only audit; not used for cutoff selection",
            "audited_states":3,"audited_candidates_per_replica":63,"input_dimension":291,"physical_dimension":33,
            "input_order":"G5 latent[256], sorted K+D P indices[33], candidate action[2]", "differences":differences}


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available() or torch.cuda.device_count() != 1 or "RTX 5090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("CUDA_VISIBLE_DEVICES=0 RTX 5090 required; no CPU fallback")
    if OUT.exists():
        raise FileExistsError(OUT)
    q = verify_source()
    audit = audit_g5(q)
    OUT.mkdir(parents=True)
    new_json(OUT / "g5_model_audit.json", {**audit, "g5_pre_confirmation_freeze_sha256":sha(G5/"pre_confirmation_freeze.json"),
        "navigation_checkpoint_sha256":EXPECTED_CHECKPOINT_SHA,"G2_force_step_csv_sha256":EXPECTED_G2_SHA,
        "model_name":MODEL,"training_seeds":list(SEEDS),"model_checkpoint_sha256":HASHES,"normalization_sha256":NORM_SHA,
        "P_min_indices":spec(MODEL)[1],"supervision":"behavior-only G5 primary ablation",
        "architecture":"291-64-ReLU-64-ReLU-1","force_target":"next 22 SOFA-substep maximum distal contact force nominal N"})
    new_json(OUT / "metadata.json", {"phase":"G6A","branch":"feature/generic-contact-force-safety",
        "study":"three frozen independent training-seed replicas; shared robust cutoff; no ensembling or retraining",
        "python":sys.executable,"torch":torch.__version__,"torch_cuda":torch.version.cuda,
        "CUDA_VISIBLE_DEVICES":"0","device_name":torch.cuda.get_device_name(0),"model_device_index":0,
        "G2_reference_quantiles_nominal_N":q})
    for group,seeds in (("calibration",CALIBRATION),("confirmation",CONFIRMATION)):
        new_json(OUT / f"{group}_seed_protocol.json", {"status":"PREDECLARED_BEFORE_CALIBRATION_LABELS_OPENED",
            "seeds":seeds,"episodes":30,"states_per_episode":4,"candidates_per_state":21,"expected_candidate_labels":2520,
            "behavior_lifecycle":"one reused Generic scene, ordered seed resets",
            "candidate_lifecycle":"new reused Generic scene, ordered full replay and restore/fork at selected states",
            "state_selection":"exact G5/G3 four-state progress/prior-force rule, group=confirmation",
            "candidate_protocol":"exact G5/G3 21 actions, group=confirmation, RNG group offset 400000",
            "first_confirmation_seed_open_only_after_shared_cutoff_freeze":group=="confirmation"})
    new_json(OUT / "calibration_selection_protocol.json", {"status":"PREDECLARED_BEFORE_CALIBRATION_LABELS_OPENED",
        "true_reference_key":"q95","F_lim_N":q["q95"],"predicted_safe_rule":"prediction <= c_pred",
        "candidate_cutoffs":"all distinct pooled three-replica calibration predictions <= F_lim, plus F_lim and one sentinel below global prediction minimum; exact-event sweep",
        "selection":"largest cutoff <= F_lim with maximum of three per-replica empirical Q95 false-safe rates <= 0.05",
        "no_mean_or_pooled_selection":True,"no_checkpoint_selection":True,
        "degenerate_action_availability":"report empirical acceptance and zero-accepted-state rate; do not force a success label",
        "uncertainty":"two-sided exact Clopper-Pearson 95% per-replica binomial CI, not a selection condition",
        "calibration_seeds":CALIBRATION,"untouched_confirmation_seeds":CONFIRMATION})
    print(f"G6A exact G5 semantics PASS; three-replica protocol predeclared in {OUT}",flush=True)


if __name__ == "__main__":
    main()
