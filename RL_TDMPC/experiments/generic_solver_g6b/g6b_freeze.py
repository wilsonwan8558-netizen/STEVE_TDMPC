"""Freeze G6B feature/model/calibration/code identities before 98000-series labels."""

from __future__ import annotations

import json
from pathlib import Path

from g6b_prepare_train import OUT,RL,HERE,SEEDS,CALIBRATION,CONFIRMATION,sha,new_json

MANIFEST=OUT/"pre_confirmation_freeze.json"


def files_to_hash():
    scripts=[HERE/name for name in ("g6b_features.py","g6b_prepare_train.py","g6b_development.py",
        "g6b_collect.py","g6b_calibrate.py","g6b_freeze.py","g6b_evaluate.py")]
    inherited=[RL/"experiments/generic_solver_g6a"/name for name in
        ("g6_collect.py","g6_metrics.py","g6_calibrate.py")]
    protocol=[RL/"experiments/generic_solver_g3/protocol.py",RL/"experiments/generic_solver_g4/g4_train.py",
        RL/"experiments/generic_solver_g4/physical_features.py"]
    data=[OUT/name for name in ("metadata.json","full_contact_feature_definition.json","model_manifest.json",
        "development_metrics.csv","training_seed_robustness.json","calibration_seed_protocol.json",
        "confirmation_seed_protocol.json","calibration_selection_protocol.json","calibration_seed_manifest.json",
        "calibration_behavior_episodes.csv","calibration_behavior_steps.csv","calibration_replay_audit.json",
        "calibration_scoring_audit.json","calibration_candidate_results.csv","cutoff_sweep.csv",
        "selected_cutoff.json","calibration_tradeoff.json","calibration_false_safe_severity.json",
        "calibration_contact_emergence.json","calibration_action_availability.json",
        "models/FULL/normalization.json")]
    data += [OUT/f"models/FULL/seed_{seed}/model.pt" for seed in SEEDS]
    states=sorted((OUT/"calibration_candidates").glob("state_*.json"))
    if len(states)!=120: raise RuntimeError("G6B calibration states incomplete")
    data += states
    if len(set(scripts+inherited+protocol+data))!=len(scripts+inherited+protocol+data):
        raise RuntimeError("Duplicate freeze paths")
    return scripts+inherited+protocol+data


def identities():
    return {str(path.relative_to(RL)) if path.is_relative_to(RL) else str(path):sha(path) for path in files_to_hash()}


def verify():
    from g6b_collect import verify_g6b
    q=verify_g6b();frozen=json.loads(MANIFEST.read_text())
    if frozen["status"]!="G6B_THREE_REPLICAS_AND_SHARED_CUTOFF_FROZEN_BEFORE_CONFIRMATION":
        raise RuntimeError("G6B freeze status changed")
    if frozen["calibration_seeds"]!=CALIBRATION or frozen["confirmation_seeds"]!=CONFIRMATION or frozen["training_seeds"]!=list(SEEDS):
        raise RuntimeError("G6B seeds changed")
    if frozen["F_lim_N"]!=q["q95"] or frozen["file_sha256"]!=identities():
        raise RuntimeError("G6B frozen data/code/model changed")
    return frozen


def main():
    if MANIFEST.exists():raise FileExistsError(MANIFEST)
    from g6b_collect import verify_g6b
    q=verify_g6b()
    selected=json.loads((OUT/"selected_cutoff.json").read_text())
    if selected["F_lim_N"]!=q["q95"] or not selected["all_replicas_meet_empirical_5pct"]:
        raise RuntimeError("G6B calibration incomplete")
    if selected["confirmation_seeds_opened_before_selection"]:
        raise RuntimeError("Confirmation leaked before selection")
    if any(OUT.glob("confirmation_behavior_*")) or (OUT/"confirmation_candidates").exists() or (OUT/"confirmation_candidate_results.csv").exists():
        raise RuntimeError("G6B confirmation opened before freeze")
    freeze={"status":"G6B_THREE_REPLICAS_AND_SHARED_CUTOFF_FROZEN_BEFORE_CONFIRMATION",
        "calibration_seeds":CALIBRATION,"confirmation_seeds":CONFIRMATION,"training_seeds":list(SEEDS),
        "F_lim_N":q["q95"],"c_pred_N":selected["c_pred_N"],"Delta_N":selected["Delta_N"],
        "feature_definition_sha256":sha(OUT/"full_contact_feature_definition.json"),
        "normalization_sha256":sha(OUT/"models/FULL/normalization.json"),
        "checkpoint_sha256":{str(seed):sha(OUT/f"models/FULL/seed_{seed}/model.pt") for seed in SEEDS},
        "candidate_protocol_sha256":sha(RL/"experiments/generic_solver_g3/protocol.py"),
        "state_selection_protocol_sha256":sha(OUT/"confirmation_seed_protocol.json"),
        "evaluation_definitions_sha256":sha(RL/"experiments/generic_solver_g6a/g6_metrics.py"),
        "no_confirmation_data_opened":True,"file_sha256":identities()}
    new_json(MANIFEST,freeze)
    verify()
    print(f"G6B pre-confirmation freeze PASS: {len(freeze['file_sha256'])} hashes",flush=True)


if __name__=="__main__":main()
