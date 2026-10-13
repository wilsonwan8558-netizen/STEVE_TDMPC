"""Immutable G7 quantile/residual/data/code freeze before 100000-series confirmation."""

from __future__ import annotations

import json
from pathlib import Path

from g7_prepare_train import OUT,RL,HERE,SEEDS,CALIBRATION,CONFIRMATION,sha,new_json

G6B=RL/"logs/generic_solver_phase_g6b_full_contact_predictor"
MANIFEST=OUT/"pre_confirmation_freeze.json"


def files_to_hash():
    code=[HERE/name for name in ("g7_prepare_train.py","g7_metrics.py","g7_development.py",
        "g7_collect.py","g7_calibrate.py","g7_freeze.py","g7_evaluate.py")]
    code += [RL/"experiments/generic_solver_g6a"/name for name in ("g6_collect.py","g6_metrics.py")]
    code += [RL/"experiments/generic_solver_g6b"/name for name in
        ("g6b_features.py","g6b_prepare_train.py","g6b_development.py")]
    code += [RL/"experiments/generic_solver_g3/protocol.py",RL/"experiments/generic_solver_g4/physical_features.py"]
    files=[OUT/name for name in ("metadata.json","g6b_input_audit.json","training_data_audit.json",
        "quantile_model_definition.json","training_seed_manifest.json","model_manifest.json",
        "models/normalization.json","development_quantile_metrics.csv","development_boundary_metrics.csv",
        "calibration_seed_protocol.json","confirmation_seed_protocol.json","residual_calibration_protocol.json",
        "calibration_seed_manifest.json","calibration_behavior_episodes.csv","calibration_behavior_steps.csv",
        "calibration_replay_audit.json","calibration_scoring_audit.json","calibration_candidate_results.csv",
        "calibration_quantile_metrics.csv","calibration_boundary_metrics.csv",
        "quantile_calibration_summary.json","selected_residual_calibration.json")]
    files += [OUT/f"models/seed_{seed}/model.pt" for seed in SEEDS]
    states=sorted((OUT/"calibration_candidates").glob("state_*.json"))
    if len(states)!=120:raise RuntimeError("G7 calibration candidate states incomplete")
    files += states
    files += [G6B/name for name in ("pre_confirmation_freeze.json","full_contact_feature_definition.json",
        "models/FULL/normalization.json","selected_cutoff.json")]
    files += [G6B/f"models/FULL/seed_{seed}/model.pt" for seed in SEEDS]
    all_files=code+files
    if len(set(all_files))!=len(all_files):raise RuntimeError("Duplicate G7 freeze paths")
    return all_files


def identities():
    return {str(p.relative_to(RL)) if p.is_relative_to(RL) else str(p):sha(p) for p in files_to_hash()}


def verify():
    from g7_collect import verify_g7
    q=verify_g7();frozen=json.loads(MANIFEST.read_text())
    if frozen["status"]!="G7_QUANTILES_AND_SHARED_RESIDUAL_FROZEN_BEFORE_CONFIRMATION":
        raise RuntimeError("G7 freeze status changed")
    if frozen["calibration_seeds"]!=CALIBRATION or frozen["confirmation_seeds"]!=CONFIRMATION or frozen["training_seeds"]!=list(SEEDS):
        raise RuntimeError("G7 seeds changed")
    if frozen["F_lim_N"]!=q["q95"] or frozen["file_sha256"]!=identities():
        raise RuntimeError("G7 frozen data/model/code changed")
    return frozen


def main():
    if MANIFEST.exists():raise FileExistsError(MANIFEST)
    from g7_collect import verify_g7
    q=verify_g7();selected=json.loads((OUT/"selected_residual_calibration.json").read_text())
    if (selected["status"]!="SMALLEST_SHARED_NONNEGATIVE_RESIDUAL_SELECTED_ON_CALIBRATION_ONLY" or
            selected["F_lim_N"]!=q["q95"] or not selected["all_replicas_meet_empirical_5pct"] or
            selected["confirmation_seeds_opened_before_selection"]):
        raise RuntimeError("G7 residual calibration incomplete")
    if any(OUT.glob("confirmation_behavior_*")) or (OUT/"confirmation_candidates").exists() or (OUT/"confirmation_candidate_results.csv").exists():
        raise RuntimeError("G7 confirmation opened before freeze")
    protocol=json.loads((OUT/"confirmation_seed_protocol.json").read_text())
    if protocol["seeds"]!=CONFIRMATION or protocol["expected_labels"]!=2520:
        raise RuntimeError("G7 confirmation protocol changed")
    freeze={"status":"G7_QUANTILES_AND_SHARED_RESIDUAL_FROZEN_BEFORE_CONFIRMATION",
        "calibration_seeds":CALIBRATION,"confirmation_seeds":CONFIRMATION,"training_seeds":list(SEEDS),
        "F_lim_N":q["q95"],"delta_cal_N":selected["delta_cal_N"],"quantile_levels":[.5,.9,.95,.99],
        "G6B_feature_definition_sha256":sha(G6B/"full_contact_feature_definition.json"),
        "G6B_normalization_sha256":sha(G6B/"models/FULL/normalization.json"),
        "G7_model_definition_sha256":sha(OUT/"quantile_model_definition.json"),
        "G7_checkpoint_sha256":{str(seed):sha(OUT/f"models/seed_{seed}/model.pt") for seed in SEEDS},
        "G6B_checkpoint_sha256":{str(seed):sha(G6B/f"models/FULL/seed_{seed}/model.pt") for seed in SEEDS},
        "G6B_frozen_cutoff_sha256":sha(G6B/"selected_cutoff.json"),
        "state_selection_protocol_sha256":sha(OUT/"confirmation_seed_protocol.json"),
        "candidate_protocol_sha256":sha(RL/"experiments/generic_solver_g3/protocol.py"),
        "evaluation_definitions_sha256":sha(HERE/"g7_metrics.py"),
        "no_confirmation_data_opened":True,"file_sha256":identities()}
    new_json(MANIFEST,freeze);verify()
    print(f"G7 pre-confirmation freeze PASS: {len(freeze['file_sha256'])} hashes",flush=True)


if __name__=="__main__":main()
