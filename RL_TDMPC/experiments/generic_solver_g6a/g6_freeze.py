"""Immutable three-replica G6A cutoff/data/code freeze before 96000-series seeds."""

from __future__ import annotations

import json
from pathlib import Path

from g6_prepare import OUT, G5, G2, G1B, RL, MODEL, SEEDS, HASHES, NORM_SHA, CALIBRATION, CONFIRMATION, verify_source, sha

HERE=Path(__file__).resolve().parent
MANIFEST=OUT/"pre_confirmation_freeze.json"


def files_to_hash():
    files=[HERE/name for name in ("g6_prepare.py","g6_collect.py","g6_metrics.py","g6_calibrate.py","g6_freeze.py","g6_evaluate.py")]
    files += [OUT/name for name in (
        "metadata.json","g5_model_audit.json","calibration_seed_protocol.json","confirmation_seed_protocol.json",
        "calibration_selection_protocol.json","calibration_seed_manifest.json","calibration_behavior_episodes.csv",
        "calibration_behavior_steps.csv","calibration_replay_audit.json","calibration_scoring_audit.json",
        "calibration_candidate_results.csv","cutoff_sweep.csv","selected_cutoff.json","calibration_tradeoff.json",
        "calibration_false_safe_severity.json","calibration_contact_emergence.json",
        "calibration_reference_diagnostics.json","calibration_action_availability.json",
        "plots/calibration_acceptance_vs_false_safe.png")]
    states=sorted((OUT/"calibration_candidates").glob("state_*.json"))
    if len(states)!=120:
        raise RuntimeError("G6A calibration state files incomplete")
    files += states
    files += [G5/"pre_confirmation_freeze.json",G5/"minimal_subset_selection.json",
              G5/"models"/MODEL/"normalization.json",G2/"step_level_force.csv",G2/"force_distribution_summary.json"]
    files += [G5/"models"/MODEL/f"seed_{seed}"/"model.pt" for seed in SEEDS]
    checkpoint=Path(json.loads((G1B/"selected_checkpoint.json").read_text())["path"])
    files.append(checkpoint)
    return files


def identities():
    files=files_to_hash()
    if len(set(files))!=len(files):
        raise RuntimeError("Duplicate G6A freeze path")
    return {str(path.relative_to(RL)) if path.is_relative_to(RL) else str(path):sha(path) for path in files}


def verify():
    verify_source()
    data=json.loads(MANIFEST.read_text())
    if data["status"]!="G6A_SHARED_CUTOFF_AND_THREE_REPLICAS_FROZEN_BEFORE_CONFIRMATION":
        raise RuntimeError("G6A freeze status invalid")
    if data["calibration_seeds"]!=CALIBRATION or data["confirmation_seeds"]!=CONFIRMATION or data["training_seeds"]!=list(SEEDS):
        raise RuntimeError("G6A frozen seeds changed")
    if data["file_sha256"]!=identities():
        raise RuntimeError("G6A frozen models, cutoff, source, data or code changed")
    return data


def main():
    if MANIFEST.exists():
        raise FileExistsError(MANIFEST)
    q=verify_source()
    selected=json.loads((OUT/"selected_cutoff.json").read_text())
    if selected["status"]!="SHARED_WORST_REPLICA_Q95_CUTOFF_SELECTED_ON_CALIBRATION_ONLY" or not selected["all_replicas_meet_empirical_5pct"]:
        raise RuntimeError("G6A calibration cutoff selection incomplete")
    if selected["F_lim_N"]!=q["q95"] or abs(selected["F_lim_N"]-selected["c_pred_N"]-selected["Delta_N"])>1e-12:
        raise RuntimeError("G6A selected cutoff/margin inconsistent")
    if selected["confirmation_seeds_opened_before_selection"]:
        raise RuntimeError("G6A confirmation leakage")
    if any(OUT.glob("confirmation_behavior_*")) or (OUT/"confirmation_candidates").exists() or (OUT/"confirmation_candidate_results.csv").exists():
        raise RuntimeError("G6A confirmation opened before freeze")
    protocol=json.loads((OUT/"confirmation_seed_protocol.json").read_text())
    if protocol["seeds"]!=CONFIRMATION or protocol["expected_candidate_labels"]!=2520:
        raise RuntimeError("G6A confirmation protocol changed")
    hashes=identities()
    if hashes[str((G5/"models"/MODEL/"normalization.json").relative_to(RL))]!=NORM_SHA:
        raise RuntimeError("G5 normalization hash changed")
    for seed in SEEDS:
        if hashes[str((G5/"models"/MODEL/f"seed_{seed}"/"model.pt").relative_to(RL))]!=HASHES[seed]:
            raise RuntimeError("G5 replica hash changed")
    MANIFEST.write_text(json.dumps({"status":"G6A_SHARED_CUTOFF_AND_THREE_REPLICAS_FROZEN_BEFORE_CONFIRMATION",
        "calibration_seeds":CALIBRATION,"confirmation_seeds":CONFIRMATION,"training_seeds":list(SEEDS),
        "model_checkpoint_sha256":{str(k):v for k,v in HASHES.items()},
        "representation":"Z+K+D; G5 sorted P indices 0-29,35-37; behavior-only, three independent replicas",
        "normalization_sha256":NORM_SHA,"F_lim_N":q["q95"],"c_pred_N":selected["c_pred_N"],
        "Delta_N":selected["Delta_N"],"selection_rule":"largest cutoff satisfying max across replica Q95 FSR <=5% on calibration only",
        "state_selection_protocol_sha256":sha(OUT/"confirmation_seed_protocol.json"),
        "candidate_protocol_sha256":sha(RL/"experiments/generic_solver_g3/protocol.py"),
        "evaluation_definitions_sha256":sha(HERE/"g6_metrics.py"),
        "no_confirmation_data_opened":True,"file_sha256":hashes},indent=2,sort_keys=True,allow_nan=False)+"\n")
    verify()
    print(f"G6A pre-confirmation shared cutoff freeze PASS: {len(hashes)} hashes",flush=True)


if __name__=="__main__":
    main()
