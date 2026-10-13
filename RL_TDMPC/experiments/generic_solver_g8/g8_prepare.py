"""Predeclare and hash the entire G8 offline repair protocol before test seeds."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
RL=HERE.parents[1]
for name in ("generic_solver_g7","generic_solver_g6b","generic_solver_g6a","generic_solver_g3"):
    sys.path.insert(0,str(RL/"experiments"/name))
from g7_freeze import verify as verify_g7
from g7_prepare_train import OUT as G7, SEEDS as MODEL_SEEDS, sha, new_json
from g6b_freeze import verify as verify_g6b
from g6b_prepare_train import OUT as G6B
from g6_prepare import verify_source
from g8_repair import validate_candidates

OUT=RL/"logs/generic_solver_phase_g8_offline_quantile_shield"
TEST_SEEDS=list(range(101000,101030))
FROZEN=OUT/"g8_protocol_freeze.json"
ALIAS=OUT/"pre_confirmation_freeze.json"  # read by the inherited audit collector


def cuda_gate():
    if (os.environ.get("CUDA_VISIBLE_DEVICES")!="0" or not torch.cuda.is_available() or
            torch.cuda.device_count()!=1 or "RTX 5090" not in torch.cuda.get_device_name(0)):
        raise RuntimeError("G8 requires CUDA_VISIBLE_DEVICES=0 RTX 5090; no CPU fallback")


def audit_sources():
    g7=verify_g7();g6b=verify_g6b();q=verify_source()
    g7_selected=json.loads((G7/"selected_residual_calibration.json").read_text())
    g6b_selected=json.loads((G6B/"selected_cutoff.json").read_text())
    g7_models=json.loads((G7/"model_manifest.json").read_text())
    g6b_models=json.loads((G6B/"model_manifest.json").read_text())
    if (g7_selected["delta_cal_N"]!=0 or g7_selected["F_lim_N"]!=q["q95"] or
            g6b_selected["F_lim_N"]!=q["q95"] or
            g6b_selected["c_pred_N"]!=2.105934143066406 or
            g7_models["training_seeds"]!=list(MODEL_SEEDS)):
        raise RuntimeError("Frozen G7/G6B decision identities changed")
    if sha(G7/"models/normalization.json")!=sha(G6B/"models/FULL/normalization.json"):
        raise RuntimeError("Exact full-contact input normalization changed")
    old_states=sorted((G7/"confirmation_candidates").glob("state_*.json"))
    if len(old_states)!=120:raise RuntimeError("Frozen G7 candidate protocol sample unavailable")
    for path in old_states:
        state=json.loads(path.read_text())
        validate_candidates(state["candidates"],state["behavior_action"])
    return q,g7,g6b,g7_models,g6b_models,g7_selected,g6b_selected


def identities():
    code=[HERE/name for name in ("g8_repair.py","g8_prepare.py","g8_collect.py","g8_analyze.py")]
    code += [RL/"experiments/generic_solver_g3/protocol.py",RL/"experiments/generic_solver_g6a/g6_collect.py",
        RL/"experiments/generic_solver_g7/g7_collect.py",RL/"experiments/generic_solver_g7/g7_metrics.py",
        RL/"experiments/generic_solver_g6b/g6b_features.py"]
    data=[OUT/name for name in ("metadata.json","g7_predictor_audit.json","seed_manifest.json",
        "confirmation_seed_protocol.json","repair_rule.json")]
    data += [G7/name for name in ("pre_confirmation_freeze.json","quantile_model_definition.json",
        "selected_residual_calibration.json","models/normalization.json")]
    data += [G7/f"models/seed_{seed}/model.pt" for seed in MODEL_SEEDS]
    data += [G6B/name for name in ("pre_confirmation_freeze.json","selected_cutoff.json",
        "full_contact_feature_definition.json","models/FULL/normalization.json")]
    data += [G6B/f"models/FULL/seed_{seed}/model.pt" for seed in MODEL_SEEDS]
    paths=code+data
    if len(paths)!=len(set(paths)):raise RuntimeError("Duplicate G8 freeze path")
    return {str(p.relative_to(RL)) if p.is_relative_to(RL) else str(p):sha(p) for p in paths}


def verify():
    q,*_=audit_sources()
    freeze=json.loads(FROZEN.read_text())
    alias=json.loads(ALIAS.read_text())
    if freeze!=alias or freeze["status"]!="G8_OFFLINE_REPAIR_PROTOCOL_FROZEN_BEFORE_TEST" or freeze["test_seeds"]!=TEST_SEEDS:
        raise RuntimeError("G8 pretest protocol freeze changed")
    if freeze["F_lim_N"]!=q["q95"] or freeze["file_sha256"]!=identities():
        raise RuntimeError("G8 frozen rules/models/data/code changed")
    return freeze


def main():
    cuda_gate()
    if OUT.exists():raise FileExistsError(OUT)
    q,g7,g6b,g7_models,g6b_models,g7_selected,g6b_selected=audit_sources()
    OUT.mkdir(parents=True)
    new_json(OUT/"metadata.json",{"phase":"G8","branch":"feature/generic-contact-force-safety",
        "study":"offline one-step action repair only; no repaired action executed in future episodes",
        "python":sys.executable,"torch":torch.__version__,"torch_cuda":torch.version.cuda,
        "CUDA_VISIBLE_DEVICES":os.environ["CUDA_VISIBLE_DEVICES"],"visible_cuda_device_count":torch.cuda.device_count(),
        "model_device_index":0,"device_name":torch.cuda.get_device_name(0),
        "F_lim_N":q["q95"],"G2_Q99_nominal_N":q["q99"],
        "test_seeds":TEST_SEEDS,"model_training_seeds":list(MODEL_SEEDS)})
    new_json(OUT/"g7_predictor_audit.json",{"status":"FROZEN_G7_THREE_REPLICAS_AND_G6B_BASELINE_RECOVERED",
        "G7_freeze_status":g7["status"],"G7_pre_confirmation_freeze_sha256":sha(G7/"pre_confirmation_freeze.json"),
        "G7_quantile_model_definition_sha256":sha(G7/"quantile_model_definition.json"),
        "G7_normalization_sha256":sha(G7/"models/normalization.json"),
        "G7_checkpoint_sha256":g7_models["checkpoint_sha256"],
        "G7_Q95_residual_N":g7_selected["delta_cal_N"],
        "G6B_freeze_status":g6b["status"],"G6B_pre_confirmation_freeze_sha256":sha(G6B/"pre_confirmation_freeze.json"),
        "G6B_checkpoint_sha256":g6b_models["checkpoint_sha256"],
        "G6B_point_cutoff_N":g6b_selected["c_pred_N"],
        "exact_G6B_full_contact_state_dimension":306,"model_input_dimension":308,
        "frozen_prior_candidate_states_audited":120,"candidate_zero_exact_nominal_pass":True,
        "no_retraining_recalibration_seed_selection_or_ensemble":True})
    new_json(OUT/"seed_manifest.json",{"status":"PREDECLARED_BEFORE_101000_SERIES_OPENED",
        "seeds":TEST_SEEDS,"episodes":30,"states_per_episode":4,"candidates_per_state":21,
        "expected_states":120,"expected_candidate_labels":2520,
        "behavior_lifecycle":"one reused Generic scene, ordered resets",
        "candidate_lifecycle":"fresh reused Generic scene, ordered full replay/restore/fork",
        "state_selection":"exact frozen G3/G5 four-state progress/prior-force, group=confirmation",
        "candidate_protocol":"exact frozen G3/G5 21 actions, candidate 0 exact nominal, group=confirmation"})
    new_json(OUT/"confirmation_seed_protocol.json",{"status":"PREDECLARED_BEFORE_101000_SERIES_OPENED",
        "seeds":TEST_SEEDS,"episodes":30,"states_per_episode":4,"candidates_per_state":21,
        "expected_candidate_labels":2520,"behavior_lifecycle":"one reused Generic scene, ordered resets",
        "candidate_lifecycle":"fresh reused Generic scene, ordered replay/restore/fork",
        "state_selection":"exact frozen G3/G5 rule, group=confirmation",
        "candidate_protocol":"exact frozen G3/G5 21 actions, candidate 0 nominal"})
    new_json(OUT/"repair_rule.json",{"status":"FROZEN_BEFORE_TEST_LABELS",
        "nominal_candidate_id":0,"candidate_count":21,"action_space":"normalized 2D translation/rotation in [-1,1]^2",
        "distance":"ordinary Euclidean L2 on stored float32 normalized action pairs evaluated as float64",
        "tie_break":"smallest candidate_id after minimum distance",
        "G7_predicted_safe":"frozen Q95 <= exact G2 Q95 F_lim, delta_cal=0",
        "G6B_predicted_safe":"frozen point prediction <= 2.105934143066406 nominal N",
        "true_safe":"true next-interval distal maximum <= exact G2 Q95 F_lim",
        "selection":"candidate 0 if predicted/true safe; else nearest safe candidate; else None/infeasible",
        "predicted_infeasible_fallback":"none; do not select minimum predicted force",
        "oracle":"offline discrete true-force reference only, never deployable",
        "failure_taxonomy_priority":["CANDIDATE_SET_INFEASIBLE","PREDICTED_INFEASIBLE","UNSAFE_PASS_THROUGH","WRONG_REPAIR","OTHER_AMBIGUOUS"],
        "no_online_execution_or_subsequent_state_change":True,
        "repair_code_sha256":sha(HERE/"g8_repair.py")})
    if any(OUT.glob("confirmation_behavior_*")) or (OUT/"confirmation_candidates").exists():
        raise RuntimeError("G8 test seeds opened before freeze")
    frozen={"status":"G8_OFFLINE_REPAIR_PROTOCOL_FROZEN_BEFORE_TEST",
        "test_seeds":TEST_SEEDS,"model_training_seeds":list(MODEL_SEEDS),
        "F_lim_N":q["q95"],"G2_Q99_nominal_N":q["q99"],
        "G7_delta_cal_N":0.,"G6B_point_cutoff_N":g6b_selected["c_pred_N"],
        "repair_rule_sha256":sha(OUT/"repair_rule.json"),
        "candidate_protocol_sha256":sha(RL/"experiments/generic_solver_g3/protocol.py"),
        "state_selection_protocol_sha256":sha(OUT/"seed_manifest.json"),
        "no_test_data_opened":True,"file_sha256":identities()}
    new_json(FROZEN,frozen);new_json(ALIAS,frozen)
    verify()
    print(f"G8 offline repair protocol freeze PASS: {len(frozen['file_sha256'])} hashes; nominal candidate audited in 120 frozen prior states",flush=True)


def reseal_before_test():
    """Correct the omitted inherited collector protocol file before any G8 seed runs."""
    cuda_gate()
    if any(OUT.glob("confirmation_behavior_*")) or (OUT/"confirmation_candidates").exists() or (OUT/"candidate_results.csv").exists():
        raise RuntimeError("Cannot amend G8 preflight after any test data opened")
    q,*_=audit_sources()
    old=json.loads(FROZEN.read_text())
    if old!=json.loads(ALIAS.read_text()) or old["status"]!="G8_OFFLINE_REPAIR_PROTOCOL_FROZEN_BEFORE_TEST" or old["test_seeds"]!=TEST_SEEDS:
        raise RuntimeError("Prior G8 preflight freeze identity invalid")
    new_json(OUT/"preflight_freeze_superseded_before_test.json",old)
    new_json(OUT/"confirmation_seed_protocol.json",{"status":"PREDECLARED_BEFORE_101000_SERIES_OPENED",
        "seeds":TEST_SEEDS,"episodes":30,"states_per_episode":4,"candidates_per_state":21,
        "expected_candidate_labels":2520,"behavior_lifecycle":"one reused Generic scene, ordered resets",
        "candidate_lifecycle":"fresh reused Generic scene, ordered replay/restore/fork",
        "state_selection":"exact frozen G3/G5 rule, group=confirmation",
        "candidate_protocol":"exact frozen G3/G5 21 actions, candidate 0 nominal"})
    amended={**old,"F_lim_N":q["q95"],"collector_seed_protocol_sha256":sha(OUT/"confirmation_seed_protocol.json"),
        "preflight_amendment":"inherited collector required confirmation_seed_protocol.json; first attempt halted before scene/test seed",
        "file_sha256":identities()}
    FROZEN.write_text(json.dumps(amended,indent=2,sort_keys=True,allow_nan=False)+"\n")
    ALIAS.write_text(json.dumps(amended,indent=2,sort_keys=True,allow_nan=False)+"\n")
    verify()
    print(f"G8 amended pretest freeze PASS: {len(amended['file_sha256'])} hashes; no 101000-series seed opened",flush=True)


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--reseal-before-test",action="store_true")
    arguments=parser.parse_args()
    reseal_before_test() if arguments.reseal_before_test else main()
