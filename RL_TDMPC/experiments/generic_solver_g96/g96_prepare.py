"""Audit exact frozen G9.5 plan/profile data before G9.6 training."""

from __future__ import annotations

import csv
import json
import sys

import numpy as np
import torch

from g96_common import (HERE, OUT, G2, G9, G95, CONFIRM, GROUPS, SEEDS,
                        F_LIM, TAUS, frozen_sources, training_arrays, json_new, sha)


def check_plan_rows(p):
    selection = json.loads((G95 / "dataset_split.json").read_text())
    summary = json.loads((G95 / "development_dataset_summary.json").read_text())
    group_seed = {group: set(selection["development_groups"][group]) for group in GROUPS}
    if (list(map(len, (group_seed[g] for g in GROUPS))) != [140, 30, 30] or
            any(group_seed[a] & group_seed[b] for a in GROUPS for b in GROUPS if a != b) or
            summary["candidate_forks"] != 7200 or summary["complete_R2_rows"] != 7193 or
            summary["censored_one_step_terminal_rows"] != 7):
        raise RuntimeError("G9.6 frozen G9.5 split/count changed")
    cache = {}
    counts = {g: 0 for g in GROUPS}
    keys = set()
    max_abs = 0.0
    for index in range(len(p["x"])):
        group = GROUPS[int(p["group_index"][index])]
        sid = int(p["state_id_within_group"][index])
        state_key = (group, sid)
        if state_key not in cache:
            path = G95 / "development_states" / f"state_{group}_{sid:04d}.json"
            cache[state_key] = json.loads(path.read_text())
        item = cache[state_key]
        candidate_id = int(p["candidate_id"][index])
        plan = item["plans"][candidate_id]
        if (int(p["seed"][index]) != item["seed"] or
                int(p["decision_index"][index]) != item["decision_index"] or
                item["seed"] not in group_seed[group] or
                plan["candidate_id"] != candidate_id or
                not plan["fork"]["complete_two_intervals"] or
                (group, sid, candidate_id) in keys):
            raise RuntimeError(f"G9.6 original candidate/state mismatch at row {index}")
        keys.add((group, sid, candidate_id))
        action = np.asarray(plan["actions"], dtype=np.float32)
        raw_x = np.r_[np.asarray(item["current_latent"], dtype=np.float32),
                      np.asarray(item["E1_current_physical_estimate"], dtype=np.float32),
                      action.reshape(-1)]
        raw_f = np.asarray([plan["fork"]["F1_true_N"], plan["fork"]["F2_true_N"],
                            plan["fork"]["R2_true_N"]], dtype=np.float32)
        stored_f = np.asarray([p["F1_true_N"][index], p["F2_true_N"][index],
                               p["R2_true_N"][index]], dtype=np.float32)
        if (not np.array_equal(p["x"][index], raw_x) or
                not np.array_equal(p["action_pair"][index], action) or
                not np.array_equal(raw_f, stored_f) or
                raw_f[2] != max(raw_f[0], raw_f[1])):
            raise RuntimeError(f"G9.6 G9.5 F1/F2/x row mismatch {index}")
        max_abs = max(max_abs, float(np.max(np.abs(raw_f-stored_f))))
        counts[group] += 1
    for group in GROUPS:
        if len([k for k in cache if k[0] == group]) != {"train": 560, "validation": 120,
                                                        "development_test": 120}[group]:
            raise RuntimeError(f"G9.6 state coverage changed in {group}")
    if counts != {"train": 5038, "validation": 1080, "development_test": 1075}:
        raise RuntimeError("G9.6 complete-row split count changed")
    censored = sum(not plan["fork"]["complete_two_intervals"]
                   for item in cache.values() for plan in item["plans"])
    if censored != 7 or len(keys) != 7193:
        raise RuntimeError("G9.6 censored/completed state coverage changed")
    return {"planner_complete_rows": len(keys), "planner_censored_forks": censored,
            "planner_states": len(cache), "planner_rows_by_group": counts,
            "max_abs_original_json_vs_npz_force_N": max_abs}


def check_behavior_rows(d):
    with (G2 / "step_level_force.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    archived = {(int(row["seed"]), int(row["step_index"])): row for row in rows}
    if len(archived) != 15500 or len(d["seed"]) != 14300:
        raise RuntimeError("G9.6 archived behavior count changed")
    for index, (seed, decision) in enumerate(zip(d["seed"], d["decision_index"])):
        pair = []
        forces = []
        for step in (int(decision), int(decision)+1):
            row = archived[(int(seed), step)]
            pair.append(np.asarray(json.loads(row["selected_action_json"]), dtype=np.float32))
            forces.append(np.float32(row["distal_contact_force_max_N"]))
        if (not np.array_equal(d["behavior_action_sequence"][index, :2], np.asarray(pair)) or
                not np.array_equal(d["future_force_sequence"][index, :2], np.asarray(forces))):
            raise RuntimeError(f"G9.6 behavior F1/F2/action alignment failed {seed}/{decision}")
    return {"behavior_complete_rows": len(d["seed"]),
            "behavior_force_action_alignment": "EXACT_FLOAT32_TO_ARCHIVED_G2"}


def prepare():
    source = frozen_sources()
    if OUT.exists():
        raise FileExistsError(OUT)
    bx, by, bm, p, py, pm, mean, scale = training_arrays()
    plan_audit = check_plan_rows(p)
    with np.load(G9 / "development_temporal_behavior.npz") as z:
        d = {key: z[key].copy() for key in z.files}
    behavior_audit = check_behavior_rows(d)
    if (not np.array_equal(np.max(py, axis=1).astype(np.float32), p["R2_true_N"]) or
            not np.all(np.isfinite(bx)) or not np.all(np.isfinite(p["x"]))):
        raise RuntimeError("G9.6 profile target/input finite max alignment failed")
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "models/scalar_baseline").mkdir(parents=True)
    for index in (1, 2, 3):
        (OUT / "models" / f"stepwise_seed_{index}").mkdir()
    json_new(OUT / "metadata.json", {"phase": "G9.6", "branch": "feature/generic-contact-force-safety",
        "runtime": {"python": sys.executable, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                    "cuda_device_index": 0, "cuda_device_name": torch.cuda.get_device_name(0)},
        "frozen_G95_sources": source, "F_lim_N": F_LIM, "H": 2,
        "training_seeds": SEEDS, "confirmation_seeds_unopened": CONFIRM,
        "only_methodological_change": "scalar max R2 target -> separate F1/F2 quantile heads"})
    json_new(OUT / "g95_data_audit.json", {"status": "PASS_EXACT_G95_F1_F2_ALIGNMENT",
        **plan_audit, **behavior_audit,
        "g95_dataset_sha256": sha(G95 / "development_planner_dataset.npz"),
        "g9_behavior_dataset_sha256": sha(G9 / "development_temporal_behavior.npz"),
        "g95_split_sha256": sha(G95 / "dataset_split.json"),
        "g95_parent_audit_sha256": sha(G95 / "development_parent_replay_audit.json"),
        "G95_confirmation_not_used_for_G96_training": True,
        "all_complete_planner_R2_equal_max_F1_F2": True})
    json_new(OUT / "model_definition.json", {"status": "PREDECLARED_BEFORE_G96_TRAINING",
        "input": "exact 298D G9.5 M1: 256D latent + 38D frozen E1 current-state estimate + original 2x2 action plan",
        "architecture": "Linear(298,64)-ReLU-Linear(64,64)-ReLU-Linear(64,8); reshape 2x4; independent per-step cumulative softplus quantiles",
        "per_step_quantiles": TAUS, "target": "[F1_true_N,F2_true_N], direct two complete true-SOFA control intervals",
        "loss": "mean pinball over F1 four quantiles + mean pinball over F2 four quantiles, equal step weighting",
        "optimizer": "Adam lr=0.001 batch=512 max_epochs=60 patience=8; validation min pinball improves by >1e-5",
        "normalization": "exact frozen G9 E1/H2 behavior-TRAIN input mean/std",
        "train_validation_distribution": "same behavior+planner rows as G9.5 M1, same episode/state split",
        "training_seeds": SEEDS, "no_best_seed_selection": True,
        "scalar_baseline": "frozen G9.5 M1 planner-conditioned three checkpoints and per-seed validation correction",
        "stepwise_decision": "Q95(F1)+delta1<=F_lim AND Q95(F2)+delta2<=F_lim",
        "ranking_diagnostic_only": "max(corrected Q95(F1),corrected Q95(F2))",
        "temporal_categories": ["SAFE-SAFE", "SAFE-HIGH", "HIGH-SAFE", "HIGH-HIGH"],
        "no_future_behavior_action_as_planner_candidate": True})
    json_new(OUT / "calibration_definition.json", {"status": "PREDECLARED_BEFORE_G96_TRAINING_AND_DEVELOPMENT_TEST",
        "selection_data": "VALIDATION behavior+planner complete H2 rows only",
        "per_step_rule": "if mean raw Q95 coverage across three fixed replicas is <0.95, shared delta=max(0, pooled seed-by-row validation residual 95th percentile with higher order statistic); otherwise delta=0",
        "shared_across_three_seeds": True, "one_delta_per_step": True,
        "no_development_test_or_confirmation_calibration": True})
    json_new(OUT / "models/scalar_baseline/manifest.json", {"status": "FROZEN_G95_M1_SCALAR_BASELINE_NO_RETRAIN",
        "replicas": source["scalar_baseline"]})
    print("G9.6 exact G9.5 F1/F2/action/feature audit PASS: 7193 complete planner rows, 7 censored; 14300 behavior rows", flush=True)


if __name__ == "__main__":
    prepare()
