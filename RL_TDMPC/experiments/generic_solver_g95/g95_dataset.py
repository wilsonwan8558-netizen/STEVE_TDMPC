"""Assemble complete true-SOFA planner R2 labels without crossing state splits."""

from __future__ import annotations

import argparse
import json

import numpy as np

from g95_common import OUT, F_LIM, PLANS_PER_STATE, json_new


def build():
    audit = json.loads((OUT / "development_parent_replay_audit.json").read_text())
    if audit["status"] != "PASS" or audit["states"] != 800 or audit["planner_candidate_forks"] != 7200:
        raise RuntimeError("G9.5 full development parent/candidate replay incomplete")
    split = json.loads((OUT / "dataset_split.json").read_text())
    groups = ("train", "validation", "development_test")
    records = {key: [] for key in ("x", "R2_true_N", "F1_true_N", "F2_true_N", "seed",
        "decision_index", "state_global_id", "state_id_within_group", "candidate_id",
        "group_index", "source_index", "planner_score", "action_pair")}
    source_names = ("sampled_high_score", "sampled_medium_score", "sampled_low_score",
                    "sampled_action_diversity", "optimized_mean_executed_first_action")
    by_group = {g: {"states": 0, "candidate_forks": 0, "complete_R2": 0, "censored": 0,
                    "true_safe": 0, "true_high_risk": 0} for g in groups}
    state_summary = []
    global_state = 0
    for group_index, group in enumerate(groups):
        for state in split["selected_development_states"][group]:
            path = OUT / "development_states" / f"state_{group}_{int(state['state_id']):04d}.json"
            item = json.loads(path.read_text())
            if ((item["group"], item["state_id"], item["seed"], item["decision_index"]) !=
                    (group, state["state_id"], state["seed"], state["decision_index"]) or
                    len(item["plans"]) != PLANS_PER_STATE):
                raise RuntimeError(f"G9.5 state identity/candidate count changed {path}")
            latent = np.asarray(item["current_latent"], dtype=np.float32)
            estimate = np.asarray(item["E1_current_physical_estimate"], dtype=np.float32)
            if latent.shape != (256,) or estimate.shape != (38,):
                raise RuntimeError(f"G9.5 E1 current state dimension changed {path}")
            local_complete = local_safe = 0
            by_group[group]["states"] += 1
            for candidate_id, plan in enumerate(item["plans"]):
                if int(plan["candidate_id"]) != candidate_id or plan["source"] not in source_names:
                    raise RuntimeError(f"G9.5 candidate order/source changed {path}")
                by_group[group]["candidate_forks"] += 1
                fork = plan["fork"]
                if not fork["complete_two_intervals"]:
                    if fork["R2_true_N"] is not None or fork["F2_true_N"] is not None:
                        raise RuntimeError("G9.5 censored fork was incorrectly assigned R2")
                    by_group[group]["censored"] += 1
                    continue
                first, second, risk = [float(fork[key]) for key in ("F1_true_N", "F2_true_N", "R2_true_N")]
                if abs(risk-max(first, second)) > 1e-8 or risk < 0:
                    raise RuntimeError("G9.5 direct two-step target changed")
                action = np.asarray(plan["actions"], dtype=np.float32)
                x = np.r_[latent, estimate, action.reshape(-1)]
                if x.shape != (298,) or not np.all(np.isfinite(x)):
                    raise RuntimeError("G9.5 planner risk input invalid")
                records["x"].append(x)
                records["R2_true_N"].append(risk)
                records["F1_true_N"].append(first)
                records["F2_true_N"].append(second)
                records["seed"].append(item["seed"])
                records["decision_index"].append(item["decision_index"])
                records["state_global_id"].append(global_state)
                records["state_id_within_group"].append(item["state_id"])
                records["candidate_id"].append(candidate_id)
                records["group_index"].append(group_index)
                records["source_index"].append(source_names.index(plan["source"]))
                records["planner_score"].append(np.nan if plan["planner_score"] is None else float(plan["planner_score"]))
                records["action_pair"].append(action)
                by_group[group]["complete_R2"] += 1
                local_complete += 1
                if risk <= F_LIM:
                    by_group[group]["true_safe"] += 1
                    local_safe += 1
                else:
                    by_group[group]["true_high_risk"] += 1
            state_summary.append({"group": group, "state_global_id": global_state,
                "state_id_within_group": item["state_id"], "seed": item["seed"],
                "decision_index": item["decision_index"], "candidate_forks": PLANS_PER_STATE,
                "complete_R2": local_complete, "censored": PLANS_PER_STATE-local_complete,
                "true_safe_plans": local_safe})
            global_state += 1
    arrays = {name: np.asarray(values, dtype=(np.int32 if name in ("seed", "decision_index",
        "state_global_id", "state_id_within_group", "candidate_id", "group_index", "source_index") else np.float32))
        for name, values in records.items()}
    if (global_state != 800 or arrays["x"].shape[1] != 298 or
            sum(g["candidate_forks"] for g in by_group.values()) != 7200 or
            sum(g["censored"] for g in by_group.values()) != audit["censored_one_step_terminal_forks"]):
        raise RuntimeError("G9.5 planner development dataset count/shape mismatch")
    if (len(set(zip(arrays["group_index"], arrays["seed"]))) != 200 or
            len(set(arrays["state_global_id"])) > 800):
        raise RuntimeError("G9.5 planner development split identity changed")
    np.savez_compressed(OUT / "development_planner_dataset.npz", **arrays)
    json_new(OUT / "development_dataset_summary.json", {"status": "COMPLETE_G95_PLANNER_TRUE_SOFA_LABELS",
        "state_count": global_state, "candidate_forks": 7200,
        "complete_R2_rows": len(arrays["x"]),
        "censored_one_step_terminal_rows": 7200-len(arrays["x"]),
        "by_group": by_group, "state_summary": state_summary,
        "source_names_by_index": source_names,
        "all_candidate_sequences_from_frozen_planner": True,
        "censored_rows_excluded_from_training_and_metrics": True})
    print(f"G9.5 planner dataset built: 800 states, {len(arrays['x'])} complete R2 plans, {7200-len(arrays['x'])} censored", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("build",), required=True)
    if parser.parse_args().stage == "build":
        build()


if __name__ == "__main__":
    main()
