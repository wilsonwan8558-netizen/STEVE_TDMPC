"""Frozen M0/M1 and exact-G9 scoring on untouched true planner-plan confirmation."""

from __future__ import annotations

import argparse
import json

import numpy as np

from g95_common import OUT, CONFIRM, F_LIM, PLANS_PER_STATE, SEEDS, json_new
from g95_confirmation_data import verify_freeze
from g95_eval import csv_new, load_replica, calculate
from g9_risk import load_model as load_g9, predict


def assemble():
    verify_freeze()
    audit = json.loads((OUT / "confirmation_parent_replay_audit.json").read_text())
    if (audit["status"] != "PASS" or audit["planner_states"] != 120 or
            audit["planner_candidate_forks"] != 120*PLANS_PER_STATE or
            audit["ordered_reused_scene_seeds"] != list(CONFIRM)):
        raise RuntimeError("G9.5 prospective confirmation parent/candidate audit incomplete")
    selection = json.loads((OUT / "confirmation_state_selection.json").read_text())
    fields = ("x", "R2_true_N", "F1_true_N", "F2_true_N", "seed", "decision_index",
              "state_global_id", "candidate_id", "source", "planner_score", "action_pair")
    records = {key: [] for key in fields}
    state_summary = []
    censored = 0
    for state in selection["states"]:
        sid = int(state["state_id"])
        item = json.loads((OUT / "confirmation_states" / f"state_{sid:04d}.json").read_text())
        if ((item["state_id"], item["seed"], item["decision_index"]) !=
                (sid, state["seed"], state["decision_index"]) or len(item["plans"]) != PLANS_PER_STATE):
            raise RuntimeError(f"G9.5 confirmation state identity changed {sid}")
        latent = np.asarray(item["current_latent"], dtype=np.float32)
        estimate = np.asarray(item["E1_current_physical_estimate"], dtype=np.float32)
        if latent.shape != (256,) or estimate.shape != (38,):
            raise RuntimeError(f"G9.5 confirmation E1 current-state feature changed {sid}")
        complete = safe = 0
        for candidate_id, plan in enumerate(item["plans"]):
            if int(plan["candidate_id"]) != candidate_id:
                raise RuntimeError(f"G9.5 confirmation candidate indexing changed {sid}/{candidate_id}")
            fork = plan["fork"]
            if not fork["complete_two_intervals"]:
                if fork["F2_true_N"] is not None or fork["R2_true_N"] is not None:
                    raise RuntimeError("G9.5 confirmation censored plan incorrectly given R2")
                censored += 1
                continue
            action = np.asarray(plan["actions"], dtype=np.float32)
            x = np.r_[latent, estimate, action.reshape(-1)]
            f1, f2, risk = [float(fork[key]) for key in ("F1_true_N", "F2_true_N", "R2_true_N")]
            if (action.shape != (2, 2) or x.shape != (298,) or not np.all(np.isfinite(x)) or
                    abs(risk-max(f1, f2)) > 1e-8):
                raise RuntimeError("G9.5 confirmation original planner action/force target changed")
            records["x"].append(x)
            records["R2_true_N"].append(risk)
            records["F1_true_N"].append(f1)
            records["F2_true_N"].append(f2)
            records["seed"].append(item["seed"])
            records["decision_index"].append(item["decision_index"])
            records["state_global_id"].append(sid)
            records["candidate_id"].append(candidate_id)
            records["source"].append(plan["source"])
            records["planner_score"].append(np.nan if plan["planner_score"] is None else float(plan["planner_score"]))
            records["action_pair"].append(action)
            complete += 1
            safe += risk <= F_LIM
        state_summary.append({"state_id": sid, "seed": item["seed"],
            "decision_index": item["decision_index"], "candidate_forks": PLANS_PER_STATE,
            "complete_R2": complete, "censored": PLANS_PER_STATE-complete,
            "true_safe_plans": safe})
    if len(records["x"])+censored != 120*PLANS_PER_STATE or censored != audit["censored_one_step_terminal_forks"]:
        raise RuntimeError("G9.5 confirmation complete/censored plan count changed")
    arrays = {}
    for key, values in records.items():
        if key == "source":
            arrays[key] = np.asarray(values, dtype="U40")
        elif key in ("seed", "decision_index", "state_global_id", "candidate_id"):
            arrays[key] = np.asarray(values, dtype=np.int32)
        else:
            arrays[key] = np.asarray(values, dtype=np.float32)
    np.savez_compressed(OUT / "confirmation_planner_dataset.npz", **arrays)
    json_new(OUT / "confirmation_dataset_summary.json", {"status": "UNTOUCHED_PROSPECTIVE_PLANNER_TRUE_SOFA_LABELS",
        "states": 120, "candidate_forks": 120*PLANS_PER_STATE,
        "complete_R2_rows": len(arrays["x"]), "censored_one_step_terminal_rows": censored,
        "state_summary": state_summary,
        "no_future_behavior_actions_as_candidate_input": True})
    print(f"G9.5 untouched prospective planner dataset: {len(arrays['x'])} complete plans, {censored} censored", flush=True)


def evaluate():
    verify_freeze()
    with np.load(OUT / "confirmation_planner_dataset.npz") as z:
        d = {key: z[key].copy() for key in z.files}
    model_rows, state_rows, plan_rows = [], [], []
    models = [(kind, seed) for kind in ("M0_behavior", "M1_planner_conditioned") for seed in SEEDS]
    models.append(("G9_exact_old_E1_H2", SEEDS[0]))
    for kind, seed in models:
        if kind == "G9_exact_old_E1_H2":
            net, mean, scale, delta = load_g9("E1", 2)
        else:
            net, mean, scale, delta = load_replica(kind, seed)
        pred = predict(net, d["x"], mean, scale)
        row, by_state = calculate(kind, seed, d, pred, delta)
        model_rows.append(row)
        state_rows.extend(by_state)
        for i in range(len(d["x"])):
            y = float(d["R2_true_N"][i])
            corrected = float(pred[i, 2]+delta)
            plan_rows.append({"model": kind, "training_seed": seed,
                "episode_seed": int(d["seed"][i]),
                "state_id": int(d["state_global_id"][i]),
                "decision_index": int(d["decision_index"][i]),
                "candidate_id": int(d["candidate_id"][i]),
                "candidate_source": str(d["source"][i]),
                "F1_true_N": float(d["F1_true_N"][i]),
                "F2_true_N": float(d["F2_true_N"][i]),
                "R2_true_N": y,
                "true_safe": int(y <= F_LIM),
                "Q50_N": float(pred[i, 0]), "Q90_N": float(pred[i, 1]),
                "Q95_N": float(pred[i, 2]), "Q99_N": float(pred[i, 3]),
                "corrected_Q95_N": corrected,
                "predicted_safe": int(corrected <= F_LIM)})
        print(f"G9.5 untouched {kind}/{seed}: safe rejected={row['safe_plan_rejected']}/{row['true_safe_plans']}, unsafe accepted={row['unsafe_plan_accepted']}/{row['true_high_risk_plans']}", flush=True)
    csv_new(OUT / "confirmation_plan_results.csv", plan_rows)
    csv_new(OUT / "confirmation_state_metrics.csv", state_rows)
    csv_new(OUT / "confirmation_model_metrics.csv", model_rows)
    old = next(r for r in model_rows if r["model"] == "G9_exact_old_E1_H2")
    paired = []
    for seed in SEEDS:
        m0 = next(r for r in model_rows if r["model"] == "M0_behavior" and r["training_seed"] == seed)
        m1 = next(r for r in model_rows if r["model"] == "M1_planner_conditioned" and r["training_seed"] == seed)
        paired.append({"training_seed": seed,
            "M0_safe_rejection": m0["safe_plan_rejection_rate"],
            "M1_safe_rejection": m1["safe_plan_rejection_rate"],
            "M1_minus_M0_safe_rejection": m1["safe_plan_rejection_rate"]-m0["safe_plan_rejection_rate"],
            "M0_unsafe_acceptance": m0["unsafe_plan_acceptance_rate"],
            "M1_unsafe_acceptance": m1["unsafe_plan_acceptance_rate"],
            "M1_minus_M0_unsafe_acceptance": m1["unsafe_plan_acceptance_rate"]-m0["unsafe_plan_acceptance_rate"],
            "M0_usable_feasible_state_fraction": m0["usable_state_fraction_among_feasible"],
            "M1_usable_feasible_state_fraction": m1["usable_state_fraction_among_feasible"],
            "M1_minus_M0_usable_fraction": m1["usable_state_fraction_among_feasible"]-m0["usable_state_fraction_among_feasible"],
            "M0_mean_within_state_spearman": m0["mean_within_state_spearman"],
            "M1_mean_within_state_spearman": m1["mean_within_state_spearman"]})
    json_new(OUT / "M0_vs_M1_comparison.json", {"status": "UNTOUCHED_PROSPECTIVE_CONFIRMATION_ALL_THREE_SEEDS",
        "same_complete_candidate_rows_for_every_model": len(d["x"]),
        "same_planner_states_for_every_model": len(set(d["state_global_id"])),
        "old_G9_exact_reference": old,
        "paired_replica_comparisons": paired,
        "no_best_seed_selection": True,
        "no_confirmation_training_or_recalibration": True,
        "classification": "See PHASE_G95_REPORT.md after evidence interpretation"})
    print("G9.5 all frozen model confirmation results saved", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("assemble", "evaluate"), required=True)
    stage = parser.parse_args().stage
    {"assemble": assemble, "evaluate": evaluate}[stage]()


if __name__ == "__main__":
    main()
