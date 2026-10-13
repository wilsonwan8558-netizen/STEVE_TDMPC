"""Score matched true-SOFA TD-MPC2 plan forks, never behavior-label substitutes."""

from __future__ import annotations

import csv
import json

import numpy as np

from g9_events import archived
from g9_risk import OUT, ESTIMATORS, F_LIM, cuda_gate, load_model, metrics, predict


def csv_new(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    cuda_gate()
    audit = json.loads((OUT / "oracle_parent_replay_audit.json").read_text())
    if audit["status"] != "PASS" or audit["parent_steps_compared"] != 1848:
        raise RuntimeError("True-SOFA plan labels lack exact reused-scene parent replay audit")
    with np.load(OUT / "g8_archived_current_features.npz") as z:
        current = {k: z[k].copy() for k in z.files}
    lookup = {(int(s), int(k)): i for i, (s, k) in enumerate(zip(current["seed"], current["decision_index"]))}
    by_seed, _ = archived()
    with (OUT / "prospective_plan_true_sofa_labels.csv").open(newline="") as stream:
        labels = list(csv.DictReader(stream))
    if len(labels) != 132:
        raise RuntimeError("Missing matched true-SOFA plan labels")
    detail, summary = [], []
    for name in ESTIMATORS:
        for horizon in (1, 2, 3):
            net, mean, scale, delta = load_model(name, horizon)
            cohort = []
            for row in labels:
                if int(row["horizon"]) != horizon or row["complete_or_success"] != "True":
                    continue
                seed, decision = int(row["episode_seed"]), int(row["decision_index"])
                i = lookup[(seed, decision)]
                plan = np.asarray(json.loads(row["plan_actions_json"]), dtype=np.float32)
                behavior = np.asarray([json.loads(r["selected_action_json"])
                    for r in by_seed[seed][decision:decision+horizon]], dtype=np.float32)
                if plan.shape != (horizon, 2) or not np.array_equal(plan[0], behavior[0]):
                    raise RuntimeError(f"TD-MPC2 plan fork indexing mismatch {seed}/{decision}/H{horizon}")
                x = np.r_[current["latent"][i], current[name][i], plan.reshape(-1)]
                cohort.append((row, x, float(np.linalg.norm(plan-behavior))))
            if not cohort:
                raise RuntimeError(f"No complete plan forks for {name}/H{horizon}")
            x = np.asarray([item[1] for item in cohort])
            y = np.asarray([float(item[0]["true_plan_R_H_N"]) for item in cohort])
            p = predict(net, x, mean, scale)
            result = metrics(y, p, delta, name, horizon, "G8_matched_plan_SOFA_forks")
            result["mean_L2_planned_vs_future_behavior_actions"] = float(np.mean([item[2] for item in cohort]))
            result["full_horizon_or_success_count"] = len(cohort)
            summary.append(result)
            for j, (row, _, distance) in enumerate(cohort):
                detail.append({"state_id": int(row["state_id"]), "episode_seed": int(row["episode_seed"]),
                    "decision_index": int(row["decision_index"]), "lookback_steps": int(row["lookback_steps"]),
                    "estimator": name, "horizon": horizon, "true_plan_R_H_N": y[j],
                    "raw_Q50_N": float(p[j, 0]), "raw_Q90_N": float(p[j, 1]),
                    "raw_Q95_N": float(p[j, 2]), "raw_Q99_N": float(p[j, 3]),
                    "corrected_Q95_N": float(p[j, 2]+delta),
                    "true_unsafe": int(y[j] > F_LIM), "predicted_accepted": int(p[j, 2]+delta <= F_LIM),
                    "planned_vs_future_behavior_action_L2": distance})
    csv_new(OUT / "prospective_plan_risk.csv", detail)
    csv_new(OUT / "prospective_plan_prediction_metrics.csv", summary)
    print(f"G9 matched planner-sequence scoring: {len(detail)} estimator-plan rows", flush=True)


if __name__ == "__main__":
    main()
