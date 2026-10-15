"""Paired original-vs-force-aware CUDA MPPI at frozen development states."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
sys.path.insert(0, str(RL / "experiments/world_model_g12"))
from g12_scale_audit import selected_states  # noqa: E402
from g12_train import OUT as G12  # noqa: E402
from tdmpc2.common import set_seed  # noqa: E402
from g121_planner import OUT, load_agent  # noqa: E402


def save_json(name: str, data: dict) -> None:
    with (OUT / name).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main() -> None:
    scale = json.loads((OUT / "lambda_scale_definition.json").read_text())
    classifications = {}
    with (OUT / "planner_state_classification.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            classifications[row["state_id"]] = row["classification"]
    split = json.loads((G12 / "force_dataset_split.json").read_text())
    with np.load(G12 / "force_transitions.npz", allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    states = selected_states(data, split["validation_seeds"], expanded=True)
    if len(states) != 16 or set(classifications) != {f"{s['seed']}_{s['step_index']}" for s in states}:
        raise RuntimeError("G12.1 development planner states changed")
    agent = load_agent(0.0)
    head = agent.force_head
    rows = []
    for state in states:
        seed, target = state["seed"], state["step_index"]
        state_id = f"{seed}_{target}"
        indices = np.flatnonzero(data["seed"] == seed)
        set_seed(seed)
        agent.lambda_F = 0.0
        agent.trace_enabled = False
        with torch.inference_mode():
            for step in range(target):
                i = int(indices[step])
                action = agent.act(data["observation"][i], first_step=step == 0, eval_mode=True)
                if not np.array_equal(action, data["action"][i]):
                    raise RuntimeError(f"G12.1 original CUDA replay action mismatch {seed}/{step}")
            mean_before = agent.previous_mean.detach().clone()
            cpu_rng = torch.get_rng_state().clone()
            cuda_rng = torch.cuda.get_rng_state().clone()
            results = []
            for weight in scale["fixed_lambda_set"]:
                agent.previous_mean.copy_(mean_before)
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state(cuda_rng)
                agent.lambda_F = float(weight)
                agent.trace_enabled = True
                agent.last_scored = None
                i = int(indices[target])
                action = agent.act(data["observation"][i], first_step=target == 0, eval_mode=True)
                trace = agent.last_scored
                if trace is None:
                    raise RuntimeError("G12.1 final candidate trace missing")
                score = trace["J_total"].squeeze(-1)
                best = int(torch.argmax(score).item())
                z = agent.model.encode(torch.from_numpy(data["observation"][i]).to("cuda:0").unsqueeze(0))
                applied_force = float((head(z, torch.from_numpy(action).to("cuda:0").unsqueeze(0))
                                       + agent.delta_F).item())
                results.append({
                    "state_id": state_id, "classification": classifications[state_id],
                    "seed": seed, "decision_index": target, "lambda_F": float(weight),
                    "selected_first_action": action.tolist(),
                    "selected_first_action_F_hat_cal_N": applied_force,
                    "selected_top_candidate_index": best,
                    "selected_top_candidate_action_sequence": trace["actions"][:, best, :].cpu().numpy().tolist(),
                    "selected_top_candidate_J_task": float(trace["J_task"][best].item()),
                    "selected_top_candidate_F_hat_cal_N": float(trace["F_hat_cal_N"][best].item()),
                    "selected_top_candidate_c_force_N2": float(trace["c_force_N2"][best].item()),
                    "selected_top_candidate_J_total": float(trace["J_total"][best].item()),
                    "final_candidate_J_task_Q90_minus_Q10": float(torch.quantile(trace["J_task"].squeeze(-1), .9)
                                                                  - torch.quantile(trace["J_task"].squeeze(-1), .1)),
                    "final_candidate_c_force_Q90_minus_Q10": float(torch.quantile(trace["c_force_N2"].squeeze(-1), .9)
                                                                  - torch.quantile(trace["c_force_N2"].squeeze(-1), .1)),
                    "final_candidate_nonzero_cost_fraction": float((trace["c_force_N2"] > 0).float().mean()),
                })
            baseline = results[0]
            if not np.array_equal(np.asarray(baseline["selected_first_action"], dtype=np.float32), data["action"][indices[target]]):
                raise RuntimeError(f"G12.1 lambda=0 exact archived-action regression failed {state_id}")
            for result in results:
                first = np.asarray(result["selected_first_action"], dtype=np.float32)
                original = np.asarray(baseline["selected_first_action"], dtype=np.float32)
                result["first_action_changed"] = bool(not np.array_equal(first, original))
                result["first_action_L2_difference"] = float(np.linalg.norm(first-original))
                result["first_action_translation_difference"] = float(first[0]-original[0])
                result["first_action_rotation_difference"] = float(first[1]-original[1])
                result["top_candidate_J_task_difference_vs_baseline"] = (
                    result["selected_top_candidate_J_task"]-baseline["selected_top_candidate_J_task"])
                result["top_candidate_F_hat_cal_difference_vs_baseline_N"] = (
                    result["selected_top_candidate_F_hat_cal_N"]-baseline["selected_top_candidate_F_hat_cal_N"])
                result["applied_action_F_hat_cal_difference_vs_baseline_N"] = (
                    result["selected_first_action_F_hat_cal_N"]-baseline["selected_first_action_F_hat_cal_N"])
                rows.append(result)
        print("G121_PLANNER_STATE", state_id, [(r["lambda_F"],r["first_action_changed"],
              r["applied_action_F_hat_cal_difference_vs_baseline_N"]) for r in results], flush=True)
    with (OUT / "planner_selection_diagnostics.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, separators=(",", ":")) if isinstance(value, list) else value
                             for key, value in row.items()})
    save_json("planner_mppi_reaudit.json", {
        "status": "FULL_CUDA_MPPI_PAIRED_DEVELOPMENT_STATES_NO_SOFA_ACTION_APPLIED",
        "state_count": len(states), "lambda_F_values": scale["fixed_lambda_set"],
        "baseline_lambda_zero_exact_archived_action_regression": "PASS_ALL_16",
        "comparison": "each lambda starts with the same previous optimized mean and identical CPU/CUDA RNG state after exact baseline replay to the parent observation",
        "selection_semantics": "first action is final weighted optimized-mean action; top final-iteration candidate is separately logged and is not the applied action",
        "by_lambda": {str(weight): {
            "action_changed_states": sum(r["first_action_changed"] for r in rows if r["lambda_F"] == weight),
            "force_inactive_changed_states": sum(r["first_action_changed"] for r in rows if r["lambda_F"] == weight and r["classification"] == "FORCE_INACTIVE"),
            "predicted_first_action_force_lower_states": sum(r["applied_action_F_hat_cal_difference_vs_baseline_N"] < 0
                                                              for r in rows if r["lambda_F"] == weight),
        } for weight in scale["fixed_lambda_set"]},
    })


if __name__ == "__main__":
    main()
