"""Inspect frozen planner candidate objectives before choosing any force weight."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

from g12_evaluate_head import OUT, CHECKPOINT, EXPECTED_CHECKPOINT_SHA, F_LIM, sha, write_csv
from g12_force_head import ForceHead
from g12_train import new_json

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tdmpc2.agent import TDMPC2Agent  # noqa: E402
from tdmpc2.common import load_torch_checkpoint, set_seed  # noqa: E402
from tdmpc2.networks import two_hot_inv  # noqa: E402
from train import build_agent_config  # noqa: E402

WEIGHTS = (0.0, 0.1, 0.5, 1.0)
H_FORCE = 2


def category(value: float) -> str:
    if value == 0:
        return "no_contact"
    if value < 0.5 * F_LIM:
        return "low_force"
    if value <= F_LIM:
        return "near_limit"
    return "above_limit"


def selected_states(data: dict, validation_seeds: list[int], *, expanded: bool) -> list[dict]:
    if expanded:
        result = []
        for label in ("no_contact", "low_force", "near_limit", "above_limit"):
            per_seed = []
            for seed in validation_seeds:
                indices = np.flatnonzero(data["seed"] == seed)
                matching = []
                for position in range(1, len(indices)):
                    force_previous = float(data["force_next_max_N"][indices[position-1]])
                    if category(force_previous) == label:
                        matching.append((force_previous, position, int(indices[position])))
                if matching:
                    matching.sort()
                    per_seed.append((seed, matching[len(matching)//2]))
            if len(per_seed) < 4:
                raise RuntimeError(f"G12 expanded audit has fewer than four distinct seeds for {label}")
            positions = np.linspace(0, len(per_seed)-1, 4, dtype=int)
            for index in positions:
                seed, (force_previous, position, row_index) = per_seed[int(index)]
                result.append({"seed": int(seed), "step_index": position,
                               "row_index": row_index, "force_previous_N": force_previous,
                               "force_state": label})
        return result
    chosen = {key: [] for key in ("no_contact", "low_force", "near_limit", "above_limit")}
    for seed in validation_seeds:
        indices = np.flatnonzero(data["seed"] == seed)
        for position in range(1, len(indices)):
            i = int(indices[position])
            force_previous = float(data["force_next_max_N"][indices[position-1]])
            label = category(force_previous)
            if len(chosen[label]) >= 2 or any(x["seed"] == seed for x in chosen[label]):
                continue
            chosen[label].append({"seed": int(seed), "step_index": position,
                                  "row_index": i, "force_previous_N": force_previous,
                                  "force_state": label})
    if any(len(group) != 2 for group in chosen.values()):
        raise RuntimeError(f"G12 cannot find two validation planner states per force category: {chosen}")
    return [item for group in chosen.values() for item in group]


def candidate_components(agent, head, latent, actions):
    """Mirror frozen _estimate_value and add a side-effect-free H_force cost."""
    returns = torch.zeros(latent.shape[0], 1, device=latent.device, dtype=latent.dtype)
    discount = 1.0
    termination = torch.zeros_like(returns)
    predicted_forces = []
    for step in range(agent.horizon):
        reward_logits = agent.model.reward(latent, actions[step])
        reward = two_hot_inv(reward_logits, agent.model.num_bins,
                             agent.model.value_min, agent.model.value_max)
        if step < H_FORCE:
            predicted_forces.append(head(latent, actions[step]))
        latent = agent.model.next(latent, actions[step])
        returns.add_(discount * (1.0 - termination) * reward)
        discount *= agent.discount
        if bool(agent.config.get("episodic", True)):
            termination = torch.clamp(
                termination + (agent.model.termination(latent) > 0.5).float(),
                max=1.0)
    final_action, _ = agent.model.pi(latent)
    final_q = agent.model.q(latent, final_action, reduction="avg")
    terminal = discount * (1.0 - termination) * final_q
    force_cost = sum(torch.relu(force - F_LIM).square() for force in predicted_forces)
    return returns, terminal, predicted_forces, force_cost


def describe(values: np.ndarray) -> dict:
    return {"min": float(np.min(values)), "q05": float(np.quantile(values, .05)),
            "q25": float(np.quantile(values, .25)), "median": float(np.median(values)),
            "q75": float(np.quantile(values, .75)), "q95": float(np.quantile(values, .95)),
            "max": float(np.max(values))}


def main(iteration: str = "final", *, expanded: bool = False) -> None:
    if iteration not in ("first", "final"):
        raise ValueError(iteration)
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1 or
            "RTX 5090" not in torch.cuda.get_device_name(0)):
        raise RuntimeError("G12 scale audit requires CUDA_VISIBLE_DEVICES=0 RTX 5090")
    manifest = json.loads((OUT / "force_dataset_manifest.json").read_text())
    split = json.loads((OUT / "force_dataset_split.json").read_text())
    definition = json.loads((OUT / "force_head_definition.json").read_text())
    if (sha(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA or
            sha(OUT / "force_transitions.npz") != manifest["joined_dataset_sha256"] or
            sha(OUT / "force_head.pt") != definition["force_head_checkpoint_sha256"]):
        raise RuntimeError("G12 scale audit frozen identity mismatch")
    if expanded:
        audit_name = f"objective_scale_audit_expanded_{iteration}.json"
        candidate_name = f"planner_candidate_diagnostics_expanded_{iteration}.csv"
    else:
        audit_name = "objective_scale_audit.json" if iteration == "final" else "objective_scale_audit_initial.json"
        candidate_name = "planner_candidate_diagnostics.csv" if iteration == "final" else "planner_candidate_diagnostics_initial.csv"
    if (OUT / audit_name).exists():
        raise FileExistsError("G12 objective-scale audit already exists")
    with np.load(OUT / "force_transitions.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    states = selected_states(data, split["validation_seeds"], expanded=expanded)
    checkpoint = load_torch_checkpoint(CHECKPOINT, map_location="cpu")
    config = checkpoint["config"]
    agent = TDMPC2Agent(14, 2, build_agent_config(config),
                        episode_length=int(config["environment"]["max_episode_steps"]),
                        device=torch.device("cuda:0"))
    agent.load_state_dict(checkpoint["agent"], load_optimizers=False)
    force_checkpoint = torch.load(OUT / "force_head.pt", map_location="cpu", weights_only=True)
    state_dict = force_checkpoint["state_dict"]
    head = ForceHead(state_dict["input_mean"], state_dict["input_std"]).to("cuda:0")
    head.load_state_dict(state_dict)
    head.eval()
    head.requires_grad_(False)
    rows = []
    summaries = []
    with torch.inference_mode():
        for state in states:
            seed, target = state["seed"], state["step_index"]
            indices = np.flatnonzero(data["seed"] == seed)
            set_seed(seed)
            captured = []
            original = agent._estimate_value

            def intercept(latent, actions):
                pre_rng = torch.cuda.get_rng_state().clone()
                value = original(latent, actions)
                captured.append((latent.detach().clone(), actions.detach().clone(),
                                 value.detach().clone(), pre_rng))
                return value

            agent._estimate_value = intercept
            try:
                for decision in range(target+1):
                    i = int(indices[decision])
                    action = agent.act(data["observation"][i],
                                       first_step=decision == 0, eval_mode=True)
                    if not np.array_equal(action, data["action"][i]):
                        raise RuntimeError(f"G12 archived CUDA planner action mismatch at {seed}/{decision}")
                    if decision < target:
                        captured.clear()
                if len(captured) != int(config["planning"]["iterations"]):
                    raise RuntimeError("G12 planner iteration count changed")
                z, actions, original_value, pre_rng = captured[0 if iteration == "first" else -1]
                final_mean = agent.previous_mean.detach().clone().cpu().numpy()
            finally:
                agent._estimate_value = original
            post_rng = torch.cuda.get_rng_state().clone()
            try:
                torch.cuda.set_rng_state(pre_rng)
                reward, terminal, force, cost = candidate_components(agent, head, z, actions)
            finally:
                torch.cuda.set_rng_state(post_rng)
            task = (reward + terminal).squeeze(-1).cpu().numpy().astype(np.float64)
            archived = original_value.squeeze(-1).cpu().numpy().astype(np.float64)
            max_difference = float(np.max(np.abs(task - archived)))
            if not np.allclose(task, archived, atol=1e-6, rtol=1e-6):
                raise RuntimeError(f"G12 objective decomposition differs from frozen planner {seed}/{target}: {max_difference}")
            reward_values = reward.squeeze(-1).cpu().numpy().astype(np.float64)
            terminal_values = terminal.squeeze(-1).cpu().numpy().astype(np.float64)
            force_1 = force[0].squeeze(-1).cpu().numpy().astype(np.float64)
            force_2 = force[1].squeeze(-1).cpu().numpy().astype(np.float64)
            costs = cost.squeeze(-1).cpu().numpy().astype(np.float64)
            candidate_actions = actions.cpu().numpy()
            chosen = {}
            for weight in WEIGHTS:
                index = int(np.argmax(task - weight*costs))
                chosen[str(weight)] = {
                    "candidate_index": index,
                    "J_task": float(task[index]),
                    "J_force": float(costs[index]),
                    "J_total": float(task[index]-weight*costs[index]),
                    "F_hat_1_N": float(force_1[index]),
                    "F_hat_2_N": float(force_2[index]),
                    "action_sequence": candidate_actions[:, index, :].tolist(),
                }
            state_id = f"{seed}_{target}"
            for candidate in range(len(task)):
                rows.append({
                    "state_id": state_id, "force_state": state["force_state"],
                    "seed": seed, "decision_index": target, "candidate_index": candidate,
                    "action_sequence_json": json.dumps(candidate_actions[:, candidate, :].tolist(), separators=(",", ":")),
                    "J_TDMPC2": task[candidate],
                    "predicted_reward_contribution": reward_values[candidate],
                    "terminal_Q_contribution": terminal_values[candidate],
                    "F_hat_1_N": force_1[candidate], "F_hat_2_N": force_2[candidate],
                    "J_force_N2": costs[candidate],
                    **{f"lambda_{weight}_times_J_force": weight*costs[candidate] for weight in WEIGHTS},
                    **{f"J_total_lambda_{weight}": task[candidate]-weight*costs[candidate] for weight in WEIGHTS},
                })
            summaries.append({
                **state, "state_id": state_id, "candidate_count": len(task),
                "planner_iterations": len(captured),
                "archived_replayed_first_action": action.tolist(),
                "final_optimized_mean_action_sequence": final_mean.tolist(),
                "objective_decomposition_max_abs_difference": max_difference,
                "J_TDMPC2": describe(task),
                "predicted_reward_contribution": describe(reward_values),
                "terminal_Q_contribution": describe(terminal_values),
                "F_hat_1_N": describe(force_1), "F_hat_2_N": describe(force_2),
                "J_force_N2": describe(costs),
                "penalty_by_lambda": {str(w): describe(w*costs) for w in WEIGHTS},
                "same_candidate_argmax_by_lambda": chosen,
                "J_task_q95_minus_q05": float(np.quantile(task,.95)-np.quantile(task,.05)),
                "J_force_q95_minus_q05": float(np.quantile(costs,.95)-np.quantile(costs,.05)),
            })
    write_csv(OUT / candidate_name, rows)
    new_json(OUT / audit_name, {
        "status": "MEASURED_BEFORE_G12_PLANNER_INTEGRATION_OR_LARGE_ROLLOUT",
        "planner_state_selection": (
            "four distinct validation seeds spread evenly through the category-eligible seed list; within each seed use the median previous-force state"
            if expanded else
            "first two distinct validation episode seeds per category of completed previous-step true force, in sorted seed/decision order"),
        "categories": "no contact 0; low 0<F<0.5F_lim; near 0.5F_lim<=F<=F_lim; above F>F_lim",
        "states": len(states), "candidate_rows": len(rows),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "force_head_checkpoint_sha256": definition["force_head_checkpoint_sha256"],
        "H_force": H_FORCE,
        "cost": "J_force=sum_(k=1..2) max(F_hat_k-F_lim,0)^2; units nominal N^2",
        "combined_objective": "J_total=J_TDMPC2-lambda_F*J_force; maximize",
        "proposed_lambda_F_values": list(WEIGHTS),
        "candidate_set": f"{iteration} iteration of exact frozen six-iteration CUDA MPPI planner, 512 candidates per state; no planner action modified",
        "selection_comparison": "same-candidate argmax only, not a rerun or replacement of MPPI optimized-mean action",
        "per_state": summaries,
    })
    print("G12_OBJECTIVE_SCALE_MEASURED", iteration, "expanded" if expanded else "initial_sample", [(s["state_id"],s["J_TDMPC2"]["median"],
          s["J_force_N2"]["median"],s["J_force_N2"]["q95"]) for s in summaries], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", choices=("first", "final"), default="final")
    parser.add_argument("--expanded", action="store_true")
    args = parser.parse_args()
    main(args.iteration, expanded=args.expanded)
