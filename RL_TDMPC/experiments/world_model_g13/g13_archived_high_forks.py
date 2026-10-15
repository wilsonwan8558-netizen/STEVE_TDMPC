"""True-SOFA G13/G12 candidate test at predeclared archived high-force parents.

Archived G2 behavior parents are off-policy for G13. The G13 planner is
sampled with its normal first-step (cold-start) MPPI distribution at each
parent; this enriched diagnostic is separate from the on-policy study.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
sys.path.insert(0, str(RL))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(RL / "experiments/world_model_g12"))
sys.path.insert(0, str(RL / "experiments/generic_solver_g2"))
sys.path.insert(0, str(RL / "experiments/generic_solver_g3"))
from tdmpc2.agent import TDMPC2Agent  # noqa: E402
from tdmpc2.common import load_torch_checkpoint, set_seed  # noqa: E402
from train import build_agent_config  # noqa: E402
from g12_force_head import ForceHead  # noqa: E402
from collect import check_scene  # noqa: E402
from restore_audit import fork_step, g2_signature, compare  # noqa: E402
from g13_env import make_g13_env  # noqa: E402
from g13_candidate_forks import (CandidateCaptureAgent, predictions,
                                 sha256, OLD_CHECKPOINT, G12, F_LIM)  # noqa: E402

OUT = RL / "logs/world_model_force_aware_planning_g13"
G2 = RL / "logs/generic_solver_phase_g2_force_baseline/step_level_force.csv"
FIELDS = ("seed", "decision_index", "stratum", "candidate_label", "candidate_rank",
          "elite", "action_translation", "action_rotation", "task_score",
          "G13_predicted_force_N", "G12_predicted_force_N", "true_force_N",
          "true_above_F_lim", "reward", "terminated", "truncated",
          "archived_behavior_force_N", "same_parent_fork_verified")


def records_from_planner(agent, weighted_mean, archived_action):
    actions, values = agent.final_actions, agent.final_values
    if actions is None or values is None or actions.shape != (3, 512, 2):
        raise RuntimeError("G13 normal first-step MPPI candidates missing")
    order = torch.argsort(values.squeeze(-1), descending=True, stable=True).cpu().numpy()
    ranks = np.r_[np.linspace(0, 63, 16, dtype=np.int64),
                  np.linspace(64, 511, 16, dtype=np.int64)]
    if len(set(ranks.tolist())) != 32:
        raise RuntimeError("Frozen archived diagnostic rank spread changed")
    records = [("archived_G2_behavior_anchor", -2, False, archived_action, float("nan")),
               ("G13_weighted_mean", -1, False, weighted_mean, float("nan"))]
    for rank in ranks:
        index = int(order[rank])
        records.append((f"sample_{index:03d}", int(rank), bool(rank < 64),
                        actions[0, index].cpu().numpy().astype(np.float32),
                        float(values[index, 0].item())))
    return records


def verify_prior(previous, row, key):
    for field in FIELDS:
        if field == "task_score" and previous[field] == "nan" and np.isnan(row[field]):
            continue
        if field in {"action_translation", "action_rotation", "task_score",
                     "G13_predicted_force_N", "G12_predicted_force_N",
                     "true_force_N", "reward", "archived_behavior_force_N"}:
            if abs(float(previous[field]) - float(row[field])) > 1e-7:
                raise RuntimeError(f"Archived high fork replay mismatch {key}/{field}")
        elif previous[field] != str(row[field]):
            raise RuntimeError(f"Archived high fork replay mismatch {key}/{field}")


def main() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("G13 archived high-force forks require CUDA_VISIBLE_DEVICES=0")
    selection = json.loads((OUT / "checkpoint_selection.json").read_text())
    manifest = json.loads((OUT / "candidate_state_manifest.json").read_text())
    source = json.loads((G12 / "force_dataset_manifest.json").read_text())
    if (selection["selected_step"] != 260000
            or sha256(Path(selection["selected_checkpoint"])) != selection["selected_checkpoint_sha256"]
            or sha256(G2) != source["source_sha256"][str(G2.resolve())]
            or sha256(G12 / "force_transitions.npz") != manifest["source_sha256"]
            or manifest["status"] != "FROZEN_BEFORE_G13_PREDICTIONS_OR_TRUE_SOFA_CANDIDATE_FORKS"
            or manifest["planned_true_sofa_forks_per_state"] != 32
            or manifest["F_lim_N"] != F_LIM):
        raise RuntimeError("Archived high-force parent or selected model identity changed")
    states = {(int(row["seed"]), int(row["decision_index"])): row
              for row in manifest["states"]}
    if len(states) != 40:
        raise RuntimeError("Archived high-force parent manifest changed")
    by_seed = {}
    with G2.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            by_seed.setdefault(int(row["seed"]), []).append(row)
    if list(by_seed) != list(range(90000, 90200)):
        raise RuntimeError("G2 archived reused-scene seed order changed")
    new_checkpoint = load_torch_checkpoint(selection["selected_checkpoint"], map_location="cpu")
    old_checkpoint = load_torch_checkpoint(OLD_CHECKPOINT, map_location="cpu")
    config = new_checkpoint["config"]
    old_config = old_checkpoint["config"]
    metadata = json.loads((OUT / "metadata.json").read_text())
    if (sha256(OLD_CHECKPOINT) != metadata["old_tdmpc2_checkpoint_sha256"]
            or sha256(G12 / "force_head.pt") != metadata["old_force_head_sha256"]):
        raise RuntimeError("Old G12 comparator identity changed")
    force_state = torch.load(G12 / "force_head.pt", map_location="cpu", weights_only=True)
    old_head = ForceHead(force_state["state_dict"]["input_mean"],
                         force_state["state_dict"]["input_std"]).to("cuda:0")
    old_head.load_state_dict(force_state["state_dict"])
    old_head.eval().requires_grad_(False)
    set_seed(90000)
    env = make_g13_env(config["environment"])
    agent = CandidateCaptureAgent(22, 2, build_agent_config(config),
                                  episode_length=int(config["environment"]["max_episode_steps"]),
                                  device=torch.device("cuda:0"))
    agent.load_state_dict(new_checkpoint["agent"], load_optimizers=False)
    old_agent = TDMPC2Agent(14, 2, build_agent_config(old_config),
                            episode_length=int(old_config["environment"]["max_episode_steps"]),
                            device=torch.device("cuda:0"))
    old_agent.load_state_dict(old_checkpoint["agent"], load_optimizers=False)
    agent.model.eval().requires_grad_(False)
    old_agent.model.eval().requires_grad_(False)
    del new_checkpoint, old_checkpoint
    output = OUT / "archived_high_force_candidate_results.csv"
    prior = {}
    if output.exists():
        with output.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                key = (int(row["seed"]), int(row["decision_index"]), row["candidate_label"])
                if key in prior:
                    raise RuntimeError("Duplicate archived high candidate row")
                prior[key] = row
    differences = {}
    selected_count = 0
    parent_steps = 0
    try:
        with output.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            if not prior:
                writer.writeheader()
                stream.flush()
            for seed in range(90000, max(seed for seed, _ in states) + 1):
                set_seed(seed)
                env.action_space.seed(seed)
                observation, _ = env.reset(seed=seed)
                check_scene(env.env)
                for decision, reference in enumerate(by_seed[seed]):
                    expected_obs = np.asarray(json.loads(reference["decision_observation_json"]), dtype=np.float32)
                    if not np.array_equal(observation[:14], expected_obs):
                        raise RuntimeError(f"Archived G2 parent observation mismatch {seed}/{decision}")
                    archived_action = np.asarray(json.loads(reference["selected_action_json"]), dtype=np.float32)
                    state = states.get((seed, decision))
                    if state is not None:
                        if abs(float(state["archived_behavior_force_N"]) -
                               float(reference["distal_contact_force_max_N"])) > 1e-12:
                            raise RuntimeError("Frozen archived behavior force anchor changed")
                        agent.capture = True
                        agent.final_actions = agent.final_values = None
                        weighted_mean = agent.act(observation, first_step=True, eval_mode=True)
                        records = records_from_planner(agent, weighted_mean, archived_action)
                        g13_pred, g12_pred = predictions(agent, old_agent, old_head,
                                                         observation, records)
                        for (label, rank, elite, candidate_action, task_score), p13, p12 in zip(
                                records, g13_pred, g12_pred):
                            result = fork_step(env.env, env.monitor, candidate_action)
                            if label == "archived_G2_behavior_anchor":
                                compare(result, g2_signature(reference), differences,
                                        f"archived high anchor {seed}/{decision}")
                            record = {
                                "seed": seed, "decision_index": decision,
                                "stratum": state["stratum"], "candidate_label": label,
                                "candidate_rank": rank, "elite": elite,
                                "action_translation": float(candidate_action[0]),
                                "action_rotation": float(candidate_action[1]),
                                "task_score": task_score,
                                "G13_predicted_force_N": float(p13),
                                "G12_predicted_force_N": float(p12),
                                "true_force_N": float(result["force_next_max_N"]),
                                "true_above_F_lim": bool(result["force_next_max_N"] > F_LIM),
                                "reward": float(result["reward"]),
                                "terminated": bool(result["terminated"]),
                                "truncated": bool(result["truncated"]),
                                "archived_behavior_force_N": state["archived_behavior_force_N"],
                                "same_parent_fork_verified": True,
                            }
                            key = (seed, decision, label)
                            if key in prior:
                                verify_prior(prior[key], record, key)
                            else:
                                writer.writerow(record)
                                stream.flush()
                        selected_count += 1
                        print("G13_ARCHIVED_HIGH_PARENT", seed, decision, state["stratum"], flush=True)
                    next_obs, reward, terminated, truncated, info = env.step(archived_action)
                    if (not np.array_equal(next_obs[:14],
                                           np.asarray(json.loads(reference["next_observation_json"]), dtype=np.float32))
                            or abs(float(reward) - float(reference["reward"])) > 1e-9
                            or float(info["distal_contact_force_max_N"]) != float(reference["distal_contact_force_max_N"])
                            or bool(terminated) != (reference["terminated"] == "True")
                            or bool(truncated) != (reference["truncated"] == "True")):
                        raise RuntimeError(f"Archived G2 parent transition mismatch {seed}/{decision}")
                    observation = next_obs
                    parent_steps += 1
                if seed % 20 == 0 or seed == max(seed for seed, _ in states):
                    print("G13_ARCHIVED_PARENT_REPLAY", seed, parent_steps, flush=True)
    finally:
        env.close()
    if selected_count != len(states):
        raise RuntimeError("Archived high-force selected parents incomplete")
    (OUT / "archived_high_fork_collection_audit.json").write_text(
        json.dumps({"status": "PASS_EXACT_G2_REUSED_SCENE_PARENT_AND_SAME_PARENT_FORKS",
                    "selected_checkpoint_sha256": selection["selected_checkpoint_sha256"],
                    "selected_parents": selected_count, "archived_parent_steps_replayed": parent_steps,
                    "max_abs_anchor_vs_G2_difference": differences,
                    "G2_source_sha256": sha256(G2),
                    "candidate_sampling": "normal G13 first-step cold-start MPPI distribution: 16 final elites and 16 final nonelites by frozen task-rank spread, plus G13 weighted mean and archived G2 behavior anchor",
                    "interpretation": "true-force-enriched off-policy diagnostic; not combined with on-policy prevalence or success rates"},
                   indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("G13_ARCHIVED_HIGH_FORKS_COMPLETE", selected_count, parent_steps, flush=True)


if __name__ == "__main__":
    main()
