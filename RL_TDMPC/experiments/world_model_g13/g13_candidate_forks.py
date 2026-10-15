"""Compare G13 and G12 on normal G13 MPPI candidates with true SOFA forks.

This is task-only G13 rollout. Fork children execute one candidate RL interval;
only the parent executes the normal weighted-mean MPPI action.
"""

from __future__ import annotations

import csv
import hashlib
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
sys.path.insert(0, str(RL / "experiments/generic_solver_g3"))
sys.path.insert(0, str(RL / "experiments/generic_solver_g2"))
from tdmpc2.agent import TDMPC2Agent  # noqa: E402
from tdmpc2.common import load_torch_checkpoint, set_seed  # noqa: E402
from train import build_agent_config  # noqa: E402
from g12_force_head import ForceHead  # noqa: E402
from restore_audit import fork_step  # noqa: E402
from collect import check_scene  # noqa: E402
from g13_env import make_g13_env  # noqa: E402
from g13_model import G13Agent  # noqa: E402

OUT = RL / "logs/world_model_force_aware_planning_g13"
G12 = RL / "logs/world_model_force_aware_planning_g12"
OLD_CHECKPOINT = RL / "runs/generic_solver_tdmpc2_baseline_300k_20261006/checkpoints/step_280000.pt"
F_LIM = 3.685331040756903
DECISIONS = (10, 25, 40, 60)
ON_POLICY_SEED_COUNT = 10
FIELDS = ("seed", "decision_index", "candidate_label", "candidate_rank",
          "elite", "action_translation", "action_rotation", "task_score",
          "G13_predicted_force_N", "G12_predicted_force_N", "true_force_N",
          "true_above_F_lim", "reward", "tip_displacement_mm",
          "insertion_delta_mm", "tip_target_distance_normalized_2d",
          "terminated", "truncated", "success_step", "same_parent_fork_verified")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class CandidateCaptureAgent(G13Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capture = False
        self.final_actions = None
        self.final_values = None

    @torch.no_grad()
    def _estimate_value(self, latent, actions):
        value = super()._estimate_value(latent, actions)
        if self.capture:
            # The last of the six calls is the final normal MPPI candidate set.
            self.final_actions = actions.detach().clone()
            self.final_values = value.detach().clone()
        return value


def selected_candidates(actions: torch.Tensor, values: torch.Tensor,
                        baseline_action: np.ndarray):
    if actions.shape != (3, 512, 2) or values.shape != (512, 1):
        raise RuntimeError("Normal G13 MPPI candidate shapes changed")
    order = torch.argsort(values.squeeze(-1), descending=True, stable=True).cpu().numpy()
    inverse_rank = np.empty(512, dtype=np.int64)
    inverse_rank[order] = np.arange(512)
    records = [("baseline_weighted_mean", -1, False,
                np.asarray(baseline_action, dtype=np.float32), float("nan"))]
    for index in range(512):
        rank = int(inverse_rank[index])
        records.append((f"sample_{index:03d}", rank, bool(rank < 64),
                        actions[0, index].cpu().numpy().astype(np.float32),
                        float(values[index, 0].item())))
    return records


def predictions(new_agent, old_agent, old_head, observation, records):
    actions = torch.as_tensor(np.stack([row[3] for row in records]),
                              dtype=torch.float32, device="cuda:0")
    with torch.inference_mode():
        g13_z = new_agent.model.encode(new_agent._tensor_observation(observation))
        g13_pred = new_agent.model.force(new_agent.model.next(
            g13_z.expand(len(records), -1), actions)).squeeze(-1).cpu().numpy()
        old_z = old_agent.model.encode(old_agent._tensor_observation(observation[:14]))
        g12_pred = old_head(old_z.expand(len(records), -1), actions).squeeze(-1).cpu().numpy()
    if not np.all(np.isfinite(g13_pred)) or not np.all(np.isfinite(g12_pred)):
        raise FloatingPointError("Nonfinite candidate force prediction")
    return g13_pred, g12_pred


def compare_record(previous: dict, record: dict, location: str) -> None:
    for key in FIELDS:
        if key == "task_score" and previous[key] == "nan" and np.isnan(record[key]):
            continue
        if key in {"action_translation", "action_rotation", "task_score",
                   "G13_predicted_force_N", "G12_predicted_force_N", "true_force_N",
                   "reward", "tip_displacement_mm", "insertion_delta_mm",
                   "tip_target_distance_normalized_2d"}:
            if abs(float(previous[key]) - float(record[key])) > 1e-7:
                raise RuntimeError(f"G13 candidate replay mismatch {location}/{key}")
        elif previous[key] != str(record[key]):
            raise RuntimeError(f"G13 candidate replay mismatch {location}/{key}")


def main() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("G13 true-SOFA candidate forks require CUDA_VISIBLE_DEVICES=0")
    selection = json.loads((OUT / "checkpoint_selection.json").read_text())
    selected_path = Path(selection["selected_checkpoint"])
    if sha256(selected_path) != selection["selected_checkpoint_sha256"]:
        raise RuntimeError("Selected G13 checkpoint changed")
    seeds = selection["ordered_development_seeds"][:ON_POLICY_SEED_COUNT]
    if seeds != list(range(350000, 350000 + ON_POLICY_SEED_COUNT)):
        raise RuntimeError("Frozen G13 candidate-evaluation development seed prefix changed")
    nav_rows = {(int(row["seed"]), int(row["checkpoint_step"])): row
                for row in csv.DictReader((OUT / "navigation_checkpoint_candidates.csv").open(newline="", encoding="utf-8"))}
    new_checkpoint = load_torch_checkpoint(selected_path, map_location="cpu")
    config = new_checkpoint["config"]
    old_checkpoint = load_torch_checkpoint(OLD_CHECKPOINT, map_location="cpu")
    old_config = old_checkpoint["config"]
    metadata = json.loads((OUT / "metadata.json").read_text())
    if (sha256(OLD_CHECKPOINT) != metadata["old_tdmpc2_checkpoint_sha256"]
            or sha256(G12 / "force_head.pt") != metadata["old_force_head_sha256"]
            or metadata["F_lim_N"] != F_LIM):
        raise RuntimeError("Old Generic/G12 comparator or F_lim changed")
    force_state = torch.load(G12 / "force_head.pt", map_location="cpu", weights_only=True)
    old_head = ForceHead(force_state["state_dict"]["input_mean"],
                         force_state["state_dict"]["input_std"]).to("cuda:0")
    old_head.load_state_dict(force_state["state_dict"])
    old_head.eval().requires_grad_(False)
    set_seed(seeds[0])
    env = make_g13_env(config["environment"])
    new_agent = CandidateCaptureAgent(22, 2, build_agent_config(config),
                                      episode_length=int(config["environment"]["max_episode_steps"]),
                                      device=torch.device("cuda:0"))
    new_agent.load_state_dict(new_checkpoint["agent"], load_optimizers=False)
    old_agent = TDMPC2Agent(14, 2, build_agent_config(old_config),
                            episode_length=int(old_config["environment"]["max_episode_steps"]),
                            device=torch.device("cuda:0"))
    old_agent.load_state_dict(old_checkpoint["agent"], load_optimizers=False)
    new_agent.model.eval().requires_grad_(False)
    old_agent.model.eval().requires_grad_(False)
    del new_checkpoint, old_checkpoint
    output = OUT / "candidate_force_discrimination.csv"
    prior = {}
    if output.exists():
        with output.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                key = (int(row["seed"]), int(row["decision_index"]), row["candidate_label"])
                if key in prior:
                    raise RuntimeError("Duplicate saved candidate fork")
                prior[key] = row
    state_count = 0
    candidate_count = 0
    try:
        with output.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            if not prior:
                writer.writeheader()
                stream.flush()
            for seed in seeds:
                set_seed(seed)
                env.action_space.seed(seed)
                observation, _ = env.reset(seed=seed)
                check_scene(env.env)
                episode_reward = 0.0
                success = simulation_error = terminated = truncated = False
                length = 0
                while not (terminated or truncated):
                    selected_state = length in DECISIONS
                    new_agent.capture = selected_state
                    new_agent.final_actions = new_agent.final_values = None
                    action = new_agent.act(observation, first_step=length == 0, eval_mode=True)
                    if selected_state:
                        if new_agent.final_actions is None or new_agent.final_values is None:
                            raise RuntimeError("Normal G13 final MPPI candidates not captured")
                        records = selected_candidates(new_agent.final_actions,
                                                      new_agent.final_values, action)
                        g13_pred, g12_pred = predictions(new_agent, old_agent, old_head,
                                                         observation, records)
                        tip_before = np.asarray(env.env._simulation.dof_positions[0], dtype=np.float64)
                        insertion_before = float(env.env._simulation.inserted_lengths[0])
                        fork_results = []
                        for (label, rank, elite, candidate_action, task_score), p13, p12 in zip(
                                records, g13_pred, g12_pred):
                            result = fork_step(env.env, env.monitor, candidate_action)
                            fork_results.append(result)
                            next_obs = np.asarray(result["observation"], dtype=np.float32)
                            record = {
                                "seed": seed, "decision_index": length,
                                "candidate_label": label, "candidate_rank": rank,
                                "elite": elite,
                                "action_translation": float(candidate_action[0]),
                                "action_rotation": float(candidate_action[1]),
                                "task_score": task_score,
                                "G13_predicted_force_N": float(p13),
                                "G12_predicted_force_N": float(p12),
                                "true_force_N": float(result["force_next_max_N"]),
                                "true_above_F_lim": bool(result["force_next_max_N"] > F_LIM),
                                "reward": float(result["reward"]),
                                "tip_displacement_mm": float(np.linalg.norm(
                                    np.asarray(result["tip_position_mm"], dtype=np.float64) - tip_before)),
                                "insertion_delta_mm": float(result["insertion_length_mm"] - insertion_before),
                                "tip_target_distance_normalized_2d": float(np.linalg.norm(next_obs[:2] - next_obs[10:12])),
                                "terminated": bool(result["terminated"]),
                                "truncated": bool(result["truncated"]),
                                "success_step": bool(result["success"]),
                                "same_parent_fork_verified": True,
                            }
                            key = (seed, length, label)
                            if key in prior:
                                compare_record(prior[key], record, f"{seed}/{length}/{label}")
                            else:
                                writer.writerow(record)
                                stream.flush()
                            candidate_count += 1
                        # Child forks must leave the parent unchanged. The
                        # normal baseline action executes once in the parent.
                        if not np.array_equal(records[0][3], action):
                            raise RuntimeError("Baseline weighted-mean action changed")
                        repeated = fork_step(env.env, env.monitor, action)
                        first = fork_results[0]
                        for key in ("observation", "reward", "terminated", "truncated",
                                    "success", "force_next_max_N", "tip_position_mm",
                                    "insertion_length_mm", "applied_action", "substeps"):
                            if repeated[key] != first[key]:
                                raise RuntimeError(f"Same-parent SOFA fork nondeterminism {seed}/{length}/{key}")
                        state_count += 1
                        print("G13_CANDIDATE_STATE", seed, length, len(records),
                              sum(r["force_next_max_N"] > F_LIM for r in fork_results), flush=True)
                    next_observation, reward, terminated, truncated, info = env.step(action)
                    if selected_state:
                        baseline = fork_results[0]
                        if (not np.array_equal(next_observation[:14],
                                               np.asarray(baseline["observation"], dtype=np.float32))
                                or float(reward) != baseline["reward"]
                                or float(info["distal_contact_force_max_N"]) != baseline["force_next_max_N"]
                                or bool(terminated) != baseline["terminated"]
                                or bool(truncated) != baseline["truncated"]):
                            raise RuntimeError(f"Parent/fork baseline transition mismatch {seed}/{length}")
                    observation = next_observation
                    episode_reward += float(reward)
                    success = success or bool(info.get("is_success", False))
                    simulation_error = simulation_error or bool(info.get("simulation_error", False))
                    length += 1
                expected = nav_rows[(seed, selection["selected_step"])]
                if (int(expected["success"]) != int(success)
                        or int(expected["episode_length"]) != length
                        or abs(float(expected["episode_reward"]) - episode_reward) > 1e-9
                        or int(expected["simulation_error"]) != int(simulation_error)):
                    raise RuntimeError(f"G13 task-only development trajectory changed {seed}")
                print("G13_CANDIDATE_EPISODE", seed, length, int(success), flush=True)
    finally:
        env.close()
    (OUT / "candidate_fork_collection_audit.json").write_text(
        json.dumps({"status": "PASS_NORMAL_G13_MPPI_AND_EXACT_SAME_PARENT_SOFA_FORKS",
                    "selected_checkpoint_sha256": selection["selected_checkpoint_sha256"],
                    "development_seeds": seeds,
                    "fixed_decision_indices": list(DECISIONS),
                    "states": state_count, "candidate_forks": candidate_count,
                    "candidate_sampling": "ALL 512 final normal six-iteration MPPI first actions (24 policy-prior, 488 Gaussian) plus weighted-mean baseline action; no candidate subsampling",
                    "H_force": 1,
                    "off_policy_archived_high_force_parents": "separate predeclared candidate_state_manifest.json; not mixed with on-policy development prevalence"},
                   indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("G13_CANDIDATE_FORKS_COMPLETE", state_count, candidate_count, flush=True)


if __name__ == "__main__":
    main()
