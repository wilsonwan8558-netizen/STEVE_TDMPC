"""Exact reused-scene Generic-SOFA one-step forks at frozen G12.2 states."""

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
sys.path.insert(0, str(RL.parent))
sys.path.insert(0, str(RL / "experiments/generic_solver_g2"))
sys.path.insert(0, str(RL / "experiments/generic_solver_g3"))
sys.path.insert(0, str(RL / "experiments/world_model_g12"))
from envs.steve_env import make_steve_env  # noqa: E402
from tdmpc2.common import load_torch_checkpoint, set_seed  # noqa: E402
from substep_monitor import DistalForceMonitor  # noqa: E402
from collect import check_scene  # noqa: E402
from restore_audit import fork_step, signature, compare, g2_signature  # noqa: E402
from g12_train import CHECKPOINT, EXPECTED_CHECKPOINT_SHA, F_LIM, OUT as G12, sha  # noqa: E402

OUT = RL / "logs/world_model_force_aware_planning_g122"
G2 = RL / "logs/generic_solver_phase_g2_force_baseline/step_level_force.csv"
FIELDS = [
    "state_id", "seed", "decision_index", "gate_class", "arm", "candidate_index",
    "action_json", "predicted_raw_force_N", "predicted_gate_force_N", "J_task",
    "true_next_step_distal_force_N", "true_above_F_gate", "reward", "success_step",
    "terminated", "truncated", "tip_displacement_mm", "insertion_delta_mm",
    "baseline_true_force_N", "true_force_difference_vs_baseline_N",
    "reward_difference_vs_baseline", "parent_replay_verified", "same_parent_via_fork",
]


def main() -> None:
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1 or
            "RTX 5090" not in torch.cuda.get_device_name(0)):
        raise RuntimeError("G12.2 true-SOFA fork requires CUDA_VISIBLE_DEVICES=0 RTX 5090")
    manifest = json.loads((G12 / "force_dataset_manifest.json").read_text())
    frozen = json.loads((OUT / "fork_selection_manifest.json").read_text())
    if (sha(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA or
            sha(G2) != manifest["source_sha256"][str(G2)] or
            frozen["F_gate_N"] != F_LIM or frozen["H_force"] != 1):
        raise RuntimeError("G12.2 frozen checkpoint, G2 source, or force gate changed")
    states = {(int(r["seed"]), int(r["decision_index"])): r for r in frozen["states"]}
    if len(states) != 9:
        raise RuntimeError("Frozen class-balanced fork state count changed")
    by_seed = {}
    with G2.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            by_seed.setdefault(int(row["seed"]), []).append(row)
    if list(by_seed) != list(range(90000, 90200)):
        raise RuntimeError("Archived G2 seed order changed")
    checkpoint = load_torch_checkpoint(CHECKPOINT, map_location="cpu")
    config = checkpoint["config"]
    output = OUT / "true_sofa_gate_forks.csv"
    old = {}
    if output.exists():
        with output.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                key = (row["state_id"], row["arm"])
                if key in old:
                    raise RuntimeError("Duplicate saved G12.2 fork row")
                old[key] = row
    differences = {}
    set_seed(90000)
    env = make_steve_env(config["environment"])
    monitor = DistalForceMonitor(env)
    selected_count = 0
    parent_steps = 0
    try:
        with output.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            if not old:
                writer.writeheader()
                stream.flush()
            for seed in range(90000, max(s for s, _ in states) + 1):
                set_seed(seed)
                env.action_space.seed(seed)
                observation, _ = env.reset(seed=seed)
                check_scene(env)
                if not hasattr(monitor, "_g122_added"):
                    env._simulation.root.addObject(monitor)
                    monitor._g122_added = True
                rows = by_seed[seed]
                for decision, reference in enumerate(rows):
                    if (seed, decision) > max(states):
                        break
                    expected_observation = np.asarray(json.loads(reference["decision_observation_json"]), dtype=np.float32)
                    if not np.array_equal(observation, expected_observation):
                        raise RuntimeError(f"G12.2 parent observation mismatch {seed}/{decision}")
                    baseline_action = np.asarray(json.loads(reference["selected_action_json"]), dtype=np.float32)
                    state = states.get((seed, decision))
                    if state is not None:
                        if not np.array_equal(baseline_action,
                                              np.asarray(state["baseline_applied_mean_action"], dtype=np.float32)):
                            raise RuntimeError(f"Frozen baseline action mismatch {seed}/{decision}")
                        selected_count += 1
                        tip_before = np.asarray(env._simulation.dof_positions[0], dtype=np.float64)
                        insertion_before = float(env._simulation.inserted_lengths[0])
                        arms = {
                            "B0_APPLIED_MEAN": (baseline_action, None, None, None),
                            "F0_BEST_CANDIDATE": (
                                np.asarray(state["baseline_best_candidate_action"], dtype=np.float32),
                                state["baseline_best_candidate_index"],
                                state["baseline_best_candidate_raw_force_N"],
                                state["baseline_best_candidate_J_task"]),
                        }
                        if state["gate_class"] == "GATE-EMPTY":
                            arms["F1_MIN_FORCE"] = (
                                np.asarray(state["min_force_candidate_action"], dtype=np.float32),
                                state["min_force_candidate_index"],
                                state["min_force_candidate_raw_force_N"],
                                state["min_force_candidate_J_task"])
                        else:
                            arms["GATED_BEST_SAFE"] = (
                                np.asarray(state["best_safe_candidate_action"], dtype=np.float32),
                                state["best_safe_candidate_index"],
                                state["best_safe_candidate_raw_force_N"],
                                state["best_safe_candidate_J_task"])
                        forks = {arm: fork_step(env, monitor, action) for arm, (action, _, _, _) in arms.items()}
                        compare(forks["B0_APPLIED_MEAN"], g2_signature(reference), differences,
                                f"G12.2 baseline fork {seed}/{decision}")
                        compare(fork_step(env, monitor, baseline_action), forks["B0_APPLIED_MEAN"],
                                differences, f"G12.2 repeated fork {seed}/{decision}")
                        baseline = forks["B0_APPLIED_MEAN"]
                        for arm, (action, candidate, raw_force, task_score) in arms.items():
                            result = forks[arm]
                            record = {
                                "state_id": state["state_id"], "seed": seed,
                                "decision_index": decision, "gate_class": state["gate_class"],
                                "arm": arm, "candidate_index": candidate,
                                "action_json": json.dumps(action.tolist(), separators=(",", ":")),
                                "predicted_raw_force_N": raw_force,
                                "predicted_gate_force_N": raw_force + frozen["delta_N"] if raw_force is not None else None,
                                "J_task": task_score,
                                "true_next_step_distal_force_N": result["force_next_max_N"],
                                "true_above_F_gate": result["force_next_max_N"] > F_LIM,
                                "reward": result["reward"], "success_step": result["success"],
                                "terminated": result["terminated"], "truncated": result["truncated"],
                                "tip_displacement_mm": float(np.linalg.norm(np.asarray(result["tip_position_mm"]) - tip_before)),
                                "insertion_delta_mm": float(result["insertion_length_mm"] - insertion_before),
                                "baseline_true_force_N": baseline["force_next_max_N"],
                                "true_force_difference_vs_baseline_N": result["force_next_max_N"] - baseline["force_next_max_N"],
                                "reward_difference_vs_baseline": result["reward"] - baseline["reward"],
                                "parent_replay_verified": True, "same_parent_via_fork": True,
                            }
                            previous = old.get((record["state_id"], arm))
                            if previous is not None:
                                for key in FIELDS:
                                    if previous[key] != str(record[key]):
                                        raise RuntimeError(f"Saved fork replay mismatch {record['state_id']}/{arm}/{key}")
                            else:
                                writer.writerow(record)
                                stream.flush()
                        print("G122_TRUE_FORK", state["state_id"], state["gate_class"],
                              {arm: forks[arm]["force_next_max_N"] for arm in arms}, flush=True)
                        parent = signature(env, monitor, baseline_action)
                        compare(parent, g2_signature(reference), differences,
                                f"G12.2 selected parent {seed}/{decision}")
                        compare(parent, baseline, differences,
                                f"G12.2 fork-parent {seed}/{decision}")
                        observation = np.asarray(parent["observation"], dtype=np.float32)
                        ended = parent["terminated"] or parent["truncated"]
                    else:
                        next_obs, reward, terminated, truncated, info = env.step(baseline_action)
                        observation = np.asarray(next_obs, dtype=np.float32)
                        if (not np.array_equal(observation, np.asarray(json.loads(reference["next_observation_json"]), dtype=np.float32)) or
                                abs(float(reward) - float(reference["reward"])) > 1e-9 or
                                bool(terminated) != (reference["terminated"] == "True") or
                                bool(truncated) != (reference["truncated"] == "True") or
                                not np.array_equal(np.asarray(info["raw_action"], dtype=np.float32),
                                                   np.asarray(json.loads(reference["applied_action_json"]), dtype=np.float32))):
                            raise RuntimeError(f"G12.2 parent transition mismatch {seed}/{decision}")
                        ended = bool(terminated or truncated)
                    parent_steps += 1
                    if ended != (decision == len(rows) - 1):
                        raise RuntimeError(f"G12.2 parent episode lifecycle mismatch {seed}/{decision}")
                print("G122_PARENT_REPLAY", seed, parent_steps, flush=True)
    finally:
        env.close()
    if selected_count != len(states):
        raise RuntimeError("Selected G12.2 fork parents incomplete")
    with (OUT / "true_sofa_gate_parent_replay_audit.json").open("x", encoding="utf-8") as stream:
        json.dump({"status": "PASS_EXACT_REUSED_SCENE_PARENT_AND_FORK",
                   "selected_parent_states": selected_count,
                   "parent_steps_replayed": parent_steps,
                   "max_abs_selected_state_difference": differences,
                   "G2_step_csv_sha256": sha(G2)}, stream, indent=2, sort_keys=True,
                  allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
