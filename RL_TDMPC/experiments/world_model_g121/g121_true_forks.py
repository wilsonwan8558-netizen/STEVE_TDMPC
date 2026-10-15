"""One-step true Generic-SOFA forks from exact G2 reused-scene parents."""

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
from envs.steve_env import make_steve_env  # noqa: E402
from tdmpc2.common import load_torch_checkpoint, set_seed  # noqa: E402
from substep_monitor import DistalForceMonitor  # noqa: E402
from collect import check_scene  # noqa: E402
from restore_audit import fork_step, signature, compare, g2_signature  # noqa: E402
from g121_planner import OUT, G12, CHECKPOINT, EXPECTED_CHECKPOINT_SHA, sha  # noqa: E402

G2 = RL / "logs/generic_solver_phase_g2_force_baseline/step_level_force.csv"


def saved_rows(path: Path) -> dict[tuple[str, str], dict]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    result = {(row["state_id"], row["lambda_F"]): row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError("Duplicate G12.1 true-SOFA fork rows")
    return result


def stable_row(row: dict) -> dict:
    return {key: str(value) for key, value in row.items()}


def main() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or "RTX 5090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("G12.1 true-SOFA fork requires CUDA_VISIBLE_DEVICES=0 RTX 5090")
    manifest = json.loads((G12 / "force_dataset_manifest.json").read_text())
    if (sha(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA or
            sha(G2) != manifest["source_sha256"][str(G2)]):
        raise RuntimeError("Frozen G2 or TD-MPC2 identity changed")
    scale = json.loads((OUT / "lambda_scale_definition.json").read_text())
    if scale["H_force"] != 1:
        raise RuntimeError("G12.1 force horizon changed")
    diagnostics = {}
    with (OUT / "planner_selection_diagnostics.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["classification"] == "FORCE_DISCRIMINATIVE":
                diagnostics.setdefault(row["state_id"], {})[row["lambda_F"]] = row
    states = {(int(state.split("_")[0]), int(state.split("_")[1])): state for state in diagnostics}
    if len(states) != 3 or set(states) != {(90000,46),(90150,62),(90199,77)}:
        raise RuntimeError("G12.1 paired true-SOFA parent state set changed")
    rows_by_seed = {}
    with G2.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            rows_by_seed.setdefault(int(row["seed"]), []).append(row)
    if list(rows_by_seed) != list(range(90000,90200)):
        raise RuntimeError("G2 scene seed order changed")
    checkpoint = load_torch_checkpoint(CHECKPOINT, map_location="cpu")
    config = checkpoint["config"]
    output = OUT / "true_sofa_one_step_forks.csv"
    old = saved_rows(output)
    fields = ["state_id", "seed", "decision_index", "lambda_F", "baseline_or_force_aware",
              "classification", "action_json", "predicted_calibrated_force_N",
              "true_next_step_distal_force_N", "true_above_F_lim", "reward", "success_step",
              "terminated", "truncated", "tip_displacement_mm", "insertion_delta_mm",
              "baseline_true_force_N", "true_force_difference_vs_baseline_N",
              "predicted_force_difference_vs_baseline_N", "reward_difference_vs_baseline",
              "parent_replay_verified", "same_saved_parent_via_fork"]
    differences = {}
    parent_steps = 0
    selected_states = 0
    set_seed(90000)
    env = make_steve_env(config["environment"])
    monitor = DistalForceMonitor(env)
    try:
        with output.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            if not old:
                writer.writeheader()
                stream.flush()
            for seed in range(90000,90200):
                set_seed(seed)
                env.action_space.seed(seed)
                observation, _ = env.reset(seed=seed)
                check_scene(env)
                if not hasattr(monitor, "_g121_added"):
                    env._simulation.root.addObject(monitor)
                    monitor._g121_added = True
                rows = rows_by_seed[seed]
                for decision, reference in enumerate(rows):
                    if (seed, decision) > max(states):
                        break
                    archived_observation = np.asarray(json.loads(reference["decision_observation_json"]), dtype=np.float32)
                    if not np.array_equal(observation, archived_observation):
                        raise RuntimeError(f"G12.1 parent observation mismatch {seed}/{decision}")
                    archived_action = np.asarray(json.loads(reference["selected_action_json"]), dtype=np.float32)
                    state_id = states.get((seed, decision))
                    if state_id is not None:
                        selected_states += 1
                        decisions = diagnostics[state_id]
                        baseline = decisions[str(scale["fixed_lambda_set"][0])]
                        if not np.array_equal(archived_action,
                                np.asarray(json.loads(baseline["selected_first_action"]), dtype=np.float32)):
                            raise RuntimeError("G12.1 baseline planned first action differs from G2")
                        tip_before = np.asarray(env._simulation.dof_positions[0], dtype=np.float64)
                        insertion_before = float(env._simulation.inserted_lengths[0])
                        baseline_fork = fork_step(env, monitor, archived_action)
                        compare(baseline_fork, g2_signature(reference), differences,
                                f"G12.1 baseline fork {state_id}")
                        forked = {str(scale["fixed_lambda_set"][0]): baseline_fork}
                        for weight in scale["fixed_lambda_set"][1:]:
                            key = str(weight)
                            action = np.asarray(json.loads(decisions[key]["selected_first_action"]), dtype=np.float32)
                            if np.array_equal(action, archived_action):
                                raise RuntimeError(f"G12.1 discriminative state had unchanged action {state_id}/{weight}")
                            forked[key] = fork_step(env, monitor, action)
                        repeated = fork_step(env, monitor, archived_action)
                        compare(repeated, baseline_fork, differences,
                                f"G12.1 repeat baseline fork {state_id}")
                        for weight in scale["fixed_lambda_set"]:
                            key = str(weight)
                            plan = decisions[key]
                            result = forked[key]
                            action = np.asarray(json.loads(plan["selected_first_action"]), dtype=np.float32)
                            true_force = float(result["force_next_max_N"])
                            predicted = float(plan["selected_first_action_F_hat_cal_N"])
                            record = {
                                "state_id": state_id, "seed": seed, "decision_index": decision,
                                "lambda_F": weight,
                                "baseline_or_force_aware": "B0" if weight == 0 else "B1",
                                "classification": "FORCE_DISCRIMINATIVE", "action_json": json.dumps(action.tolist(), separators=(",", ":")),
                                "predicted_calibrated_force_N": predicted,
                                "true_next_step_distal_force_N": true_force,
                                "true_above_F_lim": true_force > scale["F_lim_N_nominal"],
                                "reward": float(result["reward"]), "success_step": result["success"],
                                "terminated": result["terminated"], "truncated": result["truncated"],
                                "tip_displacement_mm": float(np.linalg.norm(np.asarray(result["tip_position_mm"])-tip_before)),
                                "insertion_delta_mm": float(result["insertion_length_mm"]-insertion_before),
                                "baseline_true_force_N": baseline_fork["force_next_max_N"],
                                "true_force_difference_vs_baseline_N": true_force-baseline_fork["force_next_max_N"],
                                "predicted_force_difference_vs_baseline_N": predicted-float(baseline["selected_first_action_F_hat_cal_N"]),
                                "reward_difference_vs_baseline": float(result["reward"]-baseline_fork["reward"]),
                                "parent_replay_verified": True, "same_saved_parent_via_fork": True,
                            }
                            old_record = old.get((state_id,key))
                            if old_record is not None:
                                for field in fields:
                                    if old_record[field] != str(record[field]):
                                        raise RuntimeError(f"G12.1 saved fork replay mismatch {state_id}/{key}/{field}")
                            else:
                                writer.writerow(record)
                                stream.flush()
                        print("G121_TRUE_FORK", state_id, [(key, forked[key]["force_next_max_N"])
                              for key in forked], flush=True)
                    # Only a selected step needs a force signature in the live
                    # parent; other archived actions are replayed unchanged.
                    if state_id is not None:
                        parent = signature(env, monitor, archived_action)
                        compare(parent, g2_signature(reference), differences,
                                f"G12.1 parent {state_id}")
                        compare(parent, baseline_fork, differences,
                                f"G12.1 fork-parent {state_id}")
                        observation = np.asarray(parent["observation"], dtype=np.float32)
                        ended = parent["terminated"] or parent["truncated"]
                    else:
                        next_observation, reward, terminated, truncated, info = env.step(archived_action)
                        observation = np.asarray(next_observation, dtype=np.float32)
                        if (not np.array_equal(observation, np.asarray(json.loads(reference["next_observation_json"]), dtype=np.float32)) or
                                abs(float(reward)-float(reference["reward"])) > 1e-9 or
                                bool(terminated) != (reference["terminated"] == "True") or
                                bool(truncated) != (reference["truncated"] == "True") or
                                not np.array_equal(np.asarray(info["raw_action"], dtype=np.float32),
                                                   np.asarray(json.loads(reference["applied_action_json"]), dtype=np.float32))):
                            raise RuntimeError(f"G12.1 parent transition mismatch {seed}/{decision}")
                        ended = bool(terminated or truncated)
                    parent_steps += 1
                    if ended != (decision == len(rows)-1):
                        raise RuntimeError(f"G12.1 parent episode lifecycle mismatch {seed}/{decision}")
                if seed in (90000,90150,90199):
                    print("G121_PARENT_REPLAY", seed, "steps_total", parent_steps, flush=True)
                if seed == max(states)[0]:
                    break
    finally:
        env.close()
    if selected_states != len(states):
        raise RuntimeError("G12.1 selected SOFA states incomplete")
    with (OUT / "plan_fork_parent_replay_audit.json").open("x", encoding="utf-8") as stream:
        json.dump({"status":"PASS_EXACT_REUSED_SCENE_PARENT_AND_FORK",
                   "ordered_seeds_replayed":"90000..90199, stopping after decision 77 of seed 90199",
                   "parent_steps_replayed":parent_steps,"selected_parent_states":selected_states,
                   "baseline_fork_repeat_and_parent_comparison":"PASS at each state",
                   "max_abs_selected_state_difference":differences,
                   "nonselected_parent_checks":"exact pre/post observation and applied action; reward within 1e-9; termination exact",
                   "G2_step_csv_sha256":sha(G2)},stream,indent=2,sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
