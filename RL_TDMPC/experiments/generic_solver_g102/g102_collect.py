"""Expand only G9.5 TRAIN/VALIDATION states with fixed-rotation local SOFA forks."""

from __future__ import annotations

import json
import sys

import numpy as np

HERE = __import__("pathlib").Path(__file__).resolve().parent
RL = HERE.parents[1]
for name in ("generic_solver_g102", "generic_solver_g101", "generic_solver_g10",
             "generic_solver_g96", "generic_solver_g95", "generic_solver_g85",
             "generic_solver_g8", "generic_solver_g6a", "generic_solver_g3", "generic_solver_g2"):
    sys.path.insert(0, str(RL / "experiments" / name))
sys.path.insert(0, str(RL))

from g102_prepare import OUT, G95, points, verify_source  # noqa: E402
from g10_offline import current_state  # noqa: E402
from g95_collect import archived  # noqa: E402
from g95_common import E1Representation, fork_two, json_new, sha  # noqa: E402
from g96_protocol_v2 import captured_plan_v2, canonical_plans  # noqa: E402
from g6_collect import setup  # noqa: E402
from restore_audit import signature, compare, g2_signature  # noqa: E402
from collect import check_scene  # noqa: E402
from tdmpc2.common import set_seed  # noqa: E402


def record_curve(env, monitor, row, archived_state, pair):
    nominal, second = pair
    v_nom, w_nom = float(nominal[0]), float(nominal[1])
    old_fork = archived_state["plans"][8]["fork"]
    if not old_fork["complete_two_intervals"]:
        raise RuntimeError("G10.2 archived G9.5 nominal H2 fork censored")
    true = []
    for v in points(v_nom):
        first = np.asarray((v, w_nom), dtype=np.float32)
        if np.array_equal(first, nominal):
            result = old_fork
            label_source = "EXACT_G95_ARCHIVED_NOMINAL_FORK"
        else:
            result = fork_two(env, monitor, np.stack((first, second)))
            label_source = "NEW_EXACT_PARENT_TRUE_SOFA_FORK"
        true.append({"v": float(v), "delta_v": float(np.float32(v-nominal[0])),
            "F1_true_N": result["F1_true_N"], "F2_true_N": result["F2_true_N"],
            "R2_true_N": result["R2_true_N"],
            "delta_R2_true_N": result["R2_true_N"]-old_fork["R2_true_N"]
                if result["complete_two_intervals"] else None,
            "complete_two_intervals": result["complete_two_intervals"],
            "censor_reason": result["censor_reason"], "label_source": label_source})
    exact_nominal = [r for r in true if r["label_source"] == "EXACT_G95_ARCHIVED_NOMINAL_FORK"]
    if len(exact_nominal) != 1 or exact_nominal[0]["delta_R2_true_N"] != 0:
        raise RuntimeError("G10.2 nominal fork alignment invalid")
    return {"group": row["group"], "state_id": row["state_id"],
        "seed": row["seed"], "decision_index": row["decision_index"],
        "source_sha256": row["source_sha256"],
        "nominal_pair": pair.tolist(), "v_nom": v_nom, "w_nom_fixed": w_nom,
        "second_action_fixed": second.tolist(),
        "R2_nom_true_N": old_fork["R2_true_N"], "true_curve": true,
        "new_true_SOFA_forks": sum(r["label_source"] == "NEW_EXACT_PARENT_TRUE_SOFA_FORK" for r in true),
        "one_interval_censored_forks": sum(not r["complete_two_intervals"] for r in true),
        "no_candidate_applied_to_parent": True}


def run():
    verify_source()
    manifest = json.loads((OUT / "local_dataset_manifest.json").read_text())
    selected = {(int(row["seed"]), int(row["decision_index"])): row
                for row in manifest["train_validation_states"]}
    if len(selected) != 680:
        raise RuntimeError("G10.2 exact 680 training/validation states required")
    by_seed, _ = archived()
    seeds = list(range(90000, 90200))
    env, agent, monitor = setup(seeds)
    mapper = E1Representation()
    checked = parent_steps = true_forks = censored = 0
    differences = {}
    try:
        for seed in seeds:
            set_seed(seed)
            env.action_space.seed(seed)
            observation, _ = env.reset(seed=seed)
            check_scene(env)
            if not hasattr(monitor, "_g102_added"):
                env._simulation.root.addObject(monitor)
                monitor._g102_added = True
            rows = by_seed[seed]
            obs_history, action_history = [], []
            terminated = truncated = False
            decision = 0
            while not (terminated or truncated):
                if (decision >= len(rows) or not np.array_equal(observation,
                        np.asarray(json.loads(rows[decision]["decision_observation_json"]), dtype=np.float32))):
                    raise RuntimeError(f"G10.2 parent preaction observation changed {seed}/{decision}")
                obs_history.append(observation.copy())
                row = selected.get((seed, decision))
                if row is None:
                    action = agent.act(observation, first_step=decision == 0, eval_mode=True)
                else:
                    action, final, scores, mean = captured_plan_v2(agent, observation, decision == 0)
                    candidates, _ = canonical_plans(final, scores, mean, action)
                    pair = np.asarray(candidates[8]["actions"], dtype=np.float32)
                    if not np.array_equal(pair, np.asarray(row["nominal_pair"], dtype=np.float32)):
                        raise RuntimeError("G10.2 G9.5 exact optimized-mean pair changed")
                    path = __import__("pathlib").Path(row["source_path"])
                    if sha(path) != row["source_sha256"]:
                        raise RuntimeError("G10.2 archived G9.5 state hash changed")
                    archived_state = json.loads(path.read_text())
                    x = current_state(agent, mapper, observation, obs_history, action_history, decision)
                    old_x = np.r_[archived_state["current_latent"],
                                  archived_state["E1_current_physical_estimate"]].astype(np.float32)
                    if not np.array_equal(x, old_x):
                        raise RuntimeError("G10.2 deployable latent/E1 state alignment changed")
                    out = OUT / f"{row['group']}_local_curves" / f"state_{int(row['state_id']):04d}.json"
                    if out.exists():
                        result = json.loads(out.read_text())
                        if (result["seed"], result["decision_index"], result["source_sha256"],
                                result["nominal_pair"]) != (
                                seed, decision, row["source_sha256"], pair.tolist()):
                            raise RuntimeError("G10.2 replay-resume state identity changed")
                    else:
                        result = record_curve(env, monitor, row, archived_state, pair)
                        json_new(out, result)
                    checked += 1
                    true_forks += result["new_true_SOFA_forks"]
                    censored += result["one_interval_censored_forks"]
                    if checked % 20 == 0:
                        print(f"G10.2 local expansion {checked}/680 states, "
                              f"forks={true_forks}, censored={censored}, seed={seed}", flush=True)
                if not np.array_equal(action,
                        np.asarray(json.loads(rows[decision]["selected_action_json"]), dtype=np.float32)):
                    raise RuntimeError("G10.2 frozen TD-MPC2 parent action changed")
                parent = signature(env, monitor, action)
                compare(parent, g2_signature(rows[decision]), differences,
                        f"G10.2 parent {seed}/{decision}")
                observation = np.asarray(parent["observation"], dtype=np.float32)
                terminated, truncated = parent["terminated"], parent["truncated"]
                action_history.append(action.copy())
                decision += 1
                parent_steps += 1
            if decision != len(rows):
                raise RuntimeError("G10.2 parent episode length changed")
    finally:
        env.close()
    if checked != 680 or parent_steps != 15500 or sum(len(list(
            (OUT / f"{g}_local_curves").glob("state_*.json"))) for g in ("train", "validation")) != 680:
        raise RuntimeError("G10.2 local expansion incomplete")
    json_new(OUT / "expansion_parent_replay_audit.json", {"status": "PASS_EXACT_REUSED_SCENE_PARENT",
        "ordered_seeds": seeds, "parent_steps_compared": parent_steps,
        "train_validation_states": checked, "new_true_SOFA_forks": true_forks,
        "one_interval_censored_forks": censored,
        "max_abs_parent_difference": differences,
        "all_state_representations_equal_archived_G95": True,
        "all_nominal_pairs_equal_archived_G95": True,
        "nominal_R2_reused_from_exact_G95_true_fork": True,
        "no_candidate_applied_to_parent": True,
        "confirmation_108000_108029_opened": False})
    print(f"G10.2 expansion PASS: {checked} states, {true_forks} new forks, {censored} censored", flush=True)


if __name__ == "__main__":
    run()
