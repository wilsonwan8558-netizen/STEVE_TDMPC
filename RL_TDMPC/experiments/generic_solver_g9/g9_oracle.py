"""True Generic-SOFA finite open-loop recoverability at G8.5 prior states.

The parent exactly replays all 30 G8 reused-scene episodes in order. Each
counterfactual trajectory runs in a copy-on-write child; none modifies parent.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import select
import signal
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
for folder in ("generic_solver_g85", "generic_solver_g8", "generic_solver_g6a",
               "generic_solver_g5", "generic_solver_g3", "generic_solver_g2"):
    sys.path.insert(0, str(RL / "experiments" / folder))

from g85_oracle import inputs, F_LIM, TOL, sha, json_new  # noqa: E402
from g8_prepare import TEST_SEEDS, cuda_gate  # noqa: E402
from g6_collect import setup  # noqa: E402
from restore_audit import signature, compare, g2_signature  # noqa: E402
from collect import check_scene  # noqa: E402
from protocol import generate_candidates  # noqa: E402
from tdmpc2.common import set_seed  # noqa: E402
import g5_confirm  # noqa: E402

OUT = RL / "logs/generic_solver_phase_g9_temporal_anticipatory_safety"
G8 = RL / "logs/generic_solver_phase_g8_offline_quantile_shield"
PRIOR = (1, 2, 3, 5)
DIFFICULT = tuple(json.loads((OUT / "difficult_event_manifest.json").read_text())["state_ids"])
COARSE = tuple(float(x) for x in (-1, -.5, 0, .5, 1))
WITHDRAW = ((-1., -1.), (-1., 0.), (-1., 1.), (-.5, 0.))


def fork_trajectory(env, monitor, actions, stop_when_high=False):
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 2 or np.any(~np.isfinite(actions)) or np.any(np.abs(actions) > 1):
        raise RuntimeError("Invalid true-SOFA candidate action sequence")
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        try:
            with open(os.devnull, "w") as null:
                os.dup2(null.fileno(), 1)
                os.dup2(null.fileno(), 2)
            forces = []
            terminated = truncated = success = False
            for action in actions:
                result = signature(env, monitor, action)
                forces.append(float(result["force_next_max_N"]))
                terminated, truncated, success = (bool(result["terminated"]),
                                                   bool(result["truncated"]), bool(result["success"]))
                if terminated or truncated or (stop_when_high and forces[-1] > F_LIM):
                    break
            payload = {"status": "PASS", "forces_N": forces, "steps": len(forces),
                       "terminated": terminated, "truncated": truncated, "success": success}
        except BaseException:
            payload = {"status": "ERROR", "traceback": traceback.format_exc()}
        try:
            os.write(write_fd, json.dumps(payload).encode())
        finally:
            os._exit(0)
    os.close(write_fd)
    ready, _, _ = select.select([read_fd], [], [], 120)
    if not ready:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
        os.close(read_fd)
        raise RuntimeError("G9 Generic multi-step child timed out")
    raw = os.read(read_fd, 1_000_000)
    os.close(read_fd)
    _, child_status = os.waitpid(pid, 0)
    if not raw:
        raise RuntimeError(f"G9 Generic multi-step child returned no data: {child_status}")
    result = json.loads(raw)
    if result["status"] != "PASS":
        raise RuntimeError(result["traceback"])
    result["max_force_N"] = max(result["forces_N"])
    result["complete_or_success"] = result["steps"] == len(actions) or result["success"]
    result["force_safe"] = result["complete_or_success"] and result["max_force_N"] <= F_LIM
    return result


def candidates(nominal, behavior_suffix, plan, state_id, lookback):
    H = lookback + 1  # include the force interval at t*, even for t*-5.
    nominal = np.asarray(nominal, dtype=np.float32)
    behavior_suffix = np.asarray(behavior_suffix, dtype=np.float32)
    seen, output = set(), []

    def add(kind, sequence):
        a = np.asarray(sequence, dtype=np.float32).reshape(H, 2)
        key = a.tobytes()
        if key not in seen:
            seen.add(key)
            output.append((kind, a))

    add("archived_behavior", behavior_suffix)
    if len(plan) >= min(H, 3):
        add("current_TDMPC2_plan_then_archived_suffix", np.vstack((plan[:min(H, 3)], behavior_suffix[min(H, 3):])))
    for action in WITHDRAW:
        add("constant_withdrawal", np.repeat(np.asarray(action, dtype=np.float32)[None, :], H, axis=0))
    first_actions = [np.asarray(c["action"], dtype=np.float32)
                     for c in generate_candidates(nominal, state_id*10+lookback, "confirmation")]
    first_actions += [np.asarray((t, r), dtype=np.float32) for t in COARSE for r in COARSE]
    for action in first_actions:
        add("change_first_then_archived", np.vstack((action[None, :], behavior_suffix[1:])))
        add("constant_action", np.repeat(action[None, :], H, axis=0))
        add("change_first_then_withdraw", np.vstack((action[None, :],
            np.repeat(np.asarray((-1., 0.), dtype=np.float32)[None, :], H-1, axis=0))))
    return output


def search():
    cuda_gate()
    states, _, by_seed = inputs()
    selection = json.loads((OUT / "validation_horizon_selection.json").read_text())
    if selection["status"] != "VALIDATION_ONLY_SELECTION_BEFORE_DEVELOPMENT_TEST_AND_CONFIRMATION":
        raise RuntimeError("G9 validation selection not frozen")
    protocol = {"method": "finite true-SOFA open-loop counterfactuals from exact G8 reused-scene parent",
                "G8_behavior_sha256": sha(G8 / "confirmation_behavior_steps.csv"),
                "event_ids": DIFFICULT, "prior_offsets": PRIOR,
                "oracle_horizon": "lookback+1 intervals, so the force interval at t* is included; t*-5 uses six intervals",
                "candidate_family": "archived behavior, current 3-step TD-MPC2 mean plus archived suffix, four constant withdrawals, G3 exact 21 first actions plus 5x5 coarse global actions with archived/constant/withdrawal continuation",
                "search_order": "as listed; stop after first safe witness, otherwise exhaust finite set",
                "safe_rule": "all attempted force intervals <= F_lim and full horizon completed or successful terminal; failure/truncation before horizon is not safe",
                "causal_caveat": "archived future-action continuations are hindsight existence witnesses, not deployable time-t plans",
                "no_safe_found_caveat": "finite candidate search is not an impossibility proof",
                "F_lim_N": F_LIM, "confirmation_seeds_unopened": True}
    path = OUT / "oracle_early_protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise RuntimeError("G9 early oracle protocol changed during resume")
    else:
        json_new(path, protocol)
    folder = OUT / "oracle_prior_states"
    folder.mkdir(exist_ok=True)
    selected = {}
    for (seed, d), (_, state) in states.items():
        if int(state["state_id"]) not in DIFFICULT:
            continue
        for offset in PRIOR:
            if d-offset < 2:
                raise RuntimeError(f"Insufficient pre-action history for G9 event {state['state_id']}")
            key = (seed, d-offset)
            if key in selected:
                raise RuntimeError(f"Overlapping difficult-event prior states {key}")
            selected[key] = (state, offset)
    g5_confirm.CONFIRM = TEST_SEEDS
    env, agent, monitor = setup(TEST_SEEDS)
    parent_differences = {}
    parent_steps = 0
    try:
        for seed in TEST_SEEDS:
            set_seed(seed)
            env.action_space.seed(seed)
            observation, _ = env.reset(seed=seed)
            check_scene(env)
            if not hasattr(monitor, "_g9_added"):
                env._simulation.root.addObject(monitor)
                monitor._g9_added = True
            rows = by_seed[seed]
            terminated = truncated = False
            d = 0
            while not (terminated or truncated):
                if d >= len(rows) or not np.array_equal(observation,
                    np.asarray(json.loads(rows[d]["decision_observation_json"]), dtype=np.float32)):
                    raise RuntimeError(f"G9 parent pre-action replay mismatch {seed}/{d}")
                nominal = agent.act(observation, first_step=d == 0, eval_mode=True)
                if not np.array_equal(nominal,
                    np.asarray(json.loads(rows[d]["selected_action_json"]), dtype=np.float32)):
                    raise RuntimeError(f"G9 parent selected action mismatch {seed}/{d}")
                item = selected.get((seed, d))
                if item:
                    state, offset = item
                    sid = int(state["state_id"])
                    result_path = folder / f"state_{sid:04d}_minus_{offset}.json"
                    if result_path.exists():
                        prior = json.loads(result_path.read_text())
                        if (prior["state_id"], prior["episode_seed"], prior["decision_index"], prior["lookback_steps"]) != (sid, seed, d, offset):
                            raise RuntimeError("G9 oracle resume row identity changed")
                        print(f"G9 oracle resume skipped {sid}/-{offset}", flush=True)
                    else:
                        behavior = np.asarray([json.loads(r["selected_action_json"]) for r in rows[d:d+offset+1]], dtype=np.float32)
                        if len(behavior) != offset+1 or not np.array_equal(behavior[0], nominal):
                            raise RuntimeError("G9 oracle archived future-action sequence misaligned")
                        plan = agent.previous_mean.detach().cpu().numpy().astype(np.float32)
                        if plan.shape != (3, 2) or not np.array_equal(np.clip(plan[0], -1, 1), nominal):
                            raise RuntimeError("G9 frozen TD-MPC2 CEM mean/action mismatch")
                        plan_rows = []
                        for horizon in (1, 2, 3):
                            outcome = fork_trajectory(env, monitor, plan[:horizon])
                            plan_rows.append({"horizon": horizon, "plan_actions": plan[:horizon].tolist(), **outcome})
                        attempts = 0
                        best = float("inf")
                        witness = None
                        family = candidates(nominal, behavior, plan, sid, offset)
                        for kind, sequence in family:
                            outcome = fork_trajectory(env, monitor, sequence, stop_when_high=True)
                            attempts += 1
                            best = min(best, outcome["max_force_N"])
                            if outcome["force_safe"]:
                                witness = {"kind": kind, "actions": sequence.tolist(), **outcome}
                                break
                        json_new(result_path, {"state_id": sid, "episode_seed": seed,
                            "decision_index": d, "event_decision_index": int(state["decision_index"]),
                            "lookback_steps": offset, "horizon_intervals": offset+1,
                            "candidate_family_size": len(family), "candidates_attempted": attempts,
                            "finite_search_exhausted": witness is None,
                            "safe_trajectory_found": witness is not None,
                            "witness": witness, "minimum_observed_force_N": best,
                            "plan_fork_labels": plan_rows})
                        print(f"G9 oracle state={sid} prior=-{offset} safe={witness is not None} attempted={attempts}/{len(family)}", flush=True)
                parent = signature(env, monitor, nominal)
                compare(parent, g2_signature(rows[d]), parent_differences, f"G9 parent {seed}/{d}")
                observation = np.asarray(parent["observation"], dtype=np.float32)
                terminated, truncated = parent["terminated"], parent["truncated"]
                d += 1
                parent_steps += 1
            if d != len(rows):
                raise RuntimeError(f"G9 parent episode length changed {seed}")
            print(f"G9 oracle parent seed={seed} steps={d}", flush=True)
    finally:
        env.close()
    if parent_steps != 1848 or len(list(folder.glob("state_*_minus_*.json"))) != len(DIFFICULT)*len(PRIOR):
        raise RuntimeError("G9 early-oracle replay or prior-state collection incomplete")
    json_new(OUT / "oracle_parent_replay_audit.json", {"status": "PASS", "parent_steps_compared": parent_steps,
        "ordered_reused_scene_seeds": TEST_SEEDS, "max_abs_difference": parent_differences,
        "prior_states": len(DIFFICULT)*len(PRIOR), "tolerance": TOL})


def aggregate():
    audit = json.loads((OUT / "oracle_parent_replay_audit.json").read_text())
    if audit["status"] != "PASS" or audit["parent_steps_compared"] != 1848:
        raise RuntimeError("G9 early-oracle parent replay has not passed")
    rows, plan_rows = [], []
    for sid in DIFFICULT:
        for offset in PRIOR:
            item = json.loads((OUT / "oracle_prior_states" / f"state_{sid:04d}_minus_{offset}.json").read_text())
            rows.append({k: item[k] for k in ("state_id", "episode_seed", "decision_index", "event_decision_index",
                "lookback_steps", "horizon_intervals", "candidate_family_size", "candidates_attempted",
                "finite_search_exhausted", "safe_trajectory_found", "minimum_observed_force_N")}
                        | {"witness_kind": None if item["witness"] is None else item["witness"]["kind"],
                           "witness_max_force_N": None if item["witness"] is None else item["witness"]["max_force_N"]})
            for p in item["plan_fork_labels"]:
                plan_rows.append({"state_id": sid, "episode_seed": item["episode_seed"],
                    "decision_index": item["decision_index"], "lookback_steps": offset,
                    "horizon": p["horizon"], "plan_actions_json": json.dumps(p["plan_actions"], separators=(",", ":")),
                    "true_plan_R_H_N": p["max_force_N"], "steps_executed": p["steps"],
                    "complete_or_success": p["complete_or_success"], "force_safe": p["force_safe"]})
    for path, data in ((OUT / "oracle_early_recoverability.csv", rows),
                       (OUT / "prospective_plan_true_sofa_labels.csv", plan_rows)):
        with path.open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    print(f"G9 early oracle aggregated {len(rows)} prior states, {len(plan_rows)} matched plans", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("search", "aggregate"), required=True)
    args = parser.parse_args()
    {"search": search, "aggregate": aggregate}[args.stage]()


if __name__ == "__main__":
    main()
