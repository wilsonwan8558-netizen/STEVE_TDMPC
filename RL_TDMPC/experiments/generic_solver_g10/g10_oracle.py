"""Finite true-SOFA closest-safe benchmark with G10's fixed nominal second action."""

from __future__ import annotations

import argparse
import json
import os
import select
import signal
import sys
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
for name in ("generic_solver_g10", "generic_solver_g96", "generic_solver_g95",
             "generic_solver_g85", "generic_solver_g8", "generic_solver_g6a",
             "generic_solver_g3", "generic_solver_g2"):
    sys.path.insert(0, str(RL / "experiments" / name))
sys.path.insert(0, str(RL))

from g10_prepare import OUT, G95, F_LIM, ORACLE_GRID_AXIS, REFINE_RADII, gate  # noqa: E402
from g10_repair import action_key, distance  # noqa: E402
from g10_offline import manifest, DEVELOPMENT_SEEDS  # noqa: E402
from g95_collect import archived  # noqa: E402
from g95_common import json_new  # noqa: E402
from g96_protocol_v2 import captured_plan_v2  # noqa: E402
from g85_oracle import inputs as g85_inputs  # noqa: E402
from g8_prepare import TEST_SEEDS  # noqa: E402
from g6_collect import setup  # noqa: E402
from restore_audit import signature, compare, g2_signature  # noqa: E402
from collect import check_scene  # noqa: E402
from tdmpc2.common import set_seed  # noqa: E402


def oracle_fork(env, monitor, first, second):
    """Short-circuit only an already-high F1; true-safe requires both intervals."""
    first, second = [np.asarray(a, dtype=np.float32) for a in (first, second)]
    if any(a.shape != (2,) or np.any(np.abs(a) > 1+1e-6) for a in (first, second)):
        raise RuntimeError("G10 oracle action outside frozen two-dimensional bounds")
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        try:
            with open(os.devnull, "w") as null:
                os.dup2(null.fileno(), 1)
                os.dup2(null.fileno(), 2)
            one = signature(env, monitor, first)
            f1 = float(one["force_next_max_N"])
            f2 = None
            reason = None
            if f1 > F_LIM:
                reason = "F1_ALREADY_ABOVE_LIMIT"
            elif one["terminated"] or one["truncated"]:
                reason = "TERMINAL_BEFORE_SECOND_INTERVAL"
            else:
                two = signature(env, monitor, second)
                f2 = float(two["force_next_max_N"])
            payload = {"status": "PASS", "F1_N": f1, "F2_N": f2,
                "R2_N": max(f1, f2) if f2 is not None else None,
                "true_safe": f2 is not None and max(f1, f2) <= F_LIM,
                "rejection_or_censor_reason": reason}
        except BaseException:
            payload = {"status": "ERROR", "traceback": traceback.format_exc()}
        try:
            os.write(write_fd, json.dumps(payload, allow_nan=False).encode())
        finally:
            os._exit(0)
    os.close(write_fd)
    ready, _, _ = select.select([read_fd], [], [], 120)
    if not ready:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
        os.close(read_fd)
        raise RuntimeError("G10 true-SOFA oracle fork timed out")
    raw = os.read(read_fd, 1_000_000)
    os.close(read_fd)
    _, status = os.waitpid(pid, 0)
    if not raw:
        raise RuntimeError(f"G10 true-SOFA oracle child empty status={status}")
    result = json.loads(raw)
    if result["status"] != "PASS":
        raise RuntimeError(result["traceback"])
    return result


def closest_safe(env, monitor, record, pair):
    nominal, second = np.asarray(pair, dtype=np.float32)
    grid = np.asarray([(t, r) for t in ORACLE_GRID_AXIS for r in ORACLE_GRID_AXIS], dtype=np.float32)
    observations = []
    safe = []
    for index, first in enumerate(grid):
        outcome = oracle_fork(env, monitor, first, second)
        observations.append({"action": first.tolist(), **outcome})
        if outcome["true_safe"]:
            safe.append((first.copy(), outcome))
        if (index+1) % 100 == 0:
            print(f"G10 oracle {record['cohort']} {record['state_id']}" +
                  (f"/-{record['lookback_steps']}" if record.get("lookback_steps") else "") +
                  f" grid {index+1}/441 safe={len(safe)}", flush=True)
    local_count = 0
    if safe:
        best, best_outcome = min(safe, key=lambda item: (
            distance(item[0], nominal), float(item[0][0]), float(item[0][1])))
        seen = {action_key(action) for action in grid}
        for radius in REFINE_RADII:
            offsets = radius * np.asarray([-1., -.5, 0., .5, 1.], dtype=np.float32)
            local = []
            for dt in offsets:
                for dr in offsets:
                    action = np.clip(best + np.asarray((dt, dr), dtype=np.float32), -1, 1).astype(np.float32)
                    key = action_key(action)
                    if key not in seen:
                        seen.add(key)
                        local.append(action)
            for action in local:
                outcome = oracle_fork(env, monitor, action, second)
                observations.append({"action": action.tolist(), **outcome})
                local_count += 1
                if outcome["true_safe"] and (
                        distance(action, nominal), float(action[0]), float(action[1])) < (
                        distance(best, nominal), float(best[0]), float(best[1])):
                    best, best_outcome = action.copy(), outcome
        action = best.tolist()
        risk = best_outcome["R2_N"]
        deviation = distance(best, nominal)
    else:
        action = risk = deviation = None
    return {"cohort": record["cohort"], "state_id": record["state_id"],
        "seed": record["seed"], "decision_index": record["decision_index"],
        "lookback_steps": record.get("lookback_steps"),
        "nominal_pair": pair.tolist(), "oracle_safe_action_found": bool(safe),
        "oracle_action": action, "oracle_R2_N": risk, "oracle_distance": deviation,
        "grid_evaluations": len(grid), "grid_safe_count": len(safe),
        "local_evaluations": local_count,
        "finite_search_not_global_optimum_or_infeasibility_proof": True,
        "unsafe_F1_short_circuited_without_second_step": True,
        "evaluations": observations}


def run(cohort):
    gate()
    target = "heldout_development_unsafe_nominal" if cohort == "heldout" else "difficult_event_early_intervention"
    records = [r for r in manifest() if r["cohort"] == target]
    selected = {(r["seed"], r["decision_index"]): r for r in records}
    if cohort == "heldout":
        by_seed, _ = archived()
        seeds = list(DEVELOPMENT_SEEDS)
        result_folder = OUT / "oracle_heldout_state_results"
        require_audit = OUT / "heldout_parent_replay_audit.json"
    else:
        _, _, by_seed = g85_inputs()
        seeds = TEST_SEEDS
        result_folder = OUT / "oracle_difficult_event_state_results"
        require_audit = OUT / "difficult_event_parent_replay_audit.json"
    if not require_audit.exists():
        raise RuntimeError("G10 true-SOFA predicted-repair pass must precede oracle search")
    result_folder.mkdir(exist_ok=True)
    env, agent, monitor = setup(seeds)
    checked = parent_steps = 0
    differences = {}
    try:
        for seed in seeds:
            set_seed(seed)
            env.action_space.seed(seed)
            observation, _ = env.reset(seed=seed)
            check_scene(env)
            if not hasattr(monitor, "_g10_oracle_added"):
                env._simulation.root.addObject(monitor)
                monitor._g10_oracle_added = True
            rows = by_seed[seed]
            terminated = truncated = False
            decision = 0
            while not (terminated or truncated):
                if (decision >= len(rows) or not np.array_equal(observation,
                        np.asarray(json.loads(rows[decision]["decision_observation_json"]), dtype=np.float32))):
                    raise RuntimeError(f"G10 oracle parent observation mismatch {seed}/{decision}")
                record = selected.get((seed, decision))
                if record is not None:
                    action, _, _, mean = captured_plan_v2(agent, observation, decision == 0)
                    pair = np.clip(np.asarray(mean[:2], dtype=np.float32), -1, 1)
                    if not np.allclose(pair[0], action, rtol=0, atol=1e-6):
                        raise RuntimeError("G10 oracle executable optimized mean changed")
                    expected = (OUT / ("heldout_state_results" if cohort == "heldout" else
                                       "difficult_event_state_results") /
                                (f"state_{record['state_id']:04d}.json" if cohort == "heldout" else
                                 f"state_{record['state_id']:04d}_minus_{record['lookback_steps']}.json"))
                    repair = json.loads(expected.read_text())
                    if not np.array_equal(pair, np.asarray(repair["nominal_pair"], dtype=np.float32)):
                        raise RuntimeError("G10 oracle and repair nominal pair differ")
                    path = result_folder / expected.name
                    if path.exists():
                        old = json.loads(path.read_text())
                        if old["nominal_pair"] != pair.tolist():
                            raise RuntimeError("G10 oracle replay-resume nominal pair changed")
                    else:
                        old = closest_safe(env, monitor, record, pair)
                        json_new(path, old)
                    checked += 1
                    print(f"G10 oracle {cohort} {checked}/{len(records)} safe_found={old['oracle_safe_action_found']} distance={old['oracle_distance']}", flush=True)
                else:
                    action = agent.act(observation, first_step=decision == 0, eval_mode=True)
                if not np.array_equal(action,
                        np.asarray(json.loads(rows[decision]["selected_action_json"]), dtype=np.float32)):
                    raise RuntimeError("G10 oracle parent nominal action changed")
                parent = signature(env, monitor, action)
                compare(parent, g2_signature(rows[decision]), differences,
                        f"G10 oracle parent {seed}/{decision}")
                observation = np.asarray(parent["observation"], dtype=np.float32)
                terminated, truncated = parent["terminated"], parent["truncated"]
                decision += 1
                parent_steps += 1
            if decision != len(rows):
                raise RuntimeError("G10 oracle parent episode length changed")
    finally:
        env.close()
    if checked != len(records) or len(list(result_folder.glob("state_*.json"))) != len(records):
        raise RuntimeError("G10 oracle finite search incomplete")
    json_new(OUT / f"oracle_{cohort}_parent_replay_audit.json", {"status": "PASS",
        "ordered_reused_scene_seeds": seeds, "parent_steps_compared": parent_steps,
        "states": checked, "max_abs_parent_difference": differences,
        "repaired_or_oracle_action_applied_to_parent": False,
        "confirmation_107000_series_accessed": False})
    print(f"G10 {cohort} finite true-SOFA oracle PASS {checked} states", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("heldout", "events"), required=True)
    arg = parser.parse_args()
    run(arg.cohort)
