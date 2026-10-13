"""Ordered reused-scene G10 offline true-SOFA repair and difficult-event checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
for name in ("generic_solver_g10", "generic_solver_g96", "generic_solver_g95",
             "generic_solver_g85", "generic_solver_g8", "generic_solver_g6a",
             "generic_solver_g3", "generic_solver_g2"):
    sys.path.insert(0, str(RL / "experiments" / name))
sys.path.insert(0, str(RL))

from g10_prepare import OUT, G95, F_LIM, gate  # noqa: E402
from g10_repair import FrozenRepair  # noqa: E402
from g95_collect import archived  # noqa: E402
from g95_common import E1Representation, fork_two, json_new, sha  # noqa: E402
from g96_protocol_v2 import captured_plan_v2, canonical_plans  # noqa: E402
from g85_oracle import inputs as g85_inputs  # noqa: E402
from g8_prepare import TEST_SEEDS  # noqa: E402
from g6_collect import setup  # noqa: E402
from restore_audit import signature, compare, g2_signature  # noqa: E402
from collect import check_scene  # noqa: E402
from tdmpc2.common import set_seed  # noqa: E402

DEVELOPMENT_SEEDS = tuple(range(90000, 90200))


def manifest():
    gate()
    record = json.loads((OUT / "offline_state_manifest.json").read_text())
    if (record["status"] != "SELECTED_BEFORE_G10_REPAIR_ORACLE_OUTCOMES" or
            record["heldout_unsafe_nominal_count"] != 18 or
            record["difficult_event_parent_state_count"] != 22 or
            len(record["states"]) != 40):
        raise RuntimeError("G10 offline state manifest changed")
    return record["states"]


def current_state(agent, mapper, observation, obs_history, action_history, decision):
    history_obs = np.asarray(obs_history[decision-2:decision+1], dtype=np.float32)
    history_action = np.asarray(action_history[decision-2:decision], dtype=np.float32)
    if history_obs.shape != (3, 14) or history_action.shape != (2, 2):
        raise RuntimeError("G10 frozen E1 pre-action history unavailable")
    with torch.inference_mode():
        latent = agent.model.encode(agent._tensor_observation(observation))[0].cpu().numpy()
    physical = mapper.estimate(history_obs, history_action)
    x = np.r_[latent, physical].astype(np.float32)
    if x.shape != (294,) or not np.all(np.isfinite(x)):
        raise RuntimeError("G10 frozen E1 latent/physical input invalid")
    return x


def force_row(fork):
    return {"F1_N": fork["F1_true_N"], "F2_N": fork["F2_true_N"],
            "R2_N": fork["R2_true_N"], "complete_two_intervals": fork["complete_two_intervals"],
            "true_safe": fork["complete_two_intervals"] and fork["R2_true_N"] <= F_LIM,
            "censor_reason": fork["censor_reason"]}


def evaluate_state(env, monitor, solver, record, x, pair, archived_state=None):
    proposal = solver.solve(x, pair)
    nominal = force_row(fork_two(env, monitor, pair))
    if archived_state is not None:
        expected = archived_state["plans"][8]["fork"]
        if (not nominal["complete_two_intervals"] or
                any(np.float32(nominal[key]) != np.float32(expected[old]) for key, old in
                    (("F1_N", "F1_true_N"), ("F2_N", "F2_true_N"), ("R2_N", "R2_true_N")))):
            raise RuntimeError("G10 nominal true-SOFA fork differs from archived G9.5")
    repaired = None
    if proposal["status"] == "PREDICTED_SAFE_REPAIR":
        repaired_pair = np.vstack((np.asarray(proposal["action"], dtype=np.float32), pair[1]))
        repaired = force_row(fork_two(env, monitor, repaired_pair))
    elif proposal["status"] == "NOMINAL_PREDICTED_SAFE_NO_REPAIR":
        repaired = nominal
    fallback_pair = np.vstack((np.zeros(2, dtype=np.float32), pair[1]))
    fallback = force_row(fork_two(env, monitor, fallback_pair))
    return {"cohort": record["cohort"], "state_id": record["state_id"],
        "seed": record["seed"], "decision_index": record["decision_index"],
        "event_decision_index": record.get("event_decision_index"),
        "lookback_steps": record.get("lookback_steps"),
        "nominal_pair": pair.tolist(), "proposal": proposal,
        "nominal_true": nominal, "repaired_true": repaired,
        "fallback_zero_true": fallback,
        "repaired_action_executed_only_in_counterfactual_fork": True,
        "parent_scene_unchanged_by_forks": True}


def heldout():
    records = [r for r in manifest() if r["cohort"] == "heldout_development_unsafe_nominal"]
    selected = {(r["seed"], r["decision_index"]): r for r in records}
    folder = OUT / "heldout_state_results"
    folder.mkdir(exist_ok=True)
    by_seed, _ = archived()
    mapper = E1Representation()
    solver = FrozenRepair()
    env, agent, monitor = setup(list(DEVELOPMENT_SEEDS))
    checked = 0
    parent_steps = 0
    differences = {}
    try:
        for seed in DEVELOPMENT_SEEDS:
            set_seed(seed)
            env.action_space.seed(seed)
            observation, _ = env.reset(seed=seed)
            check_scene(env)
            if not hasattr(monitor, "_g10_heldout_added"):
                env._simulation.root.addObject(monitor)
                monitor._g10_heldout_added = True
            rows = by_seed[seed]
            obs_history, action_history = [], []
            terminated = truncated = False
            decision = 0
            while not (terminated or truncated):
                if (decision >= len(rows) or not np.array_equal(observation,
                        np.asarray(json.loads(rows[decision]["decision_observation_json"]), dtype=np.float32))):
                    raise RuntimeError(f"G10 heldout pre-action parent observation changed {seed}/{decision}")
                obs_history.append(observation.copy())
                record = selected.get((seed, decision))
                if record is not None:
                    action, final_actions, scores, mean = captured_plan_v2(agent, observation, decision == 0)
                    candidates, numerical = canonical_plans(final_actions, scores, mean, action)
                    pair = np.asarray(candidates[8]["actions"], dtype=np.float32)
                    old_path = G95 / "development_states" / f"state_development_test_{record['state_id']:04d}.json"
                    old = json.loads(old_path.read_text())
                    if (sha(old_path) != record["source_state_sha256"] or
                            not np.array_equal(pair, np.asarray(record["nominal_pair"], dtype=np.float32)) or
                            not np.array_equal(pair, np.asarray(old["plans"][8]["actions"], dtype=np.float32))):
                        raise RuntimeError("G10 G9.6-v2 nominal optimized mean/candidate changed")
                    x = current_state(agent, mapper, observation, obs_history, action_history, decision)
                    archived_x = np.r_[old["current_latent"], old["E1_current_physical_estimate"]].astype(np.float32)
                    if not np.array_equal(x, archived_x):
                        raise RuntimeError("G10 frozen G9.5 E1 pre-action state changed")
                    path = folder / f"state_{record['state_id']:04d}.json"
                    if path.exists():
                        item = json.loads(path.read_text())
                        if (item["seed"], item["decision_index"], item["nominal_pair"]) != (
                                seed, decision, pair.tolist()):
                            raise RuntimeError("G10 heldout replay-resume state changed")
                    else:
                        item = evaluate_state(env, monitor, solver, record, x, pair, old)
                        item["optimized_mean_numerical_semantics"] = numerical
                        json_new(path, item)
                    checked += 1
                    print(f"G10 heldout state {checked}/18 id={record['state_id']} status={item['proposal']['status']} nominal_R2={item['nominal_true']['R2_N']:.3f} repaired_R2={item['repaired_true']['R2_N'] if item['repaired_true'] else None}", flush=True)
                else:
                    action = agent.act(observation, first_step=decision == 0, eval_mode=True)
                if not np.array_equal(action,
                        np.asarray(json.loads(rows[decision]["selected_action_json"]), dtype=np.float32)):
                    raise RuntimeError("G10 heldout nominal action changed")
                parent = signature(env, monitor, action)
                compare(parent, g2_signature(rows[decision]), differences,
                        f"G10 heldout parent {seed}/{decision}")
                observation = np.asarray(parent["observation"], dtype=np.float32)
                terminated, truncated = parent["terminated"], parent["truncated"]
                action_history.append(action.copy())
                decision += 1
                parent_steps += 1
            if decision != len(rows):
                raise RuntimeError("G10 heldout parent episode length changed")
    finally:
        env.close()
    if checked != 18 or parent_steps != 15500 or len(list(folder.glob("state_*.json"))) != 18:
        raise RuntimeError("G10 heldout true-SOFA replay incomplete")
    json_new(OUT / "heldout_parent_replay_audit.json", {"status": "PASS_EXACT_G2_G95_G96_PARENT_AND_NOMINAL_FORK",
        "ordered_reused_scene_seeds": list(DEVELOPMENT_SEEDS),
        "parent_steps_compared": parent_steps, "selected_unsafe_states": checked,
        "max_abs_parent_difference": differences,
        "no_repaired_action_applied_to_parent": True,
        "confirmation_107000_series_accessed": False})
    print("G10 heldout 18-state true-SOFA repair audit PASS", flush=True)


def events():
    records = [r for r in manifest() if r["cohort"] == "difficult_event_early_intervention"]
    selected = {(r["seed"], r["decision_index"]): r for r in records}
    folder = OUT / "difficult_event_state_results"
    folder.mkdir(exist_ok=True)
    _, _, by_seed = g85_inputs()
    mapper = E1Representation()
    solver = FrozenRepair()
    env, agent, monitor = setup(TEST_SEEDS)
    checked = parent_steps = 0
    differences = {}
    try:
        for seed in TEST_SEEDS:
            set_seed(seed)
            env.action_space.seed(seed)
            observation, _ = env.reset(seed=seed)
            check_scene(env)
            if not hasattr(monitor, "_g10_events_added"):
                env._simulation.root.addObject(monitor)
                monitor._g10_events_added = True
            rows = by_seed[seed]
            obs_history, action_history = [], []
            terminated = truncated = False
            decision = 0
            while not (terminated or truncated):
                if (decision >= len(rows) or not np.array_equal(observation,
                        np.asarray(json.loads(rows[decision]["decision_observation_json"]), dtype=np.float32))):
                    raise RuntimeError(f"G10 difficult-event parent observation changed {seed}/{decision}")
                obs_history.append(observation.copy())
                record = selected.get((seed, decision))
                if record is not None:
                    action, final_actions, scores, mean = captured_plan_v2(agent, observation, decision == 0)
                    candidates, numerical = canonical_plans(final_actions, scores, mean, action)
                    pair = np.asarray(candidates[8]["actions"], dtype=np.float32)
                    x = current_state(agent, mapper, observation, obs_history, action_history, decision)
                    path = folder / f"state_{record['state_id']:04d}_minus_{record['lookback_steps']}.json"
                    if path.exists():
                        item = json.loads(path.read_text())
                        if (item["seed"], item["decision_index"], item["nominal_pair"]) != (
                                seed, decision, pair.tolist()):
                            raise RuntimeError("G10 difficult-event replay-resume state changed")
                    else:
                        item = evaluate_state(env, monitor, solver, record, x, pair)
                        item["optimized_mean_numerical_semantics"] = numerical
                        json_new(path, item)
                    checked += 1
                    print(f"G10 event {checked}/22 id={record['state_id']}/-{record['lookback_steps']} status={item['proposal']['status']} nominal_R2={item['nominal_true']['R2_N']}", flush=True)
                else:
                    action = agent.act(observation, first_step=decision == 0, eval_mode=True)
                if not np.array_equal(action,
                        np.asarray(json.loads(rows[decision]["selected_action_json"]), dtype=np.float32)):
                    raise RuntimeError("G10 difficult-event nominal action changed")
                parent = signature(env, monitor, action)
                compare(parent, g2_signature(rows[decision]), differences,
                        f"G10 difficult-event parent {seed}/{decision}")
                observation = np.asarray(parent["observation"], dtype=np.float32)
                terminated, truncated = parent["terminated"], parent["truncated"]
                action_history.append(action.copy())
                decision += 1
                parent_steps += 1
            if decision != len(rows):
                raise RuntimeError("G10 difficult-event episode length changed")
    finally:
        env.close()
    if checked != 22 or len(list(folder.glob("state_*_minus_*.json"))) != 22:
        raise RuntimeError("G10 difficult-event true-SOFA replay incomplete")
    json_new(OUT / "difficult_event_parent_replay_audit.json", {
        "status": "PASS_EXACT_G8_PARENT_AND_G96_V2_NOMINAL_PLAN",
        "ordered_reused_scene_seeds": TEST_SEEDS, "parent_steps_compared": parent_steps,
        "selected_prior_states": checked, "max_abs_parent_difference": differences,
        "no_repaired_action_applied_to_parent": True,
        "confirmation_107000_series_accessed": False})
    print("G10 difficult-event 22-state true-SOFA repair audit PASS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("heldout", "events"), required=True)
    {"heldout": heldout, "events": events}[parser.parse_args().stage]()
