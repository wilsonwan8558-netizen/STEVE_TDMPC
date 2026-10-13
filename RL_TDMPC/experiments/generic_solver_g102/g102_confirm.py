"""Open 108000–108029 only after freeze; passive selection then exact replay/forks."""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

HERE = __import__("pathlib").Path(__file__).resolve().parent
RL = HERE.parents[1]
for name in ("generic_solver_g102", "generic_solver_g101", "generic_solver_g10",
             "generic_solver_g96", "generic_solver_g95", "generic_solver_g8",
             "generic_solver_g6a", "generic_solver_g3", "generic_solver_g2"):
    sys.path.insert(0, str(RL / "experiments" / name))
sys.path.insert(0, str(RL))

from g102_prepare import OUT, CONFIRM_SEEDS, F_LIM, verify_source  # noqa: E402
from g102_eval_common import LocalEnsemble, search_points, key  # noqa: E402
from g10_offline import current_state  # noqa: E402
from g95_common import E1Representation, fork_two, json_new, sha  # noqa: E402
from g96_protocol_v2 import captured_plan_v2, canonical_plans  # noqa: E402
from g6_collect import setup  # noqa: E402
from restore_audit import signature, compare  # noqa: E402
from collect import check_scene  # noqa: E402
from tdmpc2.common import set_seed  # noqa: E402


def freeze_gate():
    verify_source()
    path = OUT / "pre_confirmation_freeze.json"
    frozen = json.loads(path.read_text())
    if (frozen["status"] != "FROZEN_BEFORE_UNTOUCHED_108000_CONFIRMATION" or
            frozen["H"] != 2 or frozen["F_lim_N"] != F_LIM):
        raise RuntimeError("G10.2 pre-confirmation freeze invalid")
    for name, digest in frozen["frozen_output_file_sha256"].items():
        if sha(OUT / name) != digest:
            raise RuntimeError(f"G10.2 frozen output changed: {name}")
    for name, digest in frozen["frozen_code_sha256"].items():
        if sha(HERE / name) != digest:
            raise RuntimeError(f"G10.2 frozen implementation changed: {name}")
    for name, digest in frozen["development_event_feature_sha256"].items():
        if sha(OUT / "development_event_features" / name) != digest:
            raise RuntimeError(f"G10.2 frozen development event features changed: {name}")
    manifest = json.loads((OUT / "confirmation_seed_manifest.json").read_text())
    if manifest["ordered_seeds"] != list(CONFIRM_SEEDS):
        raise RuntimeError("G10.2 confirmation seed range/order changed")
    return frozen


def selected_decisions(steps):
    length = len(steps)
    if length < 4:
        raise RuntimeError("G10.2 confirmation episode too short for two selected states")
    early = min(range(2, length), key=lambda d: (abs(d/(length-1)-.25), d))
    high = max((d for d in range(2, length) if d != early),
               key=lambda d: (steps[d-1]["outcome"]["force_next_max_N"], -d))
    return [{"decision_index": early, "selection_slot": "quarter_progress"},
            {"decision_index": high, "selection_slot": "maximum_prior_executed_force"}]


def behavior():
    frozen = freeze_gate()
    folder = OUT / "confirmation_parent_episodes"
    folder.mkdir(exist_ok=True)
    env, agent, monitor = setup(list(CONFIRM_SEEDS))
    total_steps = 0
    try:
        for seed in CONFIRM_SEEDS:
            set_seed(seed)
            env.action_space.seed(seed)
            observation, _ = env.reset(seed=seed)
            check_scene(env)
            if not hasattr(monitor, "_g102_confirm_added"):
                env._simulation.root.addObject(monitor)
                monitor._g102_confirm_added = True
            steps = []
            terminated = truncated = False
            decision = 0
            while not (terminated or truncated):
                before = np.asarray(observation, dtype=np.float32).tolist()
                action = agent.act(observation, first_step=decision == 0, eval_mode=True)
                outcome = signature(env, monitor, action)
                steps.append({"decision_index": decision, "observation_before": before,
                              "selected_action": np.asarray(action, dtype=np.float32).tolist(),
                              "outcome": outcome})
                observation = np.asarray(outcome["observation"], dtype=np.float32)
                terminated, truncated = outcome["terminated"], outcome["truncated"]
                decision += 1
            path = folder / f"seed_{seed}.json"
            item = {"seed": seed, "steps": steps, "selected": selected_decisions(steps),
                    "selection_uses_only_episode_progress_and_prior_executed_force": True,
                    "no_candidate_force_forks": True,
                    "pre_confirmation_freeze_sha256": sha(OUT / "pre_confirmation_freeze.json")}
            if path.exists():
                if json.loads(path.read_text()) != item:
                    raise RuntimeError("G10.2 confirmation first-pass ordered replay changed")
            else:
                json_new(path, item)
            total_steps += len(steps)
            print(f"G10.2 confirmation passive seed={seed} steps={len(steps)}", flush=True)
    finally:
        env.close()
    if len(list(folder.glob("seed_*.json"))) != len(CONFIRM_SEEDS):
        raise RuntimeError("G10.2 confirmation first pass incomplete")
    selection = []
    episode_hashes = {}
    for seed in CONFIRM_SEEDS:
        path = folder / f"seed_{seed}.json"
        episode_hashes[str(seed)] = sha(path)
        episode = json.loads(path.read_text())
        for picked in episode["selected"]:
            selection.append({"seed": seed, **picked,
                "episode_sha256": episode_hashes[str(seed)]})
    if len(selection) != 60:
        raise RuntimeError("G10.2 confirmation 60 selected states required")
    json_new(OUT / "confirmation_state_selection.json", {
        "status": "PASSIVE_FIRST_PASS_SELECTION_BEFORE_ANY_CANDIDATE_FORCE_FORK",
        "pre_confirmation_freeze_sha256": sha(OUT / "pre_confirmation_freeze.json"),
        "episode_sha256": episode_hashes, "selected_states": selection,
        "selected_state_count": len(selection), "ordered_seeds": list(CONFIRM_SEEDS),
        "parent_steps": total_steps, "no_future_repair_success_selection": True})
    print("G10.2 confirmation passive selection PASS 60 states", flush=True)


def curve(env, monitor, ensemble, x, pair, selected):
    nominal, second = pair
    vs = search_points(nominal[0])
    predicted = ensemble.score(x, pair, vs)
    forks = []
    for v in vs:
        first = np.asarray((v, nominal[1]), dtype=np.float32)
        outcome = fork_two(env, monitor, np.stack((first, second)))
        forks.append({"v": float(v), "F1_true_N": outcome["F1_true_N"],
            "F2_true_N": outcome["F2_true_N"], "R2_true_N": outcome["R2_true_N"],
            "complete_two_intervals": outcome["complete_two_intervals"],
            "censor_reason": outcome["censor_reason"],
            "true_safe": bool(outcome["complete_two_intervals"] and outcome["R2_true_N"] <= F_LIM)})
    return {"seed": selected["seed"], "decision_index": selected["decision_index"],
        "selection_slot": selected["selection_slot"], "source_episode_sha256": selected["episode_sha256"],
        "nominal_pair": pair.tolist(), "x_294": x.tolist(), "v_nom": float(nominal[0]),
        "w_nom_fixed": float(nominal[1]), "second_action_fixed": second.tolist(),
        "nominal_G10_Q95_replicas_N": predicted["nominal_Q95_replicas_N"].tolist(),
        "nominal_conservative_Q95_N": predicted["nominal_conservative_Q95_N"],
        "tested_v": vs.tolist(),
        "predicted_delta_quantiles_replicas_N": predicted["delta_quantiles_replicas_N"].tolist(),
        "predicted_absolute_R2_N": predicted["predicted_absolute_R2_N"].tolist(),
        "predicted_corrected_Q95_spread_N": predicted["corrected_Q95_spread_N"].tolist(),
        "predicted_force_constraint_pass": predicted["force_constraint_pass"].tolist(),
        "predicted_uncertainty_constraint_pass": predicted["uncertainty_constraint_pass"].tolist(),
        "predicted_safe": predicted["predicted_safe"].tolist(),
        "true_curve": forks,
        "true_fork_count": len(forks),
        "one_interval_censored_fork_count": sum(not r["complete_two_intervals"] for r in forks),
        "no_repaired_action_applied_to_parent": True}


def curves():
    frozen = freeze_gate()
    selected_source = json.loads((OUT / "confirmation_state_selection.json").read_text())
    if (selected_source["status"] != "PASSIVE_FIRST_PASS_SELECTION_BEFORE_ANY_CANDIDATE_FORCE_FORK" or
            selected_source["pre_confirmation_freeze_sha256"] != sha(OUT / "pre_confirmation_freeze.json")):
        raise RuntimeError("G10.2 confirmation passive first pass not frozen")
    selected = {(r["seed"], r["decision_index"]): r for r in selected_source["selected_states"]}
    if len(selected) != 60:
        raise RuntimeError("G10.2 confirmation selected state identity duplicate")
    folder = OUT / "confirmation_state_curves"
    folder.mkdir(exist_ok=True)
    ensemble = LocalEnsemble()
    mapper = E1Representation()
    env, agent, monitor = setup(list(CONFIRM_SEEDS))
    checked = parent_steps = force_forks = censored = 0
    differences = {}
    try:
        for seed in CONFIRM_SEEDS:
            episode_path = OUT / "confirmation_parent_episodes" / f"seed_{seed}.json"
            if sha(episode_path) != selected_source["episode_sha256"][str(seed)]:
                raise RuntimeError("G10.2 confirmation passive parent episode changed")
            episode = json.loads(episode_path.read_text())
            steps = episode["steps"]
            set_seed(seed)
            env.action_space.seed(seed)
            observation, _ = env.reset(seed=seed)
            check_scene(env)
            if not hasattr(monitor, "_g102_confirm_curve_added"):
                env._simulation.root.addObject(monitor)
                monitor._g102_confirm_curve_added = True
            obs_history, action_history = [], []
            terminated = truncated = False
            decision = 0
            while not (terminated or truncated):
                if (decision >= len(steps) or not np.array_equal(observation,
                        np.asarray(steps[decision]["observation_before"], dtype=np.float32))):
                    raise RuntimeError("G10.2 confirmation passive/replay observation mismatch")
                obs_history.append(observation.copy())
                chosen = selected.get((seed, decision))
                if chosen is not None:
                    action, final, scores, mean = captured_plan_v2(agent, observation, decision == 0)
                    candidates, _ = canonical_plans(final, scores, mean, action)
                    pair = np.asarray(candidates[8]["actions"], dtype=np.float32)
                    x = current_state(agent, mapper, observation, obs_history, action_history, decision)
                    path = folder / f"seed_{seed}_decision_{decision:04d}.json"
                    if path.exists():
                        item = json.loads(path.read_text())
                        if (item["source_episode_sha256"], item["nominal_pair"], item["x_294"]) != (
                                chosen["episode_sha256"], pair.tolist(), x.tolist()):
                            raise RuntimeError("G10.2 confirmation curve replay-resume changed")
                    else:
                        item = curve(env, monitor, ensemble, x, pair, chosen)
                        json_new(path, item)
                    checked += 1
                    force_forks += item["true_fork_count"]
                    censored += item["one_interval_censored_fork_count"]
                    print(f"G10.2 confirmation {checked}/60 seed={seed} decision={decision} "
                          f"forks={item['true_fork_count']} censored={item['one_interval_censored_fork_count']}", flush=True)
                else:
                    action = agent.act(observation, first_step=decision == 0, eval_mode=True)
                if not np.array_equal(action,
                        np.asarray(steps[decision]["selected_action"], dtype=np.float32)):
                    raise RuntimeError("G10.2 confirmation passive/replay selected action mismatch")
                parent = signature(env, monitor, action)
                compare(parent, steps[decision]["outcome"], differences,
                        f"G10.2 confirmation {seed}/{decision}")
                observation = np.asarray(parent["observation"], dtype=np.float32)
                terminated, truncated = parent["terminated"], parent["truncated"]
                action_history.append(action.copy())
                decision += 1
                parent_steps += 1
            if decision != len(steps):
                raise RuntimeError("G10.2 confirmation passive/replay episode length mismatch")
    finally:
        env.close()
    if checked != 60 or parent_steps != selected_source["parent_steps"] or len(list(
            folder.glob("seed_*_decision_*.json"))) != 60:
        raise RuntimeError("G10.2 confirmation curves or parent replay incomplete")
    json_new(OUT / "confirmation_parent_replay_audit.json", {
        "status": "PASS_EXACT_PASSIVE_PARENT_AND_FORK_REPLAY",
        "ordered_seeds": list(CONFIRM_SEEDS), "states": checked,
        "parent_steps_compared": parent_steps,
        "true_SOFA_two_action_forks": force_forks,
        "one_interval_censored_forks": censored,
        "max_abs_parent_difference": differences,
        "pre_confirmation_freeze_sha256": sha(OUT / "pre_confirmation_freeze.json"),
        "no_candidate_applied_to_parent": True,
        "107000_107199_online_seeds_opened": False})
    print(f"G10.2 confirmation curves PASS {checked} states, {force_forks} forks", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("behavior", "curves"), required=True)
    stage = parser.parse_args().stage
    behavior() if stage == "behavior" else curves()
