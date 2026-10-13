"""Recover deployable latent/E1 features for G10.1 events without new force forks."""

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

from g102_prepare import OUT, G101, G10, verify_source  # noqa: E402
from g10_offline import current_state  # noqa: E402
from g10_repair import FrozenRepair  # noqa: E402
from g85_oracle import inputs as g85_inputs  # noqa: E402
from g8_prepare import TEST_SEEDS  # noqa: E402
from g95_common import E1Representation, json_new, sha  # noqa: E402
from g96_protocol_v2 import captured_plan_v2, canonical_plans  # noqa: E402
from g6_collect import setup  # noqa: E402
from restore_audit import signature, compare, g2_signature  # noqa: E402
from collect import check_scene  # noqa: E402
from tdmpc2.common import set_seed  # noqa: E402


def run():
    verify_source()
    manifest = json.loads((OUT / "local_dataset_manifest.json").read_text())
    selected = {(int(r["seed"]), int(r["decision_index"])): r
                for r in manifest["development_test_states"]
                if r["cohort"] == "difficult_event_early_intervention"}
    if len(selected) != 22:
        raise RuntimeError("G10.2 exact 22 difficult early-event states required")
    _, _, by_seed = g85_inputs()
    seeds = list(TEST_SEEDS)
    env, agent, monitor = setup(seeds)
    mapper, predictor = E1Representation(), FrozenRepair()
    checked = parent_steps = 0
    differences = {}
    try:
        for seed in seeds:
            set_seed(seed)
            env.action_space.seed(seed)
            observation, _ = env.reset(seed=seed)
            check_scene(env)
            if not hasattr(monitor, "_g102_feature_added"):
                env._simulation.root.addObject(monitor)
                monitor._g102_feature_added = True
            rows = by_seed[seed]
            obs_history, action_history = [], []
            terminated = truncated = False
            decision = 0
            while not (terminated or truncated):
                if decision >= len(rows) or not np.array_equal(observation,
                        np.asarray(json.loads(rows[decision]["decision_observation_json"]), dtype=np.float32)):
                    raise RuntimeError(f"G10.2 event parent observation changed {seed}/{decision}")
                obs_history.append(observation.copy())
                row = selected.get((seed, decision))
                if row is not None:
                    action, final, scores, mean = captured_plan_v2(agent, observation, decision == 0)
                    candidates, _ = canonical_plans(final, scores, mean, action)
                    pair = np.asarray(candidates[8]["actions"], dtype=np.float32)
                    if not np.array_equal(pair, np.asarray(row["nominal_pair"], dtype=np.float32)):
                        raise RuntimeError("G10.2 event exact G10.1 nominal pair changed")
                    x = current_state(agent, mapper, observation, obs_history, action_history, decision)
                    old_name = f"state_{int(row['state_id']):04d}_minus_{int(row['lookback_steps'])}.json"
                    old = json.loads((G10 / "difficult_event_state_results" / old_name).read_text())
                    single = predictor.scores(x, pair[0][None, :], pair[1])[0]
                    if not np.allclose(single, old["proposal"]["nominal_corrected_Q95_N"],
                                       rtol=0, atol=1e-7):
                        raise RuntimeError("G10.2 event latent/E1 predictor alignment changed")
                    out = OUT / "development_event_features" / old_name
                    if out.exists():
                        prior = json.loads(out.read_text())
                        if (prior["seed"], prior["decision_index"], prior["G101_curve_sha256"],
                                prior["x_294"]) != (
                                seed, decision, row["curve_sha256"], x.tolist()):
                            raise RuntimeError("G10.2 event feature replay-resume changed")
                    else:
                        json_new(out, {"cohort": row["cohort"], "state_id": row["state_id"],
                            "seed": seed, "decision_index": decision,
                            "lookback_steps": row["lookback_steps"],
                            "G101_curve_sha256": row["curve_sha256"],
                            "nominal_pair": pair.tolist(), "x_294": x.tolist(),
                            "G10_single_action_corrected_Q95_N": single.tolist(),
                            "no_force_fork_recollected": True})
                    checked += 1
                else:
                    action = agent.act(observation, first_step=decision == 0, eval_mode=True)
                if not np.array_equal(action,
                        np.asarray(json.loads(rows[decision]["selected_action_json"]), dtype=np.float32)):
                    raise RuntimeError("G10.2 event parent action changed")
                parent = signature(env, monitor, action)
                compare(parent, g2_signature(rows[decision]), differences,
                        f"G10.2 event parent {seed}/{decision}")
                observation = np.asarray(parent["observation"], dtype=np.float32)
                terminated, truncated = parent["terminated"], parent["truncated"]
                action_history.append(action.copy())
                decision += 1
                parent_steps += 1
            if decision != len(rows):
                raise RuntimeError("G10.2 event parent episode length changed")
    finally:
        env.close()
    if checked != 22 or parent_steps != 1848 or len(list(
            (OUT / "development_event_features").glob("state_*.json"))) != 22:
        raise RuntimeError("G10.2 event feature recovery incomplete")
    json_new(OUT / "development_event_feature_replay_audit.json", {
        "status": "PASS_EXACT_G10_1_EVENT_FEATURE_REPLAY",
        "ordered_seeds": seeds, "states": checked, "parent_steps_compared": parent_steps,
        "max_abs_parent_difference": differences,
        "G10_single_action_Q95_matches_at_1e-7_N": True,
        "new_true_SOFA_force_forks": 0,
        "confirmation_108000_108029_opened": False})
    print("G10.2 event feature recovery PASS 22 states, zero new force forks", flush=True)


if __name__ == "__main__":
    run()
