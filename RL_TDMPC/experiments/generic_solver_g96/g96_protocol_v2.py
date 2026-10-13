"""Authorized executable-optimized-mean repair and full ordered development reaudit."""

from __future__ import annotations

import json

import numpy as np

from g96_common import OUT, G95, assert_prepared, json_new, sha
from g95_common import choose_plans
from g95_collect import archived, selected_states
from g6_collect import setup
from restore_audit import signature, compare, g2_signature
from collect import check_scene
from tdmpc2.common import set_seed

ATOL = 1e-6
RTOL = 0.0
DEVELOPMENT_SEEDS = tuple(range(90000, 90200))
NEW_CONFIRMATION_SEEDS = tuple(range(105000, 105030))


def captured_plan_v2(agent, observation, first_step):
    """G9.5 passive CEM tap with only the authorized executed-action tolerance."""
    original = agent._estimate_value
    calls = []

    def tapped(latent, actions):
        value = original(latent, actions)
        calls.append((actions.detach().cpu().numpy().copy(), value.detach().cpu().numpy().copy()))
        return value

    agent._estimate_value = tapped
    try:
        selected = agent.act(observation, first_step=first_step, eval_mode=True)
    finally:
        agent._estimate_value = original
    if len(calls) != 6:
        raise RuntimeError(f"TD-MPC2 final CEM capture expected 6 value calls, got {len(calls)}")
    actions, values = calls[-1]
    mean = agent.previous_mean.detach().cpu().numpy().copy()
    if (actions.shape != (3, 512, 2) or values.shape != (512, 1) or
            mean.shape != (3, 2) or
            not np.allclose(np.clip(mean[0], -1, 1), selected, rtol=RTOL, atol=ATOL) or
            np.any(~np.isfinite(actions)) or np.any(np.abs(actions) > 1+1e-6)):
        raise RuntimeError("Frozen TD-MPC2 candidate shape/range/action invariant failed")
    scores = np.nan_to_num(values[:, 0]).astype(np.float64)
    return selected, actions.astype(np.float32), scores, mean.astype(np.float32)


def canonical_plans(final_actions, scores, optimized_mean, executed_action):
    """Keep eight sampled G9.5 identities; execute only clipped mean's first two actions."""
    raw = np.asarray(optimized_mean, dtype=np.float32)
    executed = np.asarray(executed_action, dtype=np.float32)
    if raw.shape != (3, 2) or executed.shape != (2,):
        raise RuntimeError("G9.6 v2 planner mean/action shape changed")
    canonical = np.clip(raw, -1.0, 1.0).astype(np.float32)
    if not np.allclose(canonical[0], executed, rtol=RTOL, atol=ATOL):
        raise RuntimeError("G9.6 v2 canonical executable mean does not match executed action")
    plans = choose_plans(final_actions, scores, canonical)
    if len(plans) != 9 or not np.array_equal(np.asarray(plans[-1]["actions"], dtype=np.float32), canonical[:2]):
        raise RuntimeError("G9.6 v2 optimized-mean plan not exact executable pair")
    # Only the ninth mean is subject to the new allclose rule. The frozen
    # eight sampled identities are exact float32/index identities; some were
    # already numerically close under 1e-6 before this amendment.
    ambiguous_pairs = []
    for i in range(len(plans)-1):
        sampled = np.asarray(plans[i]["actions"], dtype=np.float32)
        if np.allclose(sampled, canonical[:2], rtol=RTOL, atol=ATOL):
            ambiguous_pairs.append([i, len(plans)-1])
    if ambiguous_pairs:
        raise RuntimeError(f"G9.6 v2 mean/sampled full-plan ambiguity: {ambiguous_pairs}")
    changed_by_clipping = not np.array_equal(raw[:2], canonical[:2])
    changed_by_tolerance_only = not np.array_equal(canonical[0], executed)
    return plans, {"changed_by_clipping": changed_by_clipping,
                   "changed_by_tolerance_only": changed_by_tolerance_only,
                   "ambiguous_matches": 0,
                   "raw_optimized_mean_pair": raw[:2].tolist(),
                   "canonical_executable_mean_pair": canonical[:2].tolist()}


def audit():
    assert_prepared()
    output = OUT / "protocol_reaudit.json"
    if output.exists():
        raise FileExistsError(output)
    amendment = json.loads((OUT / "protocol_amendment.json").read_text())
    if (amendment["status"] != "PREDECLARED_G96_EXECUTABLE_OPTIMIZED_MEAN_REPAIR_BEFORE_105000_CONFIRMATION" or
            float(amendment["atol"]) != ATOL or float(amendment["rtol"]) != RTOL):
        raise RuntimeError("G9.6 v2 amendment changed")
    stopped = json.loads((OUT / "confirmation_protocol_stop.json").read_text())
    raw_failed = np.asarray(stopped["optimized_mean_first_action_float32"], dtype=np.float32)
    executed_failed = np.asarray(stopped["executed_clipped_first_action_float32"], dtype=np.float32)
    failed_canonical = np.clip(raw_failed, -1, 1)
    if (np.array_equal(raw_failed, executed_failed) or
            not np.array_equal(failed_canonical, executed_failed) or
            not np.allclose(failed_canonical, executed_failed, rtol=RTOL, atol=ATOL)):
        raise RuntimeError("G9.6 v2 does not resolve exact observed 104001 boundary case")
    by_seed, _ = archived()
    selected = selected_states()
    if len(selected) != 800:
        raise RuntimeError("G9.6 v2 development state selection changed")
    env, agent, monitor = setup(list(DEVELOPMENT_SEEDS))
    states = parent_steps = changed_clip = changed_tol = unexpected_identity = preexisting_sampled_near_pairs = 0
    differences = {}
    try:
        for seed in DEVELOPMENT_SEEDS:
            set_seed(seed)
            env.action_space.seed(seed)
            observation, _ = env.reset(seed=seed)
            check_scene(env)
            if not hasattr(monitor, "_g96_v2_reaudit_added"):
                env._simulation.root.addObject(monitor)
                monitor._g96_v2_reaudit_added = True
            rows = by_seed[seed]
            terminated = truncated = False
            decision = 0
            while not (terminated or truncated):
                if (decision >= len(rows) or not np.array_equal(observation,
                    np.asarray(json.loads(rows[decision]["decision_observation_json"]), dtype=np.float32))):
                    raise RuntimeError(f"G9.6 v2 development parent observation mismatch {seed}/{decision}")
                state = selected.get((seed, decision))
                if state is not None:
                    executed, final_actions, scores, raw_mean = captured_plan_v2(
                        agent, observation, decision == 0)
                    plans, changes = canonical_plans(final_actions, scores, raw_mean, executed)
                    archived_path = (G95 / "development_states" /
                        f"state_{state['group']}_{int(state['state_id']):04d}.json")
                    old = json.loads(archived_path.read_text())
                    if (old["seed"], old["decision_index"], len(old["plans"])) != (seed, decision, 9):
                        raise RuntimeError("G9.6 v2 archived state identity changed")
                    if not np.array_equal(np.asarray(old["planner_mean_action"], dtype=np.float32), executed):
                        raise RuntimeError("G9.6 v2 executed nominal planner action changed")
                    for new, previous in zip(plans, old["plans"]):
                        if (new["candidate_id"] != previous["candidate_id"] or
                                new["source"] != previous["source"] or
                                new["population_index"] != previous["population_index"] or
                                not np.array_equal(np.asarray(new["actions"], dtype=np.float32),
                                                   np.asarray(previous["actions"], dtype=np.float32))):
                            unexpected_identity += 1
                            raise RuntimeError(f"G9.6 v2 changed existing candidate identity {seed}/{decision}")
                    for i in range(8):
                        for j in range(i+1, 8):
                            if np.allclose(np.asarray(old["plans"][i]["actions"], dtype=np.float32),
                                           np.asarray(old["plans"][j]["actions"], dtype=np.float32),
                                           rtol=RTOL, atol=ATOL):
                                preexisting_sampled_near_pairs += 1
                    states += 1
                    changed_clip += changes["changed_by_clipping"]
                    changed_tol += changes["changed_by_tolerance_only"]
                    if states % 100 == 0:
                        print(f"G9.6 v2 development planner reaudit states={states}/800", flush=True)
                else:
                    executed = agent.act(observation, first_step=decision == 0, eval_mode=True)
                if not np.array_equal(executed,
                    np.asarray(json.loads(rows[decision]["selected_action_json"]), dtype=np.float32)):
                    raise RuntimeError(f"G9.6 v2 nominal action changed {seed}/{decision}")
                parent = signature(env, monitor, executed)
                compare(parent, g2_signature(rows[decision]), differences,
                        f"G9.6 v2 development parent {seed}/{decision}")
                observation = np.asarray(parent["observation"], dtype=np.float32)
                terminated, truncated = parent["terminated"], parent["truncated"]
                decision += 1
                parent_steps += 1
            if decision != len(rows):
                raise RuntimeError(f"G9.6 v2 development episode length changed {seed}")
    finally:
        env.close()
    if states != 800 or parent_steps != 15500 or unexpected_identity != 0:
        raise RuntimeError("G9.6 v2 development full ordered replay incomplete")
    json_new(output, {"status": "PASS_ZERO_AMBIGUOUS_CANDIDATES_EXACT_DEVELOPMENT_REPLAY",
        "amendment_sha256": sha(OUT / "protocol_amendment.json"),
        "ordered_reused_scene_seeds": list(DEVELOPMENT_SEEDS),
        "parent_steps_compared": parent_steps, "planner_states_checked": states,
        "candidate_plans_checked": states*9,
        "previously_valid_candidate_identities_changed": unexpected_identity,
        "development_matches_changed_by_clipping": changed_clip,
        "development_matches_changed_by_atol_only": changed_tol,
        "amendment_introduced_ambiguous_mean_sampled_full_plan_matches": 0,
        "preexisting_sampled_sampled_pairs_near_under_mean_atol_not_rematched": preexisting_sampled_near_pairs,
        "old_104001_boundary_case": {"raw_first_action": raw_failed.tolist(),
            "canonical_first_action": failed_canonical.tolist(),
            "actually_executed_first_action": executed_failed.tolist(),
            "resolved_by_clipping": True,
            "resolved_by_tolerance_only": False,
            "maximum_abs_raw_vs_executed": float(np.max(np.abs(raw_failed-executed_failed)))},
        "max_abs_parent_difference": differences,
        "G95_dataset_sha256": sha(G95 / "development_planner_dataset.npz"),
        "G95_candidate_protocol_sha256": sha(G95 / "planner_candidate_protocol.json"),
        "model_checkpoints_not_retrained": True,
        "105000_series_accessed": False})
    print(f"G9.6 v2 protocol reaudit PASS: {states} states, clip-changed={changed_clip}, tolerance-only={changed_tol}, ambiguous=0", flush=True)


if __name__ == "__main__":
    audit()
