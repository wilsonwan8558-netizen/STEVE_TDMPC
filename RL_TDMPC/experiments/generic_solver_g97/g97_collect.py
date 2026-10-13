"""Replay existing development scenes for missing pre-action state, not new labels."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import select
import signal
import sys
import traceback
import zlib

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
for folder in ("generic_solver_g97", "generic_solver_g96", "generic_solver_g95",
               "generic_solver_g6a", "generic_solver_g4", "generic_solver_g3", "generic_solver_g2"):
    sys.path.insert(0, str(RL / "experiments" / folder))
sys.path.insert(0, str(RL))

from g97_prepare import OUT, G95, GROUPS, source_gate  # noqa: E402
from g97_state import PrivilegedExtractor, RAW_DIM  # noqa: E402
from g95_collect import archived, selected_states  # noqa: E402
from g95_common import E1Representation, json_new, sha  # noqa: E402
from g6_collect import setup  # noqa: E402
from g4_data import physical_lookup  # noqa: E402
from restore_audit import signature, compare, g2_signature  # noqa: E402
from collect import check_scene  # noqa: E402
from tdmpc2.common import set_seed  # noqa: E402

DEVELOPMENT = tuple(range(90000, 90200))


def repeat_fork(env, monitor, action_pair):
    """Exact COW snapshot: return two force outcomes and full observable next states."""
    pair = np.asarray(action_pair, dtype=np.float32)
    if pair.shape != (2, 2) or not np.all(np.isfinite(pair)):
        raise RuntimeError("G9.7 repeatability action pair invalid")
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        try:
            with open(os.devnull, "w") as null:
                os.dup2(null.fileno(), 1)
                os.dup2(null.fileno(), 2)
            extractor = PrivilegedExtractor(env)
            outcomes, next_states = [], []
            for action in pair:
                outcome = signature(env, monitor, action)
                outcomes.append(outcome)
                state, _ = extractor.extract()
                next_states.append(state.tolist())
                if outcome["terminated"] or outcome["truncated"]:
                    break
            payload = {"status": "PASS", "outcomes": outcomes, "next_states": next_states}
        except BaseException:
            payload = {"status": "ERROR", "traceback": traceback.format_exc()}
        try:
            packed = zlib.compress(json.dumps(payload, allow_nan=False).encode(), level=3)
            for start in range(0, len(packed), 16384):
                os.write(write_fd, packed[start:start+16384])
        finally:
            os._exit(0)
    os.close(write_fd)
    chunks = []
    while True:
        ready, _, _ = select.select([read_fd], [], [], 120)
        if not ready:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            os.close(read_fd)
            raise RuntimeError("G9.7 repeated two-step child timed out")
        block = os.read(read_fd, 65536)
        if not block:
            break
        chunks.append(block)
    os.close(read_fd)
    _, status = os.waitpid(pid, 0)
    if not chunks:
        raise RuntimeError(f"G9.7 repeated fork produced no payload: status={status}")
    payload = json.loads(zlib.decompress(b"".join(chunks)))
    if payload["status"] != "PASS":
        raise RuntimeError(payload["traceback"])
    if len(payload["outcomes"]) != 2 or len(payload["next_states"]) != 2:
        raise RuntimeError("G9.7 predeclared repeatability case censored unexpectedly")
    next_states = np.asarray(payload["next_states"], dtype=np.float32)
    if next_states.shape != (2, RAW_DIM) or not np.all(np.isfinite(next_states)):
        raise RuntimeError("G9.7 repeatability next-state shape/finiteness changed")
    f1, f2 = (float(outcome["force_next_max_N"]) for outcome in payload["outcomes"])
    return {"F1_N": f1, "F2_N": f2, "R2_N": max(f1, f2),
            "outcomes": payload["outcomes"], "next_states": next_states}


def repeat_case(env, monitor, case):
    outcomes = [repeat_fork(env, monitor, case["action_pair"]) for _ in range(10)]
    for item in outcomes:
        for source, result in (("archived_F1_N", item["F1_N"]),
                               ("archived_F2_N", item["F2_N"]),
                               ("archived_R2_N", item["R2_N"])):
            if np.float32(result) != np.float32(case[source]):
                raise RuntimeError(f"G9.7 repeatability fork differs from archived G9.5 {case['category']} {source}")
    values = np.asarray([[item["F1_N"], item["F2_N"], item["R2_N"]] for item in outcomes], dtype=np.float64)
    states = np.stack([item["next_states"] for item in outcomes]).astype(np.float64)
    reference_values = values[0]
    reference_states = states[0]
    result = {key: {"max_abs_difference": float(np.max(np.abs(values[:, j]-reference_values[j]))),
                    "mean_abs_difference": float(np.mean(np.abs(values[:, j]-reference_values[j]))),
                    "population_std": float(np.std(values[:, j]))}
              for j, key in enumerate(("F1_N", "F2_N", "R2_N"))}
    result["next_state"] = {"max_abs_difference": float(np.max(np.abs(states-reference_states))),
        "mean_abs_difference": float(np.mean(np.abs(states-reference_states))),
        "population_std_mean_across_coordinates": float(np.mean(np.std(states, axis=0))),
        "coordinate_count_per_post_action_state": RAW_DIM,
        "states_per_repeat": 2}
    return result, values


def development():
    source_gate()
    definition = json.loads((OUT / "privileged_state_definition.json").read_text())
    if definition["raw_feature_source_sha256"] != sha(HERE / "g97_state.py"):
        raise RuntimeError("G9.7 privileged extractor changed after definition")
    if (OUT / "development_privileged_replay_audit.json").exists():
        raise FileExistsError("G9.7 development replay already completed")
    cases = json.loads((OUT / "repeatability_protocol.json").read_text())["cases"]
    by_key = {(int(c["seed"]), int(c["decision_index"])): c for c in cases}
    if len(by_key) != 3:
        raise RuntimeError("G9.7 fixed repeatability cases changed")
    by_seed, _ = archived()
    selected = selected_states()
    physical = physical_lookup()
    mapper = E1Representation()
    state_folder = OUT / "development_privileged_states"
    state_folder.mkdir(exist_ok=True)
    env, agent, monitor = setup(list(DEVELOPMENT))
    parent_steps = state_count = 0
    differences = {}
    repeat_metrics = []
    repeat_rows = []
    try:
        for seed in DEVELOPMENT:
            set_seed(seed)
            env.action_space.seed(seed)
            observation, _ = env.reset(seed=seed)
            check_scene(env)
            if not hasattr(monitor, "_g97_added"):
                env._simulation.root.addObject(monitor)
                monitor._g97_added = True
            extractor = PrivilegedExtractor(env)
            rows = by_seed[seed]
            history_obs, history_action = [], []
            decision = 0
            terminated = truncated = False
            while not (terminated or truncated):
                if (decision >= len(rows) or not np.array_equal(observation,
                    np.asarray(json.loads(rows[decision]["decision_observation_json"]), dtype=np.float32))):
                    raise RuntimeError(f"G9.7 development observation mismatch {seed}/{decision}")
                history_obs.append(observation.copy())
                state = selected.get((seed, decision))
                if state is not None:
                    raw, audit = extractor.extract()
                    archived_p = physical[(seed, decision)]
                    if not np.array_equal(raw[:38], archived_p):
                        raise RuntimeError(f"G9.7 exact G4 physical feature mismatch {seed}/{decision}")
                    group, local_id = state["group"], int(state["state_id"])
                    old_path = G95 / "development_states" / f"state_{group}_{local_id:04d}.json"
                    old = json.loads(old_path.read_text())
                    if (old["seed"], old["decision_index"]) != (seed, decision):
                        raise RuntimeError("G9.7 archived G9.5 state identity changed")
                    with torch.inference_mode():
                        latent = agent.model.encode(agent._tensor_observation(observation))[0].cpu().numpy()
                    if not np.array_equal(latent, np.asarray(old["current_latent"], dtype=np.float32)):
                        raise RuntimeError(f"G9.7 exact current latent mismatch {seed}/{decision}")
                    history = np.asarray(history_obs[decision-2:decision+1], dtype=np.float32)
                    preceding = np.asarray(history_action[decision-2:decision], dtype=np.float32)
                    estimate = mapper.estimate(history, preceding)
                    if not np.array_equal(estimate, np.asarray(old["E1_current_physical_estimate"], dtype=np.float32)):
                        raise RuntimeError(f"G9.7 exact E1 estimate mismatch {seed}/{decision}")
                    path = state_folder / f"state_{group}_{local_id:04d}.npz"
                    if path.exists():
                        with np.load(path) as saved:
                            if (int(saved["seed"]) != seed or int(saved["decision_index"]) != decision or
                                    not np.array_equal(saved["raw"], raw)):
                                raise RuntimeError(f"G9.7 replay-resume privileged state mismatch {path}")
                    else:
                        np.savez_compressed(path, raw=raw, seed=np.int32(seed),
                                            decision_index=np.int32(decision),
                                            group_index=np.int32(GROUPS.index(group)),
                                            state_id_within_group=np.int32(local_id),
                                            active_contact_groups=np.int32(audit["active_contact_groups"]))
                    state_count += 1
                    case = by_key.get((seed, decision))
                    if case is not None:
                        result, values = repeat_case(env, monitor, case)
                        repeat_metrics.append({"category": case["category"],
                            "seed": seed, "decision_index": decision,
                            "state_global_id": case["state_global_id"],
                            "candidate_id": case["candidate_id"], **result})
                        for repetition, triple in enumerate(values):
                            repeat_rows.append({"category": case["category"], "seed": seed,
                                "decision_index": decision, "candidate_id": case["candidate_id"],
                                "repetition": repetition, "F1_N": float(triple[0]),
                                "F2_N": float(triple[1]), "R2_N": float(triple[2])})
                        print(f"G9.7 repeatability {case['category']} 10 forks PASS", flush=True)
                    if state_count % 100 == 0:
                        print(f"G9.7 pre-action privileged states {state_count}/800", flush=True)
                action = agent.act(observation, first_step=decision == 0, eval_mode=True)
                if (not np.array_equal(action,
                    np.asarray(json.loads(rows[decision]["selected_action_json"]), dtype=np.float32)) or
                    (state is not None and not np.array_equal(action,
                    np.asarray(old["planner_mean_action"], dtype=np.float32)))):
                    raise RuntimeError(f"G9.7 frozen nominal action changed {seed}/{decision}")
                parent = signature(env, monitor, action)
                compare(parent, g2_signature(rows[decision]), differences,
                        f"G9.7 development parent {seed}/{decision}")
                observation = np.asarray(parent["observation"], dtype=np.float32)
                terminated, truncated = parent["terminated"], parent["truncated"]
                history_action.append(action.copy())
                decision += 1
                parent_steps += 1
            if decision != len(rows):
                raise RuntimeError(f"G9.7 development episode length changed {seed}")
    finally:
        env.close()
    if (parent_steps != 15500 or state_count != 800 or len(repeat_metrics) != 3 or
            len(list(state_folder.glob("state_*.npz"))) != 800):
        raise RuntimeError("G9.7 development privileged/parent/repeat replay incomplete")
    with (OUT / "repeatability_results.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(repeat_rows[0]))
        writer.writeheader()
        writer.writerows(repeat_rows)
    json_new(OUT / "repeatability_summary.json", {"status": "THREE_CASES_10_REPEATS_EACH",
        "method": "copy-on-write full live process/scene snapshot per child; no serialized partial SOFA state restore",
        "cases": repeat_metrics, "repetitions_per_case": 10,
        "all_force_and_observable_next_state_max_abs_zero": all(
            all(row[key]["max_abs_difference"] == 0 for key in ("F1_N", "F2_N", "R2_N", "next_state"))
            for row in repeat_metrics),
        "claim_limit": "three cases only; inaccessible simulator internals still prevent a full Markovity claim",
        "training_samples_added_from_repeats": 0})
    json_new(OUT / "development_privileged_replay_audit.json", {
        "status": "PASS_EXACT_G2_G4_G95_PREACTION_ALIGNMENT",
        "ordered_reused_scene_development_seeds": list(DEVELOPMENT),
        "parent_steps_compared": parent_steps, "selected_states": state_count,
        "raw_privileged_dimension": RAW_DIM,
        "repeatability_cases": len(repeat_metrics),
        "repeatability_forks": len(repeat_rows),
        "max_abs_parent_difference": differences,
        "exact_G4_P_38D_per_state": True, "exact_G95_latent_E1_per_state": True,
        "G95_existing_candidate_actions_F1_F2_R2_not_recollected": True,
        "confirmation_106000_series_accessed": False})
    print(f"G9.7 development replay PASS {state_count} states/{parent_steps} parent steps", flush=True)


def assemble():
    source_gate()
    audit = json.loads((OUT / "development_privileged_replay_audit.json").read_text())
    if (audit["status"] != "PASS_EXACT_G2_G4_G95_PREACTION_ALIGNMENT" or
            audit["selected_states"] != 800):
        raise RuntimeError("G9.7 development privileged replay not complete")
    split = json.loads((G95 / "dataset_split.json").read_text())
    states, seeds, decisions, group_indices, local_ids = [], [], [], [], []
    for gi, group in enumerate(GROUPS):
        for state in split["selected_development_states"][group]:
            sid = int(state["state_id"])
            path = OUT / "development_privileged_states" / f"state_{group}_{sid:04d}.npz"
            with np.load(path) as z:
                raw = z["raw"].copy()
                if (int(z["seed"]) != int(state["seed"]) or
                        int(z["decision_index"]) != int(state["decision_index"]) or
                        int(z["group_index"]) != gi or
                        int(z["state_id_within_group"]) != sid or raw.shape != (RAW_DIM,)):
                    raise RuntimeError(f"G9.7 raw privileged state/group alignment changed {path}")
            states.append(raw)
            seeds.append(state["seed"])
            decisions.append(state["decision_index"])
            group_indices.append(gi)
            local_ids.append(sid)
    matrix = np.asarray(states, dtype=np.float32)
    if matrix.shape != (800, RAW_DIM) or not np.all(np.isfinite(matrix)):
        raise RuntimeError("G9.7 privileged state matrix incomplete")
    np.savez_compressed(OUT / "development_privileged_states.npz",
        raw=matrix, state_global_id=np.arange(800, dtype=np.int32),
        seed=np.asarray(seeds, dtype=np.int32),
        decision_index=np.asarray(decisions, dtype=np.int32),
        group_index=np.asarray(group_indices, dtype=np.int32),
        state_id_within_group=np.asarray(local_ids, dtype=np.int32))
    with np.load(G95 / "development_planner_dataset.npz") as z:
        planner = {key: z[key] for key in z.files}
    for key in ("seed", "decision_index", "group_index", "state_id_within_group"):
        if not np.array_equal(planner[key], np.asarray((seeds, decisions, group_indices, local_ids)[
                ("seed", "decision_index", "group_index", "state_id_within_group").index(key)], dtype=np.int32)[planner["state_global_id"]]):
            raise RuntimeError(f"G9.7 planner row/raw state misalignment {key}")
    json_new(OUT / "dataset_alignment_audit.json", {
        "status": "PASS_ORIGINAL_7193_G95_LABELS_ACTIONS_AND_800_PRIVILEGED_STATES_ALIGNED",
        "privileged_state_matrix_sha256": sha(OUT / "development_privileged_states.npz"),
        "original_G95_planner_dataset_sha256": sha(G95 / "development_planner_dataset.npz"),
        "raw_state_dimension": RAW_DIM, "states_by_group": {g: [560, 120, 120][i] for i, g in enumerate(GROUPS)},
        "complete_planner_rows": len(planner["x"]),
        "F1_F2_R2_action_targets_reused_without_refork": True,
        "confirmation_106000_series_accessed": False})
    print("G9.7 exact privileged/G9.5 planner row alignment PASS", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("development", "assemble"), required=True)
    {"development": development, "assemble": assemble}[parser.parse_args().stage]()


if __name__ == "__main__":
    main()
