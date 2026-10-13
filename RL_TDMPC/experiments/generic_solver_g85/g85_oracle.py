"""G8.5: true-SOFA one-step action-space search at the 16 archived G8 unsafe states.

No learned force model is evaluated. All forks inherit the same ordered reused-scene
G8 parent lifecycle; repaired actions are never applied to the parent episode.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
for name in ("generic_solver_g8", "generic_solver_g6a", "generic_solver_g5", "generic_solver_g3", "generic_solver_g2"):
    sys.path.insert(0, str(RL / "experiments" / name))

from g8_prepare import OUT as G8, TEST_SEEDS, verify as verify_g8, cuda_gate  # noqa: E402
from g6_collect import setup  # noqa: E402
from restore_audit import fork_step, signature, compare, g2_signature  # noqa: E402
from collect import check_scene  # noqa: E402
from physical_features import extract  # noqa: E402
from tdmpc2.common import set_seed  # noqa: E402
import g5_confirm  # noqa: E402

OUT = RL / "logs/generic_solver_phase_g85_continuous_oracle_feasibility"
STATES = G8 / "confirmation_candidates"
F_LIM = 3.685331040756903
GRID_VALUES = np.linspace(-1.0, 1.0, 41, dtype=np.float64)
GRID = [(float(t), float(r)) for t in GRID_VALUES for r in GRID_VALUES]
LOCAL_OFFSETS = np.linspace(-0.02, 0.02, 5, dtype=np.float64)
TOL = 1e-8
FIELDS = ("state_id", "episode_seed", "decision_index", "translation", "rotation", "F_true_N", "true_safe")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def csv_new(path, rows, fields=FIELDS):
    if not rows:
        raise RuntimeError(f"No rows for {path}")
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def inputs():
    verify_g8()
    audit = json.loads((G8 / "confirmation_replay_audit.json").read_text())
    if audit["status"] != "PASS" or audit["candidate_evaluations"] != 2520:
        raise RuntimeError("G8 replay/fork audit not valid")
    with (G8 / "nominal_state_summary.csv").open(newline="") as stream:
        nominal = list(csv.DictReader(stream))
    with (G8 / "confirmation_behavior_steps.csv").open(newline="") as stream:
        behavior = list(csv.DictReader(stream))
    if len(nominal) != 120 or sorted({int(r["seed"]) for r in behavior}) != TEST_SEEDS:
        raise RuntimeError("Archived G8 states/behavior incomplete")
    unsafe = [r for r in nominal if int(r["nominal_true_unsafe"])]
    if len(unsafe) != 16 or sum(int(r["true_candidate_set_infeasible"]) for r in unsafe) != 12:
        raise RuntimeError("G8 unsafe/old-infeasible counts changed")
    states = {}
    manifest = []
    for row in unsafe:
        state_id = int(row["state_id"])
        path = STATES / f"state_{state_id:04d}.json"
        state = json.loads(path.read_text())
        seed, decision = int(row["episode_seed"]), int(row["decision_index"])
        if (state["state_id"], state["seed"], state["decision_index"]) != (state_id, seed, decision):
            raise RuntimeError(f"Archived G8 state identity changed: {state_id}")
        action = np.asarray(state["behavior_action"], dtype=np.float32)
        c0 = state["candidates"][0]
        if not np.array_equal(action, np.asarray(c0["action"], dtype=np.float32)):
            raise RuntimeError(f"G8 nominal candidate mismatch: {state_id}")
        if abs(float(c0["force_next_max_N"]) - float(row["nominal_true_force_N"])) > TOL:
            raise RuntimeError(f"G8 nominal force mismatch: {state_id}")
        states[(seed, decision)] = (row, state)
        manifest.append({"state_id": state_id, "episode_seed": seed, "decision_index": decision,
            "nominal_action": action.tolist(), "nominal_F_true_N": float(row["nominal_true_force_N"]),
            "old21_true_safe_count": int(row["true_safe_candidate_count"]),
            "old21_infeasible": bool(int(row["true_candidate_set_infeasible"])),
            "archived_state_sha256": sha(path)})
    manifest.sort(key=lambda r: r["state_id"])
    rows_by_seed = {}
    for row in behavior:
        rows_by_seed.setdefault(int(row["seed"]), []).append(row)
    return states, manifest, rows_by_seed


def prepare():
    cuda_gate()
    if OUT.exists():
        raise FileExistsError(OUT)
    states, manifest, behavior = inputs()
    OUT.mkdir(parents=True)
    (OUT / "force_landscapes").mkdir()
    (OUT / "per_state").mkdir()
    json_new(OUT / "recovered_g8_state_manifest.json", {"states": manifest,
        "count": 16, "old21_infeasible_count": 12,
        "state_recovery": "ordered reused-scene replay; archived files are observations/features and fork outcomes, not standalone SOFA snapshots"})
    json_new(OUT / "metadata.json", {"phase": "G8.5", "branch": "feature/generic-contact-force-safety",
        "F_lim_N": F_LIM, "seed_order": TEST_SEEDS, "grid_axis": GRID_VALUES.tolist(),
        "global_grid_actions_per_state": len(GRID), "expected_global_evaluations": 16 * len(GRID),
        "action_realization": "float32 normalized translation/rotation passed to env.step",
        "local_refinement": "For every state, 5x5 offset grid at [-0.02,-0.01,0,0.01,0.02] around top three lowest-force global actions and top three nominal-nearest true-safe global actions; clip to [-1,1], deduplicate realized float32 actions, exclude global actions",
        "local_refinement_scope": "all 16 states, including those without a safe global action",
        "safe_rule": "true max-substep distal force <= exact frozen G2 Q95 F_lim",
        "nearest_safe_rule": "minimum Euclidean distance in realized float32 normalized 2-D actions, tie lexicographic translation then rotation",
        "no_predictor_evaluation": True, "no_repaired_action_applied_to_parent": True,
        "finite_search_caveat": "No-safe-found is empirical over the prescribed grid/refinement, not a proof of mathematical infeasibility",
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python": sys.executable, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0), "device_index": 0,
        "g8_protocol_freeze_sha256": sha(G8 / "g8_protocol_freeze.json"),
        "g8_behavior_steps_sha256": sha(G8 / "confirmation_behavior_steps.csv"),
        "g8_nominal_summary_sha256": sha(G8 / "nominal_state_summary.csv"),
        "g85_oracle_code_sha256": sha(__file__)})
    print(f"G8.5 prepared {len(states)} unsafe states; {sum(map(len, behavior.values()))} archived parent steps", flush=True)


def verify_prepared():
    cuda_gate()
    states, manifest, rows_by_seed = inputs()
    meta = json.loads((OUT / "metadata.json").read_text())
    if (meta["g85_oracle_code_sha256"] != sha(__file__) or
        meta["g8_protocol_freeze_sha256"] != sha(G8 / "g8_protocol_freeze.json") or
        meta["g8_behavior_steps_sha256"] != sha(G8 / "confirmation_behavior_steps.csv") or
        meta["g8_nominal_summary_sha256"] != sha(G8 / "nominal_state_summary.csv") or
        meta["expected_global_evaluations"] != 26896):
        raise RuntimeError("G8.5 prepared protocol/source changed")
    if json.loads((OUT / "recovered_g8_state_manifest.json").read_text())["states"] != manifest:
        raise RuntimeError("G8.5 state manifest changed")
    return states, rows_by_seed


def make_row(state, action, force):
    return {"state_id": state["state_id"], "episode_seed": state["seed"],
        "decision_index": state["decision_index"], "translation": float(action[0]),
        "rotation": float(action[1]), "F_true_N": float(force),
        "true_safe": int(force <= F_LIM)}


def action_key(action):
    a = np.asarray(action, dtype=np.float32)
    return (float(a[0]), float(a[1]))


def local_actions(global_rows, nominal):
    safe = [r for r in global_rows if r["true_safe"]]
    low = sorted(global_rows, key=lambda r: (r["F_true_N"], r["translation"], r["rotation"]))[:3]
    near = sorted(safe, key=lambda r: ((r["translation"]-nominal[0])**2 +
        (r["rotation"]-nominal[1])**2, r["translation"], r["rotation"]))[:3]
    global_keys = {action_key((r["translation"], r["rotation"])) for r in global_rows}
    seen = set()
    local = []
    for center in low + near:
        for dt in LOCAL_OFFSETS:
            for dr in LOCAL_OFFSETS:
                a = action_key((np.clip(center["translation"] + dt, -1, 1),
                                np.clip(center["rotation"] + dr, -1, 1)))
                if a not in global_keys and a not in seen:
                    seen.add(a)
                    local.append(a)
    return local


def write_state(path, rows):
    partial = path.with_suffix(".partial")
    if path.exists() or partial.exists():
        raise FileExistsError(path if path.exists() else partial)
    csv_new(partial, rows)
    os.rename(partial, path)


def scene_pass(stage):
    states, rows_by_seed = verify_prepared()
    if stage == "replay" and (OUT / "replay_audit.json").exists():
        raise FileExistsError(OUT / "replay_audit.json")
    if stage == "search":
        audit = json.loads((OUT / "replay_audit.json").read_text())
        if audit["status"] != "PASS" or audit["unsafe_nominal_fork_count"] != 16:
            raise RuntimeError("G8.5 nominal replay audit not passed")
    g5_confirm.CONFIRM = TEST_SEEDS
    env, agent, monitor = setup(TEST_SEEDS)
    differences = {}
    nominal_forks = 0
    parent_steps = 0
    try:
        for seed in TEST_SEEDS:
            set_seed(seed)
            env.action_space.seed(seed)
            observation, _ = env.reset(seed=seed)
            check_scene(env)
            if not hasattr(monitor, "_g85_added"):
                env._simulation.root.addObject(monitor)
                monitor._g85_added = True
            rows = rows_by_seed[seed]
            terminated = truncated = False
            d = 0
            while not (terminated or truncated):
                if d >= len(rows):
                    raise RuntimeError(f"G8.5 parent replay overran {seed}")
                row = rows[d]
                if not np.array_equal(observation, np.asarray(json.loads(row["decision_observation_json"]), dtype=np.float32)):
                    raise RuntimeError(f"G8.5 observation replay mismatch {seed}/{d}")
                item = states.get((seed, d))
                if item is not None:
                    archived, state = item
                    physical, audit = extract(env)
                    compare({"physical": physical}, {"physical": state["physical"]}, differences, f"G8.5 physical {seed}/{d}")
                    delta = abs(audit["current_distal_force_final_N"] - float(rows[d-1]["distal_contact_force_final_N"]))
                    differences["preaction_previous_final_force_N"] = max(differences.get("preaction_previous_final_force_N", 0.0), delta)
                    if delta > TOL:
                        raise RuntimeError(f"G8.5 preaction force alignment mismatch {seed}/{d}")
                action = agent.act(observation, first_step=d == 0, eval_mode=True)
                if not np.array_equal(action, np.asarray(json.loads(row["selected_action_json"]), dtype=np.float32)):
                    raise RuntimeError(f"G8.5 nominal action replay mismatch {seed}/{d}")
                if item is not None:
                    archived, state = item
                    if not np.array_equal(action, np.asarray(state["behavior_action"], dtype=np.float32)):
                        raise RuntimeError(f"G8.5 archived state action mismatch {seed}/{d}")
                    nominal = fork_step(env, monitor, action)
                    compare(nominal, state["candidates"][0]["outcome"], differences, f"G8.5 archived nominal fork {seed}/{d}")
                    compare(nominal, g2_signature(row), differences, f"G8.5 nominal row fork {seed}/{d}")
                    if abs(nominal["force_next_max_N"] - float(archived["nominal_true_force_N"])) > TOL:
                        raise RuntimeError(f"G8.5 nominal force mismatch {seed}/{d}")
                    nominal_forks += 1
                    if stage == "search":
                        gid = state["state_id"]
                        global_path = OUT / "per_state" / f"state_{gid:04d}_global.csv"
                        local_path = OUT / "per_state" / f"state_{gid:04d}_local.csv"
                        if global_path.exists() and local_path.exists():
                            with global_path.open(newline="") as stream:
                                prior = list(csv.DictReader(stream))
                            with local_path.open(newline="") as stream:
                                prior_local = list(csv.DictReader(stream))
                            if (len(prior) != 1681 or not prior_local or
                                any(int(r["state_id"]) != gid for r in prior + prior_local)):
                                raise RuntimeError(f"G8.5 completed state corrupt: {gid}")
                            print(f"G8.5 resume replayed state {gid} without duplicate labels", flush=True)
                        elif global_path.exists() and not local_path.exists():
                            with global_path.open(newline="") as stream:
                                prior = list(csv.DictReader(stream))
                            if len(prior) != 1681 or any(int(r["state_id"]) != gid for r in prior):
                                raise RuntimeError(f"G8.5 incomplete global state corrupt: {gid}")
                            global_rows = [{"state_id": gid, "translation": float(r["translation"]),
                                "rotation": float(r["rotation"]), "F_true_N": float(r["F_true_N"]),
                                "true_safe": int(r["true_safe"])} for r in prior]
                            local_rows = []
                            for coord in local_actions(global_rows, action):
                                result = fork_step(env, monitor, np.asarray(coord, dtype=np.float32))
                                local_rows.append(make_row(state, coord, result["force_next_max_N"]))
                            write_state(local_path, local_rows)
                            print(f"G8.5 resumed local refinement state {gid} local={len(local_rows)}", flush=True)
                        elif local_path.exists():
                            raise RuntimeError(f"G8.5 local output exists without grid: {gid}")
                        else:
                            global_rows = []
                            for coord in GRID:
                                realized = action_key(coord)
                                result = fork_step(env, monitor, np.asarray(realized, dtype=np.float32))
                                global_rows.append(make_row(state, realized, result["force_next_max_N"]))
                            local_rows = []
                            for coord in local_actions(global_rows, action):
                                result = fork_step(env, monitor, np.asarray(coord, dtype=np.float32))
                                local_rows.append(make_row(state, coord, result["force_next_max_N"]))
                            # Every state has at least the three lowest-force centers for refinement.
                            if not local_rows:
                                raise RuntimeError(f"No G8.5 local refinement actions: {gid}")
                            write_state(global_path, global_rows)
                            write_state(local_path, local_rows)
                            print(f"G8.5 searched state {gid} seed={seed} d={d} grid=1681 local={len(local_rows)} min={min(r['F_true_N'] for r in global_rows + local_rows):.6f}", flush=True)
                parent = signature(env, monitor, action)
                compare(parent, g2_signature(row), differences, f"G8.5 parent {seed}/{d}")
                if item is not None:
                    compare(parent, nominal, differences, f"G8.5 parent-nominal fork {seed}/{d}")
                observation = np.asarray(parent["observation"], dtype=np.float32)
                terminated, truncated = parent["terminated"], parent["truncated"]
                d += 1
                parent_steps += 1
            if d != len(rows):
                raise RuntimeError(f"G8.5 episode length changed {seed}")
            print(f"G8.5 {stage} parent seed={seed} steps={d}", flush=True)
    finally:
        env.close()
    if parent_steps != 1848 or nominal_forks != 16:
        raise RuntimeError(f"G8.5 replay incomplete steps={parent_steps} forks={nominal_forks}")
    result = {"status": "PASS", "stage": stage, "ordered_reused_scene_seeds": TEST_SEEDS,
        "parent_steps_compared": parent_steps, "unsafe_nominal_fork_count": nominal_forks,
        "max_abs_difference": differences, "tolerance": TOL}
    json_new(OUT / ("replay_audit.json" if stage == "replay" else "search_parent_audit.json"), result)
    print(f"G8.5 {stage} audit PASS: {parent_steps} steps, {nominal_forks} unsafe-state forks", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "replay", "search"), required=True)
    args = parser.parse_args()
    {"prepare": prepare, "replay": lambda: scene_pass("replay"),
     "search": lambda: scene_pass("search")}[args.stage]()


if __name__ == "__main__":
    main()
