"""Reproduce G3 development forks and collect pre-action P in one reused scene.

On resume, all preceding seeds and forks are executed again; completed episode
feature files are compared, never skipped. This preserves scene history.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
sys.path.insert(0, str(RL / "experiments/generic_solver_g3"))
sys.path.insert(0, str(RL / "experiments/generic_solver_g2"))
sys.path.insert(0, str(RL))
sys.path.insert(0, str(RL.parent))

from envs.steve_env import make_steve_env  # noqa: E402
from tdmpc2.agent import TDMPC2Agent  # noqa: E402
from tdmpc2.common import load_torch_checkpoint, set_seed  # noqa: E402
from train import build_agent_config  # noqa: E402
from substep_monitor import DistalForceMonitor  # noqa: E402
from collect import check_scene  # noqa: E402
from prepare import G1B, G2, OUT as G3, CONFIG, EXPECTED_CHECKPOINT_SHA, EXPECTED_G2_SHA, sha, check_force_source  # noqa: E402
from protocol import generate_candidates  # noqa: E402
from restore_audit import fork_step, signature, compare, g2_signature, TOLERANCE  # noqa: E402
from physical_features import extract, DIMENSION  # noqa: E402

OUT = RL / "logs/generic_solver_phase_g4_state_observability"
EPISODES = OUT / "development_physical_by_episode"


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temp, path)


def save_episode(path, seed, decisions, features):
    temp = path.with_suffix(".partial.npz")
    np.savez_compressed(temp, seed=np.full(len(decisions), seed, dtype=np.int32),
                        decision_index=np.asarray(decisions, dtype=np.int32),
                        physical=np.asarray(features, dtype=np.float32))
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    status_path = OUT / "physical_state_alignment_audit.json"
    EPISODES.mkdir(parents=True, exist_ok=True)
    if status_path.exists():
        old = json.loads(status_path.read_text())
        recoverable_preflight_failure = old["status"] == "FAIL_STOP" and old["parent_steps"] == 0 and not any(EPISODES.glob("seed_*.npz"))
        if not args.resume or (old["status"] not in ("RUNNING", "INTERRUPTED") and not recoverable_preflight_failure):
            raise RuntimeError(f"Cannot start over existing audit {old['status']}")
    status = {"status": "RUNNING", "scene_lifecycle": "one reused scene, seeds 90000-90199, full historical replay on resume",
              "parent_episodes": 0, "parent_steps": 0, "physical_rows": 0,
              "selected_states": 0, "candidate_forks": 0,
              "tolerance": TOLERANCE, "max_abs_difference": {}, "resume": args.resume,
              "source_g3_behavior_sha256": sha(G3 / "behavior_features.npz"),
              "source_g2_step_sha256": EXPECTED_G2_SHA}
    write_json(status_path, status)
    try:
        check_force_source()
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Frozen CUDA_VISIBLE_DEVICES=0 required")
        definition = json.loads((OUT / "privileged_feature_definition.json").read_text())
        if sha(HERE / "physical_features.py") != definition["extractor_sha256"] or DIMENSION != definition["dimension"]:
            raise RuntimeError("Frozen P definition changed")
        checkpoint_path = Path(json.loads((G1B / "selected_checkpoint.json").read_text())["path"])
        if sha(checkpoint_path) != EXPECTED_CHECKPOINT_SHA or sha(G2 / "step_level_force.csv") != EXPECTED_G2_SHA:
            raise RuntimeError("Frozen source changed")
        selection = json.loads((G3 / "development_state_selection.json").read_text())["states"]
        selected = {(s["seed"], s["decision_index"]): s for group in ("train", "validation", "development_test") for s in selection[group]}
        if len(selected) != 800:
            raise RuntimeError("Expected 800 selected states")
        with (G2 / "step_level_force.csv").open(newline="") as stream:
            rows_by_seed = {}
            for row in csv.DictReader(stream):
                rows_by_seed.setdefault(int(row["seed"]), []).append(row)
        with np.load(G3 / "behavior_features.npz") as source:
            lookup = {(int(seed), int(d)): i for i, (seed, d) in enumerate(zip(source["seed"], source["decision_index"]))}
            source_latent = source["latent"]
        checkpoint = load_torch_checkpoint(checkpoint_path, map_location="cpu")
        config = checkpoint["config"]
        if config != yaml.safe_load(CONFIG.read_text()):
            raise RuntimeError("Frozen config mismatch")
        set_seed(90000)
        env = make_steve_env(config["environment"])
        agent = TDMPC2Agent(14, 2, build_agent_config(config), episode_length=int(config["environment"]["max_episode_steps"]), device=torch.device("cuda:0"))
        agent.load_state_dict(checkpoint["agent"], load_optimizers=False)
        monitor = DistalForceMonitor(env)
        try:
            for seed in range(90000, 90200):
                set_seed(seed)
                env.action_space.seed(seed)
                observation, _ = env.reset(seed=seed)
                check_scene(env)
                if not hasattr(monitor, "_g4_added"):
                    env._simulation.root.addObject(monitor)
                    monitor._g4_added = True
                rows = rows_by_seed[seed]
                decisions, physical = [], []
                terminated = truncated = False
                d = 0
                while not (terminated or truncated):
                    if d >= len(rows):
                        raise RuntimeError(f"Parent episode overran G2 {seed}")
                    row = rows[d]
                    if not np.array_equal(observation, np.asarray(json.loads(row["decision_observation_json"]), dtype=np.float32)):
                        raise RuntimeError(f"Decision observation changed {seed}/{d}")
                    if d >= 1:
                        vector, audit = extract(env)
                        delta = abs(audit["current_distal_force_final_N"] - float(rows[d-1]["distal_contact_force_final_N"]))
                        status["max_abs_difference"]["prior_final_force_N"] = max(status["max_abs_difference"].get("prior_final_force_N", 0.), delta)
                        if delta > TOLERANCE:
                            raise RuntimeError(f"Pre-action P force misaligned {seed}/{d}: {delta}")
                        decisions.append(d)
                        physical.append(vector)
                        status["physical_rows"] += 1
                    action = agent.act(observation, first_step=d == 0, eval_mode=True)
                    if not np.array_equal(action, np.asarray(json.loads(row["selected_action_json"]), dtype=np.float32)):
                        raise RuntimeError(f"Behavior action changed {seed}/{d}")
                    state = selected.get((seed, d))
                    forked_behavior = None
                    if state is not None:
                        with torch.inference_mode():
                            latent = agent.model.encode(agent._tensor_observation(observation))[0].cpu().numpy()
                        if not np.array_equal(latent, source_latent[lookup[(seed, d)]]):
                            raise RuntimeError(f"Frozen latent changed {seed}/{d}")
                        saved = json.loads((G3 / "development_candidates" / state["group"] / f"state_{state['state_id']:04d}.json").read_text())
                        if saved["seed"] != seed or saved["decision_index"] != d or not np.array_equal(np.asarray(saved["observation"], np.float32), observation):
                            raise RuntimeError(f"Selected G3 state misaligned {seed}/{d}")
                        if not np.array_equal(np.asarray(saved["latent"], np.float32), latent):
                            raise RuntimeError(f"Selected G3 latent misaligned {seed}/{d}")
                        candidates = generate_candidates(action, state["state_id"], state["group"])
                        if len(candidates) != 21 or len(saved["candidates"]) != 21:
                            raise RuntimeError("Candidate count changed")
                        for candidate, reference in zip(candidates, saved["candidates"]):
                            if candidate != {key: reference[key] for key in candidate}:
                                raise RuntimeError(f"Candidate action protocol changed {seed}/{d}/{candidate['candidate_id']}")
                            actual = fork_step(env, monitor, np.asarray(candidate["action"], dtype=np.float32))
                            compare(actual, reference["outcome"], status["max_abs_difference"], f"G3 candidate {seed}/{d}/{candidate['candidate_id']}")
                            if candidate["candidate_id"] == 0:
                                forked_behavior = actual
                                compare(actual, g2_signature(row), status["max_abs_difference"], f"G2 behavior fork {seed}/{d}")
                            status["candidate_forks"] += 1
                        status["selected_states"] += 1
                    parent = signature(env, monitor, action)
                    compare(parent, g2_signature(row), status["max_abs_difference"], f"G2 parent {seed}/{d}")
                    if forked_behavior is not None:
                        compare(parent, forked_behavior, status["max_abs_difference"], f"parent-fork {seed}/{d}")
                    observation = np.asarray(parent["observation"], dtype=np.float32)
                    terminated, truncated = parent["terminated"], parent["truncated"]
                    d += 1
                    status["parent_steps"] += 1
                if d != len(rows) or len(physical) != d-1:
                    raise RuntimeError(f"G2 episode length or P row count changed {seed}")
                path = EPISODES / f"seed_{seed}.npz"
                if path.exists():
                    with np.load(path) as previous:
                        if not np.array_equal(previous["decision_index"], decisions) or not np.array_equal(previous["physical"], physical):
                            raise RuntimeError(f"Resumed P extraction not bitwise identical {seed}")
                else:
                    save_episode(path, seed, decisions, physical)
                status["parent_episodes"] += 1
                status["last_seed"] = seed
                write_json(status_path, status)
                print(f"aligned seed={seed} episodes={status['parent_episodes']}/200 states={status['selected_states']}/800 forks={status['candidate_forks']}/16800", flush=True)
        finally:
            env.close()
        if (status["parent_episodes"], status["parent_steps"], status["physical_rows"], status["selected_states"], status["candidate_forks"]) != (200, 15500, 15300, 800, 16800):
            raise RuntimeError("G4 physical alignment totals incomplete")
        status["status"] = "PASS_G3_G2_PARENT_CANDIDATE_AND_PREACTION_PHYSICAL_ALIGNMENT"
    except KeyboardInterrupt:
        status["status"] = "INTERRUPTED"
        raise
    except BaseException:
        status["status"] = "FAIL_STOP"
        status["traceback"] = traceback.format_exc()
        raise
    finally:
        write_json(status_path, status)


if __name__ == "__main__":
    main()
