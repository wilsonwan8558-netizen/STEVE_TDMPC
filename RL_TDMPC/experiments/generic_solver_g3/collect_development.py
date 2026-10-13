"""Collect G3 Generic one-step candidate labels with one reused SOFA scene.

Resume replays all earlier seeds from 90000 with a new scene and skips only
already completed candidate *files*. Every parent step is re-executed and
checked against G2, preserving historical scene history.
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(RL / "experiments/generic_solver_g2"))
sys.path.insert(0, str(RL))
sys.path.insert(0, str(RL.parent))

from envs.steve_env import make_steve_env  # noqa: E402
from tdmpc2.agent import TDMPC2Agent  # noqa: E402
from tdmpc2.common import load_torch_checkpoint, set_seed  # noqa: E402
from train import build_agent_config  # noqa: E402
from substep_monitor import DistalForceMonitor  # noqa: E402
from collect import check_scene  # noqa: E402
from prepare import G1B, G2, OUT, CONFIG, EXPECTED_CHECKPOINT_SHA, EXPECTED_G2_SHA, sha, check_force_source  # noqa: E402
from protocol import generate_candidates  # noqa: E402
from restore_audit import fork_step, signature, compare, g2_signature, TOLERANCE  # noqa: E402


def atomic_json(path: Path, data) -> None:
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False)+"\n")
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    status_path = OUT / "development_collection_status.json"
    state_dir = OUT / "development_candidates"
    state_dir.mkdir(parents=True, exist_ok=True)
    status = {"status": "RUNNING", "method": "one reused Generic scene; complete parent replay from seed 90000 on resume",
              "completed_states": 0, "parent_episodes_compared": 0, "parent_steps_compared": 0,
              "one_step_evaluations_completed": 0, "max_abs_difference": {}, "resume": bool(args.resume)}
    if status_path.exists():
        old = json.loads(status_path.read_text())
        if not args.resume or old["status"] not in ("RUNNING", "INTERRUPTED"):
            raise RuntimeError(f"Cannot resume collection status {old['status']}")
    atomic_json(status_path, status)
    try:
        check_force_source()
        audit = json.loads((OUT / "counterfactual_restore_audit.json").read_text())
        if audit["status"] != "PASS_GENERIC_FORK_RESTORE_PARENT_AND_G2_FORCE":
            raise RuntimeError("Generic fork audit has not passed")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Frozen CUDA device not visible")
        checkpoint_path = Path(json.loads((G1B / "selected_checkpoint.json").read_text())["path"])
        if sha(checkpoint_path) != EXPECTED_CHECKPOINT_SHA or sha(G2 / "step_level_force.csv") != EXPECTED_G2_SHA:
            raise RuntimeError("Frozen checkpoint or G2 dataset identity changed")
        selection = json.loads((OUT / "development_state_selection.json").read_text())["states"]
        selected = {(s["seed"], s["decision_index"]): s for group in ("train", "validation", "development_test") for s in selection[group]}
        if len(selected) != 800:
            raise RuntimeError("Expected 800 selected development states")
        with (G2 / "step_level_force.csv").open(newline="") as stream:
            rows_by_seed: dict[int, list[dict]] = {}
            for row in csv.DictReader(stream):
                rows_by_seed.setdefault(int(row["seed"]), []).append(row)
        with np.load(OUT / "behavior_features.npz") as feature_file:
            lookup = {(int(seed), int(d)): i for i, (seed, d) in enumerate(zip(feature_file["seed"], feature_file["decision_index"]))}
            cached_latent = feature_file["latent"]
        checkpoint = load_torch_checkpoint(checkpoint_path, map_location="cpu")
        config = checkpoint["config"]
        if config != yaml.safe_load(CONFIG.read_text()):
            raise RuntimeError("Frozen config mismatch")
        set_seed(90000)
        env = make_steve_env(config["environment"])
        agent = TDMPC2Agent(14, 2, build_agent_config(config), episode_length=int(config["environment"]["max_episode_steps"]), device=torch.device("cuda:0"))
        agent.load_state_dict(checkpoint["agent"], load_optimizers=False)
        monitor = DistalForceMonitor(env)
        completed = 0
        try:
            for seed in range(90000, 90200):
                set_seed(seed)
                env.action_space.seed(seed)
                observation, _ = env.reset(seed=seed)
                check_scene(env)
                if not hasattr(monitor, "_g3_added"):
                    env._simulation.root.addObject(monitor)
                    monitor._g3_added = True
                rows = rows_by_seed[seed]
                terminated = truncated = False
                d = 0
                while not (terminated or truncated):
                    if d >= len(rows):
                        raise RuntimeError(f"G2 parent episode overrun {seed}")
                    row = rows[d]
                    if not np.array_equal(observation, np.asarray(json.loads(row["decision_observation_json"]), dtype=np.float32)):
                        raise RuntimeError(f"G2 decision observation differs {seed}/{d}")
                    action = agent.act(observation, first_step=d == 0, eval_mode=True)
                    if not np.array_equal(action, np.asarray(json.loads(row["selected_action_json"]), dtype=np.float32)):
                        raise RuntimeError(f"G2 behavior action differs {seed}/{d}")
                    state = selected.get((seed, d))
                    forked_behavior = None
                    if state is not None:
                        with torch.inference_mode():
                            latent = agent.model.encode(agent._tensor_observation(observation))[0].cpu().numpy()
                        if not np.array_equal(latent, cached_latent[lookup[(seed, d)]]):
                            raise RuntimeError(f"G2 latent differs {seed}/{d}")
                        group, state_id = state["group"], state["state_id"]
                        path = state_dir / group / f"state_{state_id:04d}.json"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        candidates = generate_candidates(action, state_id, group)
                        if path.exists():
                            saved = json.loads(path.read_text())
                            if saved["seed"] != seed or saved["decision_index"] != d or len(saved["candidates"]) != 21:
                                raise RuntimeError(f"Completed state file invalid {path}")
                            if saved["candidates"][0]["action"] != candidates[0]["action"]:
                                raise RuntimeError(f"Completed behavior action changed {path}")
                            forked_behavior = saved["candidates"][0]["outcome"]
                        else:
                            outcomes = []
                            for candidate in candidates:
                                outcome = fork_step(env, monitor, np.asarray(candidate["action"], dtype=np.float32))
                                if candidate["candidate_id"] == 0:
                                    forked_behavior = outcome
                                    compare(outcome, g2_signature(row), status["max_abs_difference"], f"behavior fork G2 {seed}/{d}")
                                outcomes.append({**candidate, "force_next_max_N": outcome["force_next_max_N"], "outcome": outcome})
                            atomic_json(path, {**state, "observation": observation.tolist(), "latent": latent.tolist(),
                                               "behavior_action": action.tolist(), "candidates": outcomes})
                            status["one_step_evaluations_completed"] += 21
                        completed += 1
                        status["completed_states"] = completed
                        status["last_state"] = {"seed": seed, "decision_index": d, "group": group, "state_id": state_id}
                        atomic_json(status_path, status)
                        print(f"state {completed}/800 {group} seed={seed} d={d}", flush=True)
                    parent = signature(env, monitor, action)
                    compare(parent, g2_signature(row), status["max_abs_difference"], f"parent G2 {seed}/{d}")
                    if forked_behavior is not None:
                        compare(forked_behavior, parent, status["max_abs_difference"], f"behavior fork parent {seed}/{d}")
                    observation = np.asarray(parent["observation"], dtype=np.float32)
                    terminated, truncated = parent["terminated"], parent["truncated"]
                    d += 1
                    status["parent_steps_compared"] += 1
                if d != len(rows):
                    raise RuntimeError(f"G2 parent episode length differs {seed}")
                status["parent_episodes_compared"] += 1
                atomic_json(status_path, status)
                print(f"episode {seed} parent_steps={d} completed_states={completed}", flush=True)
        finally:
            env.close()
        if completed != 800 or status["parent_steps_compared"] != 15500:
            raise RuntimeError("Incomplete development collection")
        status["status"] = "COMPLETE_ALL_800_STATES_AND_G2_PARENT_REPRODUCED"
    except KeyboardInterrupt:
        status["status"] = "INTERRUPTED"
        raise
    except BaseException:
        status["status"] = "FAIL_STOP"
        status["traceback"] = traceback.format_exc()
        raise
    finally:
        atomic_json(status_path, status)


if __name__ == "__main__":
    main()
