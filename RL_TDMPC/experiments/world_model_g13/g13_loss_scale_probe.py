"""Training-seed-only G13 initial loss-scale probe, without fitting a model."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
sys.path.insert(0, str(RL))
sys.path.insert(0, str(HERE))
from tdmpc2.common import set_seed  # noqa: E402
from train import build_agent_config  # noqa: E402
from g13_env import make_g13_env  # noqa: E402
from g13_model import G13Agent, G13ReplayBuffer  # noqa: E402

OUT = RL / "logs/world_model_force_aware_planning_g13"
BASELINE_CONFIG = RL / "runs/generic_solver_tdmpc2_baseline_300k_20261006/config.yaml"
SEED = 1
STEPS = 1000
PROBE_BATCHES = 8


def main() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("G13 loss-scale probe requires CUDA_VISIBLE_DEVICES=0")
    config = yaml.safe_load(BASELINE_CONFIG.read_text())
    config["training"]["lambda_force_train"] = 1.0  # raw unweighted force term
    set_seed(SEED)
    env = make_g13_env(config["environment"])
    replay = G13ReplayBuffer(STEPS + 200, 22, 2,
                             int(config["training"]["horizon"]),
                             int(config["training"]["batch_size"]), seed=SEED)
    collected = 0
    episode_index = 0
    force_values = []
    try:
        while collected < STEPS:
            seed = SEED + episode_index
            env.action_space.seed(seed)
            obs, _ = env.reset(seed=seed)
            observations = [obs.copy()]
            actions, rewards, terminated, forces = [], [], [], []
            done = False
            while not done and collected < STEPS:
                action = env.action_space.sample().astype(np.float32)
                obs, reward, term, trunc, info = env.step(action)
                collected += 1
                observations.append(obs.copy())
                actions.append(action.copy())
                rewards.append(float(reward))
                terminated.append(bool(term))
                force = float(info["distal_contact_force_max_N"])
                forces.append(force)
                force_values.append(force)
                done = bool(term or trunc)
            replay.add_episode(observations, actions, rewards, terminated, forces)
            episode_index += 1
            print("G13_LOSS_PROBE_EPISODE", episode_index, collected, len(actions), flush=True)
    finally:
        env.close()
    set_seed(SEED)
    agent = G13Agent(22, 2, build_agent_config(config),
                     episode_length=int(config["environment"]["max_episode_steps"]),
                     device=torch.device("cuda:0"))
    samples = []
    with torch.no_grad():
        for _ in range(PROBE_BATCHES):
            terms, _ = agent.loss_terms(replay.sample(torch.device("cuda:0")))
            samples.append({key: float(value.cpu()) for key, value in terms.items()})
    values = {key: [sample[key] for sample in samples] for key in samples[0]}
    summary = {key: {"mean": float(np.mean(item)),
                     "min": float(np.min(item)), "max": float(np.max(item))}
               for key, item in values.items()}
    force_values = np.asarray(force_values)
    with (OUT / "training_loss_scale_probe.json").open("x", encoding="utf-8") as stream:
        json.dump({
            "status": "INITIAL_RANDOM_POLICY_TRAINING_SEED_LOSS_MAGNITUDES_NO_UPDATES",
            "training_seed_start": SEED, "steps": STEPS,
            "episodes": episode_index, "probe_batches": PROBE_BATCHES,
            "batch_size": int(config["training"]["batch_size"]),
            "raw_force_coefficient_for_probe": 1.0,
            "force_target_nonzero_fraction": float(np.mean(force_values > 0)),
            "force_target_above_F_lim_fraction": float(np.mean(force_values > 3.685331040756903)),
            "force_target_q50_q90_q99_max_N": [float(np.quantile(force_values, q)) for q in (.5, .9, .99, 1)],
            "loss_terms": summary,
            "samples": samples,
        }, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print("G13_LOSS_SCALE_PROBE", json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
