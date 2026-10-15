"""Validate and join frozen Generic-SOFA transitions for G12 force-head fitting."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

RL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RL))
from tdmpc2.agent import TDMPC2Agent  # noqa: E402
from tdmpc2.common import load_torch_checkpoint  # noqa: E402
from train import build_agent_config  # noqa: E402

OUT = RL / "logs/world_model_force_aware_planning_g12"
G2 = RL / "logs/generic_solver_phase_g2_force_baseline"
G3 = RL / "logs/generic_solver_phase_g3_force_predictor"
CHECKPOINT = RL / "runs/generic_solver_tdmpc2_baseline_300k_20261006/checkpoints/step_280000.pt"
EXPECTED = {
    CHECKPOINT: "476564f8d301ae0c76c418d8e61a4129037b30de7c6872377b563c70e84acd99",
    G2 / "step_level_force.csv": "cc15380d976e75ed262b372445e1fabeaf04f4ecd6592c6e027a8e7acf1af737",
    G3 / "behavior_features.npz": "116112721a51111298fde49cd26434f89dd2de3f978f395fe7207719bb263ef8",
    G3 / "development_split.json": "e55172e189f419b7033f2b8df1535dcc0c84bdc17aa9dc545a8bda5669c48e60",
    RL / "experiments/generic_solver_g2/force_reconstruction.py": "76c920ce5122cb8da97ba100851985267e734b2877fc41ab6726217cdb13fa26",
    RL / "experiments/generic_solver_g2/substep_monitor.py": "744759896f36d284c9ba039a537a9f5334a5e6f5d99752f036062ae55ced3534",
    RL.parent / "eve/intervention/simulation/sofabeamadapter.py": "f6e397c8f43cfe0d32d4d5e0b834ae6f8f0ebb7119a2f3f5756ef47aa5e27b67",
}
F_LIM = 3.685331040756903


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def new_json(name: str, data: dict) -> None:
    path = OUT / name
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main() -> None:
    audit = json.loads((OUT / "tdmpc2_planner_audit.json").read_text())
    if audit["status"] != "COMPLETE_BEFORE_G12_SOURCE_MODIFICATION":
        raise RuntimeError("G12 planner audit absent or invalid")
    for path, expected in EXPECTED.items():
        if sha(path) != expected:
            raise RuntimeError(f"Frozen G12 source/dataset identity changed: {path}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or "RTX 5090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("G12 force dataset validation requires CUDA_VISIBLE_DEVICES=0 RTX 5090")
    if (OUT / "force_transitions.npz").exists():
        raise FileExistsError("G12 joined dataset already exists")

    with (G2 / "step_level_force.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 15500:
        raise RuntimeError("Frozen G2 behavior dataset changed size")
    with np.load(G3 / "behavior_features.npz", allow_pickle=False) as data:
        features = {key: data[key] for key in data.files}
    for key, shape in {"seed": (15500,), "decision_index": (15500,),
                       "observation": (15500, 14), "latent": (15500, 256),
                       "action": (15500, 2), "next_observation": (15500, 14),
                       "force_next_max_N": (15500,)}.items():
        if features[key].shape != shape:
            raise RuntimeError(f"G3 feature shape changed: {key}")

    rewards = np.empty(15500, dtype=np.float64)
    terminated = np.empty(15500, dtype=np.bool_)
    truncated = np.empty(15500, dtype=np.bool_)
    episode_lengths = {seed: 0 for seed in range(90000, 90200)}
    for i, row in enumerate(rows):
        seed = int(row["seed"])
        step = int(row["step_index"])
        if (seed not in episode_lengths or step != episode_lengths[seed] or
                int(features["seed"][i]) != seed or int(features["decision_index"][i]) != step):
            raise RuntimeError(f"G12 seed/step misalignment at row {i}")
        for key, column in (("observation", "decision_observation_json"),
                            ("action", "selected_action_json"),
                            ("next_observation", "next_observation_json")):
            value = np.asarray(json.loads(row[column]), dtype=np.float32)
            if not np.array_equal(features[key][i], value):
                raise RuntimeError(f"G12 {key} mismatch at row {i}")
        if float(features["force_next_max_N"][i]) != float(row["distal_contact_force_max_N"]):
            raise RuntimeError(f"G12 force label mismatch at row {i}")
        rewards[i] = float(row["reward"])
        terminated[i] = row["terminated"] == "True"
        truncated[i] = row["truncated"] == "True"
        if i and step and not np.array_equal(features["next_observation"][i-1], features["observation"][i]):
            raise RuntimeError(f"G12 previous next observation != current observation at row {i}")
        episode_lengths[seed] += 1
    if (not np.all(np.isfinite(rewards)) or not np.all(np.isfinite(features["latent"])) or
            not np.all(np.isfinite(features["force_next_max_N"])) or
            np.any(features["force_next_max_N"] < 0)):
        raise RuntimeError("G12 nonfinite reward/latent/force or negative force")
    with (G2 / "episode_level_force.csv").open(newline="", encoding="utf-8") as stream:
        episodes = list(csv.DictReader(stream))
    if (len(episodes) != 200 or [int(row["seed"]) for row in episodes] != list(range(90000, 90200)) or
            any(episode_lengths[int(row["seed"])] != int(row["episode_length"]) for row in episodes)):
        raise RuntimeError("G12 episode index or lengths changed")

    checkpoint = load_torch_checkpoint(CHECKPOINT, map_location="cpu")
    if (checkpoint["format_version"] != 1 or checkpoint["algorithm"] != "TD-MPC2" or
            checkpoint["total_env_steps"] != 280000):
        raise RuntimeError("G12 checkpoint format changed")
    config = checkpoint["config"]
    agent = TDMPC2Agent(14, 2, build_agent_config(config),
                        episode_length=int(config["environment"]["max_episode_steps"]),
                        device=torch.device("cuda:0"))
    agent.load_state_dict(checkpoint["agent"], load_optimizers=False)
    with torch.inference_mode():
        for i in (0, 1, 100, 1000, 15499):
            latent = agent.model.encode(agent._tensor_observation(features["observation"][i]))[0].cpu().numpy()
            if not np.array_equal(latent, features["latent"][i]):
                raise RuntimeError(f"G12 cached latent differs from frozen encoder at row {i}")

    split_source = json.loads((G3 / "development_split.json").read_text())
    split = {key: split_source[key] for key in ("train", "validation", "development_test")}
    if (len(split["train"]) != 140 or len(split["validation"]) != 30 or
            len(split["development_test"]) != 30 or
            sorted(split["train"] + split["validation"] + split["development_test"]) != list(range(90000, 90200))):
        raise RuntimeError("G12 frozen episode split invalid")
    np.savez_compressed(OUT / "force_transitions.npz", **features, reward=rewards,
                        terminated=terminated, truncated=truncated)
    new_json("force_dataset_split.json", {
        "status": "FROZEN_BEFORE_G12_FORCE_HEAD_TRAINING",
        "split_unit": "whole reused-scene episode seed, never random steps",
        "split_source": str(G3 / "development_split.json"),
        "split_source_sha256": EXPECTED[G3 / "development_split.json"],
        "train_seeds": split["train"], "validation_seeds": split["validation"],
        "heldout_test_seeds": split["development_test"],
        "final_confirmation_seeds": "not selected; requires a separate later reuse audit",
    })
    new_json("force_dataset_manifest.json", {
        "status": "VALIDATED_EXISTING_GENERIC_SOFA_TRANSITIONS",
        "joined_dataset": str(OUT / "force_transitions.npz"),
        "joined_dataset_sha256": sha(OUT / "force_transitions.npz"),
        "source_sha256": {str(path): digest for path, digest in EXPECTED.items()},
        "rows": 15500, "episodes": 200,
        "fields": [*features, "reward", "terminated", "truncated"],
        "pre_action_input": "float32 observation and frozen checkpoint encoder latent z_t; normalized float32 selected action a_t",
        "transition": "one RL env.step(a_t), 22 SOFA substeps, next_observation and reward",
        "force_target": "true max-substep distal 3 mm contact force during this same transition, nominal simulation N",
        "force_target_field": "force_next_max_N",
        "F_lim_N_nominal": F_LIM,
        "exact_source_row_alignment_checked": 15500,
        "exact_cached_encoder_samples_checked": [0, 1, 100, 1000, 15499],
        "high_force_rows": int(np.count_nonzero(features["force_next_max_N"] > F_LIM)),
        "zero_force_rows": int(np.count_nonzero(features["force_next_max_N"] == 0)),
        "python": sys.executable, "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "cuda_device": torch.cuda.get_device_name(0),
    })
    print("G12_EXISTING_FORCE_TRANSITIONS_VALIDATED", len(rows), flush=True)


if __name__ == "__main__":
    main()
