"""Freeze the Generic G3 development protocol and extract passive policy latents."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
sys.path.insert(0, str(RL))
from tdmpc2.agent import TDMPC2Agent  # noqa: E402
from tdmpc2.common import load_torch_checkpoint  # noqa: E402
from train import build_agent_config  # noqa: E402

G2 = RL / "logs/generic_solver_phase_g2_force_baseline"
G1B = RL / "logs/generic_solver_phase_g1b_checkpoint_sweep"
OUT = RL / "logs/generic_solver_phase_g3_force_predictor"
CONFIG = RL / "runs/generic_solver_tdmpc2_baseline_300k_20261006/config.yaml"
EXPECTED_CHECKPOINT_SHA = "476564f8d301ae0c76c418d8e61a4129037b30de7c6872377b563c70e84acd99"
EXPECTED_G2_SHA = "cc15380d976e75ed262b372445e1fabeaf04f4ecd6592c6e027a8e7acf1af737"
EXPECTED_FORCE_RECONSTRUCTION_SHA = "76c920ce5122cb8da97ba100851985267e734b2877fc41ab6726217cdb13fa26"
EXPECTED_MONITOR_SHA = "744759896f36d284c9ba039a537a9f5334a5e6f5d99752f036062ae55ced3534"
EXPECTED_ADAPTER_SHA = "f6e397c8f43cfe0d32d4d5e0b834ae6f8f0ebb7119a2f3f5756ef47aa5e27b67"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_force_source() -> None:
    expected = ((RL / "experiments/generic_solver_g2/force_reconstruction.py", EXPECTED_FORCE_RECONSTRUCTION_SHA),
                (RL / "experiments/generic_solver_g2/substep_monitor.py", EXPECTED_MONITOR_SHA),
                (RL.parent / "eve/intervention/simulation/sofabeamadapter.py", EXPECTED_ADAPTER_SHA))
    for path, digest in expected:
        if sha(path) != digest:
            raise RuntimeError(f"Frozen G2 force/adapter source changed: {path}")


def save(name: str, data) -> None:
    path = OUT / name
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    check_force_source()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Frozen G3 protocol requires CUDA_VISIBLE_DEVICES=0 and one visible device")
    if "RTX 5090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Frozen CUDA device is not RTX 5090")
    checkpoint_record = json.loads((G1B / "selected_checkpoint.json").read_text())
    checkpoint_path = Path(checkpoint_record["path"])
    if checkpoint_record["step"] != 280000 or sha(checkpoint_path) != checkpoint_record["sha256"] or sha(checkpoint_path) != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("Frozen checkpoint identity failed")
    step_path = G2 / "step_level_force.csv"
    if sha(step_path) != EXPECTED_G2_SHA:
        raise RuntimeError("Frozen G2 step dataset identity failed")
    with step_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    with (G2 / "episode_level_force.csv").open(newline="") as stream:
        episodes = list(csv.DictReader(stream))
    if len(rows) != 15500 or len(episodes) != 200 or [int(r["seed"]) for r in episodes] != list(range(90000, 90200)):
        raise RuntimeError("G2 episode/step counts changed")
    grouped = {seed: [] for seed in range(90000, 90200)}
    for row in rows:
        seed, decision = int(row["seed"]), int(row["step_index"])
        if seed not in grouped or decision != len(grouped[seed]) or int(row["episode_index"]) != seed - 90000:
            raise RuntimeError(f"G2 indexing failure: {seed}/{decision}")
        grouped[seed].append(row)
    if any(len(grouped[int(ep["seed"])]) != int(ep["episode_length"]) for ep in episodes):
        raise RuntimeError("G2 episode lengths do not align")
    rng = np.random.default_rng(20261008)
    shuffled = rng.permutation(np.arange(90000, 90200)).tolist()
    split = {"train": sorted(shuffled[:140]), "validation": sorted(shuffled[140:170]),
             "development_test": sorted(shuffled[170:]), "split_rng": "NumPy default_rng(20261008).permutation",
             "source_seeds": [90000, 90199], "confirmation_seeds_predeclared": [92000, 92029]}
    if any(set(split[a]) & set(split[b]) for a, b in (("train", "validation"), ("train", "development_test"), ("validation", "development_test"))):
        raise RuntimeError("Split overlap")
    checkpoint = load_torch_checkpoint(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    if checkpoint["format_version"] != 1 or checkpoint["algorithm"] != "TD-MPC2" or checkpoint["total_env_steps"] != 280000 or config != yaml.safe_load(CONFIG.read_text()):
        raise RuntimeError("Frozen checkpoint/config validation failed")
    agent = TDMPC2Agent(14, 2, build_agent_config(config), episode_length=int(config["environment"]["max_episode_steps"]), device=torch.device("cuda:0"))
    agent.load_state_dict(checkpoint["agent"], load_optimizers=False)
    agent.model.eval()
    observations = np.asarray([json.loads(row["decision_observation_json"]) for row in rows], dtype=np.float32)
    actions = np.asarray([json.loads(row["selected_action_json"]) for row in rows], dtype=np.float32)
    next_observations = np.asarray([json.loads(row["next_observation_json"]) for row in rows], dtype=np.float32)
    force = np.asarray([float(row["distal_contact_force_max_N"]) for row in rows], dtype=np.float64)
    if observations.shape != (15500, 14) or actions.shape != (15500, 2) or next_observations.shape != (15500, 14) or not np.all(np.isfinite(force)) or np.any(force < 0):
        raise RuntimeError("G2 shape or force validation failed")
    for seed, ep_rows in grouped.items():
        for d in range(1, len(ep_rows)):
            a = np.asarray(json.loads(ep_rows[d]["decision_observation_json"]), dtype=np.float32)
            b = np.asarray(json.loads(ep_rows[d-1]["next_observation_json"]), dtype=np.float32)
            if not np.array_equal(a, b):
                raise RuntimeError(f"Decision/next observation alignment failure at {seed}/{d}")
    # The planner encodes one observation at a time; GPU GEMM kernels can yield
    # slightly different bits when a batch is used, so mirror its exact call.
    with torch.inference_mode():
        latents = np.stack([agent.model.encode(agent._tensor_observation(observation))[0].cpu().numpy()
                            for observation in observations]).astype(np.float32)
    if latents.shape != (15500, 256) or not np.all(np.isfinite(latents)):
        raise RuntimeError("Frozen latent extraction failed")
    for i in (0, 1, 100, 1000, 15499):
        with torch.inference_mode():
            direct = agent.model.encode(agent._tensor_observation(observations[i]))[0].cpu().numpy()
        if not np.array_equal(latents[i], direct):
            raise RuntimeError(f"Batched/direct latent mismatch at row {i}")
    OUT.mkdir(parents=True, exist_ok=True)
    save("development_split.json", split)
    save("checkpoint_audit.json", {"path": str(checkpoint_path), "sha256": sha(checkpoint_path), "format_version": 1,
                                   "total_env_steps": 280000, "latent_dim": 256, "encoder": "agent.model.encode(agent._tensor_observation(o_d))",
                                   "observation_dtype": "float32", "device": "cuda:0", "single_observation_exact_samples": [0, 1, 100, 1000, 15499]})
    save("force_target_definition.json", {"decision_index": "zero-based step_index in G2", "input": "[z_d, a_d]",
                                          "target_field": "force_next_max_N", "g2_source_field": "distal_contact_force_max_N",
                                          "meaning": "maximum distal 3 mm vector-sum-before-norm contact force across all 22 SOFA substeps in env.step(a_d)",
                                          "force_source": "G2 Generic constraintForces mapped by contact Jacobian and divided by 0.006 s",
                                          "units": "nominal simulation N", "source_sha256": EXPECTED_G2_SHA})
    quantiles = json.loads((G2 / "force_distribution_summary.json").read_text())["all_rl_steps"]
    save("metadata.json", {"phase": "G3", "status": "DEVELOPMENT_PROTOCOL_FROZEN", "g2_step_sha256": EXPECTED_G2_SHA,
                           "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA, "quantiles_nominal_N": {k: quantiles[k] for k in ("q90", "q95", "q97.5", "q99")},
                           "python": sys.executable, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                           "cuda_device": torch.cuda.get_device_name(0), "cuda_visible_index": 0, "physical_cuda_index": 0})
    feature_path = OUT / "behavior_features.npz"
    if feature_path.exists():
        raise FileExistsError(feature_path)
    np.savez_compressed(feature_path, seed=np.asarray([int(row["seed"]) for row in rows], dtype=np.int32),
                        decision_index=np.asarray([int(row["step_index"]) for row in rows], dtype=np.int32),
                        observation=observations, latent=latents, action=actions, next_observation=next_observations,
                        force_next_max_N=force)
    save("data_manifest.json", {"behavior_features": str(feature_path), "behavior_features_sha256": sha(feature_path),
                                "behavior_rows": 15500, "split_episode_counts": {k: len(split[k]) for k in ("train", "validation", "development_test")},
                                "latent_extraction": "frozen checkpoint, float32 pre-action G2 observation, CUDA encode; passive offline extraction",
                                "g2_row_alignment": "exact previous next_observation == current decision_observation within each episode"})
    print(f"G3 protocol and 15,500 latents frozen at {OUT}", flush=True)


if __name__ == "__main__":
    main()
