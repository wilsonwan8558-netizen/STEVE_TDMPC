"""Check G1 config against the trusted historical Original checkpoint and G0.5."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[3]
HISTORICAL_CHECKPOINT = Path("/home/bizon/Desktop/stEVE/RL_TDMPC/checkpoints/step_260000.pt")
EXPECTED_HISTORICAL_SHA = "4ee8567514d80cf545e53b0a93c9400aaf90fb0eaba086c760c5cc59e14c1778"
EXPECTED_ADAPTER_SHA = "f6e397c8f43cfe0d32d4d5e0b834ae6f8f0ebb7119a2f3f5756ef47aa5e27b67"
RUN_REL = "runs/generic_solver_tdmpc2_baseline_300k_20261006"
UNCHANGED_TRAINING_PATHS = (
    "RL_TDMPC/train.py", "RL_TDMPC/tdmpc2", "RL_TDMPC/envs/steve_env.py",
    "RL_TDMPC/configs/steve.yaml",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha256(HISTORICAL_CHECKPOINT) != EXPECTED_HISTORICAL_SHA:
        raise RuntimeError("Historical Original checkpoint SHA256 changed")
    if sha256(REPO / "eve/intervention/simulation/sofabeamadapter.py") != EXPECTED_ADAPTER_SHA:
        raise RuntimeError("Frozen G0/G0.5 Generic adapter SHA256 changed")
    phase = json.loads((REPO / "RL_TDMPC/logs/generic_solver_phase_g05_long_horizon_validation/metadata.json").read_text())
    if phase.get("status") != "COMPLETE_CONDITIONAL_GO":
        raise RuntimeError("G0.5 conditional validation artifact missing")
    # The trusted July checkpoint was pickled with NumPy 2. Its config is
    # authoritative; these aliases only let NumPy 1.24 unpickle its metadata.
    sys.modules["numpy._core"] = np.core
    sys.modules["numpy._core.multiarray"] = np.core.multiarray
    sys.modules["numpy._core.numeric"] = np.core.numeric
    sys.modules["numpy._core.umath"] = np.core.umath
    sys.modules["numpy._core._multiarray_umath"] = np.core._multiarray_umath
    historical = torch.load(HISTORICAL_CHECKPOINT, map_location="cpu", weights_only=False)
    if historical["format_version"] != 1 or historical["algorithm"] != "TD-MPC2":
        raise RuntimeError("Unexpected historical checkpoint format/algorithm")
    if historical["total_env_steps"] != 260000 or historical["config"]["training"]["seed"] != 1:
        raise RuntimeError("Unexpected historical step or training seed")
    with args.config.open(encoding="utf-8") as stream:
        new_config = yaml.safe_load(stream)
    expected = copy.deepcopy(historical["config"])
    expected["training"]["total_steps"] = 300000
    expected["logging"]["directory"] = f"{RUN_REL}/logs"
    expected["checkpoint"]["directory"] = f"{RUN_REL}/checkpoints"
    expected["evaluation"]["log_directory"] = f"{RUN_REL}/evaluation"
    if new_config != expected:
        raise RuntimeError("G1 config differs from checkpoint beyond budget/output paths")
    with (REPO / "RL_TDMPC/configs/steve.yaml").open(encoding="utf-8") as stream:
        repository_default = yaml.safe_load(stream)
    if historical["config"] != repository_default:
        raise RuntimeError("Historical checkpoint config differs from repository default")
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    if branch != "experiment/generic-solver-baseline":
        raise RuntimeError(f"Wrong worktree branch: {branch}")
    unchanged = subprocess.run(
        ["git", "diff", "--quiet", "2a26bd7", "HEAD", "--", *UNCHANGED_TRAINING_PATHS],
        cwd=REPO, check=False,
    ).returncode == 0
    if not unchanged:
        raise RuntimeError("Original TD-MPC2 implementation differs from initial baseline commit")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("G1 requires the single CUDA_VISIBLE_DEVICES=0 GPU")
    visible_device_name = torch.cuda.get_device_name(0)
    if "RTX 5090" not in visible_device_name:
        raise RuntimeError(f"Unexpected CUDA device: {visible_device_name}")
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    output = {
        "status": "PASS",
        "historical_command_note": "The literal July shell invocation is not archived. The checkpoint-embedded config exactly equals the repository default; the documented entrypoint is python RL_TDMPC/train.py. No historical CLI overrides are assumed.",
        "historical_checkpoint": str(HISTORICAL_CHECKPOINT),
        "historical_checkpoint_sha256": EXPECTED_HISTORICAL_SHA,
        "historical_checkpoint_format_version": historical["format_version"],
        "historical_checkpoint_step": historical["total_env_steps"],
        "historical_training_seed": historical["config"]["training"]["seed"],
        "historical_training_budget": historical["config"]["training"]["total_steps"],
        "historical_checkpoint_interval": historical["config"]["checkpoint"]["interval"],
        "historical_evaluation_cadence": "No evaluation in train.py; evaluation config is for separate evaluate.py runs.",
        "only_g1_config_differences": ["training.total_steps: 1000000 -> 300000", "logging.directory", "checkpoint.directory", "evaluation.log_directory"],
        "g1_config": str(args.config.resolve()),
        "g1_config_sha256": sha256(args.config),
        "branch": branch,
        "git_commit": commit,
        "original_train_code_unchanged_since_initial_commit": unchanged,
        "frozen_adapter_sha256": EXPECTED_ADAPTER_SHA,
        "g05_decision": phase["decision"],
        "python": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_device_count": torch.cuda.device_count(),
        "physical_cuda_device_index": 0,
        "visible_cuda_device_index": 0,
        "cuda_device_name": visible_device_name,
        "cuda_free_bytes_before_launch": free_bytes,
        "cuda_total_bytes": total_bytes,
        "training_total_steps": new_config["training"]["total_steps"],
        "checkpoint_steps_requested": list(range(200000, 300001, 10000)),
        "force_in_observation_reward_planner_or_termination": False,
        "safety_model_or_shield_active": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
