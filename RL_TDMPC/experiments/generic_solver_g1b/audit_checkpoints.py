"""Require and hash all six predeclared G1B maturity checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import torch
import yaml

STEPS = (200000, 220000, 240000, 260000, 280000, 300000)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_hash(state: dict) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output = args.output.resolve()
    with (run_dir / "config.yaml").open(encoding="utf-8") as stream:
        expected_config = yaml.safe_load(stream)
    records = []
    for step in STEPS:
        path = run_dir / "checkpoints" / f"step_{step}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Required G1B checkpoint missing: {path}")
        file_hash = sha256(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if (checkpoint.get("format_version") != 1 or
                checkpoint.get("algorithm") != "TD-MPC2" or
                checkpoint.get("total_env_steps") != step or
                checkpoint.get("config") != expected_config or
                checkpoint["config"]["training"]["seed"] != 1):
            raise RuntimeError(f"Checkpoint identity/config mismatch at step {step}")
        records.append({
            "step": step, "path": str(path), "sha256": file_hash,
            "model_state_sha256": model_hash(checkpoint["agent"]["model"]),
            "format_version": checkpoint["format_version"],
            "training_seed": checkpoint["config"]["training"]["seed"],
            "episode_index": int(checkpoint["episode_index"]),
            "agent_update_count": int(checkpoint["agent"]["update_count"]),
        })
    repo = Path(__file__).resolve().parents[3]
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "ALL_SIX_CHECKPOINTS_VERIFIED",
        "branch": subprocess.check_output(["git", "branch", "--show-current"],
                                          cwd=repo, text=True).strip(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                              cwd=repo, text=True).strip(),
        "training_config_path": str(run_dir / "config.yaml"),
        "training_config_sha256": sha256(run_dir / "config.yaml"),
        "checkpoint_records": records,
    }
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
