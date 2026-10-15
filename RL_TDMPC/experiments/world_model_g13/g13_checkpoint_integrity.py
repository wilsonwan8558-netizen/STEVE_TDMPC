"""Verify the four predeclared late G13 checkpoints before selection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
OUT = RL / "logs/world_model_force_aware_planning_g13"
RUN = RL / "runs/geometry_force_tdmpc2_g13_300k_20261014/checkpoints"
STEPS = (240000, 260000, 280000, 300000)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    target = OUT / "checkpoint_integrity_audit.json"
    if target.exists():
        raise FileExistsError(target)
    frozen = json.loads((OUT / "g13_training_config.json").read_text())
    if frozen["selected_checkpoint_candidates"] != list(STEPS):
        raise RuntimeError("Late checkpoint set changed")
    rows = []
    for step in STEPS:
        path = RUN / f"step_{step}.pt"
        state = torch.load(path, map_location="cpu", weights_only=False)
        model = state["agent"]["model"]
        nonfinite = [key for key, value in model.items()
                     if torch.is_tensor(value) and not bool(torch.isfinite(value).all())]
        if (state.get("format_version") != 2
                or state.get("algorithm") != "Geometry-Force-TD-MPC2"
                or state.get("total_env_steps") != step
                or state.get("observation_dim") != 22
                or state.get("lambda_force_train") != 0.5
                or state.get("config") != frozen["config"]
                or not any(key.startswith("force_head.") for key in model)
                or nonfinite):
            raise RuntimeError(f"Invalid G13 checkpoint {step}: nonfinite={nonfinite[:5]}")
        rows.append({"step": step, "path": str(path.resolve()),
                     "sha256": sha256(path), "format_version": state["format_version"],
                     "observation_dim": state["observation_dim"],
                     "model_tensor_count": sum(torch.is_tensor(value) for value in model.values()),
                     "nonfinite_model_tensors": 0,
                     "force_head_parameter_keys": sorted(key for key in model if key.startswith("force_head.")),
                     "update_count": state["agent"]["update_count"]})
    target.write_text(json.dumps({
        "status": "PASS_FROZEN_PREDECLARED_G13_CHECKPOINTS_FINITE_AND_FORMAT_MATCHED",
        "candidate_steps": list(STEPS), "checkpoints": rows,
        "checkpoint_selection_not_based_on_force_planning": True,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("G13_CHECKPOINT_INTEGRITY_PASS", [row["step"] for row in rows], flush=True)


if __name__ == "__main__":
    main()
