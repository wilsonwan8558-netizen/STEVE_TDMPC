"""Read-only pre-action audit of compact Generic SOFA physical features."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys

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
from force_reconstruction import reconstruct_distal_contact_force  # noqa: E402
from restore_audit import signature, compare, g2_signature  # noqa: E402
from prepare import G1B, G2, CONFIG, EXPECTED_CHECKPOINT_SHA, EXPECTED_G2_SHA, sha, check_force_source  # noqa: E402

OUT = RL / "logs/generic_solver_phase_g4_state_observability"
PROBE_SEEDS = (90000, 90001, 90002)


def physical_snapshot(env):
    sim = env._simulation
    collision = sim._instruments_combined.getChild("CollisionModel")
    dofs = collision.CollisionDOFs
    positions = np.asarray(dofs.position.value, dtype=float)
    velocity = np.asarray(dofs.velocity.value, dtype=float)
    edges = np.asarray(collision.collisEdgeSet.findData("edges").value, dtype=int)
    solver = sim.root.getObject("Generic")
    multipliers = np.asarray(solver.findData("constraintForces").value, dtype=float).reshape(-1)
    sample = reconstruct_distal_contact_force(multipliers, dofs.findData("constraint").value,
                                               positions, edges, .006, 3.0)
    return {"collision_positions_shape": list(positions.shape), "collision_velocity_shape": list(velocity.shape),
            "edge_shape": list(edges.shape), "multiplier_count": len(multipliers),
            "tip_position_mm": positions[-1].tolist(), "tip_velocity_scene_units_per_s": velocity[-1].tolist(),
            "last5_positions_mm": positions[-5:].tolist(), "last5_velocity": velocity[-5:].tolist(),
            "current_distal_force_final_N": sample.magnitude_N,
            "current_distal_force_vector_N": sample.vector_N.tolist(), "current_distal_contact_count": sample.contact_count,
            "insertion_length_mm": float(sim.inserted_lengths[0]), "rotation": float(sim.rotations[0])}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "physical_source_probe.json"
    if path.exists():
        raise FileExistsError(path)
    check_force_source()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Frozen RTX 5090 CUDA_VISIBLE_DEVICES=0 required")
    checkpoint_path = Path(json.loads((G1B / "selected_checkpoint.json").read_text())["path"])
    if sha(checkpoint_path) != EXPECTED_CHECKPOINT_SHA or sha(G2 / "step_level_force.csv") != EXPECTED_G2_SHA:
        raise RuntimeError("Frozen checkpoint or G2 step dataset changed")
    with (G2 / "step_level_force.csv").open(newline="") as stream:
        rows_by_seed = {}
        for row in csv.DictReader(stream):
            seed = int(row["seed"])
            if seed in PROBE_SEEDS:
                rows_by_seed.setdefault(seed, []).append(row)
    checkpoint = load_torch_checkpoint(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    if config != yaml.safe_load(CONFIG.read_text()):
        raise RuntimeError("Frozen config changed")
    set_seed(PROBE_SEEDS[0])
    env = make_steve_env(config["environment"])
    agent = TDMPC2Agent(14, 2, build_agent_config(config), episode_length=int(config["environment"]["max_episode_steps"]), device=torch.device("cuda:0"))
    agent.load_state_dict(checkpoint["agent"], load_optimizers=False)
    monitor = DistalForceMonitor(env)
    report = {"status": "RUNNING", "seeds": PROBE_SEEDS, "snapshots": [], "max_difference": {},
              "source_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA, "source_g2_step_sha256": EXPECTED_G2_SHA}
    try:
        for seed in PROBE_SEEDS:
            set_seed(seed)
            env.action_space.seed(seed)
            observation, _ = env.reset(seed=seed)
            check_scene(env)
            if not hasattr(monitor, "_g4_added"):
                env._simulation.root.addObject(monitor)
                monitor._g4_added = True
            rows = rows_by_seed[seed]
            terminated = truncated = False
            d = 0
            while not (terminated or truncated):
                row = rows[d]
                if not np.array_equal(observation, np.asarray(json.loads(row["decision_observation_json"]), dtype=np.float32)):
                    raise RuntimeError(f"Decision observation differs {seed}/{d}")
                if d in (1, 5, min(15, len(rows)-3), len(rows)//2):
                    snap = physical_snapshot(env)
                    expected_final = float(rows[d-1]["distal_contact_force_final_N"])
                    difference = abs(snap["current_distal_force_final_N"]-expected_final)
                    report["max_difference"]["previous_step_final_force_N"] = max(report["max_difference"].get("previous_step_final_force_N", 0.0), difference)
                    if difference > 1e-8:
                        raise RuntimeError(f"Pre-action force does not align with previous final substep {seed}/{d}: {difference}")
                    if snap["collision_positions_shape"][0] < 5 or snap["collision_velocity_shape"] != snap["collision_positions_shape"]:
                        raise RuntimeError("No reliable five-node position/velocity feature")
                    if not np.all(np.isfinite(snap["last5_velocity"])):
                        raise RuntimeError("Nonfinite distal collision-node velocity")
                    report["snapshots"].append({"seed": seed, "decision_index": d, "previous_step_final_force_N": expected_final, **snap})
                action = agent.act(observation, first_step=d == 0, eval_mode=True)
                if not np.array_equal(action, np.asarray(json.loads(row["selected_action_json"]), dtype=np.float32)):
                    raise RuntimeError(f"Behavior action differs {seed}/{d}")
                parent = signature(env, monitor, action)
                compare(parent, g2_signature(row), report["max_difference"], f"G2 parent {seed}/{d}")
                observation = np.asarray(parent["observation"], dtype=np.float32)
                terminated, truncated = parent["terminated"], parent["truncated"]
                d += 1
            if d != len(rows):
                raise RuntimeError(f"Episode length differs {seed}")
        report["status"] = "PASS_PREACTION_PHYSICAL_SOURCE_AND_G2_ALIGNMENT"
    finally:
        env.close()
        path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(f"{report['status']}: {len(report['snapshots'])} snapshots", flush=True)


if __name__ == "__main__":
    main()
