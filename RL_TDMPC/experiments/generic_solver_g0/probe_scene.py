"""Inspect the installed Generic SOFA scene without training a controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from envs.steve_env import StEVEEnv  # noqa: E402


def describe_data(obj, names):
    result = {}
    for name in names:
        data = obj.findData(name)
        value = data.value
        result[name] = value.tolist() if isinstance(value, np.ndarray) else str(value)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=90000)
    parser.add_argument("--insertion-steps", type=int, default=0)
    args = parser.parse_args()
    env = StEVEEnv()
    try:
        observation, info = env.reset(seed=args.seed)
        for _ in range(args.insertion_steps):
            observation, _, terminated, truncated, info = env.step(
                np.asarray([1., 0.], dtype=np.float32))
            if terminated or truncated:
                break
        sim = env._simulation
        solver = sim.root.getObject("Generic")
        contact = sim.root.getObject("DefaultContactManager")
        if solver is None or solver.getClassName() != "GenericConstraintSolver":
            raise RuntimeError("Generic solver missing from initialized SOFA scene")
        if contact is None:
            raise RuntimeError("Contact manager missing")
        collision = sim._instruments_combined.getChild("CollisionModel")
        dofs = np.asarray(sim._instruments_combined.DOFs.position.value)
        collision_dofs = np.asarray(collision.CollisionDOFs.position.value)
        matrix_descriptions = {}
        for label, obj in (("parent", sim._instruments_combined.DOFs),
                           ("collision", collision.CollisionDOFs)):
            for field in ("constraint", "force"):
                try:
                    value = obj.findData(field).value
                    matrix_descriptions[f"{label}_{field}"] = {
                        "python_type": str(type(value)), "shape": list(getattr(value, "shape", [])),
                        "text_prefix": str(value)[:240]}
                except (AttributeError, RuntimeError, ValueError) as error:
                    matrix_descriptions[f"{label}_{field}"] = {"error": str(error)}
        data = {"seed": args.seed, "insertion_steps": args.insertion_steps,
                "observation_shape": list(observation.shape),
                "observation_finite": bool(np.all(np.isfinite(observation))),
                "action_shape": list(env.action_space.shape),
                "initial_info": {k: str(v) for k, v in info.items()},
                "solver": describe_data(solver, ["resolutionMethod", "maxIterations", "tolerance",
                                                "computeConstraintForces", "constraintForces",
                                                "currentNumConstraints", "currentNumConstraintGroups",
                                                "currentIterations", "currentError"]),
                "contact_manager": describe_data(contact, ["response", "responseParams"]),
                "root_dt": float(sim.root.dt.value),
                "parent_dof_shape": list(dofs.shape),
                "parent_first_position": dofs[0, :3].tolist(),
                "parent_last_position": dofs[-1, :3].tolist(),
                "parent_position_range": [np.min(dofs[:, :3], axis=0).tolist(),
                                          np.max(dofs[:, :3], axis=0).tolist()],
                "collision_dof_shape": list(collision_dofs.shape),
                "collision_first_position": collision_dofs[0, :3].tolist(),
                "collision_last_position": collision_dofs[-1, :3].tolist(),
                "collision_position_range": [np.min(collision_dofs[:, :3], axis=0).tolist(),
                                             np.max(collision_dofs[:, :3], axis=0).tolist()],
                "mechanical_field_descriptions": matrix_descriptions,
                "tip_position": np.asarray(sim.dof_positions[0]).tolist()}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(data, indent=2, sort_keys=True), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
