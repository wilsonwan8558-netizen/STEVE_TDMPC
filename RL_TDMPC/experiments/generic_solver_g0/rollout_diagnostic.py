"""Fixed open-loop LCP/Generic SOFA diagnostics; no TD-MPC2 training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

SCENARIOS = {
    "zero": (90000, [([0., 0.], 12)]),
    "translation": (90000, [([0.5, 0.], 14)]),
    "rotation": (90000, [([0., 0.5], 14)]),
    "insert_then_rotate": (90008, [([1., 0.], 14), ([0., 0.5], 14)]),
    "combined": (90000, [([0.5, 0.5], 14)]),
    "contact_heavy": (90008, [([1., 0.], 14), ([1., 1.], 14)]),
}


def data_value(obj, name):
    try:
        return obj.findData(name).value
    except (AttributeError, RuntimeError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--solver", choices=("LCP", "Generic"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenarios", nargs="+", choices=tuple(SCENARIOS), default=list(SCENARIOS))
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    sys.path.insert(0, str(source_root / "RL_TDMPC"))
    sys.path.insert(0, str(source_root))
    from envs.steve_env import StEVEEnv  # noqa: E402

    report = {"source_root": str(source_root), "solver": args.solver,
              "scenario_spec": SCENARIOS, "scenarios": {}}
    if args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        if previous.get("source_root") != str(source_root) or previous.get("solver") != args.solver:
            raise RuntimeError("Cannot append to a rollout from a different source or solver")
        report["scenarios"].update(previous["scenarios"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for scenario in args.scenarios:
        seed, segments = SCENARIOS[scenario]
        env = StEVEEnv()
        try:
            observation, info = env.reset(seed=seed)
            sim = env._simulation
            solver = sim.root.getObject(args.solver)
            if solver is None:
                raise RuntimeError(f"Expected {args.solver} solver missing")
            # LCP exposure is enabled only for this readout, not in its source
            # configuration; this does not change the contact resolution.
            if args.solver == "LCP":
                solver.findData("computeConstraintForces").value = True
            trajectory = []
            last_tip = np.asarray(sim.dof_positions[0], dtype=float)
            step_index = 0
            for action, count in segments:
                for _ in range(count):
                    step_index += 1
                    observation, reward, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
                    dofs = np.asarray(sim._instruments_combined.DOFs.position.value, dtype=float)
                    collision = np.asarray(sim._instruments_combined.getChild("CollisionModel").CollisionDOFs.position.value, dtype=float)
                    tip = np.asarray(sim.dof_positions[0], dtype=float)
                    force = np.asarray(data_value(solver, "constraintForces"), dtype=float)
                    constraints = data_value(solver, "currentNumConstraints")
                    iterations = data_value(solver, "currentIterations")
                    residual = data_value(solver, "currentError")
                    finite = bool(np.all(np.isfinite(observation)) and np.isfinite(reward) and
                                  np.all(np.isfinite(dofs)) and np.all(np.isfinite(collision)) and
                                  np.all(np.isfinite(force)))
                    spacing = np.linalg.norm(np.diff(dofs[:, :3], axis=0), axis=1)
                    row = {"step": step_index, "action": action,
                           "reward": float(reward), "tip_position_mm": tip.tolist(),
                           "tip_displacement_mm": float(np.linalg.norm(tip-last_tip)),
                           "insertion_length_mm": float(sim.inserted_lengths[0]),
                           "observation": observation.astype(float).tolist(),
                           "terminated": bool(terminated), "truncated": bool(truncated),
                           "success": bool(info.get("is_success", False)),
                           "simulation_error": bool(info.get("simulation_error", False)),
                           "finite": finite, "max_abs_parent_position_mm": float(np.max(np.abs(dofs[:, :3]))),
                           "max_adjacent_parent_dof_spacing_mm": float(np.max(spacing)),
                           "constraint_dimension": int(force.size),
                           "nonzero_constraint_rows": int(np.count_nonzero(np.abs(force) > 1e-12)),
                           "max_abs_constraint_multiplier": float(np.max(np.abs(force))) if force.size else 0.,
                           "constraint_multiplier_l2": float(np.linalg.norm(force)),
                           "currentNumConstraints": int(constraints) if constraints is not None else None,
                           "currentIterations": int(iterations) if iterations is not None else None,
                           "currentError": float(residual) if residual is not None else None}
                    trajectory.append(row)
                    last_tip = tip
                    if not finite or terminated or truncated:
                        break
                if trajectory and (not trajectory[-1]["finite"] or trajectory[-1]["terminated"] or trajectory[-1]["truncated"]):
                    break
            report["scenarios"][scenario] = {"seed": seed, "steps": len(trajectory),
                "terminated": trajectory[-1]["terminated"] if trajectory else False,
                "truncated": trajectory[-1]["truncated"] if trajectory else False,
                "success": trajectory[-1]["success"] if trajectory else False,
                "finite_all_steps": all(row["finite"] for row in trajectory),
                "simulation_error_any": any(row["simulation_error"] for row in trajectory),
                "contact_sampled_steps": sum(row["nonzero_constraint_rows"] > 0 for row in trajectory),
                "max_abs_constraint_multiplier": max((row["max_abs_constraint_multiplier"] for row in trajectory), default=0.),
                "max_solver_iterations": max((row["currentIterations"] or 0 for row in trajectory), default=0),
                "max_solver_residual": max((row["currentError"] or 0. for row in trajectory), default=0.),
                "max_tip_displacement_mm": max((row["tip_displacement_mm"] for row in trajectory), default=0.),
                "max_adjacent_parent_dof_spacing_mm": max((row["max_adjacent_parent_dof_spacing_mm"] for row in trajectory), default=0.),
                "final_insertion_length_mm": trajectory[-1]["insertion_length_mm"] if trajectory else 0.,
                "trajectory": trajectory}
            args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
            print(f"{args.solver} {scenario}: {len(trajectory)} steps, contact samples="
                  f"{report['scenarios'][scenario]['contact_sampled_steps']}, "
                  f"finite={report['scenarios'][scenario]['finite_all_steps']}", flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
