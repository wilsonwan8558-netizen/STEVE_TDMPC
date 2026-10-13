"""Small live-scene check of the passive Generic 3 mm substep path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
RL_ROOT = HERE.parents[1]
sys.path.insert(0, str(RL_ROOT))
sys.path.insert(0, str(RL_ROOT.parent))
sys.path.insert(0, str(HERE.parents[0] / "generic_solver_g05"))

from envs.steve_env import StEVEEnv  # noqa: E402
from collect_long_horizon import actions_for_program  # noqa: E402
from force_reconstruction import reconstruct_distal_contact_force  # noqa: E402
from substep_monitor import CONTROL_SUBSTEPS, DISTAL_LENGTH_MM, SOFA_DT_S, DistalForceMonitor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if args.steps <= 0:
        raise ValueError("Need positive probe length")
    env = StEVEEnv()
    records = []
    try:
        env.reset(seed=90008)
        sim = env._simulation
        solver = sim.root.getObject("Generic")
        contact = sim.root.getObject("DefaultContactManager")
        scene = {
            "solver": solver.getClassName(),
            "resolution_method": str(solver.findData("resolutionMethod").value),
            "tolerance": float(solver.findData("tolerance").value),
            "max_iterations": int(solver.findData("maxIterations").value),
            "compute_constraint_forces": bool(solver.findData("computeConstraintForces").value),
            "contact_response": str(contact.findData("response").value),
            "friction": str(contact.findData("responseParams").value),
            "dt_s": float(sim.root.dt.value),
        }
        expected = ("GenericConstraintSolver", "ProjectedGaussSeidel", 1e-4,
                    2000, True, "FrictionContactConstraint", "mu=0.001", 0.006)
        if tuple(scene.values()) != expected:
            raise RuntimeError(f"Frozen Generic scene changed: {scene}")
        monitor = DistalForceMonitor(env)
        sim.root.addObject(monitor)
        for index, action in enumerate(actions_for_program("contact_heavy")):
            if index >= args.steps:
                break
            monitor.begin_step()
            env.step(action)
            force = monitor.finish_step()
            collision = sim._instruments_combined.getChild("CollisionModel")
            dofs = collision.CollisionDOFs
            endpoint = reconstruct_distal_contact_force(
                np.asarray(solver.findData("constraintForces").value, dtype=float),
                dofs.findData("constraint").value,
                np.asarray(dofs.position.value, dtype=float),
                np.asarray(collision.collisEdgeSet.findData("edges").value, dtype=int),
                SOFA_DT_S, DISTAL_LENGTH_MM)
            if abs(force.final_N-endpoint.magnitude_N) > 1e-12:
                raise RuntimeError("Monitor final-substep force differs from endpoint snapshot")
            records.append({"step": index+1, "force_max_N": force.maximum_N,
                            "force_final_N": force.final_N,
                            "force_integral_Ns": force.integral_Ns,
                            "distal_contact_count": force.contact_count,
                            "contact_substeps": force.contact_substeps,
                            "substeps": force.substep_count})
        summary = {
            "status": "PASS", "scene": scene, "seed": 90008,
            "program": "G0.5 contact_heavy first actions", "control_steps": len(records),
            "sofa_substeps": monitor.total_substeps,
            "expected_substeps": len(records)*CONTROL_SUBSTEPS,
            "distal_region_mm": DISTAL_LENGTH_MM,
            "contact_control_steps": sum(row["distal_contact_count"] > 0 for row in records),
            "maximum_distal_force_N": max(row["force_max_N"] for row in records),
            "maximum_endpoint_missed_peak_N": max(row["force_max_N"]-row["force_final_N"]
                                                  for row in records),
            "max_endpoint_reconstruction_difference_N": 0.0,
            "records": records,
        }
        if summary["sofa_substeps"] != summary["expected_substeps"]:
            raise RuntimeError("Passive monitor missed a SOFA substep")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
