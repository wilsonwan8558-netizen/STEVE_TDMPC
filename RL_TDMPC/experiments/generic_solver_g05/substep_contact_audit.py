"""Passive per-SOFA-substep audit of the frozen contact-heavy G0.5 script."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import Sofa

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "RL_TDMPC"))
from envs.steve_env import StEVEEnv  # noqa: E402
from collect_long_horizon import (DT, DISTAL_ARCLENGTH_MM, CONTROL_SUBSTEPS,
                                  GeometryAudit, actions_for_program,
                                  jacobian_times_lambda)  # noqa: E402

FIELDS = ["substep", "control_step", "within_control_substep", "simulated_time_s",
          "distal_force_N", "global_contact_reaction_N", "contact_groups",
          "constraint_rows", "solver_iterations", "solver_error_scaled_ratio",
          "distal_penetration_max_mm", "global_penetration_max_mm",
          "global_signed_exterior_max_mm", "inserted_length_mm", "tip_x_mm",
          "tip_y_mm", "tip_z_mm", "finite"]


class PassiveMonitor(Sofa.Core.Controller):
    def __init__(self, env: StEVEEnv, geometry: GeometryAudit):
        super().__init__(name="g05_passive_contact_audit")
        self.env = env
        self.geometry = geometry
        self.rows = []

    def onAnimateEndEvent(self, _event):
        sim = self.env._simulation
        solver = sim.root.getObject("Generic")
        collision = sim._instruments_combined.getChild("CollisionModel")
        points = np.asarray(collision.CollisionDOFs.position.value, dtype=float)
        edge_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        distance_from_tip = np.r_[np.cumsum(edge_lengths[::-1])[::-1], 0.0]
        active = edge_lengths > 1e-6
        samples = np.vstack((points, .5*(points[:-1]+points[1:])[active]))
        arc = np.r_[distance_from_tip,
                    .5*(distance_from_tip[:-1]+distance_from_tip[1:])[active]]
        signed = self.geometry.signed_exterior(samples)
        global_pen = float(np.maximum(signed, 0.0).max())
        distal_pen = float(np.maximum(signed[arc <= DISTAL_ARCLENGTH_MM], 0.0).max())
        lambdas = np.asarray(solver.findData("constraintForces").value, dtype=float)
        n_constraints = int(solver.findData("currentNumConstraints").value)
        if len(lambdas) != n_constraints:
            raise RuntimeError("Substep multiplier dimension mismatch")
        jacobian = collision.CollisionDOFs.findData("constraint").value
        nodal = jacobian_times_lambda(jacobian, lambdas, len(points))
        distal = np.linalg.norm(np.sum(nodal[distance_from_tip <= DISTAL_ARCLENGTH_MM], axis=0)/DT)
        global_reaction = np.linalg.norm(np.sum(nodal, axis=0)/DT)
        error = float(solver.findData("currentError").value)
        substep = len(self.rows)+1
        # The adapter refreshes its cached dof_positions/inserted_lengths only
        # after all 22 animate calls. Read live SOFA objects inside this event.
        tip = np.asarray(sim._instruments_combined.DOFs.position.value[-1, :3], dtype=float)
        self.rows.append({
            "substep": substep, "control_step": (substep-1)//CONTROL_SUBSTEPS+1,
            "within_control_substep": (substep-1)%CONTROL_SUBSTEPS+1,
            "simulated_time_s": substep*DT, "distal_force_N": float(distal),
            "global_contact_reaction_N": float(global_reaction),
            "contact_groups": int(solver.findData("currentNumConstraintGroups").value),
            "constraint_rows": n_constraints,
            "solver_iterations": int(solver.findData("currentIterations").value),
            "solver_error_scaled_ratio": error/(1e-4*max(n_constraints, 1)),
            "distal_penetration_max_mm": distal_pen,
            "global_penetration_max_mm": global_pen,
            "global_signed_exterior_max_mm": float(np.max(signed)),
            "inserted_length_mm": float(sim._instruments_combined.m_ircontroller.xtip.value[0]),
            "tip_x_mm": float(tip[0]), "tip_y_mm": float(tip[1]),
            "tip_z_mm": float(tip[2]),
            "finite": bool(np.all(np.isfinite(points)) and np.all(np.isfinite(lambdas)) and
                           np.all(np.isfinite(signed)) and np.isfinite(distal)),
        })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    env = StEVEEnv()
    try:
        env.reset(seed=90008)
        geometry = GeometryAudit(env)
        # Match collect_long_horizon.main: first reset constructs the geometry
        # audit; run_one then resets the same scene again before action 1.
        env.reset(seed=90008)
        monitor = PassiveMonitor(env, geometry)
        env._simulation.root.addObject(monitor)
        action_count = 0
        for action in actions_for_program("contact_heavy"):
            env.step(action)
            action_count += 1
            if len(monitor.rows) != action_count*CONTROL_SUBSTEPS:
                raise RuntimeError("Passive monitor missed SOFA substeps")
        reference = out / "shards/scripted_contact_heavy/long_horizon_steps.csv"
        with reference.open(newline="", encoding="utf-8") as stream:
            endpoint_rows = list(csv.DictReader(stream))
        if len(endpoint_rows) != action_count:
            raise RuntimeError("Reference contact-heavy trace has a different duration")
        comparisons = (("distal_force_N", "distal_force_N"),
                       ("global_contact_reaction_N", "global_contact_reaction_N"),
                       ("global_penetration_max_mm", "global_penetration_max_mm"),
                       ("distal_penetration_max_mm", "distal_penetration_max_mm"),
                       ("global_signed_exterior_max_mm", "signed_exterior_max_mm"),
                       ("inserted_length_mm", "inserted_length_mm"),
                       ("tip_x_mm", "tip_x_mm"), ("tip_y_mm", "tip_y_mm"),
                       ("tip_z_mm", "tip_z_mm"),
                       ("solver_iterations", "solver_iterations"),
                       ("contact_groups", "contact_groups"))
        differences = {name: max(abs(float(monitor.rows[(i+1)*CONTROL_SUBSTEPS-1][name])-
                                     float(endpoint_rows[i][other]))
                                 for i in range(action_count))
                       for name, other in comparisons}
        endpoint_force = max(float(row["distal_force_N"]) for row in endpoint_rows)
        all_force = max(float(row["distal_force_N"]) for row in monitor.rows)
        endpoint_crossing = max(float(row["global_penetration_max_mm"]) for row in endpoint_rows)
        all_crossing = max(float(row["global_penetration_max_mm"]) for row in monitor.rows)
        with (out / "substep_contact_heavy.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(monitor.rows)
        summary = {
            "status": "COMPLETE", "seed": 90008, "family": "contact_heavy",
            "control_steps": action_count, "sofa_substeps": len(monitor.rows),
            "sofa_dt_s": DT, "simulated_time_s": len(monitor.rows)*DT,
            "passive_monitor_endpoint_max_abs_differences": differences,
            "passive_monitor_behaviorally_equivalent_at_endpoints": all(value < 1e-8 for value in differences.values()),
            "endpoint_max_distal_force_N": endpoint_force,
            "all_substeps_max_distal_force_N": all_force,
            "endpoint_max_centerline_crossing_mm": endpoint_crossing,
            "all_substeps_max_centerline_crossing_mm": all_crossing,
            "substeps_with_centerline_crossing": sum(row["global_penetration_max_mm"] > 1e-6 for row in monitor.rows),
            "substeps_force_without_contact": sum(row["contact_groups"] == 0 and row["distal_force_N"] > 1e-10 for row in monitor.rows),
            "all_finite": all(row["finite"] for row in monitor.rows),
            "max_solver_iterations": max(row["solver_iterations"] for row in monitor.rows),
            "max_scaled_solver_error_ratio": max(row["solver_error_scaled_ratio"] for row in monitor.rows),
        }
        (out / "substep_contact_heavy_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
