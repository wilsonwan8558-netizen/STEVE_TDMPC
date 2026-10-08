"""Known SOFA ConstantForceField loads on a distal mapped guidewire collision DOF.

The vessel mesh remains the existing rigid surface. This is an offline
diagnostic scene only; the environment/training code contains no load field.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from envs.steve_env import StEVEEnv  # noqa: E402
import eve  # noqa: E402

NODE = 53
SEED = 90008


def jacobian_times_lambda(text: str, multipliers: np.ndarray, n_nodes: int) -> np.ndarray:
    """Reconstruct guidewire collision-node reaction J_contact^T lambda."""
    result = np.zeros((n_nodes, 3), dtype=float)
    for line in text.splitlines():
        tokens = line.split()
        if not tokens:
            continue
        row, count = int(tokens[0]), int(tokens[1])
        cursor = 2
        for _ in range(count):
            index = int(tokens[cursor])
            result[index] += multipliers[row] * np.asarray(tokens[cursor+1:cursor+4], dtype=float)
            cursor += 4
        if cursor != len(tokens):
            raise RuntimeError("Unexpected Vec3 collision-constraint Jacobian row format")
    return result


class LoadedAdapter(eve.intervention.simulation.SofaBeamAdapter):
    def _load_plugins(self):
        super()._load_plugins()
        self.root.addObject("RequiredPlugin", pluginName="Sofa.Component.MechanicalLoad")

    def _add_devices(self, devices, insertion_point, insertion_direction):
        super()._add_devices(devices, insertion_point, insertion_direction)
        collision = self._instruments_combined.getChild("CollisionModel")
        self.calibration_load = collision.addObject(
            "ConstantForceField", template="Vec3d", name="distal_known_load",
            indices=[NODE], force=[0., 0., 0.])


def collision_reaction(sim):
    solver = sim.root.getObject("Generic")
    collision = sim._instruments_combined.getChild("CollisionModel")
    multipliers = np.asarray(solver.findData("constraintForces").value, dtype=float)
    points = np.asarray(collision.CollisionDOFs.position.value, dtype=float)
    text = collision.CollisionDOFs.findData("constraint").value
    force = jacobian_times_lambda(text, multipliers, len(points))
    return force, multipliers, text, points


def normal_at_node(text, node):
    for line in text.splitlines():
        t = line.split()
        if not t:
            continue
        row, count = int(t[0]), int(t[1])
        at = 2
        for _ in range(count):
            col = int(t[at])
            if col == node and row % 3 == 0:
                vector = np.asarray(t[at+1:at+4], dtype=float)
                return vector/np.linalg.norm(vector)
            at += 4
    raise RuntimeError(f"No contact-normal Jacobian row at mapped guidewire node {node}")


def run(applied: np.ndarray, hold_steps: int):
    env = StEVEEnv()
    samples = []
    try:
        env.reset(seed=SEED)
        sim = env._simulation
        if not isinstance(sim, LoadedAdapter):
            raise RuntimeError("Diagnostic ConstantForceField adapter not installed")
        for _ in range(14):
            env.step(np.asarray([1., 0.], dtype=np.float32))
        baseline_force, multipliers, jacobian, positions = collision_reaction(sim)
        if len(multipliers) == 0:
            raise RuntimeError("No guidewire-vessel contact at selected load state")
        normal = normal_at_node(jacobian, NODE)
        sim.calibration_load.findData("force").value = applied
        for index in range(hold_steps):
            obs, reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
            field, lambda_values, _, positions = collision_reaction(sim)
            samples.append({"hold_step": index+1,
                            "full_contact_reaction_impulse": np.sum(field, axis=0).tolist(),
                            "loaded_node_contact_reaction_impulse": field[NODE].tolist(),
                            "loaded_node_position_mm": positions[NODE].tolist(),
                            "tip_position_mm": np.asarray(sim.dof_positions[0]).tolist(),
                            "constraint_dimension": len(lambda_values),
                            "solver_iterations": int(sim.root.getObject("Generic").findData("currentIterations").value),
                            "solver_error": float(sim.root.getObject("Generic").findData("currentError").value),
                            "observation_finite": bool(np.all(np.isfinite(obs))),
                            "terminated": bool(terminated), "truncated": bool(truncated),
                            "simulation_error": bool(info.get("simulation_error", False))})
            if terminated or truncated:
                break
        mean_all = np.mean([r["full_contact_reaction_impulse"] for r in samples[-3:]], axis=0)
        mean_node = np.mean([r["loaded_node_contact_reaction_impulse"] for r in samples[-3:]], axis=0)
        return {"applied_vector_nominal_N": applied.tolist(),
                "applied_magnitude_nominal_N": float(np.linalg.norm(applied)),
                "node_index": NODE, "node_distance_to_tip_at_load_mm": float(np.linalg.norm(positions[NODE]-positions[-1])),
                "initial_contact_normal_from_J": normal.tolist(),
                "initial_constraint_dimension": len(multipliers),
                "initial_reaction_impulse_all": np.sum(baseline_force, axis=0).tolist(),
                "mean_last3_reaction_impulse_all": mean_all.tolist(),
                "mean_last3_reaction_impulse_node": mean_node.tolist(),
                "samples": samples}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hold-steps", type=int, default=6)
    parser.add_argument("--magnitudes", type=float, nargs="+", default=[.02, .05, .1, .2])
    args = parser.parse_args()
    if args.hold_steps < 3 or any(magnitude <= 0 for magnitude in args.magnitudes):
        raise ValueError("Need at least 3 hold steps and strictly positive force magnitudes")
    original = eve.intervention.simulation.SofaBeamAdapter
    eve.intervention.simulation.SofaBeamAdapter = LoadedAdapter
    try:
        # First obtain a deterministic normal from the same insertion state.
        anchor = run(np.zeros(3), args.hold_steps)
        normal = np.asarray(anchor["initial_contact_normal_from_J"], dtype=float)
        rows = [anchor]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        for magnitude in args.magnitudes:
            row = run(-normal*magnitude, args.hold_steps)
            rows.append(row)
            args.output.write_text(json.dumps({"status": "RUNNING", "rows": rows},
                                             indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"load {magnitude}: constraints {row['samples'][-1]['constraint_dimension']}", flush=True)
        base = np.asarray(anchor["mean_last3_reaction_impulse_all"])
        dt = .006
        summary = []
        for row in rows[1:]:
            applied = np.asarray(row["applied_vector_nominal_N"])
            delta = np.asarray(row["mean_last3_reaction_impulse_all"])-base
            raw = float(np.linalg.norm(delta))
            magnitude = float(np.linalg.norm(applied))
            summary.append({"applied_force_nominal_N": magnitude,
                            "applied_direction": applied.tolist(),
                            "reaction_impulse_delta": delta.tolist(),
                            "reaction_raw_norm": raw,
                            "reaction_divided_by_dt_norm": raw/dt,
                            "raw_relative_error": abs(raw-magnitude)/magnitude,
                            "divided_by_dt_relative_error": abs(raw/dt-magnitude)/magnitude,
                            "direction_cosine_against_opposing_applied": float(np.dot(delta,-applied)/(raw*magnitude)) if raw else None,
                            "final_constraint_dimension": row["samples"][-1]["constraint_dimension"],
                            "node_distance_to_tip_mm": row["node_distance_to_tip_at_load_mm"]})
        args.output.write_text(json.dumps({"status": "COMPLETE", "seed": SEED,
                                          "load_node": NODE, "dt": dt,
                                          "hold_steps": args.hold_steps, "rows": rows,
                                          "comparisons": summary}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        csv_path = args.output.with_suffix(".csv")
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        print("Guidewire controlled-load diagnostic complete", flush=True)
    finally:
        eve.intervention.simulation.SofaBeamAdapter = original


if __name__ == "__main__":
    main()
