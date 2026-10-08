"""Audit SOFA contact exposure and vessel-mesh sign before G0.5 rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pyvista as pv
import vtk

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from envs.steve_env import StEVEEnv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    env = StEVEEnv()
    try:
        env.reset(seed=90008)
        sim = env._simulation
        solver = sim.root.getObject("Generic")
        contact = sim.root.getObject("DefaultContactManager")
        narrow = sim.root.getObject("BVHNarrowPhase")
        if solver is None or contact is None or narrow is None:
            raise RuntimeError("Expected frozen G0 scene components missing")
        mesh_path = env.intervention.vessel_tree.mesh_path
        mesh = pv.read(mesh_path).triangulate()
        implicit = vtk.vtkImplicitPolyDataDistance()
        implicit.SetInput(mesh)
        centers = np.asarray(env.intervention.vessel_tree.centerline_coordinates)
        center_samples = centers[np.linspace(0, len(centers)-1, 12, dtype=int)]
        far_point = np.asarray(mesh.center) + 2 * np.asarray(mesh.length) * np.ones(3)
        containment = pv.PolyData(np.vstack((center_samples, far_point))).select_enclosed_points(
            mesh, tolerance=0.0, check_surface=True)
        inside_mask = np.asarray(containment["SelectedPoints"], dtype=bool)
        initial_tip = np.asarray(sim.dof_positions[0])
        for _ in range(14):
            env.step(np.asarray([1., 0.], dtype=np.float32))
        collision = sim._instruments_combined.getChild("CollisionModel")
        collision_points = np.asarray(collision.CollisionDOFs.position.value)
        edges = np.asarray(collision.collisEdgeSet.findData("edges").value)
        ordered_edges = np.column_stack((np.arange(len(collision_points)-1),
                                         np.arange(1, len(collision_points))))
        data = {
            "mesh_points": mesh.n_points,
            "mesh_faces": mesh.n_cells,
            "mesh_is_manifold": bool(mesh.is_manifold),
            "mesh_n_open_edges": int(mesh.n_open_edges),
            "mesh_bounds": list(mesh.bounds),
            "centerline_implicit_signed_mm": [float(implicit.EvaluateFunction(x)) for x in center_samples],
            "far_point_implicit_signed_mm": float(implicit.EvaluateFunction(far_point)),
            "centerline_topological_inside": inside_mask[:-1].tolist(),
            "far_point_topological_inside": bool(inside_mask[-1]),
            "initial_tip_implicit_signed_mm": float(implicit.EvaluateFunction(initial_tip)),
            "inserted_tip_implicit_signed_mm": float(implicit.EvaluateFunction(sim.dof_positions[0])),
            "inserted_collision_implicit_signed_mm": [float(implicit.EvaluateFunction(x)) for x in collision_points],
            "collision_edge_shape": list(edges.shape),
            "collision_edge_index_range": [int(np.min(edges)), int(np.max(edges))] if edges.size else None,
            "collision_edges_ordered_chain": bool(np.array_equal(edges, ordered_edges)),
            "localmindistance_contactDistance_mm": float(sim.root.getObject("localmindistance").findData("contactDistance").value),
            "narrow_phase_python_detection_methods": [name for name in dir(narrow) if "detect" in name.lower() or "output" in name.lower() or "contact" in name.lower()],
            "solver": {name: str(solver.findData(name).value) for name in
                       ("resolutionMethod", "tolerance", "maxIterations", "computeConstraintForces")},
            "contact": {name: str(contact.findData(name).value) for name in
                        ("response", "responseParams")},
            "dt_s": float(sim.root.dt.value),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({key: value for key, value in data.items() if key != "inserted_collision_implicit_signed_mm"}, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
