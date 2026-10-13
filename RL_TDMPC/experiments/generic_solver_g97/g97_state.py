"""Documented, pre-action Generic-SOFA privileged observable state."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

RL = Path(__file__).resolve().parents[2]
for folder in ("generic_solver_g4", "generic_solver_g2"):
    sys.path.insert(0, str(RL / "experiments" / folder))

from physical_features import extract as extract_g4_physical  # noqa: E402
from force_reconstruction import _parse_jacobian  # noqa: E402

DT = 0.006
PARENT_N = 63
COLLISION_N = 76

# These are all current/pre-action fields. Free state and force fields are the
# last completed substep's exposed SOFA buffers, not a prediction of a_t.
GROUPS = (
    ("exact_G4_physical", 38, "G4 physical_features.extract(env)",
     "Current tip/distal shape, velocity, reconstructed distal contact, insertion/rotation"),
    ("guidewire_rigid_position", PARENT_N * 7, "InstrumentCombined.DOFs.position.value",
     "All rigid generalized coordinates: xyz and quaternion for each guidewire node"),
    ("guidewire_rigid_velocity", PARENT_N * 6, "InstrumentCombined.DOFs.velocity.value",
     "All generalized linear/angular node velocities"),
    ("guidewire_free_position", PARENT_N * 7, "InstrumentCombined.DOFs.free_position.value",
     "Exposed unconstrained-position buffer from the completed pre-action substep"),
    ("guidewire_free_velocity", PARENT_N * 6, "InstrumentCombined.DOFs.free_velocity.value",
     "Exposed unconstrained-velocity buffer from the completed pre-action substep"),
    ("guidewire_force", PARENT_N * 6, "InstrumentCombined.DOFs.force.value",
     "Distributed generalized force buffer at the current completed state"),
    ("collision_position", COLLISION_N * 3, "CollisionModel.CollisionDOFs.position.value",
     "Full mapped collision-node geometry, including shaft and distal tip"),
    ("collision_velocity", COLLISION_N * 3, "CollisionModel.CollisionDOFs.velocity.value",
     "Full mapped collision-node velocities"),
    ("collision_free_position", COLLISION_N * 3, "CollisionModel.CollisionDOFs.free_position.value",
     "Exposed unconstrained mapped collision-node positions"),
    ("contact_node_normal_force", COLLISION_N * 3,
     "Generic.constraintForces + CollisionDOFs.constraint normal rows / dt",
     "Current per-collision-node normal reaction vectors from J^T lambda / dt"),
    ("contact_node_tangent_force", COLLISION_N * 3,
     "Generic.constraintForces + CollisionDOFs.constraint tangent rows / dt",
     "Current per-collision-node tangential reaction vectors from J^T lambda / dt"),
    ("contact_node_indicator", COLLISION_N,
     "Nonzero support of current CollisionDOFs.constraint rows",
     "Which mapped guidewire nodes participate in current contact constraints"),
    ("nearest_vessel_vertex_displacement", COLLISION_N * 3,
     "vesselTree.dofs.position.value KD-tree nearest vertex minus CollisionDOFs.position",
     "Guidewire-to-nearest-static-vessel-vertex vectors; vertex proxy, not exact wall distance"),
    ("nearest_vessel_vertex_distance", COLLISION_N,
     "Norm of nearest-vessel-vertex displacement",
     "Approximate local vessel proximity for every mapped guidewire node"),
    ("solver_summary", 4,
     "Generic.currentNumConstraints/currentNumConstraintGroups/currentIterations/currentError",
     "Current exposed projected-Gauss-Seidel contact/solve summary"),
    ("device_controller_state", 4,
     "m_ircontroller.xtip/rotationInstrument/indexFirstNode",
     "Current insertion, sin/cos rotation, and active first node"),
)
RAW_DIM = sum(group[1] for group in GROUPS)


def definition():
    offset = 0
    features = []
    for name, dim, source, meaning in GROUPS:
        features.append({"name": name, "dimension": dim, "indices": [offset, offset + dim],
                         "source_object_or_field": source, "physical_meaning": meaning,
                         "availability": "simulator-only", "pre_action": True,
                         "future_or_post_action_information": False})
        offset += dim
    assert offset == RAW_DIM
    return {"raw_privileged_extra_dimension": RAW_DIM, "features": features,
            "timing": "at decision t after previous completed RL interval, before selected candidate action a_t",
            "full_Markov_state_claimed": False,
            "name": "privileged observable simulator state",
            "contact_force_units": "nominal N, J_contact^T lambda / 0.006 s",
            "vessel_proximity_caveat": "nearest mesh vertex, not exact triangle-surface signed distance"}


class PrivilegedExtractor:
    def __init__(self, env):
        self.env = env
        vertices = np.asarray(env._simulation._vessel_object.dofs.position.value, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.all(np.isfinite(vertices)):
            raise RuntimeError("G9.7 vessel vertices unavailable")
        self.vertex_count = len(vertices)
        self.vertex_tree = cKDTree(vertices)
        self.vertices = vertices.copy()

    @staticmethod
    def _array(value, shape, name):
        data = np.asarray(value, dtype=np.float64)
        if data.shape != shape or not np.all(np.isfinite(data)):
            raise RuntimeError(f"G9.7 {name} shape/finiteness mismatch: {data.shape} != {shape}")
        return data.copy()

    def extract(self):
        """Return raw feature vector and audit, with no scene mutation or future read."""
        sim = self.env._simulation
        parent = sim._instruments_combined.DOFs
        collision = sim._instruments_combined.getChild("CollisionModel")
        dofs = collision.CollisionDOFs
        solver = sim.root.getObject("Generic")
        controller = sim._instruments_combined.m_ircontroller
        exact_p, physical_audit = extract_g4_physical(self.env)
        if exact_p.shape != (38,):
            raise RuntimeError("G9.7 exact G4 physical feature dimension changed")
        position = self._array(parent.position.value, (PARENT_N, 7), "parent position")
        velocity = self._array(parent.velocity.value, (PARENT_N, 6), "parent velocity")
        free_position = self._array(parent.free_position.value, (PARENT_N, 7), "parent free position")
        free_velocity = self._array(parent.free_velocity.value, (PARENT_N, 6), "parent free velocity")
        force = self._array(parent.force.value, (PARENT_N, 6), "parent force")
        collision_position = self._array(dofs.position.value, (COLLISION_N, 3), "collision position")
        collision_velocity = self._array(dofs.velocity.value, (COLLISION_N, 3), "collision velocity")
        collision_free_position = self._array(dofs.free_position.value, (COLLISION_N, 3), "collision free position")
        multipliers = np.asarray(solver.findData("constraintForces").value, dtype=np.float64).reshape(-1)
        n_rows = int(solver.findData("currentNumConstraints").value)
        n_groups = int(solver.findData("currentNumConstraintGroups").value)
        if (len(multipliers) != n_rows or n_rows != 3 * n_groups or
                not np.all(np.isfinite(multipliers))):
            raise RuntimeError("G9.7 current contact row/multiplier alignment changed")
        rows = _parse_jacobian(dofs.findData("constraint").value)
        if any(r >= n_rows for r in rows):
            raise RuntimeError("G9.7 current collision Jacobian row beyond multipliers")
        normal = np.zeros((COLLISION_N, 3), dtype=np.float64)
        tangent = np.zeros_like(normal)
        active = np.zeros(COLLISION_N, dtype=np.float64)
        for row, columns in rows.items():
            for node, direction in columns.items():
                if node < 0 or node >= COLLISION_N:
                    raise RuntimeError("G9.7 collision Jacobian node out of range")
                if np.linalg.norm(direction) <= 0:
                    continue
                (normal if row % 3 == 0 else tangent)[node] += direction * multipliers[row] / DT
                active[node] = 1.0
        if n_groups < physical_audit["current_distal_contact_count"]:
            raise RuntimeError("G9.7 distal contact count exceeds global groups")
        _, vertex_index = self.vertex_tree.query(collision_position, k=1)
        vessel_displacement = self.vertices[vertex_index] - collision_position
        vessel_distance = np.linalg.norm(vessel_displacement, axis=1)
        solver_summary = np.asarray((n_rows, n_groups,
            int(solver.findData("currentIterations").value),
            float(solver.findData("currentError").value)), dtype=np.float64)
        xtip = np.asarray(controller.xtip.value, dtype=np.float64).reshape(-1)
        rotation = np.asarray(controller.rotationInstrument.value, dtype=np.float64).reshape(-1)
        index_first = int(controller.indexFirstNode.value)
        if xtip.shape != (1,) or rotation.shape != (1,):
            raise RuntimeError("G9.7 one-instrument controller state changed")
        controller_state = np.asarray((xtip[0], np.sin(rotation[0]),
                                       np.cos(rotation[0]), index_first), dtype=np.float64)
        values = (exact_p, position, velocity, free_position, free_velocity,
                  force, collision_position, collision_velocity, collision_free_position,
                  normal, tangent, active, vessel_displacement, vessel_distance,
                  solver_summary, controller_state)
        raw = np.concatenate([np.asarray(value, dtype=np.float64).reshape(-1) for value in values]).astype(np.float32)
        if raw.shape != (RAW_DIM,) or not np.all(np.isfinite(raw)):
            raise RuntimeError("G9.7 raw privileged feature failed shape/finiteness check")
        return raw, {"vessel_vertex_count": self.vertex_count,
                     "active_contact_groups": n_groups,
                     "contact_node_count": int(np.sum(active)),
                     "current_distal_force_N": physical_audit["current_distal_force_final_N"],
                     "physical_first_38": exact_p.tolist()}
