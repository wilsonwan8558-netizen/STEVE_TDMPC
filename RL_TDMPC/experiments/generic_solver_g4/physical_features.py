"""Frozen, diagnostic-only 38-D pre-action Generic SOFA feature vector."""

from __future__ import annotations

import numpy as np

from force_reconstruction import reconstruct_distal_contact_force, _tip_arc_distances

DIMENSION = 38
SOFA_DT_S = 0.006
DISTAL_LENGTH_MM = 3.0


def extract(env) -> tuple[np.ndarray, dict]:
    """Read only the current decision instant, before candidate env.step()."""
    sim = env._simulation
    collision = sim._instruments_combined.getChild("CollisionModel")
    dofs = collision.CollisionDOFs
    positions = np.asarray(dofs.position.value, dtype=np.float64)
    velocity = np.asarray(dofs.velocity.value, dtype=np.float64)
    edges = np.asarray(collision.collisEdgeSet.findData("edges").value, dtype=np.int64)
    if (positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 5 or
            velocity.shape != positions.shape or not np.all(np.isfinite(positions)) or
            not np.all(np.isfinite(velocity))):
        raise RuntimeError("Invalid fixed distal collision position/velocity source")
    arc = _tip_arc_distances(positions, edges)
    if any(i not in arc for i in range(len(positions)-5, len(positions))):
        raise RuntimeError("Last five collision nodes are not in the tip-connected chain")
    solver = sim.root.getObject("Generic")
    multipliers = np.asarray(solver.findData("constraintForces").value, dtype=np.float64).reshape(-1)
    sample = reconstruct_distal_contact_force(
        multipliers, dofs.findData("constraint").value, positions, edges,
        SOFA_DT_S, DISTAL_LENGTH_MM)
    distal = positions[-5:]
    distal_velocity = velocity[-5:]
    tip = distal[-1]
    rotation = float(sim.rotations[0])
    insertion = float(sim.inserted_lengths[0])
    features = np.concatenate((
        tip,
        (distal[:-1] - tip).reshape(-1),
        distal_velocity.reshape(-1),
        sample.vector_N,
        [sample.magnitude_N, float(sample.contact_count), insertion,
         np.sin(rotation), np.cos(rotation)],
    )).astype(np.float32)
    if features.shape != (DIMENSION,) or not np.all(np.isfinite(features)):
        raise RuntimeError("Privileged feature vector shape or finiteness failure")
    audit = {"current_distal_force_final_N": sample.magnitude_N,
             "current_distal_contact_count": sample.contact_count,
             "collision_node_count": len(positions), "collision_edge_count": len(edges)}
    return features, audit
