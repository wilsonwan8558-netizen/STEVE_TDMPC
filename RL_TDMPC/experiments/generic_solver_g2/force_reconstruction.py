"""Generic-solver distal guidewire contact reaction in nominal simulation N.

The distal contact-location mapping is the prior validated 3 mm definition,
ported without its LCP result data. Generic's current ``constraintForces``
and mapped guidewire collision Jacobian are read from the *same* SOFA substep.
G0 controlled loading established that J_contact^T lambda / dt has the scene's
nominal N scale. Multipliers alone are not forces.
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np


DISTAL_LENGTH_MM = 3.0
_ZERO_TOL = 1e-12
_WEIGHT_TOL = 1e-6


@dataclass(frozen=True)
class DistalContactForceSample:
    """One Generic substep's distal resultant; N is nominal simulation N."""

    vector_N: np.ndarray
    magnitude_N: float
    contact_count: int


def _parse_jacobian(value: str) -> Dict[int, Dict[int, np.ndarray]]:
    """Parse SOFA SparseMatrixDeriv rows: row, count, (node, x, y, z)*."""

    rows: Dict[int, Dict[int, np.ndarray]] = {}
    for line in value.splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 2:
            raise ValueError(f"Malformed collision constraint row: {line!r}")
        row, count = int(parts[0]), int(parts[1])
        if row < 0 or count < 0 or row in rows or len(parts) != 2 + 4 * count:
            raise ValueError(f"Malformed collision constraint row: {line!r}")
        columns: Dict[int, np.ndarray] = {}
        for offset in range(2, len(parts), 4):
            node = int(parts[offset])
            if node < 0 or node in columns:
                raise ValueError(f"Malformed collision constraint column: {line!r}")
            direction = np.asarray(parts[offset + 1 : offset + 4], dtype=float)
            if not np.all(np.isfinite(direction)):
                raise ValueError(f"Non-finite collision constraint row: {line!r}")
            columns[node] = direction
        rows[row] = columns
    return rows


def _tip_arc_distances(positions: np.ndarray, edges: np.ndarray) -> Dict[int, float]:
    """Walk the active collision-edge chain outward from its highest-index tip."""

    n_nodes = len(positions)
    if n_nodes == 0:
        raise ValueError("No guidewire collision nodes")
    tip = n_nodes - 1
    if n_nodes == 1:
        return {tip: 0.0}
    if edges.size == 0:
        raise ValueError("No active guidewire collision edges")
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("Collision edges must have shape (n, 2)")

    neighbors: Dict[int, set[int]] = {}
    for start, end in edges:
        a, b = int(start), int(end)
        if a < 0 or b < 0 or a >= n_nodes or b >= n_nodes or a == b:
            raise ValueError("Invalid guidewire collision edge")
        neighbors.setdefault(a, set()).add(b)
        neighbors.setdefault(b, set()).add(a)
    if len(neighbors.get(tip, ())) != 1:
        raise ValueError("Distal collision node is not a chain endpoint")

    distances = {tip: 0.0}
    previous = None
    current = tip
    while True:
        adjacent = neighbors[current] - ({previous} if previous is not None else set())
        if not adjacent:
            break
        if len(adjacent) != 1:
            raise ValueError("Guidewire collision edges branch")
        next_node = adjacent.pop()
        if next_node in distances:
            raise ValueError("Guidewire collision edges contain a cycle")
        segment_mm = float(np.linalg.norm(positions[next_node] - positions[current]))
        if not np.isfinite(segment_mm):
            raise ValueError("Non-finite guidewire collision edge")
        # SOFA retains collapsed, uninserted proximal edges in this topology.
        # They contribute zero *spatial* arc length but do not invalidate a
        # contact elsewhere on the inserted guidewire.
        distances[next_node] = distances[current] + segment_mm
        previous, current = current, next_node
    return distances


def _contact_arc_length(
    normal_row: Dict[int, np.ndarray],
    distances: Dict[int, float],
    active_edges: Dict[frozenset[int], float],
) -> tuple[float, set[int]]:
    """Locate a point or line contact from its current normal-row support."""

    nonzero = {
        node: direction
        for node, direction in normal_row.items()
        if np.linalg.norm(direction) > _ZERO_TOL
    }
    if not nonzero or len(nonzero) > 2:
        raise ValueError("Contact normal must have one or two collision nodes")
    if any(node not in distances for node in nonzero):
        raise ValueError("Contact node is outside the tip-connected edge chain")
    if len(nonzero) == 1:
        node = next(iter(nonzero))
        return distances[node], {node}

    nodes = set(nonzero)
    edge = frozenset(nodes)
    if edge not in active_edges:
        raise ValueError("Line-contact support is not a current collision edge")
    if active_edges[edge] <= _ZERO_TOL:
        raise ValueError("Active line contact lies on a degenerate collision edge")
    combined = sum(nonzero.values(), np.zeros(3))
    denominator = float(np.dot(combined, combined))
    if denominator <= _ZERO_TOL**2:
        raise ValueError("Degenerate line-contact normal")
    weights = {
        node: float(np.dot(direction, combined) / denominator)
        for node, direction in nonzero.items()
    }
    if not np.isclose(sum(weights.values()), 1.0, atol=_WEIGHT_TOL):
        raise ValueError("Line-contact interpolation weights do not sum to one")
    for node, direction in nonzero.items():
        weight = weights[node]
        if weight < -_WEIGHT_TOL or weight > 1.0 + _WEIGHT_TOL:
            raise ValueError("Line-contact interpolation is outside its edge")
        if np.linalg.norm(direction - weight * combined) > _WEIGHT_TOL * np.linalg.norm(combined):
            raise ValueError("Line-contact Jacobian vectors are not collinear")
    arc_mm = sum(weights[node] * distances[node] for node in nodes)
    return float(arc_mm), nodes


def reconstruct_distal_contact_force(
    multipliers: np.ndarray,
    jacobian_text: str,
    positions: np.ndarray,
    edges: np.ndarray,
    dt: float,
    distal_length_mm: float = DISTAL_LENGTH_MM,
) -> DistalContactForceSample:
    """Return the distal vector sum of active Generic normal/tangent rows.

    For a line contact crossing the 3 mm boundary, the normal-row barycentric
    weights locate its point along the current collision edge. Its *entire*
    three-row contribution is included iff that location is at most 3 mm from
    the distal tip; equality is included. No scalar multiplier magnitudes are
    summed. ``dt`` is the SOFA substep, not the outer RL interval.
    """

    dt = float(dt)
    distal_length_mm = float(distal_length_mm)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("SOFA substep dt must be positive and finite")
    if not np.isfinite(distal_length_mm) or distal_length_mm <= 0.0:
        raise ValueError("Distal region length must be positive and finite")

    lam = np.asarray(multipliers, dtype=float).reshape(-1).copy()
    if not np.all(np.isfinite(lam)) or len(lam) % 3:
        raise ValueError("Generic multipliers must be finite frictional row triplets")
    rows = _parse_jacobian(jacobian_text)
    if any(row >= len(lam) for row in rows):
        raise ValueError("Collision Jacobian row exceeds multiplier vector")
    if len(lam) == 0:
        if rows:
            raise ValueError("Collision Jacobian has rows without multipliers")
        return DistalContactForceSample(np.zeros(3), 0.0, 0)

    coords = np.asarray(positions, dtype=float)
    topology_edges = np.asarray(edges, dtype=int)
    if coords.ndim != 2 or coords.shape[1] != 3 or not np.all(np.isfinite(coords)):
        raise ValueError("Collision positions must be finite xyz coordinates")
    distances = _tip_arc_distances(coords, topology_edges)
    active_edges = {
        frozenset((int(a), int(b))): float(np.linalg.norm(coords[int(a)] - coords[int(b)]))
        for a, b in topology_edges
    }

    resultant = np.zeros(3)
    contact_count = 0
    for normal_row_id in range(0, len(lam), 3):
        normal_lambda = float(lam[normal_row_id])
        if normal_lambda < -_ZERO_TOL:
            raise ValueError("Negative unilateral normal multiplier")
        if normal_lambda <= _ZERO_TOL:
            if np.any(np.abs(lam[normal_row_id + 1 : normal_row_id + 3]) > _ZERO_TOL):
                raise ValueError("Tangential multiplier without active normal contact")
            continue
        normal_row = rows.get(normal_row_id)
        if normal_row is None:
            raise ValueError("Active contact has no guidewire collision Jacobian row")
        arc_mm, support = _contact_arc_length(normal_row, distances, active_edges)
        if arc_mm > distal_length_mm:
            continue
        for row_id in range(normal_row_id, normal_row_id + 3):
            row = rows.get(row_id, {})
            nonzero_support = {
                node for node, direction in row.items()
                if np.linalg.norm(direction) > _ZERO_TOL
            }
            if not nonzero_support.issubset(support):
                raise ValueError("Normal and tangent rows have different collision support")
            if not nonzero_support and abs(lam[row_id]) > _ZERO_TOL:
                raise ValueError("Nonzero contact multiplier has no Jacobian row")
            for direction in row.values():
                resultant += direction * float(lam[row_id]) / dt
        contact_count += 1
    magnitude = float(np.linalg.norm(resultant))
    return DistalContactForceSample(resultant, magnitude, contact_count)
