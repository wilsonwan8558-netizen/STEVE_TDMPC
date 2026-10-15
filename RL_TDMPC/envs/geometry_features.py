"""Current-state vessel-relative geometry for the G13 observation.

The query uses only the current guidewire tracking geometry and the static
BranchWithRadii lumen model. It never reads contact multipliers, force, or
future simulation state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eve.util.coordtransform import vessel_cs_to_tracking3d, tracking3d_to_2d

TRACKING_SPACING_MM = 1.0
GEOMETRY_SCALE_MM = 20.0
OBSERVATION_DIM = 22
ALIGNMENT_ATOL_MM = 1e-4


@dataclass(frozen=True)
class GeometryResult:
    normalized: np.ndarray
    clearance_mm: np.ndarray
    radius_mm: float
    bend_angle_rad: float
    sin_delta_theta: float
    cos_delta_theta: float
    direction_valid: float
    sampled_tracking_2d_mm: np.ndarray
    sampled_vessel_3d_mm: np.ndarray
    tracking_alignment_error_mm: float
    selected_branch: str
    selected_segment: int


class GeometryFeatureExtractor:
    """Deterministic signed centerline-radius lumen geometry query."""

    def __init__(self) -> None:
        self._branch_identity = None
        self._starts = self._vectors = self._length2 = None
        self._radius_start = self._radius_delta = None
        self._tangent_2d = None
        self._segment_branches: list[str] = []
        self._segment_indices: list[int] = []

    def _cache_vessel(self, env) -> None:
        branches = env.intervention.vessel_tree.branches
        if not branches:
            raise RuntimeError("Vessel branches are not initialized")
        identity = id(branches)
        if identity == self._branch_identity:
            return
        starts, ends, r0, r1, names, indices = [], [], [], [], [], []
        for branch in branches:
            if len(branch.coordinates) < 2 or len(branch.radii) != len(branch.coordinates):
                raise RuntimeError(f"Invalid BranchWithRadii geometry: {branch.name}")
            for i in range(len(branch.coordinates) - 1):
                starts.append(branch.coordinates[i])
                ends.append(branch.coordinates[i + 1])
                r0.append(branch.radii[i])
                r1.append(branch.radii[i + 1])
                names.append(branch.name)
                indices.append(i)
        self._starts = np.asarray(starts, dtype=np.float64)
        ends = np.asarray(ends, dtype=np.float64)
        self._vectors = ends - self._starts
        self._length2 = np.einsum("ij,ij->i", self._vectors, self._vectors)
        if np.any(self._length2 <= 1e-12):
            raise RuntimeError("Zero-length vessel centerline segment")
        self._radius_start = np.asarray(r0, dtype=np.float64)
        self._radius_delta = np.asarray(r1, dtype=np.float64) - self._radius_start
        if not np.all(self._radius_start > 0) or not np.all(self._radius_start + self._radius_delta > 0):
            raise RuntimeError("Nonpositive vessel radius")
        fluoro = env.intervention.fluoroscopy
        projected_starts = tracking3d_to_2d(vessel_cs_to_tracking3d(
            self._starts, fluoro.image_rot_zx, fluoro.image_center,
            fluoro.field_of_view)).astype(np.float64)
        projected_ends = tracking3d_to_2d(vessel_cs_to_tracking3d(
            ends, fluoro.image_rot_zx, fluoro.image_center,
            fluoro.field_of_view)).astype(np.float64)
        if projected_starts.shape != (len(starts), 2) or projected_ends.shape != (len(starts), 2):
            raise RuntimeError("Vessel projection clipped or changed shape")
        self._tangent_2d = projected_ends - projected_starts
        self._segment_branches = names
        self._segment_indices = indices
        self._branch_identity = identity

    @staticmethod
    def _sample_tracking_pair(env) -> tuple[np.ndarray, np.ndarray]:
        """Mirror Tracking2D's 1 mm projected-point selection, paired with 3D."""
        fluoro = env.intervention.fluoroscopy
        points2 = np.asarray(fluoro.tracking2d, dtype=np.float32)
        points3 = np.asarray(fluoro.tracking3d, dtype=np.float32)
        if points2.ndim != 2 or points2.shape[1] != 2 or points3.shape != (len(points2), 3) or not len(points2):
            raise RuntimeError("Current 2D/3D wire tracking arrays are inconsistent")
        sampled2 = [points2[0]]
        sampled3 = [points3[0]]
        acc_dist = 0.0
        for p2, next2, p3, next3 in zip(points2[:-1], points2[1:], points3[:-1], points3[1:]):
            if len(sampled2) >= 3 or np.all(p2 == next2):
                break
            length = np.linalg.norm(next2 - p2)
            dist_to_point = TRACKING_SPACING_MM - acc_dist
            acc_dist += length
            while acc_dist >= TRACKING_SPACING_MM and len(sampled2) < 3:
                fraction = dist_to_point / length
                sampled2.append(p2 + (next2 - p2) * fraction)
                sampled3.append(p3 + (next3 - p3) * fraction)
                acc_dist -= TRACKING_SPACING_MM
        while len(sampled2) < 3:
            sampled2.append(sampled2[-1])
            sampled3.append(sampled3[-1])
        return np.asarray(sampled2, dtype=np.float32), np.asarray(sampled3, dtype=np.float32)

    def _nearest_lumen(self, point: np.ndarray) -> tuple[float, float, int]:
        relative = point.astype(np.float64) - self._starts
        t = np.clip(np.einsum("ij,ij->i", relative, self._vectors) / self._length2, 0, 1)
        closest = self._starts + t[:, None] * self._vectors
        radius = self._radius_start + t * self._radius_delta
        signed_clearance = radius - np.linalg.norm(point - closest, axis=1)
        # max signed clearance approximates the union of the branch tubes;
        # negative means outside every represented lumen tube.
        best = int(np.argmax(signed_clearance))
        return float(signed_clearance[best]), float(radius[best]), best

    def compute(self, env) -> GeometryResult:
        self._cache_vessel(env)
        sample2, sample3 = self._sample_tracking_pair(env)
        observation = env._env.observation.observations["position"].wrapped_obs.obs
        reference2 = np.asarray(observation, dtype=np.float32)[:3]
        error = float(np.max(np.abs(sample2 - reference2)))
        if not np.isfinite(error) or error > ALIGNMENT_ATOL_MM:
            raise RuntimeError(f"G13 distal tracking alignment error {error} mm")
        clearances = []
        radii = []
        segments = []
        for point in sample3:
            clearance, radius, segment = self._nearest_lumen(point)
            clearances.append(clearance)
            radii.append(radius)
            segments.append(segment)
        clearances = np.asarray(clearances, dtype=np.float64)
        tip_segment = segments[0]
        wire = sample2[0].astype(np.float64) - sample2[1].astype(np.float64)
        vessel = self._tangent_2d[tip_segment]
        wire_norm = float(np.linalg.norm(wire))
        vessel_norm = float(np.linalg.norm(vessel))
        valid = float(wire_norm > 1e-6 and vessel_norm > 1e-6)
        if valid:
            wire /= wire_norm
            vessel = vessel / vessel_norm
            sin_delta = float(vessel[0] * wire[1] - vessel[1] * wire[0])
            cos_delta = float(np.clip(np.dot(vessel, wire), -1, 1))
        else:
            # A neutral angle is accompanied by validity=0, so it cannot be
            # confused with a measured aligned configuration.
            sin_delta, cos_delta = 0.0, 1.0
        distal = sample2[0].astype(np.float64) - sample2[1].astype(np.float64)
        proximal = sample2[1].astype(np.float64) - sample2[2].astype(np.float64)
        if np.linalg.norm(distal) > 1e-6 and np.linalg.norm(proximal) > 1e-6:
            bend = float(np.arccos(np.clip(np.dot(distal, proximal) /
                                           (np.linalg.norm(distal) * np.linalg.norm(proximal)), -1, 1)))
        else:
            bend = 0.0
        normalized = np.asarray([
            *np.tanh(clearances / GEOMETRY_SCALE_MM),
            sin_delta, cos_delta, bend / np.pi,
            np.tanh(radii[0] / GEOMETRY_SCALE_MM), valid,
        ], dtype=np.float32)
        if normalized.shape != (8,) or not np.all(np.isfinite(normalized)):
            raise FloatingPointError("Nonfinite or wrong-shaped G13 geometry features")
        return GeometryResult(
            normalized=normalized, clearance_mm=clearances.astype(np.float32),
            radius_mm=float(radii[0]), bend_angle_rad=bend,
            sin_delta_theta=sin_delta, cos_delta_theta=cos_delta,
            direction_valid=valid, sampled_tracking_2d_mm=sample2,
            sampled_vessel_3d_mm=sample3,
            tracking_alignment_error_mm=error,
            selected_branch=self._segment_branches[tip_segment],
            selected_segment=self._segment_indices[tip_segment],
        )
