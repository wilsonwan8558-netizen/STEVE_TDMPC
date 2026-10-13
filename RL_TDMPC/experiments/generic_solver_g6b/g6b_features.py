"""One frozen pre-action full contact-aware feature vector; no candidate outcomes."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import eve
import numpy as np
import pyvista as pv
import vtk

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generic_solver_g5"))
from g5_data import selected_indices

MESH_CANONICAL_SHA256 = "934640044a63635e2f17bdb2548e6f082b1b2d492ee8231b3533fd82e66ccb9f"
BASE_INDICES = selected_indices(("K", "D"))
EXTRA_NAMES = (
    "current_distal_contact_force_vector_x_N", "current_distal_contact_force_vector_y_N",
    "current_distal_contact_force_vector_z_N", "current_distal_contact_force_magnitude_N",
    "current_distal_contact_indicator", "current_distal_contact_group_count",
    "tip_to_wall_centerline_distance_mm", "tip_signed_inside_wall_distance_mm",
    "minimum_five_distal_node_wall_distance_mm", "mean_five_distal_node_wall_distance_mm",
    "minimum_five_distal_signed_inside_wall_distance_mm",
    "tip_outward_wall_normal_x", "tip_outward_wall_normal_y", "tip_outward_wall_normal_z",
    "tip_velocity_outward_wall_normal_mm_per_s", "tip_tangent_outward_wall_normal_cosine",
    "current_contact_force_outward_wall_normal_N",
)


class FrozenVesselGeometry:
    def __init__(self):
        vessel=eve.intervention.vesseltree.AorticArch(seed=30,scaling_xyzd=[1.,1.,1.,.75])
        vessel.reset()
        mesh=pv.read(vessel.mesh_path).triangulate()
        if not mesh.is_manifold or mesh.n_open_edges!=0 or mesh.n_points!=636 or mesh.n_cells!=1268:
            raise RuntimeError("Frozen closed vessel surface changed")
        canonical=hashlib.sha256(np.ascontiguousarray(mesh.points).tobytes()+np.ascontiguousarray(mesh.faces).tobytes()).hexdigest()
        if canonical!=MESH_CANONICAL_SHA256:
            raise RuntimeError(f"Frozen vessel geometry changed: {canonical}")
        implicit=vtk.vtkImplicitPolyDataDistance()
        implicit.SetInput(mesh)
        centers=np.asarray(vessel.centerline_coordinates,dtype=float)
        probes=centers[np.linspace(0,len(centers)-1,12,dtype=int)]
        far=np.asarray(mesh.center)+2*float(mesh.length)*np.ones(3)
        if not all(float(implicit.EvaluateFunction(point))>0 for point in probes) or float(implicit.EvaluateFunction(far))>=0:
            raise RuntimeError("Frozen vessel signed-distance orientation changed")
        self.mesh=mesh
        self.implicit=implicit
        self.canonical_sha256=canonical

    def extra(self,p):
        """Compute at decision d before a_d, using only exact current G4 P and static wall."""
        p=np.asarray(p,dtype=np.float64)
        if p.shape!=(38,) or not np.all(np.isfinite(p)):
            raise RuntimeError("Exact pre-action G4 P unavailable")
        tip=p[:3]
        distal=np.vstack((tip+p[3:15].reshape(4,3),tip))
        signed=np.asarray([float(self.implicit.EvaluateFunction(x)) for x in distal],dtype=np.float64)
        absolute=np.abs(signed)
        gradient=[0.,0.,0.]
        self.implicit.EvaluateGradient(tip,gradient)
        gradient=np.asarray(gradient,dtype=np.float64)
        length=float(np.linalg.norm(gradient))
        if not np.isfinite(length) or length<1e-8:
            raise RuntimeError("Undefined tip wall normal")
        # G0.5/G6B audit: implicit distance is positive inside; negative gradient points outward.
        outward=-gradient/length
        tangent=tip-distal[-2]
        tangent_length=float(np.linalg.norm(tangent))
        if not np.isfinite(tangent_length) or tangent_length<1e-8:
            raise RuntimeError("Undefined distal tip tangent")
        velocity_tip=p[27:30]
        force_vector=p[30:33]
        contact_count=float(p[34])
        extra=np.r_[force_vector,p[33],float(contact_count>0),contact_count,
                    absolute[-1],signed[-1],np.min(absolute),np.mean(absolute),np.min(signed),
                    outward,np.dot(velocity_tip,outward),np.dot(tangent/tangent_length,outward),
                    np.dot(force_vector,outward)].astype(np.float32)
        if extra.shape!=(len(EXTRA_NAMES),) or not np.all(np.isfinite(extra)):
            raise RuntimeError("Invalid full contact-aware pre-action extras")
        return extra


def full_state(z,p,geometry):
    z=np.asarray(z,dtype=np.float64)
    p=np.asarray(p,dtype=np.float64)
    if z.shape!=(256,) or p.shape!=(38,):
        raise RuntimeError("G6B frozen latent/P shapes changed")
    vector=np.r_[z,p[BASE_INDICES],geometry.extra(p)]
    if vector.shape!=(256+33+len(EXTRA_NAMES),) or not np.all(np.isfinite(vector)):
        raise RuntimeError("Invalid G6B state vector")
    return vector
