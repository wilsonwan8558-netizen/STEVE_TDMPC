"""Passive 3 mm distal-force monitor at each completed Generic SOFA substep."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import Sofa

from force_reconstruction import reconstruct_distal_contact_force


SOFA_DT_S = 0.006
CONTROL_SUBSTEPS = 22
DISTAL_LENGTH_MM = 3.0


@dataclass(frozen=True)
class StepForce:
    maximum_N: float
    mean_N: float
    final_N: float
    integral_Ns: float
    contact_count: int
    contact_substeps: int
    max_vector_N: tuple[float, float, float]
    substep_count: int
    max_constraint_rows: int
    max_contact_groups: int


class DistalForceMonitor(Sofa.Core.Controller):
    """Read-only callback. ``begin_step``/``finish_step`` bracket one env.step."""

    def __init__(self, env):
        super().__init__(name="g2_passive_distal_force_monitor")
        self.env = env
        self.enabled = False
        self.samples = []
        self.error = None
        self.total_substeps = 0

    def begin_step(self) -> None:
        if self.enabled:
            raise RuntimeError("Previous control step was not finished")
        self.samples = []
        self.error = None
        self.enabled = True

    def onAnimateEndEvent(self, _event) -> None:
        if not self.enabled or self.error is not None:
            return
        try:
            sim = self.env._simulation
            solver = sim.root.getObject("Generic")
            collision = sim._instruments_combined.getChild("CollisionModel")
            multipliers = np.asarray(solver.findData("constraintForces").value,
                                     dtype=float).reshape(-1)
            n_rows = int(solver.findData("currentNumConstraints").value)
            n_groups = int(solver.findData("currentNumConstraintGroups").value)
            if len(multipliers) != n_rows or n_rows != 3*n_groups:
                raise RuntimeError(
                    f"Generic active-contact row mismatch: {len(multipliers)}, "
                    f"{n_rows}, {n_groups}")
            dofs = collision.CollisionDOFs
            points = np.asarray(dofs.position.value, dtype=float)
            edges = np.asarray(collision.collisEdgeSet.findData("edges").value,
                               dtype=int)
            sample = reconstruct_distal_contact_force(
                multipliers, dofs.findData("constraint").value,
                points, edges, SOFA_DT_S, DISTAL_LENGTH_MM)
            if not np.isfinite(sample.magnitude_N) or sample.magnitude_N < 0:
                raise RuntimeError("Nonfinite or negative distal force")
            self.samples.append((sample, n_rows, n_groups))
            self.total_substeps += 1
        except Exception as exc:  # SOFA may log callback errors without raising.
            self.error = exc

    def finish_step(self) -> StepForce:
        self.enabled = False
        if self.error is not None:
            raise RuntimeError("Passive substep force reconstruction failed") from self.error
        if len(self.samples) != CONTROL_SUBSTEPS:
            raise RuntimeError(
                f"Expected {CONTROL_SUBSTEPS} SOFA substeps, got {len(self.samples)}")
        force = np.asarray([row[0].magnitude_N for row in self.samples], dtype=float)
        if not np.all(np.isfinite(force)):
            raise RuntimeError("Nonfinite per-substep force")
        argmax = int(np.argmax(force))
        max_sample = self.samples[argmax][0]
        return StepForce(
            maximum_N=float(np.max(force)), mean_N=float(np.mean(force)),
            final_N=float(force[-1]), integral_Ns=float(np.sum(force)*SOFA_DT_S),
            contact_count=max(row[0].contact_count for row in self.samples),
            contact_substeps=sum(row[0].contact_count > 0 for row in self.samples),
            max_vector_N=tuple(float(x) for x in max_sample.vector_N),
            substep_count=len(self.samples),
            max_constraint_rows=max(row[1] for row in self.samples),
            max_contact_groups=max(row[2] for row in self.samples),
        )
