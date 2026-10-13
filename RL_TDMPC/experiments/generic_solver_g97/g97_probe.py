"""Read-only development-scene inventory for prospective privileged-state design."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RL = Path(__file__).resolve().parents[2]
for folder in ("generic_solver_g96", "generic_solver_g95", "generic_solver_g6a",
               "generic_solver_g3", "generic_solver_g2"):
    sys.path.insert(0, str(RL / "experiments" / folder))
sys.path.insert(0, str(RL))
sys.path.insert(0, str(RL.parent))

from g6_collect import setup  # noqa: E402
from g95_collect import archived  # noqa: E402
from restore_audit import signature, compare, g2_signature  # noqa: E402
from collect import check_scene  # noqa: E402
from tdmpc2.common import set_seed  # noqa: E402
from g97_state import PrivilegedExtractor, RAW_DIM  # noqa: E402


def field(obj, name):
    try:
        value = obj.findData(name).value
        array = np.asarray(value)
        if array.dtype.kind in "iufb":
            return {"type": type(value).__name__, "shape": list(array.shape),
                    "finite": bool(np.isfinite(array).all()),
                    "min": float(array.min()) if array.size else None,
                    "max": float(array.max()) if array.size else None}
        return {"type": type(value).__name__, "shape": list(array.shape),
                "prefix": str(value)[:120]}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def snapshot(env, seed, decision):
    sim = env._simulation
    parent = sim._instruments_combined.DOFs
    collision = sim._instruments_combined.getChild("CollisionModel")
    vessel = sim._vessel_object.dofs
    solver = sim.root.getObject("Generic")
    controller = sim._instruments_combined.m_ircontroller
    feature = None
    if decision > 0:
        raw, audit = PrivilegedExtractor(env).extract()
        feature = {"dimension": len(raw), "expected_dimension": RAW_DIM,
                   "minimum": float(np.min(raw)), "maximum": float(np.max(raw)),
                   "audit": {key: value for key, value in audit.items() if key != "physical_first_38"}}
    return {"seed": seed, "decision": decision, "privileged_feature": feature,
            "parent": {key: field(parent, key) for key in
                       ("position", "velocity", "rest_position", "force", "free_position", "free_velocity")},
            "collision": {key: field(collision.CollisionDOFs, key) for key in
                          ("position", "velocity", "force", "free_position", "constraint")},
            "collision_edges": field(collision.collisEdgeSet, "edges"),
            "vessel": {key: field(vessel, key) for key in ("position", "velocity")},
            "solver": {key: field(solver, key) for key in
                       ("constraintForces", "currentNumConstraints", "currentNumConstraintGroups",
                        "currentIterations", "currentError")},
            "controller": {key: field(controller, key) for key in
                           ("xtip", "rotationInstrument", "indexFirstNode", "controlledInstrument")}}


def main():
    seeds = (90000, 90001)
    by_seed, _ = archived()
    env, _, monitor = setup(list(seeds))
    differences = {}
    output = []
    try:
        for seed in seeds:
            set_seed(seed)
            env.action_space.seed(seed)
            observation, _ = env.reset(seed=seed)
            check_scene(env)
            if not hasattr(monitor, "_g97_probe_added"):
                env._simulation.root.addObject(monitor)
                monitor._g97_probe_added = True
            rows = by_seed[seed]
            for d, row in enumerate(rows):
                if d in (0, 5, 10, 20, len(rows)-1):
                    output.append(snapshot(env, seed, d))
                if not np.array_equal(observation,
                    np.asarray(json.loads(row["decision_observation_json"]), dtype=np.float32)):
                    raise RuntimeError(f"probe parent observation changed {seed}/{d}")
                action = np.asarray(json.loads(row["selected_action_json"]), dtype=np.float32)
                parent = signature(env, monitor, action)
                compare(parent, g2_signature(row), differences, f"probe {seed}/{d}")
                observation = np.asarray(parent["observation"], dtype=np.float32)
            print(f"G9.7 probe seed={seed} steps={len(rows)}", flush=True)
    finally:
        env.close()
    print(json.dumps({"snapshots": output, "max_parent_difference": differences},
                     indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
