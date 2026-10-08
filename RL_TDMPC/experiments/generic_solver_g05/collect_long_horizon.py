"""Frozen-Generic simulator diagnostics; no controller or physics edits.

The mesh is closed, but its normal winding makes vtkImplicitPolyDataDistance
positive inside. We verify that orientation and negate it, so positive
`signed_exterior_mm` means the mapped wire centerline is outside the vessel.
Only RL-control endpoints are sampled; an independent substep probe may be
used to assess transient extrema.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pyvista as pv
import vtk

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "RL_TDMPC"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generic_solver_g0"))
from envs.steve_env import StEVEEnv  # noqa: E402
from calibrate_wire import jacobian_times_lambda  # noqa: E402

DT = 0.006
CONTROL_SUBSTEPS = int((1.0 / 7.5) / DT)
DISTAL_ARCLENGTH_MM = 20.0
NUMERICAL_DISTANCE_EPS_MM = 1e-6

PROGRAMS = {
    "zero": [([0.0, 0.0], 400)],
    "sustained_moderate_translation": [([0.12, 0.0], 200)],
    "sustained_moderate_rotation": [([0.4, 0.0], 20), ([0.0, 0.4], 300)],
    "translation_plus_rotation": [([0.12, 0.25], 180), ([0.0, 0.25], 120)],
    "alternating_translation": [([0.4, 0.0], 20)] +
        [([0.2 if i % 2 == 0 else -0.2, 0.0], 25) for i in range(12)],
    "alternating_rotation": [([0.4, 0.0], 20)] +
        [([0.0, 0.5 if i % 2 == 0 else -0.5], 25) for i in range(12)],
    "contact_heavy": [([1.0, 0.0], 14), ([0.6, 0.6], 10)] +
        [([0.2, 0.6], 20), ([0.0, -0.6], 20), ([-0.2, 0.6], 20), ([0.0, -0.6], 20)] * 4,
}

STEP_FIELDS = [
    "family", "seed", "step", "simulated_time_s", "action_translation", "action_rotation",
    "terminated", "truncated", "success", "simulation_error", "all_finite",
    "contact_groups", "constraint_rows", "active_multiplier_rows", "solver_iterations",
    "solver_error", "solver_error_scaled_ratio", "max_abs_multiplier",
    "distal_force_N", "global_contact_reaction_N", "force_without_contact",
    "distal_contact_node_count", "contact_groups_with_exterior_nodes",
    "distal_penetration_max_mm", "distal_penetration_mean_mm", "distal_penetration_q95_mm",
    "distal_penetration_mean_positive_mm", "distal_penetration_q95_positive_mm",
    "distal_penetrating_samples", "distal_geometry_samples",
    "global_penetration_max_mm", "global_penetration_mean_mm", "global_penetration_q95_mm",
    "global_penetration_mean_positive_mm", "global_penetration_q95_positive_mm",
    "global_penetrating_samples", "global_geometry_samples",
    "signed_exterior_max_mm", "signed_exterior_min_mm",
    "inserted_length_mm", "tip_x_mm", "tip_y_mm", "tip_z_mm",
    "active_collision_length_mm", "min_active_collision_edge_mm", "max_collision_edge_mm",
    "max_collision_turn_deg", "max_parent_quaternion_norm_error",
    "parent_tip_collision_tip_mismatch_mm", "max_abs_parent_coordinate_mm",
]


def actions_for_program(name: str):
    for action, count in PROGRAMS[name]:
        for _ in range(count):
            yield np.asarray(action, dtype=np.float32)


def actions_for_random_seed(seed: int, count: int = 200):
    """Deterministic bounded-insertion random program, independent of task policy."""
    rng = np.random.default_rng(seed + 730000)
    target_mm = 70.0
    for index in range(count):
        if index % 20 == 0:
            target_mm = float(rng.uniform(40.0, 145.0))
        rotation = float(rng.uniform(-0.5, 0.5))
        # Initial attempt; caller replaces translation using measured insertion.
        yield target_mm, rotation


def random_action(env: StEVEEnv, target_mm: float, rotation: float) -> np.ndarray:
    delta = ((target_mm - float(env._simulation.inserted_lengths[0])) /
             (50.0 * CONTROL_SUBSTEPS * DT))
    return np.asarray([np.clip(delta, -.3, .3), rotation], dtype=np.float32)


def replay_random_seed(env: StEVEEnv, seed: int) -> int:
    """Replay a saved seed normally to restore one-scene history on resume."""
    env.reset(seed=seed)
    for step, (target_mm, rotation) in enumerate(actions_for_random_seed(seed), 1):
        _, _, terminated, truncated, _ = env.step(random_action(env, target_mm, rotation))
        if terminated or truncated:
            return step
    return 200


class GeometryAudit:
    def __init__(self, env: StEVEEnv):
        mesh = pv.read(env.intervention.vessel_tree.mesh_path).triangulate()
        if not mesh.is_manifold or mesh.n_open_edges != 0:
            raise RuntimeError("Vessel mesh is not a closed manifold; signed distance invalid")
        implicit = vtk.vtkImplicitPolyDataDistance()
        implicit.SetInput(mesh)
        centers = np.asarray(env.intervention.vessel_tree.centerline_coordinates)
        center_samples = centers[np.linspace(0, len(centers)-1, 12, dtype=int)]
        far = np.asarray(mesh.center) + 2 * float(mesh.length) * np.ones(3)
        if not all(implicit.EvaluateFunction(x) > 0 for x in center_samples):
            raise RuntimeError("Mesh centerline interior sign changed")
        if implicit.EvaluateFunction(far) >= 0:
            raise RuntimeError("Exterior sign changed")
        self.implicit = implicit

    def signed_exterior(self, points: np.ndarray) -> np.ndarray:
        return -np.fromiter((self.implicit.EvaluateFunction(point) for point in points),
                            dtype=float, count=len(points))


def penetration_stats(signed: np.ndarray) -> dict[str, float | int]:
    positive = np.maximum(signed, 0.0)
    above_noise = positive[positive > NUMERICAL_DISTANCE_EPS_MM]
    return {
        "max_mm": float(np.max(positive)) if len(positive) else 0.0,
        "mean_mm": float(np.mean(positive)) if len(positive) else 0.0,
        "q95_mm": float(np.quantile(positive, .95)) if len(positive) else 0.0,
        "mean_positive_mm": float(np.mean(above_noise)) if len(above_noise) else 0.0,
        "q95_positive_mm": float(np.quantile(above_noise, .95)) if len(above_noise) else 0.0,
        "penetrating_samples": int(len(above_noise)),
        "geometry_samples": int(len(signed)),
    }


def contact_row_nodes(jacobian: str) -> dict[int, list[int]]:
    mapping = {}
    for line in jacobian.splitlines():
        tokens = line.split()
        if not tokens:
            continue
        row = int(tokens[0])
        count = int(tokens[1])
        mapping[row] = [int(tokens[2 + 4*i]) for i in range(count)]
    return mapping


def snapshot(env: StEVEEnv, geometry: GeometryAudit, *, family: str, seed: int,
             step: int, action: np.ndarray, observation: np.ndarray, reward: float,
             terminated: bool, truncated: bool, info: dict) -> dict:
    sim = env._simulation
    solver = sim.root.getObject("Generic")
    collision = sim._instruments_combined.getChild("CollisionModel")
    points = np.asarray(collision.CollisionDOFs.position.value, dtype=float)
    parent = np.asarray(sim._instruments_combined.DOFs.position.value, dtype=float)
    edges = np.asarray(collision.collisEdgeSet.findData("edges").value, dtype=int)
    expected = np.column_stack((np.arange(len(points)-1), np.arange(1, len(points))))
    if not np.array_equal(edges, expected):
        raise RuntimeError("Guidewire collision polyline topology changed")
    edge_vectors = np.diff(points, axis=0)
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    distance_from_tip = np.r_[np.cumsum(edge_lengths[::-1])[::-1], 0.0]
    active = edge_lengths > NUMERICAL_DISTANCE_EPS_MM
    midpoints = .5 * (points[:-1] + points[1:])
    samples = np.vstack((points, midpoints[active]))
    sample_arc_distance = np.r_[distance_from_tip,
                                .5 * (distance_from_tip[:-1] + distance_from_tip[1:])[active]]
    signed = geometry.signed_exterior(samples)
    distal_sample = sample_arc_distance <= DISTAL_ARCLENGTH_MM
    global_pen = penetration_stats(signed)
    distal_pen = penetration_stats(signed[distal_sample])

    multipliers = np.asarray(solver.findData("constraintForces").value, dtype=float)
    n_constraints = int(solver.findData("currentNumConstraints").value)
    n_groups = int(solver.findData("currentNumConstraintGroups").value)
    iterations = int(solver.findData("currentIterations").value)
    error = float(solver.findData("currentError").value)
    if len(multipliers) != n_constraints:
        raise RuntimeError("Generic constraintForces dimension no longer matches constraints")
    jacobian = collision.CollisionDOFs.findData("constraint").value
    nodal_impulse = jacobian_times_lambda(jacobian, multipliers, len(points))
    distal_nodes = distance_from_tip <= DISTAL_ARCLENGTH_MM
    distal_force = float(np.linalg.norm(np.sum(nodal_impulse[distal_nodes], axis=0) / DT))
    global_force = float(np.linalg.norm(np.sum(nodal_impulse, axis=0) / DT))
    row_nodes = contact_row_nodes(jacobian)
    signed_nodes = signed[:len(points)]
    exterior_groups = sum(any(signed_nodes[node] > NUMERICAL_DISTANCE_EPS_MM
                              for node in row_nodes.get(3*group, []))
                          for group in range(n_groups))

    valid_turn_pairs = active[:-1] & active[1:]
    first = edge_vectors[:-1][valid_turn_pairs]
    second = edge_vectors[1:][valid_turn_pairs]
    if len(first):
        cosine = (np.sum(first * second, axis=1) /
                  (np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)))
        max_turn = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))).max())
    else:
        max_turn = 0.0
    quaternion_norm_error = np.abs(np.linalg.norm(parent[:, 3:7], axis=1)-1.0)
    finite = bool(np.all(np.isfinite(observation)) and np.isfinite(reward) and
                  np.all(np.isfinite(points)) and np.all(np.isfinite(parent)) and
                  np.all(np.isfinite(multipliers)) and np.all(np.isfinite(signed)))
    tip = np.asarray(sim.dof_positions[0], dtype=float)
    result = {
        "family": family, "seed": seed, "step": step,
        "simulated_time_s": step * CONTROL_SUBSTEPS * DT,
        "action_translation": float(action[0]), "action_rotation": float(action[1]),
        "terminated": bool(terminated), "truncated": bool(truncated),
        "success": bool(info.get("is_success", False)),
        "simulation_error": bool(info.get("simulation_error", False)), "all_finite": finite,
        "contact_groups": n_groups, "constraint_rows": n_constraints,
        "active_multiplier_rows": int(np.count_nonzero(np.abs(multipliers) > 1e-12)),
        "solver_iterations": iterations, "solver_error": error,
        "solver_error_scaled_ratio": error/(1e-4*max(n_constraints, 1)),
        "max_abs_multiplier": float(np.max(np.abs(multipliers))) if len(multipliers) else 0.0,
        "distal_force_N": distal_force, "global_contact_reaction_N": global_force,
        "force_without_contact": bool(n_groups == 0 and distal_force > 1e-10),
        "distal_contact_node_count": int(sum(np.linalg.norm(nodal_impulse[distal_nodes], axis=1) > 1e-12)),
        "contact_groups_with_exterior_nodes": exterior_groups,
        "signed_exterior_max_mm": float(np.max(signed)),
        "signed_exterior_min_mm": float(np.min(signed)),
        "inserted_length_mm": float(sim.inserted_lengths[0]),
        "tip_x_mm": float(tip[0]), "tip_y_mm": float(tip[1]), "tip_z_mm": float(tip[2]),
        "active_collision_length_mm": float(np.sum(edge_lengths[active])),
        "min_active_collision_edge_mm": float(np.min(edge_lengths[active])) if np.any(active) else 0.0,
        "max_collision_edge_mm": float(np.max(edge_lengths)) if len(edge_lengths) else 0.0,
        "max_collision_turn_deg": max_turn,
        "max_parent_quaternion_norm_error": float(np.max(quaternion_norm_error)),
        "parent_tip_collision_tip_mismatch_mm": float(np.linalg.norm(parent[-1, :3] - points[-1])),
        "max_abs_parent_coordinate_mm": float(np.max(np.abs(parent[:, :3]))),
    }
    for prefix, values in (("distal", distal_pen), ("global", global_pen)):
        result.update({f"{prefix}_penetration_{key}" if key.endswith("mm") else
                       f"{prefix}_{key}": value for key, value in values.items()})
    return result


def write_summary_row(path: Path, summary: dict) -> None:
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary))
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow(summary)
        stream.flush()


def run_one(env: StEVEEnv, geometry: GeometryAudit, *, family: str, seed: int,
            actions, step_writer, step_stream, summary_path: Path, nominal_stop: bool) -> dict:
    observation, _ = env.reset(seed=seed)
    first_done = None
    rows = []
    for step, action_spec in enumerate(actions, start=1):
        if family == "randomized":
            target_mm, rotation = action_spec
            action = random_action(env, target_mm, rotation)
        else:
            action = action_spec
        observation, reward, terminated, truncated, info = env.step(action)
        row = snapshot(env, geometry, family=family, seed=seed, step=step, action=action,
                       observation=observation, reward=reward, terminated=terminated,
                       truncated=truncated, info=info)
        rows.append(row)
        if first_done is None and (terminated or truncated):
            first_done = step
        if not row["all_finite"] or row["simulation_error"]:
            break
        if nominal_stop and (terminated or truncated):
            break
    if not rows:
        raise RuntimeError("No simulator steps executed")
    planned_steps = 200 if nominal_stop else sum(count for _, count in PROGRAMS[family])
    summary = {"family": family, "seed": seed, "steps": len(rows),
               "simulated_time_s": rows[-1]["simulated_time_s"],
               "first_nominal_done_step": first_done or "",
               "completed_program": bool((nominal_stop and first_done is not None) or
                                         (not nominal_stop and len(rows) == planned_steps)),
               "all_finite": all(r["all_finite"] for r in rows),
               "simulation_error_any": any(r["simulation_error"] for r in rows),
               "solver_iteration_cap_hits": sum(r["solver_iterations"] >= 2000 for r in rows),
               "solver_scaled_error_exceedances": sum(r["solver_error_scaled_ratio"] > 1.0 for r in rows),
               "max_distal_force_N": max(r["distal_force_N"] for r in rows),
               "q95_distal_force_N": float(np.quantile([r["distal_force_N"] for r in rows], .95)),
               "max_distal_penetration_mm": max(r["distal_penetration_max_mm"] for r in rows),
               "q95_distal_penetration_mm": float(np.quantile([r["distal_penetration_max_mm"] for r in rows], .95)),
               "max_global_penetration_mm": max(r["global_penetration_max_mm"] for r in rows),
               "q95_global_penetration_mm": float(np.quantile([r["global_penetration_max_mm"] for r in rows], .95)),
               "max_solver_iterations": max(r["solver_iterations"] for r in rows),
               "max_solver_scaled_error_ratio": max(r["solver_error_scaled_ratio"] for r in rows),
               "max_parent_quaternion_norm_error": max(r["max_parent_quaternion_norm_error"] for r in rows),
               "max_tip_mapping_mismatch_mm": max(r["parent_tip_collision_tip_mismatch_mm"] for r in rows)}
    step_writer.writerows(rows)
    step_stream.flush()
    write_summary_row(summary_path, summary)
    print(f"{family} seed={seed}: {len(rows)} steps, max outside="
          f"{summary['max_global_penetration_mm']:.6g} mm, max distal force="
          f"{summary['max_distal_force_N']:.6g} N", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("scripted", "randomized"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--families", nargs="+", choices=list(PROGRAMS), default=list(PROGRAMS),
                        help="Scripted families to run; default is all")
    parser.add_argument("--seed-start", type=int, default=91000,
                        help="First randomized seed in this independent scene shard")
    parser.add_argument("--seed-count", type=int, default=50,
                        help="Number of consecutive randomized seeds in this scene shard")
    parser.add_argument("--resume", action="store_true",
                        help="Replay completed random seeds in order, then record unfinished seeds")
    parser.add_argument("--test-steps", type=int, default=None,
                        help="Short diagnostic of only the first family/seed; never use as final validation")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    step_path = out / ("long_horizon_steps.csv" if args.mode == "scripted" else "randomized_steps.csv")
    summary_path = out / ("long_horizon_results.csv" if args.mode == "scripted" else "randomized_seed_results.csv")
    if args.seed_start < 91000 or args.seed_count <= 0 or args.seed_start + args.seed_count > 91050:
        raise ValueError("Randomized shard must be a nonempty subset of 91000..91049")
    seed_plan = list(range(args.seed_start, args.seed_start + args.seed_count))
    if args.mode == "randomized":
        (out / "randomized_seed_plan.json").write_text(json.dumps({
            "environment_seeds": seed_plan, "full_planned_seed_range": [91000, 91049],
            "action_rng_seed_rule": "environment_seed + 730000",
            "action_program": "Every 20 steps sample insertion target uniformly in [40,145] mm; each step translation drives toward target clipped to [-0.3,0.3], rotation sampled uniformly in [-0.5,0.5].",
            "nominal_episode_steps": 200, "solver_or_policy_training": False,
        }, indent=2) + "\n", encoding="utf-8")
    families = args.families if args.mode == "scripted" else ["randomized"]
    if args.resume and args.mode != "randomized":
        raise ValueError("Resume applies to randomized shards only")
    completed_seeds = []
    if args.resume and summary_path.exists():
        with summary_path.open(newline="", encoding="utf-8") as source:
            summaries = list(csv.DictReader(source))
        completed_seeds = [int(row["seed"]) for row in summaries]
        if completed_seeds != seed_plan[:len(completed_seeds)]:
            raise RuntimeError("Saved randomized seeds are not a contiguous plan prefix")
        if not step_path.exists():
            raise RuntimeError("Summary exists without step trace")
        with step_path.open(newline="", encoding="utf-8") as source:
            previous_steps = list(csv.DictReader(source))
        counts = {seed: sum(int(row["seed"]) == seed for row in previous_steps)
                  for seed in completed_seeds}
        if any(counts[seed] != int(summaries[index]["steps"])
               for index, seed in enumerate(completed_seeds)):
            raise RuntimeError("Saved completed-seed step counts do not match summaries")
        if len(previous_steps) != sum(counts.values()):
            archive = out / "randomized_steps_interrupted.csv"
            if archive.exists():
                raise RuntimeError("Interrupted-step archive already exists; inspect before resuming")
            step_path.rename(archive)
            with step_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=STEP_FIELDS)
                writer.writeheader()
                writer.writerows(row for row in previous_steps if int(row["seed"]) in counts)
    with step_path.open("a" if args.resume and step_path.exists() else "w",
                        newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=STEP_FIELDS)
        if stream.tell() == 0:
            writer.writeheader()
        for family in families:
            seed = 90008 if family == "contact_heavy" else 90000
            if args.mode == "scripted":
                env = StEVEEnv()
                try:
                    env.reset(seed=seed)
                    geometry = GeometryAudit(env)
                    actions = actions_for_program(family)
                    if args.test_steps is not None:
                        actions = (action for _, action in zip(range(args.test_steps), actions))
                    run_one(env, geometry, family=family, seed=seed, actions=actions,
                            step_writer=writer, step_stream=stream,
                            summary_path=summary_path, nominal_stop=False)
                finally:
                    env.close()
            else:
                env = StEVEEnv()
                try:
                    env.reset(seed=seed_plan[0])
                    geometry = GeometryAudit(env)
                    for index, completed_seed in enumerate(completed_seeds):
                        replayed_steps = replay_random_seed(env, completed_seed)
                        if replayed_steps != int(summaries[index]["steps"]):
                            raise RuntimeError(f"Replay changed episode length for seed {completed_seed}")
                        print(f"replayed seed={completed_seed} ({replayed_steps} steps)", flush=True)
                    for random_seed in seed_plan[:1 if args.test_steps is not None else None]:
                        if random_seed in completed_seeds:
                            continue
                        actions = actions_for_random_seed(random_seed)
                        if args.test_steps is not None:
                            actions = (action for _, action in zip(range(args.test_steps), actions))
                        run_one(env, geometry, family=family, seed=random_seed, actions=actions,
                                step_writer=writer, step_stream=stream,
                                summary_path=summary_path, nominal_stop=True)
                        stream.flush()
                finally:
                    env.close()
            stream.flush()
            if args.test_steps is not None:
                break


if __name__ == "__main__":
    main()
