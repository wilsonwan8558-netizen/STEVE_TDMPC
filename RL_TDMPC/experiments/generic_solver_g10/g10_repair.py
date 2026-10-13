"""Frozen three-replica Q95 force constraint and deterministic continuous repair."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
for name in ("generic_solver_g10", "generic_solver_g95", "generic_solver_g9"):
    sys.path.insert(0, str(RL / "experiments" / name))

from g10_prepare import OUT, G95, SEEDS, F_LIM, GRID_AXIS, REFINE_RADII, gate  # noqa: E402
from g95_eval import load_replica  # noqa: E402
from g9_risk import predict  # noqa: E402


def action_key(action):
    return tuple(np.asarray(action, dtype=np.float32).tolist())


def distance(action, nominal):
    return float(np.linalg.norm(np.asarray(action, dtype=np.float64)-
                                np.asarray(nominal, dtype=np.float64)))


class FrozenRepair:
    def __init__(self):
        gate()
        definition = json.loads((OUT / "frozen_predictor_definition.json").read_text())
        if (definition["family"] != "G9.5_M1_planner_conditioned" or
                definition["training_seeds"] != list(SEEDS) or
                definition["multi_replica_rule"] !=
                "predicted-safe iff all three frozen corrected Q95 values <= F_lim; no best-seed selection"):
            raise RuntimeError("G10 frozen predictor definition changed")
        self.models = [load_replica("M1_planner_conditioned", seed) for seed in SEEDS]
        for seed, (_, _, _, delta) in zip(SEEDS, self.models):
            if delta != definition["replica_validation_only_Q95_delta_N"][str(seed)]:
                raise RuntimeError("G10 frozen replica validation correction changed")
        protocol = json.loads((OUT / "action_search_protocol.json").read_text())
        if (not np.array_equal(np.asarray(protocol["grid_axis"], dtype=np.float32), GRID_AXIS) or
                protocol["refinement_radii"] != list(REFINE_RADII)):
            raise RuntimeError("G10 predeclared action search changed")

    def scores(self, state_294, current_actions, second_action):
        state = np.asarray(state_294, dtype=np.float32)
        current = np.asarray(current_actions, dtype=np.float32).reshape(-1, 2)
        second = np.asarray(second_action, dtype=np.float32)
        if (state.shape != (294,) or second.shape != (2,) or
                np.any(~np.isfinite(state)) or np.any(~np.isfinite(current)) or
                np.any(np.abs(current) > 1+1e-6) or np.any(np.abs(second) > 1+1e-6)):
            raise RuntimeError("G10 predictor state/action invalid")
        n = len(current)
        x = np.c_[np.broadcast_to(state, (n, 294)), current,
                  np.broadcast_to(second, (n, 2))]
        if x.shape != (n, 298):
            raise RuntimeError("G10 frozen 298D predictor input changed")
        corrected = np.stack([predict(net, x, mean, scale)[:, 2] + delta
                              for net, mean, scale, delta in self.models], axis=1)
        if corrected.shape != (n, len(SEEDS)) or not np.all(np.isfinite(corrected)):
            raise RuntimeError("G10 Q95 force prediction invalid")
        return corrected

    def solve(self, state_294, nominal_pair):
        pair = np.asarray(nominal_pair, dtype=np.float32)
        if pair.shape != (2, 2) or np.any(np.abs(pair) > 1+1e-6):
            raise RuntimeError("G10 frozen nominal two-action plan invalid")
        nominal, second = pair
        nominal_scores = self.scores(state_294, nominal[None, :], second)[0]
        evaluations = 1
        if np.all(nominal_scores <= F_LIM):
            return {"status": "NOMINAL_PREDICTED_SAFE_NO_REPAIR",
                "action": nominal.tolist(), "nominal_action": nominal.tolist(),
                "second_action_fixed": second.tolist(), "distance": 0.,
                "delta_translation": 0., "delta_rotation": 0.,
                "nominal_corrected_Q95_N": nominal_scores.tolist(),
                "repaired_corrected_Q95_N": nominal_scores.tolist(),
                "max_corrected_Q95_N": float(np.max(nominal_scores)),
                "predictor_evaluations": evaluations,
                "grid_predicted_safe_count": None, "refinement_predictor_evaluations": 0}
        grid = np.asarray([(t, r) for t in GRID_AXIS for r in GRID_AXIS], dtype=np.float32)
        scores = self.scores(state_294, grid, second)
        evaluations += len(grid)
        safe = np.flatnonzero(np.all(scores <= F_LIM, axis=1))
        if not len(safe):
            return {"status": "INFEASIBLE_ON_PREDECLARED_SEARCH",
                "action": None, "nominal_action": nominal.tolist(),
                "second_action_fixed": second.tolist(), "distance": None,
                "delta_translation": None, "delta_rotation": None,
                "nominal_corrected_Q95_N": nominal_scores.tolist(),
                "repaired_corrected_Q95_N": None,
                "max_corrected_Q95_N": None,
                "predictor_evaluations": evaluations,
                "grid_predicted_safe_count": int(len(safe)),
                "refinement_predictor_evaluations": 0}
        best_idx = min(safe, key=lambda j: (distance(grid[j], nominal),
                                          float(grid[j, 0]), float(grid[j, 1])))
        best = grid[best_idx].copy()
        best_scores = scores[best_idx].copy()
        seen = {action_key(a) for a in grid}
        local_count = 0
        for radius in REFINE_RADII:
            offsets = radius * np.asarray([-1., -.5, 0., .5, 1.], dtype=np.float32)
            local = []
            for dt in offsets:
                for dr in offsets:
                    candidate = np.clip(best + np.asarray([dt, dr], dtype=np.float32), -1, 1).astype(np.float32)
                    key = action_key(candidate)
                    if key not in seen:
                        seen.add(key)
                        local.append(candidate)
            if local:
                local = np.asarray(local, dtype=np.float32)
                local_scores = self.scores(state_294, local, second)
                evaluations += len(local)
                local_count += len(local)
                for j in np.flatnonzero(np.all(local_scores <= F_LIM, axis=1)):
                    candidate = local[j]
                    if (distance(candidate, nominal), float(candidate[0]), float(candidate[1])) < (
                            distance(best, nominal), float(best[0]), float(best[1])):
                        best = candidate.copy()
                        best_scores = local_scores[j].copy()
        deviation = best.astype(np.float64)-nominal.astype(np.float64)
        if np.any(best_scores > F_LIM):
            raise RuntimeError("G10 returned predicted-unsafe repaired action")
        return {"status": "PREDICTED_SAFE_REPAIR", "action": best.tolist(),
            "nominal_action": nominal.tolist(), "second_action_fixed": second.tolist(),
            "distance": float(np.linalg.norm(deviation)),
            "delta_translation": float(deviation[0]), "delta_rotation": float(deviation[1]),
            "nominal_corrected_Q95_N": nominal_scores.tolist(),
            "repaired_corrected_Q95_N": best_scores.tolist(),
            "max_corrected_Q95_N": float(np.max(best_scores)),
            "predictor_evaluations": evaluations,
            "grid_predicted_safe_count": int(len(safe)),
            "refinement_predictor_evaluations": local_count}
