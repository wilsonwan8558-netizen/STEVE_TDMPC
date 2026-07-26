"""Safety-cost schema and conversion for the stEVE environment adapter."""

from __future__ import annotations

from typing import Any, Mapping, Tuple

import numpy as np


SAFETY_COST_NAMES: Tuple[str, ...] = (
    "collision_association",
    "max_curvature_mm_inv",
    "normalized_command_motion_error",
)


def zero_safety_cost() -> np.ndarray:
    """Return a fresh reset-time safety-cost vector."""

    return np.zeros(len(SAFETY_COST_NAMES), dtype=np.float32)


def safety_cost_from_metrics(
    safety_metrics: Mapping[str, Any],
    translation_speed_limit_mm_s: float,
) -> np.ndarray:
    """Convert monitoring metrics into the fixed, unweighted cost vector."""

    speed_limit = float(translation_speed_limit_mm_s)
    if not np.isfinite(speed_limit) or speed_limit <= 0.0:
        raise ValueError(
            "translation_speed_limit_mm_s must be finite and strictly positive"
        )

    collision_detected = safety_metrics["collision_association_detected"]
    if type(collision_detected) is not bool:
        raise TypeError(
            "collision_association_detected must be a Python bool"
        )
    components = {
        "collision_association": float(collision_detected),
        "max_curvature_mm_inv": float(
            safety_metrics["max_curvature_mm_inv"]
        ),
        "normalized_command_motion_error": float(
            safety_metrics["command_motion_error_mm_s"]
        )
        / speed_limit,
    }
    safety_cost = np.asarray(
        [components[name] for name in SAFETY_COST_NAMES], dtype=np.float32
    )
    if safety_cost.shape != (len(SAFETY_COST_NAMES),):
        raise RuntimeError(
            f"Expected safety cost shape ({len(SAFETY_COST_NAMES)},), "
            f"got {safety_cost.shape}"
        )
    if not np.all(np.isfinite(safety_cost)):
        raise FloatingPointError("Safety cost contains NaN or infinity")
    if np.any(safety_cost < 0.0):
        raise ValueError("Safety cost values must be nonnegative")
    return safety_cost
