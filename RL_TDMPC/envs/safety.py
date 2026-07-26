"""Safety-cost schema and conversion for the stEVE environment adapter."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

import numpy as np
from eve.intervention import translation_block_reason_name


SAFETY_COST_NAMES: Tuple[str, ...] = (
    "filtered_max_curvature_mm_inv",
    "normalized_requested_applied_translation_error",
)
CURVATURE_STRATUM_NAMES: Tuple[str, ...] = (
    "low",
    "medium",
    "high",
    "extreme",
)
DEFAULT_CURVATURE_BOUNDARIES_MM_INV: Tuple[float, ...] = (
    0.05,
    0.10,
    0.25,
)


def validate_curvature_boundaries(
    boundaries: Sequence[float],
    *,
    source: str = "Curvature boundaries",
) -> Tuple[float, ...]:
    """Validate the three ordered diagnostic boundaries used for sampling."""

    if isinstance(boundaries, (str, bytes)) or not isinstance(
        boundaries, Sequence
    ):
        raise TypeError(f"{source} must be a sequence of real numbers")
    if len(boundaries) != len(CURVATURE_STRATUM_NAMES) - 1:
        raise ValueError(
            f"{source} must contain {len(CURVATURE_STRATUM_NAMES) - 1} "
            f"values, got {len(boundaries)}"
        )
    parsed = []
    for index, value in enumerate(boundaries):
        if isinstance(value, bool):
            raise TypeError(f"{source}[{index}] must be a real number, not bool")
        try:
            converted = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(
                f"{source}[{index}] must be a real number"
            ) from exc
        if not np.isfinite(converted) or converted < 0.0:
            raise ValueError(
                f"{source}[{index}] must be finite and nonnegative"
            )
        parsed.append(converted)
    if any(left >= right for left, right in zip(parsed, parsed[1:])):
        raise ValueError(f"{source} must be strictly increasing")
    return tuple(parsed)


def curvature_stratum_id(
    filtered_max_curvature_mm_inv: float,
    boundaries: Sequence[float] = DEFAULT_CURVATURE_BOUNDARIES_MM_INV,
) -> int:
    """Map one finite, nonnegative curvature to its canonical stratum ID."""

    try:
        curvature = float(filtered_max_curvature_mm_inv)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("Curvature must be a real number") from exc
    if not np.isfinite(curvature) or curvature < 0.0:
        raise ValueError("Curvature must be finite and nonnegative")
    validated_boundaries = validate_curvature_boundaries(boundaries)
    return int(
        np.searchsorted(
            np.asarray(validated_boundaries, dtype=np.float64),
            curvature,
            side="right",
        )
    )


def curvature_stratum_name(stratum_id: int) -> str:
    """Return the canonical name for a validated curvature-stratum ID."""

    if isinstance(stratum_id, (bool, np.bool_)) or not isinstance(
        stratum_id, (int, np.integer)
    ):
        raise TypeError("Curvature-stratum ID must be an integer")
    parsed = int(stratum_id)
    if not 0 <= parsed < len(CURVATURE_STRATUM_NAMES):
        raise ValueError(
            f"Curvature-stratum ID must be in "
            f"[0, {len(CURVATURE_STRATUM_NAMES) - 1}], got {parsed}"
        )
    return CURVATURE_STRATUM_NAMES[parsed]


def safety_aux_metadata_from_metrics(
    safety_metrics: Mapping[str, Any],
    curvature_boundaries_mm_inv: Sequence[
        float
    ] = DEFAULT_CURVATURE_BOUNDARIES_MM_INV,
) -> Tuple[int, int]:
    """Validate reason metadata and derive a curvature-stratum ID."""

    if not isinstance(safety_metrics, Mapping):
        raise TypeError("safety_metrics must be a mapping")
    reason_id_value = safety_metrics["translation_block_reason_id"]
    if isinstance(reason_id_value, (bool, np.bool_)) or not isinstance(
        reason_id_value, (int, np.integer)
    ):
        raise TypeError("translation_block_reason_id must be an integer")
    reason_id = int(reason_id_value)
    reason_name = translation_block_reason_name(reason_id)
    received_reason_name = safety_metrics["translation_block_reason"]
    if not isinstance(received_reason_name, str):
        raise TypeError("translation_block_reason must be a string")
    if received_reason_name != reason_name:
        raise ValueError(
            "Translation-block reason ID/name mismatch: "
            f"{reason_id}/{received_reason_name!r}; expected {reason_name!r}"
        )
    stratum_id = curvature_stratum_id(
        safety_metrics["filtered_max_curvature_mm_inv"],
        curvature_boundaries_mm_inv,
    )
    return reason_id, stratum_id


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

    components = {
        "filtered_max_curvature_mm_inv": float(
            safety_metrics["filtered_max_curvature_mm_inv"]
        ),
        "normalized_requested_applied_translation_error": float(
            safety_metrics["requested_applied_translation_error_mm_s"]
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
