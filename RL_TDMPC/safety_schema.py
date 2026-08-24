"""Backend-independent safety-cost schema definitions and validation.

This module intentionally has no stEVE, SOFA, NVIDIA, Warp, Gymnasium, or
PyTorch dependency.  Environment adapters, replay, models, and checkpoint
validation can therefore agree on one ordered schema without importing either
simulation backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence, Tuple


LEGACY_STEVE_SAFETY_COST_NAMES: Tuple[str, ...] = (
    "filtered_max_curvature_mm_inv",
    "normalized_requested_applied_translation_error",
)
LEGACY_STEVE_PRIMARY_RISK_CHANNEL = (
    "normalized_requested_applied_translation_error"
)
# Compatibility alias used by the staged WorldModel/Agent generalization.
LEGACY_SAFETY_COST_NAMES = LEGACY_STEVE_SAFETY_COST_NAMES

NVIDIA_GUIDED_SAFETY_COST_NAMES: Tuple[str, ...] = (
    "contact_force",
    "pre_penetration",
    "containment_failure",
    "clearance",
    "curvature",
    "tracking",
)
NVIDIA_GUIDED_PRIMARY_RISK_CHANNEL = "tracking"

# This historical dataset/checkpoint layout was explicitly retired before the
# current two-channel stEVE schema.  Continue to reject it rather than treating
# it as an arbitrary new three-channel backend and silently changing meaning.
UNSUPPORTED_LEGACY_THREE_CHANNEL_SAFETY_COST_NAMES: Tuple[str, ...] = (
    "collision_association",
    "max_curvature_mm_inv",
    "normalized_command_motion_error",
)

# Kept here so algorithm/config code need not import ``eve.intervention`` merely
# to validate legacy auxiliary metadata.
TRANSLATION_BLOCK_REASON_NAMES: Tuple[str, ...] = (
    "none",
    "lower_insertion_boundary",
    "device_length_limit",
    "vessel_tree_end",
    "other",
)


def validate_safety_cost_names(
    names: Sequence[str],
    *,
    source: str = "Safety schema",
) -> Tuple[str, ...]:
    """Return a validated, ordered tuple of unique nonempty channel names."""

    if isinstance(names, (str, bytes)) or not isinstance(names, Sequence):
        raise TypeError(f"{source} safety_cost_names must be a sequence of strings")
    received = tuple(names)
    if not received:
        raise ValueError(f"{source} safety_cost_names must not be empty")
    for index, name in enumerate(received):
        if not isinstance(name, str):
            raise TypeError(
                f"{source} safety_cost_names[{index}] must be a string, "
                f"got {type(name).__name__}"
            )
        if not name.strip():
            raise ValueError(
                f"{source} safety_cost_names[{index}] must be nonempty"
            )
    if len(set(received)) != len(received):
        raise ValueError(
            f"{source} safety_cost_names must be unique, got {received}"
        )
    if received == UNSUPPORTED_LEGACY_THREE_CHANNEL_SAFETY_COST_NAMES:
        raise ValueError(
            f"{source} uses the unsupported legacy three-channel safety-cost "
            f"schema {received}; configure an explicitly supported ordered schema"
        )
    return received


def validate_safety_dim(
    safety_dim: int,
    names: Sequence[str],
    *,
    source: str = "Safety schema",
) -> int:
    """Validate an exact integer dimension against the ordered channel names."""

    if isinstance(safety_dim, bool) or not isinstance(safety_dim, Integral):
        raise TypeError(f"{source} safety_dim must be an integer, not bool")
    parsed = int(safety_dim)
    expected = len(tuple(names))
    if parsed <= 0:
        raise ValueError(f"{source} safety_dim must be positive, got {parsed}")
    if parsed != expected:
        raise ValueError(
            f"{source} safety_dim {parsed} does not match "
            f"len(safety_cost_names)={expected}"
        )
    return parsed


@dataclass(frozen=True)
class SafetyCostSchema:
    """Immutable ordered safety-cost interface shared across components."""

    safety_cost_names: Tuple[str, ...]
    safety_dim: int

    @classmethod
    def create(
        cls,
        names: Sequence[str],
        safety_dim: int,
        *,
        source: str = "Safety schema",
    ) -> "SafetyCostSchema":
        validated_names = validate_safety_cost_names(names, source=source)
        validated_dim = validate_safety_dim(
            safety_dim,
            validated_names,
            source=source,
        )
        return cls(validated_names, validated_dim)

    def index(self, channel_name: str, *, source: str = "Safety schema") -> int:
        """Resolve one channel by name, failing clearly when it is absent."""

        if not isinstance(channel_name, str) or not channel_name.strip():
            raise ValueError(f"{source} channel name must be a nonempty string")
        try:
            return self.safety_cost_names.index(channel_name)
        except ValueError as exc:
            raise ValueError(
                f"{source} channel {channel_name!r} is absent from ordered schema "
                f"{self.safety_cost_names}"
            ) from exc


def validate_safety_schema(
    names: Sequence[str],
    safety_dim: int,
    *,
    source: str = "Safety schema",
) -> Tuple[str, ...]:
    """Validate ordered names and their exact dimension, returning the names."""

    return SafetyCostSchema.create(
        names,
        safety_dim,
        source=source,
    ).safety_cost_names


def translation_block_reason_name(reason_id: int) -> str:
    """Resolve the canonical legacy stEVE translation-block reason name."""

    if isinstance(reason_id, bool) or not isinstance(reason_id, Integral):
        raise TypeError("Translation-block reason ID must be an integer")
    parsed = int(reason_id)
    if not 0 <= parsed < len(TRANSLATION_BLOCK_REASON_NAMES):
        raise ValueError(
            "Translation-block reason ID must be in "
            f"[0, {len(TRANSLATION_BLOCK_REASON_NAMES) - 1}], got {parsed}"
        )
    return TRANSLATION_BLOCK_REASON_NAMES[parsed]


__all__ = [
    "LEGACY_STEVE_PRIMARY_RISK_CHANNEL",
    "LEGACY_STEVE_SAFETY_COST_NAMES",
    "LEGACY_SAFETY_COST_NAMES",
    "NVIDIA_GUIDED_PRIMARY_RISK_CHANNEL",
    "NVIDIA_GUIDED_SAFETY_COST_NAMES",
    "SafetyCostSchema",
    "TRANSLATION_BLOCK_REASON_NAMES",
    "UNSUPPORTED_LEGACY_THREE_CHANNEL_SAFETY_COST_NAMES",
    "translation_block_reason_name",
    "validate_safety_cost_names",
    "validate_safety_dim",
    "validate_safety_schema",
]
