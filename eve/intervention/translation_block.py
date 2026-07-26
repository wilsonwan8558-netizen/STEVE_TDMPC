"""Canonical translation-intervention metadata shared with RL adapters."""

from __future__ import annotations

from typing import Tuple

import numpy as np


TRANSLATION_BLOCK_REASON_NAMES: Tuple[str, ...] = (
    "none",
    "lower_insertion_boundary",
    "device_length_limit",
    "vessel_tree_end",
    "other",
)
TRANSLATION_MISMATCH_TOLERANCE_MM_S = 1e-9


def translation_block_reason_name(reason_id: int) -> str:
    """Return the canonical name for a validated translation-block reason ID."""

    if isinstance(reason_id, (bool, np.bool_)) or not isinstance(
        reason_id, (int, np.integer)
    ):
        raise TypeError("Translation-block reason ID must be an integer")
    parsed = int(reason_id)
    if not 0 <= parsed < len(TRANSLATION_BLOCK_REASON_NAMES):
        raise ValueError(
            "Translation-block reason ID must be in "
            f"[0, {len(TRANSLATION_BLOCK_REASON_NAMES) - 1}], got {parsed}"
        )
    return TRANSLATION_BLOCK_REASON_NAMES[parsed]

