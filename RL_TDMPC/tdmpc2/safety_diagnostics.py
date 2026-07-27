"""Diagnostics for the two-channel latent safety prediction head.

The curvature ranges in this module are diagnostic buckets only.  They are
chosen to make errors at different numerical scales visible and must not be
interpreted as clinical safety thresholds.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from envs.safety import CURVATURE_STRATUM_NAMES
from eve.intervention import TRANSLATION_BLOCK_REASON_NAMES

from .common import (
    DEFAULT_SAFETY_AUX_VALIDATION_TRANSLATION_THRESHOLD_CANDIDATES,
)
from .replay_buffer import REPLAY_SAFETY_COST_NAMES


CURVATURE_DIAGNOSTIC_BOUNDARIES_MM_INV: Tuple[float, float] = (0.05, 0.1)
DEFAULT_TRANSLATION_POSITIVE_THRESHOLD = 1e-6
DEFAULT_FIXED_VALIDATION_BATCH_SIZE = 512
DEFAULT_TRANSLATION_THRESHOLD_CANDIDATES: Tuple[float, ...] = tuple(
    DEFAULT_SAFETY_AUX_VALIDATION_TRANSLATION_THRESHOLD_CANDIDATES
)

_CURVATURE_INDEX = 0
_TRANSLATION_ERROR_INDEX = 1
_CHANNEL_LABELS = ("curvature", "translation_error")


def _normalise_prefix(prefix: str) -> str:
    if not isinstance(prefix, str):
        raise TypeError(f"prefix must be a string, got {type(prefix).__name__}")
    if prefix and not prefix.endswith("_"):
        return f"{prefix}_"
    return prefix


def _validate_channel_order(
    model: Any, safety_cost_names: Sequence[str]
) -> None:
    try:
        requested_names = tuple(str(name) for name in safety_cost_names)
    except TypeError as exc:
        raise TypeError("safety_cost_names must be an iterable of strings") from exc
    if requested_names != REPLAY_SAFETY_COST_NAMES:
        raise ValueError(
            "Safety diagnostics require channels "
            f"{REPLAY_SAFETY_COST_NAMES} in this exact order, got "
            f"{requested_names}"
        )

    if not hasattr(model, "safety_cost_names"):
        raise TypeError("model is missing safety_cost_names metadata")
    model_names = tuple(str(name) for name in model.safety_cost_names)
    if model_names != REPLAY_SAFETY_COST_NAMES:
        raise ValueError(
            "WorldModel safety channels do not match the required order: "
            f"expected {REPLAY_SAFETY_COST_NAMES}, got {model_names}"
        )
    if int(getattr(model, "safety_dim", -1)) != len(
        REPLAY_SAFETY_COST_NAMES
    ):
        raise ValueError(
            "WorldModel safety_dim must be "
            f"{len(REPLAY_SAFETY_COST_NAMES)}, got "
            f"{getattr(model, 'safety_dim', None)}"
        )
    for method_name in (
        "transform_safety_targets",
        "decode_safety_transformed",
    ):
        if not callable(getattr(model, method_name, None)):
            raise TypeError(f"model is missing callable {method_name}()")


def _as_floating_tensor(value: Any, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
    else:
        try:
            tensor = torch.as_tensor(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be convertible to a tensor") from exc
    if not tensor.is_floating_point():
        raise TypeError(
            f"{name} must use a floating-point dtype, got {tensor.dtype}"
        )
    return tensor


def _prepare_inputs(
    prediction_transformed: Any,
    target: Any,
    model: Any,
    safety_cost_names: Sequence[str],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_channel_order(model, safety_cost_names)
    prediction_transformed = _as_floating_tensor(
        prediction_transformed, "prediction_transformed"
    )
    target = _as_floating_tensor(target, "target")

    expected_dim = len(REPLAY_SAFETY_COST_NAMES)
    if (
        prediction_transformed.ndim == 0
        or prediction_transformed.shape[-1] != expected_dim
    ):
        raise ValueError(
            "prediction_transformed must have final dimension "
            f"{expected_dim}, got shape {tuple(prediction_transformed.shape)}"
        )
    if target.shape != prediction_transformed.shape:
        raise ValueError(
            "target and prediction_transformed must have identical shapes, "
            f"got {tuple(target.shape)} and "
            f"{tuple(prediction_transformed.shape)}"
        )
    if prediction_transformed.numel() == 0:
        raise ValueError("Safety diagnostic inputs must contain at least one sample")
    if not bool(torch.isfinite(prediction_transformed).all()):
        raise FloatingPointError(
            "prediction_transformed contains NaN or infinity"
        )
    if not bool(torch.isfinite(target).all()):
        raise FloatingPointError("target contains NaN or infinity")
    if bool(torch.any(target < 0)):
        raise ValueError("target must be nonnegative in both safety channels")

    target = target.to(
        device=prediction_transformed.device,
        dtype=prediction_transformed.dtype,
    )
    if not bool(torch.isfinite(target).all()):
        raise FloatingPointError(
            "target became non-finite when converted to prediction dtype/device"
        )

    with torch.no_grad():
        target_transformed = model.transform_safety_targets(target)
        prediction = model.decode_safety_transformed(prediction_transformed)
    for value, name in (
        (target_transformed, "transformed target"),
        (prediction, "decoded safety prediction"),
    ):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"WorldModel returned a non-tensor {name}")
        if value.shape != target.shape:
            raise ValueError(
                f"{name} must have shape {tuple(target.shape)}, "
                f"got {tuple(value.shape)}"
            )
        if not value.is_floating_point():
            raise TypeError(f"{name} must use a floating-point dtype")
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"{name} contains NaN or infinity")
    if bool(torch.any(target_transformed < 0)):
        raise ValueError("WorldModel produced a negative transformed target")
    if bool(torch.any(prediction < 0)):
        raise ValueError("WorldModel produced a negative decoded prediction")

    # Float64 reductions avoid float32 overflow in squared-error and moment
    # diagnostics. The model transformations above still use the model input
    # dtype, so the reported decoded values match normal inference semantics.
    return tuple(
        value.reshape(-1, expected_dim).to(dtype=torch.float64)
        for value in (
            prediction_transformed,
            target_transformed,
            prediction,
            target,
        )
    )


def _nan(reference: torch.Tensor) -> torch.Tensor:
    return reference.new_tensor(float("nan"))


def _stable_mean(value: torch.Tensor) -> torch.Tensor:
    if value.numel() == 0:
        return _nan(value)
    scale = torch.amax(torch.abs(value))
    if bool(scale == 0):
        return torch.zeros((), dtype=value.dtype, device=value.device)
    result = torch.mean(value / scale) * scale
    return torch.nan_to_num(
        result,
        nan=float("nan"),
        posinf=torch.finfo(value.dtype).max,
        neginf=-torch.finfo(value.dtype).max,
    )


def _stable_std(value: torch.Tensor) -> torch.Tensor:
    """Return population standard deviation without overflowing finite input."""

    if value.numel() == 0:
        return _nan(value)
    scale = torch.amax(torch.abs(value))
    if bool(scale == 0):
        return torch.zeros((), dtype=value.dtype, device=value.device)
    normalised = value / scale
    centred = normalised - torch.mean(normalised)
    result = torch.sqrt(torch.mean(centred.square())) * scale
    return torch.nan_to_num(
        result,
        nan=float("nan"),
        posinf=torch.finfo(value.dtype).max,
        neginf=0.0,
    )


def _finite_absolute_error(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    finfo = torch.finfo(prediction.dtype)
    return torch.nan_to_num(
        torch.abs(prediction - target),
        nan=finfo.max,
        posinf=finfo.max,
        neginf=finfo.max,
    )


def _stable_mae(error: torch.Tensor) -> torch.Tensor:
    return _stable_mean(error)


def _stable_rmse(error: torch.Tensor) -> torch.Tensor:
    if error.numel() == 0:
        return _nan(error)
    scale = torch.amax(torch.abs(error))
    if bool(scale == 0):
        return torch.zeros((), dtype=error.dtype, device=error.device)
    result = torch.sqrt(torch.mean((error / scale).square())) * scale
    return torch.nan_to_num(
        result,
        nan=float("nan"),
        posinf=torch.finfo(error.dtype).max,
        neginf=0.0,
    )


def _pearson_correlation(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Return Pearson correlation, or NaN when it is not defined."""

    if prediction.numel() < 2:
        return _nan(prediction)
    prediction_scale = torch.amax(torch.abs(prediction))
    target_scale = torch.amax(torch.abs(target))
    if bool(prediction_scale == 0) or bool(target_scale == 0):
        return _nan(prediction)

    prediction_normalised = prediction / prediction_scale
    target_normalised = target / target_scale
    prediction_centred = (
        prediction_normalised - torch.mean(prediction_normalised)
    )
    target_centred = target_normalised - torch.mean(target_normalised)
    prediction_energy = torch.sum(prediction_centred.square())
    target_energy = torch.sum(target_centred.square())
    if (
        not bool(torch.isfinite(prediction_energy))
        or not bool(torch.isfinite(target_energy))
        or bool(prediction_energy <= 0)
        or bool(target_energy <= 0)
    ):
        return _nan(prediction)

    denominator = torch.sqrt(prediction_energy * target_energy)
    correlation = torch.sum(
        prediction_centred * target_centred
    ) / denominator
    if not bool(torch.isfinite(correlation)):
        return _nan(prediction)
    return torch.clamp(correlation, min=-1.0, max=1.0)


def _finite_ratio(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
) -> torch.Tensor:
    """Return a finite scalar ratio, or NaN when the denominator is zero."""

    if bool(denominator == 0):
        return _nan(numerator)
    finfo = torch.finfo(numerator.dtype)
    return torch.nan_to_num(
        numerator / denominator,
        nan=float("nan"),
        posinf=finfo.max,
        neginf=-finfo.max,
    )


def _validated_group_ids(
    value: Any,
    *,
    name: str,
    sample_count: int,
    group_count: int,
    device: torch.device,
) -> torch.Tensor:
    """Validate one stored categorical label vector without re-bucketing it."""

    try:
        tensor = (
            value.detach()
            if isinstance(value, torch.Tensor)
            else torch.as_tensor(value)
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be convertible to a tensor") from exc
    if tensor.ndim != 1 or int(tensor.numel()) != sample_count:
        raise ValueError(
            f"{name} must have shape ({sample_count},), got "
            f"{tuple(tensor.shape)}"
        )
    integer_dtypes = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
    if tensor.dtype not in integer_dtypes:
        raise TypeError(f"{name} must use an integer dtype, got {tensor.dtype}")
    tensor = tensor.to(device=device, dtype=torch.int64)
    if bool(torch.any(tensor < 0)) or bool(torch.any(tensor >= group_count)):
        raise ValueError(
            f"{name} values must be in [0, {group_count - 1}]"
        )
    return tensor


def _validated_group_names(
    value: Sequence[str],
    *,
    name: str,
) -> Tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of names")
    try:
        names = tuple(str(item) for item in value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of names") from exc
    if not names or len(set(names)) != len(names):
        raise ValueError(f"{name} must contain unique group names")
    return names


def _fixed_validation_group_metrics(
    *,
    mask: torch.Tensor,
    prediction_transformed: torch.Tensor,
    target_transformed: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, Any]:
    """Compute complete two-channel metrics for one fixed stored group."""

    count = int(torch.count_nonzero(mask))
    channels: Dict[str, Dict[str, torch.Tensor]] = {}
    for channel_index, channel_label in enumerate(_CHANNEL_LABELS):
        if count == 0:
            unavailable = _nan(prediction)
            channels[channel_label] = {
                "mae": unavailable.clone(),
                "mae_transformed": unavailable.clone(),
                "prediction_mean": unavailable.clone(),
                "target_mean": unavailable.clone(),
                "prediction_max": unavailable.clone(),
                "target_max": unavailable.clone(),
                "pearson": unavailable.clone(),
                "prediction_max_to_target_max_ratio": unavailable.clone(),
                "target_max_to_prediction_max_ratio": unavailable.clone(),
            }
            continue

        channel_prediction = prediction[mask, channel_index]
        channel_target = target[mask, channel_index]
        channel_prediction_transformed = prediction_transformed[
            mask, channel_index
        ]
        channel_target_transformed = target_transformed[mask, channel_index]
        prediction_max = torch.amax(channel_prediction)
        target_max = torch.amax(channel_target)
        channels[channel_label] = {
            "mae": _stable_mae(
                _finite_absolute_error(channel_prediction, channel_target)
            ),
            "mae_transformed": _stable_mae(
                _finite_absolute_error(
                    channel_prediction_transformed,
                    channel_target_transformed,
                )
            ),
            "prediction_mean": _stable_mean(channel_prediction),
            "target_mean": _stable_mean(channel_target),
            "prediction_max": prediction_max,
            "target_max": target_max,
            "pearson": _pearson_correlation(
                channel_prediction,
                channel_target,
            ),
            "prediction_max_to_target_max_ratio": _finite_ratio(
                prediction_max,
                target_max,
            ),
            "target_max_to_prediction_max_ratio": _finite_ratio(
                target_max,
                prediction_max,
            ),
        }
    return {"count": count, "channels": channels}


def _validated_translation_threshold_candidates(
    value: Sequence[float],
) -> Tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(
            "translation_threshold_candidates must be a sequence of numbers"
        )
    if not value:
        raise ValueError(
            "translation_threshold_candidates must not be empty"
        )
    parsed: List[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool):
            raise TypeError(
                "translation_threshold_candidates "
                f"[{index}] must be a real number, not bool"
            )
        try:
            threshold = float(item)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(
                "translation_threshold_candidates "
                f"[{index}] must be a real number"
            ) from exc
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError(
                "translation_threshold_candidates must contain finite, "
                "nonnegative values"
            )
        if parsed and threshold <= parsed[-1]:
            raise ValueError(
                "translation_threshold_candidates must be strictly increasing"
            )
        parsed.append(threshold)
    return tuple(parsed)


def _binary_operating_point(
    prediction: torch.Tensor,
    target_positive: torch.Tensor,
    threshold: float,
) -> Dict[str, Any]:
    """Return a complete confusion-table row for ``prediction > threshold``."""

    predicted_positive = prediction > threshold
    true_positive = int(
        torch.count_nonzero(target_positive & predicted_positive)
    )
    false_positive = int(
        torch.count_nonzero(~target_positive & predicted_positive)
    )
    false_negative = int(
        torch.count_nonzero(target_positive & ~predicted_positive)
    )
    true_negative = int(
        torch.count_nonzero(~target_positive & ~predicted_positive)
    )
    actual_positive = true_positive + false_negative
    actual_negative = true_negative + false_positive
    predicted_positive_count = true_positive + false_positive
    sample_count = actual_positive + actual_negative

    precision = (
        true_positive / predicted_positive_count
        if predicted_positive_count > 0
        else None
    )
    recall = (
        true_positive / actual_positive
        if actual_positive > 0
        else None
    )
    specificity = (
        true_negative / actual_negative
        if actual_negative > 0
        else None
    )
    false_positive_rate = (
        false_positive / actual_negative
        if actual_negative > 0
        else None
    )
    false_negative_rate = (
        false_negative / actual_positive
        if actual_positive > 0
        else None
    )
    f1_denominator = 2 * true_positive + false_positive + false_negative
    f1 = (
        2 * true_positive / f1_denominator
        if f1_denominator > 0
        else None
    )
    balanced_accuracy = (
        (recall + specificity) / 2.0
        if recall is not None and specificity is not None
        else None
    )
    return {
        "threshold": float(threshold),
        "sample_count": sample_count,
        "actual_positive_count": actual_positive,
        "actual_negative_count": actual_negative,
        "predicted_positive_count": predicted_positive_count,
        "predicted_negative_count": sample_count - predicted_positive_count,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "balanced_accuracy": balanced_accuracy,
    }


def _linear_quantile(
    values: torch.Tensor,
    probability: float,
) -> Optional[float]:
    """Version-stable linear quantile for one finite float64 vector."""

    if values.numel() == 0:
        return None
    sorted_values = torch.sort(values.to(dtype=torch.float64)).values
    position = (int(sorted_values.numel()) - 1) * float(probability)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    lower = float(sorted_values[lower_index])
    upper = float(sorted_values[upper_index])
    fraction = position - lower_index
    return float(lower + (upper - lower) * fraction)


def _prediction_quantiles(
    values: torch.Tensor,
    percentiles: Sequence[int],
) -> Dict[str, Optional[float]]:
    return {
        f"p{percentile}": _linear_quantile(
            values,
            float(percentile) / 100.0,
        )
        for percentile in percentiles
    }


def _binary_ranking_auc(
    prediction: torch.Tensor,
    target_positive: torch.Tensor,
) -> Tuple[Optional[float], Optional[float]]:
    """Return deterministic tie-aware ROC AUC and step-integral PR AUC."""

    scores = [float(value) for value in prediction.to(dtype=torch.float64)]
    labels = [bool(value) for value in target_positive]
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        return None, None

    ascending = sorted(
        range(len(scores)),
        key=lambda index: (scores[index], index),
    )
    ranks = [0.0] * len(scores)
    start = 0
    while start < len(ascending):
        stop = start + 1
        score = scores[ascending[start]]
        while stop < len(ascending) and scores[ascending[stop]] == score:
            stop += 1
        # Ranks are one-based; tied scores receive their average rank.
        average_rank = ((start + 1) + stop) / 2.0
        for position in range(start, stop):
            ranks[ascending[position]] = average_rank
        start = stop
    positive_rank_sum = sum(
        rank for rank, label in zip(ranks, labels) if label
    )
    roc_auc = (
        positive_rank_sum
        - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)

    descending = sorted(
        range(len(scores)),
        key=lambda index: (-scores[index], index),
    )
    true_positive = 0
    false_positive = 0
    previous_recall = 0.0
    average_precision = 0.0
    start = 0
    while start < len(descending):
        stop = start + 1
        score = scores[descending[start]]
        while stop < len(descending) and scores[descending[stop]] == score:
            stop += 1
        for position in range(start, stop):
            if labels[descending[position]]:
                true_positive += 1
            else:
                false_positive += 1
        recall = true_positive / positive_count
        precision = true_positive / (true_positive + false_positive)
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        start = stop
    return float(roc_auc), float(average_precision)


def _selected_operating_point(
    row: Mapping[str, Any],
) -> Dict[str, Any]:
    """Copy one row without sharing mutable source lists."""

    return {
        key: list(value) if key == "sources" else value
        for key, value in row.items()
    }


def calibrate_translation_blockage_thresholds(
    prediction: Any,
    target: Any,
    translation_block_reason_id: Any,
    *,
    threshold_candidates: Sequence[float] = (
        DEFAULT_TRANSLATION_THRESHOLD_CANDIDATES
    ),
    requested_threshold: float = DEFAULT_TRANSLATION_POSITIVE_THRESHOLD,
    translation_block_reason_names: Sequence[str] = (
        TRANSLATION_BLOCK_REASON_NAMES
    ),
    curvature_stratum_id: Optional[Any] = None,
    curvature_stratum_names: Sequence[str] = CURVATURE_STRATUM_NAMES,
) -> Dict[str, Any]:
    """Calibrate diagnostic blockage thresholds on one fixed validation split.

    All configured candidates are expressed in the decoded, original
    normalized translation-error units. Automatic score boundaries make the
    table include all-positive and all-negative operating points, but only
    configured candidates are eligible for deterministic selection.
    """

    prediction_tensor = _as_floating_tensor(
        prediction,
        "translation prediction",
    ).reshape(-1).to(dtype=torch.float64, device="cpu")
    target_tensor = _as_floating_tensor(
        target,
        "translation target",
    ).reshape(-1).to(dtype=torch.float64, device="cpu")
    if (
        prediction_tensor.numel() == 0
        or prediction_tensor.shape != target_tensor.shape
    ):
        raise ValueError(
            "Translation prediction and target must be equal non-empty vectors"
        )
    if not bool(torch.isfinite(prediction_tensor).all()) or not bool(
        torch.isfinite(target_tensor).all()
    ):
        raise FloatingPointError(
            "Translation calibration inputs contain NaN or infinity"
        )
    if bool(torch.any(prediction_tensor < 0)) or bool(
        torch.any(target_tensor < 0)
    ):
        raise ValueError(
            "Translation calibration inputs must be nonnegative"
        )
    configured_candidates = _validated_translation_threshold_candidates(
        threshold_candidates
    )
    requested = float(requested_threshold)
    if not math.isfinite(requested) or requested < 0.0:
        raise ValueError(
            "requested_threshold must be finite and nonnegative"
        )
    reason_names = _validated_group_names(
        translation_block_reason_names,
        name="translation_block_reason_names",
    )
    reason_ids = _validated_group_ids(
        translation_block_reason_id,
        name="translation_block_reason_id",
        sample_count=int(target_tensor.numel()),
        group_count=len(reason_names),
        device=target_tensor.device,
    )
    stratum_names = _validated_group_names(
        curvature_stratum_names,
        name="curvature_stratum_names",
    )
    stratum_ids = (
        _validated_group_ids(
            curvature_stratum_id,
            name="curvature_stratum_id",
            sample_count=int(target_tensor.numel()),
            group_count=len(stratum_names),
            device=target_tensor.device,
        )
        if curvature_stratum_id is not None
        else None
    )
    target_positive = target_tensor > 0.0

    candidate_sources: Dict[float, set[str]] = {}

    def add_candidate(threshold: float, source: str) -> None:
        candidate_sources.setdefault(float(threshold), set()).add(source)

    for threshold in configured_candidates:
        add_candidate(threshold, "configured")
    add_candidate(requested, "requested_evaluation_threshold")
    minimum = torch.amin(prediction_tensor)
    maximum = torch.amax(prediction_tensor)
    lower_boundary = float(
        torch.nextafter(
            minimum,
            minimum.new_tensor(float("-inf")),
        )
    )
    upper_boundary = float(
        torch.nextafter(
            maximum,
            maximum.new_tensor(float("inf")),
        )
    )
    if math.isfinite(lower_boundary):
        add_candidate(lower_boundary, "automatic_all_positive_boundary")
    if math.isfinite(upper_boundary):
        add_candidate(upper_boundary, "automatic_all_negative_boundary")

    configured_set = set(configured_candidates)
    candidate_table = []
    for threshold in sorted(candidate_sources):
        row = _binary_operating_point(
            prediction_tensor,
            target_positive,
            threshold,
        )
        row["sources"] = sorted(candidate_sources[threshold])
        row["eligible_for_selection"] = threshold in configured_set
        candidate_table.append(row)
    eligible = [
        row for row in candidate_table if row["eligible_for_selection"]
    ]

    max_f1_rows = [
        row
        for row in eligible
        if row["f1"] is not None and row["recall"] is not None
    ]
    max_f1 = (
        _selected_operating_point(
            min(
                max_f1_rows,
                key=lambda row: (
                    -float(row["f1"]),
                    -float(row["recall"]),
                    (
                        float(row["false_positive_rate"])
                        if row["false_positive_rate"] is not None
                        else float("inf")
                    ),
                    float(row["threshold"]),
                ),
            )
        )
        if max_f1_rows
        else None
    )
    recall_constrained_rows = [
        row
        for row in eligible
        if row["recall"] is not None
        and row["recall"] >= 0.95
        and row["false_positive_rate"] is not None
    ]
    recall_constrained = (
        _selected_operating_point(
            min(
                recall_constrained_rows,
                key=lambda row: (
                    float(row["false_positive_rate"]),
                    float(row["threshold"]),
                    -(
                        float(row["precision"])
                        if row["precision"] is not None
                        else float("-inf")
                    ),
                ),
            )
        )
        if recall_constrained_rows
        else None
    )
    fpr_constrained_rows = [
        row
        for row in eligible
        if row["false_positive_rate"] is not None
        and row["false_positive_rate"] <= 0.05
        and row["recall"] is not None
    ]
    fpr_constrained = (
        _selected_operating_point(
            min(
                fpr_constrained_rows,
                key=lambda row: (
                    -float(row["recall"]),
                    float(row["threshold"]),
                    -(
                        float(row["precision"])
                        if row["precision"] is not None
                        else float("-inf")
                    ),
                ),
            )
        )
        if fpr_constrained_rows
        else None
    )

    def masked_mean(mask: torch.Tensor) -> Optional[float]:
        return (
            float(torch.mean(prediction_tensor[mask]))
            if int(torch.count_nonzero(mask)) > 0
            else None
        )

    none_id = (
        reason_names.index("none")
        if "none" in reason_names
        else None
    )
    none_mask = (
        reason_ids == none_id
        if none_id is not None
        else torch.zeros_like(target_positive)
    )
    roc_auc, pr_auc = _binary_ranking_auc(
        prediction_tensor,
        target_positive,
    )
    return {
        "schema_version": 1,
        "units": "normalized_requested_applied_translation_error",
        "decision_rule": "decoded_prediction > threshold",
        "threshold_note": (
            "Validation diagnostic only; no calibrated threshold is a "
            "clinical safety threshold."
        ),
        "configured_candidate_thresholds": list(configured_candidates),
        "requested_evaluation_threshold": requested,
        "automatic_boundaries": {
            "all_positive_threshold": lower_boundary,
            "all_negative_threshold": upper_boundary,
            "eligible_for_selection": False,
        },
        "candidate_count": len(candidate_table),
        "selection_candidate_count": len(eligible),
        "candidate_table": candidate_table,
        "selection_rules": {
            "max_f1": (
                "F1 descending, recall descending, FPR ascending, "
                "threshold ascending."
            ),
            "recall_at_least_0_95_lowest_fpr": (
                "Eligible recall >= 0.95; FPR ascending, threshold "
                "ascending, precision descending."
            ),
            "fpr_at_most_0_05_highest_recall": (
                "Eligible FPR <= 0.05; recall descending, threshold "
                "ascending, precision descending."
            ),
            "selection_scope": (
                "Configured candidates only; automatic boundaries and an "
                "unconfigured requested threshold are report-only."
            ),
        },
        "selections": {
            "max_f1": max_f1,
            "recall_at_least_0_95_lowest_fpr": recall_constrained,
            "fpr_at_most_0_05_highest_recall": fpr_constrained,
        },
        "class_counts": {
            "sample_count": int(target_tensor.numel()),
            "positive_count": int(torch.count_nonzero(target_positive)),
            "negative_count": int(torch.count_nonzero(~target_positive)),
        },
        "roc_auc": roc_auc,
        "roc_auc_method": "tie-aware Mann-Whitney rank statistic",
        "pr_auc": pr_auc,
        "pr_auc_method": "average-precision step integral",
        "auc_requires_both_classes": True,
        "prediction_means": {
            "overall": float(torch.mean(prediction_tensor)),
            "target_negative": masked_mean(~target_positive),
            "target_positive": masked_mean(target_positive),
            "per_translation_reason": {
                name: masked_mean(reason_ids == index)
                for index, name in enumerate(reason_names)
            },
            "per_curvature_stratum": (
                {
                    name: masked_mean(stratum_ids == index)
                    for index, name in enumerate(stratum_names)
                }
                if stratum_ids is not None
                else None
            ),
        },
        "prediction_quantiles": {
            "none_reason": {
                "count": int(torch.count_nonzero(none_mask)),
                **_prediction_quantiles(
                    prediction_tensor[none_mask],
                    (50, 90, 95, 99),
                ),
            },
            "positive_target": {
                "count": int(torch.count_nonzero(target_positive)),
                **_prediction_quantiles(
                    prediction_tensor[target_positive],
                    (1, 5, 10, 50),
                ),
            },
        },
    }


def _translation_blockage_metrics(
    *,
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
) -> Dict[str, Any]:
    """Compute non-clinical blockage classification diagnostics."""

    translation_target = target[:, _TRANSLATION_ERROR_INDEX]
    translation_prediction = prediction[:, _TRANSLATION_ERROR_INDEX]
    actual_positive = translation_target > 0
    predicted_positive = translation_prediction > threshold
    true_positive = torch.count_nonzero(actual_positive & predicted_positive)
    false_positive = torch.count_nonzero(~actual_positive & predicted_positive)
    false_negative = torch.count_nonzero(actual_positive & ~predicted_positive)
    true_negative = torch.count_nonzero(~actual_positive & ~predicted_positive)
    actual_positive_count = true_positive + false_negative
    predicted_positive_count = true_positive + false_positive

    if int(actual_positive_count) == 0:
        precision = recall = f1 = _nan(target)
    else:
        precision = (
            true_positive.to(dtype=torch.float64)
            / predicted_positive_count.to(dtype=torch.float64)
            if int(predicted_positive_count) > 0
            else target.new_tensor(0.0, dtype=torch.float64)
        )
        recall = (
            true_positive.to(dtype=torch.float64)
            / actual_positive_count.to(dtype=torch.float64)
        )
        denominator = precision + recall
        f1 = (
            2.0 * precision * recall / denominator
            if bool(denominator > 0)
            else target.new_tensor(0.0, dtype=torch.float64)
        )
    return {
        "threshold": float(threshold),
        "threshold_note": (
            "Diagnostic normalized translation-error threshold only; "
            "not a clinical safety threshold."
        ),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "actual_positive_count": actual_positive_count,
        "predicted_positive_count": predicted_positive_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def aggregate_fixed_safety_validation(
    prediction_transformed: Any,
    target: Any,
    translation_block_reason_id: Any,
    curvature_stratum_id: Any,
    model: Any,
    *,
    translation_positive_threshold: float = (
        DEFAULT_TRANSLATION_POSITIVE_THRESHOLD
    ),
    translation_threshold_candidates: Sequence[float] = (
        DEFAULT_TRANSLATION_THRESHOLD_CANDIDATES
    ),
    safety_cost_names: Sequence[str] = REPLAY_SAFETY_COST_NAMES,
    translation_block_reason_names: Sequence[str] = (
        TRANSLATION_BLOCK_REASON_NAMES
    ),
    curvature_stratum_names: Sequence[str] = CURVATURE_STRATUM_NAMES,
) -> Dict[str, Any]:
    """Aggregate one complete fixed auxiliary validation split.

    Group membership comes from the dataset's stored reason and curvature
    labels.  It is deliberately not inferred again from continuous targets.
    Predictions from all inference minibatches must be concatenated before
    calling this function so Pearson correlations, maxima, and ratios describe
    the complete fixed split rather than an average of minibatch statistics.
    """

    threshold = float(translation_positive_threshold)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError(
            "translation_positive_threshold must be finite and nonnegative"
        )
    reason_names = _validated_group_names(
        translation_block_reason_names,
        name="translation_block_reason_names",
    )
    stratum_names = _validated_group_names(
        curvature_stratum_names,
        name="curvature_stratum_names",
    )
    (
        prediction_transformed,
        target_transformed,
        prediction,
        target,
    ) = _prepare_inputs(
        prediction_transformed,
        target,
        model,
        safety_cost_names,
    )
    sample_count = int(target.shape[0])
    reason_ids = _validated_group_ids(
        translation_block_reason_id,
        name="translation_block_reason_id",
        sample_count=sample_count,
        group_count=len(reason_names),
        device=target.device,
    )
    stratum_ids = _validated_group_ids(
        curvature_stratum_id,
        name="curvature_stratum_id",
        sample_count=sample_count,
        group_count=len(stratum_names),
        device=target.device,
    )
    complete_mask = torch.ones(
        sample_count,
        dtype=torch.bool,
        device=target.device,
    )
    return {
        "sample_count": sample_count,
        "overall": _fixed_validation_group_metrics(
            mask=complete_mask,
            prediction_transformed=prediction_transformed,
            target_transformed=target_transformed,
            prediction=prediction,
            target=target,
        ),
        "per_translation_reason": {
            reason_name: _fixed_validation_group_metrics(
                mask=reason_ids == reason_id,
                prediction_transformed=prediction_transformed,
                target_transformed=target_transformed,
                prediction=prediction,
                target=target,
            )
            for reason_id, reason_name in enumerate(reason_names)
        },
        "per_curvature_stratum": {
            stratum_name: _fixed_validation_group_metrics(
                mask=stratum_ids == stratum_id,
                prediction_transformed=prediction_transformed,
                target_transformed=target_transformed,
                prediction=prediction,
                target=target,
            )
            for stratum_id, stratum_name in enumerate(stratum_names)
        },
        "translation_blockage": _translation_blockage_metrics(
            prediction=prediction,
            target=target,
            threshold=threshold,
        ),
        "translation_threshold_calibration": (
            calibrate_translation_blockage_thresholds(
                prediction=prediction[:, _TRANSLATION_ERROR_INDEX],
                target=target[:, _TRANSLATION_ERROR_INDEX],
                translation_block_reason_id=reason_ids,
                threshold_candidates=translation_threshold_candidates,
                requested_threshold=threshold,
                translation_block_reason_names=reason_names,
                curvature_stratum_id=stratum_ids,
                curvature_stratum_names=stratum_names,
            )
        ),
    }


def evaluate_fixed_safety_validation(
    agent: Any,
    validation_batch: Mapping[str, Any],
    *,
    batch_size: int = DEFAULT_FIXED_VALIDATION_BATCH_SIZE,
    translation_positive_threshold: float = (
        DEFAULT_TRANSLATION_POSITIVE_THRESHOLD
    ),
    translation_threshold_candidates: Sequence[float] = (
        DEFAULT_TRANSLATION_THRESHOLD_CANDIDATES
    ),
) -> Dict[str, Any]:
    """Predict and aggregate one complete immutable validation batch.

    ``validation_batch`` is the state mapping returned by
    ``SafetyAuxDataset.validation_buffer().state_dict()``.  Its requested
    normalized ``action`` field is intentionally used instead of
    ``applied_action``.  Inference may be minibatched, but aggregation happens
    exactly once after all predictions have been concatenated.
    """

    if not isinstance(validation_batch, Mapping):
        raise TypeError("validation_batch must be a mapping")
    required = {
        "size",
        "observation",
        "action",
        "safety_cost",
        "translation_block_reason_id",
        "curvature_stratum_id",
        "safety_cost_names",
        "translation_block_reason_names",
        "curvature_stratum_names",
    }
    missing = sorted(required - validation_batch.keys())
    if missing:
        raise ValueError(
            f"Fixed validation batch is missing required fields {missing}"
        )
    if isinstance(batch_size, bool):
        raise TypeError("Fixed validation batch_size must be an integer")
    try:
        numeric_batch_size = float(batch_size)
        parsed_batch_size = int(batch_size)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(
            "Fixed validation batch_size must be a positive integer"
        ) from exc
    if (
        not math.isfinite(numeric_batch_size)
        or numeric_batch_size != parsed_batch_size
        or parsed_batch_size <= 0
    ):
        raise ValueError(
            "Fixed validation batch_size must be a positive integer"
        )

    sample_count = int(validation_batch["size"])
    if sample_count <= 0:
        raise ValueError("Fixed validation batch must contain at least one sample")
    observation = torch.as_tensor(validation_batch["observation"])
    action = torch.as_tensor(validation_batch["action"])
    target = torch.as_tensor(validation_batch["safety_cost"])
    expected_observation_shape = (sample_count, int(agent.observation_dim))
    expected_action_shape = (sample_count, int(agent.action_dim))
    expected_target_shape = (
        sample_count,
        len(REPLAY_SAFETY_COST_NAMES),
    )
    for tensor, expected_shape, name in (
        (observation, expected_observation_shape, "observation"),
        (action, expected_action_shape, "action"),
        (target, expected_target_shape, "safety_cost"),
    ):
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"Fixed validation {name} must have shape {expected_shape}, "
                f"got {tuple(tensor.shape)}"
            )
        if not tensor.is_floating_point():
            raise TypeError(
                f"Fixed validation {name} must use a floating-point dtype"
            )
        if not bool(torch.isfinite(tensor).all()):
            raise FloatingPointError(
                f"Fixed validation {name} contains NaN or infinity"
            )
    if bool(torch.any(target < 0)):
        raise ValueError("Fixed validation safety_cost must be nonnegative")

    model = agent.model
    was_training = model.training
    model.train(False)
    try:
        with torch.inference_mode():
            prediction_batches = []
            for start in range(0, sample_count, parsed_batch_size):
                stop = min(start + parsed_batch_size, sample_count)
                observation_batch = observation[start:stop].to(
                    device=agent.device,
                    dtype=torch.float32,
                )
                action_batch = action[start:stop].to(
                    device=agent.device,
                    dtype=torch.float32,
                )
                latent = model.encode(observation_batch)
                prediction_batch = model.safety_transformed(
                    latent,
                    action_batch,
                )
                expected_shape = (
                    stop - start,
                    len(REPLAY_SAFETY_COST_NAMES),
                )
                if tuple(prediction_batch.shape) != expected_shape:
                    raise RuntimeError(
                        "Safety Head returned transformed prediction shape "
                        f"{tuple(prediction_batch.shape)}; expected "
                        f"{expected_shape}"
                    )
                if not bool(torch.isfinite(prediction_batch).all()):
                    raise FloatingPointError(
                        "Safety Head returned non-finite fixed-validation "
                        "predictions"
                    )
                prediction_batches.append(prediction_batch)
            prediction_transformed = torch.cat(prediction_batches, dim=0)
            target_device = target.to(
                device=agent.device,
                dtype=prediction_transformed.dtype,
            )
            metrics = aggregate_fixed_safety_validation(
                prediction_transformed=prediction_transformed,
                target=target_device,
                translation_block_reason_id=validation_batch[
                    "translation_block_reason_id"
                ],
                curvature_stratum_id=validation_batch[
                    "curvature_stratum_id"
                ],
                model=model,
                translation_positive_threshold=(
                    translation_positive_threshold
                ),
                translation_threshold_candidates=(
                    translation_threshold_candidates
                ),
                safety_cost_names=validation_batch["safety_cost_names"],
                translation_block_reason_names=validation_batch[
                    "translation_block_reason_names"
                ],
                curvature_stratum_names=validation_batch[
                    "curvature_stratum_names"
                ],
            )
            transformed_target = model.transform_safety_targets(target_device)
            channel_loss = F.smooth_l1_loss(
                prediction_transformed,
                transformed_target,
                reduction="none",
            ).mean(dim=0)
            curvature_loss = channel_loss[_CURVATURE_INDEX]
            translation_loss = channel_loss[_TRANSLATION_ERROR_INDEX]
            combined_loss = (
                float(agent.safety_curvature_loss_coef) * curvature_loss
                + float(agent.safety_translation_error_loss_coef)
                * translation_loss
            )
            metrics["loss"] = {
                "combined": combined_loss,
                "curvature": curvature_loss,
                "translation_error": translation_loss,
                "temporal_rho_weighting_applied": False,
            }
            metrics["inference_batch_count"] = len(prediction_batches)
            metrics["inference_batch_size"] = parsed_batch_size
            return metrics
    finally:
        model.train(was_training)


def flatten_fixed_safety_validation_metrics(
    metrics: Mapping[str, Any],
    *,
    prefix: str = "aux_val_",
) -> Dict[str, float]:
    """Flatten nested fixed-validation metrics for scalar training logs."""

    if not isinstance(metrics, Mapping):
        raise TypeError("Fixed validation metrics must be a mapping")
    normalized_prefix = _normalise_prefix(prefix)

    def scalar(value: Any, name: str) -> float:
        if value is None:
            return float("nan")
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"{name} must be scalar")
            value = value.detach().cpu().item()
        if isinstance(value, bool):
            return float(value)
        try:
            return float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(f"{name} must be numeric") from exc

    output: Dict[str, float] = {
        f"{normalized_prefix}sample_count": scalar(
            metrics["sample_count"],
            "sample_count",
        ),
        f"{normalized_prefix}safety_loss": scalar(
            metrics["loss"]["combined"],
            "loss.combined",
        ),
        f"{normalized_prefix}safety_curvature_loss": scalar(
            metrics["loss"]["curvature"],
            "loss.curvature",
        ),
        f"{normalized_prefix}safety_translation_error_loss": scalar(
            metrics["loss"]["translation_error"],
            "loss.translation_error",
        ),
    }

    metric_names = (
        "mae",
        "mae_transformed",
        "prediction_mean",
        "target_mean",
        "prediction_max",
        "target_max",
        "pearson",
        "prediction_max_to_target_max_ratio",
        "target_max_to_prediction_max_ratio",
    )

    def add_group(base: str, group: Mapping[str, Any]) -> None:
        output[f"{normalized_prefix}{base}count"] = scalar(
            group["count"],
            f"{base}count",
        )
        for channel_name, channel in group["channels"].items():
            for metric_name in metric_names:
                output[
                    f"{normalized_prefix}{base}{channel_name}_{metric_name}"
                ] = scalar(
                    channel[metric_name],
                    f"{base}{channel_name}.{metric_name}",
                )

    add_group("", metrics["overall"])
    # Keep the shorter externally requested name while retaining the canonical
    # channel label used everywhere else in the Safety model.
    output[f"{normalized_prefix}translation_mae"] = output[
        f"{normalized_prefix}translation_error_mae"
    ]
    for reason_name, group in metrics["per_translation_reason"].items():
        add_group(f"reason_{reason_name}_", group)
    for stratum_name, group in metrics["per_curvature_stratum"].items():
        add_group(f"curvature_stratum_{stratum_name}_", group)

    classification = metrics["translation_blockage"]
    for name in (
        "true_positive",
        "false_positive",
        "false_negative",
        "true_negative",
        "actual_positive_count",
        "predicted_positive_count",
        "precision",
        "recall",
        "f1",
    ):
        output[
            f"{normalized_prefix}translation_blockage_{name}"
        ] = scalar(
            classification[name],
            f"translation_blockage.{name}",
        )

    calibration = metrics["translation_threshold_calibration"]
    output[
        f"{normalized_prefix}translation_calibration_roc_auc"
    ] = scalar(calibration["roc_auc"], "calibration.roc_auc")
    output[
        f"{normalized_prefix}translation_calibration_pr_auc"
    ] = scalar(calibration["pr_auc"], "calibration.pr_auc")
    output[
        f"{normalized_prefix}translation_calibration_candidate_count"
    ] = scalar(
        calibration["candidate_count"],
        "calibration.candidate_count",
    )
    output[
        f"{normalized_prefix}translation_calibration_selection_candidate_count"
    ] = scalar(
        calibration["selection_candidate_count"],
        "calibration.selection_candidate_count",
    )
    for group_name, value in calibration["prediction_means"][
        "per_translation_reason"
    ].items():
        output[
            f"{normalized_prefix}translation_prediction_mean_reason_{group_name}"
        ] = scalar(
            value,
            f"calibration.prediction_means.reason.{group_name}",
        )
    for group_name, value in (
        calibration["prediction_means"]["per_curvature_stratum"] or {}
    ).items():
        output[
            f"{normalized_prefix}translation_prediction_mean_stratum_{group_name}"
        ] = scalar(
            value,
            f"calibration.prediction_means.stratum.{group_name}",
        )
    for group_name in ("none_reason", "positive_target"):
        group = calibration["prediction_quantiles"][group_name]
        output[
            f"{normalized_prefix}translation_{group_name}_count"
        ] = scalar(
            group["count"],
            f"calibration.quantiles.{group_name}.count",
        )
        for quantile_name, value in group.items():
            if quantile_name == "count":
                continue
            output[
                f"{normalized_prefix}translation_{group_name}_{quantile_name}"
            ] = scalar(
                value,
                f"calibration.quantiles.{group_name}.{quantile_name}",
            )

    selected_metric_names = (
        "threshold",
        "precision",
        "recall",
        "f1",
        "specificity",
        "false_positive_rate",
        "false_negative_rate",
        "balanced_accuracy",
    )
    for selection_name, selection in calibration["selections"].items():
        for metric_name in selected_metric_names:
            output[
                f"{normalized_prefix}translation_calibration_"
                f"{selection_name}_{metric_name}"
            ] = scalar(
                selection.get(metric_name) if selection is not None else None,
                f"calibration.selections.{selection_name}.{metric_name}",
            )
    return output


def _validate_curvature_boundaries(
    boundaries: Tuple[float, float],
) -> Tuple[float, float]:
    if not isinstance(boundaries, (tuple, list)) or len(boundaries) != 2:
        raise TypeError(
            "curvature_boundaries must contain exactly two numeric values"
        )
    low_max, medium_max = (float(value) for value in boundaries)
    if (
        not math.isfinite(low_max)
        or not math.isfinite(medium_max)
        or low_max < 0.0
        or medium_max <= low_max
    ):
        raise ValueError(
            "curvature_boundaries must be finite and satisfy "
            "0 <= low_max < medium_max"
        )
    return low_max, medium_max


def _add_masked_metrics(
    metrics: Dict[str, torch.Tensor],
    *,
    base_key: str,
    mask: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    absolute_error: torch.Tensor,
) -> None:
    count = torch.count_nonzero(mask)
    metrics[f"{base_key}_count"] = count
    if int(count) == 0:
        unavailable = _nan(prediction)
        metrics[f"{base_key}_mae"] = unavailable.clone()
        metrics[f"{base_key}_pred_mean"] = unavailable.clone()
        metrics[f"{base_key}_target_mean"] = unavailable.clone()
        return
    metrics[f"{base_key}_mae"] = _stable_mae(absolute_error[mask])
    metrics[f"{base_key}_pred_mean"] = _stable_mean(prediction[mask])
    metrics[f"{base_key}_target_mean"] = _stable_mean(target[mask])


def safety_batch_diagnostics(
    prediction_transformed: Any,
    target: Any,
    model: Any,
    *,
    prefix: str = "",
    curvature_boundaries: Tuple[float, float] = (
        CURVATURE_DIAGNOSTIC_BOUNDARIES_MM_INV
    ),
    safety_cost_names: Sequence[str] = REPLAY_SAFETY_COST_NAMES,
) -> Dict[str, torch.Tensor]:
    """Compute detached scalar diagnostics for one replay or validation batch.

    ``prediction_transformed`` and ``target`` may have any non-empty prefix
    shape, but their final dimension must contain the canonical two safety
    channels. ``target`` is always interpreted in original physical units.
    Translation positives use the replay target's exact ``target > 0``
    semantics.
    """

    prefix = _normalise_prefix(prefix)
    low_max, medium_max = _validate_curvature_boundaries(
        curvature_boundaries
    )
    (
        prediction_transformed,
        target_transformed,
        prediction,
        target,
    ) = _prepare_inputs(
        prediction_transformed,
        target,
        model,
        safety_cost_names,
    )

    metrics: Dict[str, torch.Tensor] = {}
    for channel_index, channel_label in enumerate(_CHANNEL_LABELS):
        channel_prediction = prediction[:, channel_index]
        channel_target = target[:, channel_index]
        transformed_error = _finite_absolute_error(
            prediction_transformed[:, channel_index],
            target_transformed[:, channel_index],
        )
        original_error = _finite_absolute_error(
            channel_prediction, channel_target
        )
        base_key = f"{prefix}safety_{channel_label}"
        metrics[f"{prefix}safety_pred_{channel_label}_mean"] = _stable_mean(
            channel_prediction
        )
        metrics[f"{prefix}safety_target_{channel_label}_mean"] = _stable_mean(
            channel_target
        )
        metrics[f"{prefix}safety_pred_{channel_label}_max"] = torch.amax(
            channel_prediction
        )
        metrics[f"{prefix}safety_target_{channel_label}_max"] = torch.amax(
            channel_target
        )
        metrics[f"{base_key}_mae_transformed"] = _stable_mae(
            transformed_error
        )
        metrics[f"{base_key}_mae"] = _stable_mae(original_error)

    translation_target = target[:, _TRANSLATION_ERROR_INDEX]
    translation_prediction = prediction[:, _TRANSLATION_ERROR_INDEX]
    translation_error = _finite_absolute_error(
        translation_prediction, translation_target
    )
    positive_mask = translation_target > 0
    zero_mask = translation_target == 0
    sample_count = translation_target.numel()
    positive_count = torch.count_nonzero(positive_mask)
    metrics[
        f"{prefix}safety_translation_error_positive_count"
    ] = positive_count
    metrics[
        f"{prefix}safety_translation_error_positive_fraction"
    ] = positive_count.to(dtype=torch.float64) / sample_count
    metrics[
        f"{prefix}safety_translation_error_zero_fraction"
    ] = torch.count_nonzero(zero_mask).to(dtype=torch.float64) / sample_count

    positive_key = f"{prefix}safety_translation_error_positive"
    zero_key = f"{prefix}safety_translation_error_zero"
    _add_masked_metrics(
        metrics,
        base_key=positive_key,
        mask=positive_mask,
        prediction=translation_prediction,
        target=translation_target,
        absolute_error=translation_error,
    )
    # Only zero-target MAE is required here; retaining the other two moments
    # makes train/validation reports symmetric and is inexpensive.
    _add_masked_metrics(
        metrics,
        base_key=zero_key,
        mask=zero_mask,
        prediction=translation_prediction,
        target=translation_target,
        absolute_error=translation_error,
    )

    curvature_target = target[:, _CURVATURE_INDEX]
    curvature_prediction = prediction[:, _CURVATURE_INDEX]
    curvature_error = _finite_absolute_error(
        curvature_prediction, curvature_target
    )
    curvature_masks = {
        "low": curvature_target < low_max,
        "medium": (curvature_target >= low_max)
        & (curvature_target < medium_max),
        "high": curvature_target >= medium_max,
    }
    for group_name, mask in curvature_masks.items():
        _add_masked_metrics(
            metrics,
            base_key=f"{prefix}safety_curvature_{group_name}",
            mask=mask,
            prediction=curvature_prediction,
            target=curvature_target,
            absolute_error=curvature_error,
        )
    return metrics


def aggregate_safety_evaluation(
    prediction_transformed: Any,
    target: Any,
    model: Any,
    *,
    translation_positive_threshold: float = (
        DEFAULT_TRANSLATION_POSITIVE_THRESHOLD
    ),
    prefix: str = "",
    safety_cost_names: Sequence[str] = REPLAY_SAFETY_COST_NAMES,
) -> Dict[str, torch.Tensor]:
    """Aggregate episode-level Safety Head prediction metrics.

    The ground-truth positive class follows the cleaned replay schema's exact
    ``target > 0`` semantics. ``translation_positive_threshold`` is applied
    only to continuous predictions in original normalized-error units.  It
    detects intervention-level translation blockage and is not a clinical
    safety threshold.
    """

    prefix = _normalise_prefix(prefix)
    threshold = float(translation_positive_threshold)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError(
            "translation_positive_threshold must be finite and nonnegative"
        )
    (
        prediction_transformed,
        target_transformed,
        prediction,
        target,
    ) = _prepare_inputs(
        prediction_transformed,
        target,
        model,
        safety_cost_names,
    )

    metrics: Dict[str, torch.Tensor] = {}
    transition_count = int(target.shape[0])
    metrics[f"{prefix}safety_transition_count"] = torch.tensor(
        transition_count, dtype=torch.int64, device=target.device
    )

    for channel_index, channel_label in enumerate(_CHANNEL_LABELS):
        channel_prediction = prediction[:, channel_index]
        channel_target = target[:, channel_index]
        original_error = _finite_absolute_error(
            channel_prediction, channel_target
        )
        transformed_error = _finite_absolute_error(
            prediction_transformed[:, channel_index],
            target_transformed[:, channel_index],
        )
        base_key = f"{prefix}safety_{channel_label}"
        metrics[f"{base_key}_mae"] = _stable_mae(original_error)
        metrics[f"{base_key}_rmse"] = _stable_rmse(original_error)
        metrics[f"{base_key}_mae_transformed"] = _stable_mae(
            transformed_error
        )
        metrics[f"{base_key}_pearson"] = _pearson_correlation(
            channel_prediction, channel_target
        )
        metrics[f"{base_key}_pred_mean"] = _stable_mean(channel_prediction)
        metrics[f"{base_key}_pred_std"] = _stable_std(channel_prediction)
        metrics[f"{base_key}_pred_max"] = torch.amax(channel_prediction)
        metrics[f"{base_key}_target_mean"] = _stable_mean(channel_target)
        metrics[f"{base_key}_target_std"] = _stable_std(channel_target)
        metrics[f"{base_key}_target_max"] = torch.amax(channel_target)

    translation_target = target[:, _TRANSLATION_ERROR_INDEX]
    translation_prediction = prediction[:, _TRANSLATION_ERROR_INDEX]
    translation_error = _finite_absolute_error(
        translation_prediction, translation_target
    )
    positive_mask = translation_target > 0
    zero_mask = translation_target == 0
    positive_count = torch.count_nonzero(positive_mask)
    zero_count = torch.count_nonzero(zero_mask)
    metrics[
        f"{prefix}safety_translation_error_positive_count"
    ] = positive_count
    metrics[
        f"{prefix}safety_translation_error_positive_fraction"
    ] = positive_count.to(dtype=torch.float64) / transition_count
    metrics[
        f"{prefix}safety_translation_error_zero_fraction"
    ] = zero_count.to(dtype=torch.float64) / transition_count

    if int(positive_count) == 0:
        metrics[
            f"{prefix}safety_translation_error_positive_mae"
        ] = _nan(target)
    else:
        metrics[
            f"{prefix}safety_translation_error_positive_mae"
        ] = _stable_mae(translation_error[positive_mask])
    metrics[f"{prefix}safety_translation_error_zero_mae"] = (
        _stable_mae(translation_error[zero_mask])
        if int(zero_count) > 0
        else _nan(target)
    )

    classification_target = positive_mask
    classification_prediction = translation_prediction > threshold
    classification_positive_count = torch.count_nonzero(
        classification_target
    )
    if int(classification_positive_count) == 0:
        unavailable = _nan(target)
        metrics[
            f"{prefix}safety_translation_error_precision"
        ] = unavailable.clone()
        metrics[
            f"{prefix}safety_translation_error_recall"
        ] = unavailable.clone()
        metrics[
            f"{prefix}safety_translation_error_f1"
        ] = unavailable.clone()
    else:
        true_positive = torch.count_nonzero(
            classification_target & classification_prediction
        ).to(dtype=torch.float64)
        predicted_positive = torch.count_nonzero(
            classification_prediction
        ).to(dtype=torch.float64)
        actual_positive = classification_positive_count.to(dtype=torch.float64)
        precision = (
            true_positive / predicted_positive
            if bool(predicted_positive > 0)
            else target.new_tensor(0.0)
        )
        recall = true_positive / actual_positive
        denominator = precision + recall
        f1 = (
            2.0 * precision * recall / denominator
            if bool(denominator > 0)
            else target.new_tensor(0.0)
        )
        metrics[
            f"{prefix}safety_translation_error_precision"
        ] = precision
        metrics[f"{prefix}safety_translation_error_recall"] = recall
        metrics[f"{prefix}safety_translation_error_f1"] = f1
    return metrics


# Backward-friendly descriptive aliases for callers that refer to a diagnostic
# batch or one complete episode explicitly.
safety_prediction_diagnostics = safety_batch_diagnostics
aggregate_safety_episode = aggregate_safety_evaluation


__all__ = [
    "CURVATURE_DIAGNOSTIC_BOUNDARIES_MM_INV",
    "DEFAULT_FIXED_VALIDATION_BATCH_SIZE",
    "DEFAULT_TRANSLATION_POSITIVE_THRESHOLD",
    "DEFAULT_TRANSLATION_THRESHOLD_CANDIDATES",
    "aggregate_fixed_safety_validation",
    "aggregate_safety_episode",
    "aggregate_safety_evaluation",
    "calibrate_translation_blockage_thresholds",
    "evaluate_fixed_safety_validation",
    "flatten_fixed_safety_validation_metrics",
    "safety_batch_diagnostics",
    "safety_prediction_diagnostics",
]
