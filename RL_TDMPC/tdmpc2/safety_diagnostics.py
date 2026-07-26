"""Diagnostics for the two-channel latent safety prediction head.

The curvature ranges in this module are diagnostic buckets only.  They are
chosen to make errors at different numerical scales visible and must not be
interpreted as clinical safety thresholds.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Sequence, Tuple

import torch

from .replay_buffer import REPLAY_SAFETY_COST_NAMES


CURVATURE_DIAGNOSTIC_BOUNDARIES_MM_INV: Tuple[float, float] = (0.05, 0.1)
DEFAULT_TRANSLATION_POSITIVE_THRESHOLD = 1e-6

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
    "DEFAULT_TRANSLATION_POSITIVE_THRESHOLD",
    "aggregate_safety_episode",
    "aggregate_safety_evaluation",
    "safety_batch_diagnostics",
    "safety_prediction_diagnostics",
]
