#!/usr/bin/env python3
"""Evaluate multi-step latent Safety rollouts without changing TD-MPC2 planning.

The evaluator has two deliberately separate phases:

1. Collect complete real stEVE trajectories without invoking any function in
   this module from ``agent.act()`` or ``TDMPC2Agent._plan()``.
2. Freeze those trajectories and run teacher-forced and open-loop Safety
   inference offline.

Temporal alignment is strict.  At rollout offset ``k`` the prediction is
``safety(z_t+k, requested_action_t+k)`` and is compared with
``safety_cost_t+k``.  The prediction is made *before*
``dynamics(z_t+k, requested_action_t+k)`` advances the imagined latent.  No
target is shifted by one transition, incomplete terminal windows are not
padded, and applied actions are never substituted for requested actions.

Translation Safety is controller-feasibility risk produced by the existing
execution-layer action mask.  It is not contact force, guidewire slippage,
tissue damage, or a clinical safety measure.  Curvature strata and thresholds
in this script are diagnostic only.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional
from typing import Sequence, Tuple

import numpy as np
import torch

from collect_safety_dataset import _at_tree_end
from envs.safety import (
    CURVATURE_STRATUM_NAMES,
    SAFETY_COST_NAMES,
    safety_aux_metadata_from_metrics,
)
from envs.steve_env import StEVEEnv, make_steve_env
from evaluate_safety_head import load_evaluation_model
from eve.intervention import TRANSLATION_BLOCK_REASON_NAMES
from tdmpc2.common import (
    curvature_boundaries_from_diagnostics,
    set_seed,
)
from train import validate_collected_safety_cost


CURVATURE_INDEX = 0
TRANSLATION_INDEX = 1
FIXED_TRANSLATION_THRESHOLD = 0.2
DEFAULT_HORIZONS = (1, 3, 5, 10)
DEFAULT_LOOKBACKS = (1, 3, 5, 10)
DEFAULT_SEEDS = (110000, 120000, 130000, 140000)
DEFAULT_OUTPUT_ROOT = Path("/tmp/steve_safety_rollout")


def _as_finite_float_array(
    value: Any,
    *,
    name: str,
    ndim: Optional[int] = None,
) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float32)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise FloatingPointError(f"{name} contains NaN or infinity")
    return array


def _model_device(model: torch.nn.Module, requested: torch.device) -> torch.device:
    try:
        parameter_device = next(model.parameters()).device
    except StopIteration:
        parameter_device = requested
    if parameter_device != requested:
        raise ValueError(
            f"Requested rollout device {requested} does not match model device "
            f"{parameter_device}"
        )
    return parameter_device


def _decode_safety(
    model: torch.nn.Module,
    latent: torch.Tensor,
    action: torch.Tensor,
) -> torch.Tensor:
    transformed = model.safety_transformed(latent, action)
    expected_shape = (*latent.shape[:-1], len(SAFETY_COST_NAMES))
    if tuple(transformed.shape) != expected_shape:
        raise RuntimeError(
            "Safety Head transformed output has shape "
            f"{tuple(transformed.shape)}; expected {expected_shape}"
        )
    if not bool(torch.isfinite(transformed).all()):
        raise FloatingPointError(
            "Safety Head transformed output contains NaN or infinity"
        )
    decoded = model.decode_safety_transformed(transformed)
    if tuple(decoded.shape) != expected_shape:
        raise RuntimeError(
            f"Decoded Safety output has shape {tuple(decoded.shape)}; "
            f"expected {expected_shape}"
        )
    if not bool(torch.isfinite(decoded).all()):
        raise FloatingPointError(
            "Decoded Safety output contains NaN or infinity"
        )
    if bool(torch.any(decoded < 0)):
        raise ValueError("Decoded Safety output must be nonnegative")
    return decoded


def teacher_forced_rollout(
    model: torch.nn.Module,
    observations: Any,
    actions: Any,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Predict immediate Safety from every real encoded observation.

    ``observations[k]``, ``actions[k]``, and the caller's
    ``safety_cost[k]`` all refer to the same transition.  The function uses
    requested normalized actions and never advances learned dynamics.
    """

    observation_array = _as_finite_float_array(
        observations, name="teacher observations", ndim=2
    ).astype(np.float32, copy=False)
    action_array = _as_finite_float_array(
        actions, name="teacher actions", ndim=2
    ).astype(np.float32, copy=False)
    if observation_array.shape[0] != action_array.shape[0]:
        raise ValueError(
            "Teacher observations/actions must contain the same number of "
            "same-step transitions"
        )
    if action_array.shape[1] <= 0:
        raise ValueError("Teacher action dimension must be positive")
    _model_device(model, device)
    with torch.no_grad():
        observation_tensor = torch.as_tensor(
            observation_array, dtype=torch.float32, device=device
        )
        action_tensor = torch.as_tensor(
            action_array, dtype=torch.float32, device=device
        )
        latent = model.encode(observation_tensor)
        if latent.ndim != 2 or latent.shape[0] != observation_tensor.shape[0]:
            raise RuntimeError("Encoder returned an invalid teacher latent shape")
        if not bool(torch.isfinite(latent).all()):
            raise FloatingPointError("Teacher latent contains NaN or infinity")
        prediction = _decode_safety(model, latent, action_tensor)
    return {
        "prediction": prediction.detach().cpu().numpy().astype(np.float32),
        "latent": latent.detach().cpu().numpy().astype(np.float32),
    }


def open_loop_rollout(
    model: torch.nn.Module,
    initial_observation: Any,
    actions: Any,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Imagine Safety for one action sequence, encoding only its first state.

    At every offset the exact order is:

    ``safety_hat_k = safety(z_hat_k, action_k)``
    ``z_hat_k+1 = dynamics(z_hat_k, action_k)``.

    This ordering is the central same-step alignment invariant.
    """

    initial_array = _as_finite_float_array(
        initial_observation, name="initial observation", ndim=1
    ).astype(np.float32, copy=False)
    action_array = _as_finite_float_array(
        actions, name="open-loop actions", ndim=2
    ).astype(np.float32, copy=False)
    _model_device(model, device)
    predictions: List[torch.Tensor] = []
    latents: List[torch.Tensor] = []
    with torch.no_grad():
        latent = model.encode(
            torch.as_tensor(
                initial_array, dtype=torch.float32, device=device
            ).unsqueeze(0)
        )
        if latent.ndim != 2 or latent.shape[0] != 1:
            raise RuntimeError("Encoder returned an invalid initial latent shape")
        latents.append(latent)
        action_tensor = torch.as_tensor(
            action_array, dtype=torch.float32, device=device
        )
        for offset in range(action_tensor.shape[0]):
            action = action_tensor[offset : offset + 1]
            predictions.append(_decode_safety(model, latent, action))
            latent = model.next(latent, action)
            if not bool(torch.isfinite(latent).all()):
                raise FloatingPointError(
                    f"Open-loop latent at offset {offset + 1} is non-finite"
                )
            latents.append(latent)
    return {
        "prediction": torch.cat(predictions, dim=0)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32),
        "latent": torch.cat(latents, dim=0)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32),
    }


def _batched_open_loop_rollout(
    model: torch.nn.Module,
    initial_observations: np.ndarray,
    action_sequences: np.ndarray,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Vectorized open-loop inference for ``(windows, horizon, action_dim)``."""

    initial = _as_finite_float_array(
        initial_observations, name="batched initial observations", ndim=2
    ).astype(np.float32, copy=False)
    actions = _as_finite_float_array(
        action_sequences, name="batched action sequences", ndim=3
    ).astype(np.float32, copy=False)
    if initial.shape[0] != actions.shape[0]:
        raise ValueError("Batched rollout window counts do not match")
    _model_device(model, device)
    prediction_steps: List[torch.Tensor] = []
    latent_steps: List[torch.Tensor] = []
    with torch.no_grad():
        latent = model.encode(
            torch.as_tensor(initial, dtype=torch.float32, device=device)
        )
        latent_steps.append(latent)
        action_tensor = torch.as_tensor(
            actions, dtype=torch.float32, device=device
        )
        for offset in range(actions.shape[1]):
            action = action_tensor[:, offset]
            prediction_steps.append(_decode_safety(model, latent, action))
            latent = model.next(latent, action)
            if not bool(torch.isfinite(latent).all()):
                raise FloatingPointError(
                    f"Batched open-loop latent at offset {offset + 1} "
                    "contains NaN or infinity"
                )
            latent_steps.append(latent)
    return {
        "prediction": torch.stack(prediction_steps, dim=1)
        .cpu()
        .numpy()
        .astype(np.float32),
        "latent": torch.stack(latent_steps, dim=1)
        .cpu()
        .numpy()
        .astype(np.float32),
    }


def build_episode_windows(
    episode_ids: Any,
    horizon: int,
) -> List[np.ndarray]:
    """Return sequential, unpadded, non-crossing transition-index windows."""

    if isinstance(horizon, bool) or int(horizon) != horizon or int(horizon) <= 0:
        raise ValueError("horizon must be a positive integer")
    parsed_horizon = int(horizon)
    identifiers = np.asarray(episode_ids)
    if identifiers.ndim != 1:
        raise ValueError("episode_ids must be one-dimensional")
    windows: List[np.ndarray] = []
    start = 0
    while start < identifiers.size:
        stop = start + 1
        while stop < identifiers.size and identifiers[stop] == identifiers[start]:
            stop += 1
        episode_length = stop - start
        for offset in range(max(episode_length - parsed_horizon + 1, 0)):
            windows.append(
                np.arange(
                    start + offset,
                    start + offset + parsed_horizon,
                    dtype=np.int64,
                )
            )
        start = stop
    return windows


def aggregate_future_event_targets_and_scores(
    true_translation: Any,
    predicted_translation: Any,
    discount: float,
) -> Dict[str, Any]:
    """Derive blockage-within-window labels and max/discounted rollout risk."""

    target = _as_finite_float_array(
        true_translation, name="true translation", ndim=None
    )
    prediction = _as_finite_float_array(
        predicted_translation, name="predicted translation", ndim=None
    )
    if target.shape != prediction.shape or target.ndim not in (1, 2):
        raise ValueError(
            "true/predicted translation must have equal 1-D or 2-D shapes"
        )
    parsed_discount = float(discount)
    if not np.isfinite(parsed_discount) or not 0.0 <= parsed_discount <= 1.0:
        raise ValueError("discount must be finite and in [0, 1]")
    if np.any(target < 0.0) or np.any(prediction < 0.0):
        raise ValueError("Translation targets and predictions must be nonnegative")
    weights = np.power(
        parsed_discount,
        np.arange(target.shape[-1], dtype=np.float64),
    )
    true_event = np.any(target > 0.0, axis=-1)
    risk_max = np.max(prediction, axis=-1)
    risk_sum = np.sum(prediction * weights, axis=-1)
    return {
        "true_event": true_event,
        "risk_max": risk_max,
        "risk_sum": risk_sum,
    }


def compute_latent_drift(
    predicted_latents: Any,
    encoded_latents: Any,
) -> Dict[str, np.ndarray]:
    """Compare latents along their final feature dimension at each offset."""

    predicted = _as_finite_float_array(
        predicted_latents, name="predicted latents"
    ).astype(np.float64, copy=False)
    encoded = _as_finite_float_array(
        encoded_latents, name="encoded latents"
    ).astype(np.float64, copy=False)
    if predicted.shape != encoded.shape or predicted.ndim < 2:
        raise ValueError(
            "predicted/encoded latents must have identical shapes with a "
            "final latent feature dimension"
        )
    difference = predicted - encoded
    mse = np.mean(np.square(difference), axis=-1)
    l1 = np.mean(np.abs(difference), axis=-1)
    numerator = np.sum(predicted * encoded, axis=-1)
    denominator = np.linalg.norm(predicted, axis=-1) * np.linalg.norm(
        encoded, axis=-1
    )
    cosine = np.ones_like(numerator, dtype=np.float64)
    nonzero = denominator > 0.0
    cosine[nonzero] = numerator[nonzero] / denominator[nonzero]
    cosine = np.clip(cosine, -1.0, 1.0)
    for value, name in ((mse, "mse"), (l1, "l1"), (cosine, "cosine")):
        if not np.all(np.isfinite(value)):
            raise FloatingPointError(f"Latent drift {name} is non-finite")
    return {
        "mse": mse,
        "l1": l1,
        "cosine_similarity": cosine,
    }


def evaluate_early_warnings(
    episode_ids: Any,
    translation_targets: Any,
    open_loop_predictor: Callable[[int, int], np.ndarray],
    lookbacks: Sequence[int],
    threshold: float,
    actual_step_durations: Any,
    *,
    reason_ids: Optional[Any] = None,
    episode_steps: Optional[Any] = None,
) -> Dict[str, Any]:
    """Evaluate warnings strictly before every recorded blockage transition.

    A lookback of ``L`` asks the model to imagine from real steps
    ``event-L`` through ``event-1``.  Each such imagined sequence includes the
    event action, so the internal rollout can be ``L+1`` transitions.  This is
    required to measure a literal 10-step lead without attributing risk from a
    window that does not contain that event.
    """

    identifiers = np.asarray(episode_ids)
    target = _as_finite_float_array(
        translation_targets, name="early-warning target", ndim=1
    )
    durations = _as_finite_float_array(
        actual_step_durations, name="actual step durations", ndim=1
    )
    if identifiers.ndim != 1 or not (
        identifiers.shape == target.shape == durations.shape
    ):
        raise ValueError("Early-warning arrays must be aligned one-dimensional data")
    if np.any(target < 0.0) or np.any(durations < 0.0):
        raise ValueError("Early-warning targets/durations must be nonnegative")
    parsed_threshold = float(threshold)
    if not np.isfinite(parsed_threshold) or parsed_threshold < 0.0:
        raise ValueError("Early-warning threshold must be finite and nonnegative")
    parsed_lookbacks = tuple(dict.fromkeys(int(value) for value in lookbacks))
    if not parsed_lookbacks or any(value <= 0 for value in parsed_lookbacks):
        raise ValueError("lookbacks must contain positive integers")
    reasons = (
        np.zeros(target.shape, dtype=np.int64)
        if reason_ids is None
        else np.asarray(reason_ids)
    )
    steps = (
        np.arange(1, target.size + 1, dtype=np.int64)
        if episode_steps is None
        else np.asarray(episode_steps)
    )
    if reasons.shape != target.shape or steps.shape != target.shape:
        raise ValueError("reason_ids/episode_steps must align with targets")

    episode_start = np.empty(target.shape, dtype=np.int64)
    current_start = 0
    for index in range(target.size):
        if index == 0 or identifiers[index] != identifiers[index - 1]:
            current_start = index
        episode_start[index] = current_start

    # One blockage event is the 0 -> positive onset of a contiguous blocked
    # run.  A second blocked command on the immediately following transition
    # is not a new event: counting it would incorrectly treat a prediction
    # made after blockage had already started as an "early" warning.
    positive = target > 0.0
    previous_positive_same_episode = np.zeros(target.shape, dtype=bool)
    if target.size > 1:
        previous_positive_same_episode[1:] = (
            positive[:-1] & (identifiers[1:] == identifiers[:-1])
        )
    event_indices = np.flatnonzero(positive & ~previous_positive_same_episode)

    event_records: List[Dict[str, Any]] = []
    for event_index in event_indices:
        immediate_prediction = np.asarray(
            open_loop_predictor(int(event_index), 1), dtype=np.float64
        )
        if immediate_prediction.shape != (1,) or not np.all(
            np.isfinite(immediate_prediction)
        ):
            raise ValueError("Immediate early-warning prediction must have shape (1,)")
        by_lookback: Dict[str, Any] = {}
        for lookback in parsed_lookbacks:
            minimum_start = max(
                int(episode_start[event_index]),
                int(event_index) - lookback,
            )
            earliest: Optional[int] = None
            maximum_risk: Optional[float] = None
            for start_index in range(minimum_start, int(event_index)):
                rollout_horizon = int(event_index) - start_index + 1
                prediction = np.asarray(
                    open_loop_predictor(start_index, rollout_horizon),
                    dtype=np.float64,
                )
                if prediction.shape != (rollout_horizon,) or not np.all(
                    np.isfinite(prediction)
                ):
                    raise ValueError(
                        "Early-warning predictor returned invalid or padded data"
                    )
                risk = float(np.max(prediction))
                maximum_risk = (
                    risk if maximum_risk is None else max(maximum_risk, risk)
                )
                if earliest is None and risk > parsed_threshold:
                    earliest = start_index
            lead_steps = (
                0 if earliest is None else int(event_index) - int(earliest)
            )
            lead_seconds = (
                0.0
                if earliest is None
                else float(np.sum(durations[earliest:event_index]))
            )
            by_lookback[str(lookback)] = {
                "earliest_warning_index": (
                    None if earliest is None else int(earliest)
                ),
                "earliest_warning_episode_step": (
                    None if earliest is None else int(steps[earliest])
                ),
                "lead_steps": int(lead_steps),
                "lead_seconds": float(lead_seconds),
                "missed": earliest is None,
                "max_risk_before_event": maximum_risk,
            }
        event_records.append(
            {
                "episode_id": (
                    identifiers[event_index].item()
                    if isinstance(identifiers[event_index], np.generic)
                    else identifiers[event_index]
                ),
                "event_index": int(event_index),
                "event_step": int(steps[event_index]),
                "reason_id": int(reasons[event_index]),
                "reason": TRANSLATION_BLOCK_REASON_NAMES[int(reasons[event_index])],
                "immediate_prediction": float(immediate_prediction[0]),
                "immediate_detection": bool(
                    immediate_prediction[0] > parsed_threshold
                ),
                "by_lookback": by_lookback,
            }
        )
    return {"events": event_records}


def _strict_json_value(value: Any, path: str = "$") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, np.generic):
        return _strict_json_value(value.item(), path)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite JSON number at {path}")
        return value
    if isinstance(value, np.ndarray):
        return [
            _strict_json_value(item, f"{path}[{index}]")
            for index, item in enumerate(value.tolist())
        ]
    if isinstance(value, Mapping):
        return {
            str(key): _strict_json_value(item, f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _strict_json_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"Unsupported JSON value {type(value).__name__} at {path}")


def strict_json_dumps(value: Any) -> str:
    """Serialize JSON with finite numbers only; unavailable metrics use null."""

    return json.dumps(
        _strict_json_value(value),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )


def write_strict_json(path: Path, value: Any) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(strict_json_dumps(value) + "\n", encoding="utf-8")


def _finite_metric(value: float) -> Optional[float]:
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _mean(value: np.ndarray) -> Optional[float]:
    return None if value.size == 0 else _finite_metric(np.mean(value))


def _std(value: np.ndarray) -> Optional[float]:
    return None if value.size == 0 else _finite_metric(np.std(value))


def _pearson(target: np.ndarray, prediction: np.ndarray) -> Optional[float]:
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if (
        target.size < 2
        or np.std(target) == 0.0
        or np.std(prediction) == 0.0
    ):
        return None
    return _finite_metric(np.corrcoef(target, prediction)[0, 1])


def _regression_metrics(
    target: Any,
    prediction: Any,
) -> Dict[str, Any]:
    target_array = _as_finite_float_array(
        target, name="regression target"
    ).astype(np.float64, copy=False).reshape(-1)
    prediction_array = _as_finite_float_array(
        prediction, name="regression prediction"
    ).astype(np.float64, copy=False).reshape(-1)
    if target_array.shape != prediction_array.shape:
        raise ValueError("Regression target/prediction shapes do not match")
    error = prediction_array - target_array
    return {
        "count": int(target_array.size),
        "mae": _finite_metric(np.mean(np.abs(error))),
        "rmse": _finite_metric(np.sqrt(np.mean(np.square(error)))),
        "bias_prediction_minus_target": _finite_metric(np.mean(error)),
        "pearson": _pearson(target_array, prediction_array),
        "target_mean": _finite_metric(np.mean(target_array)),
        "target_std": _finite_metric(np.std(target_array)),
        "target_max": _finite_metric(np.max(target_array)),
        "prediction_mean": _finite_metric(np.mean(prediction_array)),
        "prediction_std": _finite_metric(np.std(prediction_array)),
        "prediction_max": _finite_metric(np.max(prediction_array)),
    }


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positive_count = int(np.count_nonzero(labels))
    negative_count = int(labels.size - positive_count)
    if positive_count == 0 or negative_count == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        stop = start + 1
        while stop < scores.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    rank_sum = float(np.sum(ranks[labels]))
    auc = (
        rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)
    return _finite_metric(auc)


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positive_count = int(np.count_nonzero(labels))
    if positive_count == 0:
        return None
    # Group equal scores so the result does not depend on within-tie ordering.
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    true_positive = 0
    false_positive = 0
    previous_recall = 0.0
    area = 0.0
    start = 0
    while start < labels.size:
        stop = start + 1
        while stop < labels.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        true_positive += int(np.count_nonzero(sorted_labels[start:stop]))
        false_positive += int(stop - start) - int(
            np.count_nonzero(sorted_labels[start:stop])
        )
        recall = true_positive / positive_count
        precision = true_positive / max(true_positive + false_positive, 1)
        area += (recall - previous_recall) * precision
        previous_recall = recall
        start = stop
    return _finite_metric(area)


def _classification_metrics(
    labels: Any,
    scores: Any,
    threshold: float,
) -> Dict[str, Any]:
    label_array = np.asarray(labels, dtype=bool).reshape(-1)
    score_array = _as_finite_float_array(
        scores, name="classification scores"
    ).astype(np.float64, copy=False).reshape(-1)
    if label_array.shape != score_array.shape or label_array.size == 0:
        raise ValueError("Classification labels/scores must be nonempty and aligned")
    prediction = score_array > float(threshold)
    tp = int(np.count_nonzero(label_array & prediction))
    fp = int(np.count_nonzero(~label_array & prediction))
    tn = int(np.count_nonzero(~label_array & ~prediction))
    fn = int(np.count_nonzero(label_array & ~prediction))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else None
    fpr = fp / (fp + tn) if fp + tn else None
    f1 = (
        None
        if recall is None
        else (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0.0
            else 0.0
        )
    )
    tnr = None if fpr is None else 1.0 - fpr
    balanced = (
        None if recall is None or tnr is None else 0.5 * (recall + tnr)
    )
    return {
        "count": int(label_array.size),
        "positive_count": int(np.count_nonzero(label_array)),
        "negative_count": int(np.count_nonzero(~label_array)),
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": float(precision),
        "recall": None if recall is None else float(recall),
        "f1": None if f1 is None else float(f1),
        "fpr": None if fpr is None else float(fpr),
        "balanced_accuracy": (
            None if balanced is None else float(balanced)
        ),
        "roc_auc": _roc_auc(label_array, score_array),
        "pr_auc": _average_precision(label_array, score_array),
    }


def _masked_mae(
    target: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, Any]:
    count = int(np.count_nonzero(mask))
    return {
        "count": count,
        "mae": (
            None
            if count == 0
            else _finite_metric(
                np.mean(np.abs(prediction[mask] - target[mask]))
            )
        ),
    }


def _teacher_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    reason_ids: np.ndarray,
    stratum_ids: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    if target.shape != prediction.shape or target.shape[1] != 2:
        raise ValueError("Teacher target/prediction must have shape (N, 2)")
    curvature_target = target[:, CURVATURE_INDEX]
    curvature_prediction = prediction[:, CURVATURE_INDEX]
    translation_target = target[:, TRANSLATION_INDEX]
    translation_prediction = prediction[:, TRANSLATION_INDEX]
    curvature = _regression_metrics(curvature_target, curvature_prediction)
    curvature["strata"] = {
        name: _masked_mae(
            curvature_target,
            curvature_prediction,
            stratum_ids == index,
        )
        for index, name in enumerate(CURVATURE_STRATUM_NAMES)
    }
    translation = _regression_metrics(
        translation_target, translation_prediction
    )
    translation["reason_groups"] = {
        name: _masked_mae(
            translation_target,
            translation_prediction,
            reason_ids == TRANSLATION_BLOCK_REASON_NAMES.index(name),
        )
        for name in (
            "none",
            "lower_insertion_boundary",
            "vessel_tree_end",
        )
    }
    translation["classification"] = _classification_metrics(
        translation_target > 0.0,
        translation_prediction,
        threshold,
    )
    return {
        "transition_count": int(target.shape[0]),
        "curvature": curvature,
        "translation": translation,
    }


def _offset_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    stratum_ids: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    curvature = _regression_metrics(
        target[:, CURVATURE_INDEX], prediction[:, CURVATURE_INDEX]
    )
    curvature_target = target[:, CURVATURE_INDEX]
    curvature_prediction = prediction[:, CURVATURE_INDEX]
    curvature.update(
        {
            "high_stratum": _masked_mae(
                curvature_target,
                curvature_prediction,
                stratum_ids == CURVATURE_STRATUM_NAMES.index("high"),
            ),
            "extreme_stratum": _masked_mae(
                curvature_target,
                curvature_prediction,
                stratum_ids == CURVATURE_STRATUM_NAMES.index("extreme"),
            ),
            "prediction_max_to_target_max_ratio": (
                None
                if np.max(curvature_target) == 0.0
                else _finite_metric(
                    np.max(curvature_prediction) / np.max(curvature_target)
                )
            ),
        }
    )
    translation = _regression_metrics(
        target[:, TRANSLATION_INDEX], prediction[:, TRANSLATION_INDEX]
    )
    translation["classification"] = _classification_metrics(
        target[:, TRANSLATION_INDEX] > 0.0,
        prediction[:, TRANSLATION_INDEX],
        threshold,
    )
    return {
        "curvature": curvature,
        "translation": translation,
    }


def _future_event_metrics(
    true_translation: np.ndarray,
    predicted_translation: np.ndarray,
    discount: float,
    threshold: float,
) -> Dict[str, Any]:
    signals = aggregate_future_event_targets_and_scores(
        true_translation,
        predicted_translation,
        discount,
    )
    labels = np.asarray(signals["true_event"], dtype=bool)
    max_scores = np.asarray(signals["risk_max"], dtype=np.float64)
    sum_scores = np.asarray(signals["risk_sum"], dtype=np.float64)
    return {
        "window_count": int(labels.size),
        "blockage_event_window_count": int(np.count_nonzero(labels)),
        "max_risk": _classification_metrics(labels, max_scores, threshold),
        "discounted_sum_ranking": {
            "discount": float(discount),
            "roc_auc": _roc_auc(labels, sum_scores),
            "pr_auc": _average_precision(labels, sum_scores),
            "target_positive_mean": _mean(sum_scores[labels]),
            "target_negative_mean": _mean(sum_scores[~labels]),
        },
        "score_summary": {
            "max_mean": _mean(max_scores),
            "max_max": (
                None if max_scores.size == 0 else float(np.max(max_scores))
            ),
            "sum_mean": _mean(sum_scores),
            "sum_max": (
                None if sum_scores.size == 0 else float(np.max(sum_scores))
            ),
        },
    }


def _curvature_horizon_metrics(
    true_curvature: np.ndarray,
    predicted_curvature: np.ndarray,
    boundaries: Sequence[float],
) -> Dict[str, Any]:
    true_max = np.max(true_curvature, axis=1)
    prediction_max = np.max(predicted_curvature, axis=1)
    regression = _regression_metrics(true_max, prediction_max)
    underpredicted = prediction_max < true_max
    positive = true_max > 0.0
    severe = positive & (prediction_max < 0.75 * true_max)
    high_threshold = float(boundaries[1])
    extreme_threshold = float(boundaries[2])
    return {
        "horizon_maximum": regression,
        "underprediction_bias_true_minus_prediction": _finite_metric(
            np.mean(true_max - prediction_max)
        ),
        "underprediction_fraction": float(np.mean(underpredicted)),
        "severe_underprediction_definition": (
            "predicted_max < 0.75 * true_max for true_max > 0"
        ),
        "severe_underprediction_positive_count": int(np.count_nonzero(positive)),
        "severe_underprediction_fraction": (
            None
            if not np.any(positive)
            else float(np.mean(severe[positive]))
        ),
        "high_within_h": _classification_metrics(
            true_max >= high_threshold,
            prediction_max,
            high_threshold,
        ),
        "extreme_within_h": _classification_metrics(
            true_max >= extreme_threshold,
            prediction_max,
            extreme_threshold,
        ),
    }


def _false_warning_metrics(
    window_indices: np.ndarray,
    episode_ids: np.ndarray,
    true_translation: np.ndarray,
    predicted_translation: np.ndarray,
    threshold: float,
    environment_step_count: int,
) -> Dict[str, Any]:
    labels = np.any(true_translation > 0.0, axis=1)
    scores = np.max(predicted_translation, axis=1)
    negative = ~labels
    false_warning = negative & (scores > threshold)
    runs: List[int] = []
    current_run = 0
    previous_start: Optional[int] = None
    previous_episode: Any = None
    for row, is_false in enumerate(false_warning):
        start = int(window_indices[row, 0])
        episode = episode_ids[start]
        consecutive = (
            previous_start is not None
            and start == previous_start + 1
            and episode == previous_episode
        )
        if bool(is_false):
            if not consecutive:
                if current_run:
                    runs.append(current_run)
                current_run = 0
            current_run += 1
        elif current_run:
            runs.append(current_run)
            current_run = 0
        previous_start = start
        previous_episode = episode
    if current_run:
        runs.append(current_run)
    false_count = int(np.count_nonzero(false_warning))
    negative_count = int(np.count_nonzero(negative))
    return {
        "negative_window_count": negative_count,
        "false_warning_window_count": false_count,
        "false_warning_fraction": (
            None if negative_count == 0 else false_count / negative_count
        ),
        "false_warnings_per_100_environment_steps": (
            None
            if environment_step_count <= 0
            else 100.0 * false_count / environment_step_count
        ),
        "consecutive_run_count": len(runs),
        "mean_consecutive_run_duration_steps": (
            None if not runs else float(np.mean(runs))
        ),
        "maximum_consecutive_run_duration_steps": (
            0 if not runs else int(max(runs))
        ),
    }


def _metric_gap(
    teacher: Optional[float],
    opened: Optional[float],
    *,
    order: str,
) -> Optional[float]:
    if teacher is None or opened is None:
        return None
    if order == "open_minus_teacher":
        return float(opened - teacher)
    if order == "teacher_minus_open":
        return float(teacher - opened)
    raise ValueError(f"Unknown gap order {order}")


def _all_numeric_gaps(
    teacher: Any,
    opened: Any,
) -> Any:
    """Return an exhaustive ``open_loop - teacher_forced`` metric gap tree."""

    if isinstance(teacher, Mapping) and isinstance(opened, Mapping):
        return {
            str(key): _all_numeric_gaps(teacher[key], opened[key])
            for key in teacher
            if key in opened
        }
    if isinstance(teacher, (list, tuple)) and isinstance(opened, (list, tuple)):
        if len(teacher) != len(opened):
            return None
        return [
            _all_numeric_gaps(teacher_item, open_item)
            for teacher_item, open_item in zip(teacher, opened)
        ]
    if teacher is None or opened is None:
        return None
    if isinstance(teacher, (bool, np.bool_)) or isinstance(
        opened, (bool, np.bool_)
    ):
        return None
    if isinstance(teacher, (int, float, np.number)) and isinstance(
        opened, (int, float, np.number)
    ):
        teacher_value = float(teacher)
        open_value = float(opened)
        if math.isfinite(teacher_value) and math.isfinite(open_value):
            return open_value - teacher_value
    return None


def _teacher_open_gap(
    teacher_offsets: Sequence[Mapping[str, Any]],
    open_offsets: Sequence[Mapping[str, Any]],
    teacher_event: Mapping[str, Any],
    open_event: Mapping[str, Any],
    teacher_curvature: Mapping[str, Any],
    open_curvature: Mapping[str, Any],
) -> Dict[str, Any]:
    offset_records = []
    for offset, (teacher, opened) in enumerate(
        zip(teacher_offsets, open_offsets)
    ):
        teacher_translation = teacher["translation"]
        open_translation = opened["translation"]
        teacher_class = teacher_translation["classification"]
        open_class = open_translation["classification"]
        offset_records.append(
            {
                "offset": offset,
                "translation_mae_increase": _metric_gap(
                    teacher_translation["mae"],
                    open_translation["mae"],
                    order="open_minus_teacher",
                ),
                "translation_f1_decrease": _metric_gap(
                    teacher_class["f1"],
                    open_class["f1"],
                    order="teacher_minus_open",
                ),
                "translation_recall_decrease": _metric_gap(
                    teacher_class["recall"],
                    open_class["recall"],
                    order="teacher_minus_open",
                ),
                "translation_fpr_increase": _metric_gap(
                    teacher_class["fpr"],
                    open_class["fpr"],
                    order="open_minus_teacher",
                ),
                "curvature_mae_increase": _metric_gap(
                    teacher["curvature"]["mae"],
                    opened["curvature"]["mae"],
                    order="open_minus_teacher",
                ),
            }
        )
    teacher_event_class = teacher_event["max_risk"]
    open_event_class = open_event["max_risk"]
    return {
        "all_numeric_gap_convention": "open_loop_minus_teacher_forced",
        "all_numeric_by_offset": [
            _all_numeric_gaps(teacher, opened)
            for teacher, opened in zip(teacher_offsets, open_offsets)
        ],
        "all_numeric_future_blockage": _all_numeric_gaps(
            teacher_event, open_event
        ),
        "all_numeric_curvature_horizon_risk": _all_numeric_gaps(
            teacher_curvature, open_curvature
        ),
        "by_offset": offset_records,
        "future_event": {
            "f1_decrease": _metric_gap(
                teacher_event_class["f1"],
                open_event_class["f1"],
                order="teacher_minus_open",
            ),
            "recall_decrease": _metric_gap(
                teacher_event_class["recall"],
                open_event_class["recall"],
                order="teacher_minus_open",
            ),
            "fpr_increase": _metric_gap(
                teacher_event_class["fpr"],
                open_event_class["fpr"],
                order="open_minus_teacher",
            ),
        },
        "curvature_horizon_maximum_error_increase": _metric_gap(
            teacher_curvature["horizon_maximum"]["mae"],
            open_curvature["horizon_maximum"]["mae"],
            order="open_minus_teacher",
        ),
    }


class TrajectoryRecorder:
    """In-memory sequential transition recorder, never connected to replay."""

    _ARRAY_FIELDS = (
        "observation",
        "next_observation",
        "command_action",
        "requested_action",
        "applied_action",
        "safety_cost",
        "block_reason_id",
        "block_reason",
        "curvature_stratum_id",
        "curvature_stratum",
        "reward",
        "terminated",
        "truncated",
        "episode_id",
        "episode_step",
        "environment_seed",
        "mode",
        "variant",
        "actual_step_duration_s",
        "simulation_error",
    )

    def __init__(self, observation_dim: int, action_dim: int) -> None:
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self._records: List[Dict[str, Any]] = []

    def add(
        self,
        *,
        observation: np.ndarray,
        next_observation: np.ndarray,
        command_action: np.ndarray,
        info: Mapping[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
        episode_id: str,
        environment_seed: int,
        mode: str,
        variant: str,
        curvature_boundaries: Sequence[float],
    ) -> None:
        observation_array = np.asarray(observation, dtype=np.float32)
        next_array = np.asarray(next_observation, dtype=np.float32)
        command = np.asarray(command_action, dtype=np.float32)
        requested = np.asarray(info["requested_action"], dtype=np.float32)
        applied = np.asarray(info["applied_action"], dtype=np.float32)
        if observation_array.shape != (self.observation_dim,) or (
            next_array.shape != (self.observation_dim,)
        ):
            raise ValueError("Recorded observation shape does not match checkpoint")
        for value, name in (
            (command, "command action"),
            (requested, "requested action"),
            (applied, "applied action"),
        ):
            if value.shape != (self.action_dim,) or not np.all(
                np.isfinite(value)
            ):
                raise ValueError(f"Recorded {name} is invalid")
        if not np.allclose(command, requested, rtol=0.0, atol=1e-6):
            raise RuntimeError(
                "Environment requested action differs from the already-selected "
                "normalized command"
            )
        safety_cost = validate_collected_safety_cost(
            info.get("safety_cost"),
            safety_cost_names=SAFETY_COST_NAMES,
            source=f"{episode_id} step {info.get('episode_step')} safety_cost",
        )
        metrics = info.get("safety_metrics")
        reason_id, stratum_id = safety_aux_metadata_from_metrics(
            metrics, curvature_boundaries
        )
        duration = float(metrics["actual_simulation_time_s"])
        if (
            not np.all(np.isfinite(observation_array))
            or not np.all(np.isfinite(next_array))
            or not np.isfinite(float(reward))
            or not np.isfinite(duration)
            or duration < 0.0
        ):
            raise FloatingPointError("Recorded transition contains non-finite data")
        self._records.append(
            {
                "observation": observation_array.copy(),
                "next_observation": next_array.copy(),
                "command_action": command.copy(),
                "requested_action": requested.copy(),
                "applied_action": applied.copy(),
                "safety_cost": safety_cost.copy(),
                "block_reason_id": int(reason_id),
                "block_reason": TRANSLATION_BLOCK_REASON_NAMES[reason_id],
                "curvature_stratum_id": int(stratum_id),
                "curvature_stratum": CURVATURE_STRATUM_NAMES[stratum_id],
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "episode_id": str(episode_id),
                "episode_step": int(info["episode_step"]),
                "environment_seed": int(environment_seed),
                "mode": str(mode),
                "variant": str(variant),
                "actual_step_duration_s": duration,
                "simulation_error": bool(info.get("simulation_error", False)),
            }
        )

    @property
    def transition_count(self) -> int:
        return len(self._records)

    def arrays(self) -> Dict[str, np.ndarray]:
        if not self._records:
            raise RuntimeError("Trajectory recorder contains no transitions")
        arrays: Dict[str, np.ndarray] = {}
        for field in self._ARRAY_FIELDS:
            values = [record[field] for record in self._records]
            if field in (
                "observation",
                "next_observation",
                "command_action",
                "requested_action",
                "applied_action",
                "safety_cost",
            ):
                arrays[field] = np.stack(values).astype(np.float32, copy=False)
            elif field in (
                "block_reason_id",
                "curvature_stratum_id",
                "episode_step",
                "environment_seed",
            ):
                arrays[field] = np.asarray(values, dtype=np.int64)
            elif field in (
                "terminated",
                "truncated",
                "simulation_error",
            ):
                arrays[field] = np.asarray(values, dtype=np.bool_)
            elif field in ("reward", "actual_step_duration_s"):
                arrays[field] = np.asarray(values, dtype=np.float64)
            else:
                arrays[field] = np.asarray(values, dtype=np.str_)
        validate_trajectory_arrays(
            arrays,
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
        )
        return arrays


def validate_trajectory_arrays(
    data: Mapping[str, np.ndarray],
    *,
    observation_dim: int,
    action_dim: int,
) -> None:
    missing = set(TrajectoryRecorder._ARRAY_FIELDS) - set(data)
    if missing:
        raise KeyError(f"Trajectory data is missing fields {sorted(missing)}")
    count = int(np.asarray(data["episode_id"]).shape[0])
    if count <= 0:
        raise ValueError("Trajectory data must contain transitions")
    for field in TrajectoryRecorder._ARRAY_FIELDS:
        if np.asarray(data[field]).shape[0] != count:
            raise ValueError(f"Trajectory field {field} has inconsistent length")
    expected_shapes = {
        "observation": (count, observation_dim),
        "next_observation": (count, observation_dim),
        "command_action": (count, action_dim),
        "requested_action": (count, action_dim),
        "applied_action": (count, action_dim),
        "safety_cost": (count, len(SAFETY_COST_NAMES)),
    }
    for field, shape in expected_shapes.items():
        value = np.asarray(data[field])
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise ValueError(
                f"Trajectory field {field} must have finite shape {shape}, "
                f"got {value.shape}"
            )
    if np.any(np.asarray(data["safety_cost"]) < 0.0):
        raise ValueError("Trajectory safety costs must be nonnegative")
    if not np.allclose(
        data["command_action"],
        data["requested_action"],
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("Trajectory commands/requested actions are misaligned")
    identifiers = np.asarray(data["episode_id"])
    steps = np.asarray(data["episode_step"])
    seen = set()
    previous: Optional[str] = None
    expected_step = 1
    for index, identifier_value in enumerate(identifiers):
        identifier = str(identifier_value)
        if identifier != previous:
            if identifier in seen:
                raise ValueError(
                    f"Episode {identifier!r} is not stored in one contiguous block"
                )
            seen.add(identifier)
            previous = identifier
            expected_step = 1
        if int(steps[index]) != expected_step:
            raise ValueError(
                f"Episode {identifier!r} expected step {expected_step}, "
                f"got {steps[index]}"
            )
        expected_step += 1
    for field in ("reward", "actual_step_duration_s"):
        if not np.all(np.isfinite(np.asarray(data[field]))):
            raise FloatingPointError(f"Trajectory field {field} is non-finite")


def _merge_trajectory_arrays(
    datasets: Sequence[Mapping[str, np.ndarray]],
    *,
    observation_dim: int,
    action_dim: int,
) -> Dict[str, np.ndarray]:
    if not datasets:
        raise ValueError("At least one trajectory dataset is required")
    merged = {
        field: np.concatenate(
            [np.asarray(dataset[field]) for dataset in datasets], axis=0
        )
        for field in TrajectoryRecorder._ARRAY_FIELDS
    }
    validate_trajectory_arrays(
        merged,
        observation_dim=observation_dim,
        action_dim=action_dim,
    )
    return merged


def _step_and_record(
    env: StEVEEnv,
    recorder: TrajectoryRecorder,
    observation: np.ndarray,
    action: np.ndarray,
    *,
    episode_id: str,
    seed: int,
    mode: str,
    variant: str,
    curvature_boundaries: Sequence[float],
) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
    selected_action = np.asarray(action, dtype=np.float32).copy()
    result = env.step(selected_action)
    recorder.add(
        observation=observation,
        next_observation=result[0],
        command_action=selected_action,
        info=result[4],
        reward=result[1],
        terminated=result[2],
        truncated=result[3],
        episode_id=episode_id,
        environment_seed=seed,
        mode=mode,
        variant=variant,
        curvature_boundaries=curvature_boundaries,
    )
    return result


def _collect_policy_trajectories(
    env: StEVEEnv,
    agent: Any,
    recorder: TrajectoryRecorder,
    *,
    checkpoint_label: str,
    episodes: int,
    base_seed: int,
    curvature_boundaries: Sequence[float],
) -> Dict[str, Any]:
    records = []
    for episode_index in range(int(episodes)):
        seed = int(base_seed + episode_index)
        set_seed(seed)
        observation, _ = env.reset(seed=seed)
        terminated = truncated = False
        length = 0
        reward_sum = 0.0
        episode_id = f"policy:{checkpoint_label}:{seed}"
        success = False
        while not (terminated or truncated):
            # This is the final action selected by the unchanged planner.  No
            # Safety rollout or Safety prediction is run until collection of
            # every real trajectory has completed.
            action = np.asarray(
                agent.act(
                    observation,
                    first_step=length == 0,
                    eval_mode=True,
                ),
                dtype=np.float32,
            ).copy()
            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = _step_and_record(
                env,
                recorder,
                observation,
                action,
                episode_id=episode_id,
                seed=seed,
                mode="policy",
                variant="policy_action",
                curvature_boundaries=curvature_boundaries,
            )
            length += 1
            reward_sum += float(reward)
            success = success or bool(info.get("is_success", False))
        records.append(
            {
                "episode_id": episode_id,
                "seed": seed,
                "length": length,
                "reward": reward_sum,
                "success": success,
            }
        )
        print(
            f"collect policy checkpoint={checkpoint_label} "
            f"episode={episode_index + 1}/{episodes} seed={seed} length={length}"
        )
    return {"episodes": records}


def _collect_random_trajectories(
    env: StEVEEnv,
    recorder: TrajectoryRecorder,
    *,
    episodes: int,
    base_seed: int,
    curvature_boundaries: Sequence[float],
) -> Dict[str, Any]:
    records = []
    for episode_index in range(int(episodes)):
        seed = int(base_seed + episode_index)
        rng = np.random.default_rng(seed)
        observation, _ = env.reset(seed=seed)
        terminated = truncated = False
        length = 0
        episode_id = f"random:{seed}"
        while not (terminated or truncated):
            action = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
            (
                observation,
                _,
                terminated,
                truncated,
                _,
            ) = _step_and_record(
                env,
                recorder,
                observation,
                action,
                episode_id=episode_id,
                seed=seed,
                mode="random",
                variant="uniform_random_action",
                curvature_boundaries=curvature_boundaries,
            )
            length += 1
        records.append(
            {"episode_id": episode_id, "seed": seed, "length": length}
        )
        print(
            f"collect random episode={episode_index + 1}/{episodes} "
            f"seed={seed} length={length}"
        )
    return {"episodes": records}


def _collect_lower_boundary_trajectories(
    env: StEVEEnv,
    recorder: TrajectoryRecorder,
    *,
    repetitions: int,
    base_seed: int,
    curvature_boundaries: Sequence[float],
) -> Dict[str, Any]:
    # The approach first moves away from zero insertion, returns near the
    # boundary through several real transitions, runs zero/forward controls,
    # and only then executes moderate and maximum blocked retractions.
    plan = (
        ("pre_zero", np.asarray([0.0, 0.0], dtype=np.float32), "none"),
        ("pre_forward", np.asarray([0.5, 0.0], dtype=np.float32), "none"),
        ("approach_retract_1", np.asarray([-0.2, 0.0], dtype=np.float32), "none"),
        ("approach_retract_2", np.asarray([-0.2, 0.0], dtype=np.float32), "none"),
        ("near_boundary_zero", np.asarray([0.0, 0.0], dtype=np.float32), "none"),
        ("near_boundary_forward", np.asarray([0.2, 0.0], dtype=np.float32), "none"),
        ("return_near_boundary", np.asarray([-0.2, 0.0], dtype=np.float32), "none"),
        (
            "blocked_retraction_moderate",
            np.asarray([-0.5, 0.0], dtype=np.float32),
            "lower_insertion_boundary",
        ),
        (
            "blocked_retraction_maximum",
            np.asarray([-1.0, 0.0], dtype=np.float32),
            "lower_insertion_boundary",
        ),
        ("post_block_zero", np.asarray([0.0, 0.0], dtype=np.float32), "none"),
    )
    constructions = []
    for repetition in range(int(repetitions)):
        seed = int(base_seed + repetition)
        observation, _ = env.reset(seed=seed)
        terminated = truncated = False
        episode_id = f"lower_boundary:{seed}"
        mismatches = []
        for variant, action, expected_reason in plan:
            if terminated or truncated:
                break
            (
                observation,
                _,
                terminated,
                truncated,
                info,
            ) = _step_and_record(
                env,
                recorder,
                observation,
                action,
                episode_id=episode_id,
                seed=seed,
                mode="lower_boundary",
                variant=variant,
                curvature_boundaries=curvature_boundaries,
            )
            actual_reason = str(
                info["safety_metrics"]["translation_block_reason"]
            )
            if actual_reason != expected_reason:
                mismatches.append(
                    {
                        "variant": variant,
                        "expected": expected_reason,
                        "actual": actual_reason,
                    }
                )
        if mismatches or terminated or truncated:
            raise RuntimeError(
                f"Lower-boundary construction failed for seed {seed}: "
                f"mismatches={mismatches}, terminated={terminated}, "
                f"truncated={truncated}"
            )
        constructions.append(
            {
                "episode_id": episode_id,
                "seed": seed,
                "transition_count": len(plan),
                "matched": True,
            }
        )
        print(
            f"collect lower-boundary sequence={repetition + 1}/"
            f"{repetitions} seed={seed}"
        )
    return {"successful_sequences": constructions}


def _collect_tree_end_trajectories(
    env: StEVEEnv,
    recorder: TrajectoryRecorder,
    *,
    repetitions: int,
    base_seed: int,
    curvature_boundaries: Sequence[float],
) -> Dict[str, Any]:
    approach_action = np.asarray([1.0, 0.0], dtype=np.float32)
    controls = (
        ("tree_end_zero", np.asarray([0.0, 0.0], dtype=np.float32), "none"),
        (
            "blocked_forward_moderate",
            np.asarray([0.5, 0.0], dtype=np.float32),
            "vessel_tree_end",
        ),
        (
            "blocked_forward_maximum",
            np.asarray([1.0, 0.0], dtype=np.float32),
            "vessel_tree_end",
        ),
        (
            "tree_end_rotation_only",
            np.asarray([0.0, 0.5], dtype=np.float32),
            "none",
        ),
        (
            "tree_end_retraction",
            np.asarray([-0.5, 0.0], dtype=np.float32),
            "none",
        ),
    )
    successes = []
    failures = []
    attempt = 0
    maximum_attempts = max(int(repetitions) * 20, int(repetitions))
    while len(successes) < int(repetitions) and attempt < maximum_attempts:
        seed = int(base_seed + attempt)
        attempt += 1
        observation, _ = env.reset(seed=seed)
        terminated = truncated = False
        approach_steps = 0
        episode_id = f"tree_end:{seed}"
        while not _at_tree_end(env) and not (terminated or truncated):
            (
                observation,
                _,
                terminated,
                truncated,
                _,
            ) = _step_and_record(
                env,
                recorder,
                observation,
                approach_action,
                episode_id=episode_id,
                seed=seed,
                mode="tree_end",
                variant="tree_end_approach",
                curvature_boundaries=curvature_boundaries,
            )
            approach_steps += 1
        if terminated or truncated or not _at_tree_end(env):
            failures.append(
                {
                    "episode_id": episode_id,
                    "seed": seed,
                    "approach_steps": approach_steps,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
            print(
                f"collect tree-end candidate seed={seed} ended before endpoint "
                f"after {approach_steps} steps"
            )
            continue
        mismatches = []
        for variant, action, expected_reason in controls:
            (
                observation,
                _,
                terminated,
                truncated,
                info,
            ) = _step_and_record(
                env,
                recorder,
                observation,
                action,
                episode_id=episode_id,
                seed=seed,
                mode="tree_end",
                variant=variant,
                curvature_boundaries=curvature_boundaries,
            )
            actual_reason = str(
                info["safety_metrics"]["translation_block_reason"]
            )
            if actual_reason != expected_reason:
                mismatches.append(
                    {
                        "variant": variant,
                        "expected": expected_reason,
                        "actual": actual_reason,
                    }
                )
            if terminated or truncated:
                break
        if mismatches or terminated or truncated:
            failures.append(
                {
                    "episode_id": episode_id,
                    "seed": seed,
                    "approach_steps": approach_steps,
                    "mismatches": mismatches,
                    "ended_during_controls": True,
                }
            )
            continue
        successes.append(
            {
                "episode_id": episode_id,
                "seed": seed,
                "approach_steps": approach_steps,
                "matched": True,
            }
        )
        print(
            f"collect tree-end sequence={len(successes)}/{repetitions} "
            f"seed={seed} approach_steps={approach_steps}"
        )
    if len(successes) != int(repetitions):
        raise RuntimeError(
            f"Collected only {len(successes)}/{repetitions} successful tree-end "
            f"sequences after {attempt} disjoint seeds"
        )
    return {
        "successful_sequences": successes,
        "failed_approach_sequences": failures,
        "attempt_count": attempt,
    }


def _trajectory_summary(data: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    modes = tuple(dict.fromkeys(str(value) for value in data["mode"]))
    return {
        "transition_count": int(data["episode_id"].shape[0]),
        "episode_count": int(np.unique(data["episode_id"]).size),
        "simulation_error_count": int(np.count_nonzero(data["simulation_error"])),
        "modes": {
            mode: {
                "transition_count": int(np.count_nonzero(data["mode"] == mode)),
                "episode_count": int(
                    np.unique(data["episode_id"][data["mode"] == mode]).size
                ),
                "environment_seeds": sorted(
                    int(value)
                    for value in np.unique(
                        data["environment_seed"][data["mode"] == mode]
                    )
                ),
                "translation_block_reasons": {
                    name: int(
                        np.count_nonzero(
                            (data["mode"] == mode)
                            & (
                                data["block_reason_id"]
                                == TRANSLATION_BLOCK_REASON_NAMES.index(name)
                            )
                        )
                    )
                    for name in TRANSLATION_BLOCK_REASON_NAMES
                },
                "curvature_strata": {
                    name: int(
                        np.count_nonzero(
                            (data["mode"] == mode)
                            & (data["curvature_stratum_id"] == index)
                        )
                    )
                    for index, name in enumerate(CURVATURE_STRATUM_NAMES)
                },
            }
            for mode in modes
        },
    }


def _save_trajectory_npz(
    destination: Path,
    data: Mapping[str, np.ndarray],
    *,
    checkpoint_path: Path,
    horizons: Sequence[int],
    seeds: Sequence[int],
) -> Path:
    path = destination.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(data)
    payload.update(
        {
            "safety_cost_names": np.asarray(SAFETY_COST_NAMES, dtype=np.str_),
            "translation_block_reason_names": np.asarray(
                TRANSLATION_BLOCK_REASON_NAMES, dtype=np.str_
            ),
            "curvature_stratum_names": np.asarray(
                CURVATURE_STRATUM_NAMES, dtype=np.str_
            ),
            "checkpoint_path": np.asarray(str(checkpoint_path), dtype=np.str_),
            "evaluation_horizons": np.asarray(horizons, dtype=np.int64),
            "seed_bases": np.asarray(seeds, dtype=np.int64),
            "temporal_alignment": np.asarray(
                "prediction[k]=safety(z_t+k,requested_action_t+k) "
                "versus safety_cost_t+k; prediction precedes dynamics",
                dtype=np.str_,
            ),
        }
    )
    np.savez_compressed(path, **payload)
    return path


def _aggregate_latent_drift(
    drift: Mapping[str, np.ndarray],
    mask: np.ndarray,
) -> List[Dict[str, Any]]:
    output = []
    for offset in range(drift["mse"].shape[1]):
        output.append(
            {
                "offset": offset,
                "count": int(np.count_nonzero(mask)),
                "mse": _mean(drift["mse"][mask, offset]),
                "l1": _mean(drift["l1"][mask, offset]),
                "cosine_similarity": _mean(
                    drift["cosine_similarity"][mask, offset]
                ),
            }
        )
    return output


def _evaluate_horizon(
    model: torch.nn.Module,
    device: torch.device,
    data: Mapping[str, np.ndarray],
    teacher_prediction: np.ndarray,
    teacher_latent: np.ndarray,
    *,
    horizon: int,
    threshold: float,
    discount: float,
    curvature_boundaries: Sequence[float],
) -> Dict[str, Any]:
    window_list = build_episode_windows(data["episode_id"], horizon)
    if not window_list:
        raise RuntimeError(
            f"No complete real trajectory windows exist for horizon {horizon}"
        )
    windows = np.stack(window_list)
    action_sequences = data["requested_action"][windows]
    opened = _batched_open_loop_rollout(
        model,
        data["observation"][windows[:, 0]],
        action_sequences,
        device,
    )
    open_prediction = opened["prediction"]
    open_latent = opened["latent"]
    teacher_window_prediction = teacher_prediction[windows]

    reference_prefix = teacher_latent[windows]
    with torch.no_grad():
        final_reference = model.encode(
            torch.as_tensor(
                data["next_observation"][windows[:, -1]],
                dtype=torch.float32,
                device=device,
            )
        )
    if not bool(torch.isfinite(final_reference).all()):
        raise FloatingPointError("Final encoded reference latent is non-finite")
    reference_latent = np.concatenate(
        (
            reference_prefix,
            final_reference.detach().cpu().numpy()[:, None, :],
        ),
        axis=1,
    )
    drift = compute_latent_drift(open_latent, reference_latent)
    target = data["safety_cost"][windows]
    strata = data["curvature_stratum_id"][windows]
    window_modes = data["mode"][windows[:, 0]]
    available_modes = tuple(dict.fromkeys(str(value) for value in window_modes))
    groups = ("all", *available_modes)
    group_reports: Dict[str, Any] = {}
    for mode in groups:
        mask = (
            np.ones(windows.shape[0], dtype=bool)
            if mode == "all"
            else window_modes == mode
        )
        if not np.any(mask):
            continue
        mode_target = target[mask]
        mode_teacher = teacher_window_prediction[mask]
        mode_open = open_prediction[mask]
        mode_strata = strata[mask]
        mode_windows = windows[mask]
        teacher_offsets = [
            _offset_metrics(
                mode_target[:, offset],
                mode_teacher[:, offset],
                mode_strata[:, offset],
                threshold,
            )
            for offset in range(horizon)
        ]
        open_offsets = [
            _offset_metrics(
                mode_target[:, offset],
                mode_open[:, offset],
                mode_strata[:, offset],
                threshold,
            )
            for offset in range(horizon)
        ]
        teacher_event = _future_event_metrics(
            mode_target[:, :, TRANSLATION_INDEX],
            mode_teacher[:, :, TRANSLATION_INDEX],
            discount,
            threshold,
        )
        open_event = _future_event_metrics(
            mode_target[:, :, TRANSLATION_INDEX],
            mode_open[:, :, TRANSLATION_INDEX],
            discount,
            threshold,
        )
        teacher_curvature = _curvature_horizon_metrics(
            mode_target[:, :, CURVATURE_INDEX],
            mode_teacher[:, :, CURVATURE_INDEX],
            curvature_boundaries,
        )
        open_curvature = _curvature_horizon_metrics(
            mode_target[:, :, CURVATURE_INDEX],
            mode_open[:, :, CURVATURE_INDEX],
            curvature_boundaries,
        )
        environment_step_count = (
            int(data["episode_id"].size)
            if mode == "all"
            else int(np.count_nonzero(data["mode"] == mode))
        )
        group_reports[mode] = {
            "valid_window_count": int(np.count_nonzero(mask)),
            "valid_predicted_transition_count": int(
                np.count_nonzero(mask) * horizon
            ),
            "windows_containing_blockage": int(
                np.count_nonzero(
                    np.any(
                        mode_target[:, :, TRANSLATION_INDEX] > 0.0,
                        axis=1,
                    )
                )
            ),
            "windows_containing_high_or_extreme_curvature": int(
                np.count_nonzero(np.any(mode_strata >= 2, axis=1))
            ),
            "teacher_forced_by_offset": teacher_offsets,
            "open_loop_by_offset": open_offsets,
            "latent_drift_by_offset": _aggregate_latent_drift(drift, mask),
            "future_blockage": {
                "teacher_forced": teacher_event,
                "open_loop": open_event,
            },
            "false_warnings": {
                "teacher_forced": _false_warning_metrics(
                    mode_windows,
                    data["episode_id"],
                    mode_target[:, :, TRANSLATION_INDEX],
                    mode_teacher[:, :, TRANSLATION_INDEX],
                    threshold,
                    environment_step_count,
                ),
                "open_loop": _false_warning_metrics(
                    mode_windows,
                    data["episode_id"],
                    mode_target[:, :, TRANSLATION_INDEX],
                    mode_open[:, :, TRANSLATION_INDEX],
                    threshold,
                    environment_step_count,
                ),
            },
            "curvature_horizon_risk": {
                "teacher_forced": teacher_curvature,
                "open_loop": open_curvature,
            },
            "teacher_open_gap": _teacher_open_gap(
                teacher_offsets,
                open_offsets,
                teacher_event,
                open_event,
                teacher_curvature,
                open_curvature,
            ),
        }
    return {
        "horizon": int(horizon),
        "no_terminal_padding": True,
        "groups": group_reports,
    }


def _aggregate_early_warning_records(
    records: Sequence[Mapping[str, Any]],
    lookbacks: Sequence[int],
) -> Dict[str, Any]:
    output: Dict[str, Any] = {
        "event_count": len(records),
        "immediate_detection_count": int(
            sum(bool(record["immediate_detection"]) for record in records)
        ),
        "immediate_detection_rate": (
            None
            if not records
            else float(
                np.mean(
                    [
                        bool(record["immediate_detection"])
                        for record in records
                    ]
                )
            )
        ),
        "by_lookback": {},
    }
    for lookback in lookbacks:
        entries = [
            record["by_lookback"][str(int(lookback))] for record in records
        ]
        detected = [entry for entry in entries if not entry["missed"]]
        leads = np.asarray(
            [entry["lead_steps"] for entry in detected], dtype=np.float64
        )
        seconds = np.asarray(
            [entry["lead_seconds"] for entry in detected], dtype=np.float64
        )
        event_count = len(entries)
        output["by_lookback"][str(int(lookback))] = {
            "event_count": event_count,
            "detected_event_count": len(detected),
            "detection_rate": (
                None if event_count == 0 else len(detected) / event_count
            ),
            "missed_event_count": event_count - len(detected),
            "lead_steps": {
                "mean": _mean(leads),
                "median": (
                    None
                    if leads.size == 0
                    else _finite_metric(np.median(leads))
                ),
                "minimum": (
                    None if leads.size == 0 else int(np.min(leads))
                ),
                "maximum": (
                    None if leads.size == 0 else int(np.max(leads))
                ),
            },
            "lead_seconds_actual_sofa_time": {
                "mean": _mean(seconds),
                "median": (
                    None
                    if seconds.size == 0
                    else _finite_metric(np.median(seconds))
                ),
                "minimum": (
                    None if seconds.size == 0 else float(np.min(seconds))
                ),
                "maximum": (
                    None if seconds.size == 0 else float(np.max(seconds))
                ),
            },
            "fraction_warned_at_least_1_step_early": (
                None
                if event_count == 0
                else sum(entry["lead_steps"] >= 1 for entry in entries)
                / event_count
            ),
            "fraction_warned_at_least_3_steps_early": (
                None
                if event_count == 0
                else sum(entry["lead_steps"] >= 3 for entry in entries)
                / event_count
            ),
            "fraction_warned_at_least_5_steps_early": (
                None
                if event_count == 0
                else sum(entry["lead_steps"] >= 5 for entry in entries)
                / event_count
            ),
        }
    return output


def _evaluate_early_warning_report(
    model: torch.nn.Module,
    device: torch.device,
    data: Mapping[str, np.ndarray],
    *,
    lookbacks: Sequence[int],
    threshold: float,
    nominal_step_duration_s: float,
) -> Dict[str, Any]:
    cache: Dict[Tuple[int, int], np.ndarray] = {}

    def predictor(start_index: int, horizon: int) -> np.ndarray:
        key = (int(start_index), int(horizon))
        if key not in cache:
            stop = start_index + horizon
            if stop > data["episode_id"].size:
                raise ValueError("Early-warning rollout exceeds trajectory")
            if np.unique(data["episode_id"][start_index:stop]).size != 1:
                raise ValueError("Early-warning rollout crosses episode boundary")
            rollout = open_loop_rollout(
                model,
                data["observation"][start_index],
                data["requested_action"][start_index:stop],
                device,
            )
            cache[key] = rollout["prediction"][:, TRANSLATION_INDEX]
        return cache[key].copy()

    result = evaluate_early_warnings(
        data["episode_id"],
        data["safety_cost"][:, TRANSLATION_INDEX],
        predictor,
        lookbacks,
        threshold,
        data["actual_step_duration_s"],
        reason_ids=data["block_reason_id"],
        episode_steps=data["episode_step"],
    )
    for record in result["events"]:
        index = int(record["event_index"])
        record["mode"] = str(data["mode"][index])
        record["variant"] = str(data["variant"][index])
        record["environment_seed"] = int(data["environment_seed"][index])
        for lookback in lookbacks:
            entry = record["by_lookback"][str(int(lookback))]
            entry["nominal_lead_seconds_at_image_frequency"] = (
                float(entry["lead_steps"]) * float(nominal_step_duration_s)
            )

    reason_reports = {}
    for reason in ("lower_insertion_boundary", "vessel_tree_end"):
        selected = [
            record for record in result["events"] if record["reason"] == reason
        ]
        reason_reports[reason] = _aggregate_early_warning_records(
            selected, lookbacks
        )
    mode_reports = {}
    for mode in tuple(dict.fromkeys(str(value) for value in data["mode"])):
        selected = [
            record for record in result["events"] if record["mode"] == mode
        ]
        mode_reports[mode] = _aggregate_early_warning_records(
            selected, lookbacks
        )
    return {
        "definition": (
            "A blockage event is the 0-to-positive onset of one contiguous "
            "blocked run. Warnings must occur on a real transition before "
            "that onset. The imagined rollout includes the future blocked "
            "action; an immediate-only prediction is reported separately."
        ),
        "literal_lookback_note": (
            "A literal L-step warning requires an internal L+1-transition "
            "rollout so that the event transition is included."
        ),
        "open_loop_only": True,
        "events": result["events"],
        "by_blockage_reason": reason_reports,
        "by_trajectory_mode": mode_reports,
    }


def _evaluate_checkpoint(
    model: torch.nn.Module,
    device: torch.device,
    data: Mapping[str, np.ndarray],
    *,
    horizons: Sequence[int],
    lookbacks: Sequence[int],
    threshold: float,
    discount: float,
    curvature_boundaries: Sequence[float],
    nominal_step_duration_s: float,
) -> Dict[str, Any]:
    teacher = teacher_forced_rollout(
        model,
        data["observation"],
        data["requested_action"],
        device,
    )
    teacher_prediction = teacher["prediction"]
    teacher_latent = teacher["latent"]
    teacher_by_mode: Dict[str, Any] = {}
    modes = tuple(dict.fromkeys(str(value) for value in data["mode"]))
    for mode in ("all", *modes):
        mask = (
            np.ones(data["episode_id"].shape[0], dtype=bool)
            if mode == "all"
            else data["mode"] == mode
        )
        teacher_by_mode[mode] = _teacher_metrics(
            data["safety_cost"][mask],
            teacher_prediction[mask],
            data["block_reason_id"][mask],
            data["curvature_stratum_id"][mask],
            threshold,
        )
    horizon_reports = {}
    for horizon in horizons:
        print(f"evaluate latent rollout horizon={horizon}")
        horizon_reports[str(int(horizon))] = _evaluate_horizon(
            model,
            device,
            data,
            teacher_prediction,
            teacher_latent,
            horizon=int(horizon),
            threshold=threshold,
            discount=discount,
            curvature_boundaries=curvature_boundaries,
        )
    early_warning = _evaluate_early_warning_report(
        model,
        device,
        data,
        lookbacks=lookbacks,
        threshold=threshold,
        nominal_step_duration_s=nominal_step_duration_s,
    )
    return {
        "teacher_forced": teacher_by_mode,
        "horizons": horizon_reports,
        "early_warning": early_warning,
    }


def _checkpoint_label(path: Path, used: Iterable[str]) -> str:
    parent = (
        path.parent.parent.name
        if path.parent.name == "checkpoints" and path.parent.parent.name
        else path.parent.name
    )
    candidate = f"{parent}_{path.stem}" if parent else path.stem
    candidate = re.sub(r"[^A-Za-z0-9_.-]+", "_", candidate).strip("_")
    if not candidate:
        candidate = "checkpoint"
    base = candidate
    suffix = 2
    used_set = set(used)
    while candidate in used_set:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def _parse_arguments(
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument(
        "--checkpoint",
        type=Path,
        help="One strict calibrated TD-MPC2 checkpoint",
    )
    checkpoint_group.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        help="Multiple strict calibrated TD-MPC2 checkpoints",
    )
    parser.add_argument("--policy-episodes", type=int, default=5)
    parser.add_argument("--random-episodes", type=int, default=5)
    parser.add_argument("--lower-boundary-repetitions", type=int, default=20)
    parser.add_argument("--tree-end-repetitions", type=int, default=20)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs=4,
        metavar=("POLICY", "RANDOM", "LOWER", "TREE"),
        default=DEFAULT_SEEDS,
        help=(
            "Four disjoint base seeds for policy, random, lower-boundary, "
            "and tree-end trajectories"
        ),
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=DEFAULT_HORIZONS,
        help=(
            "Requested rollout horizons. H=1,3,5,10 and the checkpoint's "
            "configured MPC horizon are always included."
        ),
    )
    parser.add_argument(
        "--lookbacks",
        type=int,
        nargs="+",
        default=DEFAULT_LOOKBACKS,
        help="Early-warning lookbacks; defaults to 1, 3, 5, and 10",
    )
    parser.add_argument(
        "--translation-threshold",
        type=float,
        default=FIXED_TRANSLATION_THRESHOLD,
        help=(
            "Primary Translation diagnostic threshold. Commit 4.7A fixes "
            "this at 0.2; no recalibration is performed."
        ),
    )
    parser.add_argument(
        "--safety-discount",
        type=float,
        default=1.0,
        help="Analysis-only discounted future-risk aggregation coefficient",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "report.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "REPORT.md",
    )
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "trajectories",
        help="Temporary NPZ trajectory directory (use /tmp for research runs)",
    )
    return parser.parse_args(argv)


def _validate_arguments(args: argparse.Namespace) -> None:
    for name in (
        "policy_episodes",
        "random_episodes",
        "lower_boundary_repetitions",
        "tree_end_repetitions",
    ):
        value = getattr(args, name)
        if isinstance(value, bool) or value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be nonnegative")
    if (
        args.policy_episodes
        + args.random_episodes
        + args.lower_boundary_repetitions
        + args.tree_end_repetitions
        <= 0
    ):
        raise ValueError("At least one trajectory collection count must be positive")
    if any(int(value) <= 0 for value in args.horizons):
        raise ValueError("--horizons must contain positive integers")
    if any(int(value) <= 0 for value in args.lookbacks):
        raise ValueError("--lookbacks must contain positive integers")
    threshold = float(args.translation_threshold)
    if (
        not np.isfinite(threshold)
        or not math.isclose(
            threshold,
            FIXED_TRANSLATION_THRESHOLD,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(
            "Commit 4.7A primary --translation-threshold is fixed at 0.2"
        )
    discount = float(args.safety_discount)
    if not np.isfinite(discount) or not 0.0 <= discount <= 1.0:
        raise ValueError("--safety-discount must be finite and in [0, 1]")
    seeds = tuple(int(value) for value in args.seeds)
    if len(set(seeds)) != len(seeds):
        raise ValueError("The four collection seed bases must be disjoint")


def _markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Multi-Step Latent Safety Rollout Evaluation",
        "",
        "This report is post-hoc evaluation only. Safety predictions did not "
        "change TD-MPC2 planning, MPPI scores, reward/value calculations, or "
        "executed actions.",
        "",
        "Temporal alignment: "
        "`prediction[k] = safety(z_t+k, requested_action_t+k)` is compared "
        "with `safety_cost_t+k`, before advancing latent dynamics.",
        "",
        "## Collection",
        "",
        "| Checkpoint | Transitions | Episodes | Trajectory file |",
        "|---|---:|---:|---|",
    ]
    for label, checkpoint in report["checkpoints"].items():
        summary = checkpoint["trajectory_summary"]
        lines.append(
            f"| {label} | {summary['transition_count']} | "
            f"{summary['episode_count']} | "
            f"`{checkpoint['trajectory_path']}` |"
        )
    lines.extend(
        [
            "",
            "## All-mode teacher-forced metrics",
            "",
            "| Checkpoint | Curvature MAE | Curvature Pearson | "
            "Translation MAE | Translation F1@0.2 | "
            "Translation recall@0.2 | Translation FPR@0.2 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    def render(value: Optional[float]) -> str:
        return "null" if value is None else f"{float(value):.6g}"

    for label, checkpoint in report["checkpoints"].items():
        teacher = checkpoint["evaluation"]["teacher_forced"]["all"]
        translation_class = teacher["translation"]["classification"]
        lines.append(
            f"| {label} | {render(teacher['curvature']['mae'])} | "
            f"{render(teacher['curvature']['pearson'])} | "
            f"{render(teacher['translation']['mae'])} | "
            f"{render(translation_class['f1'])} | "
            f"{render(translation_class['recall'])} | "
            f"{render(translation_class['fpr'])} |"
        )
    lines.extend(
        [
            "",
            "## Open-loop all-mode horizon summary",
            "",
            "| Checkpoint | H | Windows | Event F1@0.2 | Event recall | "
            "Event FPR | Curvature-max MAE | Final latent MSE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, checkpoint in report["checkpoints"].items():
        for horizon, horizon_report in checkpoint["evaluation"][
            "horizons"
        ].items():
            group = horizon_report["groups"]["all"]
            event = group["future_blockage"]["open_loop"]["max_risk"]
            curvature = group["curvature_horizon_risk"]["open_loop"][
                "horizon_maximum"
            ]
            final_drift = group["latent_drift_by_offset"][-1]
            lines.append(
                f"| {label} | {horizon} | "
                f"{group['valid_window_count']} | {render(event['f1'])} | "
                f"{render(event['recall'])} | {render(event['fpr'])} | "
                f"{render(curvature['mae'])} | "
                f"{render(final_drift['mse'])} |"
            )
    lines.extend(
        [
            "",
            "## Early warning",
            "",
            "Immediate detection is reported separately and is never counted "
            "as an early warning. Actual SOFA step durations are accumulated "
            "for lead-time seconds.",
            "",
            "| Checkpoint | Reason | Lookback | Events | Detection rate | "
            "Mean lead steps | Mean actual seconds |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label, checkpoint in report["checkpoints"].items():
        groups = checkpoint["evaluation"]["early_warning"][
            "by_blockage_reason"
        ]
        for reason, reason_report in groups.items():
            for lookback, lookback_report in reason_report[
                "by_lookback"
            ].items():
                mean_seconds = lookback_report[
                    "lead_seconds_actual_sofa_time"
                ]["mean"]
                lines.append(
                    f"| {label} | {reason} | {lookback} | "
                    f"{lookback_report['event_count']} | "
                    f"{render(lookback_report['detection_rate'])} | "
                    f"{render(lookback_report['lead_steps']['mean'])} | "
                    f"{render(mean_seconds)} |"
                )
    lines.extend(
        [
            "",
            "The JSON report contains per-mode teacher metrics, every "
            "open-loop offset, latent drift, future-event rankings, false "
            "warning runs, curvature horizon risk, all event records, and "
            "teacher/open-loop gaps.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_markdown_report(report), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_arguments(argv)
    _validate_arguments(args)
    requested_paths = (
        [args.checkpoint] if args.checkpoint is not None else args.checkpoints
    )
    checkpoint_paths = [
        path.expanduser().resolve() for path in requested_paths
    ]
    missing = [str(path) for path in checkpoint_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Expected calibrated checkpoint(s) are missing: "
            + ", ".join(missing)
        )

    loaded = []
    labels: List[str] = []
    for checkpoint_path in checkpoint_paths:
        label = _checkpoint_label(checkpoint_path, labels)
        checkpoint, config, agent, device = load_evaluation_model(
            checkpoint_path,
            requested_device=args.device,
        )
        labels.append(label)
        loaded.append(
            {
                "path": checkpoint_path,
                "label": label,
                "checkpoint": checkpoint,
                "config": config,
                "agent": agent,
                "device": device,
            }
        )
        print(f"loaded strict checkpoint label={label} path={checkpoint_path}")

    first_config = loaded[0]["config"]
    first_agent = loaded[0]["agent"]
    for item in loaded[1:]:
        if strict_json_dumps(item["config"]["environment"]) != strict_json_dumps(
            first_config["environment"]
        ):
            raise ValueError(
                "All checkpoints must use the same stEVE environment config"
            )
        if (
            item["agent"].observation_dim != first_agent.observation_dim
            or item["agent"].action_dim != first_agent.action_dim
        ):
            raise ValueError(
                "All checkpoints must have matching observation/action schemas"
            )
    configured_horizons = {
        int(item["config"]["planning"]["horizon"]) for item in loaded
    }
    if len(configured_horizons) != 1:
        raise ValueError("All checkpoints must use one configured MPC horizon")
    configured_horizon = next(iter(configured_horizons))
    horizons = tuple(
        sorted(
            {
                *(int(value) for value in args.horizons),
                *DEFAULT_HORIZONS,
                configured_horizon,
            }
        )
    )
    lookbacks = tuple(
        sorted({*(int(value) for value in args.lookbacks), *DEFAULT_LOOKBACKS})
    )
    curvature_boundaries = curvature_boundaries_from_diagnostics(first_config)
    seeds = tuple(int(value) for value in args.seeds)

    environment = make_steve_env(first_config["environment"])
    observation_dim = int(np.prod(environment.observation_space.shape))
    action_dim = int(np.prod(environment.action_space.shape))
    if (
        observation_dim != first_agent.observation_dim
        or action_dim != first_agent.action_dim
    ):
        environment.close()
        raise ValueError("Environment schema does not match strict checkpoints")

    try:
        shared_recorder = TrajectoryRecorder(observation_dim, action_dim)
        collection_report: Dict[str, Any] = {}
        if args.random_episodes:
            collection_report["random"] = _collect_random_trajectories(
                environment,
                shared_recorder,
                episodes=args.random_episodes,
                base_seed=seeds[1],
                curvature_boundaries=curvature_boundaries,
            )
        if args.lower_boundary_repetitions:
            collection_report[
                "lower_boundary"
            ] = _collect_lower_boundary_trajectories(
                environment,
                shared_recorder,
                repetitions=args.lower_boundary_repetitions,
                base_seed=seeds[2],
                curvature_boundaries=curvature_boundaries,
            )
        if args.tree_end_repetitions:
            collection_report["tree_end"] = _collect_tree_end_trajectories(
                environment,
                shared_recorder,
                repetitions=args.tree_end_repetitions,
                base_seed=seeds[3],
                curvature_boundaries=curvature_boundaries,
            )
        shared_arrays = (
            shared_recorder.arrays()
            if shared_recorder.transition_count
            else None
        )

        for item in loaded:
            policy_recorder = TrajectoryRecorder(observation_dim, action_dim)
            policy_report = None
            if args.policy_episodes:
                policy_report = _collect_policy_trajectories(
                    environment,
                    item["agent"],
                    policy_recorder,
                    checkpoint_label=item["label"],
                    episodes=args.policy_episodes,
                    base_seed=seeds[0],
                    curvature_boundaries=curvature_boundaries,
                )
            datasets = []
            if policy_recorder.transition_count:
                datasets.append(policy_recorder.arrays())
            if shared_arrays is not None:
                datasets.append(shared_arrays)
            item["trajectory"] = _merge_trajectory_arrays(
                datasets,
                observation_dim=observation_dim,
                action_dim=action_dim,
            )
            item["policy_collection"] = policy_report
    finally:
        environment.close()

    # All real actions and trajectories are frozen before the first call to
    # teacher_forced_rollout/open_loop_rollout.  This makes evaluator-to-MPC
    # influence impossible by construction.
    trajectory_directory = args.trajectory_dir.expanduser().resolve()
    trajectory_directory.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {
        "schema": "steve_tdmpc2_safety_rollout_evaluation_v1",
        "scope": {
            "post_hoc_evaluation_only": True,
            "mpc_scoring_modified": False,
            "action_selection_modified": False,
            "reward_or_value_modified": False,
            "execution_layer_action_mask_unchanged": True,
            "translation_semantics": (
                "controller-feasibility risk, not force, slippage, tissue "
                "injury, vessel-wall damage, or clinical safety"
            ),
        },
        "temporal_alignment": {
            "prediction": (
                "safety_hat_t+k = safety_head(z_hat_t+k, "
                "requested_action_t+k)"
            ),
            "target": "safety_cost_t+k",
            "order": "predict Safety before dynamics at every offset",
            "target_shift": 0,
            "terminal_padding": False,
        },
        "configuration": {
            "checkpoint_paths": [str(path) for path in checkpoint_paths],
            "configured_mpc_horizon": configured_horizon,
            "evaluated_horizons": list(horizons),
            "early_warning_lookbacks": list(lookbacks),
            "translation_threshold": float(args.translation_threshold),
            "threshold_recalibrated": False,
            "safety_discount": float(args.safety_discount),
            "seed_bases": {
                "policy": seeds[0],
                "random": seeds[1],
                "lower_boundary": seeds[2],
                "tree_end": seeds[3],
            },
            "identical_seed_bases_for_all_checkpoints": True,
            "curvature_boundaries_mm_inv": list(curvature_boundaries),
            "environment_image_frequency_hz": float(
                first_config["environment"]["image_frequency"]
            ),
            "nominal_environment_step_duration_s": float(
                1.0 / first_config["environment"]["image_frequency"]
            ),
            "device": str(loaded[0]["device"]),
        },
        "collection": collection_report,
        "checkpoints": {},
    }
    for item in loaded:
        data = item["trajectory"]
        trajectory_path = _save_trajectory_npz(
            trajectory_directory / f"{item['label']}.npz",
            data,
            checkpoint_path=item["path"],
            horizons=horizons,
            seeds=seeds,
        )
        print(
            f"offline evaluation checkpoint={item['label']} "
            f"transitions={data['episode_id'].size}"
        )
        evaluation = _evaluate_checkpoint(
            item["agent"].model,
            item["device"],
            data,
            horizons=horizons,
            lookbacks=lookbacks,
            threshold=float(args.translation_threshold),
            discount=float(args.safety_discount),
            curvature_boundaries=curvature_boundaries,
            nominal_step_duration_s=float(
                1.0 / first_config["environment"]["image_frequency"]
            ),
        )
        report["checkpoints"][item["label"]] = {
            "path": str(item["path"]),
            "checkpoint_step": int(item["checkpoint"]["total_env_steps"]),
            "trajectory_path": str(trajectory_path),
            "trajectory_summary": _trajectory_summary(data),
            "policy_collection": item["policy_collection"],
            "evaluation": evaluation,
        }

    report["planning_isolation"] = {
        "collection_completed_before_safety_inference": True,
        "agent_act_imports_evaluator": False,
        "planner_source_modified_by_this_change": False,
        "safety_prediction_used_for_action_selection": False,
        "focused_test": "RL_TDMPC/smoke_test_safety_rollout.py",
    }
    write_strict_json(args.output_json, report)
    _write_markdown(args.output_markdown, report)
    print(f"wrote strict JSON report: {args.output_json.expanduser().resolve()}")
    print(
        "wrote Markdown report: "
        f"{args.output_markdown.expanduser().resolve()}"
    )


if __name__ == "__main__":
    main()
