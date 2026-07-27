"""Shared configuration, reproducibility, logging, and checkpoint helpers."""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import torch
import yaml
from eve.intervention import TRANSLATION_BLOCK_REASON_NAMES


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DIAGNOSTICS_CONFIG = {
    "validation_interval": 100,
    "gradient_interval": 100,
    "curvature_low_max_mm_inv": 0.05,
    "curvature_medium_max_mm_inv": 0.1,
    "curvature_high_max_mm_inv": 0.25,
}
SAFETY_AUX_TRANSLATION_GROUP_NAMES = tuple(
    TRANSLATION_BLOCK_REASON_NAMES
)
DEFAULT_SAFETY_AUX_VALIDATION_TRANSLATION_THRESHOLD_CANDIDATES = (
    0.0,
    1.0e-6,
    1.0e-5,
    1.0e-4,
    1.0e-3,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.5,
    1.0,
)
DEFAULT_SAFETY_AUX_CONFIG = {
    "enabled": False,
    "dataset_path": None,
    "loss_coef": 1.0,
    "curvature_loss_coef": 1.0,
    "translation_loss_coef": 1.0,
    "translation_group_weights": {
        "none": 2.0,
        "lower_insertion_boundary": 1.0,
        "device_length_limit": 1.0,
        "vessel_tree_end": 1.0,
        "other": 1.0,
    },
    "translation_zero_calibration_coef": 0.1,
    "validation_translation_threshold_candidates": list(
        DEFAULT_SAFETY_AUX_VALIDATION_TRANSLATION_THRESHOLD_CANDIDATES
    ),
    "batch_size": 64,
    "update_interval": 1,
    "sampling_mode": "mixed",
    "translation_fraction": 0.5,
    "curvature_fraction": 0.5,
    "sample_with_replacement": True,
}
COMMIT_46C_SAFETY_AUX_CONFIG_KEYS = {
    "enabled",
    "dataset_path",
    "loss_coef",
    "batch_size",
    "update_interval",
    "sampling_mode",
    "translation_fraction",
    "curvature_fraction",
    "sample_with_replacement",
}
LEGACY_SAFETY_AUX_CONFIG_KEYS = {
    "enabled",
    "capacity",
    "translation_fraction",
    "curvature_fraction",
}
SAFETY_AUX_SAMPLING_MODES = {
    "uniform",
    "translation_balanced",
    "curvature_balanced",
    "mixed",
}


def load_config(path: os.PathLike) -> Dict[str, Any]:
    """Load a YAML configuration and perform basic structural validation."""

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration {config_path} must contain a mapping")
    required = {
        "environment",
        "model",
        "safety",
        "training",
        "planning",
        "logging",
        "checkpoint",
        "evaluation",
    }
    missing = required.difference(config)
    if missing:
        raise KeyError(f"Configuration is missing sections: {sorted(missing)}")
    return config


def build_safety_agent_config(
    config: Mapping[str, Any],
    safety_cost_names: Sequence[str],
) -> Dict[str, Any]:
    """Validate nested safety settings and return the agent's flat schema."""

    names = tuple(str(name) for name in safety_cost_names)
    if len(names) != 2:
        raise ValueError(
            "Safety configuration requires exactly two ordered cost channels; "
            f"got {names}"
        )
    if len(set(names)) != len(names):
        raise ValueError(f"Safety cost channel names must be unique; got {names}")

    safety = config.get("safety")
    if not isinstance(safety, Mapping):
        raise TypeError("Configuration section 'safety' must be a mapping")
    expected_keys = {
        "loss_coef",
        "curvature_loss_coef",
        "translation_error_loss_coef",
        "curvature_scale_mm_inv",
        "translation_error_scale",
    }
    unexpected_keys = sorted(set(safety) - expected_keys)
    if unexpected_keys:
        raise ValueError(
            "Configuration safety section has unexpected keys: "
            f"{unexpected_keys}"
        )

    def validated_value(key: str, *, strictly_positive: bool) -> float:
        if key not in safety:
            raise KeyError(f"Configuration safety section is missing {key!r}")
        value = safety[key]
        if isinstance(value, bool):
            raise TypeError(f"safety.{key} must be a real number, not bool")
        try:
            converted = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(f"safety.{key} must be a real number") from exc
        if not np.isfinite(converted):
            raise ValueError(f"safety.{key} must be finite")
        if strictly_positive and converted <= 0.0:
            raise ValueError(f"safety.{key} must be strictly positive")
        if not strictly_positive and converted < 0.0:
            raise ValueError(f"safety.{key} must be nonnegative")
        if strictly_positive:
            with np.errstate(over="ignore", under="ignore"):
                replay_value = np.asarray(converted, dtype=np.float32).item()
            if not np.isfinite(replay_value) or replay_value <= 0.0:
                raise ValueError(
                    f"safety.{key} must remain finite and strictly positive "
                    "when represented as float32 replay data"
                )
        return converted

    return {
        "safety_cost_names": names,
        "safety_dim": len(names),
        "safety_loss_coef": validated_value(
            "loss_coef", strictly_positive=False
        ),
        "safety_curvature_loss_coef": validated_value(
            "curvature_loss_coef", strictly_positive=False
        ),
        "safety_translation_error_loss_coef": validated_value(
            "translation_error_loss_coef", strictly_positive=False
        ),
        "safety_curvature_scale_mm_inv": validated_value(
            "curvature_scale_mm_inv", strictly_positive=True
        ),
        "safety_translation_error_scale": validated_value(
            "translation_error_scale", strictly_positive=True
        ),
    }


def build_diagnostics_agent_config(
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate diagnostic controls and return their flat agent schema.

    Curvature boundaries are used only to partition logged prediction
    statistics. They are numerical analysis bins, not clinical safety limits.
    """

    diagnostics = config.get("diagnostics", DEFAULT_DIAGNOSTICS_CONFIG)
    if not isinstance(diagnostics, Mapping):
        raise TypeError("Configuration section 'diagnostics' must be a mapping")
    expected_keys = {
        "validation_interval",
        "gradient_interval",
        "curvature_low_max_mm_inv",
        "curvature_medium_max_mm_inv",
        "curvature_high_max_mm_inv",
    }
    unexpected_keys = sorted(set(diagnostics) - expected_keys)
    if unexpected_keys:
        raise ValueError(
            "Configuration diagnostics section has unexpected keys: "
            f"{unexpected_keys}"
        )
    # The third boundary was added for auxiliary sampling. Older format-v3
    # checkpoint configs remain valid and resolve it to the documented default.
    required_keys = expected_keys - {"curvature_high_max_mm_inv"}
    missing_keys = sorted(required_keys - set(diagnostics))
    if missing_keys:
        raise KeyError(
            "Configuration diagnostics section is missing keys: "
            f"{missing_keys}"
        )

    def positive_integer(key: str) -> int:
        value = diagnostics[key]
        if isinstance(value, bool):
            raise TypeError(f"diagnostics.{key} must be an integer, not bool")
        try:
            numeric = float(value)
            converted = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(
                f"diagnostics.{key} must be a positive integer"
            ) from exc
        if (
            not np.isfinite(numeric)
            or numeric != converted
            or converted <= 0
        ):
            raise ValueError(
                f"diagnostics.{key} must be a positive integer"
            )
        return converted

    def nonnegative_float(key: str) -> float:
        value = diagnostics[key]
        if isinstance(value, bool):
            raise TypeError(
                f"diagnostics.{key} must be a real number, not bool"
            )
        try:
            converted = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(
                f"diagnostics.{key} must be a real number"
            ) from exc
        if not np.isfinite(converted) or converted < 0.0:
            raise ValueError(
                f"diagnostics.{key} must be finite and nonnegative"
            )
        return converted

    low_boundary = nonnegative_float("curvature_low_max_mm_inv")
    medium_boundary = nonnegative_float("curvature_medium_max_mm_inv")
    high_boundary_configured = "curvature_high_max_mm_inv" in diagnostics
    if high_boundary_configured:
        high_boundary = nonnegative_float("curvature_high_max_mm_inv")
    else:
        high_boundary = float(
            DEFAULT_DIAGNOSTICS_CONFIG["curvature_high_max_mm_inv"]
        )
    if low_boundary >= medium_boundary:
        raise ValueError(
            "Diagnostics curvature boundaries must be strictly increasing: "
            "curvature_low_max_mm_inv must be smaller than "
            "curvature_medium_max_mm_inv"
        )
    if high_boundary_configured and medium_boundary >= high_boundary:
        raise ValueError(
            "Diagnostics curvature boundaries must be strictly increasing: "
            "curvature_medium_max_mm_inv must be smaller than "
            "curvature_high_max_mm_inv"
        )

    return {
        "validation_interval": positive_integer("validation_interval"),
        "gradient_interval": positive_integer("gradient_interval"),
        "curvature_low_max_mm_inv": low_boundary,
        "curvature_medium_max_mm_inv": medium_boundary,
        "curvature_high_max_mm_inv": high_boundary,
    }


def curvature_boundaries_from_diagnostics(
    config: Mapping[str, Any],
) -> tuple[float, ...]:
    """Return the single configured source of auxiliary curvature boundaries."""

    from envs.safety import validate_curvature_boundaries

    diagnostics = build_diagnostics_agent_config(config)
    return validate_curvature_boundaries(
        (
            diagnostics["curvature_low_max_mm_inv"],
            diagnostics["curvature_medium_max_mm_inv"],
            diagnostics["curvature_high_max_mm_inv"],
        ),
        source="Diagnostics curvature boundaries",
    )


def build_safety_aux_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate offline balanced and calibrated Safety supervision controls.

    Commit-4.6C replaces the former online collection buffer with a strict
    offline dataset. Commit-4.6D adds auxiliary-only channel/group calibration
    and explicit validation-threshold candidates. Exact older schemas remain
    loadable only when disabled; an enabled pre-4.6D schema is rejected rather
    than silently receiving calibration defaults during strict resume.
    """

    safety_aux = config.get("safety_aux", DEFAULT_SAFETY_AUX_CONFIG)
    if not isinstance(safety_aux, Mapping):
        raise TypeError("Configuration section 'safety_aux' must be a mapping")
    received_keys = set(safety_aux)
    if received_keys == LEGACY_SAFETY_AUX_CONFIG_KEYS:
        legacy_enabled = safety_aux["enabled"]
        if type(legacy_enabled) is not bool:
            raise TypeError("safety_aux.enabled must be a bool")
        if legacy_enabled:
            raise ValueError(
                "The legacy four-field safety_aux collection schema cannot be "
                "enabled for offline auxiliary supervision; configure "
                "dataset_path and the complete Commit-4.6D safety_aux schema"
            )

        capacity_value = safety_aux["capacity"]
        if isinstance(capacity_value, bool) or not isinstance(
            capacity_value, (int, np.integer)
        ):
            raise TypeError(
                "legacy safety_aux.capacity must be a positive integer"
            )
        if int(capacity_value) <= 0:
            raise ValueError(
                "legacy safety_aux.capacity must be a positive integer"
            )
        normalized = copy_safety_aux_defaults()
        normalized["translation_fraction"] = _safety_aux_fraction(
            safety_aux["translation_fraction"],
            "translation_fraction",
        )
        normalized["curvature_fraction"] = _safety_aux_fraction(
            safety_aux["curvature_fraction"],
            "curvature_fraction",
        )
        # These fractions were inert while the legacy auxiliary path was
        # disabled, and older checkpoints did not require them to sum to one.
        # The resolved 4.6D schema defaults to mixed sampling, whose strict
        # parser does require that invariant. Preserve valid legacy ratios and
        # canonicalize otherwise-inert invalid ratios so a disabled checkpoint
        # remains stable when its materialized config is parsed again on save.
        if not np.isclose(
            normalized["translation_fraction"]
            + normalized["curvature_fraction"],
            1.0,
            rtol=0.0,
            atol=1.0e-12,
        ):
            normalized["translation_fraction"] = float(
                DEFAULT_SAFETY_AUX_CONFIG["translation_fraction"]
            )
            normalized["curvature_fraction"] = float(
                DEFAULT_SAFETY_AUX_CONFIG["curvature_fraction"]
            )
        return normalized

    if received_keys == COMMIT_46C_SAFETY_AUX_CONFIG_KEYS:
        prior_enabled = safety_aux["enabled"]
        if type(prior_enabled) is not bool:
            raise TypeError("safety_aux.enabled must be a bool")
        if prior_enabled:
            raise ValueError(
                "Enabled safety_aux uses a pre-Commit-4.6D schema and is "
                "missing required auxiliary calibration metadata; explicitly "
                "configure channel coefficients, canonical translation group "
                "weights, zero calibration, and validation threshold candidates"
            )
        normalized_prior = copy_safety_aux_defaults()
        normalized_prior.update(copy.deepcopy(dict(safety_aux)))
        safety_aux = normalized_prior
        received_keys = set(safety_aux)

    expected_keys = set(DEFAULT_SAFETY_AUX_CONFIG)
    unexpected_keys = sorted(received_keys - expected_keys)
    if unexpected_keys:
        raise ValueError(
            "Configuration safety_aux section has unexpected keys: "
            f"{unexpected_keys}"
        )
    missing_keys = sorted(expected_keys - received_keys)
    if missing_keys:
        raise KeyError(
            "Configuration safety_aux section is missing keys: "
            f"{missing_keys}"
        )

    enabled = safety_aux["enabled"]
    if type(enabled) is not bool:
        raise TypeError("safety_aux.enabled must be a bool")

    dataset_path_value = safety_aux["dataset_path"]
    if dataset_path_value is None:
        dataset_path = None
    elif isinstance(dataset_path_value, (str, os.PathLike)) and not isinstance(
        dataset_path_value, (bytes, bytearray)
    ):
        dataset_path = os.fspath(dataset_path_value)
        if not isinstance(dataset_path, str) or not dataset_path.strip():
            raise ValueError(
                "safety_aux.dataset_path must be null or a non-empty path"
            )
    else:
        raise TypeError(
            "safety_aux.dataset_path must be null or a path-like string"
        )
    if enabled and dataset_path is None:
        raise ValueError(
            "safety_aux.dataset_path must be a non-empty path when "
            "safety_aux.enabled=true"
        )

    loss_coef = _safety_aux_finite_float(
        safety_aux["loss_coef"],
        "loss_coef",
        minimum=0.0,
    )
    curvature_loss_coef = _safety_aux_finite_float(
        safety_aux["curvature_loss_coef"],
        "curvature_loss_coef",
        minimum=0.0,
    )
    translation_loss_coef = _safety_aux_finite_float(
        safety_aux["translation_loss_coef"],
        "translation_loss_coef",
        minimum=0.0,
    )
    translation_group_weights = _safety_aux_translation_group_weights(
        safety_aux["translation_group_weights"]
    )
    translation_zero_calibration_coef = _safety_aux_finite_float(
        safety_aux["translation_zero_calibration_coef"],
        "translation_zero_calibration_coef",
        minimum=0.0,
    )
    validation_translation_threshold_candidates = (
        _safety_aux_validation_threshold_candidates(
            safety_aux["validation_translation_threshold_candidates"]
        )
    )
    batch_size = _safety_aux_positive_integer(
        safety_aux["batch_size"],
        "batch_size",
    )
    update_interval = _safety_aux_positive_integer(
        safety_aux["update_interval"],
        "update_interval",
    )

    sampling_mode = safety_aux["sampling_mode"]
    if not isinstance(sampling_mode, str):
        raise TypeError("safety_aux.sampling_mode must be a string")
    if sampling_mode not in SAFETY_AUX_SAMPLING_MODES:
        raise ValueError(
            "safety_aux.sampling_mode must be one of "
            f"{sorted(SAFETY_AUX_SAMPLING_MODES)}, got {sampling_mode!r}"
        )

    translation_fraction = _safety_aux_fraction(
        safety_aux["translation_fraction"],
        "translation_fraction",
    )
    curvature_fraction = _safety_aux_fraction(
        safety_aux["curvature_fraction"],
        "curvature_fraction",
    )
    if sampling_mode == "mixed" and not np.isclose(
        translation_fraction + curvature_fraction,
        1.0,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            "safety_aux.translation_fraction and "
            "safety_aux.curvature_fraction must sum to 1 for mixed sampling"
        )

    sample_with_replacement = safety_aux["sample_with_replacement"]
    if type(sample_with_replacement) is not bool:
        raise TypeError("safety_aux.sample_with_replacement must be a bool")

    # Current configs explicitly carry the third auxiliary boundary and should
    # always validate it. Older format-v3 checkpoint configs may contain only
    # the two diagnostic boundaries; preserve those configs while auxiliary
    # replay is disabled, but require a valid three-boundary schema before the
    # auxiliary replay can be enabled.
    diagnostics = config.get("diagnostics")
    high_boundary_is_explicit = (
        diagnostics is None
        or (
            isinstance(diagnostics, Mapping)
            and "curvature_high_max_mm_inv" in diagnostics
        )
    )
    build_diagnostics_agent_config(config)
    if enabled or high_boundary_is_explicit:
        curvature_boundaries_from_diagnostics(config)
    return {
        "enabled": enabled,
        "dataset_path": dataset_path,
        "loss_coef": loss_coef,
        "curvature_loss_coef": curvature_loss_coef,
        "translation_loss_coef": translation_loss_coef,
        "translation_group_weights": translation_group_weights,
        "translation_zero_calibration_coef": (
            translation_zero_calibration_coef
        ),
        "validation_translation_threshold_candidates": (
            validation_translation_threshold_candidates
        ),
        "batch_size": batch_size,
        "update_interval": update_interval,
        "sampling_mode": sampling_mode,
        "translation_fraction": translation_fraction,
        "curvature_fraction": curvature_fraction,
        "sample_with_replacement": sample_with_replacement,
    }


def copy_safety_aux_defaults() -> Dict[str, Any]:
    """Return a defensive copy of nested auxiliary defaults."""

    return copy.deepcopy(DEFAULT_SAFETY_AUX_CONFIG)


def _safety_aux_translation_group_weights(
    value: Any,
) -> Dict[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError(
            "safety_aux.translation_group_weights must be a mapping"
        )
    received_names = tuple(value.keys())
    expected_names = SAFETY_AUX_TRANSLATION_GROUP_NAMES
    missing = [name for name in expected_names if name not in value]
    unexpected = [name for name in received_names if name not in expected_names]
    if missing or unexpected:
        raise ValueError(
            "safety_aux.translation_group_weights keys mismatch; "
            f"missing={missing}, unexpected={unexpected}"
        )
    if received_names != expected_names:
        raise ValueError(
            "safety_aux.translation_group_weights must use canonical order "
            f"{expected_names}, got {received_names}"
        )
    weights = {
        name: _safety_aux_finite_float(
            value[name],
            f"translation_group_weights.{name}",
            minimum=0.0,
        )
        for name in expected_names
    }
    if not any(weight > 0.0 for weight in weights.values()):
        raise ValueError(
            "safety_aux.translation_group_weights must contain at least one "
            "positive weight"
        )
    return weights


def _safety_aux_validation_threshold_candidates(
    value: Any,
) -> list[float]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(
            "safety_aux.validation_translation_threshold_candidates must be "
            "a list or tuple"
        )
    if not value:
        raise ValueError(
            "safety_aux.validation_translation_threshold_candidates must not "
            "be empty"
        )
    candidates = [
        _safety_aux_finite_float(
            candidate,
            f"validation_translation_threshold_candidates[{index}]",
            minimum=0.0,
        )
        for index, candidate in enumerate(value)
    ]
    if any(
        current <= previous
        for previous, current in zip(candidates, candidates[1:])
    ):
        raise ValueError(
            "safety_aux.validation_translation_threshold_candidates must be "
            "strictly increasing with no duplicates"
        )
    return candidates


def _safety_aux_positive_integer(value: Any, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"safety_aux.{key} must be a positive integer")
    converted = int(value)
    if converted <= 0:
        raise ValueError(f"safety_aux.{key} must be a positive integer")
    return converted


def _safety_aux_finite_float(
    value: Any,
    key: str,
    *,
    minimum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"safety_aux.{key} must be a real number")
    converted = float(value)
    if not np.isfinite(converted) or converted < minimum:
        raise ValueError(
            f"safety_aux.{key} must be finite and at least {minimum}"
        )
    return converted


def _safety_aux_fraction(value: Any, key: str) -> float:
    converted = _safety_aux_finite_float(value, key, minimum=0.0)
    if converted > 1.0:
        raise ValueError(f"safety_aux.{key} must be in [0, 1]")
    return converted


def resolve_project_path(path: os.PathLike) -> Path:
    """Resolve paths relative to ``RL_TDMPC`` rather than the current shell."""

    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (PROJECT_DIR / candidate).resolve()


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    """Select CUDA when requested/available, with an explicit CPU fallback."""

    requested = str(requested).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"device={requested!r} was requested, but torch.cuda.is_available() is False"
        )
    return torch.device(requested)


def scalar_metrics(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert scalar values into strict JSON, using null when unavailable."""

    output: Dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().mean().cpu().item()
        elif isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            value = None
        if isinstance(value, (int, float, bool, str)) or value is None:
            output[key] = value
        else:
            output[key] = str(value)
    return output


class MetricLogger:
    """Append metrics to JSONL and episode/update CSV files."""

    def __init__(self, log_dir: os.PathLike) -> None:
        self.log_dir = resolve_project_path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.log_dir / "metrics.jsonl"
        self._csv_headers: Dict[str, Iterable[str]] = {}

    def log(self, category: str, metrics: Mapping[str, Any]) -> Dict[str, Any]:
        record = {"category": category, **scalar_metrics(metrics)}
        with self.jsonl_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

        csv_path = self.log_dir / f"{category}.csv"
        fields = tuple(record.keys())
        known_fields = self._csv_headers.get(category)
        if known_fields is None:
            if csv_path.exists() and csv_path.stat().st_size > 0:
                with csv_path.open("r", encoding="utf-8", newline="") as stream:
                    known_fields = tuple(next(csv.reader(stream)))
            else:
                known_fields = fields
            self._csv_headers[category] = known_fields

        # Each category uses a stable schema. Unknown later metrics remain in JSONL.
        row = {field: record.get(field, "") for field in known_fields}
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0
        with csv_path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(known_fields))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        return record


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Optional[Mapping[str, Any]]) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(payload: Mapping[str, Any], destination: os.PathLike) -> Path:
    """Write a checkpoint through a temporary file and atomically replace it."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, destination)
    return destination


def atomic_json_save(
    payload: Mapping[str, Any],
    destination: os.PathLike,
) -> Path:
    """Serialize strict JSON through a temporary file and atomically replace it."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(
                dict(payload),
                stream,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_torch_checkpoint(
    path: os.PathLike, *, map_location: Any = "cpu"
) -> Dict[str, Any]:
    """Load one of this integration's trusted full-state checkpoints."""

    # Explicit ``weights_only=False`` is required because resumable files also
    # contain optimizer, replay, RNG, and config state; spelling it out avoids
    # PyTorch's implicit-pickle FutureWarning.
    return torch.load(
        Path(path).expanduser().resolve(),
        map_location=map_location,
        weights_only=False,
    )


def apply_cli_overrides(
    config: MutableMapping[str, Any],
    *,
    total_steps: Optional[int] = None,
    checkpoint_interval: Optional[int] = None,
    device: Optional[str] = None,
    safety_loss_coef: Optional[float] = None,
) -> MutableMapping[str, Any]:
    if total_steps is not None:
        config["training"]["total_steps"] = int(total_steps)
    if checkpoint_interval is not None:
        config["checkpoint"]["interval"] = int(checkpoint_interval)
    if device is not None:
        config["training"]["device"] = device
    if safety_loss_coef is not None:
        config["safety"]["loss_coef"] = float(safety_loss_coef)
    return config
