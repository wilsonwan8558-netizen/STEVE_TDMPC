"""Shared configuration, reproducibility, logging, and checkpoint helpers."""

from __future__ import annotations

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


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DIAGNOSTICS_CONFIG = {
    "validation_interval": 100,
    "gradient_interval": 100,
    "curvature_low_max_mm_inv": 0.05,
    "curvature_medium_max_mm_inv": 0.1,
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
    }
    unexpected_keys = sorted(set(diagnostics) - expected_keys)
    if unexpected_keys:
        raise ValueError(
            "Configuration diagnostics section has unexpected keys: "
            f"{unexpected_keys}"
        )
    missing_keys = sorted(expected_keys - set(diagnostics))
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
    if low_boundary >= medium_boundary:
        raise ValueError(
            "diagnostics.curvature_low_max_mm_inv must be smaller than "
            "diagnostics.curvature_medium_max_mm_inv"
        )

    return {
        "validation_interval": positive_integer("validation_interval"),
        "gradient_interval": positive_integer("gradient_interval"),
        "curvature_low_max_mm_inv": low_boundary,
        "curvature_medium_max_mm_inv": medium_boundary,
    }


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
