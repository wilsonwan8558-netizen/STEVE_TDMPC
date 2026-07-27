#!/usr/bin/env python3
"""Train TD-MPC2 on the normalized stEVE endovascular environment."""

from __future__ import annotations

import argparse
import copy
import json
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

from envs.safety import SAFETY_COST_NAMES
from envs.steve_env import make_steve_env
from tdmpc2.agent import SAFETY_MODEL_SCHEMA_VERSION, TDMPC2Agent
from tdmpc2.common import (
    PROJECT_DIR,
    MetricLogger,
    apply_cli_overrides,
    atomic_json_save,
    atomic_torch_save,
    build_diagnostics_agent_config,
    build_safety_agent_config,
    build_safety_aux_config,
    capture_rng_state,
    curvature_boundaries_from_diagnostics,
    load_torch_checkpoint,
    load_config,
    resolve_project_path,
    restore_rng_state,
    select_device,
    set_seed,
)
from tdmpc2.replay_buffer import (
    SAFETY_COST_SCHEMA_VERSION,
    EpisodeReplayBuffer,
    validate_safety_cost_names,
)
from tdmpc2.safety_diagnostics import (
    evaluate_fixed_safety_validation,
    flatten_fixed_safety_validation_metrics,
)
from tdmpc2.safety_aux_supervision import (
    SAFETY_AUXILIARY_CHECKPOINT_KEY,
    SafetyAuxiliarySupervisor,
)


DEFAULT_CONFIG = PROJECT_DIR / "configs" / "steve.yaml"
CHECKPOINT_FORMAT_VERSION = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "YAML config (default: saved checkpoint config when resuming, "
            "otherwise configs/steve.yaml)"
        ),
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=None, help="Override total steps")
    parser.add_argument("--seed-steps", type=int, default=None)
    parser.add_argument("--initial-updates", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument(
        "--safety-loss-coef",
        type=float,
        default=None,
        help=(
            "Override safety.loss_coef (for example 0.0, 0.1, or 1.0); "
            "must be finite and nonnegative"
        ),
    )
    parser.add_argument("--device", type=str, default=None, help="auto, cpu, or cuda")
    return parser.parse_args()


def build_agent_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    training_horizon = int(config["training"]["horizon"])
    planning_horizon = int(config["planning"]["horizon"])
    if training_horizon != planning_horizon:
        raise ValueError("training.horizon and planning.horizon must match")
    return {
        **dict(config["model"]),
        **dict(config["training"]),
        **dict(config["planning"]),
        **build_safety_agent_config(config, SAFETY_COST_NAMES),
        **build_diagnostics_agent_config(config),
    }


def validate_checkpoint_schema(
    checkpoint: Mapping[str, Any],
    *,
    config: Optional[Mapping[str, Any]] = None,
    source: str = "Checkpoint",
    auxiliary_dataset_path_override: Optional[Path] = None,
) -> None:
    """Validate format-v3 safety metadata and all internal/external links."""

    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"{source} must be a mapping")

    def exact_integer(value: Any, label: str) -> int:
        if isinstance(value, bool):
            raise TypeError(f"{source} {label} must be an integer, not bool")
        try:
            numeric = float(value)
            converted = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{source} {label} must be an integer") from exc
        if not np.isfinite(numeric) or numeric != converted:
            raise ValueError(f"{source} {label} must be an integer")
        return converted

    if "format_version" not in checkpoint:
        raise ValueError(
            f"{source} has no format_version and predates the required "
            "format-v3 two-channel safety-model checkpoint schema"
        )
    format_version = exact_integer(
        checkpoint["format_version"], "format_version"
    )
    if format_version != CHECKPOINT_FORMAT_VERSION:
        legacy_note = (
            " This checkpoint predates the format-v3 two-channel safety-model "
            "metadata and cannot be resumed or evaluated without retraining."
            if format_version < CHECKPOINT_FORMAT_VERSION
            else ""
        )
        raise ValueError(
            f"{source} format_version {format_version} is unsupported; expected "
            f"{CHECKPOINT_FORMAT_VERSION}.{legacy_note}"
        )

    expected_names = validate_safety_cost_names(
        SAFETY_COST_NAMES,
        source="Environment safety schema",
    )
    if "safety_cost_names" not in checkpoint:
        raise ValueError(
            f"{source} is missing required safety_cost_names metadata"
        )
    checkpoint_names = validate_safety_cost_names(
        checkpoint["safety_cost_names"],
        source=source,
    )
    if checkpoint_names != expected_names:
        raise ValueError(
            f"{source} safety_cost_names {checkpoint_names} do not match "
            f"environment order {expected_names}"
        )

    if "safety_dim" not in checkpoint:
        raise ValueError(f"{source} is missing required safety_dim metadata")
    safety_dim = exact_integer(checkpoint["safety_dim"], "safety_dim")
    if safety_dim != 2 or safety_dim != len(checkpoint_names):
        raise ValueError(
            f"{source} safety_dim {safety_dim} does not match the required "
            f"two-channel dimension {len(checkpoint_names)}"
        )

    if "safety_cost_schema_version" not in checkpoint:
        raise ValueError(
            f"{source} is missing required safety_cost_schema_version metadata"
        )
    schema_version = exact_integer(
        checkpoint["safety_cost_schema_version"],
        "safety_cost_schema_version",
    )
    if schema_version != SAFETY_COST_SCHEMA_VERSION:
        raise ValueError(
            f"{source} safety_cost_schema_version {schema_version} is unsupported; "
            f"expected {SAFETY_COST_SCHEMA_VERSION}"
        )

    if "safety_model_schema_version" not in checkpoint:
        raise ValueError(
            f"{source} is missing required safety_model_schema_version metadata"
        )
    model_schema_version = exact_integer(
        checkpoint["safety_model_schema_version"],
        "safety_model_schema_version",
    )
    if model_schema_version != SAFETY_MODEL_SCHEMA_VERSION:
        raise ValueError(
            f"{source} safety_model_schema_version {model_schema_version} is "
            f"unsupported; expected {SAFETY_MODEL_SCHEMA_VERSION}"
        )

    if "config" not in checkpoint:
        raise ValueError(f"{source} is missing its embedded config")
    embedded_config = checkpoint["config"]
    if not isinstance(embedded_config, Mapping):
        raise TypeError(f"{source} embedded config must be a mapping")
    embedded_agent_config = build_agent_config(embedded_config)
    embedded_safety_aux_config = build_safety_aux_config(embedded_config)
    embedded_safety_aux_enabled = bool(
        embedded_safety_aux_config["enabled"]
    )
    embedded_curvature_boundaries = (
        curvature_boundaries_from_diagnostics(embedded_config)
        if embedded_safety_aux_enabled
        else None
    )
    embedded_names = tuple(embedded_agent_config["safety_cost_names"])
    if embedded_names != checkpoint_names:
        raise ValueError(
            f"{source} top-level safety_cost_names {checkpoint_names} do not "
            f"match embedded config order {embedded_names}"
        )
    if int(embedded_agent_config["safety_dim"]) != safety_dim:
        raise ValueError(
            f"{source} top-level safety_dim {safety_dim} does not match "
            f"embedded config safety_dim {embedded_agent_config['safety_dim']}"
        )

    def positive_checkpoint_float(key: str) -> float:
        if key not in checkpoint:
            raise ValueError(f"{source} is missing required {key} metadata")
        value = checkpoint[key]
        if isinstance(value, bool):
            raise TypeError(f"{source} {key} must be a real number, not bool")
        try:
            converted = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(f"{source} {key} must be a real number") from exc
        if not np.isfinite(converted) or converted <= 0.0:
            raise ValueError(f"{source} {key} must be finite and positive")
        return converted

    checkpoint_curvature_scale = positive_checkpoint_float(
        "safety_curvature_scale_mm_inv"
    )
    checkpoint_translation_scale = positive_checkpoint_float(
        "safety_translation_error_scale"
    )
    embedded_curvature_scale = float(
        embedded_agent_config["safety_curvature_scale_mm_inv"]
    )
    embedded_translation_scale = float(
        embedded_agent_config["safety_translation_error_scale"]
    )
    if checkpoint_curvature_scale != embedded_curvature_scale:
        raise ValueError(
            f"{source} top-level safety_curvature_scale_mm_inv "
            f"{checkpoint_curvature_scale} does not match embedded config "
            f"value {embedded_curvature_scale}"
        )
    if checkpoint_translation_scale != embedded_translation_scale:
        raise ValueError(
            f"{source} top-level safety_translation_error_scale "
            f"{checkpoint_translation_scale} does not match embedded config "
            f"value {embedded_translation_scale}"
        )

    if "safety_config" not in checkpoint:
        raise ValueError(f"{source} is missing required safety_config metadata")
    checkpoint_safety_config = checkpoint["safety_config"]
    if not isinstance(checkpoint_safety_config, Mapping):
        raise TypeError(f"{source} safety_config must be a mapping")
    embedded_safety_config = embedded_config["safety"]
    if dict(checkpoint_safety_config) != dict(embedded_safety_config):
        raise ValueError(
            f"{source} top-level safety_config does not match its embedded "
            "config['safety']"
        )

    if config is not None:
        requested_agent_config = build_agent_config(config)
        requested_safety_aux_config = build_safety_aux_config(config)
        requested_names = tuple(requested_agent_config["safety_cost_names"])
        if requested_names != checkpoint_names:
            raise ValueError(
                f"{source} safety-cost order {checkpoint_names} does not match "
                f"requested config order {requested_names}"
            )
        if int(requested_agent_config["safety_dim"]) != safety_dim:
            raise ValueError(
                f"{source} safety_dim {safety_dim} does not match requested "
                f"config safety_dim {requested_agent_config['safety_dim']}"
            )
        requested_curvature_scale = float(
            requested_agent_config["safety_curvature_scale_mm_inv"]
        )
        requested_translation_scale = float(
            requested_agent_config["safety_translation_error_scale"]
        )
        if requested_curvature_scale != checkpoint_curvature_scale:
            raise ValueError(
                f"{source} curvature scale {checkpoint_curvature_scale} does "
                f"not match requested config value {requested_curvature_scale}"
            )
        if requested_translation_scale != checkpoint_translation_scale:
            raise ValueError(
                f"{source} translation-error scale "
                f"{checkpoint_translation_scale} does not match requested "
                f"config value {requested_translation_scale}"
            )
        requested_safety = config.get("safety")
        if not isinstance(requested_safety, Mapping):
            raise TypeError("Requested config safety section must be a mapping")
        if dict(requested_safety) != dict(checkpoint_safety_config):
            raise ValueError(
                f"{source} safety_config does not match the requested "
                "config['safety']"
            )
        requested_safety_aux_enabled = bool(
            requested_safety_aux_config["enabled"]
        )
        if requested_safety_aux_enabled != embedded_safety_aux_enabled:
            raise ValueError(
                f"{source} safety_aux.enabled does not match the requested "
                "config"
            )
        if embedded_safety_aux_enabled:
            comparable_requested_aux = copy.deepcopy(
                requested_safety_aux_config
            )
            comparable_embedded_aux = copy.deepcopy(
                embedded_safety_aux_config
            )
            if auxiliary_dataset_path_override is not None:
                comparable_requested_aux.pop("dataset_path", None)
                comparable_embedded_aux.pop("dataset_path", None)
            if comparable_requested_aux != comparable_embedded_aux:
                raise ValueError(
                    f"{source} resolved safety_aux config does not match the "
                    "requested config"
                )
            requested_curvature_boundaries = (
                curvature_boundaries_from_diagnostics(config)
            )
            if requested_curvature_boundaries != embedded_curvature_boundaries:
                raise ValueError(
                    f"{source} auxiliary curvature boundaries do not match the "
                    "requested diagnostics config"
                )

    agent_state = checkpoint.get("agent")
    if not isinstance(agent_state, Mapping):
        raise TypeError(f"{source} agent state must be a mapping")
    required_agent_safety_keys = {
        "safety_model_schema_version",
        "safety_cost_names",
        "safety_dim",
        "safety_config",
    }
    missing_agent_safety_keys = sorted(
        required_agent_safety_keys - agent_state.keys()
    )
    if missing_agent_safety_keys:
        raise ValueError(
            f"{source} agent state predates the required Safety-Aware schema; "
            f"missing metadata {missing_agent_safety_keys}"
        )
    agent_schema_version = exact_integer(
        agent_state["safety_model_schema_version"],
        "agent safety_model_schema_version",
    )
    if agent_schema_version != SAFETY_MODEL_SCHEMA_VERSION:
        raise ValueError(
            f"{source} agent safety_model_schema_version "
            f"{agent_schema_version} does not match required version "
            f"{SAFETY_MODEL_SCHEMA_VERSION}"
        )
    agent_names = tuple(agent_state["safety_cost_names"])
    if agent_names != checkpoint_names:
        raise ValueError(
            f"{source} agent safety_cost_names {agent_names} do not match "
            f"top-level order {checkpoint_names}"
        )
    agent_safety_dim = exact_integer(
        agent_state["safety_dim"], "agent safety_dim"
    )
    if agent_safety_dim != safety_dim:
        raise ValueError(
            f"{source} agent safety_dim {agent_safety_dim} does not match "
            f"top-level safety_dim {safety_dim}"
        )
    agent_safety_config = agent_state["safety_config"]
    if not isinstance(agent_safety_config, Mapping):
        raise TypeError(f"{source} agent safety_config must be a mapping")
    expected_agent_safety_config = {
        "safety_loss_coef": float(
            embedded_agent_config["safety_loss_coef"]
        ),
        "safety_curvature_loss_coef": float(
            embedded_agent_config["safety_curvature_loss_coef"]
        ),
        "safety_translation_error_loss_coef": float(
            embedded_agent_config[
                "safety_translation_error_loss_coef"
            ]
        ),
        "safety_curvature_scale_mm_inv": embedded_curvature_scale,
        "safety_translation_error_scale": embedded_translation_scale,
    }
    if dict(agent_safety_config) != expected_agent_safety_config:
        raise ValueError(
            f"{source} agent safety_config does not match its embedded config"
        )

    if "safety_aux_replay" in checkpoint:
        raise ValueError(
            f"{source} contains the incompatible Commit-4.6A/B "
            "collection-only safety_aux_replay checkpoint schema; automatic "
            "migration to offline auxiliary supervision is not supported"
        )
    has_safety_aux_state = SAFETY_AUXILIARY_CHECKPOINT_KEY in checkpoint
    safety_aux_state = checkpoint.get(SAFETY_AUXILIARY_CHECKPOINT_KEY)
    if embedded_safety_aux_enabled:
        if not isinstance(safety_aux_state, Mapping):
            raise ValueError(
                f"{source} enables safety_aux but is missing a valid "
                f"{SAFETY_AUXILIARY_CHECKPOINT_KEY} state"
            )
        try:
            supervisor_config = copy.deepcopy(
                embedded_safety_aux_config
            )
            if auxiliary_dataset_path_override is not None:
                supervisor_config["dataset_path"] = str(
                    auxiliary_dataset_path_override.expanduser().resolve()
                )
            auxiliary_supervisor = SafetyAuxiliarySupervisor(
                supervisor_config,
                observation_dim=exact_integer(
                    agent_state["observation_dim"], "agent observation_dim"
                ),
                action_dim=exact_integer(
                    agent_state["action_dim"], "agent action_dim"
                ),
                safety_cost_names=checkpoint_names,
                curvature_boundaries_mm_inv=embedded_curvature_boundaries,
                seed=exact_integer(
                    embedded_config["training"]["seed"],
                    "embedded training seed",
                ),
            )
            auxiliary_supervisor.load_state_dict(
                safety_aux_state,
                expected_normal_update_count=exact_integer(
                    agent_state.get("update_count", 0),
                    "agent update_count",
                ),
                allow_dataset_path_override=(
                    auxiliary_dataset_path_override is not None
                ),
            )
        except (
            FileNotFoundError,
            TypeError,
            ValueError,
            FloatingPointError,
        ) as exc:
            raise type(exc)(
                f"{source} has invalid offline Safety auxiliary state or "
                f"dataset: {exc}"
            ) from exc
    elif has_safety_aux_state:
        raise ValueError(
            f"{source} contains {SAFETY_AUXILIARY_CHECKPOINT_KEY} while "
            "safety_aux.enabled=false"
        )

    replay_state = checkpoint.get("replay")
    if replay_state is not None:
        if not isinstance(replay_state, Mapping):
            raise TypeError(f"{source} replay state must be a mapping")
        if "safety_cost_names" not in replay_state:
            raise ValueError(
                f"{source} replay state is missing required safety_cost_names metadata"
            )
        replay_names = validate_safety_cost_names(
            replay_state["safety_cost_names"],
            source=f"{source} replay state",
        )
        if checkpoint_names != replay_names:
            raise ValueError(
                f"{source} top-level and replay safety-cost schemas disagree"
            )
        replay_schema_version = replay_state.get("safety_cost_schema_version")
        if replay_schema_version is None:
            raise ValueError(
                f"{source} replay state is missing required "
                "safety_cost_schema_version metadata"
            )
        parsed_replay_schema_version = exact_integer(
            replay_schema_version,
            "replay safety_cost_schema_version",
        )
        if parsed_replay_schema_version != SAFETY_COST_SCHEMA_VERSION:
            raise ValueError(
                f"{source} replay safety_cost_schema_version "
                f"{parsed_replay_schema_version} is unsupported; expected "
                f"{SAFETY_COST_SCHEMA_VERSION}"
            )
        replay_safety_dim = replay_state.get("safety_cost_dim")
        if replay_safety_dim is None:
            raise ValueError(
                f"{source} replay state is missing required safety_cost_dim metadata"
            )
        parsed_replay_safety_dim = exact_integer(
            replay_safety_dim, "replay safety_cost_dim"
        )
        if parsed_replay_safety_dim != safety_dim:
            raise ValueError(
                f"{source} replay safety_cost_dim {parsed_replay_safety_dim} does "
                f"not match top-level safety_dim {safety_dim}"
            )


def validate_collected_safety_cost(
    value: Any,
    *,
    safety_cost_names: Sequence[str] = SAFETY_COST_NAMES,
    source: str = "Environment info['safety_cost']",
) -> np.ndarray:
    """Validate and copy one environment transition's safety-cost vector."""

    validated_names = validate_safety_cost_names(
        safety_cost_names,
        source=f"{source} names",
    )
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{source} must be a numpy.ndarray, got {type(value).__name__}")
    expected_shape = (len(validated_names),)
    if value.shape != expected_shape:
        if value.shape == (len(validated_names) + 1,):
            raise ValueError(
                f"{source} uses the unsupported legacy three-channel shape "
                f"{value.shape}; expected {expected_shape}"
            )
        raise ValueError(
            f"{source} must have shape {expected_shape}, got {value.shape}"
        )
    if value.dtype != np.float32:
        raise TypeError(f"{source} must use float32, got {value.dtype}")
    if not np.all(np.isfinite(value)):
        raise FloatingPointError(f"{source} contains NaN or infinity")
    if np.any(value < 0.0):
        raise ValueError(f"{source} values must be nonnegative")
    return value.copy()


def save_checkpoint(
    *,
    path: Path,
    config: Mapping[str, Any],
    agent: TDMPC2Agent,
    replay: EpisodeReplayBuffer,
    total_env_steps: int,
    episode_index: int,
    success_count: int,
    include_replay: bool,
    safety_auxiliary: Optional[SafetyAuxiliarySupervisor] = None,
) -> Path:
    agent_config = build_agent_config(config)
    expected_names = validate_safety_cost_names(
        agent_config["safety_cost_names"],
        source="Checkpoint config",
    )
    replay_names = validate_safety_cost_names(
        replay.safety_cost_names,
        source="Replay buffer",
    )
    if replay_names != expected_names:
        raise ValueError(
            "Replay safety-cost order does not match the checkpoint config: "
            f"{replay_names} != {expected_names}"
        )
    safety_dim = int(agent_config["safety_dim"])
    if safety_dim != len(expected_names) or safety_dim != 2:
        raise ValueError(
            f"Checkpoint safety_dim must be 2, got {safety_dim}"
        )
    safety_config = config.get("safety")
    if not isinstance(safety_config, Mapping):
        raise TypeError("Checkpoint config safety section must be a mapping")
    safety_aux_config = build_safety_aux_config(config)
    safety_aux_enabled = bool(safety_aux_config["enabled"])
    if safety_aux_enabled != (safety_auxiliary is not None):
        raise ValueError(
            "safety_aux.enabled and the checkpoint auxiliary supervisor disagree"
        )
    if safety_auxiliary is not None:
        expected_boundaries = curvature_boundaries_from_diagnostics(config)
        if safety_auxiliary.observation_dim != agent.observation_dim:
            raise ValueError(
                "Safety auxiliary dataset observation dimension does not match "
                "the agent"
            )
        if safety_auxiliary.action_dim != agent.action_dim:
            raise ValueError(
                "Safety auxiliary dataset action dimension does not match the agent"
            )
        if safety_auxiliary.safety_cost_names != expected_names:
            raise ValueError(
                "Safety auxiliary dataset safety-cost order does not match "
                "the checkpoint config"
            )
        if (
            safety_auxiliary.curvature_boundaries_mm_inv
            != expected_boundaries
        ):
            raise ValueError(
                "Safety auxiliary dataset curvature boundaries do not match "
                "diagnostics"
            )
        if safety_auxiliary.last_normal_update_count != agent.update_count:
            raise ValueError(
                "Safety auxiliary normal-update counter does not match the "
                "agent update_count"
            )
    payload: Dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "algorithm": "TD-MPC2",
        "official_reference_commit": "e9f59321933cbc8e11a002b842adc7d4ffae8ff1",
        "safety_cost_schema_version": SAFETY_COST_SCHEMA_VERSION,
        "safety_model_schema_version": SAFETY_MODEL_SCHEMA_VERSION,
        "safety_cost_names": expected_names,
        "safety_dim": safety_dim,
        "safety_curvature_scale_mm_inv": float(
            agent_config["safety_curvature_scale_mm_inv"]
        ),
        "safety_translation_error_scale": float(
            agent_config["safety_translation_error_scale"]
        ),
        "safety_config": copy.deepcopy(dict(safety_config)),
        "config": copy.deepcopy(dict(config)),
        "agent": agent.state_dict(),
        "total_env_steps": int(total_env_steps),
        "episode_index": int(episode_index),
        "success_count": int(success_count),
        "rng_state": capture_rng_state(),
        "saved_at_unix": time.time(),
    }
    if include_replay:
        payload["replay"] = replay.state_dict()
    if safety_auxiliary is not None:
        payload[SAFETY_AUXILIARY_CHECKPOINT_KEY] = (
            safety_auxiliary.state_dict()
        )
    validate_checkpoint_schema(
        payload,
        config=config,
        source=f"Checkpoint payload for {path}",
    )
    return atomic_torch_save(payload, path)


class LossAccumulator:
    def __init__(self) -> None:
        self.sums: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}
        self.weighted_sums: Dict[str, float] = {}
        self.weights: Dict[str, float] = {}
        self.observed_keys = set()

    def add(self, metrics: Mapping[str, float]) -> None:
        for key, value in metrics.items():
            converted = float(value)
            self.observed_keys.add(key)
            # NaN represents an unavailable conditional diagnostic (for
            # example, positive-target MAE when this batch has no positives).
            # Ignore it when another batch supplies a defined value, but retain
            # NaN when the metric was unavailable for the entire log window.
            if np.isnan(converted):
                continue
            if key.endswith("_count"):
                # Counts describe coverage over the complete logging window,
                # so summing is more useful than a mean-per-batch count.
                self.sums[key] = self.sums.get(key, 0.0) + converted
                self.counts[key] = self.counts.get(key, 0) + 1
                continue

            # Conditional translation/curvature metrics have a sibling count.
            # Weight them by their number of valid samples so rare/small groups
            # are not overrepresented merely because batches are averaged.
            count_key = None
            for suffix in ("_mae", "_pred_mean", "_target_mean"):
                if key.endswith(suffix):
                    candidate = f"{key[:-len(suffix)]}_count"
                    if candidate in metrics:
                        count_key = candidate
                        break
            if count_key is not None:
                weight = float(metrics[count_key])
                if np.isfinite(weight) and weight > 0.0:
                    self.weighted_sums[key] = (
                        self.weighted_sums.get(key, 0.0)
                        + converted * weight
                    )
                    self.weights[key] = self.weights.get(key, 0.0) + weight
                continue

            self.sums[key] = self.sums.get(key, 0.0) + converted
            self.counts[key] = self.counts.get(key, 0) + 1

    def pop_means(self) -> Dict[str, float]:
        if not self.observed_keys:
            return {}
        means: Dict[str, float] = {}
        for key in self.observed_keys:
            if key.endswith("_count") and self.counts.get(key, 0) > 0:
                means[key] = self.sums[key]
            elif self.weights.get(key, 0.0) > 0.0:
                means[key] = self.weighted_sums[key] / self.weights[key]
            elif self.counts.get(key, 0) > 0:
                means[key] = self.sums[key] / self.counts[key]
            else:
                means[key] = float("nan")
        self.sums.clear()
        self.counts.clear()
        self.weighted_sums.clear()
        self.weights.clear()
        self.observed_keys.clear()
        return means


def safety_aux_sampling_log_record(
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    """Flatten one scheduled sampler result without logging dataset indices."""

    metrics = SafetyAuxiliarySupervisor.sampling_metrics(metadata)
    return {
        "normal_update_count": int(metadata["normal_update_count"]),
        "auxiliary_update_count": int(metadata["auxiliary_update_count"]),
        "requested_sampling_mode": str(metadata["mode"]),
        "requested_batch_size": int(metadata["requested_batch_size"]),
        "actual_sample_count": int(metadata["returned_batch_size"]),
        "sample_with_replacement": bool(metadata["with_replacement"]),
        "replacement_used": bool(metadata["replacement_used"]),
        "unique_sample_count": int(metadata["unique_sample_count"]),
        "duplicate_exposure_fraction": float(
            metadata["duplicate_exposure_fraction"]
        ),
        "missing_translation_groups": ",".join(
            metadata["missing_translation_groups"]
        ),
        "missing_curvature_groups": ",".join(
            metadata["missing_curvature_groups"]
        ),
        **metrics,
    }


def train(config: Dict[str, Any], resume_path: Optional[Path] = None) -> None:
    training = config["training"]
    checkpoint_config = config["checkpoint"]
    seed = int(training["seed"])
    set_seed(seed)
    device = select_device(training["device"])
    print(f"Device: {device}")
    safety_cost_names = validate_safety_cost_names(
        SAFETY_COST_NAMES,
        source="Environment",
    )
    raw_diagnostics = config.get("diagnostics")
    diagnostics_config = build_diagnostics_agent_config(config)
    safety_aux_config = build_safety_aux_config(config)
    diagnostics_high_is_explicit = (
        raw_diagnostics is None
        or (
            isinstance(raw_diagnostics, Mapping)
            and "curvature_high_max_mm_inv" in raw_diagnostics
        )
    )
    # Diagnostics are not part of the format-v3 Safety model schema. Materialize
    # defaults here so Commit-4 checkpoints/configs remain loadable while every
    # new run records its exact effective diagnostic settings.
    if diagnostics_high_is_explicit or bool(safety_aux_config["enabled"]):
        config["diagnostics"] = copy.deepcopy(diagnostics_config)
    else:
        config["diagnostics"] = {
            key: copy.deepcopy(value)
            for key, value in diagnostics_config.items()
            if key != "curvature_high_max_mm_inv"
        }
    config["safety_aux"] = copy.deepcopy(safety_aux_config)
    curvature_boundaries = (
        curvature_boundaries_from_diagnostics(config)
        if bool(safety_aux_config["enabled"])
        else None
    )
    agent_config = build_agent_config(config)
    startup_configuration = {
        "safety": copy.deepcopy(dict(config["safety"])),
        "diagnostics": copy.deepcopy(dict(config["diagnostics"])),
        "safety_aux": copy.deepcopy(safety_aux_config),
        "curvature_stratum_boundaries_mm_inv": (
            list(curvature_boundaries)
            if curvature_boundaries is not None
            else None
        ),
        "safety_cost_names": list(safety_cost_names),
        "safety_dim": int(agent_config["safety_dim"]),
    }
    print(
        "Safety and diagnostics configuration:\n"
        + json.dumps(startup_configuration, indent=2, sort_keys=True)
    )

    env = make_steve_env(config["environment"])
    try:
        observation_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        agent = TDMPC2Agent(
            observation_dim,
            action_dim,
            agent_config,
            episode_length=int(
                config["environment"]["max_episode_steps"]
            ),
            device=device,
        )
        replay = EpisodeReplayBuffer(
            int(training["replay_capacity"]),
            observation_dim,
            action_dim,
            int(training["horizon"]),
            int(training["batch_size"]),
            safety_cost_names=safety_cost_names,
            seed=seed,
        )
        safety_auxiliary = (
            SafetyAuxiliarySupervisor(
                safety_aux_config,
                observation_dim=observation_dim,
                action_dim=action_dim,
                safety_cost_names=safety_cost_names,
                curvature_boundaries_mm_inv=curvature_boundaries,
                seed=seed,
            )
            if bool(safety_aux_config["enabled"])
            else None
        )
        logger = MetricLogger(config["logging"]["directory"])
    except BaseException:
        env.close()
        raise
    try:
        started_at_unix = time.time()
        run_id = datetime.fromtimestamp(
            started_at_unix,
            tz=timezone.utc,
        ).strftime("%Y%m%dT%H%M%S%fZ")
        run_metadata = {
            "run_id": run_id,
            "algorithm": "TD-MPC2",
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "started_at_unix": started_at_unix,
            "started_at_utc": datetime.fromtimestamp(
                started_at_unix,
                tz=timezone.utc,
            ).isoformat(),
            "command": shlex.join([sys.executable, *sys.argv]),
            "argv": list(sys.argv),
            "resume_checkpoint": (
                str(resume_path.expanduser().resolve())
                if resume_path is not None
                else None
            ),
            "device": str(device),
            **startup_configuration,
        }
        if safety_auxiliary is not None:
            run_metadata["safety_aux_dataset"] = (
                safety_auxiliary.metadata_summary()
            )
    except BaseException:
        env.close()
        raise

    total_env_steps = 0
    episode_index = 0
    successes = 0
    if resume_path is not None:
        try:
            resolved_resume = resume_path.expanduser().resolve()
            checkpoint = load_torch_checkpoint(
                resolved_resume, map_location=device
            )
            validate_checkpoint_schema(
                checkpoint,
                config=config,
                source=f"Resume checkpoint {resolved_resume}",
            )
            agent.load_state_dict(
                checkpoint["agent"], load_optimizers=True
            )
            if "replay" in checkpoint:
                replay.load_state_dict(checkpoint["replay"])
            if safety_auxiliary is not None:
                safety_auxiliary.load_state_dict(
                    checkpoint[SAFETY_AUXILIARY_CHECKPOINT_KEY],
                    expected_normal_update_count=agent.update_count,
                )
            total_env_steps = int(checkpoint["total_env_steps"])
            episode_index = int(checkpoint.get("episode_index", 0))
            successes = int(checkpoint.get("success_count", 0))
            restore_rng_state(checkpoint.get("rng_state"))
            print(
                f"Resumed {resolved_resume} at step {total_env_steps:,}; "
                f"replay={len(replay):,}, updates={agent.update_count:,}."
            )
        except BaseException:
            env.close()
            raise

    try:
        total_steps = int(training["total_steps"])
    except BaseException:
        env.close()
        raise
    if total_env_steps >= total_steps:
        env.close()
        raise ValueError(
            f"Checkpoint is already at step {total_env_steps}, which is not below "
            f"training.total_steps={total_steps}. Use --steps with a larger value."
        )

    try:
        run_metadata.update(
            {
                "initial_total_env_steps": total_env_steps,
                "initial_update_count": agent.update_count,
            }
        )
        metadata_path = atomic_json_save(
            run_metadata,
            logger.log_dir / f"run_metadata_{run_id}.json",
        )
        # Keep a convenient latest-run pointer without sacrificing the immutable
        # per-run metadata needed for controlled coefficient comparisons.
        atomic_json_save(
            run_metadata,
            logger.log_dir / "run_metadata.json",
        )
        print(f"Run metadata: {metadata_path}")

        checkpoint_dir = resolve_project_path(
            checkpoint_config["directory"]
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_interval = int(checkpoint_config["interval"])
        update_log_interval = max(
            1, int(config["logging"]["update_interval"])
        )
        loss_accumulator = LossAccumulator()
        start_time = time.time()
        interrupted = False
        training_failed = False
        update_transaction_failed = False
    except BaseException:
        env.close()
        raise

    try:
        while total_env_steps < total_steps:
            episode_seed = seed + episode_index
            env.action_space.seed(episode_seed)
            observation, _ = env.reset(seed=episode_seed)
            episode_observations: List[np.ndarray] = [observation.copy()]
            episode_actions: List[np.ndarray] = []
            episode_rewards: List[float] = []
            episode_safety_costs: List[np.ndarray] = []
            episode_terminated: List[bool] = []
            episode_reward = 0.0
            success = False
            simulation_error = False

            terminated = truncated = False
            while not (terminated or truncated) and total_env_steps < total_steps:
                use_random_action = (
                    total_env_steps < int(training["seed_steps"])
                    or agent.update_count == 0
                )
                if use_random_action:
                    action = env.action_space.sample().astype(np.float32)
                else:
                    action = agent.act(
                        observation,
                        first_step=len(episode_actions) == 0,
                        eval_mode=False,
                    )

                next_observation, reward, terminated, truncated, info = env.step(action)
                total_env_steps += 1
                if total_env_steps >= total_steps and not (terminated or truncated):
                    # The requested training budget cuts this physical episode
                    # short. Treat it as a time-limit truncation for logging;
                    # it still bootstraps in replay because terminated=False.
                    truncated = True
                episode_observations.append(next_observation.copy())
                episode_actions.append(action.copy())
                episode_rewards.append(float(reward))
                transition_safety_cost = validate_collected_safety_cost(
                    info["safety_cost"],
                    safety_cost_names=safety_cost_names,
                )
                episode_safety_costs.append(transition_safety_cost)
                episode_terminated.append(bool(terminated))
                episode_reward += float(reward)
                success = success or bool(info.get("is_success", False))
                simulation_error = simulation_error or bool(
                    info.get("simulation_error", False)
                )
                observation = next_observation

                episode_done = terminated or truncated or total_env_steps >= total_steps
                if episode_done:
                    replay.add_episode(
                        episode_observations,
                        episode_actions,
                        episode_rewards,
                        episode_terminated,
                        safety_cost=episode_safety_costs,
                    )

                if (
                    total_env_steps >= int(training["seed_steps"])
                    and replay.can_sample()
                ):
                    if agent.update_count == 0:
                        updates = int(training["initial_updates"])
                        print(f"Pretraining world model for {updates:,} updates...")
                    else:
                        updates = int(training["updates_per_step"])
                    for _ in range(updates):
                        if safety_auxiliary is None:
                            # Keep the baseline call path and RNG consumption
                            # identical when offline supervision is disabled.
                            update_metrics = agent.update(replay)
                        else:
                            agent_update_started = False
                            agent_update_completed = False
                            try:
                                auxiliary_batch, sampling_metadata = (
                                    safety_auxiliary.begin_update(
                                        agent.update_count + 1
                                    )
                                )
                                agent_update_started = True
                                update_metrics = agent.update(
                                    replay,
                                    safety_aux_batch=auxiliary_batch,
                                    safety_aux_loss_coef=float(
                                        safety_aux_config["loss_coef"]
                                    ),
                                    safety_aux_curvature_loss_coef=float(
                                        safety_aux_config[
                                            "curvature_loss_coef"
                                        ]
                                    ),
                                    safety_aux_translation_loss_coef=float(
                                        safety_aux_config[
                                            "translation_loss_coef"
                                        ]
                                    ),
                                    safety_aux_translation_group_weights=(
                                        copy.deepcopy(
                                            safety_aux_config[
                                                "translation_group_weights"
                                            ]
                                        )
                                    ),
                                    safety_aux_translation_zero_calibration_coef=(
                                        float(
                                            safety_aux_config[
                                                "translation_zero_calibration_coef"
                                            ]
                                        )
                                    ),
                                )
                                agent_update_completed = True
                                safety_auxiliary.commit_update(
                                    agent.update_count
                                )
                            except BaseException:
                                if agent_update_completed:
                                    # If interruption landed immediately
                                    # before/during commit, finish the cheap
                                    # counter commit so the completed model
                                    # update remains checkpoint-consistent.
                                    if safety_auxiliary.update_pending:
                                        try:
                                            safety_auxiliary.commit_update(
                                                agent.update_count
                                            )
                                        except BaseException:
                                            update_transaction_failed = True
                                elif agent_update_started:
                                    # Model/main-replay updates cannot be
                                    # rolled back safely after an interrupted
                                    # agent.update. Never checkpoint this
                                    # potentially partial state.
                                    update_transaction_failed = True
                                    safety_auxiliary.rollback_update()
                                else:
                                    # begin_update is atomic and already
                                    # restored its sampler/counters.
                                    safety_auxiliary.rollback_update()
                                raise
                            update_metrics.update(
                                safety_auxiliary.sampling_metrics(
                                    sampling_metadata
                                )
                            )
                            if sampling_metadata is not None:
                                logger.log(
                                    "safety_aux_sampling",
                                    {
                                        "run_id": run_id,
                                        "total_env_steps": total_env_steps,
                                        **safety_aux_sampling_log_record(
                                            sampling_metadata
                                        ),
                                    },
                                )
                            if (
                                agent.update_count
                                % int(diagnostics_config["validation_interval"])
                                == 0
                            ):
                                fixed_validation = (
                                    evaluate_fixed_safety_validation(
                                        agent,
                                        safety_auxiliary.validation_batch(),
                                        batch_size=max(
                                            1,
                                            int(
                                                safety_aux_config[
                                                    "batch_size"
                                                ]
                                            ),
                                        ),
                                        translation_threshold_candidates=(
                                            safety_aux_config[
                                                "validation_translation_threshold_candidates"
                                            ]
                                        ),
                                    )
                                )
                                fixed_validation_flat = (
                                    flatten_fixed_safety_validation_metrics(
                                        fixed_validation,
                                        prefix="aux_val_",
                                    )
                                )
                                logger.log(
                                    "safety_aux_validation",
                                    {
                                        "run_id": run_id,
                                        "total_env_steps": total_env_steps,
                                        "update_count": agent.update_count,
                                        "dataset_fingerprint": (
                                            safety_auxiliary.fingerprint
                                        ),
                                        **fixed_validation_flat,
                                    },
                                )
                        loss_accumulator.add(update_metrics)

                if (
                    agent.update_count
                    and total_env_steps % update_log_interval == 0
                ):
                    update_metrics = loss_accumulator.pop_means()
                    if update_metrics:
                        update_metrics.update(
                            {
                                "run_id": run_id,
                                "total_env_steps": total_env_steps,
                                "update_count": agent.update_count,
                                "replay_size": len(replay),
                                "elapsed_seconds": time.time() - start_time,
                            }
                        )
                        record = logger.log("updates", update_metrics)
                        print(
                            f"step={total_env_steps:,} "
                            f"loss={record['total_loss']:.4f} "
                            f"replay={len(replay):,}"
                        )

                if (
                    checkpoint_interval > 0
                    and total_env_steps % checkpoint_interval == 0
                ):
                    path = checkpoint_dir / f"step_{total_env_steps}.pt"
                    save_checkpoint(
                        path=path,
                        config=config,
                        agent=agent,
                        replay=replay,
                        total_env_steps=total_env_steps,
                        # A live SOFA state is intentionally not serialized;
                        # resume therefore starts the following episode seed.
                        episode_index=episode_index + 1,
                        success_count=(
                            successes + int(success) if episode_done else successes
                        ),
                        include_replay=bool(checkpoint_config["save_replay"]),
                        safety_auxiliary=safety_auxiliary,
                    )
                    print(f"Saved checkpoint: {path}")

            successes += int(success)
            episode_index += 1
            episode_metrics = {
                "run_id": run_id,
                "total_env_steps": total_env_steps,
                "episode": episode_index,
                "episode_reward": episode_reward,
                "episode_length": len(episode_actions),
                "success": success,
                "success_rate": successes / episode_index,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "simulation_error": simulation_error,
                "replay_size": len(replay),
            }
            logger.log("episodes", episode_metrics)
            print(
                f"episode={episode_index} step={total_env_steps:,} "
                f"reward={episode_reward:.3f} length={len(episode_actions)} "
                f"success={int(success)}"
            )
    except KeyboardInterrupt:
        interrupted = True
        if update_transaction_failed:
            print(
                "Interrupted inside agent.update(); not writing potentially "
                "inconsistent state. Resume from the previous checkpoint if "
                "one exists because model/main-replay updates cannot be "
                "rolled back atomically."
            )
        else:
            print("Interrupted; saving a resumable checkpoint...")
    except BaseException:
        training_failed = True
        raise
    finally:
        try:
            should_save_final = (
                bool(checkpoint_config.get("save_final", True))
                or interrupted
            ) and not training_failed and not update_transaction_failed
            if should_save_final:
                final_path = checkpoint_dir / f"step_{total_env_steps}.pt"
                save_checkpoint(
                    path=final_path,
                    config=config,
                    agent=agent,
                    replay=replay,
                    total_env_steps=total_env_steps,
                    episode_index=episode_index,
                    success_count=successes,
                    include_replay=bool(checkpoint_config["save_replay"]),
                    safety_auxiliary=safety_auxiliary,
                )
                print(f"Saved final checkpoint: {final_path}")
        finally:
            env.close()


def main() -> None:
    args = parse_args()
    if args.config is not None:
        config = load_config(args.config)
    elif args.resume is not None:
        resume_metadata = load_torch_checkpoint(args.resume, map_location="cpu")
        validate_checkpoint_schema(
            resume_metadata,
            source=f"Resume checkpoint {args.resume.expanduser().resolve()}",
        )
        if "config" not in resume_metadata:
            raise KeyError("Resume checkpoint has no embedded config; pass --config")
        config = copy.deepcopy(resume_metadata["config"])
    else:
        config = load_config(DEFAULT_CONFIG)
    apply_cli_overrides(
        config,
        total_steps=args.steps,
        checkpoint_interval=args.checkpoint_interval,
        device=args.device,
        safety_loss_coef=args.safety_loss_coef,
    )
    if args.seed_steps is not None:
        config["training"]["seed_steps"] = args.seed_steps
    if args.initial_updates is not None:
        config["training"]["initial_updates"] = args.initial_updates
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.num_samples is not None:
        config["planning"]["num_samples"] = args.num_samples
        config["planning"]["num_elites"] = min(
            int(config["planning"]["num_elites"]), args.num_samples
        )
        config["planning"]["num_pi_trajs"] = min(
            int(config["planning"]["num_pi_trajs"]), args.num_samples
        )
    if args.iterations is not None:
        config["planning"]["iterations"] = args.iterations
    train(config, args.resume)


if __name__ == "__main__":
    main()
