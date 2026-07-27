#!/usr/bin/env python3
"""Evaluate the auxiliary TD-MPC2 Safety Head without changing MPC actions.

The regular evaluation path first obtains the agent's final action, then
predicts the immediate two-channel safety cost from ``(observation_t,
action_t)`` before passing that unchanged action to stEVE.  The predictions
are monitoring-only and never participate in planning or action selection.

``--targeted-blockage`` additionally executes real, evaluation-only control
commands for known intervention-level blockage cases.  It does not modify
the environment dynamics and does not inject synthetic labels.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from envs.safety import CURVATURE_STRATUM_NAMES, SAFETY_COST_NAMES
from envs.steve_env import StEVEEnv, make_steve_env
from eve.intervention import TRANSLATION_BLOCK_REASON_NAMES
from tdmpc2.agent import TDMPC2Agent
from tdmpc2.common import (
    curvature_boundaries_from_diagnostics,
    load_config,
    load_torch_checkpoint,
    select_device,
    set_seed,
)
from tdmpc2.safety_aux_dataset import load_dataset
from tdmpc2.safety_diagnostics import (
    aggregate_safety_evaluation,
    evaluate_fixed_safety_validation,
)
from train import (
    build_agent_config,
    validate_checkpoint_schema,
    validate_collected_safety_cost,
)


DEFAULT_POSITIVE_THRESHOLD = 1.0e-6
DEFAULT_OFFLINE_VALIDATION_BATCH_SIZE = 512
_TARGETED_TREE_END_SEED = 301


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Accepting an explicit argument sequence keeps the entry point importable
    and straightforward to exercise in smoke tests.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional config override; checkpoint config is used by default",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help=(
            "Number of policy-evaluation episodes. Without --safety-dataset, "
            "the default is evaluation.episodes. With --safety-dataset, policy "
            "evaluation runs only when this option is explicitly supplied."
        ),
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=DEFAULT_POSITIVE_THRESHOLD,
        help=(
            "Diagnostic threshold in normalized translation-error units "
            "(default: 1e-6; not a clinical safety threshold)"
        ),
    )
    parser.add_argument(
        "--targeted-blockage",
        action="store_true",
        help=(
            "Also execute real lower-boundary and practical forward-blockage "
            "diagnostic actions"
        ),
    )
    parser.add_argument(
        "--safety-dataset",
        type=Path,
        default=None,
        help=(
            "Strict Commit-4.6B Safety auxiliary dataset. When supplied alone, "
            "evaluate its complete fixed validation split without creating a "
            "SOFA environment."
        ),
    )
    parser.add_argument(
        "--safety-batch-size",
        type=int,
        default=DEFAULT_OFFLINE_VALIDATION_BATCH_SIZE,
        help=(
            "Inference minibatch size for fixed offline validation "
            f"(default: {DEFAULT_OFFLINE_VALIDATION_BATCH_SIZE})"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optionally write the complete report to this user-selected path",
    )
    return parser.parse_args(argv)


def _validate_positive_threshold(value: float) -> float:
    threshold = float(value)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError(
            "--positive-threshold must be finite and nonnegative"
        )
    return threshold


def load_evaluation_model(
    checkpoint_path: Path,
    *,
    config_path: Optional[Path] = None,
    requested_device: Optional[str] = None,
    auxiliary_dataset_override: Optional[Path] = None,
) -> Tuple[
    Dict[str, Any],
    Dict[str, Any],
    TDMPC2Agent,
    torch.device,
]:
    """Strictly load a checkpoint and model without constructing stEVE/SOFA."""

    resolved_checkpoint = checkpoint_path.expanduser().resolve()
    checkpoint = load_torch_checkpoint(resolved_checkpoint, map_location="cpu")
    if config_path is not None:
        config = load_config(config_path)
    elif "config" in checkpoint:
        config = copy.deepcopy(checkpoint["config"])
    else:
        raise KeyError(
            "Checkpoint has no saved config; pass --config explicitly"
        )

    # This validates format version, ordered channel names, transform scales,
    # replay/model schema versions, auxiliary identity, and cross-links.
    validate_checkpoint_schema(
        checkpoint,
        config=config,
        source=f"Safety evaluation checkpoint {resolved_checkpoint}",
        auxiliary_dataset_path_override=auxiliary_dataset_override,
    )

    device = select_device(
        requested_device or str(config["training"]["device"])
    )
    agent_state = checkpoint["agent"]
    observation_dim = int(agent_state["observation_dim"])
    action_dim = int(agent_state["action_dim"])
    if observation_dim <= 0 or action_dim <= 0:
        raise ValueError(
            "Checkpoint observation/action dimensions must be positive"
        )
    agent = TDMPC2Agent(
        observation_dim,
        action_dim,
        build_agent_config(config),
        episode_length=int(config["environment"]["max_episode_steps"]),
        device=device,
    )
    # World-model loading remains strict, but evaluation intentionally avoids
    # restoring optimizer state.
    agent.load_state_dict(
        agent_state,
        load_optimizers=False,
    )
    agent.model.train(False)
    return checkpoint, config, agent, device


def _make_evaluation_environment(
    config: Mapping[str, Any],
    agent: TDMPC2Agent,
) -> StEVEEnv:
    """Construct stEVE only for explicitly requested online diagnostics."""

    env = make_steve_env(config["environment"])
    try:
        observation_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        if (
            observation_dim != agent.observation_dim
            or action_dim != agent.action_dim
        ):
            raise ValueError(
                "Environment observation/action dimensions "
                f"({observation_dim}, {action_dim}) do not match checkpoint "
                f"dimensions ({agent.observation_dim}, {agent.action_dim})"
            )
    except Exception:
        env.close()
        raise
    return env


def load_evaluation_components(
    checkpoint_path: Path,
    *,
    config_path: Optional[Path] = None,
    requested_device: Optional[str] = None,
) -> Tuple[
    Dict[str, Any],
    Dict[str, Any],
    StEVEEnv,
    TDMPC2Agent,
    torch.device,
]:
    """Strictly load the model and construct an online evaluation environment."""

    checkpoint, config, agent, device = load_evaluation_model(
        checkpoint_path,
        config_path=config_path,
        requested_device=requested_device,
    )
    env = _make_evaluation_environment(config, agent)
    return checkpoint, config, env, agent, device


def _synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def predict_safety_before_step(
    agent: TDMPC2Agent,
    observation: np.ndarray,
    final_action: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Predict safety for one final action without mutating that action or MPC.

    Returns the original-unit prediction, its transformed representation, and
    encoder-plus-Safety-Head inference time in milliseconds.
    """

    observation_array = np.asarray(observation, dtype=np.float32)
    action_array = np.asarray(final_action, dtype=np.float32)
    if observation_array.shape != (agent.observation_dim,):
        raise ValueError(
            "Safety evaluation observation has shape "
            f"{observation_array.shape}; expected ({agent.observation_dim},)"
        )
    if action_array.shape != (agent.action_dim,):
        raise ValueError(
            f"Safety evaluation action has shape {action_array.shape}; "
            f"expected ({agent.action_dim},)"
        )
    if not np.all(np.isfinite(observation_array)):
        raise FloatingPointError(
            "Safety evaluation observation contains NaN or infinity"
        )
    if not np.all(np.isfinite(action_array)):
        raise FloatingPointError(
            "Safety evaluation action contains NaN or infinity"
        )
    action_before_prediction = action_array.copy()

    with torch.no_grad():
        observation_tensor = torch.as_tensor(
            observation_array,
            dtype=torch.float32,
            device=agent.device,
        ).unsqueeze(0)
        action_tensor = torch.as_tensor(
            action_array,
            dtype=torch.float32,
            device=agent.device,
        ).unsqueeze(0)

        _synchronize_if_cuda(agent.device)
        start_ns = time.perf_counter_ns()
        latent = agent.model.encode(observation_tensor)
        prediction_transformed = agent.model.safety_transformed(
            latent,
            action_tensor,
        )
        prediction = agent.model.decode_safety_transformed(
            prediction_transformed
        )
        _synchronize_if_cuda(agent.device)
        inference_ms = (time.perf_counter_ns() - start_ns) / 1.0e6

        expected_shape = (1, len(SAFETY_COST_NAMES))
        if (
            tuple(prediction_transformed.shape) != expected_shape
            or tuple(prediction.shape) != expected_shape
        ):
            raise RuntimeError(
                "Safety Head returned transformed/decoded shapes "
                f"{tuple(prediction_transformed.shape)}/"
                f"{tuple(prediction.shape)}; expected {expected_shape}"
            )

    prediction_array = (
        prediction.squeeze(0).detach().cpu().numpy().astype(np.float32)
    )
    transformed_array = (
        prediction_transformed.squeeze(0)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    if not np.all(np.isfinite(prediction_array)):
        raise FloatingPointError(
            "Safety Head returned a non-finite original-unit prediction"
        )
    if not np.all(np.isfinite(transformed_array)):
        raise FloatingPointError(
            "Safety Head returned a non-finite transformed prediction"
        )
    if np.any(prediction_array < 0.0):
        raise ValueError("Safety Head returned a negative decoded prediction")
    if not np.array_equal(action_array, action_before_prediction):
        raise RuntimeError("Safety prediction unexpectedly modified the action")
    if not np.isfinite(inference_ms) or inference_ms < 0.0:
        raise FloatingPointError("Safety inference time is invalid")
    return prediction_array, transformed_array, float(inference_ms)


def collect_policy_evaluation(
    env: StEVEEnv,
    agent: TDMPC2Agent,
    *,
    episodes: int,
    base_seed: int,
) -> Dict[str, Any]:
    """Collect policy actions, predictions, and post-step stEVE targets."""

    if episodes <= 0:
        raise ValueError("Evaluation episode count must be positive")

    targets: List[np.ndarray] = []
    predictions: List[np.ndarray] = []
    transformed_predictions: List[np.ndarray] = []
    inference_times_ms: List[float] = []
    episode_records: List[Dict[str, Any]] = []

    for episode_index in range(episodes):
        episode_seed = int(base_seed + episode_index)
        # MPPI is sampling based. Re-seeding before reset makes final actions
        # reproducible without changing the planner.
        set_seed(episode_seed)
        observation, _ = env.reset(seed=episode_seed)
        terminated = truncated = False
        episode_reward = 0.0
        episode_length = 0
        success = False
        simulation_error = False

        while not (terminated or truncated):
            action = agent.act(
                observation,
                first_step=episode_length == 0,
                eval_mode=True,
            )
            action_for_environment = np.asarray(
                action, dtype=np.float32
            ).copy()
            prediction, transformed, inference_ms = (
                predict_safety_before_step(
                    agent,
                    observation,
                    action_for_environment,
                )
            )
            (
                next_observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action_for_environment)
            target = validate_collected_safety_cost(
                info.get("safety_cost"),
                safety_cost_names=SAFETY_COST_NAMES,
                source=(
                    "Safety evaluation "
                    f"episode {episode_index + 1} step {episode_length + 1} "
                    "info['safety_cost']"
                ),
            )

            predictions.append(prediction)
            transformed_predictions.append(transformed)
            targets.append(target)
            inference_times_ms.append(inference_ms)
            episode_reward += float(reward)
            episode_length += 1
            success = success or bool(info.get("is_success", False))
            simulation_error = simulation_error or bool(
                info.get("simulation_error", False)
            )
            observation = next_observation

        episode_records.append(
            {
                "episode": episode_index + 1,
                "seed": episode_seed,
                "reward": float(episode_reward),
                "length": int(episode_length),
                "success": bool(success),
                "simulation_error": bool(simulation_error),
            }
        )
        print(
            f"episode={episode_index + 1} reward={episode_reward:.3f} "
            f"length={episode_length} success={int(success)}"
        )

    if not targets:
        raise RuntimeError("Safety evaluation collected no transitions")
    return {
        "target": np.stack(targets).astype(np.float32, copy=False),
        "prediction": np.stack(predictions).astype(np.float32, copy=False),
        "prediction_transformed": np.stack(
            transformed_predictions
        ).astype(np.float32, copy=False),
        "inference_times_ms": np.asarray(
            inference_times_ms, dtype=np.float64
        ),
        "episodes": episode_records,
    }


def _targeted_transition_record(
    *,
    case: str,
    action: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    inference_ms: float,
    info: Mapping[str, Any],
    positive_threshold: float,
    steps_to_case: int = 1,
) -> Dict[str, Any]:
    translation_index = SAFETY_COST_NAMES.index(
        "normalized_requested_applied_translation_error"
    )
    metrics = info.get("safety_metrics", {})
    if not isinstance(metrics, Mapping):
        metrics = {}
    return {
        "case": case,
        "status": "observed",
        "steps_to_case": int(steps_to_case),
        "action": np.asarray(action, dtype=np.float32).tolist(),
        "prediction": {
            name: float(prediction[index])
            for index, name in enumerate(SAFETY_COST_NAMES)
        },
        "target": {
            name: float(target[index])
            for index, name in enumerate(SAFETY_COST_NAMES)
        },
        "translation_error_absolute_error": float(
            abs(
                float(prediction[translation_index])
                - float(target[translation_index])
            )
        ),
        # Ground-truth positivity follows the cleaned replay target's exact
        # semantics. The configurable threshold applies only to predictions.
        "translation_target_positive": bool(
            target[translation_index] > 0.0
        ),
        "translation_prediction_positive": bool(
            prediction[translation_index] > positive_threshold
        ),
        "translation_action_blocked": bool(
            metrics.get("translation_action_blocked", False)
        ),
        "requested_translation_speed_mm_s": float(
            metrics.get("requested_translation_speed_mm_s", 0.0)
        ),
        "applied_translation_speed_mm_s": float(
            metrics.get("applied_translation_speed_mm_s", 0.0)
        ),
        "inserted_length_mm": float(
            metrics.get("inserted_length_mm", 0.0)
        ),
        "safety_inference_ms": float(inference_ms),
    }


def _is_at_tree_end(env: StEVEEnv) -> bool:
    """Read the same tree-end predicate used by MonoPlaneStatic."""

    from eve.intervention.vesseltree.vesseltree import at_tree_end

    positions = np.asarray(
        env.intervention.simulation.dof_positions,
        dtype=np.float64,
    )
    if positions.ndim != 2 or positions.shape[0] == 0:
        return False
    return bool(
        at_tree_end(
            positions[0],
            env.intervention.vessel_tree,
        )
    )


def evaluate_targeted_blockage(
    env: StEVEEnv,
    agent: TDMPC2Agent,
    *,
    base_seed: int,
    positive_threshold: float,
) -> Dict[str, Any]:
    """Exercise real lower-boundary and practical forward blockage cases."""

    cases: Dict[str, Any] = {}

    # InsertionPoint reset places the device at zero inserted length. A real
    # maximum retraction command is then blocked by MonoPlaneStatic's lower
    # insertion-boundary mask.
    lower_seed = int(base_seed)
    observation, _ = env.reset(seed=lower_seed)
    retract_action = np.asarray([-1.0, 0.0], dtype=np.float32)
    prediction, _, inference_ms = predict_safety_before_step(
        agent,
        observation,
        retract_action,
    )
    _, _, _, _, info = env.step(retract_action.copy())
    target = validate_collected_safety_cost(
        info.get("safety_cost"),
        safety_cost_names=SAFETY_COST_NAMES,
        source="Lower-boundary targeted info['safety_cost']",
    )
    lower_record = _targeted_transition_record(
        case="lower_insertion_boundary_retraction",
        action=retract_action,
        prediction=prediction,
        target=target,
        inference_ms=inference_ms,
        info=info,
        positive_threshold=positive_threshold,
    )
    if not lower_record["translation_action_blocked"]:
        lower_record["status"] = "not_observed"
        lower_record["reason"] = (
            "The real retraction command was not reported as blocked by "
            "the intervention."
        )
    cases["lower_insertion_boundary"] = lower_record

    # Seed 301 is the repository's deterministic regression case for reaching
    # a fixed-tree endpoint with repeated real forward commands. We never
    # alter stop_device_at_tree_end or any other environment dynamics.
    observation, _ = env.reset(seed=_TARGETED_TREE_END_SEED)
    forward_action = np.asarray([1.0, 0.0], dtype=np.float32)
    forward_record: Optional[Dict[str, Any]] = None
    forward_is_tree_end = False
    forward_is_maximum_length = False
    stop_reason = "No forward blockage occurred within the episode limit."
    for step_index in range(env.max_episode_steps):
        prediction, _, inference_ms = predict_safety_before_step(
            agent,
            observation,
            forward_action,
        )
        (
            next_observation,
            _,
            terminated,
            truncated,
            info,
        ) = env.step(forward_action.copy())
        metrics = info.get("safety_metrics", {})
        blocked = bool(
            isinstance(metrics, Mapping)
            and metrics.get("translation_action_blocked", False)
        )
        if blocked:
            target = validate_collected_safety_cost(
                info.get("safety_cost"),
                safety_cost_names=SAFETY_COST_NAMES,
                source="Forward targeted info['safety_cost']",
            )
            forward_record = _targeted_transition_record(
                case="forward_blockage",
                action=forward_action,
                prediction=prediction,
                target=target,
                inference_ms=inference_ms,
                info=info,
                positive_threshold=positive_threshold,
                steps_to_case=step_index + 1,
            )
            forward_is_tree_end = _is_at_tree_end(env)
            inserted_length = float(
                metrics.get("inserted_length_mm", 0.0)
            )
            maximum_length = float(
                env.intervention.device_lengths_maximum[0]
            )
            raw_action = np.asarray(
                info.get("raw_action", [0.0, 0.0]),
                dtype=np.float64,
            )
            one_frame_motion = (
                abs(float(raw_action[0]))
                / float(env.intervention.fluoroscopy.image_frequency)
            )
            forward_is_maximum_length = bool(
                not forward_is_tree_end
                and inserted_length + one_frame_motion >= maximum_length
            )
            break
        if terminated or truncated:
            stop_reason = (
                "The episode terminated or truncated before a forward "
                "translation blockage was observed."
            )
            break
        observation = next_observation

    unsupported_maximum = {
        "case": "device_maximum_length_forward",
        "status": "unsupported",
        "reason": (
            "The unchanged environment's tree-end mask stops this device "
            "before its 450 mm maximum length, so a distinct maximum-length "
            "case cannot be reached safely in this task."
        ),
    }
    unsupported_tree_end = {
        "case": "vessel_tree_end_forward",
        "status": "not_observed",
        "reason": stop_reason,
        "seed": _TARGETED_TREE_END_SEED,
    }
    if forward_record is None:
        cases["vessel_tree_end"] = unsupported_tree_end
        cases["device_maximum_length"] = unsupported_maximum
    elif forward_is_tree_end:
        forward_record["case"] = "vessel_tree_end_forward"
        forward_record["seed"] = _TARGETED_TREE_END_SEED
        cases["vessel_tree_end"] = forward_record
        cases["device_maximum_length"] = unsupported_maximum
    elif forward_is_maximum_length:
        forward_record["case"] = "device_maximum_length_forward"
        forward_record["seed"] = _TARGETED_TREE_END_SEED
        cases["device_maximum_length"] = forward_record
        cases["vessel_tree_end"] = {
            **unsupported_tree_end,
            "reason": (
                "A maximum-device-length blockage occurred before a "
                "tree-end case was observed."
            ),
        }
    else:
        # A real blockage was observed, but available state could not
        # unambiguously attribute it to either requested controlled case.
        ambiguous = {
            **forward_record,
            "case": "unattributed_forward_blockage",
            "status": "observed_unattributed",
            "seed": _TARGETED_TREE_END_SEED,
        }
        cases["unattributed_forward_blockage"] = ambiguous
        cases["vessel_tree_end"] = {
            **unsupported_tree_end,
            "reason": "Observed forward blockage could not be attributed safely.",
        }
        cases["device_maximum_length"] = {
            **unsupported_maximum,
            "reason": "Observed forward blockage could not be attributed safely.",
        }
    return cases


def _inference_metrics(values_ms: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values_ms, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Inference-time samples must be a non-empty vector")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise FloatingPointError("Inference-time samples are invalid")
    return {
        "safety_inference_count": int(values.size),
        "safety_inference_ms_mean": float(np.mean(values)),
        "safety_inference_ms_std": float(np.std(values)),
        "safety_inference_ms_median": float(np.median(values)),
        "safety_inference_ms_p95": float(np.percentile(values, 95.0)),
        "safety_inference_ms_max": float(np.max(values)),
    }


def aggregate_collected_evaluation(
    collected: Mapping[str, Any],
    agent: TDMPC2Agent,
    *,
    positive_threshold: float,
) -> Dict[str, Any]:
    """Aggregate channel-specific prediction and timing metrics."""

    with torch.no_grad():
        aggregate = aggregate_safety_evaluation(
            prediction_transformed=collected["prediction_transformed"],
            target=collected["target"],
            model=agent.model,
            translation_positive_threshold=positive_threshold,
            safety_cost_names=SAFETY_COST_NAMES,
        )
    return {
        **dict(aggregate),
        **_inference_metrics(
            np.asarray(collected["inference_times_ms"])
        ),
    }


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        numeric = float(value)
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a positive integer") from exc
    if not np.isfinite(numeric) or numeric != parsed or parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _compare_fixed_split_metadata(
    checkpoint_split: Any,
    dataset_split: Mapping[str, Any],
) -> None:
    """Require checkpoint and supplied-dataset split metadata to be identical."""

    if not isinstance(checkpoint_split, Mapping):
        raise TypeError(
            "Checkpoint safety_auxiliary dataset split must be a mapping"
        )
    if set(checkpoint_split) != set(dataset_split):
        raise ValueError(
            "Checkpoint auxiliary dataset split keys do not match the "
            "supplied fixed dataset"
        )
    for key, expected in dataset_split.items():
        received = checkpoint_split[key]
        if isinstance(expected, np.ndarray):
            if (
                not isinstance(received, np.ndarray)
                or received.dtype != expected.dtype
                or not np.array_equal(received, expected)
            ):
                raise ValueError(
                    "Checkpoint auxiliary dataset fixed split mismatch in "
                    f"{key!r}"
                )
        elif received != expected:
            raise ValueError(
                "Checkpoint auxiliary dataset fixed split mismatch in "
                f"{key!r}: {received!r} != {expected!r}"
            )


def _validate_checkpoint_dataset_identity(
    checkpoint: Mapping[str, Any],
    config: Mapping[str, Any],
    dataset_state: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate Commit-4.6C checkpoint identity when metadata is available."""

    raw_aux_config = config.get("safety_aux", {})
    auxiliary_enabled = bool(
        isinstance(raw_aux_config, Mapping)
        and raw_aux_config.get("enabled", False)
    )
    checkpoint_auxiliary = checkpoint.get("safety_auxiliary")
    if checkpoint_auxiliary is None:
        return {
            "checked": False,
            "checkpoint_auxiliary_enabled": auxiliary_enabled,
            "reason": (
                "Checkpoint has no safety_auxiliary dataset identity metadata."
            ),
        }
    if not auxiliary_enabled:
        raise ValueError(
            "Checkpoint contains safety_auxiliary metadata while its saved "
            "configuration has safety_aux.enabled=false"
        )
    if not isinstance(checkpoint_auxiliary, Mapping):
        raise TypeError("Checkpoint safety_auxiliary metadata must be a mapping")
    if checkpoint_auxiliary.get("enabled") is not True:
        raise ValueError(
            "Checkpoint safety_auxiliary metadata must record enabled=true"
        )
    checkpoint_dataset = checkpoint_auxiliary.get("dataset")
    if not isinstance(checkpoint_dataset, Mapping):
        raise TypeError(
            "Checkpoint safety_auxiliary.dataset metadata must be a mapping"
        )

    replay_state = dataset_state["replay_state"]
    expected_identity = {
        "fingerprint": dataset_state["fingerprint"],
        "dataset_schema_version": int(dataset_state["schema_version"]),
        "replay_schema_version": int(replay_state["schema_version"]),
        "observation_dim": int(replay_state["observation_dim"]),
        "action_dim": int(replay_state["action_dim"]),
        "safety_cost_names": tuple(replay_state["safety_cost_names"]),
        "translation_block_reason_names": tuple(
            replay_state["translation_block_reason_names"]
        ),
        "curvature_stratum_names": tuple(
            replay_state["curvature_stratum_names"]
        ),
        "curvature_boundaries_mm_inv": tuple(
            float(value)
            for value in replay_state["curvature_boundaries_mm_inv"]
        ),
    }
    missing = sorted(
        {
            "resolved_path",
            "split",
            *expected_identity.keys(),
        }
        - checkpoint_dataset.keys()
    )
    if missing:
        raise ValueError(
            "Checkpoint safety_auxiliary.dataset metadata is missing "
            f"{missing}"
        )
    resolved_path = checkpoint_dataset["resolved_path"]
    if not isinstance(resolved_path, (str, Path)) or not str(resolved_path):
        raise ValueError(
            "Checkpoint auxiliary dataset resolved_path must be non-empty"
        )

    sequence_fields = {
        "safety_cost_names",
        "translation_block_reason_names",
        "curvature_stratum_names",
        "curvature_boundaries_mm_inv",
    }
    integer_fields = {
        "dataset_schema_version",
        "replay_schema_version",
        "observation_dim",
        "action_dim",
    }
    for key, expected in expected_identity.items():
        received = checkpoint_dataset[key]
        if key in sequence_fields:
            received = tuple(received)
        elif key in integer_fields:
            received = int(received)
        if received != expected:
            raise ValueError(
                "Checkpoint auxiliary dataset identity mismatch in "
                f"{key!r}: {received!r} != {expected!r}"
            )
    _compare_fixed_split_metadata(
        checkpoint_dataset["split"],
        dataset_state["split"],
    )
    return {
        "checked": True,
        "checkpoint_auxiliary_enabled": True,
        "fingerprint_match": True,
        "schema_match": True,
        "dimension_match": True,
        "channel_and_group_schema_match": True,
        "curvature_boundaries_match": True,
        "fixed_split_match": True,
        "path_equality_required": False,
        "checkpoint_recorded_path": str(resolved_path),
    }


def evaluate_fixed_safety_dataset(
    dataset_path: Path,
    checkpoint: Mapping[str, Any],
    config: Mapping[str, Any],
    agent: TDMPC2Agent,
    *,
    positive_threshold: float,
    batch_size: int = DEFAULT_OFFLINE_VALIDATION_BATCH_SIZE,
) -> Dict[str, Any]:
    """Evaluate every stored validation transition once without stEVE/SOFA."""

    parsed_batch_size = _positive_integer(
        batch_size,
        "--safety-batch-size",
    )
    expected_boundaries = curvature_boundaries_from_diagnostics(config)
    resolved_path = dataset_path.expanduser().resolve()
    dataset = load_dataset(
        resolved_path,
        expected_curvature_boundaries_mm_inv=expected_boundaries,
    )
    dataset_state = dataset.state_dict()
    replay_state = dataset_state["replay_state"]
    if int(replay_state["observation_dim"]) != agent.observation_dim:
        raise ValueError(
            "Safety dataset observation_dim does not match checkpoint agent: "
            f"{replay_state['observation_dim']} != {agent.observation_dim}"
        )
    if int(replay_state["action_dim"]) != agent.action_dim:
        raise ValueError(
            "Safety dataset action_dim does not match checkpoint agent: "
            f"{replay_state['action_dim']} != {agent.action_dim}"
        )
    if tuple(replay_state["safety_cost_names"]) != tuple(
        agent.safety_cost_names
    ):
        raise ValueError(
            "Safety dataset cost channel order does not match checkpoint agent"
        )
    if tuple(replay_state["translation_block_reason_names"]) != tuple(
        TRANSLATION_BLOCK_REASON_NAMES
    ):
        raise ValueError(
            "Safety dataset translation-block reason schema is incompatible"
        )
    if tuple(replay_state["curvature_stratum_names"]) != tuple(
        CURVATURE_STRATUM_NAMES
    ):
        raise ValueError(
            "Safety dataset curvature-stratum schema is incompatible"
        )

    checkpoint_identity = _validate_checkpoint_dataset_identity(
        checkpoint,
        config,
        dataset_state,
    )
    if dataset.validation_size <= 0:
        raise ValueError(
            "Safety auxiliary dataset has no fixed validation samples"
        )
    validation_state = dataset.validation_buffer().state_dict()
    sample_count = int(validation_state["size"])
    if sample_count != dataset.validation_size:
        raise RuntimeError(
            "Fixed validation buffer size does not match stored split metadata"
        )

    targets = validation_state["safety_cost"]
    stored_strata = validation_state["curvature_stratum_id"]
    derived_strata = np.searchsorted(
        np.asarray(expected_boundaries, dtype=np.float64),
        targets[:, 0].astype(np.float64),
        side="right",
    ).astype(np.int64)
    if not np.array_equal(stored_strata, derived_strata):
        mismatch_count = int(np.count_nonzero(stored_strata != derived_strata))
        raise ValueError(
            "Safety dataset stored curvature strata disagree with its targets "
            f"and boundaries for {mismatch_count} validation samples"
        )

    # The shared helper uses validation_state["action"], which is the requested
    # normalized action.  It never substitutes the post-mask applied_action.
    _synchronize_if_cuda(agent.device)
    evaluation_start_ns = time.perf_counter_ns()
    metrics = evaluate_fixed_safety_validation(
        agent,
        validation_state,
        batch_size=parsed_batch_size,
        translation_positive_threshold=positive_threshold,
    )
    _synchronize_if_cuda(agent.device)
    evaluation_ms = (time.perf_counter_ns() - evaluation_start_ns) / 1.0e6
    reason_ids = validation_state["translation_block_reason_id"]
    target_positive = targets[:, 1] > 0.0
    reason_positive = reason_ids != TRANSLATION_BLOCK_REASON_NAMES.index(
        "none"
    )
    return {
        "dataset": {
            "path": resolved_path,
            "fingerprint": dataset.fingerprint,
            "dataset_schema_version": dataset.schema_version,
            "replay_schema_version": int(replay_state["schema_version"]),
            "total_size": dataset.total_size,
            "train_size": dataset.train_size,
            "validation_size": dataset.validation_size,
            "observation_dim": int(replay_state["observation_dim"]),
            "action_dim": int(replay_state["action_dim"]),
            "safety_cost_names": replay_state["safety_cost_names"],
            "translation_block_reason_names": replay_state[
                "translation_block_reason_names"
            ],
            "curvature_stratum_names": replay_state[
                "curvature_stratum_names"
            ],
            "curvature_boundaries_mm_inv": replay_state[
                "curvature_boundaries_mm_inv"
            ],
            "split": {
                key: value
                for key, value in dataset_state["split"].items()
                if key not in {"train_indices", "validation_indices"}
            },
        },
        "checkpoint_dataset_identity": checkpoint_identity,
        "evaluation": {
            "action_field": "action",
            "action_semantics": (
                "Requested normalized action supplied to the environment; "
                "applied_action is not used as Safety Head input."
            ),
            "fixed_split": "validation",
            "sample_count": sample_count,
            "inference_batch_size": parsed_batch_size,
            "inference_batch_count": int(metrics["inference_batch_count"]),
            "complete_validation_evaluation_ms": float(evaluation_ms),
            "prediction_aggregation": (
                "All minibatch predictions were concatenated before one "
                "complete-split metric aggregation."
            ),
            "translation_target_reason_disagreement_count": int(
                np.count_nonzero(target_positive != reason_positive)
            ),
            "curvature_target_stratum_disagreement_count": 0,
        },
        "metrics": metrics,
    }


def _episode_summary(
    episode_records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    rewards = np.asarray(
        [record["reward"] for record in episode_records],
        dtype=np.float64,
    )
    lengths = np.asarray(
        [record["length"] for record in episode_records],
        dtype=np.float64,
    )
    successes = np.asarray(
        [record["success"] for record in episode_records],
        dtype=np.float64,
    )
    return {
        "episode_count": int(len(episode_records)),
        "mean_episode_reward": float(np.mean(rewards)),
        "std_episode_reward": float(np.std(rewards)),
        "mean_episode_length": float(np.mean(lengths)),
        "success_rate": float(np.mean(successes)),
        "simulation_error_episode_count": int(
            sum(
                bool(record["simulation_error"])
                for record in episode_records
            )
        ),
    }


def _json_safe(value: Any) -> Any:
    """Convert tensors/NumPy values and unavailable non-finite metrics."""

    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _json_safe(value.detach().cpu().item())
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, str):
        return value
    return str(value)


def _print_metric_report(metrics: Mapping[str, Any]) -> None:
    print("\nSafety Head aggregate metrics:")
    safe_metrics = _json_safe(metrics)
    for name in sorted(safe_metrics):
        value = safe_metrics[name]
        if value is None:
            rendered = "unavailable"
        elif isinstance(value, float):
            rendered = f"{value:.8g}"
        else:
            rendered = str(value)
        print(f"  {name}: {rendered}")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    """Run a complete diagnostic evaluation and return a JSON-safe report."""

    positive_threshold = _validate_positive_threshold(
        args.positive_threshold
    )
    offline_batch_size = _positive_integer(
        getattr(
            args,
            "safety_batch_size",
            DEFAULT_OFFLINE_VALIDATION_BATCH_SIZE,
        ),
        "--safety-batch-size",
    )
    safety_dataset_path = getattr(args, "safety_dataset", None)
    (
        checkpoint,
        config,
        agent,
        device,
    ) = load_evaluation_model(
        args.checkpoint,
        config_path=args.config,
        requested_device=args.device,
        auxiliary_dataset_override=safety_dataset_path,
    )
    evaluation_config = config["evaluation"]
    policy_requested = (
        safety_dataset_path is None or args.episodes is not None
    )
    targeted_requested = bool(args.targeted_blockage)
    episodes = (
        int(
            args.episodes
            if args.episodes is not None
            else evaluation_config["episodes"]
        )
        if policy_requested
        else 0
    )
    base_seed = int(
        args.seed
        if args.seed is not None
        else evaluation_config["seed"]
    )
    if policy_requested and episodes <= 0:
        raise ValueError("Evaluation episode count must be positive")

    modes = []
    if safety_dataset_path is not None:
        modes.append("fixed-offline-validation")
    if policy_requested:
        modes.append("policy")
    if targeted_requested:
        modes.append("targeted-blockage")
    print(
        "Safety Head evaluation: "
        f"device={device}, modes={','.join(modes)}, "
        f"episodes={episodes if policy_requested else 'disabled'}, "
        f"seed={base_seed}, "
        f"positive_threshold={positive_threshold:g}"
    )
    print(
        "Safety predictions are diagnostic only and are not used by MPC "
        "or action selection."
    )
    env: Optional[StEVEEnv] = None
    collected: Optional[Dict[str, Any]] = None
    policy_metrics: Optional[Dict[str, Any]] = None
    targeted: Optional[Dict[str, Any]] = None
    fixed_validation: Optional[Dict[str, Any]] = None
    try:
        if safety_dataset_path is not None:
            fixed_validation = evaluate_fixed_safety_dataset(
                safety_dataset_path,
                checkpoint,
                config,
                agent,
                positive_threshold=positive_threshold,
                batch_size=offline_batch_size,
            )

        if policy_requested or targeted_requested:
            env = _make_evaluation_environment(config, agent)
        if policy_requested:
            if env is None:
                raise RuntimeError(
                    "Policy evaluation requested without an environment"
                )
            collected = collect_policy_evaluation(
                env,
                agent,
                episodes=episodes,
                base_seed=base_seed,
            )
            policy_metrics = aggregate_collected_evaluation(
                collected,
                agent,
                positive_threshold=positive_threshold,
            )
        if targeted_requested:
            if env is None:
                raise RuntimeError(
                    "Targeted blockage evaluation requested without an "
                    "environment"
                )
            targeted = evaluate_targeted_blockage(
                env,
                agent,
                base_seed=base_seed + episodes,
                positive_threshold=positive_threshold,
            )
    finally:
        if env is not None:
            env.close()

    episode_records = collected["episodes"] if collected is not None else []
    episode_summary = (
        _episode_summary(episode_records)
        if episode_records
        else None
    )

    report = _json_safe(
        {
            "checkpoint": args.checkpoint.expanduser().resolve(),
            "checkpoint_step": int(
                checkpoint.get("total_env_steps", -1)
            ),
            "device": str(device),
            "safety_cost_names": SAFETY_COST_NAMES,
            "positive_threshold": positive_threshold,
            "positive_threshold_note": (
                "Diagnostic intervention-blockage threshold only; "
                "not a clinical safety threshold."
            ),
            "evaluation_modes": modes,
            "online_environment_constructed": bool(
                policy_requested or targeted_requested
            ),
            "episode_summary": episode_summary,
            "episodes": episode_records,
            "metrics": policy_metrics,
            "fixed_validation": fixed_validation,
            "targeted_blockage": targeted,
            "planning_isolation": (
                "Safety predictions never altered MPC scores/actions. Offline "
                "validation used only stored observations and requested actions."
            ),
        }
    )
    if report["metrics"] is not None:
        _print_metric_report(report["metrics"])
    summary = report["episode_summary"]
    if summary is not None:
        print(
            "\nEpisode summary: "
            f"reward={summary['mean_episode_reward']:.3f} +/- "
            f"{summary['std_episode_reward']:.3f}, "
            f"length={summary['mean_episode_length']:.1f}, "
            f"success_rate={summary['success_rate']:.3f}"
        )
    if report["fixed_validation"] is not None:
        offline_report = report["fixed_validation"]
        offline_metrics = offline_report["metrics"]
        print(
            "\nFixed auxiliary validation: "
            f"samples={offline_metrics['sample_count']}, "
            f"fingerprint={offline_report['dataset']['fingerprint']}"
        )
        print(
            json.dumps(
                {
                    "overall": offline_metrics["overall"],
                    "translation_blockage": offline_metrics[
                        "translation_blockage"
                    ],
                    "per_translation_reason": offline_metrics[
                        "per_translation_reason"
                    ],
                    "per_curvature_stratum": offline_metrics[
                        "per_curvature_stratum"
                    ],
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
    if report["targeted_blockage"] is not None:
        print("\nTargeted blockage diagnostics:")
        print(
            json.dumps(
                report["targeted_blockage"],
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )

    if args.output_json is not None:
        output_path = args.output_json.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as stream:
            json.dump(
                report,
                stream,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
        print(f"Saved Safety Head evaluation report: {output_path}")
    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
