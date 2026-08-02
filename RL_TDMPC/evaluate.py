#!/usr/bin/env python3
"""Load a TD-MPC2 checkpoint and run reproducible evaluation episodes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import cv2
import numpy as np
import torch
from eve.intervention import TRANSLATION_BLOCK_REASON_NAMES

from envs.safety import SAFETY_COST_NAMES
from envs.steve_env import make_steve_env
from train import resolved_safety_mpc_config, validate_checkpoint_schema
from tdmpc2.agent import TDMPC2Agent
from tdmpc2.common import (
    MetricLogger,
    atomic_json_save,
    build_diagnostics_agent_config,
    build_integral_lagrangian_config,
    build_safety_agent_config,
    build_safety_mpc_agent_config,
    load_config,
    load_torch_checkpoint,
    select_device,
    set_seed,
)
from tdmpc2.integral_lagrangian import IntegralLagrangianController


BASELINE_EVALUATION_REPORT_SCHEMA_VERSION = 1
EVALUATION_REPORT_SCHEMA_VERSION = 2
_PLANNER_METRIC_NAMES = (
    "safety_mpc_selected_risk",
    "safety_mpc_selected_penalty",
    "safety_mpc_task_scale",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional config override; checkpoint config is used by default",
    )
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--eval-safety-mpc",
        choices=("checkpoint", "disabled", "enabled"),
        default="checkpoint",
        help=(
            "Evaluation-only planner setting. The checkpoint setting is used "
            "by default; disabled/enabled are explicit runtime-only overrides."
        ),
    )
    parser.add_argument(
        "--eval-safety-mpc-alpha",
        type=float,
        default=None,
        help=(
            "Required nonnegative alpha with --eval-safety-mpc enabled; "
            "rejected for checkpoint/disabled modes."
        ),
    )
    parser.add_argument(
        "--eval-integral-lagrangian",
        choices=("enabled", "disabled"),
        default=None,
        help=(
            "Explicit evaluation-only dual-controller override. By default "
            "the checkpoint/config integral_lagrangian setting is used."
        ),
    )
    parser.add_argument("--eval-dual-initial-alpha", type=float, default=None)
    parser.add_argument("--eval-dual-integral-gain", type=float, default=None)
    parser.add_argument("--eval-dual-cost-limit", type=float, default=None)
    parser.add_argument("--eval-dual-alpha-max", type=float, default=None)
    parser.add_argument(
        "--eval-dual-window-episodes", type=int, default=None
    )
    parser.add_argument(
        "--eval-dual-warmup-episodes", type=int, default=None
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write strict per-seed and aggregate evaluation results as JSON",
    )
    parser.add_argument(
        "--log-directory",
        type=Path,
        default=None,
        help="Optional isolated MetricLogger directory for this evaluation",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Display evaluation in a SofaPygame OpenGL window",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Save rendered evaluation to an MP4 file (also enables rendering)",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=None,
        help="Output video frame rate (default: environment render FPS)",
    )
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
        **build_safety_mpc_agent_config(config),
        **build_diagnostics_agent_config(config),
    }


def resolve_evaluation_safety_mpc_config(
    checkpoint_config: Mapping[str, Any],
    *,
    override: str,
    alpha: Optional[float],
) -> Dict[str, Any]:
    """Resolve an explicit runtime-only planner override.

    This helper is deliberately evaluation-local. Checkpoint validation and
    agent loading must already have used the unmodified checkpoint settings.
    """

    checkpoint_settings = resolved_safety_mpc_config(checkpoint_config)
    if override == "checkpoint":
        if alpha is not None:
            raise ValueError(
                "--eval-safety-mpc-alpha is only valid with "
                "--eval-safety-mpc enabled"
            )
        return copy.deepcopy(checkpoint_settings)
    if override == "disabled":
        if alpha is not None:
            raise ValueError(
                "--eval-safety-mpc-alpha is only valid with "
                "--eval-safety-mpc enabled"
            )
        evaluation_settings = copy.deepcopy(checkpoint_settings)
        evaluation_settings["enabled"] = False
        return evaluation_settings
    if override != "enabled":
        raise ValueError(f"Unsupported evaluation Safety-MPC override {override!r}")
    if alpha is None:
        raise ValueError(
            "--eval-safety-mpc enabled requires --eval-safety-mpc-alpha"
        )
    if isinstance(alpha, bool) or not np.isfinite(float(alpha)) or float(alpha) < 0:
        raise ValueError(
            "Evaluation Safety-MPC alpha must be finite and nonnegative"
        )
    evaluation_settings = copy.deepcopy(checkpoint_settings)
    evaluation_settings["enabled"] = True
    evaluation_settings["alpha"] = float(alpha)
    # Reuse the production parser so evaluation cannot introduce a looser
    # configuration schema than training.
    resolved_safety_mpc_config({"safety_mpc": evaluation_settings})
    return evaluation_settings


def resolve_evaluation_integral_lagrangian_config(
    checkpoint_config: Mapping[str, Any],
    *,
    override: Optional[str],
    initial_alpha: Optional[float],
    integral_gain: Optional[float],
    cost_limit: Optional[float],
    alpha_max: Optional[float],
    rolling_window_episodes: Optional[int],
    warmup_episodes: Optional[int],
) -> Dict[str, Any]:
    """Resolve strict evaluation-only dual settings without mutating config."""

    checkpoint_settings = build_integral_lagrangian_config(checkpoint_config)
    overrides = {
        "initial_alpha": initial_alpha,
        "integral_gain": integral_gain,
        "cost_limit": cost_limit,
        "alpha_max": alpha_max,
        "rolling_window_episodes": rolling_window_episodes,
        "warmup_episodes": warmup_episodes,
    }
    supplied_keys = [key for key, value in overrides.items() if value is not None]
    if override is None:
        if supplied_keys:
            raise ValueError(
                "Dual parameter overrides require explicit "
                "--eval-integral-lagrangian enabled"
            )
        return copy.deepcopy(checkpoint_settings)
    if override == "disabled":
        if supplied_keys:
            raise ValueError(
                "Dual parameter overrides are invalid when Integral "
                "Lagrangian evaluation is disabled"
            )
        resolved = copy.deepcopy(checkpoint_settings)
        resolved["enabled"] = False
        return build_integral_lagrangian_config(
            {"integral_lagrangian": resolved}
        )
    if override != "enabled":
        raise ValueError(
            f"Unsupported Integral Lagrangian override {override!r}"
        )
    resolved = copy.deepcopy(checkpoint_settings)
    resolved["enabled"] = True
    for key, value in overrides.items():
        if value is not None:
            resolved[key] = value
    return build_integral_lagrangian_config(
        {"integral_lagrangian": resolved}
    )


def validate_evaluation_planner_compatibility(
    safety_mpc: Mapping[str, Any],
    integral_lagrangian: Mapping[str, Any],
    *,
    mpc_enabled: bool = True,
) -> None:
    """Reject an active dual controller without active Safety-MPC scoring."""

    if integral_lagrangian["enabled"] and not safety_mpc["enabled"]:
        raise ValueError(
            "Integral Lagrangian evaluation requires Safety-MPC to be enabled"
        )
    if integral_lagrangian["enabled"] and not mpc_enabled:
        raise ValueError(
            "Integral Lagrangian evaluation requires planning.mpc=true"
        )
    if (
        safety_mpc["enabled"]
        and float(safety_mpc["alpha"]) > 0.0
        and not mpc_enabled
    ):
        raise ValueError(
            "Active Safety-MPC evaluation requires planning.mpc=true"
        )


def apply_evaluation_safety_mpc_config(
    agent: TDMPC2Agent,
    *,
    checkpoint_settings: Mapping[str, Any],
    evaluation_settings: Mapping[str, Any],
) -> None:
    """Apply only enabled/alpha to an already strictly loaded temporary agent."""

    immutable_keys = (
        "translation_risk_cap",
        "minimum_task_scale",
        "aggregation",
    )
    for key in immutable_keys:
        if evaluation_settings[key] != checkpoint_settings[key]:
            raise ValueError(
                f"Evaluation-only Safety-MPC cannot override {key}"
            )
    agent.set_safety_mpc_runtime_alpha(
        float(evaluation_settings["alpha"]),
        enabled=bool(evaluation_settings["enabled"]),
    )


def _update_state_digest(digest: Any, value: Any) -> None:
    """Add a deterministic nested tensor/state representation to ``digest``."""

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor:")
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
        digest.update(
            tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        )
        return
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"ndarray:")
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(repr(tuple(array.shape)).encode("utf-8"))
        digest.update(array.tobytes())
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping:{")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _update_state_digest(digest, key)
            _update_state_digest(digest, value[key])
        digest.update(b"}")
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"sequence:[")
        for item in value:
            _update_state_digest(digest, item)
        digest.update(b"]")
        return
    if isinstance(value, np.generic):
        value = value.item()
    digest.update(type(value).__name__.encode("utf-8"))
    digest.update(b":")
    digest.update(repr(value).encode("utf-8"))


def state_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _update_state_digest(digest, value)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def paths_alias(first: Path, second: Path) -> bool:
    first_resolved = first.expanduser().resolve()
    second_resolved = second.expanduser().resolve()
    if first_resolved == second_resolved:
        return True
    return (
        first_resolved.exists()
        and second_resolved.exists()
        and first_resolved.samefile(second_resolved)
    )


def finite_statistics(
    values: Sequence[float],
    *,
    include_median: bool = False,
    include_std: bool = False,
    include_max: bool = False,
) -> Optional[Dict[str, Any]]:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise FloatingPointError("Evaluation statistics contain NaN or infinity")
    statistics: Dict[str, Any] = {
        "count": int(array.size),
        "mean": float(np.mean(array)),
    }
    if include_median:
        statistics["median"] = float(np.median(array))
    if include_std:
        statistics["std"] = float(np.std(array))
    if include_max:
        statistics["max"] = float(np.max(array))
    return statistics


def compute_episode_tree_end_blockage_fraction(
    vessel_tree_end_blockage_count: int,
    episode_transition_count: int,
) -> float:
    """Compute the sole real-environment feedback used by the dual update."""

    for name, value in (
        ("vessel_tree_end_blockage_count", vessel_tree_end_blockage_count),
        ("episode_transition_count", episode_transition_count),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"{name} must be an integer")
        if int(value) < 0:
            raise ValueError(f"{name} must be nonnegative")
    if vessel_tree_end_blockage_count > episode_transition_count:
        raise ValueError(
            "vessel_tree_end_blockage_count cannot exceed "
            "episode_transition_count"
        )
    return float(vessel_tree_end_blockage_count) / max(
        int(episode_transition_count), 1
    )


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    for label, destination in (
        ("--output-json", args.output_json),
        ("--video", args.video),
    ):
        if destination is not None and paths_alias(
            destination,
            checkpoint_path,
        ):
            raise ValueError(
                f"{label} must not overwrite or alias the checkpoint path"
            )
    checkpoint_file_hash_before = file_sha256(checkpoint_path)
    checkpoint = load_torch_checkpoint(checkpoint_path, map_location="cpu")
    if args.config is not None:
        config = load_config(args.config)
    elif "config" in checkpoint:
        config = copy.deepcopy(checkpoint["config"])
    else:
        raise KeyError("Checkpoint has no saved config; pass --config explicitly")
    validate_checkpoint_schema(
        checkpoint,
        config=config,
        source=f"Evaluation checkpoint {checkpoint_path}",
    )
    checkpoint_config = checkpoint["config"]
    checkpoint_safety_mpc = resolved_safety_mpc_config(checkpoint_config)
    checkpoint_integral_lagrangian = build_integral_lagrangian_config(
        checkpoint_config
    )
    checkpoint_metadata_snapshot = {
        "top_level": copy.deepcopy(checkpoint.get("safety_mpc_config")),
        "embedded": copy.deepcopy(checkpoint["config"].get("safety_mpc")),
        "agent": copy.deepcopy(
            checkpoint["agent"].get("safety_mpc_config")
        ),
    }
    evaluation_safety_mpc = resolve_evaluation_safety_mpc_config(
        config,
        override=args.eval_safety_mpc,
        alpha=args.eval_safety_mpc_alpha,
    )
    evaluation_integral_lagrangian = (
        resolve_evaluation_integral_lagrangian_config(
            config,
            override=args.eval_integral_lagrangian,
            initial_alpha=args.eval_dual_initial_alpha,
            integral_gain=args.eval_dual_integral_gain,
            cost_limit=args.eval_dual_cost_limit,
            alpha_max=args.eval_dual_alpha_max,
            rolling_window_episodes=args.eval_dual_window_episodes,
            warmup_episodes=args.eval_dual_warmup_episodes,
        )
    )
    validate_evaluation_planner_compatibility(
        evaluation_safety_mpc,
        evaluation_integral_lagrangian,
        mpc_enabled=bool(config["planning"]["mpc"]),
    )
    safety_settings_report = {
        "checkpoint_safety_mpc": checkpoint_safety_mpc,
        "evaluation_safety_mpc": evaluation_safety_mpc,
        "override_applied": evaluation_safety_mpc != checkpoint_safety_mpc,
        "override_mode": args.eval_safety_mpc,
    }
    integral_settings_report = {
        "checkpoint_integral_lagrangian": checkpoint_integral_lagrangian,
        "evaluation_integral_lagrangian": evaluation_integral_lagrangian,
        "integral_lagrangian_override_applied": (
            evaluation_integral_lagrangian
            != checkpoint_integral_lagrangian
        ),
        "integral_lagrangian_override_mode": (
            args.eval_integral_lagrangian
            if args.eval_integral_lagrangian is not None
            else ("external_config" if args.config is not None else "checkpoint")
        ),
        "integral_lagrangian_effective_initial_alpha_source": (
            "integral_lagrangian.initial_alpha"
            if evaluation_integral_lagrangian["enabled"]
            else "safety_mpc.alpha"
        ),
        "effective_initial_planner_alpha": (
            evaluation_integral_lagrangian["initial_alpha"]
            if evaluation_integral_lagrangian["enabled"]
            else evaluation_safety_mpc["alpha"]
        ),
    }
    settings_report = {
        **safety_settings_report,
        **integral_settings_report,
    }
    print(
        "Safety-MPC and Integral Lagrangian checkpoint/evaluation settings:\n"
        + json.dumps(
            settings_report,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )

    evaluation = config["evaluation"]
    episodes = int(
        args.episodes
        if args.episodes is not None
        else evaluation["episodes"]
    )
    base_seed = int(args.seed if args.seed is not None else evaluation["seed"])
    if episodes <= 0:
        raise ValueError("Evaluation episode count must be positive")
    requested_device = args.device or config["training"]["device"]
    device = select_device(requested_device)
    set_seed(base_seed)

    record_video = args.video is not None
    render_enabled = args.render or record_video
    environment_config = dict(config["environment"])
    if render_enabled:
        environment_config["render_mode"] = "human"
    env = make_steve_env(environment_config)
    observation_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    agent = TDMPC2Agent(
        observation_dim,
        action_dim,
        build_agent_config(config),
        episode_length=int(config["environment"]["max_episode_steps"]),
        device=device,
    )
    agent.load_state_dict(checkpoint["agent"], load_optimizers=False)
    checkpoint_model_hash = state_sha256(checkpoint["agent"]["model"])
    loaded_model_hash = state_sha256(agent.model.state_dict())
    if loaded_model_hash != checkpoint_model_hash:
        env.close()
        raise RuntimeError(
            "Loaded evaluation model weights do not match the checkpoint"
        )
    model_hash_before = loaded_model_hash
    model_optimizer_hash_before = state_sha256(
        agent.model_optimizer.state_dict()
    )
    policy_optimizer_hash_before = state_sha256(
        agent.policy_optimizer.state_dict()
    )
    scale_hash_before = state_sha256(agent.scale.state_dict())
    agent_config_hash_before = state_sha256(agent.config)
    update_count_before = agent.update_count
    parameter_gradient_count_before = sum(
        int(parameter.grad is not None)
        for parameter in agent.model.parameters()
    )

    apply_evaluation_safety_mpc_config(
        agent,
        checkpoint_settings=checkpoint_safety_mpc,
        evaluation_settings=evaluation_safety_mpc,
    )
    integral_controller: Optional[IntegralLagrangianController] = None
    if evaluation_integral_lagrangian["enabled"]:
        integral_controller = IntegralLagrangianController(
            evaluation_integral_lagrangian
        )
        # Construction resets the controller, but keeping reset explicit makes
        # the new-evaluation-run boundary auditable.
        integral_controller.reset()
        agent.set_safety_mpc_runtime_alpha(
            integral_controller.current_alpha,
            enabled=True,
        )
    log_directory = (
        args.log_directory
        if args.log_directory is not None
        else evaluation["log_directory"]
    )
    logger = MetricLogger(log_directory)

    rewards: List[float] = []
    lengths: List[int] = []
    successes: List[bool] = []
    episode_results: List[Dict[str, Any]] = []
    selected_risks: List[float] = []
    selected_penalties: List[float] = []
    task_scales: List[float] = []
    total_translation_blockage_count = 0
    total_transition_count = 0
    blockage_episode_count = 0
    reason_totals = {
        reason: 0 for reason in TRANSLATION_BLOCK_REASON_NAMES
    }
    simulation_error_transition_count = 0
    simulation_error_episode_count = 0
    expected_active_planner_call_count = 0
    video_writer: Optional[cv2.VideoWriter] = None
    video_path = args.video.expanduser().resolve() if record_video else None
    video_fps = float(
        args.video_fps
        if args.video_fps is not None
        else env.metadata.get("render_fps", 7.5)
    )
    if video_fps <= 0:
        raise ValueError("Video FPS must be positive")

    def render_frame() -> None:
        nonlocal video_writer
        if not render_enabled:
            return
        frame = env.render()
        if not record_video:
            return
        if frame is None:
            raise RuntimeError("The environment did not return a frame for recording")
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise RuntimeError(f"Expected an HxWx3 RGB frame, got {frame.shape}")
        frame = np.clip(frame, 0, 255).astype(np.uint8, copy=False)
        if video_writer is None:
            assert video_path is not None
            video_path.parent.mkdir(parents=True, exist_ok=True)
            height, width = frame.shape[:2]
            video_writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                video_fps,
                (width, height),
            )
            if not video_writer.isOpened():
                video_writer.release()
                video_writer = None
                raise RuntimeError(f"Could not open MP4 video writer: {video_path}")
        # OpenCV video encoders expect BGR rather than the renderer's RGB.
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    try:
        for episode in range(episodes):
            episode_seed = base_seed + episode
            if integral_controller is not None:
                # Alpha is fixed for the entire episode. The controller only
                # changes its own next-episode state after termination below.
                agent.set_safety_mpc_runtime_alpha(
                    integral_controller.current_alpha,
                    enabled=True,
                )
            episode_runtime_alpha = agent.safety_mpc_runtime_alpha
            episode_safety_mpc_active = agent.safety_mpc_active
            # The world-model planner is sampling-based. Reseeding makes the
            # no-exploration eval trajectory reproducible across program runs.
            set_seed(episode_seed)
            env.action_space.seed(episode_seed)
            observation, _ = env.reset(seed=episode_seed)
            render_frame()
            terminated = truncated = False
            episode_reward = 0.0
            episode_length = 0
            success = False
            simulation_error = False
            episode_simulation_error_transitions = 0
            episode_blockage_count = 0
            episode_reason_counts = {
                reason: 0 for reason in TRANSLATION_BLOCK_REASON_NAMES
            }
            episode_selected_risks: List[float] = []
            episode_selected_penalties: List[float] = []
            episode_task_scales: List[float] = []
            while not (terminated or truncated):
                action = agent.act(
                    observation,
                    first_step=episode_length == 0,
                    eval_mode=True,
                )
                planner_metrics = agent.last_safety_mpc_metrics
                if agent.safety_mpc_active != episode_safety_mpc_active:
                    raise RuntimeError(
                        "Safety-MPC active state changed inside an episode"
                    )
                if episode_safety_mpc_active:
                    if not isinstance(planner_metrics, Mapping):
                        raise RuntimeError(
                            "Active Safety-MPC did not expose planner metrics"
                        )
                    planner_values: Dict[str, float] = {}
                    for name in _PLANNER_METRIC_NAMES:
                        if name not in planner_metrics:
                            raise KeyError(
                                f"Active Safety-MPC metric {name!r} is missing"
                            )
                        value = float(planner_metrics[name])
                        if not np.isfinite(value):
                            raise FloatingPointError(
                                f"Active Safety-MPC metric {name!r} is non-finite"
                            )
                        planner_values[name] = value
                    episode_selected_risks.append(
                        planner_values["safety_mpc_selected_risk"]
                    )
                    episode_selected_penalties.append(
                        planner_values["safety_mpc_selected_penalty"]
                    )
                    episode_task_scales.append(
                        planner_values["safety_mpc_task_scale"]
                    )
                elif planner_metrics is not None:
                    raise RuntimeError(
                        "Inactive Safety-MPC unexpectedly exposed planner metrics"
                    )

                observation, reward, terminated, truncated, info = env.step(action)
                render_frame()
                episode_reward += float(reward)
                episode_length += 1
                success = success or bool(info.get("is_success", False))
                simulation_error_step = bool(
                    info.get("simulation_error", False)
                )
                simulation_error = simulation_error or simulation_error_step
                episode_simulation_error_transitions += int(
                    simulation_error_step
                )

                safety_metrics = info.get("safety_metrics")
                if not isinstance(safety_metrics, Mapping):
                    raise TypeError(
                        "Environment info['safety_metrics'] must be a mapping"
                    )
                blocked_value = safety_metrics.get(
                    "translation_action_blocked"
                )
                if type(blocked_value) not in (bool, np.bool_):
                    raise TypeError(
                        "translation_action_blocked must be a bool"
                    )
                translation_blocked = bool(blocked_value)
                block_reason = safety_metrics.get(
                    "translation_block_reason"
                )
                if (
                    not isinstance(block_reason, str)
                    or block_reason not in episode_reason_counts
                ):
                    raise ValueError(
                        f"Unknown translation block reason {block_reason!r}"
                    )
                if translation_blocked != (block_reason != "none"):
                    raise RuntimeError(
                        "Translation blockage flag/reason mismatch: "
                        f"{translation_blocked}/{block_reason!r}"
                    )
                episode_blockage_count += int(translation_blocked)
                episode_reason_counts[block_reason] += 1

            non_none_reason_count = sum(
                count
                for reason, count in episode_reason_counts.items()
                if reason != "none"
            )
            if non_none_reason_count != episode_blockage_count:
                raise RuntimeError(
                    "Translation blockage count does not match reason totals"
                )
            planner_call_count = len(episode_selected_risks)
            if episode_safety_mpc_active:
                if not (
                    planner_call_count
                    == len(episode_selected_penalties)
                    == len(episode_task_scales)
                    == episode_length
                ):
                    raise RuntimeError(
                        "Active planner metric counts do not match transitions"
                    )
            elif planner_call_count != 0:
                raise RuntimeError("Inactive planner metric count must be zero")
            expected_active_planner_call_count += (
                episode_length if episode_safety_mpc_active else 0
            )

            episode_tree_end_blockage_count = episode_reason_counts[
                "vessel_tree_end"
            ]
            episode_tree_end_blockage_fraction = (
                compute_episode_tree_end_blockage_fraction(
                    episode_tree_end_blockage_count, episode_length
                )
            )
            integral_episode_record: Optional[Dict[str, Any]] = None
            if integral_controller is not None:
                integral_episode_record = (
                    integral_controller.observe_episode_cost(
                        episode_tree_end_blockage_fraction
                    )
                )
                if not np.isclose(
                    integral_episode_record["alpha_before_update"],
                    episode_runtime_alpha,
                    rtol=0.0,
                    atol=0.0,
                ):
                    raise RuntimeError(
                        "Integral controller alpha was not aligned with the "
                        "completed episode"
                    )

            rewards.append(episode_reward)
            lengths.append(episode_length)
            successes.append(success)
            selected_risks.extend(episode_selected_risks)
            selected_penalties.extend(episode_selected_penalties)
            task_scales.extend(episode_task_scales)
            total_transition_count += episode_length
            total_translation_blockage_count += episode_blockage_count
            blockage_episode_count += int(episode_blockage_count > 0)
            simulation_error_transition_count += (
                episode_simulation_error_transitions
            )
            simulation_error_episode_count += int(simulation_error)
            for reason, count in episode_reason_counts.items():
                reason_totals[reason] += count

            episode_result = {
                "episode_index": episode + 1,
                "seed": episode_seed,
                "reward": episode_reward,
                "length": episode_length,
                "success": success,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "transition_count": episode_length,
                "translation_blockage_count": episode_blockage_count,
                "translation_blockage_fraction": (
                    episode_blockage_count / episode_length
                ),
                "translation_blockage_episode": episode_blockage_count > 0,
                "lower_insertion_boundary_blockage_count": (
                    episode_reason_counts["lower_insertion_boundary"]
                ),
                "device_length_limit_blockage_count": (
                    episode_reason_counts["device_length_limit"]
                ),
                "vessel_tree_end_blockage_count": (
                    episode_tree_end_blockage_count
                ),
                "other_blockage_count": episode_reason_counts["other"],
                "simulation_error_transition_count": (
                    episode_simulation_error_transitions
                ),
                "simulation_error": simulation_error,
                "planner_call_count": planner_call_count,
                "selected_translation_risk": finite_statistics(
                    episode_selected_risks,
                    include_max=True,
                ),
                "selected_safety_penalty": finite_statistics(
                    episode_selected_penalties,
                    include_max=True,
                ),
                "task_scale": finite_statistics(
                    episode_task_scales,
                    include_std=True,
                ),
            }
            if integral_episode_record is not None:
                episode_result.update(
                    {
                        "episode_tree_end_blockage_fraction": (
                            episode_tree_end_blockage_fraction
                        ),
                        "integral_lagrangian": integral_episode_record,
                    }
                )
            episode_results.append(episode_result)
            episode_log_record = {
                "checkpoint_step": int(checkpoint.get("total_env_steps", -1)),
                "episode": episode + 1,
                "seed": episode_seed,
                "episode_reward": episode_reward,
                "episode_length": episode_length,
                "success": success,
                "simulation_error": simulation_error,
                "translation_blockage_count": episode_blockage_count,
                "translation_blockage_fraction": (
                    episode_blockage_count / episode_length
                ),
                "lower_insertion_boundary_blockage_count": (
                    episode_reason_counts["lower_insertion_boundary"]
                ),
                "vessel_tree_end_blockage_count": (
                    episode_tree_end_blockage_count
                ),
                "planner_call_count": planner_call_count,
                "selected_translation_risk_mean": (
                    episode_result["selected_translation_risk"]["mean"]
                    if episode_result["selected_translation_risk"]
                    else None
                ),
                "selected_safety_penalty_mean": (
                    episode_result["selected_safety_penalty"]["mean"]
                    if episode_result["selected_safety_penalty"]
                    else None
                ),
                "task_scale_mean": (
                    episode_result["task_scale"]["mean"]
                    if episode_result["task_scale"]
                    else None
                ),
            }
            if integral_episode_record is not None:
                episode_log_record.update(
                    {
                        "episode_tree_end_blockage_fraction": (
                            episode_tree_end_blockage_fraction
                        ),
                        "dual_enabled": True,
                        "dual_rolling_mean_cost": integral_episode_record[
                            "rolling_mean_cost"
                        ],
                        "dual_cost_limit": integral_episode_record[
                            "cost_limit"
                        ],
                        "dual_error": integral_episode_record["dual_error"],
                        "dual_alpha_before_update": episode_runtime_alpha,
                        "dual_alpha_after_update": integral_episode_record[
                            "alpha_after_update"
                        ],
                        "dual_alpha_update_amount": integral_episode_record[
                            "alpha_update_amount"
                        ],
                        "dual_lower_clipping_active": (
                            integral_episode_record[
                                "lower_clipping_active"
                            ]
                        ),
                        "dual_upper_clipping_active": (
                            integral_episode_record[
                                "upper_clipping_active"
                            ]
                        ),
                        "dual_warmup_active": integral_episode_record[
                            "warmup_active"
                        ],
                        "dual_update_applied": integral_episode_record[
                            "dual_update_applied"
                        ],
                    }
                )
            logger.log("episodes", episode_log_record)
            episode_message = (
                f"episode={episode + 1} reward={episode_reward:.3f} "
                f"length={episode_length} success={int(success)} "
                f"blockage={episode_blockage_count}"
            )
            if integral_episode_record is not None:
                episode_message += (
                    f" tree_end_cost={episode_tree_end_blockage_fraction:.6f} "
                    f"alpha={episode_runtime_alpha:.6f}->"
                    f"{integral_episode_record['alpha_after_update']:.6f}"
                )
            print(episode_message)
    finally:
        if video_writer is not None:
            video_writer.release()
        env.close()

    if video_path is not None:
        print(f"Saved evaluation video: {video_path}")

    if total_transition_count <= 0:
        raise RuntimeError("Evaluation produced no environment transitions")
    non_none_reason_total = sum(
        count for reason, count in reason_totals.items() if reason != "none"
    )
    if non_none_reason_total != total_translation_blockage_count:
        raise RuntimeError(
            "Aggregate blockage count does not match aggregate reason totals"
        )
    if reason_totals["none"] + non_none_reason_total != total_transition_count:
        raise RuntimeError(
            "Aggregate translation block reasons do not cover all transitions"
        )
    active_planner_call_count = len(selected_risks)
    if not (
        active_planner_call_count
        == len(selected_penalties)
        == len(task_scales)
        == expected_active_planner_call_count
    ):
        raise RuntimeError(
            "Aggregate planner metrics do not match the episodes in which "
            "Safety-MPC was active"
        )

    integral_summary = (
        integral_controller.summary()
        if integral_controller is not None
        else None
    )

    summary = {
        "checkpoint_step": int(checkpoint.get("total_env_steps", -1)),
        "episode_count": episodes,
        "success_count": int(sum(successes)),
        "success_rate": float(np.mean(successes)),
        "reward": finite_statistics(
            rewards,
            include_median=True,
            include_std=True,
        ),
        "episode_length": finite_statistics(
            lengths,
            include_median=True,
        ),
        "total_transition_count": total_transition_count,
        "translation_blockage_count": total_translation_blockage_count,
        "translation_blockage_fraction": (
            total_translation_blockage_count / total_transition_count
        ),
        "translation_blockage_episode_count": blockage_episode_count,
        "lower_insertion_boundary_blockage_count": (
            reason_totals["lower_insertion_boundary"]
        ),
        "device_length_limit_blockage_count": (
            reason_totals["device_length_limit"]
        ),
        "vessel_tree_end_blockage_count": reason_totals["vessel_tree_end"],
        "other_blockage_count": reason_totals["other"],
        "simulation_error_transition_count": (
            simulation_error_transition_count
        ),
        "simulation_error_episode_count": simulation_error_episode_count,
        "planner_call_count": active_planner_call_count,
        "selected_translation_risk": finite_statistics(
            selected_risks,
            include_max=True,
        ),
        "selected_safety_penalty": finite_statistics(
            selected_penalties,
            include_max=True,
        ),
        "task_scale": finite_statistics(
            task_scales,
            include_std=True,
        ),
    }
    if integral_summary is not None:
        summary["integral_lagrangian"] = integral_summary

    checkpoint_metadata_after = {
        "top_level": checkpoint.get("safety_mpc_config"),
        "embedded": checkpoint["config"].get("safety_mpc"),
        "agent": checkpoint["agent"].get("safety_mpc_config"),
    }
    checkpoint_file_hash_after = file_sha256(checkpoint_path)
    model_hash_after = state_sha256(agent.model.state_dict())
    model_optimizer_hash_after = state_sha256(
        agent.model_optimizer.state_dict()
    )
    policy_optimizer_hash_after = state_sha256(
        agent.policy_optimizer.state_dict()
    )
    scale_hash_after = state_sha256(agent.scale.state_dict())
    agent_config_hash_after = state_sha256(agent.config)
    parameter_gradient_count_after = sum(
        int(parameter.grad is not None)
        for parameter in agent.model.parameters()
    )
    isolation = {
        "deterministic_episode_reseeding": True,
        "eval_mode": True,
        "training_exploration_noise": False,
        "training_updates": False,
        "replay_constructed_or_updated": False,
        "checkpoint_file_sha256_before": checkpoint_file_hash_before,
        "checkpoint_file_sha256_after": checkpoint_file_hash_after,
        "checkpoint_file_unchanged": (
            checkpoint_file_hash_after == checkpoint_file_hash_before
        ),
        "checkpoint_model_state_sha256": checkpoint_model_hash,
        "loaded_model_matches_checkpoint": (
            model_hash_before == checkpoint_model_hash
        ),
        "model_state_sha256_before": model_hash_before,
        "model_state_sha256_after": model_hash_after,
        "model_parameters_unchanged": model_hash_after == model_hash_before,
        "model_optimizer_state_unchanged": (
            model_optimizer_hash_after == model_optimizer_hash_before
        ),
        "policy_optimizer_state_unchanged": (
            policy_optimizer_hash_after == policy_optimizer_hash_before
        ),
        "policy_scale_state_unchanged": scale_hash_after == scale_hash_before,
        "agent_checkpoint_config_unchanged": (
            agent_config_hash_after == agent_config_hash_before
        ),
        "checkpoint_safety_mpc_metadata_unchanged": (
            checkpoint_metadata_after == checkpoint_metadata_snapshot
        ),
        "update_count_before": update_count_before,
        "update_count_after": agent.update_count,
        "update_count_unchanged": agent.update_count == update_count_before,
        "parameter_gradient_count_before": parameter_gradient_count_before,
        "parameter_gradient_count_after": parameter_gradient_count_after,
        "parameter_gradients_absent": (
            parameter_gradient_count_before == 0
            and parameter_gradient_count_after == 0
        ),
        "runtime_override_fields": [
            (
                "safety_mpc_runtime_enabled"
                if integral_controller is not None
                else "safety_mpc_enabled"
            ),
            (
                "safety_mpc_runtime_alpha"
                if integral_controller is not None
                else "safety_mpc_alpha"
            ),
            "safety_mpc_active",
        ],
        "curvature_used_in_planning": False,
        "intervention_mask_modified": False,
    }
    if integral_controller is not None:
        isolation.update(
            {
                "integral_lagrangian_evaluation_only": True,
                "dual_feedback_uses_predicted_risk": False,
            }
        )
    required_isolation_checks = (
        "checkpoint_file_unchanged",
        "loaded_model_matches_checkpoint",
        "model_parameters_unchanged",
        "model_optimizer_state_unchanged",
        "policy_optimizer_state_unchanged",
        "policy_scale_state_unchanged",
        "agent_checkpoint_config_unchanged",
        "checkpoint_safety_mpc_metadata_unchanged",
        "update_count_unchanged",
        "parameter_gradients_absent",
    )
    failed_isolation_checks = [
        key for key in required_isolation_checks if not isolation[key]
    ]
    if failed_isolation_checks:
        raise RuntimeError(
            "Evaluation isolation checks failed: "
            f"{failed_isolation_checks}"
        )

    logger.log(
        "summary",
        {
            "checkpoint_step": summary["checkpoint_step"],
            "episode_count": episodes,
            "success_count": summary["success_count"],
            "success_rate": summary["success_rate"],
            "reward_mean": summary["reward"]["mean"],
            "reward_median": summary["reward"]["median"],
            "reward_std": summary["reward"]["std"],
            "episode_length_mean": summary["episode_length"]["mean"],
            "episode_length_median": summary["episode_length"]["median"],
            "translation_blockage_count": total_translation_blockage_count,
            "translation_blockage_fraction": (
                summary["translation_blockage_fraction"]
            ),
            "translation_blockage_episode_count": blockage_episode_count,
            "lower_insertion_boundary_blockage_count": (
                reason_totals["lower_insertion_boundary"]
            ),
            "vessel_tree_end_blockage_count": (
                reason_totals["vessel_tree_end"]
            ),
            "simulation_error_transition_count": (
                simulation_error_transition_count
            ),
            "simulation_error_episode_count": (
                simulation_error_episode_count
            ),
            "planner_call_count": active_planner_call_count,
            "selected_translation_risk_mean": (
                summary["selected_translation_risk"]["mean"]
                if summary["selected_translation_risk"]
                else None
            ),
            "selected_translation_risk_max": (
                summary["selected_translation_risk"]["max"]
                if summary["selected_translation_risk"]
                else None
            ),
            "selected_safety_penalty_mean": (
                summary["selected_safety_penalty"]["mean"]
                if summary["selected_safety_penalty"]
                else None
            ),
            "selected_safety_penalty_max": (
                summary["selected_safety_penalty"]["max"]
                if summary["selected_safety_penalty"]
                else None
            ),
            "task_scale_mean": (
                summary["task_scale"]["mean"]
                if summary["task_scale"]
                else None
            ),
            "task_scale_std": (
                summary["task_scale"]["std"]
                if summary["task_scale"]
                else None
            ),
            **(
                {
                    "dual_initial_alpha": integral_summary["initial_alpha"],
                    "dual_final_alpha": integral_summary["final_alpha"],
                    "dual_minimum_alpha": integral_summary["minimum_alpha"],
                    "dual_maximum_alpha": integral_summary["maximum_alpha"],
                    "dual_alpha_mean": integral_summary["alpha_mean"],
                    "dual_update_count": integral_summary[
                        "dual_update_count"
                    ],
                    "dual_lower_clip_count": integral_summary[
                        "lower_clip_count"
                    ],
                    "dual_upper_clip_count": integral_summary[
                        "upper_clip_count"
                    ],
                    "dual_alpha_trajectory": integral_summary[
                        "alpha_trajectory"
                    ],
                    "dual_episode_alpha_trajectory": integral_summary[
                        "episode_alpha_trajectory"
                    ],
                    "dual_alpha_statistics_basis": integral_summary[
                        "alpha_statistics_basis"
                    ],
                    "dual_episode_applied_minimum_alpha": integral_summary[
                        "episode_applied_minimum_alpha"
                    ],
                    "dual_episode_applied_maximum_alpha": integral_summary[
                        "episode_applied_maximum_alpha"
                    ],
                    "dual_episode_applied_alpha_mean": integral_summary[
                        "episode_applied_alpha_mean"
                    ],
                    "dual_observed_cost_trajectory": integral_summary[
                        "observed_cost_trajectory"
                    ],
                    "dual_rolling_cost_trajectory": integral_summary[
                        "rolling_cost_trajectory"
                    ],
                }
                if integral_summary is not None
                else {}
            ),
        },
    )

    report_settings = {
        **(
            settings_report
            if integral_controller is not None
            else safety_settings_report
        ),
        "device": str(device),
        "episode_count": episodes,
        "base_seed": base_seed,
        "seeds": [base_seed + index for index in range(episodes)],
        "max_episode_steps": int(
            config["environment"]["max_episode_steps"]
        ),
        "deterministic": True,
        "environment_config": copy.deepcopy(config["environment"]),
    }
    report_semantics = {
        "translation_safety_only": True,
        "curvature_monitoring_only": True,
        "selected_risk_is_weighted_mean_plan_risk": True,
        "blockage_source": "info.safety_metrics",
    }
    if integral_controller is not None:
        report_semantics.update(
            {
                "dual_feedback_signal": (
                    "vessel_tree_end_blockage_count / "
                    "max(episode_transition_count, 1)"
                ),
                "predicted_translation_risk_used_only_for_candidate_scoring": (
                    True
                ),
                "alpha_updated_only_after_completed_episode": True,
            }
        )

    report = {
        "schema_version": (
            EVALUATION_REPORT_SCHEMA_VERSION
            if integral_controller is not None
            else BASELINE_EVALUATION_REPORT_SCHEMA_VERSION
        ),
        "evaluation": (
            "integral_lagrangian_translation_safety_mpc"
            if integral_controller is not None
            else "same_checkpoint_translation_safety_mpc_ablation"
        ),
        "checkpoint": {
            "path": str(checkpoint_path),
            "total_env_steps": int(checkpoint.get("total_env_steps", -1)),
            "file_sha256": checkpoint_file_hash_before,
            "model_state_sha256": checkpoint_model_hash,
        },
        "settings": report_settings,
        "semantics": report_semantics,
        "episodes": episode_results,
        "summary": summary,
        "isolation": isolation,
    }
    if args.output_json is not None:
        output_json = args.output_json.expanduser().resolve()
        atomic_json_save(report, output_json)
        # Parse through the standard library to make the strict artifact part
        # of the evaluated path rather than relying only on serialization.
        with output_json.open("r", encoding="utf-8") as stream:
            parsed_report = json.load(stream)
        if parsed_report != report:
            raise RuntimeError(
                "Strict JSON report does not round-trip exactly"
            )
        print(f"Saved strict evaluation JSON: {output_json}")

    print(
        "Evaluation summary: "
        f"reward={summary['reward']['mean']:.3f} +/- "
        f"{summary['reward']['std']:.3f}, "
        f"length={summary['episode_length']['mean']:.1f}, "
        f"success_rate={summary['success_rate']:.3f}, "
        f"blockage={summary['translation_blockage_count']}/"
        f"{summary['total_transition_count']} "
        f"({summary['translation_blockage_fraction']:.4f})"
    )


if __name__ == "__main__":
    main()
