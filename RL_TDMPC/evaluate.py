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

from envs import make_env
from safety_schema import (
    LEGACY_STEVE_SAFETY_COST_NAMES,
    TRANSLATION_BLOCK_REASON_NAMES,
)
from train import (
    resolved_safety_mpc_config,
    validate_checkpoint_schema,
    validate_collected_safety_cost,
)
from tdmpc2.agent import TDMPC2Agent
from tdmpc2.adaptive_lagrangian import (
    DEFAULT_ADAPTIVE_LAGRANGIAN_CONFIG,
    AdaptiveLagrangianController,
    validate_adaptive_lagrangian_config,
)
from tdmpc2.common import (
    MetricLogger,
    atomic_json_save,
    build_diagnostics_agent_config,
    build_safety_agent_config,
    build_safety_mpc_agent_config,
    load_config,
    load_torch_checkpoint,
    select_device,
    set_seed,
)


BASELINE_EVALUATION_REPORT_SCHEMA_VERSION = 1
EVALUATION_REPORT_SCHEMA_VERSION = 2
_PLANNER_METRIC_NAMES = (
    "safety_mpc_selected_risk",
    "safety_mpc_selected_penalty",
    "safety_mpc_task_scale",
)
_LAGRANGIAN_PLANNER_METRIC_NAMES = (
    *_PLANNER_METRIC_NAMES,
    "safety_mpc_penalty_to_task_scale",
    "safety_mpc_safe_top_task_score",
    "safety_mpc_safe_top_planner_score",
    "safety_mpc_same_population_task_sacrifice",
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
        "--backend",
        choices=("steve", "nvidia-guided"),
        default=None,
        help=(
            "Evaluation backend consistency check; it must match the selected "
            "checkpoint/config because observation and Safety schemas are fixed"
        ),
    )
    parser.add_argument("--nvidia-workflow-root", type=Path, default=None)
    parser.add_argument("--nvidia-ct-cache", type=Path, default=None)
    parser.add_argument(
        "--eval-safety-mpc",
        choices=("checkpoint", "disabled", "enabled", "lagrangian"),
        default="checkpoint",
        help=(
            "Evaluation-only planner setting. The checkpoint setting is used "
            "by default; disabled/enabled/lagrangian are explicit runtime-only "
            "overrides."
        ),
    )
    parser.add_argument(
        "--eval-safety-mpc-alpha",
        type=float,
        default=None,
        help=(
            "Required positive alpha with --eval-safety-mpc enabled; rejected "
            "for checkpoint/disabled modes."
        ),
    )
    parser.add_argument("--eval-lagrangian-epsilon", type=float, default=None)
    parser.add_argument("--eval-lagrangian-eta", type=float, default=None)
    parser.add_argument(
        "--eval-lagrangian-lambda-initial", type=float, default=None
    )
    parser.add_argument(
        "--eval-lagrangian-lambda-min", type=float, default=None
    )
    parser.add_argument(
        "--eval-lagrangian-lambda-max", type=float, default=None
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
        help="Enable the selected backend's evaluation rendering path",
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
        **build_safety_agent_config(config),
        **build_safety_mpc_agent_config(config),
        **build_diagnostics_agent_config(config),
    }


def resolve_evaluation_safety_mpc_config(
    checkpoint_config: Mapping[str, Any],
    *,
    override: str,
    alpha: Optional[float],
    lagrangian_initial: Optional[float] = None,
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
    if override == "lagrangian":
        if alpha is not None:
            raise ValueError(
                "--eval-safety-mpc-alpha is invalid with lagrangian mode"
            )
        if lagrangian_initial is None:
            raise ValueError("Lagrangian mode requires a resolved lambda_initial")
        evaluation_settings = copy.deepcopy(checkpoint_settings)
        evaluation_settings["enabled"] = True
        evaluation_settings["alpha"] = float(lagrangian_initial)
        resolved_safety_mpc_config({"safety_mpc": evaluation_settings})
        return evaluation_settings
    if override != "enabled":
        raise ValueError(f"Unsupported evaluation Safety-MPC override {override!r}")
    if alpha is None:
        raise ValueError(
            "--eval-safety-mpc enabled requires --eval-safety-mpc-alpha"
        )
    if isinstance(alpha, bool) or not np.isfinite(float(alpha)) or float(alpha) <= 0:
        raise ValueError("Evaluation Safety-MPC alpha must be finite and positive")
    evaluation_settings = copy.deepcopy(checkpoint_settings)
    evaluation_settings["enabled"] = True
    evaluation_settings["alpha"] = float(alpha)
    # Reuse the production parser so evaluation cannot introduce a looser
    # configuration schema than training.
    resolved_safety_mpc_config({"safety_mpc": evaluation_settings})
    return evaluation_settings


def resolve_evaluation_adaptive_lagrangian_config(
    *,
    mode: str,
    epsilon: Optional[float],
    eta: Optional[float],
    lambda_initial: Optional[float],
    lambda_min: Optional[float],
    lambda_max: Optional[float],
) -> Optional[Dict[str, float]]:
    """Resolve strict CLI-only adaptive settings without touching config."""

    supplied = {
        "epsilon": epsilon,
        "eta": eta,
        "lambda_initial": lambda_initial,
        "lambda_min": lambda_min,
        "lambda_max": lambda_max,
    }
    supplied_names = [name for name, value in supplied.items() if value is not None]
    if mode != "lagrangian":
        if supplied_names:
            raise ValueError(
                "Adaptive Lagrangian parameters require "
                "--eval-safety-mpc lagrangian"
            )
        return None
    resolved: Dict[str, Any] = copy.deepcopy(
        DEFAULT_ADAPTIVE_LAGRANGIAN_CONFIG
    )
    for name, value in supplied.items():
        if value is not None:
            resolved[name] = value
    return validate_adaptive_lagrangian_config(resolved)


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
    include_min: bool = False,
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
    if include_min:
        statistics["min"] = float(np.min(array))
    if include_max:
        statistics["max"] = float(np.max(array))
    return statistics


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
    configured_backend = config.get("backend", "steve")
    if not isinstance(configured_backend, str):
        raise TypeError("Configuration backend must be a string")
    configured_backend = configured_backend.strip().lower()
    if args.backend is not None and args.backend != configured_backend:
        raise ValueError(
            f"--backend {args.backend!r} conflicts with configuration backend "
            f"{configured_backend!r}; evaluate with the matching checkpoint/config"
        )
    config["backend"] = configured_backend
    if args.nvidia_workflow_root is not None:
        config["environment"]["workflow_root"] = str(
            args.nvidia_workflow_root.expanduser().resolve()
        )
    if args.nvidia_ct_cache is not None:
        config["environment"]["ct_cache_path"] = str(
            args.nvidia_ct_cache.expanduser().resolve()
        )
    validate_checkpoint_schema(
        checkpoint,
        config=config,
        source=f"Evaluation checkpoint {checkpoint_path}",
    )
    checkpoint_safety_mpc = resolved_safety_mpc_config(config)
    checkpoint_metadata_snapshot = {
        "top_level": copy.deepcopy(checkpoint.get("safety_mpc_config")),
        "embedded": copy.deepcopy(checkpoint["config"].get("safety_mpc")),
        "agent": copy.deepcopy(
            checkpoint["agent"].get("safety_mpc_config")
        ),
    }
    evaluation_adaptive_lagrangian = (
        resolve_evaluation_adaptive_lagrangian_config(
            mode=args.eval_safety_mpc,
            epsilon=args.eval_lagrangian_epsilon,
            eta=args.eval_lagrangian_eta,
            lambda_initial=args.eval_lagrangian_lambda_initial,
            lambda_min=args.eval_lagrangian_lambda_min,
            lambda_max=args.eval_lagrangian_lambda_max,
        )
    )
    evaluation_safety_mpc = resolve_evaluation_safety_mpc_config(
        config,
        override=args.eval_safety_mpc,
        alpha=args.eval_safety_mpc_alpha,
        lagrangian_initial=(
            evaluation_adaptive_lagrangian["lambda_initial"]
            if evaluation_adaptive_lagrangian is not None
            else None
        ),
    )
    if evaluation_adaptive_lagrangian is not None:
        if not bool(config["planning"]["mpc"]):
            raise ValueError("Lagrangian Safety-MPC requires planning.mpc=true")
        if (
            evaluation_adaptive_lagrangian["epsilon"]
            > evaluation_safety_mpc["translation_risk_cap"]
        ):
            raise ValueError(
                "Lagrangian epsilon cannot exceed translation_risk_cap"
            )
    settings_report = {
        "checkpoint_safety_mpc": checkpoint_safety_mpc,
        "evaluation_safety_mpc": evaluation_safety_mpc,
        "override_applied": evaluation_safety_mpc != checkpoint_safety_mpc,
        "override_mode": args.eval_safety_mpc,
    }
    if evaluation_adaptive_lagrangian is not None:
        settings_report.update(
            {
                "evaluation_controller_mode": "lagrangian",
                "adaptive_lagrangian": evaluation_adaptive_lagrangian,
                "adaptive_lagrangian_override_applied": True,
            }
        )
    print(
        "Safety-MPC checkpoint/evaluation settings:\n"
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
    agent_config = build_agent_config(config)
    safety_cost_names = tuple(agent_config["safety_cost_names"])
    backend = config.get("backend", "steve")
    if not isinstance(backend, str):
        raise TypeError("Configuration backend must be a string")
    backend = backend.strip().lower()
    config["backend"] = backend

    record_video = args.video is not None
    render_enabled = args.render or record_video
    environment_config = dict(config["environment"])
    if render_enabled:
        environment_config["render_mode"] = (
            "rgb_array" if backend == "nvidia-guided" else "human"
        )
    env = make_env(environment_config, backend=backend)
    environment_safety_names = tuple(
        getattr(env, "safety_cost_names", ())
    )
    if environment_safety_names != safety_cost_names:
        env.close()
        raise ValueError(
            "Environment safety schema mismatch: expected "
            f"{safety_cost_names}, received {environment_safety_names}"
        )
    observation_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    agent = TDMPC2Agent(
        observation_dim,
        action_dim,
        agent_config,
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
    agent_safety_mpc_metadata_before = copy.deepcopy(
        agent.state_dict()["safety_mpc_config"]
    )
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
    adaptive_controller: Optional[AdaptiveLagrangianController] = None
    if evaluation_adaptive_lagrangian is not None:
        adaptive_controller = AdaptiveLagrangianController(
            evaluation_adaptive_lagrangian
        )
        agent.set_safety_mpc_runtime_alpha(
            adaptive_controller.current_lambda,
            enabled=True,
            force_active=True,
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
    lagrangian_applied_lambdas: List[float] = []
    lagrangian_risks: List[float] = []
    lagrangian_residuals: List[float] = []
    lagrangian_safe_top_task_scores: List[float] = []
    lagrangian_safe_top_planner_scores: List[float] = []
    lagrangian_same_population_task_sacrifices: List[float] = []
    lagrangian_action_norms: List[float] = []
    lagrangian_episode_summaries: List[Dict[str, Any]] = []
    total_translation_blockage_count = 0
    total_transition_count = 0
    blockage_episode_count = 0
    reason_totals = {
        reason: 0 for reason in TRANSLATION_BLOCK_REASON_NAMES
    }
    simulation_error_transition_count = 0
    simulation_error_episode_count = 0
    hard_safety_violation_transition_count = 0
    hard_safety_violation_episode_count = 0
    legacy_blockage_metrics = (
        safety_cost_names == LEGACY_STEVE_SAFETY_COST_NAMES
    )
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
            if adaptive_controller is not None:
                adaptive_controller.reset_episode()
                agent.set_safety_mpc_runtime_alpha(
                    adaptive_controller.current_lambda,
                    enabled=True,
                    force_active=True,
                )
            # The world-model planner is sampling-based. Reseeding makes the
            # no-exploration eval trajectory reproducible across program runs.
            set_seed(episode_seed)
            env.action_space.seed(episode_seed)
            observation, reset_info = env.reset(seed=episode_seed)
            reset_names = reset_info.get("safety_cost_names")
            if reset_names is not None and tuple(reset_names) != safety_cost_names:
                raise ValueError(
                    "Reset safety schema mismatch: expected "
                    f"{safety_cost_names}, received {tuple(reset_names)}"
                )
            validate_collected_safety_cost(
                reset_info["safety_cost"],
                safety_cost_names=safety_cost_names,
                received_safety_cost_names=reset_names,
                source="Evaluation reset info['safety_cost']",
            )
            render_frame()
            terminated = truncated = False
            episode_reward = 0.0
            episode_length = 0
            success = False
            simulation_error = False
            episode_simulation_error_transitions = 0
            episode_hard_safety_violation_transitions = 0
            episode_blockage_count = 0
            episode_reason_counts = {
                reason: 0 for reason in TRANSLATION_BLOCK_REASON_NAMES
            }
            episode_selected_risks: List[float] = []
            episode_selected_penalties: List[float] = []
            episode_task_scales: List[float] = []
            episode_lagrangian_steps: List[Dict[str, Any]] = []
            while not (terminated or truncated):
                lambda_used_for_action = (
                    adaptive_controller.current_lambda
                    if adaptive_controller is not None
                    else None
                )
                if (
                    adaptive_controller is not None
                    and agent.safety_mpc_runtime_alpha
                    != lambda_used_for_action
                ):
                    raise RuntimeError(
                        "Agent runtime lambda is not aligned with the controller"
                    )
                action = agent.act(
                    observation,
                    first_step=episode_length == 0,
                    eval_mode=True,
                )
                planner_metrics = agent.last_safety_mpc_metrics
                if agent.safety_mpc_active:
                    if not isinstance(planner_metrics, Mapping):
                        raise RuntimeError(
                            "Active Safety-MPC did not expose planner metrics"
                        )
                    planner_values: Dict[str, float] = {}
                    required_metrics = (
                        _LAGRANGIAN_PLANNER_METRIC_NAMES
                        if adaptive_controller is not None
                        else _PLANNER_METRIC_NAMES
                    )
                    for name in required_metrics:
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
                episode_hard_safety_violation_transitions += int(
                    bool(info.get("hard_safety_violation", False))
                )

                received_names = info.get("safety_cost_names")
                if (
                    received_names is not None
                    and tuple(received_names) != safety_cost_names
                ):
                    raise ValueError(
                        "Step safety schema mismatch: expected "
                        f"{safety_cost_names}, received {tuple(received_names)}"
                    )
                validate_collected_safety_cost(
                    info["safety_cost"],
                    safety_cost_names=safety_cost_names,
                    received_safety_cost_names=received_names,
                    source="Evaluation environment info['safety_cost']",
                )
                translation_blocked = False
                block_reason = "unavailable"
                safety_metrics = info.get("safety_metrics")
                if safety_cost_names == LEGACY_STEVE_SAFETY_COST_NAMES:
                    if not isinstance(safety_metrics, Mapping):
                        raise TypeError(
                            "Legacy stEVE info['safety_metrics'] must be a mapping"
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
                if adaptive_controller is not None:
                    assert lambda_used_for_action is not None
                    predicted_risk = planner_values[
                        "safety_mpc_selected_risk"
                    ]
                    expected_penalty_ratio = lambda_used_for_action * predicted_risk
                    if not np.isclose(
                        planner_values["safety_mpc_penalty_to_task_scale"],
                        expected_penalty_ratio,
                        rtol=1.0e-5,
                        atol=1.0e-8,
                    ):
                        raise RuntimeError(
                            "Planner penalty is not aligned with lambda_t and risk_t"
                        )
                    if isinstance(safety_metrics, Mapping):
                        inserted_length = float(
                            safety_metrics.get(
                                "inserted_length_mm",
                                info.get("guide_s_m", 0.0),
                            )
                        )
                        requested_translation = float(
                            safety_metrics.get(
                                "requested_translation_speed_mm_s",
                                info.get("velocity_cmd_m_s", 0.0),
                            )
                        )
                        applied_translation = float(
                            safety_metrics.get(
                                "applied_translation_speed_mm_s",
                                info.get("velocity_cmd_m_s", 0.0),
                            )
                        )
                    else:
                        inserted_length = float(info.get("guide_s_m", 0.0))
                        requested_translation = float(
                            info.get("velocity_cmd_m_s", 0.0)
                        )
                        applied_translation = requested_translation
                    if not all(
                        np.isfinite(value)
                        for value in (
                            inserted_length,
                            requested_translation,
                            applied_translation,
                        )
                    ):
                        raise FloatingPointError(
                            "Adaptive Lagrangian environment diagnostics are non-finite"
                        )
                    action_values = np.asarray(action, dtype=np.float64)
                    if (
                        action_values.shape != (agent.action_dim,)
                        or not np.all(np.isfinite(action_values))
                    ):
                        raise FloatingPointError(
                            "Adaptive Lagrangian selected action is invalid"
                        )
                    # Commit the controller state only after the transition and
                    # all diagnostics for (lambda_t, action_t, risk_t) validate.
                    dual_record = adaptive_controller.observe_step_risk(
                        predicted_risk
                    )
                    if dual_record["lambda_before_update"] != lambda_used_for_action:
                        raise RuntimeError(
                            "Adaptive lambda changed before the environment step"
                        )
                    agent.set_safety_mpc_runtime_alpha(
                        adaptive_controller.current_lambda,
                        enabled=True,
                        force_active=True,
                    )
                    episode_lagrangian_steps.append(
                        {
                            "timestep": episode_length,
                            **dual_record,
                            "safe_top_candidate_task_score": planner_values[
                                "safety_mpc_safe_top_task_score"
                            ],
                            "safe_top_candidate_planner_score": planner_values[
                                "safety_mpc_safe_top_planner_score"
                            ],
                            "same_population_task_sacrifice": planner_values[
                                "safety_mpc_same_population_task_sacrifice"
                            ],
                            "task_scale": planner_values[
                                "safety_mpc_task_scale"
                            ],
                            "safety_penalty": planner_values[
                                "safety_mpc_selected_penalty"
                            ],
                            "penalty_to_task_scale_ratio": planner_values[
                                "safety_mpc_penalty_to_task_scale"
                            ],
                            "selected_action": action_values.tolist(),
                            "selected_action_l2_norm": float(
                                np.linalg.norm(action_values)
                            ),
                            "inserted_length_mm": inserted_length,
                            "requested_translation_speed_mm_s": (
                                requested_translation
                            ),
                            "applied_translation_speed_mm_s": applied_translation,
                            "translation_action_blocked": translation_blocked,
                            "translation_block_reason": block_reason,
                        }
                    )
                    lagrangian_safe_top_task_scores.append(
                        planner_values["safety_mpc_safe_top_task_score"]
                    )
                    lagrangian_safe_top_planner_scores.append(
                        planner_values["safety_mpc_safe_top_planner_score"]
                    )
                    lagrangian_same_population_task_sacrifices.append(
                        planner_values[
                            "safety_mpc_same_population_task_sacrifice"
                        ]
                    )
                    lagrangian_action_norms.append(
                        float(np.linalg.norm(action_values))
                    )

            non_none_reason_count = sum(
                count
                for reason, count in episode_reason_counts.items()
                if reason != "none"
            )
            if (
                legacy_blockage_metrics
                and non_none_reason_count != episode_blockage_count
            ):
                raise RuntimeError(
                    "Translation blockage count does not match reason totals"
                )
            planner_call_count = len(episode_selected_risks)
            if agent.safety_mpc_active:
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

            episode_lagrangian_summary: Optional[Dict[str, Any]] = None
            if adaptive_controller is not None:
                episode_lagrangian_summary = adaptive_controller.summary()
                expected_step_count = episode_length
                trajectory_names = (
                    "applied_lambda_trajectory",
                    "predicted_risk_trajectory",
                    "constraint_residual_trajectory",
                    "lambda_update_trajectory",
                )
                if (
                    episode_lagrangian_summary["step_update_count"]
                    != expected_step_count
                    or len(episode_lagrangian_summary["lambda_state_trajectory"])
                    != expected_step_count + 1
                    or any(
                        len(episode_lagrangian_summary[name])
                        != expected_step_count
                        for name in trajectory_names
                    )
                    or len(episode_lagrangian_steps) != expected_step_count
                ):
                    raise RuntimeError(
                        "Adaptive Lagrangian trajectories do not align with "
                        "environment transitions"
                    )
                lagrangian_applied_lambdas.extend(
                    episode_lagrangian_summary["applied_lambda_trajectory"]
                )
                lagrangian_risks.extend(
                    episode_lagrangian_summary["predicted_risk_trajectory"]
                )
                lagrangian_residuals.extend(
                    episode_lagrangian_summary[
                        "constraint_residual_trajectory"
                    ]
                )
                lagrangian_episode_summaries.append(
                    {
                        "episode_index": episode + 1,
                        "seed": episode_seed,
                        **episode_lagrangian_summary,
                    }
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
            hard_safety_violation_transition_count += (
                episode_hard_safety_violation_transitions
            )
            hard_safety_violation_episode_count += int(
                episode_hard_safety_violation_transitions > 0
            )
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
                    episode_reason_counts["vessel_tree_end"]
                ),
                "other_blockage_count": episode_reason_counts["other"],
                "simulation_error_transition_count": (
                    episode_simulation_error_transitions
                ),
                "simulation_error": simulation_error,
                "hard_safety_violation_transition_count": (
                    episode_hard_safety_violation_transitions
                ),
                "hard_safety_violation_episode": (
                    episode_hard_safety_violation_transitions > 0
                ),
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
            if episode_lagrangian_summary is not None:
                episode_result["adaptive_lagrangian"] = {
                    "summary": episode_lagrangian_summary,
                    "steps": episode_lagrangian_steps,
                }
            episode_results.append(episode_result)
            adaptive_episode_log: Dict[str, Any] = {}
            if episode_lagrangian_summary is not None:
                adaptive_episode_log = {
                    "lagrangian_lambda_initial": (
                        episode_lagrangian_summary["lambda_initial"]
                    ),
                    "lagrangian_lambda_final": (
                        episode_lagrangian_summary["lambda_final"]
                    ),
                    "lagrangian_lambda_mean": (
                        episode_lagrangian_summary["applied_lambda_mean"]
                    ),
                    "lagrangian_lambda_min": (
                        episode_lagrangian_summary["applied_lambda_min"]
                    ),
                    "lagrangian_lambda_max": (
                        episode_lagrangian_summary["applied_lambda_max"]
                    ),
                    "lagrangian_predicted_risk_mean": (
                        episode_lagrangian_summary["predicted_risk_mean"]
                    ),
                    "lagrangian_constraint_residual_mean": (
                        episode_lagrangian_summary[
                            "constraint_residual_mean"
                        ]
                    ),
                    "lagrangian_fraction_steps_near_lambda_min": (
                        episode_lagrangian_summary[
                            "fraction_steps_near_lambda_min"
                        ]
                    ),
                    "lagrangian_fraction_steps_near_lambda_max": (
                        episode_lagrangian_summary[
                            "fraction_steps_near_lambda_max"
                        ]
                    ),
                }
            logger.log(
                "episodes",
                {
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
                        episode_reason_counts["vessel_tree_end"]
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
                    **adaptive_episode_log,
                },
            )
            print(
                f"episode={episode + 1} reward={episode_reward:.3f} "
                f"length={episode_length} success={int(success)} "
                f"blockage={episode_blockage_count}"
            )
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
    if (
        legacy_blockage_metrics
        and non_none_reason_total != total_translation_blockage_count
    ):
        raise RuntimeError(
            "Aggregate blockage count does not match aggregate reason totals"
        )
    if (
        legacy_blockage_metrics
        and reason_totals["none"] + non_none_reason_total
        != total_transition_count
    ):
        raise RuntimeError(
            "Aggregate translation block reasons do not cover all transitions"
        )
    active_planner_call_count = len(selected_risks)
    if agent.safety_mpc_active:
        if not (
            active_planner_call_count
            == len(selected_penalties)
            == len(task_scales)
            == total_transition_count
        ):
            raise RuntimeError(
                "Aggregate active planner metrics do not cover all transitions"
            )
    elif active_planner_call_count != 0:
        raise RuntimeError("Inactive Safety-MPC produced aggregate planner metrics")

    adaptive_lagrangian_summary: Optional[Dict[str, Any]] = None
    if adaptive_controller is not None:
        if not (
            len(lagrangian_applied_lambdas)
            == len(lagrangian_risks)
            == len(lagrangian_residuals)
            == len(lagrangian_safe_top_task_scores)
            == len(lagrangian_safe_top_planner_scores)
            == len(lagrangian_same_population_task_sacrifices)
            == len(lagrangian_action_norms)
            == total_transition_count
        ):
            raise RuntimeError(
                "Aggregate adaptive trajectories do not cover all transitions"
            )
        assert evaluation_adaptive_lagrangian is not None
        lambda_minimum = evaluation_adaptive_lagrangian["lambda_min"]
        lambda_maximum = evaluation_adaptive_lagrangian["lambda_max"]
        lambda_tolerance = max(
            1.0e-9,
            1.0e-6 * (lambda_maximum - lambda_minimum),
        )
        near_lambda_min_count = sum(
            value <= lambda_minimum + lambda_tolerance
            for value in lagrangian_applied_lambdas
        )
        near_lambda_max_count = sum(
            value >= lambda_maximum - lambda_tolerance
            for value in lagrangian_applied_lambdas
        )
        risk_budget = evaluation_adaptive_lagrangian["epsilon"]
        risk_violation_count = sum(
            risk > risk_budget for risk in lagrangian_risks
        )
        episode_final_lambdas = [
            item["lambda_final"] for item in lagrangian_episode_summaries
        ]
        episode_applied_lambda_near_min_count = sum(
            item["fraction_steps_near_lambda_min"] > 0.0
            for item in lagrangian_episode_summaries
        )
        episode_applied_lambda_near_max_count = sum(
            item["fraction_steps_near_lambda_max"] > 0.0
            for item in lagrangian_episode_summaries
        )
        adaptive_lagrangian_summary = {
            "config": copy.deepcopy(evaluation_adaptive_lagrangian),
            "episode_reset_enabled": True,
            "update_timing": (
                "once after env.step from the selected weighted-mean plan's "
                "predicted capped horizon-max Translation risk; updated lambda "
                "is first used by the next environment decision"
            ),
            "episode_count": episodes,
            "total_step_update_count": total_transition_count,
            "applied_lambda": finite_statistics(
                lagrangian_applied_lambdas,
                include_std=True,
                include_min=True,
                include_max=True,
            ),
            "predicted_translation_risk": finite_statistics(
                lagrangian_risks,
                include_std=True,
                include_min=True,
                include_max=True,
            ),
            "constraint_residual": finite_statistics(
                lagrangian_residuals,
                include_std=True,
                include_min=True,
                include_max=True,
            ),
            "safe_top_candidate_task_score": finite_statistics(
                lagrangian_safe_top_task_scores,
                include_std=True,
                include_min=True,
                include_max=True,
            ),
            "safe_top_candidate_planner_score": finite_statistics(
                lagrangian_safe_top_planner_scores,
                include_std=True,
                include_min=True,
                include_max=True,
            ),
            "same_population_task_sacrifice": finite_statistics(
                lagrangian_same_population_task_sacrifices,
                include_std=True,
                include_min=True,
                include_max=True,
            ),
            "selected_action_l2_norm": finite_statistics(
                lagrangian_action_norms,
                include_std=True,
                include_min=True,
                include_max=True,
            ),
            "episode_final_lambda": finite_statistics(
                episode_final_lambdas,
                include_std=True,
                include_min=True,
                include_max=True,
            ),
            "risk_above_epsilon_count": risk_violation_count,
            "risk_above_epsilon_fraction": (
                risk_violation_count / total_transition_count
            ),
            "lambda_bound_tolerance": lambda_tolerance,
            "steps_near_lambda_min_count": near_lambda_min_count,
            "steps_near_lambda_max_count": near_lambda_max_count,
            "fraction_steps_near_lambda_min": (
                near_lambda_min_count / total_transition_count
            ),
            "fraction_steps_near_lambda_max": (
                near_lambda_max_count / total_transition_count
            ),
            "episode_applied_lambda_near_min_count": (
                episode_applied_lambda_near_min_count
            ),
            "episode_applied_lambda_near_max_count": (
                episode_applied_lambda_near_max_count
            ),
            "lower_clip_count": int(
                sum(
                    item["lower_clip_count"]
                    for item in lagrangian_episode_summaries
                )
            ),
            "upper_clip_count": int(
                sum(
                    item["upper_clip_count"]
                    for item in lagrangian_episode_summaries
                )
            ),
            "episodes": lagrangian_episode_summaries,
        }

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
        "hard_safety_violation_transition_count": (
            hard_safety_violation_transition_count
        ),
        "hard_safety_violation_episode_count": (
            hard_safety_violation_episode_count
        ),
        "translation_blockage_metrics_available": legacy_blockage_metrics,
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
    if adaptive_lagrangian_summary is not None:
        summary["adaptive_lagrangian"] = adaptive_lagrangian_summary

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
    agent_safety_mpc_metadata_after = copy.deepcopy(
        agent.state_dict()["safety_mpc_config"]
    )
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
        "agent_safety_mpc_metadata_unchanged": (
            agent_safety_mpc_metadata_after
            == agent_safety_mpc_metadata_before
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
        "runtime_override_fields": (
            [
                "safety_mpc_enabled",
                "safety_mpc_alpha",
                "safety_mpc_active",
            ]
            if adaptive_controller is None
            else [
                "_safety_mpc_runtime_enabled",
                "_safety_mpc_runtime_alpha",
                "_safety_mpc_runtime_force_active",
                "safety_mpc_active",
            ]
        ),
        "adaptive_controller_serialized_to_checkpoint": False,
        "adaptive_controller_uses_model_gradients": False,
        "curvature_used_in_planning": False,
        "intervention_mask_modified": False,
    }
    required_isolation_checks = (
        "checkpoint_file_unchanged",
        "loaded_model_matches_checkpoint",
        "model_parameters_unchanged",
        "model_optimizer_state_unchanged",
        "policy_optimizer_state_unchanged",
        "policy_scale_state_unchanged",
        "agent_checkpoint_config_unchanged",
        "agent_safety_mpc_metadata_unchanged",
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

    adaptive_summary_log: Dict[str, Any] = {}
    if adaptive_lagrangian_summary is not None:
        adaptive_summary_log = {
            "lagrangian_lambda_mean": adaptive_lagrangian_summary[
                "applied_lambda"
            ]["mean"],
            "lagrangian_lambda_std": adaptive_lagrangian_summary[
                "applied_lambda"
            ]["std"],
            "lagrangian_lambda_min": adaptive_lagrangian_summary[
                "applied_lambda"
            ]["min"],
            "lagrangian_lambda_max": adaptive_lagrangian_summary[
                "applied_lambda"
            ]["max"],
            "lagrangian_predicted_risk_mean": adaptive_lagrangian_summary[
                "predicted_translation_risk"
            ]["mean"],
            "lagrangian_constraint_residual_mean": (
                adaptive_lagrangian_summary["constraint_residual"]["mean"]
            ),
            "lagrangian_risk_above_epsilon_fraction": (
                adaptive_lagrangian_summary["risk_above_epsilon_fraction"]
            ),
            "lagrangian_same_population_task_sacrifice_mean": (
                adaptive_lagrangian_summary[
                    "same_population_task_sacrifice"
                ]["mean"]
            ),
            "lagrangian_fraction_steps_near_lambda_min": (
                adaptive_lagrangian_summary[
                    "fraction_steps_near_lambda_min"
                ]
            ),
            "lagrangian_fraction_steps_near_lambda_max": (
                adaptive_lagrangian_summary[
                    "fraction_steps_near_lambda_max"
                ]
            ),
            "lagrangian_lower_clip_count": adaptive_lagrangian_summary[
                "lower_clip_count"
            ],
            "lagrangian_upper_clip_count": adaptive_lagrangian_summary[
                "upper_clip_count"
            ],
        }
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
            **adaptive_summary_log,
        },
    )

    adaptive_mode = adaptive_lagrangian_summary is not None
    report = {
        "schema_version": (
            EVALUATION_REPORT_SCHEMA_VERSION
            if adaptive_mode
            else BASELINE_EVALUATION_REPORT_SCHEMA_VERSION
        ),
        "evaluation": (
            "adaptive_lagrangian_translation_safety_mpc"
            if adaptive_mode
            else "same_checkpoint_translation_safety_mpc_ablation"
        ),
        "checkpoint": {
            "path": str(checkpoint_path),
            "total_env_steps": int(checkpoint.get("total_env_steps", -1)),
            "file_sha256": checkpoint_file_hash_before,
            "model_state_sha256": checkpoint_model_hash,
        },
        "settings": {
            **settings_report,
            "device": str(device),
            "episode_count": episodes,
            "base_seed": base_seed,
            "seeds": [base_seed + index for index in range(episodes)],
            "max_episode_steps": int(
                config["environment"]["max_episode_steps"]
            ),
            "deterministic": True,
            "environment_config": copy.deepcopy(config["environment"]),
            "backend": backend,
            "safety_cost_names": list(safety_cost_names),
            "safety_primary_risk_channel": agent_config[
                "safety_primary_risk_channel"
            ],
        },
        "semantics": {
            "primary_safety_channel_only": True,
            "primary_safety_channel": agent_config[
                "safety_primary_risk_channel"
            ],
            "non_primary_safety_channels_monitoring_only": True,
            "translation_safety_only": (
                safety_cost_names == LEGACY_STEVE_SAFETY_COST_NAMES
            ),
            "curvature_monitoring_only": True,
            "selected_risk_is_weighted_mean_plan_risk": True,
            "blockage_source": (
                "info.safety_metrics"
                if legacy_blockage_metrics
                else None
            ),
        },
        "episodes": episode_results,
        "summary": summary,
        "isolation": isolation,
    }
    if adaptive_mode:
        report["semantics"].update(
            {
                "dual_feedback_source": (
                    "selected predicted Translation risk, not blockage or success"
                ),
                "selected_risk_aggregation": "capped_horizon_max",
                "lambda_resets_each_episode": True,
                "lambda_updates_after_environment_step": True,
                "lambda_update_affects_next_decision": True,
                "planner_score": (
                    "task_score - lambda_t * task_scale * trajectory_risk"
                ),
                "epsilon_is_candidate_constant_in_lagrangian_objective": True,
            }
        )
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
