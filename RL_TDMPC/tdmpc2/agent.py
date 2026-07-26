"""Minimal single-task TD-MPC2 training and latent-MPPI planning agent.

Algorithmically this follows the official implementation at
https://github.com/nicklashansen/tdmpc2 (commit e9f5932). Environment suites,
multi-task/offline learning, pixels, Hydra, TorchRL, W&B, and compilation were
removed because they are unrelated to the stEVE state-control task.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .networks import WorldModel, soft_cross_entropy, two_hot_inv
from .replay_buffer import REPLAY_SAFETY_COST_NAMES
from .safety_diagnostics import safety_batch_diagnostics


SAFETY_MODEL_SCHEMA_VERSION = 1
_SAFETY_CONFIG_KEYS = (
    "safety_loss_coef",
    "safety_curvature_loss_coef",
    "safety_translation_error_loss_coef",
    "safety_curvature_scale_mm_inv",
    "safety_translation_error_scale",
)
_DIAGNOSTIC_CONFIG_KEYS = (
    "validation_interval",
    "gradient_interval",
    "curvature_low_max_mm_inv",
    "curvature_medium_max_mm_inv",
)


class RunningScale(torch.nn.Module):
    """Running 5th-to-95th-percentile value scale used by TD-MPC2's actor."""

    def __init__(self, tau: float, device: torch.device) -> None:
        super().__init__()
        self.tau = float(tau)
        self.register_buffer("value", torch.ones(1, device=device))

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        flat = values.detach().float().reshape(-1)
        if flat.numel() < 2:
            estimate = torch.ones_like(self.value)
        else:
            percentiles = torch.quantile(
                flat, torch.tensor([0.05, 0.95], device=flat.device)
            )
            estimate = (percentiles[1] - percentiles[0]).clamp(min=1.0).reshape(1)
        self.value.lerp_(estimate, self.tau)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values / self.value


class TDMPC2Agent:
    """TD-MPC2 agent for one low-dimensional continuous-control task."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        config: Mapping[str, Any],
        *,
        episode_length: int,
        device: torch.device,
    ) -> None:
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.config = dict(config)
        self.device = device
        required_safety_keys = {
            "safety_cost_names",
            "safety_dim",
            *_SAFETY_CONFIG_KEYS,
            *_DIAGNOSTIC_CONFIG_KEYS,
        }
        missing_safety_keys = sorted(required_safety_keys - self.config.keys())
        if missing_safety_keys:
            raise ValueError(
                "Agent config is missing required Safety-Aware keys: "
                f"{missing_safety_keys}"
            )
        self.safety_cost_names = tuple(self.config["safety_cost_names"])
        if self.safety_cost_names != REPLAY_SAFETY_COST_NAMES:
            raise ValueError(
                "Agent safety_cost_names "
                f"{self.safety_cost_names} do not match required schema "
                f"{REPLAY_SAFETY_COST_NAMES} in this exact order"
            )
        self.safety_dim = int(self.config["safety_dim"])
        if self.safety_dim != len(REPLAY_SAFETY_COST_NAMES):
            raise ValueError(
                f"Agent safety_dim must be {len(REPLAY_SAFETY_COST_NAMES)}, "
                f"got {self.safety_dim}"
            )
        self.safety_loss_coef = float(self.config["safety_loss_coef"])
        self.safety_curvature_loss_coef = float(
            self.config["safety_curvature_loss_coef"]
        )
        self.safety_translation_error_loss_coef = float(
            self.config["safety_translation_error_loss_coef"]
        )
        self.safety_curvature_scale_mm_inv = float(
            self.config["safety_curvature_scale_mm_inv"]
        )
        self.safety_translation_error_scale = float(
            self.config["safety_translation_error_scale"]
        )
        coefficient_values = {
            "safety_loss_coef": self.safety_loss_coef,
            "safety_curvature_loss_coef": self.safety_curvature_loss_coef,
            "safety_translation_error_loss_coef": (
                self.safety_translation_error_loss_coef
            ),
        }
        for name, value in coefficient_values.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        scale_values = {
            "safety_curvature_scale_mm_inv": (
                self.safety_curvature_scale_mm_inv
            ),
            "safety_translation_error_scale": self.safety_translation_error_scale,
        }
        for name, value in scale_values.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and strictly positive")
        self.validation_interval = int(self.config["validation_interval"])
        self.gradient_interval = int(self.config["gradient_interval"])
        if self.validation_interval <= 0 or self.gradient_interval <= 0:
            raise ValueError(
                "validation_interval and gradient_interval must be positive"
            )
        self.curvature_boundaries = (
            float(self.config["curvature_low_max_mm_inv"]),
            float(self.config["curvature_medium_max_mm_inv"]),
        )
        if (
            not all(
                np.isfinite(value) and value >= 0.0
                for value in self.curvature_boundaries
            )
            or self.curvature_boundaries[0] >= self.curvature_boundaries[1]
        ):
            raise ValueError(
                "Curvature diagnostic boundaries must be finite, nonnegative, "
                "and strictly increasing"
            )
        self.safety_training_enabled = self.safety_loss_coef > 0.0
        self.model = WorldModel(
            self.observation_dim, self.action_dim, self.config
        ).to(device)
        self.model.train(False)

        learning_rate = float(self.config["lr"])
        encoder_scale = float(self.config["enc_lr_scale"])
        model_groups = [
            {
                "params": self.model.encoder.parameters(),
                "lr": learning_rate * encoder_scale,
            },
            {"params": self.model.dynamics.parameters()},
            {"params": self.model.reward_head.parameters()},
            {"params": self.model.termination_head.parameters()},
            {"params": self.model.q_ensemble.parameters()},
        ]
        safety_head_parameters = list(self.model.safety_head_parameters())
        if not safety_head_parameters:
            raise RuntimeError("WorldModel returned no Safety head parameters")
        if len({id(parameter) for parameter in safety_head_parameters}) != len(
            safety_head_parameters
        ):
            raise RuntimeError("WorldModel Safety head parameters contain duplicates")
        policy_parameter_ids = {
            id(parameter) for parameter in self.model.policy.parameters()
        }
        if any(
            id(parameter) in policy_parameter_ids
            for parameter in safety_head_parameters
        ):
            raise RuntimeError(
                "WorldModel Safety head parameters must not overlap policy parameters"
            )
        model_groups.append(
            {
                "params": safety_head_parameters,
                "name": "safety",
            }
        )
        self.model_optimizer = torch.optim.Adam(model_groups, lr=learning_rate)
        self.policy_optimizer = torch.optim.Adam(
            self.model.policy.parameters(), lr=learning_rate, eps=1e-5
        )
        self.scale = RunningScale(float(self.config["tau"]), device)

        fraction = float(episode_length) / float(self.config["discount_denom"])
        discount = (fraction - 1.0) / fraction
        self.discount = float(
            np.clip(
                discount,
                float(self.config["discount_min"]),
                float(self.config["discount_max"]),
            )
        )
        self.horizon = int(self.config["horizon"])
        self.previous_mean = torch.zeros(
            self.horizon, self.action_dim, device=self.device
        )
        self.update_count = 0

    def _tensor_observation(self, observation: np.ndarray) -> torch.Tensor:
        tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        )
        if tensor.shape != (self.observation_dim,):
            raise ValueError(
                f"Expected observation ({self.observation_dim},), got {tuple(tensor.shape)}"
            )
        return tensor.unsqueeze(0)

    @torch.no_grad()
    def act(
        self,
        observation: np.ndarray,
        *,
        first_step: bool = False,
        eval_mode: bool = False,
    ) -> np.ndarray:
        observation_tensor = self._tensor_observation(observation)
        if bool(self.config.get("mpc", True)):
            action = self._plan(
                observation_tensor, first_step=first_step, eval_mode=eval_mode
            )
        else:
            latent = self.model.encode(observation_tensor)
            action, policy_info = self.model.pi(latent, deterministic=eval_mode)
            action = policy_info["mean"] if eval_mode else action
            action = action[0]
        return action.clamp(-1.0, 1.0).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def _estimate_value(
        self, latent: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        returns = torch.zeros(
            latent.shape[0], 1, device=self.device, dtype=latent.dtype
        )
        discount = 1.0
        termination = torch.zeros_like(returns)
        for step in range(self.horizon):
            reward_logits = self.model.reward(latent, actions[step])
            reward = two_hot_inv(
                reward_logits,
                self.model.num_bins,
                self.model.value_min,
                self.model.value_max,
            )
            latent = self.model.next(latent, actions[step])
            returns.add_(discount * (1.0 - termination) * reward)
            discount *= self.discount
            if bool(self.config.get("episodic", True)):
                termination = torch.clamp(
                    termination + (self.model.termination(latent) > 0.5).float(),
                    max=1.0,
                )
        final_action, _ = self.model.pi(latent)
        final_q = self.model.q(latent, final_action, reduction="avg")
        return returns + discount * (1.0 - termination) * final_q

    @torch.no_grad()
    def _plan(
        self,
        observation: torch.Tensor,
        *,
        first_step: bool,
        eval_mode: bool,
    ) -> torch.Tensor:
        latent_single = self.model.encode(observation)
        num_samples = int(self.config["num_samples"])
        num_policy = min(int(self.config["num_pi_trajs"]), num_samples)
        num_elites = int(self.config["num_elites"])
        if not 0 < num_elites <= num_samples:
            raise ValueError("num_elites must be in [1, num_samples]")

        policy_actions: Optional[torch.Tensor] = None
        if num_policy > 0:
            policy_actions = torch.empty(
                self.horizon, num_policy, self.action_dim, device=self.device
            )
            policy_latent = latent_single.repeat(num_policy, 1)
            for step in range(self.horizon):
                policy_actions[step], _ = self.model.pi(policy_latent)
                policy_latent = self.model.next(
                    policy_latent, policy_actions[step]
                )

        latent = latent_single.repeat(num_samples, 1)
        mean = torch.zeros(self.horizon, self.action_dim, device=self.device)
        std = torch.full_like(mean, float(self.config["max_std"]))
        if not first_step:
            mean[:-1].copy_(self.previous_mean[1:])
        actions = torch.empty(
            self.horizon, num_samples, self.action_dim, device=self.device
        )
        if policy_actions is not None:
            actions[:, :num_policy].copy_(policy_actions)

        score = None
        elite_actions = None
        for _ in range(int(self.config["iterations"])):
            random_actions = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
                self.horizon,
                num_samples - num_policy,
                self.action_dim,
                device=self.device,
            )
            actions[:, num_policy:].copy_(random_actions.clamp(-1.0, 1.0))
            values = torch.nan_to_num(self._estimate_value(latent, actions), nan=0.0)
            elite_indices = torch.topk(values.squeeze(-1), num_elites).indices
            elite_values = values[elite_indices]
            elite_actions = actions[:, elite_indices]
            max_value = elite_values.max(dim=0).values
            score = torch.exp(
                float(self.config["temperature"]) * (elite_values - max_value)
            )
            score = score / (score.sum(dim=0, keepdim=True) + 1e-9)
            weights = score.unsqueeze(0)
            mean = (weights * elite_actions).sum(dim=1) / (
                weights.sum(dim=1) + 1e-9
            )
            variance = (weights * (elite_actions - mean.unsqueeze(1)).pow(2)).sum(
                dim=1
            ) / (weights.sum(dim=1) + 1e-9)
            std = variance.sqrt().clamp(
                float(self.config["min_std"]), float(self.config["max_std"])
            )

        assert score is not None and elite_actions is not None
        self.previous_mean.copy_(mean)
        if eval_mode:
            # Deterministic final selection; planning samples remain reproducible
            # when evaluate.py seeds PyTorch at each episode.
            action = mean[0]
        else:
            elite_index = torch.multinomial(score.squeeze(-1), 1).squeeze(0)
            action = elite_actions[0, elite_index] + std[0] * torch.randn_like(std[0])
        return action.clamp(-1.0, 1.0)

    @torch.no_grad()
    def _td_target(
        self,
        next_latent: torch.Tensor,
        reward: torch.Tensor,
        terminated: torch.Tensor,
    ) -> torch.Tensor:
        next_action, _ = self.model.pi(next_latent)
        next_q = self.model.q(
            next_latent, next_action, reduction="min", target=True
        )
        return reward + self.discount * (1.0 - terminated) * next_q

    def _compute_safety_loss(
        self,
        rollout_latent: torch.Tensor,
        actions: torch.Tensor,
        safety_cost: torch.Tensor,
        weights: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute aligned per-channel Safety losses for ``(z_t, a_t, c_t)``."""

        if rollout_latent.ndim != 3:
            raise ValueError(
                "Safety rollout latent must have shape (H, B, latent_dim), "
                f"got {tuple(rollout_latent.shape)}"
            )
        if actions.ndim != 3 or actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"Safety actions must have shape (H, B, {self.action_dim}), "
                f"got {tuple(actions.shape)}"
            )
        expected_prefix = (
            self.horizon,
            actions.shape[1],
        )
        if tuple(rollout_latent.shape[:2]) != expected_prefix:
            raise ValueError(
                "Safety rollout latent must have leading shape "
                f"{expected_prefix}, got {tuple(rollout_latent.shape)}"
            )
        if tuple(actions.shape[:2]) != expected_prefix:
            raise ValueError(
                f"Safety actions must have leading shape {expected_prefix}, "
                f"got {tuple(actions.shape)}"
            )
        expected_cost_shape = (*expected_prefix, self.safety_dim)
        if tuple(safety_cost.shape) != expected_cost_shape:
            raise ValueError(
                f"Safety cost must have shape {expected_cost_shape}, "
                f"got {tuple(safety_cost.shape)}"
            )
        if tuple(weights.shape) != (self.horizon,):
            raise ValueError(
                f"Safety weights must have shape ({self.horizon},), "
                f"got {tuple(weights.shape)}"
            )
        if not torch.isfinite(safety_cost).all():
            raise FloatingPointError("Safety cost contains NaN or infinity")
        if torch.any(safety_cost < 0.0):
            raise ValueError("Safety cost values must be nonnegative")

        prediction_transformed = self.model.safety_transformed(
            rollout_latent,
            actions,
        )
        target_transformed = self.model.transform_safety_targets(safety_cost)
        if tuple(prediction_transformed.shape) != expected_cost_shape:
            raise RuntimeError(
                "WorldModel.safety_transformed returned shape "
                f"{tuple(prediction_transformed.shape)}; expected "
                f"{expected_cost_shape}"
            )
        if tuple(target_transformed.shape) != expected_cost_shape:
            raise RuntimeError(
                "WorldModel.transform_safety_targets returned shape "
                f"{tuple(target_transformed.shape)}; expected "
                f"{expected_cost_shape}"
            )

        element_loss = F.smooth_l1_loss(
            prediction_transformed,
            target_transformed,
            reduction="none",
        )
        # Mean over batch independently for each channel, then apply TD-MPC2's
        # rho^t weighting and normalize by the fixed training horizon.
        channel_losses = (
            element_loss.mean(dim=1) * weights.unsqueeze(-1)
        ).sum(dim=0) / self.horizon
        curvature_loss = channel_losses[0]
        translation_error_loss = channel_losses[1]
        safety_loss = (
            self.safety_curvature_loss_coef * curvature_loss
            + self.safety_translation_error_loss_coef * translation_error_loss
        )

        with torch.no_grad():
            diagnostics = safety_batch_diagnostics(
                prediction_transformed,
                safety_cost,
                self.model,
                curvature_boundaries=self.curvature_boundaries,
                safety_cost_names=self.safety_cost_names,
            )

        return {
            "safety_loss": safety_loss,
            "safety_curvature_loss": curvature_loss,
            "safety_translation_error_loss": translation_error_loss,
            **diagnostics,
        }

    @torch.no_grad()
    def safety_validation_metrics(
        self,
        batch,
        *,
        prefix: str = "val_",
    ) -> Dict[str, torch.Tensor]:
        """Evaluate one independent replay batch without changing parameters."""

        if not isinstance(prefix, str):
            raise TypeError("Validation metric prefix must be a string")
        normalized_prefix = (
            prefix if not prefix or prefix.endswith("_") else f"{prefix}_"
        )
        observations, actions, _, _, safety_cost = batch
        expected_observation_shape = (
            self.horizon + 1,
            actions.shape[1],
            self.observation_dim,
        )
        expected_action_shape = (
            self.horizon,
            actions.shape[1],
            self.action_dim,
        )
        expected_cost_shape = (
            self.horizon,
            actions.shape[1],
            self.safety_dim,
        )
        if tuple(observations.shape) != expected_observation_shape:
            raise ValueError(
                "Validation observations must have shape "
                f"{expected_observation_shape}, got {tuple(observations.shape)}"
            )
        if tuple(actions.shape) != expected_action_shape:
            raise ValueError(
                f"Validation actions must have shape {expected_action_shape}, "
                f"got {tuple(actions.shape)}"
            )
        if tuple(safety_cost.shape) != expected_cost_shape:
            raise ValueError(
                f"Validation safety cost must have shape {expected_cost_shape}, "
                f"got {tuple(safety_cost.shape)}"
            )

        was_training = self.model.training
        self.model.train(False)
        try:
            latent = self.model.encode(observations[0])
            rollout = [latent]
            for step in range(self.horizon):
                latent = self.model.next(latent, actions[step])
                rollout.append(latent)
            rollout_latent = torch.stack(rollout, dim=0)[:-1]
            prediction_transformed = self.model.safety_transformed(
                rollout_latent,
                actions,
            )
            target_transformed = self.model.transform_safety_targets(
                safety_cost
            )
            element_loss = F.smooth_l1_loss(
                prediction_transformed,
                target_transformed,
                reduction="none",
            )
            weights = torch.pow(
                torch.tensor(float(self.config["rho"]), device=self.device),
                torch.arange(self.horizon, device=self.device),
            )
            channel_losses = (
                element_loss.mean(dim=1) * weights.unsqueeze(-1)
            ).sum(dim=0) / self.horizon
            curvature_loss = channel_losses[0]
            translation_error_loss = channel_losses[1]
            combined_loss = (
                self.safety_curvature_loss_coef * curvature_loss
                + self.safety_translation_error_loss_coef
                * translation_error_loss
            )
            diagnostics = safety_batch_diagnostics(
                prediction_transformed,
                safety_cost,
                self.model,
                prefix=normalized_prefix,
                curvature_boundaries=self.curvature_boundaries,
                safety_cost_names=self.safety_cost_names,
            )
            return {
                f"{normalized_prefix}safety_loss": combined_loss,
                f"{normalized_prefix}safety_curvature_loss": curvature_loss,
                f"{normalized_prefix}safety_translation_error_loss": (
                    translation_error_loss
                ),
                **diagnostics,
            }
        finally:
            self.model.train(was_training)

    def _gradient_diagnostics(
        self,
        safety_shared_gradients: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Return sparse pre-clipping Safety and total gradient norms."""

        return {
            "safety_grad_norm_trunk": self._module_gradient_norm(
                self.model.safety_trunk
            ),
            "safety_grad_norm_curvature_branch": self._module_gradient_norm(
                self.model.safety_curvature_head
            ),
            "safety_grad_norm_translation_error_branch": (
                self._module_gradient_norm(
                    self.model.safety_translation_error_head
                )
            ),
            # These total shared-module norms make the auxiliary-only values
            # below interpretable when checking whether Safety dominates.
            "total_grad_norm_encoder": self._module_gradient_norm(
                self.model.encoder
            ),
            "total_grad_norm_dynamics": self._module_gradient_norm(
                self.model.dynamics
            ),
            **dict(safety_shared_gradients),
        }

    def _module_gradient_norm(self, module: torch.nn.Module) -> torch.Tensor:
        squared_norm = torch.zeros((), device=self.device)
        for parameter in module.parameters():
            if parameter.grad is not None:
                squared_norm = squared_norm + parameter.grad.detach().pow(2).sum()
        return squared_norm.sqrt()

    def _safety_shared_gradient_diagnostics(
        self,
        scaled_safety_loss: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Measure auxiliary-only encoder/dynamics gradients without mutation."""

        encoder_parameters = list(self.model.encoder.parameters())
        dynamics_parameters = list(self.model.dynamics.parameters())
        parameters = encoder_parameters + dynamics_parameters
        gradients = torch.autograd.grad(
            scaled_safety_loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )

        def norm(values) -> torch.Tensor:
            squared_norm = torch.zeros((), device=self.device)
            for gradient in values:
                if gradient is not None:
                    squared_norm = (
                        squared_norm + gradient.detach().pow(2).sum()
                    )
            return squared_norm.sqrt()

        encoder_count = len(encoder_parameters)
        return {
            "safety_grad_norm_encoder": norm(
                gradients[:encoder_count]
            ),
            "safety_grad_norm_dynamics": norm(
                gradients[encoder_count:]
            ),
        }

    def _disabled_safety_info(self) -> Dict[str, torch.Tensor]:
        """Return stable loss keys without evaluating the disabled Safety Head."""

        zero = torch.zeros((), device=self.device)
        unavailable = torch.full((), float("nan"), device=self.device)
        info: Dict[str, torch.Tensor] = {
            "safety_loss": zero,
            "safety_curvature_loss": zero,
            "safety_translation_error_loss": zero,
        }
        for channel in ("curvature", "translation_error"):
            for metric in (
                "pred_mean",
                "target_mean",
                "pred_max",
                "target_max",
                "mae_transformed",
                "mae",
            ):
                if metric.startswith(("pred_", "target_")):
                    statistic, reduction = metric.split("_", 1)
                    name = (
                        f"safety_{statistic}_{channel}_{reduction}"
                    )
                else:
                    name = f"safety_{channel}_{metric}"
                info[name] = unavailable
        for name in (
            "safety_translation_error_positive_count",
            "safety_translation_error_positive_fraction",
            "safety_translation_error_zero_fraction",
            "safety_translation_error_positive_mae",
            "safety_translation_error_positive_pred_mean",
            "safety_translation_error_positive_target_mean",
            "safety_translation_error_zero_count",
            "safety_translation_error_zero_mae",
            "safety_translation_error_zero_pred_mean",
            "safety_translation_error_zero_target_mean",
        ):
            info[name] = unavailable
        for group in ("low", "medium", "high"):
            for metric in ("count", "mae", "pred_mean", "target_mean"):
                info[f"safety_curvature_{group}_{metric}"] = unavailable
        return info

    def update(self, replay_buffer) -> Dict[str, float]:
        observations, actions, rewards, terminated, safety_cost = (
            replay_buffer.sample(self.device)
        )
        rho = float(self.config["rho"])
        weights = torch.pow(
            torch.tensor(rho, device=self.device),
            torch.arange(self.horizon, device=self.device),
        )

        with torch.no_grad():
            next_latent_targets = self.model.encode(observations[1:])
            td_targets = self._td_target(
                next_latent_targets, rewards, terminated
            )

        self.model.train(True)
        latent = self.model.encode(observations[0])
        latent_rollout = [latent]
        consistency_loss = torch.zeros((), device=self.device)
        for step in range(self.horizon):
            latent = self.model.next(latent, actions[step])
            consistency_loss = consistency_loss + weights[step] * F.mse_loss(
                latent, next_latent_targets[step]
            )
            latent_rollout.append(latent)
        latent_rollout_tensor = torch.stack(latent_rollout, dim=0)

        rollout_latent = latent_rollout_tensor[:-1]
        reward_logits = self.model.reward(rollout_latent, actions)
        q_logits = self.model.q(rollout_latent, actions, reduction="all")
        termination_logits = self.model.termination(
            latent_rollout_tensor[1:], logits=True
        )

        reward_element = soft_cross_entropy(
            reward_logits,
            rewards,
            self.model.num_bins,
            self.model.value_min,
            self.model.value_max,
        ).mean(dim=(1, 2))
        reward_loss = (reward_element * weights).sum() / self.horizon

        value_element = soft_cross_entropy(
            q_logits,
            td_targets.unsqueeze(0).expand(self.model.num_q, -1, -1, -1),
            self.model.num_bins,
            self.model.value_min,
            self.model.value_max,
        ).mean(dim=(0, 2, 3))
        value_loss = (value_element * weights).sum() / self.horizon
        consistency_loss = consistency_loss / self.horizon
        termination_loss = F.binary_cross_entropy_with_logits(
            termination_logits, terminated
        )
        safety_info = self._disabled_safety_info()
        if self.safety_training_enabled:
            safety_info = self._compute_safety_loss(
                latent_rollout_tensor[:-1],
                actions,
                safety_cost,
                weights,
            )
        total_loss = (
            float(self.config["consistency_coef"]) * consistency_loss
            + float(self.config["reward_coef"]) * reward_loss
            + float(self.config["value_coef"]) * value_loss
            + float(self.config["termination_coef"]) * termination_loss
            + self.safety_loss_coef * safety_info["safety_loss"]
        )

        self.model_optimizer.zero_grad(set_to_none=True)
        diagnostic_gradient_due = (
            (self.update_count + 1) % self.gradient_interval == 0
        )
        safety_shared_gradients: Dict[str, torch.Tensor] = {}
        if diagnostic_gradient_due and self.safety_training_enabled:
            safety_shared_gradients = (
                self._safety_shared_gradient_diagnostics(
                    self.safety_loss_coef * safety_info["safety_loss"]
                )
            )
        elif diagnostic_gradient_due:
            zero = torch.zeros((), device=self.device)
            safety_shared_gradients = {
                "safety_grad_norm_encoder": zero,
                "safety_grad_norm_dynamics": zero.clone(),
            }
        total_loss.backward()
        gradient_info: Dict[str, torch.Tensor] = {}
        if diagnostic_gradient_due:
            gradient_info = self._gradient_diagnostics(
                safety_shared_gradients
            )
        model_grad_norm = torch.nn.utils.clip_grad_norm_(
            [
                parameter
                for group in self.model_optimizer.param_groups
                for parameter in group["params"]
            ],
            float(self.config["grad_clip_norm"]),
        )
        self.model_optimizer.step()

        policy_info = self._update_policy(latent_rollout_tensor.detach())
        self.model.soft_update_target_q(float(self.config["tau"]))
        self.model.train(False)
        next_update_count = self.update_count + 1
        validation_info: Dict[str, torch.Tensor] = {}
        if (
            self.safety_training_enabled
            and next_update_count % self.validation_interval == 0
        ):
            diagnostic_sampler = getattr(
                replay_buffer,
                "sample_diagnostics",
                None,
            )
            if not callable(diagnostic_sampler):
                raise TypeError(
                    "Replay buffer must implement sample_diagnostics() so "
                    "validation cannot advance the training sampler"
                )
            validation_batch = diagnostic_sampler(
                self.device,
                seed=next_update_count,
            )
            validation_info = self.safety_validation_metrics(
                validation_batch,
                prefix="val_",
            )
        self.update_count = next_update_count
        return {
            "total_loss": float(total_loss.detach().cpu()),
            "consistency_loss": float(consistency_loss.detach().cpu()),
            "reward_loss": float(reward_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "termination_loss": float(termination_loss.detach().cpu()),
            "model_grad_norm": float(model_grad_norm.detach().cpu()),
            **{
                name: float(value.detach().cpu())
                for name, value in safety_info.items()
            },
            **{
                name: float(value.detach().cpu())
                for name, value in gradient_info.items()
            },
            **{
                name: float(value.detach().cpu())
                for name, value in validation_info.items()
            },
            **policy_info,
        }

    def _update_policy(self, latent_rollout: torch.Tensor) -> Dict[str, float]:
        self.policy_optimizer.zero_grad(set_to_none=True)
        action, info = self.model.pi(latent_rollout)

        critic_parameters = list(self.model.q_ensemble.parameters())
        for parameter in critic_parameters:
            parameter.requires_grad_(False)
        q_values = self.model.q(latent_rollout, action, reduction="avg")
        for parameter in critic_parameters:
            parameter.requires_grad_(True)

        self.scale.update(q_values[0])
        normalized_q = self.scale(q_values)
        weights = torch.pow(
            torch.tensor(float(self.config["rho"]), device=self.device),
            torch.arange(latent_rollout.shape[0], device=self.device),
        )
        objective = (
            float(self.config["entropy_coef"]) * info["scaled_entropy"]
            + normalized_q
        )
        policy_loss = -(objective.mean(dim=(1, 2)) * weights).mean()
        policy_loss.backward()
        policy_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.policy.parameters(), float(self.config["grad_clip_norm"])
        )
        self.policy_optimizer.step()
        return {
            "policy_loss": float(policy_loss.detach().cpu()),
            "policy_grad_norm": float(policy_grad_norm.detach().cpu()),
            "policy_entropy": float(info["entropy"].detach().mean().cpu()),
            "policy_scale": float(self.scale.value.detach().cpu()),
        }

    def _safety_state_config(self) -> Dict[str, float]:
        return {
            "safety_loss_coef": self.safety_loss_coef,
            "safety_curvature_loss_coef": self.safety_curvature_loss_coef,
            "safety_translation_error_loss_coef": (
                self.safety_translation_error_loss_coef
            ),
            "safety_curvature_scale_mm_inv": (
                self.safety_curvature_scale_mm_inv
            ),
            "safety_translation_error_scale": self.safety_translation_error_scale,
        }

    def _validate_safety_state(self, state: Mapping[str, Any]) -> None:
        required = {
            "safety_model_schema_version",
            "safety_cost_names",
            "safety_dim",
            "safety_config",
        }
        missing = sorted(required - state.keys())
        if missing:
            raise ValueError(
                "Agent checkpoint predates the required Safety Head/model "
                f"schema; missing metadata {missing}"
            )
        schema_version = int(state["safety_model_schema_version"])
        if schema_version != SAFETY_MODEL_SCHEMA_VERSION:
            raise ValueError(
                f"Agent checkpoint Safety model schema version {schema_version} "
                f"does not match required version {SAFETY_MODEL_SCHEMA_VERSION}"
            )
        received_names = tuple(state["safety_cost_names"])
        if received_names != self.safety_cost_names:
            raise ValueError(
                "Agent checkpoint safety_cost_names "
                f"{received_names} do not match current "
                f"{self.safety_cost_names} in this exact order"
            )
        received_dim = int(state["safety_dim"])
        if received_dim != self.safety_dim:
            raise ValueError(
                f"Agent checkpoint safety_dim {received_dim} does not match "
                f"current {self.safety_dim}"
            )
        received_config = state["safety_config"]
        if not isinstance(received_config, Mapping):
            raise TypeError("Agent checkpoint safety_config must be a mapping")
        expected_config = self._safety_state_config()
        missing_config = sorted(expected_config.keys() - received_config.keys())
        if missing_config:
            raise ValueError(
                "Agent checkpoint safety_config is missing required keys "
                f"{missing_config}"
            )
        unexpected_config = sorted(received_config.keys() - expected_config.keys())
        if unexpected_config:
            raise ValueError(
                "Agent checkpoint safety_config has unexpected keys "
                f"{unexpected_config}"
            )
        for name, expected_value in expected_config.items():
            received_value = float(received_config[name])
            if received_value != expected_value:
                raise ValueError(
                    f"Agent checkpoint {name}={received_value} does not match "
                    f"current value {expected_value}"
                )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "safety_model_schema_version": SAFETY_MODEL_SCHEMA_VERSION,
            "safety_cost_names": self.safety_cost_names,
            "safety_dim": self.safety_dim,
            "safety_config": self._safety_state_config(),
            "model": self.model.state_dict(),
            "model_optimizer": self.model_optimizer.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "scale": self.scale.state_dict(),
            "previous_mean": self.previous_mean.detach().cpu(),
            "update_count": self.update_count,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "discount": self.discount,
        }

    def load_state_dict(
        self, state: Mapping[str, Any], *, load_optimizers: bool = True
    ) -> None:
        self._validate_safety_state(state)
        if int(state["observation_dim"]) != self.observation_dim or int(
            state["action_dim"]
        ) != self.action_dim:
            raise ValueError("Checkpoint observation/action dimensions do not match")
        self.model.load_state_dict(state["model"])
        if load_optimizers:
            self.model_optimizer.load_state_dict(state["model_optimizer"])
            self.policy_optimizer.load_state_dict(state["policy_optimizer"])
        self.scale.load_state_dict(state["scale"])
        self.previous_mean.copy_(state["previous_mean"].to(self.device))
        self.update_count = int(state.get("update_count", 0))
        self.model.train(False)
