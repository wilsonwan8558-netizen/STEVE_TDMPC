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

from envs.safety import CURVATURE_STRATUM_NAMES
from eve.intervention import TRANSLATION_BLOCK_REASON_NAMES

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
_TRANSLATION_AUXILIARY_LOG_LABELS = {
    "none": "none",
    "lower_insertion_boundary": "lower",
    "device_length_limit": "device",
    "vessel_tree_end": "tree_end",
    "other": "other",
}


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

    @staticmethod
    def _auxiliary_nonnegative_coefficient(value: Any, name: str) -> float:
        """Strictly parse one auxiliary-only loss coefficient."""

        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be a real number, not bool")
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(f"{name} must be a real number") from exc
        if not np.isfinite(parsed) or parsed < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
        return parsed

    def _compute_auxiliary_safety_loss(
        self,
        auxiliary_batch: Mapping[str, Any],
        *,
        safety_aux_curvature_loss_coef: float = 1.0,
        safety_aux_translation_loss_coef: float = 1.0,
        safety_aux_translation_group_weights: Optional[
            Mapping[str, Any]
        ] = None,
        safety_aux_translation_zero_calibration_coef: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """Compute group-normalized detached Safety Head supervision.

        Auxiliary observations are encoded without autograd and detached again
        before entering the Safety Head. Translation Smooth-L1 losses are
        averaged inside each canonical blockage-reason group before a weighted
        mean across available groups. Curvature losses are averaged inside each
        stored stratum and then equally across available strata. The requested
        action (``batch["action"]``), rather than ``applied_action``, preserves
        the main replay's ``(z_t, a_t, c_t)`` convention.
        """

        if not isinstance(auxiliary_batch, Mapping):
            raise TypeError("Auxiliary Safety batch must be a mapping")
        required_keys = {
            "observation",
            "action",
            "safety_cost",
            "translation_block_reason_id",
            "curvature_stratum_id",
        }
        missing_keys = sorted(required_keys - auxiliary_batch.keys())
        if missing_keys:
            raise ValueError(
                "Auxiliary Safety batch is missing required fields "
                f"{missing_keys}"
            )

        curvature_coefficient = self._auxiliary_nonnegative_coefficient(
            safety_aux_curvature_loss_coef,
            "safety_aux_curvature_loss_coef",
        )
        translation_coefficient = self._auxiliary_nonnegative_coefficient(
            safety_aux_translation_loss_coef,
            "safety_aux_translation_loss_coef",
        )
        zero_calibration_coefficient = (
            self._auxiliary_nonnegative_coefficient(
                safety_aux_translation_zero_calibration_coef,
                "safety_aux_translation_zero_calibration_coef",
            )
        )

        canonical_reason_names = tuple(TRANSLATION_BLOCK_REASON_NAMES)
        if safety_aux_translation_group_weights is None:
            group_weights = {
                name: 1.0 for name in canonical_reason_names
            }
        else:
            if not isinstance(
                safety_aux_translation_group_weights,
                Mapping,
            ):
                raise TypeError(
                    "safety_aux_translation_group_weights must be a mapping"
                )
            received_names = tuple(
                safety_aux_translation_group_weights.keys()
            )
            if received_names != canonical_reason_names:
                raise ValueError(
                    "safety_aux_translation_group_weights keys must match "
                    f"canonical names and order {canonical_reason_names}, got "
                    f"{received_names}"
                )
            group_weights = {
                name: self._auxiliary_nonnegative_coefficient(
                    safety_aux_translation_group_weights[name],
                    (
                        "safety_aux_translation_group_weights"
                        f"[{name!r}]"
                    ),
                )
                for name in canonical_reason_names
            }

        def tensor_field(
            name: str,
            final_dim: int,
        ) -> torch.Tensor:
            value = auxiliary_batch[name]
            if not isinstance(value, np.ndarray):
                raise TypeError(
                    f"Auxiliary Safety {name} must be a numpy.ndarray"
                )
            if value.dtype != np.float32:
                raise TypeError(
                    f"Auxiliary Safety {name} must use float32, "
                    f"got {value.dtype}"
                )
            tensor = torch.as_tensor(value, device=self.device)
            if tensor.ndim != 2 or tensor.shape[1] != final_dim:
                raise ValueError(
                    f"Auxiliary Safety {name} must have shape "
                    f"(B, {final_dim}), got {tuple(tensor.shape)}"
                )
            if tensor.shape[0] <= 0:
                raise ValueError("Auxiliary Safety batch must not be empty")
            if not torch.isfinite(tensor).all():
                raise FloatingPointError(
                    f"Auxiliary Safety {name} contains NaN or infinity"
                )
            return tensor

        observations = tensor_field("observation", self.observation_dim)
        actions = tensor_field("action", self.action_dim)
        safety_cost = tensor_field("safety_cost", self.safety_dim)
        batch_size = observations.shape[0]
        if actions.shape[0] != batch_size or safety_cost.shape[0] != batch_size:
            raise ValueError(
                "Auxiliary Safety observation, action, and safety_cost batch "
                "dimensions must match"
            )
        if bool(torch.any(safety_cost < 0.0)):
            raise ValueError("Auxiliary Safety cost values must be nonnegative")

        def id_field(
            name: str,
            canonical_names: tuple[str, ...],
        ) -> torch.Tensor:
            value = auxiliary_batch[name]
            if not isinstance(value, np.ndarray):
                raise TypeError(
                    f"Auxiliary Safety {name} must be a numpy.ndarray"
                )
            if value.dtype != np.int64:
                raise TypeError(
                    f"Auxiliary Safety {name} must use int64, "
                    f"got {value.dtype}"
                )
            if value.shape != (batch_size,):
                raise ValueError(
                    f"Auxiliary Safety {name} must have shape "
                    f"({batch_size},), got {value.shape}"
                )
            invalid = np.unique(
                value[(value < 0) | (value >= len(canonical_names))]
            )
            if invalid.size:
                raise ValueError(
                    f"Auxiliary Safety {name} contains unknown canonical IDs "
                    f"{invalid.tolist()}; valid range is "
                    f"[0, {len(canonical_names) - 1}]"
                )
            return torch.as_tensor(
                value,
                device=self.device,
                dtype=torch.long,
            )

        reason_ids = id_field(
            "translation_block_reason_id",
            canonical_reason_names,
        )
        canonical_stratum_names = tuple(CURVATURE_STRATUM_NAMES)
        stratum_ids = id_field(
            "curvature_stratum_id",
            canonical_stratum_names,
        )

        with torch.no_grad():
            latent = self.model.encode(observations)
        latent = latent.detach()
        if latent.requires_grad:
            raise RuntimeError("Auxiliary Safety latent unexpectedly requires grad")

        prediction_transformed = self.model.safety_transformed(latent, actions)
        target_transformed = self.model.transform_safety_targets(safety_cost)
        expected_shape = (batch_size, self.safety_dim)
        if tuple(prediction_transformed.shape) != expected_shape:
            raise RuntimeError(
                "WorldModel.safety_transformed returned auxiliary shape "
                f"{tuple(prediction_transformed.shape)}; expected "
                f"{expected_shape}"
            )
        if tuple(target_transformed.shape) != expected_shape:
            raise RuntimeError(
                "WorldModel.transform_safety_targets returned auxiliary shape "
                f"{tuple(target_transformed.shape)}; expected {expected_shape}"
            )
        if not bool(torch.isfinite(prediction_transformed).all()):
            raise FloatingPointError(
                "Auxiliary Safety prediction contains NaN or infinity"
            )
        if not bool(torch.isfinite(target_transformed).all()):
            raise FloatingPointError(
                "Auxiliary Safety transformed target contains NaN or infinity"
            )

        # These transitions are independent, so no temporal rho weighting is
        # applied. Group normalization also prevents sampled group cardinality
        # from implicitly becoming an extra loss weight.
        element_loss = F.smooth_l1_loss(
            prediction_transformed,
            target_transformed,
            reduction="none",
        )
        unavailable = prediction_transformed.new_tensor(float("nan"))
        group_info: Dict[str, torch.Tensor] = {}

        curvature_group_losses = []
        for stratum_id, stratum_name in enumerate(canonical_stratum_names):
            mask = stratum_ids == stratum_id
            available = bool(torch.any(mask))
            loss = (
                element_loss[mask, 0].mean()
                if available
                else unavailable.clone()
            )
            group_info[
                f"safety_aux_curvature_{stratum_name}_loss"
            ] = loss
            group_info[
                f"safety_aux_curvature_{stratum_name}_available"
            ] = prediction_transformed.new_tensor(float(available))
            if available:
                curvature_group_losses.append(loss)
        curvature_loss = torch.stack(curvature_group_losses).mean()

        weighted_translation_losses = []
        available_translation_weight = 0.0
        available_translation_count = 0
        for reason_id, reason_name in enumerate(canonical_reason_names):
            mask = reason_ids == reason_id
            available = bool(torch.any(mask))
            loss = (
                element_loss[mask, 1].mean()
                if available
                else unavailable.clone()
            )
            log_label = _TRANSLATION_AUXILIARY_LOG_LABELS[reason_name]
            group_info[
                f"safety_aux_translation_{log_label}_loss"
            ] = loss
            group_info[
                f"safety_aux_translation_{log_label}_available"
            ] = prediction_transformed.new_tensor(float(available))
            if available:
                available_translation_count += 1
                weight = group_weights[reason_name]
                if weight > 0.0:
                    weighted_translation_losses.append(weight * loss)
                    available_translation_weight += weight
        if not weighted_translation_losses:
            available_names = tuple(
                reason_name
                for reason_id, reason_name in enumerate(canonical_reason_names)
                if bool(torch.any(reason_ids == reason_id))
            )
            raise ValueError(
                "At least one available auxiliary translation group must have "
                "a positive weight; available groups are "
                f"{available_names}"
            )
        translation_error_loss = (
            torch.stack(weighted_translation_losses).sum()
            / available_translation_weight
        )

        prediction_original = self.model.decode_safety_transformed(
            prediction_transformed
        )
        if not bool(torch.isfinite(prediction_original).all()):
            raise FloatingPointError(
                "Auxiliary Safety decoded prediction contains NaN or infinity"
            )
        translation_prediction = prediction_original[:, 1]
        translation_target = safety_cost[:, 1]
        zero_target_mask = translation_target == 0.0
        positive_target_mask = translation_target > 0.0
        none_mask = reason_ids == canonical_reason_names.index("none")

        def conditional_mean(
            values: torch.Tensor,
            mask: torch.Tensor,
        ) -> torch.Tensor:
            return (
                values[mask].mean()
                if bool(torch.any(mask))
                else unavailable.clone()
            )

        zero_calibration_loss = (
            translation_prediction[zero_target_mask].pow(2).mean()
            if bool(torch.any(zero_target_mask))
            else unavailable.clone()
        )
        zero_calibration_contribution = (
            zero_calibration_coefficient * zero_calibration_loss
            if bool(torch.any(zero_target_mask))
            else prediction_transformed.sum() * 0.0
        )
        combined_loss = (
            curvature_coefficient * curvature_loss
            + translation_coefficient
            * (translation_error_loss + zero_calibration_contribution)
        )
        if not bool(torch.isfinite(combined_loss)):
            raise FloatingPointError("Auxiliary Safety loss is not finite")
        return {
            "safety_aux_loss": combined_loss,
            "safety_aux_curvature_loss": curvature_loss,
            "safety_aux_translation_error_loss": translation_error_loss,
            "safety_aux_translation_zero_calibration_loss": (
                zero_calibration_loss
            ),
            "safety_aux_translation_positive_prediction_mean": (
                conditional_mean(
                    translation_prediction,
                    positive_target_mask,
                )
            ),
            "safety_aux_translation_none_prediction_mean": (
                conditional_mean(translation_prediction, none_mask)
            ),
            "safety_aux_translation_available_group_count": (
                combined_loss.new_tensor(float(available_translation_count))
            ),
            "safety_aux_curvature_available_group_count": (
                combined_loss.new_tensor(float(len(curvature_group_losses)))
            ),
            "safety_aux_batch_size": combined_loss.new_tensor(
                float(batch_size)
            ),
            **group_info,
        }

    def _disabled_auxiliary_info(self) -> Dict[str, torch.Tensor]:
        """Return stable unavailable metrics without evaluating auxiliary data."""

        unavailable = torch.full((), float("nan"), device=self.device)
        names = [
            "safety_aux_loss",
            "safety_aux_curvature_loss",
            "safety_aux_translation_error_loss",
            "safety_aux_translation_zero_calibration_loss",
            "safety_aux_translation_positive_prediction_mean",
            "safety_aux_translation_none_prediction_mean",
            "safety_aux_translation_available_group_count",
            "safety_aux_curvature_available_group_count",
            "safety_aux_batch_size",
        ]
        for reason_name in TRANSLATION_BLOCK_REASON_NAMES:
            log_label = _TRANSLATION_AUXILIARY_LOG_LABELS[reason_name]
            names.extend(
                (
                    f"safety_aux_translation_{log_label}_loss",
                    f"safety_aux_translation_{log_label}_available",
                )
            )
        for stratum_name in CURVATURE_STRATUM_NAMES:
            names.extend(
                (
                    f"safety_aux_curvature_{stratum_name}_loss",
                    f"safety_aux_curvature_{stratum_name}_available",
                )
            )
        return {name: unavailable.clone() for name in names}

    def _disabled_auxiliary_gradient_info(self) -> Dict[str, torch.Tensor]:
        """Return unavailable auxiliary-gradient metrics for unscheduled updates."""

        unavailable = torch.full((), float("nan"), device=self.device)
        module_names = (
            "safety_head",
            "safety_trunk",
            "safety_curvature_branch",
            "safety_translation_error_branch",
            "encoder",
            "dynamics",
            "reward_head",
            "termination_head",
            "q_ensemble",
            "policy",
        )
        output: Dict[str, torch.Tensor] = {}
        for module_name in module_names:
            for source in ("main", "aux", "combined"):
                output[
                    f"{source}_grad_norm_{module_name}"
                ] = unavailable.clone()
        output["aux_grad_norm_safety_head_clipped"] = unavailable.clone()
        output["combined_grad_norm_safety_head_after_clipping"] = (
            unavailable.clone()
        )
        return output

    def _gradient_values_norm(
        self,
        gradients,
    ) -> torch.Tensor:
        """Return an L2 norm for detached gradient tensors/``None`` values."""

        squared_norm = torch.zeros((), device=self.device)
        for gradient in gradients:
            if gradient is not None:
                squared_norm = (
                    squared_norm + gradient.detach().pow(2).sum()
                )
        return squared_norm.sqrt()

    @staticmethod
    def _combined_gradient_values(main_gradients, auxiliary_gradients):
        """Add aligned optional gradient sequences without mutating either."""

        combined = []
        for main_gradient, auxiliary_gradient in zip(
            main_gradients,
            auxiliary_gradients,
        ):
            if main_gradient is None:
                combined.append(auxiliary_gradient)
            elif auxiliary_gradient is None:
                combined.append(main_gradient)
            else:
                combined.append(main_gradient + auxiliary_gradient)
        return tuple(combined)

    def _isolated_auxiliary_gradient_step(
        self,
        *,
        total_loss: torch.Tensor,
        main_safety_objective: Optional[torch.Tensor],
        auxiliary_objective: torch.Tensor,
        diagnostic_gradient_due: bool,
        safety_shared_gradients: Mapping[str, torch.Tensor],
    ):
        """Backpropagate once while isolating auxiliary global-clip effects.

        A naive global clip of ``main + auxiliary`` gradients would let a large
        auxiliary Safety gradient change the clip factor applied to encoder,
        dynamics, reward, termination, and Q parameters.  This method preserves
        the exact main-gradient clipping operation: it snapshots pure main and
        auxiliary Safety gradients, performs the single total backward pass,
        restores pure main Safety gradients for the existing global clip, then
        independently clips and adds only the auxiliary Safety gradients before
        the single optimizer step.
        """

        safety_groups = {
            "safety_trunk": tuple(self.model.safety_trunk.parameters()),
            "safety_curvature_branch": tuple(
                self.model.safety_curvature_head.parameters()
            ),
            "safety_translation_error_branch": tuple(
                self.model.safety_translation_error_head.parameters()
            ),
        }
        safety_parameters = tuple(
            parameter
            for parameters in safety_groups.values()
            for parameter in parameters
        )
        forbidden_groups = {
            "encoder": tuple(self.model.encoder.parameters()),
            "dynamics": tuple(self.model.dynamics.parameters()),
            "reward_head": tuple(self.model.reward_head.parameters()),
            "termination_head": tuple(
                self.model.termination_head.parameters()
            ),
            "q_ensemble": tuple(self.model.q_ensemble.parameters()),
            "policy": tuple(self.model.policy.parameters()),
        }
        forbidden_parameters = tuple(
            parameter
            for parameters in forbidden_groups.values()
            for parameter in parameters
        )

        if main_safety_objective is None:
            main_safety_gradients = tuple(None for _ in safety_parameters)
        else:
            main_safety_gradients = torch.autograd.grad(
                main_safety_objective,
                safety_parameters,
                retain_graph=True,
                allow_unused=True,
            )
        auxiliary_all_gradients = torch.autograd.grad(
            auxiliary_objective,
            (*safety_parameters, *forbidden_parameters),
            retain_graph=True,
            allow_unused=True,
        )
        auxiliary_safety_gradients = tuple(
            auxiliary_all_gradients[: len(safety_parameters)]
        )
        auxiliary_forbidden_gradients = tuple(
            auxiliary_all_gradients[len(safety_parameters) :]
        )
        if any(
            gradient is not None
            for gradient in auxiliary_forbidden_gradients
        ):
            raise RuntimeError(
                "Detached auxiliary Safety loss unexpectedly has an autograd "
                "path to a forbidden non-Safety module"
            )

        total_loss.backward()
        gradient_info: Dict[str, torch.Tensor] = {}

        main_by_id = {
            id(parameter): gradient
            for parameter, gradient in zip(
                safety_parameters,
                main_safety_gradients,
            )
        }
        auxiliary_by_id = {
            id(parameter): gradient
            for parameter, gradient in zip(
                safety_parameters,
                auxiliary_safety_gradients,
            )
        }

        def values_for(parameters, source: str):
            if source == "main":
                return tuple(
                    main_by_id[id(parameter)] for parameter in parameters
                )
            if source == "aux":
                return tuple(
                    auxiliary_by_id[id(parameter)] for parameter in parameters
                )
            return self._combined_gradient_values(
                values_for(parameters, "main"),
                values_for(parameters, "aux"),
            )

        auxiliary_gradient_info: Dict[str, torch.Tensor] = {}
        for group_name, parameters in safety_groups.items():
            for source in ("main", "aux", "combined"):
                auxiliary_gradient_info[
                    f"{source}_grad_norm_{group_name}"
                ] = self._gradient_values_norm(
                    values_for(parameters, source)
                )
        for source in ("main", "aux", "combined"):
            auxiliary_gradient_info[
                f"{source}_grad_norm_safety_head"
            ] = self._gradient_values_norm(
                values_for(safety_parameters, source)
            )

        forbidden_offset = 0
        for group_name, parameters in forbidden_groups.items():
            group_size = len(parameters)
            auxiliary_values = auxiliary_forbidden_gradients[
                forbidden_offset : forbidden_offset + group_size
            ]
            forbidden_offset += group_size
            # The policy uses a separate optimizer and can still hold gradients
            # from the preceding policy step. It has no path from either model
            # objective, so do not misreport those stale tensors as main grads.
            main_values = (
                tuple(None for _ in parameters)
                if group_name == "policy"
                else tuple(parameter.grad for parameter in parameters)
            )
            combined_values = self._combined_gradient_values(
                main_values,
                auxiliary_values,
            )
            auxiliary_gradient_info[
                f"main_grad_norm_{group_name}"
            ] = self._gradient_values_norm(main_values)
            auxiliary_gradient_info[
                f"aux_grad_norm_{group_name}"
            ] = self._gradient_values_norm(auxiliary_values)
            auxiliary_gradient_info[
                f"combined_grad_norm_{group_name}"
            ] = self._gradient_values_norm(combined_values)

        # Restore pure main Safety gradients before applying TD-MPC2's existing
        # global model-gradient clip. Non-Safety gradients already contain only
        # the main objective because the auxiliary graph is detached.
        for parameter, gradient in zip(
            safety_parameters,
            main_safety_gradients,
        ):
            parameter.grad = (
                None
                if gradient is None
                else gradient.detach().clone()
            )
        if diagnostic_gradient_due:
            # Preserve the established metric semantics: these are pure-main
            # pre-clipping gradients. Separate auxiliary/combined metrics above
            # expose the additional supervision.
            gradient_info.update(
                self._gradient_diagnostics(safety_shared_gradients)
            )
        model_grad_norm = torch.nn.utils.clip_grad_norm_(
            [
                parameter
                for group in self.model_optimizer.param_groups
                for parameter in group["params"]
            ],
            float(self.config["grad_clip_norm"]),
        )

        auxiliary_norm = self._gradient_values_norm(
            auxiliary_safety_gradients
        )
        if not torch.isfinite(auxiliary_norm):
            raise FloatingPointError(
                "Auxiliary Safety gradient norm is not finite"
            )
        maximum_norm = float(self.config["grad_clip_norm"])
        clip_scale = torch.clamp(
            auxiliary_norm.new_tensor(maximum_norm)
            / (auxiliary_norm + 1.0e-6),
            max=1.0,
        )
        clipped_auxiliary_gradients = tuple(
            None
            if gradient is None
            else gradient.detach() * clip_scale
            for gradient in auxiliary_safety_gradients
        )
        for parameter, gradient in zip(
            safety_parameters,
            clipped_auxiliary_gradients,
        ):
            if gradient is None:
                continue
            if parameter.grad is None:
                parameter.grad = gradient.clone()
            else:
                parameter.grad.add_(gradient)
        auxiliary_gradient_info[
            "aux_grad_norm_safety_head_clipped"
        ] = self._gradient_values_norm(clipped_auxiliary_gradients)
        auxiliary_gradient_info[
            "combined_grad_norm_safety_head_after_clipping"
        ] = self._gradient_values_norm(
            tuple(parameter.grad for parameter in safety_parameters)
        )
        return model_grad_norm, gradient_info, auxiliary_gradient_info

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
            # These total shared-module norms make the main Safety-only values
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
        """Measure main Safety-only encoder/dynamics gradients without mutation."""

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

    def update(
        self,
        replay,
        *,
        safety_aux_batch: Optional[Mapping[str, Any]] = None,
        safety_aux_loss_coef: float = 0.0,
        safety_aux_curvature_loss_coef: float = 1.0,
        safety_aux_translation_loss_coef: float = 1.0,
        safety_aux_translation_group_weights: Optional[
            Mapping[str, Any]
        ] = None,
        safety_aux_translation_zero_calibration_coef: float = 0.0,
    ) -> Dict[str, float]:
        if isinstance(safety_aux_loss_coef, (bool, np.bool_)):
            raise TypeError(
                "safety_aux_loss_coef must be a real number, not bool"
            )
        try:
            parsed_auxiliary_loss_coef = float(safety_aux_loss_coef)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(
                "safety_aux_loss_coef must be a real number"
            ) from exc
        if (
            not np.isfinite(parsed_auxiliary_loss_coef)
            or parsed_auxiliary_loss_coef < 0.0
        ):
            raise ValueError(
                "safety_aux_loss_coef must be finite and nonnegative"
            )

        observations, actions, rewards, terminated, safety_cost = (
            replay.sample(self.device)
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
        main_total_loss = (
            float(self.config["consistency_coef"]) * consistency_loss
            + float(self.config["reward_coef"]) * reward_loss
            + float(self.config["value_coef"]) * value_loss
            + float(self.config["termination_coef"]) * termination_loss
            + self.safety_loss_coef * safety_info["safety_loss"]
        )
        auxiliary_info = self._disabled_auxiliary_info()
        auxiliary_gradient_info = (
            self._disabled_auxiliary_gradient_info()
        )
        auxiliary_objective: Optional[torch.Tensor] = None
        if safety_aux_batch is not None:
            auxiliary_info = self._compute_auxiliary_safety_loss(
                safety_aux_batch,
                safety_aux_curvature_loss_coef=(
                    safety_aux_curvature_loss_coef
                ),
                safety_aux_translation_loss_coef=(
                    safety_aux_translation_loss_coef
                ),
                safety_aux_translation_group_weights=(
                    safety_aux_translation_group_weights
                ),
                safety_aux_translation_zero_calibration_coef=(
                    safety_aux_translation_zero_calibration_coef
                ),
            )
            auxiliary_objective = (
                parsed_auxiliary_loss_coef
                * auxiliary_info["safety_aux_loss"]
            )
        total_loss = (
            main_total_loss
            if auxiliary_objective is None
            else main_total_loss + auxiliary_objective
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
        gradient_info: Dict[str, torch.Tensor] = {}
        if auxiliary_objective is None:
            # Preserve the existing update path exactly when auxiliary
            # supervision is disabled or not scheduled.
            total_loss.backward()
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
        else:
            main_safety_objective = (
                self.safety_loss_coef * safety_info["safety_loss"]
                if self.safety_training_enabled
                else None
            )
            (
                model_grad_norm,
                gradient_info,
                auxiliary_gradient_info,
            ) = self._isolated_auxiliary_gradient_step(
                total_loss=total_loss,
                main_safety_objective=main_safety_objective,
                auxiliary_objective=auxiliary_objective,
                diagnostic_gradient_due=diagnostic_gradient_due,
                safety_shared_gradients=safety_shared_gradients,
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
                replay,
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
                for name, value in auxiliary_info.items()
            },
            **{
                name: float(value.detach().cpu())
                for name, value in gradient_info.items()
            },
            **{
                name: float(value.detach().cpu())
                for name, value in auxiliary_gradient_info.items()
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
