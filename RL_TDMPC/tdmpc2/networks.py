"""Core TD-MPC2 implicit world-model networks.

This is a dependency-light single-task/state-only adaptation of the official
TD-MPC2 implementation at commit e9f59321933cbc8e11a002b842adc7d4ffae8ff1.
It retains SimNorm latents, distributional reward/value prediction, an
ensemble of Q-functions, a learned dynamics model, and a Gaussian policy
prior. Multi-task embeddings and pixel encoders are intentionally omitted.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


class SimNorm(nn.Module):
    """Normalize groups of latent features onto probability simplices."""

    def __init__(self, group_dim: int) -> None:
        super().__init__()
        self.group_dim = int(group_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] % self.group_dim != 0:
            raise ValueError(
                f"Latent dimension {value.shape[-1]} must be divisible by "
                f"SimNorm group dimension {self.group_dim}."
            )
        shape = value.shape
        value = value.reshape(*shape[:-1], -1, self.group_dim)
        return F.softmax(value, dim=-1).reshape(shape)


class NormedLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        dropout: float = 0.0,
        activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.activation = activation or nn.Mish()
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = F.linear(value, self.weight, self.bias)
        return self.activation(self.norm(self.dropout(value)))


def mlp(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    *,
    output_activation: Optional[nn.Module] = None,
    dropout: float = 0.0,
) -> nn.Sequential:
    dimensions = [input_dim, *hidden_dims, output_dim]
    layers: List[nn.Module] = []
    for index in range(len(dimensions) - 2):
        layers.append(
            NormedLinear(
                dimensions[index],
                dimensions[index + 1],
                dropout=dropout if index == 0 else 0.0,
            )
        )
    if output_activation is None:
        layers.append(nn.Linear(dimensions[-2], dimensions[-1]))
    else:
        # Official TD-MPC2 uses a LayerNorm immediately before SimNorm for
        # encoder and dynamics outputs.
        layers.append(
            NormedLinear(
                dimensions[-2],
                dimensions[-1],
                activation=output_activation,
            )
        )
    return nn.Sequential(*layers)


def symlog(value: torch.Tensor) -> torch.Tensor:
    return torch.sign(value) * torch.log1p(torch.abs(value))


def symexp(value: torch.Tensor) -> torch.Tensor:
    return torch.sign(value) * torch.expm1(torch.abs(value))


def two_hot(
    value: torch.Tensor, num_bins: int, value_min: float, value_max: float
) -> torch.Tensor:
    """Encode scalar targets using TD-MPC2's symlog two-hot representation."""

    if num_bins == 0:
        return value
    if num_bins == 1:
        return symlog(value)
    value = torch.clamp(symlog(value), value_min, value_max).squeeze(-1)
    bin_size = (value_max - value_min) / (num_bins - 1)
    location = (value - value_min) / bin_size
    lower = torch.floor(location).long().clamp(0, num_bins - 1)
    upper = (lower + 1).clamp(max=num_bins - 1)
    upper_weight = (location - lower.to(location.dtype)).unsqueeze(-1)
    lower_weight = 1.0 - upper_weight
    target = torch.zeros(
        *value.shape, num_bins, device=value.device, dtype=value.dtype
    )
    target.scatter_add_(-1, lower.unsqueeze(-1), lower_weight)
    target.scatter_add_(-1, upper.unsqueeze(-1), upper_weight)
    return target


def two_hot_inv(
    logits: torch.Tensor, num_bins: int, value_min: float, value_max: float
) -> torch.Tensor:
    if num_bins == 0:
        return logits
    if num_bins == 1:
        return symexp(logits)
    bins = torch.linspace(
        value_min, value_max, num_bins, device=logits.device, dtype=logits.dtype
    )
    value = (F.softmax(logits, dim=-1) * bins).sum(dim=-1, keepdim=True)
    return symexp(value)


def soft_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_bins: int,
    value_min: float,
    value_max: float,
) -> torch.Tensor:
    encoded = two_hot(target, num_bins, value_min, value_max)
    if num_bins <= 1:
        return F.mse_loss(logits, encoded, reduction="none")
    return -(encoded * F.log_softmax(logits, dim=-1)).sum(dim=-1, keepdim=True)


class WorldModel(nn.Module):
    """Single-task TD-MPC2 world model and target critic ensemble."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        config: Dict,
    ) -> None:
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.latent_dim = int(config["latent_dim"])
        self.num_q = int(config["num_q"])
        self.num_bins = int(config["num_bins"])
        self.value_min = float(config["vmin"])
        self.value_max = float(config["vmax"])
        self.safety_dim = int(config["safety_dim"])
        if self.safety_dim != 2:
            raise ValueError(
                f"safety_dim must be exactly 2, got {self.safety_dim}"
            )
        try:
            self.safety_cost_names = tuple(config["safety_cost_names"])
        except TypeError as exc:
            raise TypeError("safety_cost_names must be an iterable") from exc
        if len(self.safety_cost_names) != self.safety_dim:
            raise ValueError(
                "safety_cost_names must contain exactly "
                f"{self.safety_dim} entries, got {len(self.safety_cost_names)}"
            )

        configured_curvature_scale = float(
            config["safety_curvature_scale_mm_inv"]
        )
        configured_translation_error_scale = float(
            config["safety_translation_error_scale"]
        )
        if (
            not math.isfinite(configured_curvature_scale)
            or configured_curvature_scale <= 0.0
        ):
            raise ValueError(
                "safety_curvature_scale_mm_inv must be finite and positive"
            )
        if (
            not math.isfinite(configured_translation_error_scale)
            or configured_translation_error_scale <= 0.0
        ):
            raise ValueError(
                "safety_translation_error_scale must be finite and positive"
            )
        encoded_scales = torch.tensor(
            [
                configured_curvature_scale,
                configured_translation_error_scale,
            ],
            dtype=torch.float32,
        )
        if not torch.isfinite(encoded_scales).all() or torch.any(
            encoded_scales <= 0.0
        ):
            raise ValueError(
                "Safety transform scales must remain finite and positive when "
                "represented as float32 replay data"
            )
        curvature_scale, translation_error_scale = (
            float(value) for value in encoded_scales
        )
        self.safety_curvature_scale_mm_inv = curvature_scale
        self.safety_translation_error_scale = translation_error_scale

        simnorm_dim = int(config["simnorm_dim"])
        enc_dim = int(config["enc_dim"])
        mlp_dim = int(config["mlp_dim"])
        dropout = float(config.get("dropout", 0.0))
        if self.latent_dim % simnorm_dim:
            raise ValueError("model.latent_dim must be divisible by model.simnorm_dim")

        self.encoder = mlp(
            self.observation_dim,
            [enc_dim] * max(int(config.get("num_enc_layers", 2)) - 1, 1),
            self.latent_dim,
            output_activation=SimNorm(simnorm_dim),
        )
        self.dynamics = mlp(
            self.latent_dim + self.action_dim,
            [mlp_dim, mlp_dim],
            self.latent_dim,
            output_activation=SimNorm(simnorm_dim),
        )
        self.reward_head = mlp(
            self.latent_dim + self.action_dim,
            [mlp_dim, mlp_dim],
            max(self.num_bins, 1),
        )
        self.termination_head = mlp(
            self.latent_dim, [mlp_dim, mlp_dim], 1
        )
        self.policy = mlp(
            self.latent_dim, [mlp_dim, mlp_dim], 2 * self.action_dim
        )
        self.q_ensemble = nn.ModuleList(
            [
                mlp(
                    self.latent_dim + self.action_dim,
                    [mlp_dim, mlp_dim],
                    max(self.num_bins, 1),
                    dropout=dropout,
                )
                for _ in range(self.num_q)
            ]
        )
        self.apply(weight_init)
        nn.init.zeros_(self.reward_head[-1].weight)
        nn.init.zeros_(self.reward_head[-1].bias)
        for critic in self.q_ensemble:
            nn.init.zeros_(critic[-1].weight)
            nn.init.zeros_(critic[-1].bias)

        self.target_q_ensemble = deepcopy(self.q_ensemble)
        for parameter in self.target_q_ensemble.parameters():
            parameter.requires_grad_(False)

        self.register_buffer("log_std_min", torch.tensor(float(config["log_std_min"])))
        self.register_buffer(
            "log_std_range",
            torch.tensor(float(config["log_std_max"]) - float(config["log_std_min"])),
        )

        self.register_buffer(
            "safety_scales",
            encoded_scales,
        )
        # Register and initialize safety modules only after the original
        # TD-MPC2 model. This preserves the initialization RNG sequence of all
        # shared encoder/dynamics/reward/policy/Q parameters.
        self.safety_trunk = mlp(
            self.latent_dim + self.action_dim,
            [mlp_dim],
            mlp_dim,
            output_activation=nn.Mish(),
        )
        self.safety_curvature_head = nn.Linear(mlp_dim, 1)
        self.safety_translation_error_head = nn.Linear(mlp_dim, 1)
        self.safety_trunk.apply(weight_init)
        self.safety_curvature_head.apply(weight_init)
        self.safety_translation_error_head.apply(weight_init)

    def train(self, mode: bool = True):
        """Keep target critics in evaluation mode, as in the official code."""

        super().train(mode)
        self.target_q_ensemble.train(False)
        return self

    def encode(self, observation: torch.Tensor) -> torch.Tensor:
        return self.encoder(observation)

    def next(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.dynamics(torch.cat([latent, action], dim=-1))

    def reward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.reward_head(torch.cat([latent, action], dim=-1))

    def safety_head_parameters(self) -> List[nn.Parameter]:
        """Return all trainable parameters owned by the two-channel safety head."""

        return [
            *self.safety_trunk.parameters(),
            *self.safety_curvature_head.parameters(),
            *self.safety_translation_error_head.parameters(),
        ]

    def safety_transformed(
        self, latent: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Predict raw immediate safety costs in log-transformed space."""

        features = self.safety_trunk(torch.cat([latent, action], dim=-1))
        curvature = self.safety_curvature_head(features)
        translation_error = self.safety_translation_error_head(features)
        return torch.cat([curvature, translation_error], dim=-1)

    def translation_safety_transformed(
        self, latent: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Predict only normalized requested/applied Translation Safety.

        This planning-facing path deliberately does not execute the Curvature
        output branch.  The shared Safety trunk remains identical to training,
        while Curvature stays monitoring-only and cannot affect active MPPI.
        """

        features = self.safety_trunk(torch.cat([latent, action], dim=-1))
        return self.safety_translation_error_head(features)

    def transform_safety_targets(self, safety_cost: torch.Tensor) -> torch.Tensor:
        """Map nonnegative physical safety targets into transformed space."""

        self._validate_safety_tensor(safety_cost, "safety_cost")
        scales = self.safety_scales.to(
            device=safety_cost.device, dtype=safety_cost.dtype
        )
        # log(1 + cost / scale) expressed in log-space avoids overflowing the
        # intermediate division for very large but still finite targets.
        log_ratio = torch.log(safety_cost) - torch.log(scales)
        return torch.logaddexp(torch.zeros_like(log_ratio), log_ratio)

    def decode_safety_transformed(
        self, transformed: torch.Tensor
    ) -> torch.Tensor:
        """Decode transformed predictions without overflowing their dtype."""

        self._validate_safety_tensor(transformed, "transformed safety")
        finfo = torch.finfo(transformed.dtype)
        upper_values = [
            self._safe_expm1_upper(
                transformed.dtype,
                self.safety_curvature_scale_mm_inv,
            ),
            self._safe_expm1_upper(
                transformed.dtype,
                self.safety_translation_error_scale,
            ),
        ]
        upper = transformed.new_tensor(upper_values)
        nonnegative = torch.clamp(transformed, min=0.0)
        guarded = torch.minimum(nonnegative, upper)
        scales = self.safety_scales.to(
            device=transformed.device, dtype=transformed.dtype
        )
        decoded = scales * torch.expm1(guarded)
        return torch.nan_to_num(
            decoded,
            nan=0.0,
            posinf=finfo.max,
            neginf=0.0,
        ).clamp(min=0.0, max=finfo.max)

    def decode_translation_safety_transformed(
        self, transformed: torch.Tensor
    ) -> torch.Tensor:
        """Decode the one-channel Translation Safety prediction."""

        if not transformed.is_floating_point():
            raise TypeError(
                "transformed Translation Safety must use a floating-point dtype"
            )
        if transformed.ndim == 0 or transformed.shape[-1] != 1:
            raise ValueError(
                "transformed Translation Safety must have final dimension 1, "
                f"got shape {tuple(transformed.shape)}"
            )
        finfo = torch.finfo(transformed.dtype)
        upper = self._safe_expm1_upper(
            transformed.dtype,
            self.safety_translation_error_scale,
        )
        guarded = torch.clamp(transformed, min=0.0, max=upper)
        scale = transformed.new_tensor(self.safety_translation_error_scale)
        decoded = scale * torch.expm1(guarded)
        return torch.nan_to_num(
            decoded,
            nan=0.0,
            posinf=finfo.max,
            neginf=0.0,
        ).clamp(min=0.0, max=finfo.max)

    def safety(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict finite, nonnegative immediate safety costs."""

        return self.decode_safety_transformed(
            self.safety_transformed(latent, action)
        )

    def _validate_safety_tensor(
        self, value: torch.Tensor, name: str
    ) -> None:
        if not value.is_floating_point():
            raise TypeError(f"{name} must use a floating-point dtype")
        if value.ndim == 0 or value.shape[-1] != self.safety_dim:
            raise ValueError(
                f"{name} must have final dimension {self.safety_dim}, "
                f"got shape {tuple(value.shape)}"
            )

    @staticmethod
    def _safe_expm1_upper(dtype: torch.dtype, scale: float) -> float:
        """Return a conservative pre-expm1 bound for one output channel."""

        finfo = torch.finfo(dtype)
        log_dtype_max = math.log(finfo.max)
        if scale <= 1.0:
            upper = log_dtype_max
        else:
            log_ratio = log_dtype_max - math.log(scale)
            upper = (
                log_ratio
                if log_ratio > 50.0
                else math.log1p(math.exp(log_ratio))
            )
            upper = min(log_dtype_max, upper)
        margin = 2.0 * finfo.eps * max(1.0, abs(upper))
        return max(0.0, upper - margin)

    def termination(
        self, latent: torch.Tensor, *, logits: bool = False
    ) -> torch.Tensor:
        value = self.termination_head(latent)
        return value if logits else torch.sigmoid(value)

    def pi(
        self, latent: torch.Tensor, *, deterministic: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        mean, raw_log_std = self.policy(latent).chunk(2, dim=-1)
        log_std = self.log_std_min + 0.5 * self.log_std_range * (
            torch.tanh(raw_log_std) + 1.0
        )
        noise = torch.zeros_like(mean) if deterministic else torch.randn_like(mean)
        pre_tanh = mean + noise * log_std.exp()
        action = torch.tanh(pre_tanh)
        squashed_mean = torch.tanh(mean)

        residual = -0.5 * noise.pow(2) - log_std - 0.9189385332046727
        gaussian_log_prob = residual.sum(dim=-1, keepdim=True)
        scaled_log_prob = gaussian_log_prob * self.action_dim
        log_prob = gaussian_log_prob
        log_prob = log_prob - torch.log(
            torch.relu(1.0 - action.pow(2)) + 1e-6
        ).sum(dim=-1, keepdim=True)
        entropy = -log_prob
        entropy_scale = scaled_log_prob / (log_prob + 1e-8)
        return action, {
            "mean": squashed_mean,
            "log_std": log_std,
            "entropy": entropy,
            "scaled_entropy": entropy * entropy_scale,
        }

    def q_values(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        *,
        target: bool = False,
    ) -> torch.Tensor:
        features = torch.cat([latent, action], dim=-1)
        ensemble = self.target_q_ensemble if target else self.q_ensemble
        return torch.stack([critic(features) for critic in ensemble], dim=0)

    def q(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        *,
        reduction: str = "min",
        target: bool = False,
    ) -> torch.Tensor:
        all_logits = self.q_values(latent, action, target=target)
        if reduction == "all":
            return all_logits
        indices = torch.randperm(self.num_q, device=latent.device)[
            : min(2, self.num_q)
        ]
        values = two_hot_inv(
            all_logits[indices], self.num_bins, self.value_min, self.value_max
        )
        if reduction == "min":
            return values.min(dim=0).values
        if reduction == "avg":
            return values.mean(dim=0)
        raise ValueError(f"Unknown Q reduction {reduction!r}")

    @torch.no_grad()
    def soft_update_target_q(self, tau: float) -> None:
        for target_parameter, parameter in zip(
            self.target_q_ensemble.parameters(), self.q_ensemble.parameters()
        ):
            target_parameter.lerp_(parameter, float(tau))
