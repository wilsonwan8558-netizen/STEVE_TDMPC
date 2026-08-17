"""Evaluation-only step-level Lagrangian control for Translation Safety-MPC."""

from __future__ import annotations

import copy
import math
import struct
from numbers import Real
from typing import Any, Dict, Mapping


DEFAULT_ADAPTIVE_LAGRANGIAN_CONFIG = {
    "epsilon": 0.005,
    "eta": 0.2,
    "lambda_initial": 0.1,
    "lambda_min": 0.0,
    "lambda_max": 0.2,
}


def validate_adaptive_lagrangian_config(
    config: Mapping[str, Any],
) -> Dict[str, float]:
    """Return a strict finite configuration for the evaluation controller."""

    if not isinstance(config, Mapping):
        raise TypeError("Adaptive Lagrangian configuration must be a mapping")
    expected = set(DEFAULT_ADAPTIVE_LAGRANGIAN_CONFIG)
    missing = sorted(expected - set(config))
    if missing:
        raise KeyError(f"Adaptive Lagrangian configuration is missing keys: {missing}")
    unexpected = sorted(set(config) - expected)
    if unexpected:
        raise ValueError(
            "Adaptive Lagrangian configuration has unexpected keys: "
            f"{unexpected}"
        )

    resolved: Dict[str, float] = {}
    for key in DEFAULT_ADAPTIVE_LAGRANGIAN_CONFIG:
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"Adaptive Lagrangian {key} must be a real number")
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError(f"Adaptive Lagrangian {key} must be finite")
        if converted < 0.0:
            raise ValueError(f"Adaptive Lagrangian {key} must be nonnegative")
        resolved[key] = converted

    if resolved["lambda_max"] <= resolved["lambda_min"]:
        raise ValueError(
            "Adaptive Lagrangian lambda_max must be greater than lambda_min"
        )
    if not (
        resolved["lambda_min"]
        <= resolved["lambda_initial"]
        <= resolved["lambda_max"]
    ):
        raise ValueError(
            "Adaptive Lagrangian lambda_initial must lie within "
            "[lambda_min, lambda_max]"
        )
    represented: Dict[str, float] = {}
    for key in ("lambda_initial", "lambda_min", "lambda_max"):
        value = resolved[key]
        try:
            float32_value = struct.unpack("f", struct.pack("f", value))[0]
        except OverflowError as exc:
            raise ValueError(
                f"Adaptive Lagrangian {key} is not representable as float32"
            ) from exc
        if (
            not math.isfinite(float32_value)
            or (value > 0.0 and float32_value == 0.0)
        ):
            raise ValueError(
                f"Adaptive Lagrangian {key} must remain finite and nonzero "
                "when its positive value is represented as float32"
            )
        represented[key] = float32_value
    if represented["lambda_max"] <= represented["lambda_min"]:
        raise ValueError(
            "Adaptive Lagrangian lambda bounds must remain distinct in float32"
        )
    if not (
        represented["lambda_min"]
        <= represented["lambda_initial"]
        <= represented["lambda_max"]
    ):
        raise ValueError(
            "Adaptive Lagrangian lambda_initial must remain inside the bounds "
            "when represented as float32"
        )
    return resolved


class AdaptiveLagrangianController:
    """Adapt a nonnegative planner multiplier from predicted selected risk.

    The controller has no PyTorch, model, optimizer, or checkpoint dependency.
    ``reset_episode`` is mandatory at each evaluation episode boundary. The
    lambda applied at step ``t`` is updated only after observing that step's
    selected predicted Translation trajectory risk, so the result is used by
    planning at step ``t+1``.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        self._config = validate_adaptive_lagrangian_config(config)
        self.reset_episode()

    @property
    def config(self) -> Dict[str, float]:
        return copy.deepcopy(self._config)

    @property
    def current_lambda(self) -> float:
        return self._current_lambda

    def reset_episode(self) -> None:
        """Restore lambda_initial and clear every per-step trajectory."""

        self._current_lambda = float(self._config["lambda_initial"])
        self._lambda_state_trajectory = [self._current_lambda]
        self._applied_lambda_trajectory: list[float] = []
        self._risk_trajectory: list[float] = []
        self._residual_trajectory: list[float] = []
        self._update_trajectory: list[float] = []
        self._lower_clip_count = 0
        self._upper_clip_count = 0

    def observe_step_risk(self, risk: Real) -> Dict[str, Any]:
        """Update lambda once after one executed environment transition."""

        if isinstance(risk, bool) or not isinstance(risk, Real):
            raise TypeError("Predicted Translation risk must be a real number")
        predicted_risk = float(risk)
        if not math.isfinite(predicted_risk):
            raise ValueError("Predicted Translation risk must be finite")
        if predicted_risk < 0.0:
            raise ValueError("Predicted Translation risk must be nonnegative")

        lambda_before = self._current_lambda
        residual = predicted_risk - float(self._config["epsilon"])
        requested_update = float(self._config["eta"]) * residual
        raw_lambda = lambda_before + requested_update
        if not math.isfinite(residual) or not math.isfinite(raw_lambda):
            raise FloatingPointError("Adaptive Lagrangian update is non-finite")
        lower_clipping = raw_lambda < float(self._config["lambda_min"])
        upper_clipping = raw_lambda > float(self._config["lambda_max"])
        lambda_after = min(
            max(raw_lambda, float(self._config["lambda_min"])),
            float(self._config["lambda_max"]),
        )
        if not math.isfinite(lambda_after) or lambda_after < 0.0:
            raise FloatingPointError(
                "Adaptive Lagrangian clipped lambda is invalid"
            )

        self._applied_lambda_trajectory.append(lambda_before)
        self._risk_trajectory.append(predicted_risk)
        self._residual_trajectory.append(residual)
        self._update_trajectory.append(lambda_after - lambda_before)
        self._current_lambda = lambda_after
        self._lambda_state_trajectory.append(lambda_after)
        self._lower_clip_count += int(lower_clipping)
        self._upper_clip_count += int(upper_clipping)

        return {
            "step_number": len(self._risk_trajectory),
            "predicted_translation_risk": predicted_risk,
            "risk_budget_epsilon": float(self._config["epsilon"]),
            "constraint_residual": residual,
            "lambda_before_update": lambda_before,
            "lambda_after_update": lambda_after,
            "lambda_update_amount": lambda_after - lambda_before,
            "requested_lambda_update": requested_update,
            "lower_clipping_active": lower_clipping,
            "upper_clipping_active": upper_clipping,
        }

    def state_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe snapshot of the current episode state."""

        return {
            "schema_version": 1,
            "config": copy.deepcopy(self._config),
            "current_lambda": self._current_lambda,
            "lambda_state_trajectory": list(self._lambda_state_trajectory),
            "applied_lambda_trajectory": list(
                self._applied_lambda_trajectory
            ),
            "predicted_risk_trajectory": list(self._risk_trajectory),
            "constraint_residual_trajectory": list(
                self._residual_trajectory
            ),
            "lambda_update_trajectory": list(self._update_trajectory),
            "lower_clip_count": self._lower_clip_count,
            "upper_clip_count": self._upper_clip_count,
        }

    def summary(self) -> Dict[str, Any]:
        """Return complete trajectories and finite per-episode statistics."""

        applied = self._applied_lambda_trajectory
        risks = self._risk_trajectory
        residuals = self._residual_trajectory
        if not applied:
            applied = [self._current_lambda]
        tolerance = max(
            1.0e-9,
            1.0e-6
            * (float(self._config["lambda_max"]) - float(self._config["lambda_min"])),
        )
        near_min = sum(
            value <= float(self._config["lambda_min"]) + tolerance
            for value in applied
        )
        near_max = sum(
            value >= float(self._config["lambda_max"]) - tolerance
            for value in applied
        )
        return {
            "lambda_initial": float(self._config["lambda_initial"]),
            "lambda_final": self._current_lambda,
            "applied_lambda_mean": math.fsum(applied) / len(applied),
            "applied_lambda_min": min(applied),
            "applied_lambda_max": max(applied),
            "lambda_bound_tolerance": tolerance,
            "fraction_steps_near_lambda_min": near_min / len(applied),
            "fraction_steps_near_lambda_max": near_max / len(applied),
            "predicted_risk_mean": (
                math.fsum(risks) / len(risks) if risks else None
            ),
            "constraint_residual_mean": (
                math.fsum(residuals) / len(residuals) if residuals else None
            ),
            "step_update_count": len(self._risk_trajectory),
            "lower_clip_count": self._lower_clip_count,
            "upper_clip_count": self._upper_clip_count,
            "lambda_state_trajectory": list(self._lambda_state_trajectory),
            "applied_lambda_trajectory": list(
                self._applied_lambda_trajectory
            ),
            "predicted_risk_trajectory": list(self._risk_trajectory),
            "constraint_residual_trajectory": list(
                self._residual_trajectory
            ),
            "lambda_update_trajectory": list(self._update_trajectory),
        }
