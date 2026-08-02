"""Episode-level Integral Lagrangian control for evaluation-only Safety-MPC."""

from __future__ import annotations

import copy
import math
from numbers import Integral, Real
from typing import Any, Dict, Mapping


DEFAULT_INTEGRAL_LAGRANGIAN_CONFIG = {
    "enabled": False,
    "initial_alpha": 0.1,
    "integral_gain": 0.1,
    "cost_limit": 0.01,
    "alpha_max": 0.5,
    "rolling_window_episodes": 5,
    "warmup_episodes": 5,
}


def validate_integral_lagrangian_config(
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return a strict, JSON-safe Integral Lagrangian configuration."""

    if not isinstance(config, Mapping):
        raise TypeError("integral_lagrangian must be a mapping")
    expected_keys = set(DEFAULT_INTEGRAL_LAGRANGIAN_CONFIG)
    missing_keys = sorted(expected_keys - set(config))
    if missing_keys:
        raise KeyError(
            "integral_lagrangian is missing keys: " f"{missing_keys}"
        )
    unexpected_keys = sorted(set(config) - expected_keys)
    if unexpected_keys:
        raise ValueError(
            "integral_lagrangian has unexpected keys: "
            f"{unexpected_keys}"
        )

    enabled = config["enabled"]
    if type(enabled) is not bool:
        raise TypeError("integral_lagrangian.enabled must be a bool")

    def finite_float(key: str, *, strictly_positive: bool) -> float:
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(
                f"integral_lagrangian.{key} must be a real number"
            )
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError(f"integral_lagrangian.{key} must be finite")
        if strictly_positive and converted <= 0.0:
            raise ValueError(
                f"integral_lagrangian.{key} must be strictly positive"
            )
        if not strictly_positive and converted < 0.0:
            raise ValueError(
                f"integral_lagrangian.{key} must be nonnegative"
            )
        return converted

    def positive_integer(key: str) -> int:
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(
                f"integral_lagrangian.{key} must be an integer"
            )
        converted = int(value)
        if converted < 1:
            raise ValueError(
                f"integral_lagrangian.{key} must be at least 1"
            )
        return converted

    resolved = {
        "enabled": enabled,
        "initial_alpha": finite_float(
            "initial_alpha", strictly_positive=False
        ),
        "integral_gain": finite_float(
            "integral_gain", strictly_positive=False
        ),
        "cost_limit": finite_float("cost_limit", strictly_positive=False),
        "alpha_max": finite_float("alpha_max", strictly_positive=True),
        "rolling_window_episodes": positive_integer(
            "rolling_window_episodes"
        ),
        "warmup_episodes": positive_integer("warmup_episodes"),
    }
    if resolved["initial_alpha"] > resolved["alpha_max"]:
        raise ValueError(
            "integral_lagrangian.initial_alpha must be no greater than "
            "integral_lagrangian.alpha_max"
        )
    return resolved


class IntegralLagrangianController:
    """Adapt one Safety-MPC alpha after each completed evaluation episode.

    ``alpha_trajectory`` contains the initial alpha followed by the alpha after
    every observed episode. Thus an evaluation with N episodes has N+1 alpha
    states, while each episode record identifies the value used for that
    episode through ``alpha_before_update``.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        self._config = validate_integral_lagrangian_config(config)
        if not self._config["enabled"]:
            raise ValueError(
                "IntegralLagrangianController requires enabled=true"
            )
        self.reset()

    @property
    def config(self) -> Dict[str, Any]:
        """Return a copy so validated controller settings stay immutable."""

        return copy.deepcopy(self._config)

    @property
    def current_alpha(self) -> float:
        return self._current_alpha

    def reset(self) -> None:
        """Restore initial alpha and clear all episode history."""

        self._current_alpha = float(self._config["initial_alpha"])
        self._recent_episode_costs: list[float] = []
        self._observed_cost_trajectory: list[float] = []
        self._rolling_cost_trajectory: list[float] = []
        self._alpha_trajectory: list[float] = [self._current_alpha]
        self._episode_alpha_trajectory: list[float] = []
        self._dual_update_count = 0
        self._lower_clip_count = 0
        self._upper_clip_count = 0

    def observe_episode_cost(self, cost: Real) -> Dict[str, Any]:
        """Observe one completed episode and update alpha for the next one."""

        if isinstance(cost, bool) or not isinstance(cost, Real):
            raise TypeError("Episode safety cost must be a real number")
        episode_cost = float(cost)
        if not math.isfinite(episode_cost):
            raise ValueError("Episode safety cost must be finite")
        if episode_cost < 0.0 or episode_cost > 1.0:
            raise ValueError(
                "Episode tree-end blockage fraction must be in [0, 1]"
            )

        alpha_before = self._current_alpha
        prospective_costs = [*self._recent_episode_costs, episode_cost]
        window_size = int(self._config["rolling_window_episodes"])
        recent_costs = prospective_costs[-window_size:]
        rolling_mean = math.fsum(recent_costs) / len(recent_costs)
        if not math.isfinite(rolling_mean):
            raise FloatingPointError("Rolling episode safety cost is non-finite")
        dual_error = rolling_mean - float(self._config["cost_limit"])

        episode_number = len(self._observed_cost_trajectory) + 1
        warmup_active = episode_number <= int(self._config["warmup_episodes"])
        lower_clip = False
        upper_clip = False
        dual_update_applied = not warmup_active
        alpha_after = alpha_before
        if dual_update_applied:
            requested_delta = float(self._config["integral_gain"]) * dual_error
            raw_alpha = alpha_before + requested_delta
            if not math.isfinite(requested_delta) or not math.isfinite(raw_alpha):
                raise FloatingPointError(
                    "Integral Lagrangian alpha update is non-finite"
                )
            lower_clip = raw_alpha < 0.0
            upper_clip = raw_alpha > float(self._config["alpha_max"])
            alpha_after = min(
                max(raw_alpha, 0.0),
                float(self._config["alpha_max"]),
            )
            if not math.isfinite(alpha_after):
                raise FloatingPointError(
                    "Clipped Integral Lagrangian alpha is non-finite"
                )

        # Commit only after every derived value has passed validation.
        self._recent_episode_costs = recent_costs
        self._observed_cost_trajectory.append(episode_cost)
        self._rolling_cost_trajectory.append(rolling_mean)
        self._episode_alpha_trajectory.append(alpha_before)
        self._current_alpha = alpha_after
        self._alpha_trajectory.append(alpha_after)
        if dual_update_applied:
            self._dual_update_count += 1
            self._lower_clip_count += int(lower_clip)
            self._upper_clip_count += int(upper_clip)

        return {
            "episode_number": episode_number,
            "episode_tree_end_blockage_fraction": episode_cost,
            "rolling_mean_cost": rolling_mean,
            "cost_limit": float(self._config["cost_limit"]),
            "dual_error": dual_error,
            "alpha_before_update": alpha_before,
            "alpha_after_update": alpha_after,
            "alpha_update_amount": alpha_after - alpha_before,
            "lower_clipping_active": lower_clip,
            "upper_clipping_active": upper_clip,
            "warmup_active": warmup_active,
            "dual_update_applied": dual_update_applied,
        }

    def state_dict(self) -> Dict[str, Any]:
        """Return evaluation-controller state without model dependencies."""

        return {
            "schema_version": 1,
            "config": copy.deepcopy(self._config),
            "current_alpha": self._current_alpha,
            "recent_episode_costs": list(self._recent_episode_costs),
            "observed_cost_trajectory": list(
                self._observed_cost_trajectory
            ),
            "rolling_cost_trajectory": list(self._rolling_cost_trajectory),
            "alpha_trajectory": list(self._alpha_trajectory),
            "episode_alpha_trajectory": list(
                self._episode_alpha_trajectory
            ),
            "dual_update_count": self._dual_update_count,
            "lower_clip_count": self._lower_clip_count,
            "upper_clip_count": self._upper_clip_count,
        }

    def summary(self) -> Dict[str, Any]:
        """Return the complete finite run-level trajectory and statistics."""

        alpha_values = self._alpha_trajectory
        episode_alpha_values = (
            self._episode_alpha_trajectory
            if self._episode_alpha_trajectory
            else [self._current_alpha]
        )
        alpha_mean = math.fsum(alpha_values) / len(alpha_values)
        episode_alpha_mean = math.fsum(episode_alpha_values) / len(
            episode_alpha_values
        )
        return {
            "enabled": True,
            "initial_alpha": float(self._config["initial_alpha"]),
            "final_alpha": self._current_alpha,
            "minimum_alpha": min(alpha_values),
            "maximum_alpha": max(alpha_values),
            "alpha_mean": alpha_mean,
            "alpha_statistics_basis": "controller_state_trajectory",
            "episode_applied_minimum_alpha": min(episode_alpha_values),
            "episode_applied_maximum_alpha": max(episode_alpha_values),
            "episode_applied_alpha_mean": episode_alpha_mean,
            "dual_update_count": self._dual_update_count,
            "lower_clip_count": self._lower_clip_count,
            "upper_clip_count": self._upper_clip_count,
            "alpha_trajectory": list(self._alpha_trajectory),
            "episode_alpha_trajectory": list(
                self._episode_alpha_trajectory
            ),
            "observed_cost_trajectory": list(
                self._observed_cost_trajectory
            ),
            "rolling_cost_trajectory": list(self._rolling_cost_trajectory),
        }
