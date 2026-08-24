"""Lazy Gymnasium adapter for the NVIDIA guided catheter environment.

The NVIDIA workflow is a separate local codebase with GPU-only dependencies.
Nothing from that workflow is imported until :class:`NvidiaGuidedEnv` is
constructed, so importing the stEVE backend does not require Warp or the
NVIDIA catheter packages.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import gymnasium as gym
import numpy as np

from safety_schema import NVIDIA_GUIDED_SAFETY_COST_NAMES

_WORKFLOW_ENV_VAR = "I4H_WORKFLOW_ROOT"
_CT_CACHE_ENV_VAR = "I4H_CT_CACHE"
_NVIDIA_ENV_MODULE = "workflows.catheter_navigation.nvidia_catheter_env"
_NVIDIA_ENV_RELATIVE_PATH = Path(
    "workflows/catheter_navigation/nvidia_catheter_env.py"
)
_REQUIRED_CT_FILES = (
    "mu_volume.npy",
    "metadata.json",
    "centerline_points_mm.npy",
    "vessel_mask.npy",
)


def _configured_path(
    value: str | Path | None,
    *,
    environment_variable: str,
    description: str,
) -> Path:
    """Resolve one required path from explicit config or an environment variable."""

    raw_value: str | Path | None = value
    if raw_value is None:
        raw_value = os.environ.get(environment_variable)
    if raw_value is None or not str(raw_value).strip():
        raise ValueError(
            f"{description} is required for the NVIDIA guided backend. "
            f"Set it in the environment config or via {environment_variable}."
        )
    path = Path(raw_value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{description} is not a directory: {path}")
    return path


def _load_nvidia_environment_class(workflow_root: Path):
    """Import the external environment only after NVIDIA was selected."""

    expected_module_path = (workflow_root / _NVIDIA_ENV_RELATIVE_PATH).resolve()
    if not expected_module_path.is_file():
        raise FileNotFoundError(
            "NVIDIA catheter environment module was not found at "
            f"{expected_module_path}. Check workflow_root/{_WORKFLOW_ENV_VAR}."
        )

    workflow_root_text = str(workflow_root)
    if workflow_root_text not in sys.path:
        sys.path.insert(0, workflow_root_text)

    try:
        module = importlib.import_module(_NVIDIA_ENV_MODULE)
    except ImportError as exc:
        raise ImportError(
            "Failed to import the NVIDIA guided catheter environment from "
            f"{expected_module_path}. Activate the i4h_workflow environment "
            "and verify Gymnasium, Torch, Warp, fluorosim, "
            "catheter_vasculature_solver, and vasculature_digital_twin."
        ) from exc

    loaded_module_path = Path(module.__file__).resolve()
    if loaded_module_path != expected_module_path:
        raise ImportError(
            "The loaded NVIDIA catheter environment came from an unexpected "
            f"location: expected {expected_module_path}, got {loaded_module_path}."
        )
    try:
        return module.NvidiaCatheterEnv
    except AttributeError as exc:
        raise ImportError(
            f"{expected_module_path} does not define NvidiaCatheterEnv"
        ) from exc


class NvidiaGuidedEnv(gym.Env[np.ndarray, np.ndarray]):
    """Validated adapter around the external NVIDIA guided environment."""

    metadata = {"render_modes": [None, "rgb_array"], "render_fps": 30}
    safety_cost_names = NVIDIA_GUIDED_SAFETY_COST_NAMES

    def __init__(
        self,
        *,
        workflow_root: str | Path | None = None,
        ct_cache_path: str | Path | None = None,
        render_mode: str | None = None,
        max_episode_steps: int = 1200,
        max_translation_velocity_m_s: float = 0.005,
        max_rotation_velocity_rad_s: float = 1.0,
        terminate_on_hard_violation: bool = True,
        safety_config: Any = None,
    ) -> None:
        super().__init__()
        if safety_config is not None:
            raise ValueError(
                "NVIDIA safety_config overrides are intentionally unsupported "
                "in the first guided integration; omit this field to preserve "
                "the external environment's validated normalization constants"
            )
        self.workflow_root = _configured_path(
            workflow_root,
            environment_variable=_WORKFLOW_ENV_VAR,
            description="NVIDIA workflow root",
        )
        self.ct_cache_path = _configured_path(
            ct_cache_path,
            environment_variable=_CT_CACHE_ENV_VAR,
            description="NVIDIA CT cache",
        )
        missing_ct_files = [
            name
            for name in _REQUIRED_CT_FILES
            if not (self.ct_cache_path / name).is_file()
        ]
        if missing_ct_files:
            raise FileNotFoundError(
                f"NVIDIA CT cache {self.ct_cache_path} is missing required guided "
                f"assets: {missing_ct_files}"
            )

        environment_class = _load_nvidia_environment_class(self.workflow_root)
        try:
            self._env = environment_class(
                ct_dir=self.ct_cache_path,
                render_mode=render_mode,
                max_episode_steps=max_episode_steps,
                max_translation_velocity_m_s=max_translation_velocity_m_s,
                max_rotation_velocity_rad_s=max_rotation_velocity_rad_s,
                terminate_on_hard_violation=terminate_on_hard_violation,
                safety_config=safety_config,
            )
        except ImportError as exc:
            raise ImportError(
                "Failed to construct the NVIDIA guided catheter environment. "
                "Activate the i4h_workflow environment and verify its GPU "
                "simulation dependencies."
            ) from exc

        self.render_mode = render_mode
        self.action_space = self._validate_space(
            self._env.action_space,
            expected_shape=(2,),
            label="action",
        )
        self.observation_space = self._validate_space(
            self._env.observation_space,
            expected_shape=(14,),
            label="observation",
        )
        if not np.allclose(self.action_space.low, -1.0) or not np.allclose(
            self.action_space.high, 1.0
        ):
            raise ValueError(
                "NVIDIA guided actions must be normalized to [-1, 1]; got "
                f"low={self.action_space.low}, high={self.action_space.high}"
            )

    @staticmethod
    def _validate_space(
        space: gym.Space,
        *,
        expected_shape: Tuple[int, ...],
        label: str,
    ) -> gym.spaces.Box:
        if not isinstance(space, gym.spaces.Box):
            raise TypeError(
                f"NVIDIA guided {label} space must be gym.spaces.Box, "
                f"got {type(space).__name__}"
            )
        if space.shape != expected_shape:
            raise ValueError(
                f"NVIDIA guided {label} shape must be {expected_shape}, "
                f"got {space.shape}"
            )
        if space.dtype != np.dtype(np.float32):
            raise TypeError(
                f"NVIDIA guided {label} space must use float32, got {space.dtype}"
            )
        if not np.all(np.isfinite(space.low)) or not np.all(np.isfinite(space.high)):
            raise ValueError(f"NVIDIA guided {label} bounds must be finite")
        return space

    def _validate_observation(self, observation: Any) -> np.ndarray:
        output = np.asarray(observation, dtype=np.float32)
        if output.shape != (14,):
            raise ValueError(
                f"NVIDIA guided observation must have shape (14,), got {output.shape}"
            )
        if not np.all(np.isfinite(output)):
            raise FloatingPointError(
                "NVIDIA guided environment returned a non-finite observation"
            )
        if not self.observation_space.contains(output):
            raise ValueError(
                "NVIDIA guided observation lies outside its declared observation space"
            )
        return output

    def _validate_action(self, action: Any) -> np.ndarray:
        output = np.asarray(action, dtype=np.float32)
        if output.shape != (2,):
            raise ValueError(
                f"NVIDIA guided action must have shape (2,), got {output.shape}"
            )
        if not np.all(np.isfinite(output)):
            raise ValueError("NVIDIA guided action contains NaN or infinity")
        return np.clip(output, -1.0, 1.0).astype(np.float32, copy=False)

    def _adapt_info(self, info: Mapping[str, Any]) -> Dict[str, Any]:
        if not isinstance(info, Mapping):
            raise TypeError(
                "NVIDIA guided environment info must be a mapping, "
                f"got {type(info).__name__}"
            )
        output = dict(info)

        received_names = tuple(output.get("safety_component_order", ()))
        if received_names != self.safety_cost_names:
            raise ValueError(
                "NVIDIA safety-cost channel mismatch: expected "
                f"{self.safety_cost_names}, got {received_names}"
            )

        if "safety_cost_vector" not in output:
            raise KeyError("NVIDIA info is missing 'safety_cost_vector'")
        safety_cost = np.asarray(
            output["safety_cost_vector"],
            dtype=np.float32,
        )
        if safety_cost.shape != (len(self.safety_cost_names),):
            raise ValueError(
                "NVIDIA safety cost must have shape "
                f"({len(self.safety_cost_names)},), got {safety_cost.shape}"
            )
        if not np.all(np.isfinite(safety_cost)):
            raise FloatingPointError("NVIDIA safety cost contains NaN or infinity")
        if np.any(safety_cost < 0.0):
            raise ValueError("NVIDIA safety cost values must be nonnegative")

        if "safety_cost" not in output:
            raise KeyError("NVIDIA info is missing scalar 'safety_cost'")
        safety_cost_scalar = float(output["safety_cost"])
        if not np.isfinite(safety_cost_scalar) or safety_cost_scalar < 0.0:
            raise ValueError(
                "NVIDIA scalar safety cost must be finite and nonnegative"
            )

        if "hard_safety_violation" not in output:
            raise KeyError("NVIDIA info is missing 'hard_safety_violation'")

        # Preserve the workflow's scalar for logging while exposing the exact
        # six-channel vector through the common TD-MPC2 safety-cost key.
        output["safety_cost_scalar"] = safety_cost_scalar
        output["safety_cost"] = safety_cost.copy()
        output["safety_cost_vector"] = safety_cost.copy()
        output["safety_cost_names"] = self.safety_cost_names
        output["hard_safety_violation"] = bool(
            output["hard_safety_violation"]
        )
        output["is_success"] = bool(output.get("success", False))
        return output

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        observation, info = self._env.reset(seed=seed, options=options)
        return self._validate_observation(observation), self._adapt_info(info)

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        normalized_action = self._validate_action(action)
        observation, reward, terminated, truncated, info = self._env.step(
            normalized_action
        )
        finite_reward = float(reward)
        if not np.isfinite(finite_reward):
            raise FloatingPointError(
                "NVIDIA guided environment returned a non-finite reward"
            )
        return (
            self._validate_observation(observation),
            finite_reward,
            bool(terminated),
            bool(truncated),
            self._adapt_info(info),
        )

    def render(self):
        """Pass rendering through to the external environment unchanged."""

        return self._env.render()

    def close(self) -> None:
        self._env.close()


def make_nvidia_guided_env(
    config: Optional[Mapping[str, Any]] = None,
) -> NvidiaGuidedEnv:
    """Construct :class:`NvidiaGuidedEnv` from backend configuration."""

    return NvidiaGuidedEnv(**dict(config or {}))


__all__ = [
    "NVIDIA_GUIDED_SAFETY_COST_NAMES",
    "NvidiaGuidedEnv",
    "make_nvidia_guided_env",
]
