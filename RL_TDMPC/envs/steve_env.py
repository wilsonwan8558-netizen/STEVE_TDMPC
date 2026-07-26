"""Gymnasium adapter for the stEVE endovascular navigation task.

The task construction intentionally mirrors ``examples/function_check.py``.
The only functional change is the start strategy: ``InsertionPoint`` is used
instead of ``MaxDeviceLength(max_length=500)`` because the reference J-shaped
wire is 450 mm long and therefore never satisfies that reset condition.  A
fresh insertion-point reset is required for independent RL episodes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np


# Allow ``python RL_TDMPC/...`` from a source checkout before ``pip install -e .``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import eve  # noqa: E402  (must follow source-checkout path setup)


OBSERVATION_KEYS: Tuple[str, ...] = ("position", "target", "rotation")
_CURVATURE_EPSILON_MM = 1e-8


class StEVEEnv(gym.Env[np.ndarray, np.ndarray]):
    """A fixed-vector, normalized-action Gymnasium view of stEVE.

    Observation layout (14 values for the default task):

    * ``position``: five normalized 2-D tracking points (10 values)
    * ``target``: normalized 2-D target location (2 values)
    * ``rotation``: sine/cosine of the device rotation (2 values)

    The public action is ``[translation, rotation]`` in ``[-1, 1]``.  It is
    linearly mapped to the stEVE device limits, currently 50 mm/s and
    3.14 rad/s for ``JShaped``.
    """

    metadata = {"render_modes": ["human"], "render_fps": 7.5}

    def __init__(
        self,
        *,
        max_episode_steps: int = 200,
        vessel_seed: int = 30,
        image_frequency: float = 7.5,
        target_threshold: float = 5.0,
        target_branches: Sequence[str] = (
            "lcca",
            "rcca",
            "lsa",
            "rsa",
            "bct",
            "co",
        ),
        friction: float = 0.001,
        render_mode: Optional[str] = None,
        display_size: Tuple[int, int] = (600, 860),
    ) -> None:
        super().__init__()
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if render_mode not in (None, "human"):
            raise ValueError(
                f"Unsupported render_mode {render_mode!r}; expected None or 'human'."
            )

        self.max_episode_steps = int(max_episode_steps)
        self.render_mode = render_mode
        self._episode_steps = 0

        vessel_tree = eve.intervention.vesseltree.AorticArch(
            seed=int(vessel_seed),
            scaling_xyzd=[1.0, 1.0, 1.0, 0.75],
        )
        device = eve.intervention.device.JShaped()
        simulation = eve.intervention.simulation.SofaBeamAdapter(
            friction=float(friction)
        )
        fluoroscopy = eve.intervention.fluoroscopy.TrackingOnly(
            simulation=simulation,
            vessel_tree=vessel_tree,
            image_frequency=float(image_frequency),
            image_rot_zx=[20, 5],
        )
        target = eve.intervention.target.CenterlineRandom(
            vessel_tree=vessel_tree,
            fluoroscopy=fluoroscopy,
            threshold=float(target_threshold),
            branches=list(target_branches),
        )
        intervention = eve.intervention.MonoPlaneStatic(
            vessel_tree=vessel_tree,
            devices=[device],
            simulation=simulation,
            fluoroscopy=fluoroscopy,
            target=target,
        )

        # Keep the reference observation construction and flatten it below.
        position = eve.observation.Tracking2D(
            intervention=intervention, n_points=5
        )
        position = eve.observation.wrapper.NormalizeTracking2DEpisode(
            position, intervention
        )
        target_state = eve.observation.Target2D(intervention=intervention)
        target_state = eve.observation.wrapper.NormalizeTracking2DEpisode(
            target_state, intervention
        )
        rotation = eve.observation.Rotations(intervention=intervention)
        observation = eve.observation.ObsDict(
            {"position": position, "target": target_state, "rotation": rotation}
        )

        pathfinder = eve.pathfinder.BruteForceBFS(intervention=intervention)
        reward = eve.reward.Combination(
            [
                eve.reward.TargetReached(intervention=intervention, factor=1.0),
                eve.reward.PathLengthDelta(pathfinder=pathfinder, factor=0.01),
            ]
        )
        terminal = eve.terminal.TargetReached(intervention=intervention)
        truncation = eve.truncation.MaxSteps(self.max_episode_steps)
        info = eve.info.TargetReached(intervention=intervention, name="is_success")

        # Required compatibility change; see module docstring.
        start = eve.start.InsertionPoint(intervention=intervention)
        visualisation = None
        if self.render_mode == "human":
            # SofaPygame enables SOFA visual nodes in its constructor, so it
            # must be created before the first environment reset.
            visualisation = eve.visualisation.SofaPygame(
                intervention=intervention,
                display_size=tuple(int(value) for value in display_size),
            )
        self._env = eve.Env(
            intervention=intervention,
            observation=observation,
            reward=reward,
            terminal=terminal,
            truncation=truncation,
            info=info,
            start=start,
            pathfinder=pathfinder,
            visualisation=visualisation,
        )
        self._intervention = intervention
        self._simulation = simulation
        self._visualisation = visualisation
        self._previous_tip_position: Optional[np.ndarray] = None
        self._previous_inserted_length: Optional[float] = None

        self._raw_action_low = np.asarray(
            self._env.action_space.low, dtype=np.float32
        ).reshape(-1)
        self._raw_action_high = np.asarray(
            self._env.action_space.high, dtype=np.float32
        ).reshape(-1)
        if self._raw_action_low.shape != (2,):
            raise RuntimeError(
                "This adapter expects one device with translation/rotation actions; "
                f"got raw action shape {self._env.action_space.shape}."
            )

        self.action_space = gym.spaces.Box(
            low=-np.ones(2, dtype=np.float32),
            high=np.ones(2, dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-np.ones(14, dtype=np.float32),
            high=np.ones(14, dtype=np.float32),
            dtype=np.float32,
        )

    @property
    def raw_action_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return copies of the underlying stEVE velocity limits."""

        return self._raw_action_low.copy(), self._raw_action_high.copy()

    @property
    def intervention(self):
        """Expose the wrapped intervention for diagnostics only."""

        return self._intervention

    def _flatten_observation(self, observation: Mapping[str, Any]) -> np.ndarray:
        try:
            flat = np.concatenate(
                [
                    np.asarray(observation[key], dtype=np.float32).reshape(-1)
                    for key in OBSERVATION_KEYS
                ]
            )
        except KeyError as exc:
            raise KeyError(
                f"Missing stEVE observation key {exc.args[0]!r}; expected "
                f"{OBSERVATION_KEYS}."
            ) from exc
        if flat.shape != self.observation_space.shape:
            raise RuntimeError(
                f"Expected observation shape {self.observation_space.shape}, "
                f"got {flat.shape}."
            )
        if not np.all(np.isfinite(flat)):
            raise FloatingPointError("stEVE returned a non-finite observation")
        # Numerical round-off in the episode normalizer can slightly exceed bounds.
        return np.clip(flat, -1.0, 1.0).astype(np.float32, copy=False)

    def _denormalize_action(self, action: np.ndarray) -> np.ndarray:
        normalized = np.asarray(action, dtype=np.float32)
        if normalized.shape != self.action_space.shape:
            raise ValueError(
                f"Expected action shape {self.action_space.shape}, got "
                f"{normalized.shape}."
            )
        if not np.all(np.isfinite(normalized)):
            raise ValueError("Action contains NaN or infinity")
        normalized = np.clip(normalized, -1.0, 1.0)
        raw = self._raw_action_low + 0.5 * (normalized + 1.0) * (
            self._raw_action_high - self._raw_action_low
        )
        return raw.reshape(self._env.action_space.shape).astype(np.float32)

    @staticmethod
    def _finite_float(value: Any, default: float = 0.0) -> float:
        try:
            converted = float(value)
        except (TypeError, ValueError, OverflowError):
            return float(default)
        return converted if np.isfinite(converted) else float(default)

    @staticmethod
    def _curvature_metrics(positions: np.ndarray) -> Tuple[float, float]:
        """Compute circumcircle curvature over valid consecutive DOF triplets."""

        if (
            positions.ndim != 2
            or positions.shape[0] < 3
            or positions.shape[1] < 3
        ):
            return 0.0, 0.0

        p0 = positions[:-2, :3]
        p1 = positions[1:-1, :3]
        p2 = positions[2:, :3]
        u = p1 - p0
        v = p2 - p1
        chord = p2 - p0
        u_norm = np.linalg.norm(u, axis=1)
        v_norm = np.linalg.norm(v, axis=1)
        chord_norm = np.linalg.norm(chord, axis=1)
        valid = (
            np.all(np.isfinite(p0), axis=1)
            & np.all(np.isfinite(p1), axis=1)
            & np.all(np.isfinite(p2), axis=1)
            & (u_norm > _CURVATURE_EPSILON_MM)
            & (v_norm > _CURVATURE_EPSILON_MM)
            & (chord_norm > _CURVATURE_EPSILON_MM)
        )
        if not np.any(valid):
            return 0.0, 0.0

        numerator = 2.0 * np.linalg.norm(np.cross(u[valid], v[valid]), axis=1)
        denominator = u_norm[valid] * v_norm[valid] * chord_norm[valid]
        curvatures = numerator / denominator
        curvatures = curvatures[np.isfinite(curvatures)]
        if curvatures.size == 0:
            return 0.0, 0.0
        return float(np.max(curvatures)), float(np.mean(curvatures))

    def _read_simulation_state(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        try:
            positions = np.asarray(
                self._simulation.dof_positions, dtype=np.float64
            )
        except (TypeError, ValueError):
            positions = np.empty((0, 3), dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] < 3:
            positions = np.empty((0, 3), dtype=np.float64)
        else:
            positions = positions[:, :3]

        fallback_tip = (
            self._previous_tip_position
            if self._previous_tip_position is not None
            else np.zeros(3, dtype=np.float64)
        )
        if positions.shape[0] > 0 and np.all(np.isfinite(positions[0])):
            # SofaBeamAdapter reverses DOFs, so index zero is the guidewire tip.
            current_tip = positions[0].copy()
        else:
            current_tip = fallback_tip.copy()

        fallback_length = (
            self._previous_inserted_length
            if self._previous_inserted_length is not None
            else 0.0
        )
        try:
            inserted_length = self._finite_float(
                self._simulation.inserted_lengths[0], fallback_length
            )
        except (IndexError, TypeError):
            inserted_length = float(fallback_length)
        try:
            rotation = self._finite_float(self._simulation.rotations[0])
        except (IndexError, TypeError):
            rotation = 0.0
        return positions, current_tip, inserted_length, rotation

    def _build_safety_metrics(
        self,
        requested_translation_speed: float,
        *,
        initial: bool = False,
    ) -> Dict[str, Any]:
        """Build monitoring-only signals; they do not affect training or planning."""

        positions, current_tip, inserted_length, rotation = (
            self._read_simulation_state()
        )
        adapter_metrics = self._simulation.get_safety_metrics()
        if not isinstance(adapter_metrics, Mapping):
            adapter_metrics = {}
        actual_time = self._finite_float(
            adapter_metrics.get("actual_simulation_time_s", 0.0)
        )
        requested_speed = self._finite_float(requested_translation_speed)

        tip_speed = 0.0
        insertion_speed = 0.0
        if (
            not initial
            and actual_time > 0.0
            and self._previous_tip_position is not None
            and self._previous_inserted_length is not None
        ):
            tip_speed = self._finite_float(
                np.linalg.norm(current_tip - self._previous_tip_position)
                / actual_time
            )
            insertion_speed = self._finite_float(
                (inserted_length - self._previous_inserted_length)
                / actual_time
            )

        max_curvature, mean_curvature = self._curvature_metrics(positions)
        if initial:
            requested_speed = 0.0
            collision_detected = False
            max_associations = 0
        else:
            collision_detected = bool(
                adapter_metrics.get("collision_association_detected", False)
            )
            try:
                max_associations = max(
                    0,
                    int(
                        adapter_metrics.get(
                            "max_collision_model_associations", 0
                        )
                    ),
                )
            except (TypeError, ValueError, OverflowError):
                max_associations = 0

        # Motion error is only an inconsistency proxy, not confirmed slippage.
        command_motion_error = self._finite_float(
            abs(requested_speed - insertion_speed)
        )
        safety_metrics = {
            "tip_speed_mm_s": float(tip_speed),
            "insertion_speed_mm_s": float(insertion_speed),
            "requested_translation_speed_mm_s": float(requested_speed),
            "command_motion_error_mm_s": float(command_motion_error),
            "max_curvature_mm_inv": float(max_curvature),
            "mean_curvature_mm_inv": float(mean_curvature),
            "inserted_length_mm": float(inserted_length),
            "rotation_rad": float(rotation),
            # Coarse SOFA model association, not a physical contact count.
            "collision_association_detected": bool(collision_detected),
            "max_collision_model_associations": int(max_associations),
            "simulation_error": bool(self._simulation.simulation_error),
        }
        self._previous_tip_position = current_tip.copy()
        self._previous_inserted_length = float(inserted_length)
        return safety_metrics

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            # MonoPlaneStatic does not forward its reset seed to
            # SofaBeamAdapter.reset(). Seed the adapter's existing generator
            # here so InsertionPoint.reset_devices() chooses a reproducible
            # initial rotation. This wrapper-local workaround avoids changing
            # stEVE source while making evaluation reproducible.
            self._simulation._rng = np.random.default_rng(  # pylint: disable=protected-access
                seed
            )
        self._simulation.simulation_error = False
        self._previous_tip_position = None
        self._previous_inserted_length = None
        observation, info = self._env.reset(seed=seed, options=options)
        self._episode_steps = 0
        output_info = dict(info)
        output_info["is_success"] = bool(output_info.get("is_success", False))
        output_info["simulation_error"] = bool(self._simulation.simulation_error)
        output_info["safety_metrics"] = self._build_safety_metrics(
            0.0, initial=True
        )
        return self._flatten_observation(observation), output_info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        raw_action = self._denormalize_action(action)
        requested_translation_speed = float(raw_action.reshape(-1)[0])
        observation, reward, terminated, truncated, info = self._env.step(raw_action)
        self._episode_steps += 1
        reward = float(reward)
        if not np.isfinite(reward):
            raise FloatingPointError("stEVE returned a non-finite reward")

        output_info = dict(info)
        output_info["is_success"] = bool(
            output_info.get("is_success", terminated)
        )
        output_info["simulation_error"] = bool(self._simulation.simulation_error)
        output_info["raw_action"] = raw_action.reshape(-1).copy()
        output_info["episode_step"] = self._episode_steps
        output_info["safety_metrics"] = self._build_safety_metrics(
            requested_translation_speed
        )
        return (
            self._flatten_observation(observation),
            reward,
            bool(terminated),
            bool(truncated),
            output_info,
        )

    def close(self) -> None:
        # Close components explicitly so a failed visual reset (before pygame
        # was imported/initialized) does not trigger SofaPygame.close()'s
        # unconditional ``self._pygame.quit()`` and mask the original error.
        self._intervention.close()
        if self._visualisation is not None and getattr(
            self._visualisation, "_pygame", None
        ) is not None:
            self._visualisation.close()

    def render(self) -> Optional[np.ndarray]:
        """Display and return the current RGB frame in human render mode."""

        if self.render_mode != "human":
            return None
        return self._env.render()


def make_steve_env(config: Optional[Mapping[str, Any]] = None) -> StEVEEnv:
    """Construct :class:`StEVEEnv` from an optional environment config."""

    return StEVEEnv(**dict(config or {}))
