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

from .safety import safety_cost_from_metrics, zero_safety_cost


# Allow ``python RL_TDMPC/...`` from a source checkout before ``pip install -e .``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import eve  # noqa: E402  (must follow source-checkout path setup)


OBSERVATION_KEYS: Tuple[str, ...] = ("position", "target", "rotation")
_CURVATURE_EPSILON_MM = 1e-8
_CURVATURE_MINIMUM_SPACING_FRACTION = 0.05
_TRANSLATION_ACTION_TOLERANCE_MM_S = 1e-9


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
        self._translation_speed_limit_mm_s = max(
            abs(float(self._raw_action_low[0])),
            abs(float(self._raw_action_high[0])),
        )
        if (
            not np.isfinite(self._translation_speed_limit_mm_s)
            or self._translation_speed_limit_mm_s <= 0.0
        ):
            raise ValueError(
                "The physical translation-speed limit must be finite and positive"
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
    def _curvature_metrics(
        positions: np.ndarray,
    ) -> Tuple[float, float, float, float, int, int]:
        """Compute spacing-filtered circumcircle curvature over final 3-D DOFs.

        SOFA can place consecutive inactive or compressed DOFs at nearly the
        same position.  Direct circumcircle curvature over those triplets is
        numerically unstable, so the filter derives a scale from the median
        finite, positive adjacent spacing and rejects triplets containing a
        segment shorter than five percent of that spacing.
        """

        if (
            positions.ndim != 2
            or positions.shape[0] < 3
            or positions.shape[1] < 3
        ):
            return 0.0, 0.0, 0.0, 0.0, 0, 0

        positions_3d = positions[:, :3]
        triplet_count = positions_3d.shape[0] - 2
        adjacent = positions_3d[1:] - positions_3d[:-1]
        adjacent_norms = np.linalg.norm(adjacent, axis=1)
        valid_spacings = adjacent_norms[
            np.isfinite(adjacent_norms) & (adjacent_norms > 0.0)
        ]
        if valid_spacings.size == 0:
            return 0.0, 0.0, 0.0, 0.0, 0, int(triplet_count)

        median_spacing = float(np.median(valid_spacings))
        minimum_segment_length = float(
            _CURVATURE_MINIMUM_SPACING_FRACTION * median_spacing
        )

        p0 = positions_3d[:-2]
        p1 = positions_3d[1:-1]
        p2 = positions_3d[2:]
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
            & np.isfinite(u_norm)
            & np.isfinite(v_norm)
            & np.isfinite(chord_norm)
            & (u_norm > 0.0)
            & (v_norm > 0.0)
            # A segment exactly on the A5 threshold remains valid.
            & ~(u_norm < minimum_segment_length)
            & ~(v_norm < minimum_segment_length)
            & (chord_norm > _CURVATURE_EPSILON_MM)
        )
        if not np.any(valid):
            return (
                0.0,
                0.0,
                median_spacing,
                minimum_segment_length,
                0,
                int(triplet_count),
            )

        candidate_indices = np.flatnonzero(valid)
        numerator = 2.0 * np.linalg.norm(
            np.cross(u[candidate_indices], v[candidate_indices]), axis=1
        )
        denominator = (
            u_norm[candidate_indices]
            * v_norm[candidate_indices]
            * chord_norm[candidate_indices]
        )
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            candidate_curvatures = numerator / denominator
        finite_curvature = (
            np.isfinite(numerator)
            & np.isfinite(denominator)
            & (denominator > 0.0)
            & np.isfinite(candidate_curvatures)
        )
        curvatures = candidate_curvatures[finite_curvature]
        valid_triplet_count = int(curvatures.size)
        skipped_triplet_count = int(triplet_count - valid_triplet_count)
        if curvatures.size == 0:
            return (
                0.0,
                0.0,
                median_spacing,
                minimum_segment_length,
                0,
                skipped_triplet_count,
            )
        return (
            float(np.max(curvatures)),
            float(np.mean(curvatures)),
            median_spacing,
            minimum_segment_length,
            valid_triplet_count,
            skipped_triplet_count,
        )

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

    def _read_translation_action_targets(self) -> Tuple[float, float]:
        """Read the intervention's physical pre-mask and post-mask targets."""

        expected_shape = np.asarray(
            self._intervention.velocity_limits
        ).shape
        requested_action = np.asarray(
            self._intervention.requested_action, dtype=np.float64
        )
        applied_action = np.asarray(
            self._intervention.applied_action, dtype=np.float64
        )
        if requested_action.shape != expected_shape:
            raise RuntimeError(
                "MonoPlaneStatic.requested_action must have shape "
                f"{expected_shape}, got {requested_action.shape}"
            )
        if applied_action.shape != expected_shape:
            raise RuntimeError(
                "MonoPlaneStatic.applied_action must have shape "
                f"{expected_shape}, got {applied_action.shape}"
            )
        requested_translation = self._finite_float(
            requested_action.reshape(-1, 2)[0, 0]
        )
        applied_translation = self._finite_float(
            applied_action.reshape(-1, 2)[0, 0]
        )
        return requested_translation, applied_translation

    def _build_safety_metrics(
        self,
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
        requested_speed, applied_speed = (
            self._read_translation_action_targets()
        )

        tip_speed = 0.0
        observed_insertion_speed = 0.0
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
            observed_insertion_speed = self._finite_float(
                (inserted_length - self._previous_inserted_length)
                / actual_time
            )

        (
            filtered_max_curvature,
            mean_filtered_curvature,
            curvature_median_spacing,
            curvature_minimum_segment_length,
            curvature_valid_triplet_count,
            curvature_skipped_triplet_count,
        ) = self._curvature_metrics(positions)
        if initial:
            requested_speed = 0.0
            applied_speed = 0.0
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

        requested_applied_error = self._finite_float(
            abs(requested_speed - applied_speed)
        )
        # This is an observed kinematic mismatch only.  It is not confirmed
        # slippage, tissue force, or a learnable safety-cost channel.
        requested_observed_insertion_speed_error = self._finite_float(
            abs(requested_speed - observed_insertion_speed)
        )
        safety_metrics = {
            "tip_speed_mm_s": float(tip_speed),
            "observed_insertion_speed_mm_s": float(
                observed_insertion_speed
            ),
            "requested_translation_speed_mm_s": float(requested_speed),
            "applied_translation_speed_mm_s": float(applied_speed),
            "translation_action_blocked": bool(
                abs(requested_speed)
                > _TRANSLATION_ACTION_TOLERANCE_MM_S
                and requested_applied_error
                > _TRANSLATION_ACTION_TOLERANCE_MM_S
            ),
            "requested_applied_translation_error_mm_s": float(
                requested_applied_error
            ),
            "requested_observed_insertion_speed_error_mm_s": float(
                requested_observed_insertion_speed_error
            ),
            "filtered_max_curvature_mm_inv": float(
                filtered_max_curvature
            ),
            "mean_filtered_curvature_mm_inv": float(
                mean_filtered_curvature
            ),
            "curvature_median_spacing_mm": float(
                curvature_median_spacing
            ),
            "curvature_minimum_segment_length_mm": float(
                curvature_minimum_segment_length
            ),
            "curvature_valid_triplet_count": int(
                curvature_valid_triplet_count
            ),
            "curvature_skipped_triplet_count": int(
                curvature_skipped_triplet_count
            ),
            "inserted_length_mm": float(inserted_length),
            "rotation_rad": float(rotation),
            "actual_simulation_time_s": float(actual_time),
            # Persistent collision-model association monitoring only.  This
            # can survive retraction/reset in a reused SOFA scene; it is not a
            # contact event, contact count, force, or learnable cost channel.
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
        output_info["safety_metrics"] = self._build_safety_metrics(initial=True)
        output_info["safety_cost"] = zero_safety_cost()
        return self._flatten_observation(observation), output_info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        raw_action = self._denormalize_action(action)
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
        safety_metrics = self._build_safety_metrics()
        output_info["safety_metrics"] = safety_metrics
        output_info["safety_cost"] = safety_cost_from_metrics(
            safety_metrics,
            self._translation_speed_limit_mm_s,
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
