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
        observation, info = self._env.reset(seed=seed, options=options)
        self._episode_steps = 0
        output_info = dict(info)
        output_info["is_success"] = bool(output_info.get("is_success", False))
        output_info["simulation_error"] = bool(self._simulation.simulation_error)
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
