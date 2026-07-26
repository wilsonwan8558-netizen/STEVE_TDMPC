# pylint: disable=unused-argument
from typing import Any, Dict, List, Optional
import gymnasium as gym

import numpy as np
from .intervention import SimulatedIntervention
from .target import Target
from .vesseltree import VesselTree
from .vesseltree.vesseltree import at_tree_end
from .fluoroscopy import SimulatedFluoroscopy
from .device import Device
from .simulation import Simulation
from .translation_block import (
    TRANSLATION_BLOCK_REASON_NAMES,
    TRANSLATION_MISMATCH_TOLERANCE_MM_S,
)


_TRANSLATION_BLOCK_NONE = TRANSLATION_BLOCK_REASON_NAMES.index("none")
_TRANSLATION_BLOCK_LOWER_BOUNDARY = TRANSLATION_BLOCK_REASON_NAMES.index(
    "lower_insertion_boundary"
)
_TRANSLATION_BLOCK_DEVICE_LENGTH = TRANSLATION_BLOCK_REASON_NAMES.index(
    "device_length_limit"
)
_TRANSLATION_BLOCK_VESSEL_TREE_END = TRANSLATION_BLOCK_REASON_NAMES.index(
    "vessel_tree_end"
)
_TRANSLATION_BLOCK_OTHER = TRANSLATION_BLOCK_REASON_NAMES.index("other")


class MonoPlaneStatic(SimulatedIntervention):
    def __init__(
        self,
        vessel_tree: VesselTree,
        devices: List[Device],
        simulation: Simulation,
        fluoroscopy: SimulatedFluoroscopy,
        target: Target,
        stop_device_at_tree_end: bool = True,
        normalize_action: bool = False,
    ) -> None:
        self.vessel_tree = vessel_tree
        self.devices = devices
        self.target = target
        self.fluoroscopy = fluoroscopy
        self.stop_device_at_tree_end = stop_device_at_tree_end
        self.normlaize_action = normalize_action
        self.simulation = simulation
        self._np_random = np.random.default_rng()

        self.velocity_limits = np.array(
            [device.velocity_limit for device in self.devices]
        )
        # These snapshots always use physical units, independent of whether
        # the public action space is normalized.
        self._requested_action = np.zeros_like(self.velocity_limits)
        self._applied_action = np.zeros_like(self.velocity_limits)
        self._translation_block_reason_ids = np.full(
            len(self.devices),
            _TRANSLATION_BLOCK_NONE,
            dtype=np.int8,
        )
        self.last_action = np.zeros_like(self.velocity_limits)
        self._device_lengths_inserted = self.simulation.inserted_lengths
        self._device_rotations = self.simulation.rotations

    @property
    def device_lengths_inserted(self) -> List[float]:
        return self.simulation.inserted_lengths

    @property
    def device_rotations(self) -> List[float]:
        return self.simulation.rotations

    @property
    def device_lengths_maximum(self) -> List[float]:
        return [device.length for device in self.devices]

    @property
    def device_diameters(self) -> List[float]:
        return [device.sofa_device.radius * 2 for device in self.devices]

    @property
    def action_space(self) -> gym.spaces.Box:
        if self.normalize_action:
            high = np.ones_like(self.velocity_limits)
            space = gym.spaces.Box(low=-high, high=high)
        else:
            space = gym.spaces.Box(low=-self.velocity_limits, high=self.velocity_limits)
        return space

    @property
    def requested_action(self) -> np.ndarray:
        """Return the latest clipped physical command before safety masks."""

        return self._requested_action.copy()

    @property
    def applied_action(self) -> np.ndarray:
        """Return the latest physical command passed through safety masks."""

        return self._applied_action.copy()

    @property
    def translation_block_reason_ids(self) -> np.ndarray:
        """Return one canonical translation-block reason ID per device."""

        return self._translation_block_reason_ids.copy()

    def step(self, action: np.ndarray) -> None:
        # Step-local metadata must never leak across actions, including zero
        # commands following a blocked command.
        self._translation_block_reason_ids.fill(_TRANSLATION_BLOCK_NONE)
        action = np.array(action).reshape(self.velocity_limits.shape)
        if self.normalize_action:
            normalized_action = np.clip(action, -1.0, 1.0)
            high = self.velocity_limits
            low = -high
            requested_action = (
                (normalized_action + 1) / 2 * (high - low) + low
            )
        else:
            normalized_action = None
            requested_action = np.clip(
                action, -self.velocity_limits, self.velocity_limits
            )

        self._requested_action = requested_action.copy()
        applied_action = requested_action.copy()

        inserted_lengths = np.array(self.device_lengths_inserted)
        max_lengths = np.array(self.device_lengths_maximum)
        duration = 1 / self.fluoroscopy.image_frequency
        mask = (
            inserted_lengths + applied_action[:, 0] * duration <= 0.0
        )
        changed = mask & (
            applied_action[:, 0] < -TRANSLATION_MISMATCH_TOLERANCE_MM_S
        )
        self._translation_block_reason_ids[
            changed
        ] = _TRANSLATION_BLOCK_LOWER_BOUNDARY
        applied_action[mask, 0] = 0.0
        mask = (
            inserted_lengths + applied_action[:, 0] * duration >= max_lengths
        )
        changed = mask & (
            applied_action[:, 0] > TRANSLATION_MISMATCH_TOLERANCE_MM_S
        )
        self._translation_block_reason_ids[
            changed
        ] = _TRANSLATION_BLOCK_DEVICE_LENGTH
        applied_action[mask, 0] = 0.0
        tip = self.simulation.dof_positions[0]
        if self.stop_device_at_tree_end and at_tree_end(tip, self.vessel_tree):
            max_length = max(inserted_lengths)
            if max_length > 10:
                dist_to_longest = -1 * inserted_lengths + max_length
                movement = applied_action[:, 0] * duration
                mask = movement > dist_to_longest
                changed = mask & (
                    applied_action[:, 0]
                    > TRANSLATION_MISMATCH_TOLERANCE_MM_S
                )
                self._translation_block_reason_ids[
                    changed
                ] = _TRANSLATION_BLOCK_VESSEL_TREE_END
                applied_action[mask, 0] = 0.0

        unexplained = (
            self._translation_block_reason_ids == _TRANSLATION_BLOCK_NONE
        ) & (
            np.abs(requested_action[:, 0] - applied_action[:, 0])
            > TRANSLATION_MISMATCH_TOLERANCE_MM_S
        )
        self._translation_block_reason_ids[unexplained] = _TRANSLATION_BLOCK_OTHER
        self._applied_action = applied_action.copy()
        # Preserve the legacy public contract explicitly. Physical-action
        # interventions historically exposed the post-mask action because
        # ``last_action`` aliased the masked array. Normalized interventions
        # exposed the clipped normalized command because denormalization
        # rebound the local array before masking.
        self.last_action = (
            normalized_action.copy()
            if normalized_action is not None
            else self._applied_action.copy()
        )

        self.vessel_tree.step()
        self.simulation.step(self._applied_action.copy(), duration)
        self.fluoroscopy.step()
        self.target.step()

    def reset(
        self,
        episode_number: int = 0,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        vessel_seed = None if seed is None else self._np_random.integers(0, 2**31)
        self.vessel_tree.reset(episode_number, vessel_seed)
        ip_pos = self.vessel_tree.insertion.position
        ip_dir = self.vessel_tree.insertion.direction
        self.simulation.reset(
            insertion_point=ip_pos,
            insertion_direction=ip_dir,
            mesh_path=self.vessel_tree.mesh_path,
            devices=self.devices,
            coords_low=self.vessel_tree.coordinate_space.low,
            coords_high=self.vessel_tree.coordinate_space.high,
            vessel_visual_path=self.vessel_tree.visu_mesh_path,
        )
        target_seed = None if seed is None else self._np_random.integers(0, 2**31)
        self.target.reset(episode_number, target_seed)
        self.fluoroscopy.reset(episode_number)
        self._requested_action.fill(0.0)
        self._applied_action.fill(0.0)
        self._translation_block_reason_ids.fill(_TRANSLATION_BLOCK_NONE)
        self.last_action.fill(0.0)

    def _update_states(self):
        self._device_lengths_inserted = self.simulation.inserted_lengths
        self._device_rotations = self.simulation.rotations

    def close(self) -> None:
        self.simulation.close()

    def reset_devices(self) -> None:
        self.simulation.reset_devices()
