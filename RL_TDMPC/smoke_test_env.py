"""Run the pre-training acceptance checks for the stEVE adapter."""

from __future__ import annotations

import numpy as np
from gymnasium.utils.env_checker import check_env

from envs.safety import (
    CURVATURE_STRATUM_NAMES,
    SAFETY_COST_NAMES,
    curvature_stratum_id,
    curvature_stratum_name,
    safety_cost_from_metrics,
)
from envs.steve_env import StEVEEnv
from eve.intervention import TRANSLATION_BLOCK_REASON_NAMES


SAFETY_FLOAT_KEYS = {
    "tip_speed_mm_s",
    "observed_insertion_speed_mm_s",
    "requested_translation_speed_mm_s",
    "applied_translation_speed_mm_s",
    "requested_applied_translation_error_mm_s",
    "normalized_requested_applied_translation_error",
    "requested_observed_insertion_speed_error_mm_s",
    "filtered_max_curvature_mm_inv",
    "mean_filtered_curvature_mm_inv",
    "curvature_median_spacing_mm",
    "curvature_minimum_segment_length_mm",
    "inserted_length_mm",
    "rotation_rad",
    "actual_simulation_time_s",
}
SAFETY_KEYS = SAFETY_FLOAT_KEYS | {
    "translation_action_blocked",
    "translation_block_reason_id",
    "translation_block_reason",
    "curvature_valid_triplet_count",
    "curvature_skipped_triplet_count",
    "collision_association_detected",
    "max_collision_model_associations",
    "simulation_error",
}


def assert_safety_metrics(info) -> None:
    metrics = info["safety_metrics"]
    assert set(metrics) == SAFETY_KEYS
    for key in SAFETY_FLOAT_KEYS:
        assert type(metrics[key]) is float
        assert np.isfinite(metrics[key])
    assert type(metrics["translation_action_blocked"]) is bool
    assert type(metrics["translation_block_reason_id"]) is int
    assert type(metrics["translation_block_reason"]) is str
    reason_id = metrics["translation_block_reason_id"]
    assert 0 <= reason_id < len(TRANSLATION_BLOCK_REASON_NAMES)
    assert (
        metrics["translation_block_reason"]
        == TRANSLATION_BLOCK_REASON_NAMES[reason_id]
    )
    assert (
        metrics["translation_action_blocked"]
        == (metrics["translation_block_reason"] != "none")
    )
    assert type(metrics["curvature_valid_triplet_count"]) is int
    assert metrics["curvature_valid_triplet_count"] >= 0
    assert type(metrics["curvature_skipped_triplet_count"]) is int
    assert metrics["curvature_skipped_triplet_count"] >= 0
    assert type(metrics["collision_association_detected"]) is bool
    assert type(metrics["max_collision_model_associations"]) is int
    assert type(metrics["simulation_error"]) is bool


def assert_safety_cost(env: StEVEEnv, info, *, reset: bool = False) -> None:
    cost = info["safety_cost"]
    assert isinstance(cost, np.ndarray)
    assert SAFETY_COST_NAMES == (
        "filtered_max_curvature_mm_inv",
        "normalized_requested_applied_translation_error",
    )
    assert cost.shape == (len(SAFETY_COST_NAMES),)
    assert cost.dtype == np.float32
    assert np.all(np.isfinite(cost))
    assert np.all(cost >= 0.0)
    if reset:
        np.testing.assert_array_equal(
            cost, np.zeros(len(SAFETY_COST_NAMES), dtype=np.float32)
        )
        return

    cost_by_name = dict(zip(SAFETY_COST_NAMES, cost))
    metrics = info["safety_metrics"]
    raw_low, raw_high = env.raw_action_limits
    translation_limit = max(abs(float(raw_low[0])), abs(float(raw_high[0])))
    np.testing.assert_allclose(
        cost_by_name["filtered_max_curvature_mm_inv"],
        metrics["filtered_max_curvature_mm_inv"],
    )
    np.testing.assert_allclose(
        cost_by_name["normalized_requested_applied_translation_error"],
        metrics["normalized_requested_applied_translation_error"],
    )
    np.testing.assert_allclose(
        metrics["normalized_requested_applied_translation_error"],
        metrics["requested_applied_translation_error_mm_s"] / translation_limit,
    )


def legacy_unfiltered_max_curvature(positions: np.ndarray) -> float:
    """Reproduce the former unfiltered formula for the regression fixture."""

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
        & (u_norm > 1e-8)
        & (v_norm > 1e-8)
        & (chord_norm > 1e-8)
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        curvature = (
            2.0
            * np.linalg.norm(np.cross(u[valid], v[valid]), axis=1)
            / (u_norm[valid] * v_norm[valid] * chord_norm[valid])
        )
    finite = curvature[np.isfinite(curvature)]
    return float(np.max(finite)) if finite.size else 0.0


def main() -> None:
    assert TRANSLATION_BLOCK_REASON_NAMES == (
        "none",
        "lower_insertion_boundary",
        "device_length_limit",
        "vessel_tree_end",
        "other",
    )
    assert CURVATURE_STRATUM_NAMES == (
        "low",
        "medium",
        "high",
        "extreme",
    )
    for curvature, expected_id in (
        (0.0, 0),
        (0.049, 0),
        (0.05, 1),
        (0.099, 1),
        (0.10, 2),
        (0.249, 2),
        (0.25, 3),
        (1.0, 3),
    ):
        stratum_id = curvature_stratum_id(
            curvature,
            (0.05, 0.10, 0.25),
        )
        assert stratum_id == expected_id
        assert curvature_stratum_name(stratum_id) == CURVATURE_STRATUM_NAMES[
            expected_id
        ]

    (
        max_curvature,
        mean_curvature,
        median_spacing,
        minimum_segment_length,
        valid_triplets,
        skipped_triplets,
    ) = StEVEEnv._curvature_metrics(
        np.asarray([[1, 0, 0], [0, 1, 0], [-1, 0, 0]], dtype=np.float64)
    )
    np.testing.assert_allclose([max_curvature, mean_curvature], [1.0, 1.0])
    np.testing.assert_allclose(median_spacing, np.sqrt(2.0))
    np.testing.assert_allclose(
        minimum_segment_length, 0.05 * np.sqrt(2.0)
    )
    assert valid_triplets == 1
    assert skipped_triplets == 0

    # A near-duplicate, non-collinear triplet makes the old unfiltered
    # circumcircle formula exceed 1e3 mm^-1.  A5 skips the two triplets
    # touching that tiny segment while retaining two normal-scale curves.
    near_duplicate_positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1e-4, 0.0, 0.0],
            [1e-4, 2e-8, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    assert legacy_unfiltered_max_curvature(near_duplicate_positions) > 1e3
    filtered_result = StEVEEnv._curvature_metrics(
        near_duplicate_positions
    )
    assert np.all(np.isfinite(filtered_result[:4]))
    assert 0.0 < filtered_result[0] < 10.0
    assert 0.0 < filtered_result[1] < 10.0
    np.testing.assert_allclose(filtered_result[2], 0.999900000000005)
    np.testing.assert_allclose(
        filtered_result[3], 0.05 * filtered_result[2]
    )
    assert filtered_result[4:] == (2, 2)

    degenerate_result = StEVEEnv._curvature_metrics(np.zeros((3, 3)))
    assert degenerate_result == (0.0, 0.0, 0.0, 0.0, 0, 1)

    synthetic_cost = safety_cost_from_metrics(
        {
            "collision_association_detected": True,
            "filtered_max_curvature_mm_inv": 0.25,
            "requested_applied_translation_error_mm_s": 25.0,
        },
        translation_speed_limit_mm_s=50.0,
    )
    synthetic_by_name = dict(zip(SAFETY_COST_NAMES, synthetic_cost))
    assert all("collision" not in name for name in SAFETY_COST_NAMES)
    assert synthetic_by_name["filtered_max_curvature_mm_inv"] == 0.25
    assert (
        synthetic_by_name[
            "normalized_requested_applied_translation_error"
        ]
        == 0.5
    )
    # Collision association is monitoring-only and cannot alter learnable cost.
    synthetic_without_collision = safety_cost_from_metrics(
        {
            "collision_association_detected": False,
            "filtered_max_curvature_mm_inv": 0.25,
            "requested_applied_translation_error_mm_s": 25.0,
        },
        translation_speed_limit_mm_s=50.0,
    )
    np.testing.assert_array_equal(synthetic_cost, synthetic_without_collision)
    unclipped_cost = safety_cost_from_metrics(
        {
            "filtered_max_curvature_mm_inv": 0.0,
            "requested_applied_translation_error_mm_s": 100.0,
        },
        translation_speed_limit_mm_s=50.0,
    )
    assert unclipped_cost[1] == 2.0

    env = StEVEEnv()
    try:
        observation, reset_info = env.reset(seed=1)
        assert observation.shape == (14,)
        assert observation.dtype == np.float32
        assert np.all(np.isfinite(observation))
        assert env.observation_space.contains(observation)
        assert env.action_space.shape == (2,)
        assert np.all(env.action_space.low == -1)
        assert np.all(env.action_space.high == 1)
        assert_safety_metrics(reset_info)
        assert_safety_cost(env, reset_info, reset=True)
        reset_metrics = reset_info["safety_metrics"]
        assert reset_metrics["tip_speed_mm_s"] == 0.0
        assert reset_metrics["observed_insertion_speed_mm_s"] == 0.0
        assert reset_metrics["requested_translation_speed_mm_s"] == 0.0
        assert reset_metrics["applied_translation_speed_mm_s"] == 0.0
        assert not reset_metrics["translation_action_blocked"]
        assert reset_metrics["translation_block_reason_id"] == 0
        assert reset_metrics["translation_block_reason"] == "none"
        assert (
            reset_metrics["requested_applied_translation_error_mm_s"] == 0.0
        )
        assert (
            reset_metrics["normalized_requested_applied_translation_error"]
            == 0.0
        )
        assert (
            reset_metrics[
                "requested_observed_insertion_speed_error_mm_s"
            ]
            == 0.0
        )
        assert reset_metrics["actual_simulation_time_s"] == 0.0
        assert not reset_metrics["collision_association_detected"]
        assert reset_metrics["max_collision_model_associations"] == 0
        np.testing.assert_array_equal(
            env.intervention.requested_action, np.zeros((1, 2))
        )
        np.testing.assert_array_equal(
            env.intervention.applied_action, np.zeros((1, 2))
        )

        _, reward, _, _, info = env.step(np.ones(2, dtype=np.float32))
        assert np.isfinite(reward)
        np.testing.assert_allclose(info["raw_action"], [50.0, 3.14], rtol=1e-5)
        assert_safety_metrics(info)
        assert_safety_cost(env, info)
        assert info["safety_metrics"]["requested_translation_speed_mm_s"] == 50.0
        assert info["safety_metrics"]["applied_translation_speed_mm_s"] == 50.0
        assert not info["safety_metrics"]["translation_action_blocked"]
        assert info["safety_metrics"]["translation_block_reason_id"] == 0
        assert info["safety_metrics"]["translation_block_reason"] == "none"
        assert (
            info["safety_metrics"][
                "requested_applied_translation_error_mm_s"
            ]
            == 0.0
        )
        np.testing.assert_allclose(
            info["safety_metrics"]["observed_insertion_speed_mm_s"],
            50.0,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            info["safety_metrics"][
                "requested_observed_insertion_speed_error_mm_s"
            ],
            0.0,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            env.intervention.requested_action,
            [[50.0, 3.14]],
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            env.intervention.applied_action,
            [[50.0, 3.14]],
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            info["safety_metrics"]["inserted_length_mm"],
            env.intervention.device_lengths_inserted[0],
        )
        np.testing.assert_allclose(
            env.intervention.simulation.get_safety_metrics()[
                "actual_simulation_time_s"
            ],
            0.132,
            atol=1e-12,
        )

        # A reset must restore the physical device instead of leaking state across episodes.
        _, reset_info = env.reset(seed=2)
        np.testing.assert_allclose(
            env.intervention.device_lengths_inserted, [0.0], atol=1e-6
        )
        assert_safety_metrics(reset_info)
        assert_safety_cost(env, reset_info, reset=True)
        assert reset_info["safety_metrics"]["tip_speed_mm_s"] == 0.0
        assert (
            reset_info["safety_metrics"]["observed_insertion_speed_mm_s"]
            == 0.0
        )
        _, _, _, _, zero_action_info = env.step(
            np.zeros(2, dtype=np.float32)
        )
        assert_safety_metrics(zero_action_info)
        assert_safety_cost(env, zero_action_info)
        assert (
            zero_action_info["safety_metrics"][
                "requested_translation_speed_mm_s"
            ]
            == 0.0
        )
        assert (
            zero_action_info["safety_metrics"][
                "applied_translation_speed_mm_s"
            ]
            == 0.0
        )
        assert not zero_action_info["safety_metrics"][
            "translation_action_blocked"
        ]
        assert (
            zero_action_info["safety_metrics"][
                "requested_applied_translation_error_mm_s"
            ]
            == 0.0
        )
        _, _, _, _, blocked_info = env.step(
            np.asarray([-1.0, 0.0], dtype=np.float32)
        )
        np.testing.assert_allclose(
            blocked_info["safety_metrics"][
                "observed_insertion_speed_mm_s"
            ],
            0.0,
            atol=1e-5,
        )
        assert (
            blocked_info["safety_metrics"]["requested_translation_speed_mm_s"]
            == -50.0
        )
        assert (
            blocked_info["safety_metrics"]["applied_translation_speed_mm_s"]
            == 0.0
        )
        assert blocked_info["safety_metrics"]["translation_action_blocked"]
        assert (
            blocked_info["safety_metrics"]["translation_block_reason_id"]
            == TRANSLATION_BLOCK_REASON_NAMES.index(
                "lower_insertion_boundary"
            )
        )
        assert (
            blocked_info["safety_metrics"]["translation_block_reason"]
            == "lower_insertion_boundary"
        )
        np.testing.assert_allclose(
            blocked_info["safety_metrics"][
                "requested_applied_translation_error_mm_s"
            ],
            50.0,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            blocked_info["safety_metrics"][
                "requested_observed_insertion_speed_error_mm_s"
            ],
            50.0,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            env.intervention.requested_action, [[-50.0, 0.0]]
        )
        np.testing.assert_allclose(
            env.intervention.applied_action, [[0.0, 0.0]]
        )
        assert_safety_cost(env, blocked_info)
        blocked_cost = dict(
            zip(SAFETY_COST_NAMES, blocked_info["safety_cost"])
        )
        np.testing.assert_allclose(
            blocked_cost[
                "normalized_requested_applied_translation_error"
            ],
            1.0,
            atol=1e-5,
        )
        _, _, _, _, after_block_info = env.step(
            np.zeros(2, dtype=np.float32)
        )
        assert (
            after_block_info["safety_metrics"][
                "translation_block_reason_id"
            ]
            == 0
        )
        assert (
            after_block_info["safety_metrics"][
                "translation_block_reason"
            ]
            == "none"
        )
        assert not after_block_info["safety_metrics"][
            "translation_action_blocked"
        ]

        # Seed 301 reaches the fixed vessel-tree end under maximum forward
        # insertion. The intervention must preserve the requested command while
        # exposing the exact zero translation sent to SOFA after tree-end masking.
        _, reset_info = env.reset(seed=301)
        assert_safety_cost(env, reset_info, reset=True)
        forward_blocked_info = None
        for _ in range(90):
            _, _, terminated, truncated, candidate_info = env.step(
                np.asarray([1.0, 0.0], dtype=np.float32)
            )
            assert_safety_metrics(candidate_info)
            assert_safety_cost(env, candidate_info)
            if candidate_info["safety_metrics"]["translation_action_blocked"]:
                forward_blocked_info = candidate_info
                break
            assert not terminated and not truncated
        assert forward_blocked_info is not None
        forward_blocked_metrics = forward_blocked_info["safety_metrics"]
        assert forward_blocked_metrics["requested_translation_speed_mm_s"] == 50.0
        assert forward_blocked_metrics["applied_translation_speed_mm_s"] == 0.0
        assert forward_blocked_metrics["translation_action_blocked"]
        assert (
            forward_blocked_metrics["translation_block_reason_id"]
            == TRANSLATION_BLOCK_REASON_NAMES.index("vessel_tree_end")
        )
        assert (
            forward_blocked_metrics["translation_block_reason"]
            == "vessel_tree_end"
        )
        np.testing.assert_allclose(
            forward_blocked_metrics[
                "requested_applied_translation_error_mm_s"
            ],
            50.0,
        )
        forward_blocked_cost = dict(
            zip(SAFETY_COST_NAMES, forward_blocked_info["safety_cost"])
        )
        np.testing.assert_allclose(
            forward_blocked_cost[
                "normalized_requested_applied_translation_error"
            ],
            1.0,
        )

        # The production 450 mm device limit is unreachable because the fixed
        # tree ends first. Exercise the intervention mask itself with a
        # temporary, test-only short maximum; no such sample enters a dataset.
        _, reset_info = env.reset(seed=302)
        assert reset_info["safety_metrics"]["translation_block_reason"] == "none"
        device = env.intervention.devices[0]
        original_length = device.length
        try:
            device.length = 0.5
            _, _, _, _, device_limit_info = env.step(
                np.asarray([1.0, 0.0], dtype=np.float32)
            )
        finally:
            device.length = original_length
        assert_safety_metrics(device_limit_info)
        assert_safety_cost(env, device_limit_info)
        assert (
            device_limit_info["safety_metrics"]["translation_block_reason"]
            == "device_length_limit"
        )
        assert (
            device_limit_info["safety_metrics"]["translation_block_reason_id"]
            == TRANSLATION_BLOCK_REASON_NAMES.index("device_length_limit")
        )
        _, reset_info = env.reset(seed=303)
        assert reset_info["safety_metrics"]["translation_block_reason_id"] == 0
        assert reset_info["safety_metrics"]["translation_block_reason"] == "none"

        seeded_observation, _ = env.reset(seed=123)
        repeated_observation, _ = env.reset(seed=123)
        np.testing.assert_allclose(seeded_observation, repeated_observation, atol=1e-6)

        # Exercise the complete five-value step contract with random actions.
        observation, reset_info = env.reset(seed=7)
        assert_safety_cost(env, reset_info, reset=True)
        rng = np.random.default_rng(7)
        collected_observations = [observation.copy()]
        collected_actions = []
        collected_rewards = []
        collected_safety_costs = []
        collected_terminated = []
        for _ in range(10):
            action = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
            result = env.step(action)
            assert len(result) == 5
            observation, reward, terminated, truncated, info = result
            assert observation.shape == (14,)
            assert np.all(np.isfinite(observation))
            assert np.isfinite(reward)
            assert_safety_metrics(info)
            assert_safety_cost(env, info)
            np.testing.assert_allclose(
                info["safety_metrics"]["requested_translation_speed_mm_s"],
                env.intervention.requested_action[0, 0],
            )
            np.testing.assert_allclose(
                info["safety_metrics"]["applied_translation_speed_mm_s"],
                env.intervention.applied_action[0, 0],
            )
            expected_action_error = abs(
                env.intervention.requested_action[0, 0]
                - env.intervention.applied_action[0, 0]
            )
            np.testing.assert_allclose(
                info["safety_metrics"][
                    "requested_applied_translation_error_mm_s"
                ],
                expected_action_error,
            )
            assert (
                info["safety_metrics"]["translation_action_blocked"]
                == (
                    abs(env.intervention.requested_action[0, 0]) > 1e-9
                    and expected_action_error > 1e-9
                )
            )
            assert not terminated and not truncated
            collected_observations.append(observation.copy())
            collected_actions.append(action.copy())
            collected_rewards.append(float(reward))
            collected_safety_costs.append(info["safety_cost"].copy())
            collected_terminated.append(bool(terminated))
        transition_count = len(collected_actions)
        assert len(collected_observations) == transition_count + 1
        assert len(collected_rewards) == transition_count
        assert len(collected_safety_costs) == transition_count
        assert len(collected_terminated) == transition_count

        _, reset_info = env.reset(seed=123)
        assert_safety_cost(env, reset_info, reset=True)
        steps = 0
        terminated = truncated = False
        while not (terminated or truncated):
            observation, reward, terminated, truncated, info = env.step(
                np.zeros(2, dtype=np.float32)
            )
            steps += 1
            assert observation.shape == (14,)
            assert np.all(np.isfinite(observation))
            assert np.isfinite(reward)
            assert_safety_metrics(info)
            assert_safety_cost(env, info)
        assert steps == 200
        assert truncated

        # This invokes extra resets/steps and checks the full Gymnasium contract.
        check_env(env, skip_render_check=True)
        print(
            "PASS: reset/step, safety metrics/costs, 10 aligned random "
            "transitions, robust curvature, requested/applied actions, "
            "finite (14,) observations, normalized actions, and a complete "
            "200-step episode."
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
