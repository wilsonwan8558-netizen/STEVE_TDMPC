"""Run the pre-training acceptance checks for the stEVE adapter."""

from __future__ import annotations

import numpy as np
import torch
from gymnasium.utils.env_checker import check_env

from envs.safety import SAFETY_COST_NAMES, safety_cost_from_metrics
from envs.steve_env import StEVEEnv
from tdmpc2.replay_buffer import EpisodeReplayBuffer


SAFETY_FLOAT_KEYS = {
    "tip_speed_mm_s",
    "insertion_speed_mm_s",
    "requested_translation_speed_mm_s",
    "command_motion_error_mm_s",
    "max_curvature_mm_inv",
    "mean_curvature_mm_inv",
    "inserted_length_mm",
    "rotation_rad",
}
SAFETY_KEYS = SAFETY_FLOAT_KEYS | {
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
    assert type(metrics["collision_association_detected"]) is bool
    assert type(metrics["max_collision_model_associations"]) is int
    assert type(metrics["simulation_error"]) is bool


def assert_safety_cost(env: StEVEEnv, info, *, reset: bool = False) -> None:
    cost = info["safety_cost"]
    assert isinstance(cost, np.ndarray)
    assert len(SAFETY_COST_NAMES) == 3
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
        cost_by_name["collision_association"],
        float(metrics["collision_association_detected"]),
    )
    np.testing.assert_allclose(
        cost_by_name["max_curvature_mm_inv"],
        metrics["max_curvature_mm_inv"],
    )
    np.testing.assert_allclose(
        cost_by_name["normalized_command_motion_error"],
        metrics["command_motion_error_mm_s"] / translation_limit,
    )


def main() -> None:
    max_curvature, mean_curvature = StEVEEnv._curvature_metrics(
        np.asarray([[1, 0, 0], [0, 1, 0], [-1, 0, 0]], dtype=np.float64)
    )
    np.testing.assert_allclose([max_curvature, mean_curvature], [1.0, 1.0])
    assert StEVEEnv._curvature_metrics(np.zeros((3, 3))) == (0.0, 0.0)
    synthetic_cost = safety_cost_from_metrics(
        {
            "collision_association_detected": True,
            "max_curvature_mm_inv": 0.25,
            "command_motion_error_mm_s": 100.0,
        },
        translation_speed_limit_mm_s=50.0,
    )
    synthetic_by_name = dict(zip(SAFETY_COST_NAMES, synthetic_cost))
    assert synthetic_by_name["collision_association"] == 1.0
    assert synthetic_by_name["normalized_command_motion_error"] == 2.0

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
        assert reset_metrics["insertion_speed_mm_s"] == 0.0
        assert reset_metrics["requested_translation_speed_mm_s"] == 0.0
        assert reset_metrics["command_motion_error_mm_s"] == 0.0
        assert not reset_metrics["collision_association_detected"]
        assert reset_metrics["max_collision_model_associations"] == 0

        _, reward, _, _, info = env.step(np.ones(2, dtype=np.float32))
        assert np.isfinite(reward)
        np.testing.assert_allclose(info["raw_action"], [50.0, 3.14], rtol=1e-5)
        assert_safety_metrics(info)
        assert_safety_cost(env, info)
        assert info["safety_metrics"]["requested_translation_speed_mm_s"] == 50.0
        np.testing.assert_allclose(
            info["safety_metrics"]["insertion_speed_mm_s"], 50.0, atol=1e-10
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
        assert reset_info["safety_metrics"]["insertion_speed_mm_s"] == 0.0
        _, _, _, _, blocked_info = env.step(
            np.asarray([-1.0, 0.0], dtype=np.float32)
        )
        np.testing.assert_allclose(
            blocked_info["safety_metrics"]["insertion_speed_mm_s"],
            0.0,
            atol=1e-5,
        )
        assert (
            blocked_info["safety_metrics"]["requested_translation_speed_mm_s"]
            == -50.0
        )
        np.testing.assert_allclose(
            blocked_info["safety_metrics"]["command_motion_error_mm_s"],
            50.0,
            atol=1e-5,
        )
        assert_safety_cost(env, blocked_info)
        blocked_cost = dict(
            zip(SAFETY_COST_NAMES, blocked_info["safety_cost"])
        )
        np.testing.assert_allclose(
            blocked_cost["normalized_command_motion_error"], 1.0, atol=1e-5
        )

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
                info["raw_action"][0],
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
        replay = EpisodeReplayBuffer(
            100,
            14,
            2,
            3,
            2,
            safety_cost_names=SAFETY_COST_NAMES,
            seed=7,
        )
        replay.add_episode(
            collected_observations,
            collected_actions,
            collected_rewards,
            collected_terminated,
            safety_cost=collected_safety_costs,
        )
        stored_episode = replay.state_dict()["episodes"][0]
        np.testing.assert_array_equal(
            stored_episode["safety_cost"],
            np.asarray(collected_safety_costs, dtype=np.float32),
        )
        sampled_batch = replay.sample(torch.device("cpu"))
        assert sampled_batch[2].shape == (3, 2, 1)
        assert sampled_batch[4].shape == (
            3,
            2,
            len(SAFETY_COST_NAMES),
        )
        assert sampled_batch[2].shape[:2] == sampled_batch[4].shape[:2]
        assert torch.all(torch.isfinite(sampled_batch[4]))
        assert torch.all(sampled_batch[4] >= 0.0)

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
            "transitions, finite (14,) observations, normalized actions, "
            "and a complete 200-step episode."
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
