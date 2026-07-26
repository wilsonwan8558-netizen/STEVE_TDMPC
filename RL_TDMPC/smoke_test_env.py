"""Run the pre-training acceptance checks for the stEVE adapter."""

from __future__ import annotations

import numpy as np
from gymnasium.utils.env_checker import check_env

from envs.steve_env import StEVEEnv


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


def main() -> None:
    max_curvature, mean_curvature = StEVEEnv._curvature_metrics(
        np.asarray([[1, 0, 0], [0, 1, 0], [-1, 0, 0]], dtype=np.float64)
    )
    np.testing.assert_allclose([max_curvature, mean_curvature], [1.0, 1.0])
    assert StEVEEnv._curvature_metrics(np.zeros((3, 3))) == (0.0, 0.0)

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

        seeded_observation, _ = env.reset(seed=123)
        repeated_observation, _ = env.reset(seed=123)
        np.testing.assert_allclose(seeded_observation, repeated_observation, atol=1e-6)

        # Exercise the complete five-value step contract with random actions.
        env.reset(seed=7)
        rng = np.random.default_rng(7)
        for _ in range(10):
            result = env.step(
                rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
            )
            assert len(result) == 5
            observation, reward, terminated, truncated, info = result
            assert observation.shape == (14,)
            assert np.all(np.isfinite(observation))
            assert np.isfinite(reward)
            assert_safety_metrics(info)
            np.testing.assert_allclose(
                info["safety_metrics"]["requested_translation_speed_mm_s"],
                info["raw_action"][0],
            )
            if terminated or truncated:
                _, reset_info = env.reset()
                assert_safety_metrics(reset_info)

        env.reset(seed=123)
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
        assert steps == 200
        assert truncated

        # This invokes extra resets/steps and checks the full Gymnasium contract.
        check_env(env, skip_render_check=True)
        print(
            "PASS: reset/step, safety metrics, 10 random actions, finite (14,) "
            "observations, normalized actions, and a complete 200-step episode."
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
