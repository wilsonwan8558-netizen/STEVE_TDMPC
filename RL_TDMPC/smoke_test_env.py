"""Run the pre-training acceptance checks for the stEVE adapter."""

from __future__ import annotations

import numpy as np
from gymnasium.utils.env_checker import check_env

from envs.steve_env import StEVEEnv


def main() -> None:
    env = StEVEEnv()
    try:
        observation, _ = env.reset(seed=1)
        assert observation.shape == (14,)
        assert observation.dtype == np.float32
        assert np.all(np.isfinite(observation))
        assert env.observation_space.contains(observation)
        assert env.action_space.shape == (2,)
        assert np.all(env.action_space.low == -1)
        assert np.all(env.action_space.high == 1)

        _, reward, _, _, info = env.step(np.ones(2, dtype=np.float32))
        assert np.isfinite(reward)
        np.testing.assert_allclose(info["raw_action"], [50.0, 3.14], rtol=1e-5)

        # A reset must restore the physical device instead of leaking state across episodes.
        env.reset(seed=2)
        np.testing.assert_allclose(
            env.intervention.device_lengths_inserted, [0.0], atol=1e-6
        )

        seeded_observation, _ = env.reset(seed=123)
        repeated_observation, _ = env.reset(seed=123)
        np.testing.assert_allclose(seeded_observation, repeated_observation, atol=1e-6)

        steps = 0
        terminated = truncated = False
        while not (terminated or truncated):
            observation, reward, terminated, truncated, _ = env.step(
                np.zeros(2, dtype=np.float32)
            )
            steps += 1
            assert observation.shape == (14,)
            assert np.all(np.isfinite(observation))
            assert np.isfinite(reward)
        assert steps == 200
        assert truncated

        # This invokes extra resets/steps and checks the full Gymnasium contract.
        check_env(env, skip_render_check=True)
        print(
            "PASS: reset/step, finite (14,) observations, normalized actions, "
            "and a complete 200-step episode."
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
