import gymnasium as gym
import numpy as np
import pytest
from gymnasium.wrappers import NormalizeObservation
from pettingzoo.butterfly import pistonball_v6

import supersuit as ss


@pytest.mark.parametrize("env_fn", [pistonball_v6])
def test_vec_env_normalize_obs(env_fn):
    env = env_fn.parallel_env()
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 10, base_class="gymnasium")
    obs, info = env.reset()

    # Create a "dummy" class that adds gym.Env as a base class of the env
    # to satisfy the assertion.
    class PatchedEnv(env.__class__, gym.Env):
        pass

    env.__class__ = PatchedEnv

    env = NormalizeObservation(env)
    normalized_obs, normalized_info = env.reset()

    obs_range = np.amax(obs) - np.amin(obs)
    normalized_obs_range = np.amax(normalized_obs) - np.amin(normalized_obs)
    assert obs_range > 1, "Regular observation space should be greater than 1."
    assert (
        normalized_obs_range < 1.0e-4
    ), "Normalized observation space should be smaller than 1.0e-4."
    assert (
        obs_range > normalized_obs_range
    ), "Normalized observation space has more range than regular observation space."
