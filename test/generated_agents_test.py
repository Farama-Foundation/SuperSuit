import numpy as np
from pettingzoo.test import api_test, seed_test, parallel_test
from .example_envs import generated_agents_parallel_v0, generated_agents_env_v0

import supersuit
from supersuit import dtype_v0
import pytest


wrappers = [
    supersuit.dtype_v0(generated_agents_parallel_v0.env(), np.int32),
    supersuit.flatten_v0(generated_agents_parallel_v0.env()),
    supersuit.normalize_obs_v0(dtype_v0(generated_agents_parallel_v0.env(), np.float32), env_min=-1, env_max=5.0),
    supersuit.frame_stack_v1(generated_agents_parallel_v0.env(), 8),
    supersuit.reward_lambda_v0(generated_agents_parallel_v0.env(), lambda x: x / 10),
    supersuit.clip_reward_v0(generated_agents_parallel_v0.env()),
    supersuit.nan_noop_v0(generated_agents_parallel_v0.env(), 0),
    supersuit.nan_zeros_v0(generated_agents_parallel_v0.env()),
    supersuit.nan_random_v0(generated_agents_parallel_v0.env()),
    supersuit.frame_skip_v0(generated_agents_parallel_v0.env(), 4),
    supersuit.sticky_actions_v0(generated_agents_parallel_v0.env(), 0.75),
    supersuit.delay_observations_v0(generated_agents_parallel_v0.env(), 3),
    supersuit.max_observation_v0(generated_agents_parallel_v0.env(), 3),
]


@pytest.mark.parametrize("env", wrappers)
def test_pettingzoo_aec_api_par_gen(env):
    api_test(env)


wrappers = [
    supersuit.dtype_v0(generated_agents_env_v0.env(), np.int32),
    supersuit.flatten_v0(generated_agents_env_v0.env()),
    supersuit.normalize_obs_v0(dtype_v0(generated_agents_env_v0.env(), np.float32), env_min=-1, env_max=5.0),
    supersuit.frame_stack_v1(generated_agents_env_v0.env(), 8),
    supersuit.reward_lambda_v0(generated_agents_env_v0.env(), lambda x: x / 10),
    supersuit.clip_reward_v0(generated_agents_env_v0.env()),
    supersuit.nan_noop_v0(generated_agents_env_v0.env(), 0),
    supersuit.nan_zeros_v0(generated_agents_env_v0.env()),
    supersuit.nan_random_v0(generated_agents_env_v0.env()),
    supersuit.frame_skip_v0(generated_agents_env_v0.env(), 4),
    supersuit.sticky_actions_v0(generated_agents_env_v0.env(), 0.75),
    supersuit.delay_observations_v0(generated_agents_env_v0.env(), 3),
    supersuit.max_observation_v0(generated_agents_env_v0.env(), 3),
]


@pytest.mark.parametrize("env", wrappers)
def test_pettingzoo_aec_api_aec_gen(env):
    api_test(env)


parallel_wrappers = wrappers = [
    supersuit.dtype_v0(generated_agents_parallel_v0.parallel_env(), np.int32),
    supersuit.flatten_v0(generated_agents_parallel_v0.parallel_env()),
    supersuit.normalize_obs_v0(dtype_v0(generated_agents_parallel_v0.parallel_env(), np.float32), env_min=-1, env_max=5.0),
    supersuit.frame_stack_v1(generated_agents_parallel_v0.parallel_env(), 8),
    supersuit.reward_lambda_v0(generated_agents_parallel_v0.parallel_env(), lambda x: x / 10),
    supersuit.clip_reward_v0(generated_agents_parallel_v0.parallel_env()),
    supersuit.nan_noop_v0(generated_agents_parallel_v0.parallel_env(), 0),
    supersuit.nan_zeros_v0(generated_agents_parallel_v0.parallel_env()),
    supersuit.nan_random_v0(generated_agents_parallel_v0.parallel_env()),
    supersuit.frame_skip_v0(generated_agents_parallel_v0.parallel_env(), 4),
    supersuit.sticky_actions_v0(generated_agents_parallel_v0.parallel_env(), 0.75),
    supersuit.delay_observations_v0(generated_agents_parallel_v0.parallel_env(), 3),
    supersuit.max_observation_v0(generated_agents_parallel_v0.parallel_env(), 3),
]

@pytest.mark.parametrize("env", parallel_wrappers)
def test_pettingzoo_parallel_api_gen(env):
    parallel_test.parallel_api_test(env)
