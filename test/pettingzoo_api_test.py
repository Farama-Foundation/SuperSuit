import numpy as np
import pytest
from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.classic import connect_four_v3
from pettingzoo.mpe import simple_push_v3, simple_world_comm_v3
from pettingzoo.test import api_test, parallel_api_test, seed_test

import supersuit
from supersuit import (
    dtype_v0,
    frame_skip_v0,
    frame_stack_v2,
    pad_action_space_v0,
    sticky_actions_v0,
)
from supersuit.utils.basic_transforms import convert_box

BUTTERFLY_MPE_CLASSIC = [knights_archers_zombies_v10, simple_push_v3, connect_four_v3]
BUTTERFLY_MPE = [knights_archers_zombies_v10, simple_push_v3]


@pytest.mark.parametrize("env_fn", [simple_push_v3, simple_world_comm_v3])
def test_frame_stack(env_fn):
    _env = env_fn.env()
    wrapped_env = frame_stack_v2(_env)
    api_test(wrapped_env)


@pytest.mark.parametrize("env_fn", [simple_push_v3, simple_world_comm_v3])
def test_frame_stack_parallel(env_fn):
    _env = env_fn.parallel_env()
    wrapped_env = frame_stack_v2(_env)
    parallel_api_test(wrapped_env)


@pytest.mark.parametrize("env_fn", [simple_push_v3])
def test_frame_skip(env_fn):
    env = env_fn.raw_env(max_cycles=100)
    env = frame_skip_v0(env, 3)
    env.reset()
    x = 0
    for _ in env.agent_iter(25):
        assert env.unwrapped.steps == (x // 2) * 3
        action = env.action_space(env.agent_selection).sample()
        env.step(action)
        x += 1


@pytest.mark.parametrize("env_fn", [simple_push_v3])
def test_frame_skip_parallel(env_fn):
    env = env_fn.parallel_env(max_cycles=100)
    env = frame_skip_v0(env, 3)
    env.reset()
    x = 0
    while env.agents:
        assert env.unwrapped.steps == (x // 2) * 3
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        env.step(actions)
        x += env.num_agents


@pytest.mark.parametrize("env_fn", [simple_world_comm_v3, connect_four_v3])
def test_pad_action_space(env_fn):
    _env = simple_world_comm_v3.env()
    wrapped_env = pad_action_space_v0(_env)
    api_test(wrapped_env)
    seed_test(lambda: sticky_actions_v0(simple_world_comm_v3.env(), 0.5), 100)


@pytest.mark.parametrize("env_fn", [simple_world_comm_v3])
def test_pad_action_space_parallel(env_fn):
    _env = env_fn.parallel_env()
    wrapped_env = pad_action_space_v0(_env)
    parallel_api_test(wrapped_env)


@pytest.mark.parametrize("env_fn", [knights_archers_zombies_v10])
def test_color_reduction(env_fn):
    env = supersuit.color_reduction_v0(env_fn.env(vector_state=False), "R")
    api_test(env)


@pytest.mark.parametrize("env_fn", [knights_archers_zombies_v10])
def test_color_reduction_parallel(env_fn):
    env = supersuit.color_reduction_v0(env_fn.parallel_env(vector_state=False), "R")
    parallel_api_test(env)


@pytest.mark.parametrize("env_fn", [knights_archers_zombies_v10])
@pytest.mark.parametrize(
    "wrapper_kwargs",
    [dict(x_size=5, y_size=10), dict(x_size=5, y_size=10, linear_interp=True)],
)
def test_resize_dtype(env_fn, wrapper_kwargs):
    env = supersuit.resize_v1(
        dtype_v0(env_fn.env(vector_state=False), np.uint8), **wrapper_kwargs
    )
    api_test(env)


@pytest.mark.parametrize("env_fn", [knights_archers_zombies_v10])
@pytest.mark.parametrize(
    "wrapper_kwargs",
    [dict(x_size=5, y_size=10), dict(x_size=5, y_size=10, linear_interp=True)],
)
def test_resize_dtype_parallel(env_fn, wrapper_kwargs):
    env = supersuit.resize_v1(
        dtype_v0(env_fn.parallel_env(vector_state=False), np.uint8), **wrapper_kwargs
    )
    parallel_api_test(env)


@pytest.mark.parametrize("env_fn", [knights_archers_zombies_v10])
def test_dtype(env_fn):
    env = supersuit.dtype_v0(env_fn.env(), np.int32)
    api_test(env)


@pytest.mark.parametrize("env_fn", [knights_archers_zombies_v10])
def test_dtype_parallel(env_fn):
    env = supersuit.dtype_v0(env_fn.parallel_env(), np.int32)
    parallel_api_test(env)


@pytest.mark.parametrize("env_fn", BUTTERFLY_MPE_CLASSIC)
def test_flatten(env_fn):
    env = supersuit.flatten_v0(knights_archers_zombies_v10.env())
    api_test(env)


# Classic environments don't have parallel envs so this doesn't apply
@pytest.mark.parametrize("env_fn", BUTTERFLY_MPE)
def test_flatten_parallel(env_fn):
    env = supersuit.flatten_v0(env_fn.parallel_env())
    parallel_api_test(env)


@pytest.mark.parametrize("env_fn", [knights_archers_zombies_v10])
def test_reshape(env_fn):
    env = supersuit.reshape_v0(env_fn.env(vector_state=False), (512 * 512, 3))
    api_test(env)


@pytest.mark.parametrize("env_fn", [knights_archers_zombies_v10])
def test_reshape_parallel(env_fn):
    env = supersuit.reshape_v0(env_fn.parallel_env(vector_state=False), (512 * 512, 3))
    parallel_api_test(env)


# MPE environment has infinite bounds for observation space (only environments with finite bounds can be passed to normalize_obs)
@pytest.mark.parametrize("env_fn", [knights_archers_zombies_v10])
def test_normalize_obs(env_fn):
    env = supersuit.normalize_obs_v0(
        dtype_v0(env_fn.env(), np.float32), env_min=-1, env_max=5.0
    )
    api_test(env)


@pytest.mark.parametrize("env_fn", [knights_archers_zombies_v10])
def test_normalize_obs_parallel(env_fn):
    env = supersuit.normalize_obs_v0(
        dtype_v0(env_fn.parallel_env(), np.float32), env_min=-1, env_max=5.0
    )
    parallel_api_test(env)


@pytest.mark.parametrize("env_fn", BUTTERFLY_MPE_CLASSIC)
def test_pad_observations(env_fn):
    env = supersuit.pad_observations_v0(simple_world_comm_v3.env())
    api_test(env)


@pytest.mark.parametrize("env_fn", BUTTERFLY_MPE_CLASSIC)
def test_pad_observations_parallel(env_fn):
    env = supersuit.pad_observations_v0(simple_world_comm_v3.parallel_env())
    parallel_api_test(env)


@pytest.mark.skip(
    reason="Black death wrapper is only designed for parallel envs, AEC envs should simply skip the agent by setting env.agent_selection manually"
)
@pytest.mark.parametrize("env_fn", BUTTERFLY_MPE)
def test_black_death(env_fn):
    env = supersuit.black_death_v3(env_fn.env())
    api_test(env)


@pytest.mark.parametrize("env_fn", BUTTERFLY_MPE)
def test_black_death_parallel(env_fn):
    env = supersuit.black_death_v3(env_fn.parallel_env())
    parallel_api_test(env)


@pytest.mark.parametrize("env_fn", [knights_archers_zombies_v10])
@pytest.mark.parametrize("env_kwargs", [dict(type_only=True), dict(type_only=False)])
def test_agent_indicator(env_fn, env_kwargs):
    env = supersuit.agent_indicator_v0(env_fn.env(), **env_kwargs)
    api_test(env)


@pytest.mark.parametrize("env_fn", [knights_archers_zombies_v10])
@pytest.mark.parametrize("env_kwargs", [dict(type_only=True), dict(type_only=False)])
def test_agent_indicator_parallel(env_fn, env_kwargs):
    env = supersuit.agent_indicator_v0(env_fn.parallel_env(), **env_kwargs)
    parallel_api_test(env)


@pytest.mark.parametrize("env_fn", BUTTERFLY_MPE_CLASSIC)
def test_reward_lambda(env_fn):
    env = supersuit.reward_lambda_v0(env_fn.env(), lambda x: x / 10)
    api_test(env)


@pytest.mark.parametrize("env_fn", BUTTERFLY_MPE)
def test_reward_lambda_parallel(env_fn):
    env = supersuit.reward_lambda_v0(env_fn.parallel_env(), lambda x: x / 10)
    parallel_api_test(env)


@pytest.mark.parametrize("env_fn", BUTTERFLY_MPE)
def test_observation_lambda(env_fn):
    env = supersuit.observation_lambda_v0(env_fn.env(), lambda obs, obs_space: obs - 1)
    api_test(env)


# Example using observation lambda with an action masked environment (flattening the obs space while keeping action mask)
@pytest.mark.parametrize("env_fn", [connect_four_v3])
def test_observation_lambda_action_mask(env_fn):
    env = env_fn.env()
    env.reset()
    obs = env.observe(env.possible_agents[0])

    # Example: reshape the observation to flatten the first two dimensions: (6, 7, 2) -> (42, 2)
    newshape = obs["observation"].reshape((-1, 2)).shape

    def change_obs_space_fn(obs_space):
        obs_space["observation"] = convert_box(
            lambda obs: obs.reshape(newshape), old_box=obs_space["observation"]
        )
        return obs_space

    def change_observation_fn(observation, old_obs_space):
        # Reshape observation
        observation["observation"] = observation["observation"].reshape(newshape)
        # Invert the action mask (make illegal actions legal, and vice versa)
        observation["action_mask"] = 1 - observation["action_mask"]
        return observation

    env = supersuit.observation_lambda_v0(
        env_fn.env(),
        change_obs_space_fn=change_obs_space_fn,
        change_observation_fn=change_observation_fn,
    )
    api_test(env)


@pytest.mark.parametrize("env_fn", BUTTERFLY_MPE)
def test_observation_lambda_parallel(env_fn):
    pass


@pytest.mark.parametrize("env_fn", BUTTERFLY_MPE_CLASSIC)
def test_clip_reward(env_fn):
    env = supersuit.clip_reward_v0(env_fn.env())
    api_test(env)


@pytest.mark.parametrize("env_fn", BUTTERFLY_MPE)
def test_clip_reward_parallel(env_fn):
    env = supersuit.clip_reward_v0(env_fn.parallel_env())
    parallel_api_test(env)


@pytest.mark.parametrize("env_fn", BUTTERFLY_MPE_CLASSIC)
def test_nan_noop(env_fn):
    env = supersuit.nan_noop_v0(env_fn.env(), 0)
    api_test(env)


@pytest.mark.parametrize("env_fn", BUTTERFLY_MPE)
def test_nan_noop_parallel(env_fn):
    env = supersuit.nan_noop_v0(env_fn.parallel_env(), 0)
    parallel_api_test(env)


@pytest.mark.parametrize("env_fn", BUTTERFLY_MPE_CLASSIC)
def test_nan_zeros(env_fn):
    env = supersuit.nan_zeros_v0(env_fn.env())
    api_test(env)


@pytest.mark.parametrize("env_fn", BUTTERFLY_MPE)
def test_nan_zeros_parallel(env_fn):
    env = supersuit.nan_zeros_v0(env_fn.parallel_env())
    parallel_api_test(env)


@pytest.mark.parametrize("env_fn", BUTTERFLY_MPE_CLASSIC)
def test_nan_random(env_fn):
    env = supersuit.nan_random_v0(env_fn.env())
    api_test(env)


@pytest.mark.parametrize("env_fn", BUTTERFLY_MPE)
def test_nan_random_parallel(env_fn):
    env = supersuit.nan_random_v0(env_fn.parallel_env())
    parallel_api_test(env)


@pytest.mark.parametrize("env_fn", BUTTERFLY_MPE_CLASSIC)
def test_sticky_actions(env_fn):
    env = supersuit.sticky_actions_v0(env_fn.env(), 0.75)
    api_test(env)


@pytest.mark.parametrize("env_fn", BUTTERFLY_MPE)
def test_sticky_actions_parallel(env_fn):
    env = supersuit.sticky_actions_v0(env_fn.parallel_env(), 0.75)
    parallel_api_test(env)


@pytest.mark.parametrize("env_fn", [connect_four_v3])
def test_delay_observations(env_fn):
    env = supersuit.delay_observations_v0(env_fn.env(), 3)
    api_test(env)


@pytest.mark.parametrize("env_fn", BUTTERFLY_MPE)
def test_delay_observations_parallel(env_fn):
    env = supersuit.delay_observations_v0(env_fn.parallel_env(), 3)
    parallel_api_test(env)


@pytest.mark.parametrize("env_fn", BUTTERFLY_MPE_CLASSIC)
def test_max_observation(env_fn):
    env = supersuit.max_observation_v0(knights_archers_zombies_v10.env(), 3)
    api_test(env)


@pytest.mark.parametrize("env_fn", BUTTERFLY_MPE)
def test_max_observation_parallel(env_fn):
    env = supersuit.max_observation_v0(knights_archers_zombies_v10.parallel_env(), 3)
    parallel_api_test(env)
