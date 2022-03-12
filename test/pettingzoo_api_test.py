import numpy as np
from pettingzoo.test import api_test, seed_test, parallel_test
from pettingzoo.mpe import simple_push_v2, simple_world_comm_v2
from pettingzoo.butterfly import knights_archers_zombies_v9, prison_v4
from pettingzoo.classic import chess_v5
from pettingzoo.magent import combined_arms_v5

import supersuit
import pytest
from supersuit import (
    frame_stack_v1,
    pad_action_space_v0,
    frame_skip_v0,
    sticky_actions_v0,
    dtype_v0
)


def test_pettingzoo_frame_stack():
    _env = simple_push_v2.env()
    wrapped_env = frame_stack_v1(_env)
    api_test(wrapped_env)


def test_pettingzoo_frame_skip():
    env = simple_push_v2.raw_env(max_cycles=100)
    env = frame_skip_v0(env, 3)
    env.reset()
    x = 0
    for agent in env.agent_iter(25):
        assert env.env.steps == (x // 2) * 3
        action = env.action_space(env.agent_selection).sample()
        next_obs = env.step(action)
        x += 1


def test_pettingzoo_pad_action_space():
    _env = simple_world_comm_v2.env()
    wrapped_env = pad_action_space_v0(_env)
    api_test(wrapped_env)
    seed_test(lambda: sticky_actions_v0(simple_world_comm_v2.env(), 0.5), 100)


def test_pettingzoo_parallel_env():
    _env = simple_world_comm_v2.parallel_env()
    wrapped_env = pad_action_space_v0(_env)
    parallel_test.parallel_api_test(wrapped_env)


wrappers = [
    supersuit.color_reduction_v0(knights_archers_zombies_v9.env(), "R"),
    supersuit.resize_v0(dtype_v0(knights_archers_zombies_v9.env(), np.uint8), x_size=5, y_size=10),
    supersuit.resize_v0(dtype_v0(knights_archers_zombies_v9.env(), np.uint8), x_size=5, y_size=10, linear_interp=True),
    supersuit.dtype_v0(knights_archers_zombies_v9.env(), np.int32),
    supersuit.flatten_v0(knights_archers_zombies_v9.env()),
    supersuit.reshape_v0(knights_archers_zombies_v9.env(), (512 * 512, 3)),
    supersuit.normalize_obs_v0(dtype_v0(knights_archers_zombies_v9.env(), np.float32), env_min=-1, env_max=5.0),
    supersuit.frame_stack_v1(combined_arms_v5.env(), 8),
    supersuit.pad_observations_v0(simple_world_comm_v2.env()),
    supersuit.pad_action_space_v0(simple_world_comm_v2.env()),
    supersuit.black_death_v3(combined_arms_v5.env()),
    supersuit.agent_indicator_v0(knights_archers_zombies_v9.env(), True),
    supersuit.agent_indicator_v0(knights_archers_zombies_v9.env(), False),
    supersuit.reward_lambda_v0(knights_archers_zombies_v9.env(), lambda x: x / 10),
    supersuit.clip_reward_v0(combined_arms_v5.env()),
    supersuit.clip_actions_v0(prison_v4.env(continuous=True)),
    supersuit.scale_actions_v0(prison_v4.env(continuous=True), 0.5),
    supersuit.nan_noop_v0(knights_archers_zombies_v9.env(), 0),
    supersuit.nan_zeros_v0(knights_archers_zombies_v9.env()),
    supersuit.nan_zeros_v0(prison_v4.env(continuous=True)),
    supersuit.nan_random_v0(chess_v5.env()),
    supersuit.nan_random_v0(knights_archers_zombies_v9.env()),
    supersuit.frame_skip_v0(combined_arms_v5.env(), 4),
    supersuit.sticky_actions_v0(combined_arms_v5.env(), 0.75),
    supersuit.delay_observations_v0(combined_arms_v5.env(), 3),
    supersuit.max_observation_v0(knights_archers_zombies_v9.env(), 3),
]


@pytest.mark.parametrize("env", wrappers)
def test_pettingzoo_aec_api(env):
    api_test(env)


parallel_wrappers = [
    supersuit.frame_stack_v1(combined_arms_v5.parallel_env(), 8),
    supersuit.frame_stack_v1(simple_push_v2.parallel_env(), 8),
    supersuit.reward_lambda_v0(combined_arms_v5.parallel_env(), lambda x: x / 10),
    supersuit.delay_observations_v0(combined_arms_v5.parallel_env(), 3),
    supersuit.delay_observations_v0(simple_push_v2.parallel_env(), 3),
    supersuit.dtype_v0(combined_arms_v5.parallel_env(), np.int32),
    supersuit.color_reduction_v0(knights_archers_zombies_v9.parallel_env(), "R"),
    supersuit.frame_skip_v0(combined_arms_v5.parallel_env(), 4),
    supersuit.frame_skip_v0(simple_push_v2.parallel_env(), 4),
    supersuit.max_observation_v0(combined_arms_v5.parallel_env(), 4),
    supersuit.black_death_v3(combined_arms_v5.parallel_env()),
    supersuit.black_death_v3(simple_push_v2.parallel_env()),
]


@pytest.mark.parametrize("env", parallel_wrappers)
def test_pettingzoo_parallel_api(env):
    parallel_test.parallel_api_test(env)
