from pettingzoo.tests import api_test, seed_test, error_tests, parallel_test
from pettingzoo.mpe import simple_push_v2, simple_world_comm_v2
# from pettingzoo.sisl import multiwalker

from supersuit import (
    frame_stack_v1,
    pad_action_space_v0,
    frame_skip_v0,
    sticky_actions_v0,
)
import numpy as np


def test_pettinzoo_frame_stack():
    _env = simple_push_v2.env()
    wrapped_env = frame_stack_v1(_env)
    api_test.api_test(wrapped_env)


def test_pettinzoo_frame_skip():
    env = simple_push_v2.raw_env(max_cycles=100)
    env = frame_skip_v0(env, 3)
    env.reset()
    x = 0
    for agent in env.agent_iter(25):
        assert env.env.steps == (x // 2) * 3
        action = env.action_spaces[env.agent_selection].sample()
        next_obs = env.step(action)
        x += 1


def test_pettinzoo_pad_action_space():
    _env = simple_world_comm_v2.env()
    wrapped_env = pad_action_space_v0(_env)
    api_test.api_test(wrapped_env)
    seed_test.seed_test(lambda: sticky_actions_v0(simple_world_comm_v2.env(), 0.5))


def test_pettingzoo_parallel_env():
    _env = simple_world_comm_v2.parallel_env()
    wrapped_env = pad_action_space_v0(_env)
    parallel_test.parallel_play_test(wrapped_env)
