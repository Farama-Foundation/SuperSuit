from gym.spaces import Box, Discrete
import numpy as np
from .dummy_scale_action_env import DummyScaleEnv
from supersuit import (
    scale_actions_wrapper
)
import pytest

base_obs = {"a{}".format(idx): np.zeros([8, 8, 3], dtype=np.float32) + np.arange(3) + idx for idx in range(2)}
base_obs_space = {"a{}".format(idx): Box(low=np.float32(0.0), high=np.float32(10.0), shape=[8, 8, 3]) for idx in range(2)}
base_act_spaces = {"a{}".format(idx): Discrete(5) for idx in range(2)}

def test_scale_action_wrapper():
    base_env = DummyScaleEnv(base_obs, base_obs_space, base_act_spaces)
    wrapped_env = scale_actions_wrapper(base_env, 2)
    scaled_action = wrapped_env.step(2)
    assert scaled_action == 4