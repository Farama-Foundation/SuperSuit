from gym.spaces import Box, Discrete
import numpy as np
from .dummy_NaN_env import DummyNaNEnv
from supersuit import (
    nan_noop_wrapper
    nan_zeros_wrapper
    nan_random_wrapper
)
import pytest

base_obs = {"a{}".format(idx): np.zeros([8, 8, 3], dtype=np.float32) + np.arange(3) + idx for idx in range(2)}
base_obs_space = {"a{}".format(idx): Box(low=np.float32(0.0), high=np.float32(10.0), shape=[8, 8, 3]) for idx in range(2)}
base_act_spaces = {"a{}".format(idx): Discrete(5) for idx in range(2)}

def test_NaN_noop_wrapper():
    base_env = DummyNaNEnv(base_obs, base_obs_space, base_act_spaces)
    wrapped_env = nan_noop_wrapper(base_env)
    wrapped_env.step(np.nan)

def test_NaN_zeros_wrapper():
    base_env = DummyNaNEnv(base_obs, base_obs_space, base_act_spaces)
    wrapped_env = nan_zeros_wrapper(base_env)
    wrapped_env.step(np.nan)

def test_NaN_random_wrapper():
    base_env = DummyNaNEnv(base_obs, base_obs_space, base_act_spaces)
    wrapped_env = nan_random_wrapper(base_env)
    wrapped_env.step(np.nan)
