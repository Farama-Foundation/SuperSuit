import numpy as np
from gym.spaces import Box
from . import convert_box


def check_param(obs_space, should_flatten):
    assert isinstance(should_flatten, bool), "should_flatten must be bool. It is {}".format(should_flatten)

def change_obs_space(obs_space, param):
    return convert_box(lambda obs:change_observation(obs, obs_space, param), obs_space)

def change_observation(obs, obs_space,should_flatten):
    if should_flatten:
        obs = obs.flatten()
    return obs
