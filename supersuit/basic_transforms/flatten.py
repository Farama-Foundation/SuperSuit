import numpy as np
from gym.spaces import Box

def check_param(obs_space, should_flatten):
    assert isinstance(should_flatten, bool), "should_flatten must be bool. It is {}".format(should_flatten)

def change_observation(obs,should_flatten):
    if should_flatten:
        obs = obs.flatten()
    return obs
