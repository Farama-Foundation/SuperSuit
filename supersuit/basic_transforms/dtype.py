import numpy as np
from gym.spaces import Box
from . import convert_box


def check_param(obs_space, new_dtype):
    assert isinstance(new_dtype, type) or isinstance(new_dtype, np.dtype), "new_dtype must be type. It is {}".format(new_dtype)

def change_obs_space(obs_space, param):
    return convert_box(lambda obs:change_observation(obs, obs_space, param), obs_space)

def change_observation(obs, obs_space,new_dtype):
    obs = obs.astype(new_dtype)
    return obs
