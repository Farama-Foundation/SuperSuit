from skimage import measure
from gym.spaces import Box
import numpy as np
from . import convert_box


def check_param(obs_space, down_scale):
    assert isinstance(down_scale, tuple), "down_scale must be tuple. It is {}".format(down_scale)
    assert all(isinstance(ds, int) for ds in down_scale), "down_scale must be a tuple of int. It is {}".format(down_scale)
    assert len(obs_space.shape) == len(down_scale), "down scale must be of the same length as the observations of the agents"
    assert all(ds > 0 for ds in down_scale), "all down scale values must be greater than zero"

def change_obs_space(obs_space, param):
    return convert_box(lambda obs:change_observation(obs, obs_space, param), obs_space)

def change_observation(obs, obs_space, down_scale):
    mean = lambda x, axis: np.mean(x, axis=axis, dtype=obs_space.dtype)
    obs = measure.block_reduce(obs, block_size=down_scale, func=mean)
    return obs
