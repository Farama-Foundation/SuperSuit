from skimage import measure
from gym.spaces import Box
import numpy as np

def check_param(observation_spaces, down_scale):
    assert isinstance(down_scale, tuple), "down_scale must be tuple. It is {}".format(down_scale)
    assert all(isinstance(ds, int) for ds in down_scale), "down_scale must be a tuple of int. It is {}".format(down_scale)
    for obs_space in observation_spaces.values():
        assert len(obs_space.shape) == len(down_scale), "down scale must be of the same length as the observations of the agents"

def change_space(observation_spaces,down_scale):
    new_spaces = {}
    for agent in observation_spaces.keys():
        obs_space = observation_spaces[agent]
        dtype = obs_space.dtype
        shape = obs_space.shape
        new_shape = tuple([int(shape[i] / down_scale[i]) for i in range(len(shape))])
        low = obs_space.low.flatten()[:np.product(new_shape)].reshape(new_shape)
        high = obs_space.high.flatten()[:np.product(new_shape)].reshape(new_shape)
        new_spaces[agent] = Box(low=low, high=high, dtype=dtype)
    print("Mod obs space: down_scale", new_spaces)
    return new_spaces

def change_observation(obs,down_scale):
    down_scale = self.down_scale[agent]
    mean = lambda x, axis: np.mean(x, axis=axis, dtype=np.uint8)
    obs = measure.block_reduce(obs, block_size=down_scale, func=mean)
    return obs
