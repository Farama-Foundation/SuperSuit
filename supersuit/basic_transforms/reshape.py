import numpy as np
from gym.spaces import Box

def check_param(obs_space, shape):
    assert isinstance(shape, tuple), "shape must be tuple. It is {}".format(shape)
    assert all(isinstance(el,int) for el in shape), "shape must be tuple of ints, is: {}".format(shape)
    assert np.prod(shape) == np.prod(obs_space.shape), "new shape {} must have as many elements as original shape {}".format(shape,obs_space.shape)

def change_observation(obs, shape):
    obs = obs.reshape(shape)
    return obs
