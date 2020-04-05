import warnings
import numpy as np
from gym.spaces import Box
from gym import spaces

COLOR_RED_LIST = ["full", 'R', 'G', 'B']
GRAYSCALE_WEIGHTS = [0.299, 0.587, 0.114]

def check_param(observation_spaces,color_reduction):
    assert isinstance(color_reduction, str), "color_reduction must be str. It is {}".format(color_reduction)
    assert color_reduction in COLOR_RED_LIST, "color_reduction must be in {}".format(COLOR_RED_LIST)
    for space in observation_spaces.values():
        assert len(space.low.shape) == 3 and space.low.shape[2] == 3, "To apply color_reduction, shape must be a 3d image with last dimention of size 3. Shape is {}".format((space.low.shape))
    if color_reduction == "full":
        warnings.warn("You have chosen true grayscaling. It might be too slow. Choose a specific channel for better performance")

def change_space(observation_spaces, color_reduction):
    new_obs_spaces = {}
    for agent in observation_spaces.keys():
        obs_space = observation_spaces[agent]
        dtype = obs_space.dtype
        if color_reduction == 'R':
            low = obs_space.low[:, :, 0]
            high = obs_space.high[:, :, 0]
        if color_reduction == 'G':
            low = obs_space.low[:, :, 1]
            high = obs_space.high[:, :, 1]
        if color_reduction == 'B':
            low = obs_space.low[:, :, 2]
            high = obs_space.high[:, :, 2]
        if color_reduction == 'full':
            low = np.average(obs_space.low, weights=GRAYSCALE_WEIGHTS, axis=2).astype(obs_space.dtype)
            high = np.average(obs_space.high, weights=GRAYSCALE_WEIGHTS, axis=2).astype(obs_space.dtype)
        new_obs_spaces[agent] = Box(low=low, high=high, dtype=dtype)
    return new_obs_spaces

def change_observation(obs, color_reduction):
    if color_reduction == 'R':
        obs = obs[:, :, 0]
    if color_reduction == 'G':
        obs = obs[:, :, 1]
    if color_reduction == 'B':
        obs = obs[:, :, 2]
    if color_reduction == 'full':
        obs = np.average(obs, weights=GRAYSCALE_WEIGHTS, axis=2).astype(obs.dtype)
    return obs
