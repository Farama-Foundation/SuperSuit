import warnings
import numpy as np
from gym.spaces import Box
from gym import spaces
from . import convert_box

COLOR_RED_LIST = ["full", 'R', 'G', 'B']
GRAYSCALE_WEIGHTS = [0.299, 0.587, 0.114]

def check_param(space,color_reduction):
    assert isinstance(color_reduction, str), "color_reduction must be str. It is {}".format(color_reduction)
    assert color_reduction in COLOR_RED_LIST, "color_reduction must be in {}".format(COLOR_RED_LIST)
    assert len(space.low.shape) == 3 and space.low.shape[2] == 3, "To apply color_reduction, shape must be a 3d image with last dimention of size 3. Shape is {}".format((space.low.shape))
    if color_reduction == "full":
        warnings.warn("You have chosen true grayscaling. It might be too slow. Choose a specific channel for better performance")

def change_obs_space(obs_space, param):
    return convert_box(lambda obs:change_observation(obs, obs_space, param), obs_space)

def change_observation(obs,  obs_space,color_reduction):
    if color_reduction == 'R':
        obs = obs[:, :, 0]
    if color_reduction == 'G':
        obs = obs[:, :, 1]
    if color_reduction == 'B':
        obs = obs[:, :, 2]
    if color_reduction == 'full':
        obs = np.average(obs, weights=GRAYSCALE_WEIGHTS, axis=2).astype(obs.dtype)
    return obs
