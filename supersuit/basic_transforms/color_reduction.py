import warnings
import numpy as np
from gym.spaces import Box
from gym import spaces

def check_param(color_reduction):
    assert isinstance(color_reduction, str) or isinstance(color_reduction, dict), "color_reduction must be str or dict. It is {}".format(color_reduction)
    if isinstance(color_reduction, str):
        color_reduction = dict(zip(self.agents, [color_reduction for _ in enumerate(self.agents)]))
    if isinstance(color_reduction, dict):
        for agent in self.agents:
            assert agent in color_reduction.keys(), "Agent id {} is not a key of color_reduction {}".format(agent, color_reduction)
            assert color_reduction[agent] in COLOR_RED_LIST, "color_reduction must be in {}".format(COLOR_RED_LIST)
            assert len(self.observation_spaces[agent].low.shape) == 3, "To apply color_reduction, length of shape of obs space of the agent should be 3. It is {}".format(len(self.observation_spaces[agent].low.shape))
            if color_reduction[agent] == "full":
                warnings.warn("You have chosen true grayscaling. It might be too slow. Choose a specific channel for better performance")

def change_space(observation_spaces,color_reduction):
    for agent in self.agents:
        obs_space = self.observation_spaces[agent]
        dtype = obs_space.dtype
        color_reduction = self.color_reduction[agent]
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
            low = np.average(obs_space.low, weights=[0.299, 0.587, 0.114], axis=2).astype(obs_space.dtype)
            high = np.average(obs_space.high, weights=[0.299, 0.587, 0.114], axis=2).astype(obs_space.dtype)
        self.observation_spaces[agent] = Box(low=low, high=high, dtype=dtype)
    print("Mod obs space: color_reduction", self.observation_spaces)

def change_observation(obs,color_reduction):
    color_reduction = self.color_reduction[agent]
    if color_reduction == 'R':
        obs = obs[:, :, 0]
    if color_reduction == 'G':
        obs = obs[:, :, 1]
    if color_reduction == 'B':
        obs = obs[:, :, 2]
    if color_reduction == 'full':
        obs = np.average(obs, weights=[0.299, 0.587, 0.114], axis=2).astype(obs.dtype)
    return obs
