import numpy as np
import copy
from gym.spaces import Box
from gym import spaces
import warnings
from skimage import measure
from pettingzoo import AECEnv

from .frame_stack import stack_obs_space, stack_obs

COLOR_RED_LIST = ["full", 'R', 'G', 'B']
OBS_RESHAPE_LIST = ["expand", "flatten"]


class BaseWrapper(AECEnv):
    def __init__(self, env):
        '''
        Creates a wrapper around `env`. Extend this class to create changes to the space.
        '''
        super().__init__()
        self.env = env

        self.agents = self.env.agents
        self.agent_selection = self.env.agent_selection
        self.observation_spaces = self.env.observation_spaces
        self.action_spaces = copy.copy(self.env.action_spaces)
        self.orig_action_spaces = self.env.action_spaces

        self.rewards = self.env.rewards
        self.dones = self.env.dones
        self.infos = self.env.infos

        self.agent_order = self.env.agent_order

        self._check_wrapper_params()

        self._modify_spaces()


    def _check_wrapper_params(self):
        raise NotImplementedError()

    def _modify_spaces(self):
        raise NotImplementedError()

    def _modify_action(self, agent, action):
        raise NotImplementedError()

    def _modify_observation(self, agent, observation):
        raise NotImplementedError()

    def _update_step(self, agent, observation):
        raise NotImplementedError()

    def close(self):
        self.env.close()

    def render(self, mode='human'):
        self.env.render(mode)

    def reset(self, observe=True):
        observation = self.env.reset(observe)
        agent = self.env.agent_selection

        observation = self._modify_observation(agent,observation)
        self._update_step(agent,observation)
        return observation

    def observe(self, agent):
        obs = self.env.observe(agent)
        observation = self._modify_observation(agent, obs)
        return observation

    def step(self, action, observe=True):
        agent = self.env.agent_selection
        action = self._modify_action(agent, action)

        next_obs = self.env.step(action, observe=observe)

        new_agent = self.env.agent_selection
        self._update_step(new_agent,next_obs)
        next_obs = self._modify_observation(new_agent,next_obs)

        self.agent_selection = self.env.agent_selection
        self.rewards = self.env.rewards
        self.dones = self.env.dones
        self.infos = self.env.infos

        return next_obs
