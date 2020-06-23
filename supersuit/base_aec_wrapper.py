import numpy as np
import copy
from gym.spaces import Box
from gym import spaces
import warnings
from skimage import measure
from pettingzoo import AECEnv


class BaseWrapper(AECEnv):

    metadata = {'render.modes': ['human']}

    def __init__(self, env):
        '''
        Creates a wrapper around `env`. Extend this class to create changes to the space.
        '''
        super().__init__()
        self.env = env

        self.num_agents = self.env.num_agents
        self.agents = self.env.agents
        self.observation_spaces = copy.copy(self.env.observation_spaces)
        self.action_spaces = copy.copy(self.env.action_spaces)

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

        self.agent_selection = self.env.agent_selection
        self.rewards = self.env.rewards
        self.dones = self.env.dones
        self.infos = self.env.infos

        self._update_step(agent,observation)
        if observe:
            observation = self._modify_observation(agent,observation)
            return observation
        else:
            return None

    def observe(self, agent):
        obs = self.env.observe(agent)
        observation = self._modify_observation(agent, obs)
        return observation

    def step(self, action, observe=True):
        agent = self.env.agent_selection
        cur_act_space = self.action_spaces[agent]
        assert not isinstance(cur_act_space,Box) or cur_act_space.shape == action.shape, "the shape of the action {} is not equal to the shape of the action space {}".format(action.shape,cur_act_space.shape)
        action = self._modify_action(agent, action)
        next_obs = self.env.step(action, observe=observe)
        new_agent = self.env.agent_selection

        self.agent_selection = self.env.agent_selection
        self.rewards = self.env.rewards
        self.dones = self.env.dones
        self.infos = self.env.infos
        self._update_step(new_agent,next_obs)

        if observe:
            next_obs = self._modify_observation(new_agent,next_obs)
            return next_obs
        else:
            return None
