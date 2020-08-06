import numpy as np
import copy
from gym.spaces import Box
from gym import spaces
import warnings
from pettingzoo.utils.wrappers import AgentIterWrapper as PettingzooWrap


class BaseWrapper(PettingzooWrap):

    metadata = {'render.modes': ['human']}

    def __init__(self, env):
        '''
        Creates a wrapper around `env`. Extend this class to create changes to the space.
        '''
        super().__init__(env)

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

    def reset(self, observe=True):
        observation = super().reset(observe)
        agent = self.env.agent_selection

        self._update_step(agent,observation)
        if observe:
            observation = self._modify_observation(agent,observation)
            return observation
        else:
            return None

    def observe(self, agent):
        obs = super().observe(agent)
        observation = self._modify_observation(agent, obs)
        return observation

    def step(self, action, observe=True):
        agent = self.env.agent_selection
        cur_act_space = self.action_spaces[agent]
        assert not isinstance(cur_act_space,Box) or cur_act_space.shape == action.shape, "the shape of the action {} is not equal to the shape of the action space {}".format(action.shape,cur_act_space.shape)
        action = self._modify_action(agent, action)
        next_obs = super().step(action, observe=observe)

        new_agent = self.env.agent_selection

        self._update_step(new_agent,next_obs)

        if observe:
            next_obs = self._modify_observation(new_agent,next_obs)
            return next_obs
        else:
            return None
