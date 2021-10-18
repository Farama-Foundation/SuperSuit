import functools
import gym
import numpy as np
from gym.spaces import Box, Space
from supersuit.utils.base_aec_wrapper import BaseWrapper
from supersuit.utils.wrapper_chooser import WrapperChooser


class aec_observation_lambda(BaseWrapper):
    def __init__(self, env, change_observation_fn, change_obs_space_fn=None):
        assert callable(change_observation_fn), "change_observation_fn needs to be a function. It is {}".format(change_observation_fn)
        assert change_obs_space_fn is None or callable(change_obs_space_fn), "change_obs_space_fn needs to be a function. It is {}".format(change_obs_space_fn)

        self.change_observation_fn = change_observation_fn
        self.change_obs_space_fn = change_obs_space_fn

        super().__init__(env)

        if hasattr(self, 'possible_agents'):
            for agent in self.possible_agents:
                # call any validation logic in this function
                self.observation_space(agent)

    def _modify_action(self, agent, action):
        return action

    def _check_wrapper_params(self):
        if self.change_obs_space_fn is None and hasattr(self, 'possible_agents'):
            for agent in self.possible_agents:
                assert isinstance(self.observation_space(agent), Box), "the observation_lambda_wrapper only allows the change_obs_space_fn argument to be optional for Box observation spaces"

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        if self.change_obs_space_fn is None:
            space = self.env.observation_space(agent)
            try:
                trans_low = self.change_observation_fn(space.low, space, agent)
                trans_high = self.change_observation_fn(space.high, space, agent)
            except TypeError:
                trans_low = self.change_observation_fn(space.low, space)
                trans_high = self.change_observation_fn(space.high, space)
            new_low = np.minimum(trans_low, trans_high)
            new_high = np.maximum(trans_low, trans_high)

            return Box(low=new_low, high=new_high, dtype=new_low.dtype)
        else:
            old_obs_space = self.env.observation_space(agent)
            try:
                return self.change_obs_space_fn(old_obs_space, agent)
            except TypeError:
                return self.change_obs_space_fn(old_obs_space)

    def _modify_observation(self, agent, observation):
        old_obs_space = self.env.observation_space(agent)
        try:
            return self.change_observation_fn(observation, old_obs_space, agent)
        except TypeError:
            return self.change_observation_fn(observation, old_obs_space)


class gym_observation_lambda(gym.Wrapper):
    def __init__(self, env, change_observation_fn, change_obs_space_fn=None):
        assert callable(change_observation_fn), "change_observation_fn needs to be a function. It is {}".format(change_observation_fn)
        assert change_obs_space_fn is None or callable(change_obs_space_fn), "change_obs_space_fn needs to be a function. It is {}".format(change_obs_space_fn)
        self.change_observation_fn = change_observation_fn
        self.change_obs_space_fn = change_obs_space_fn

        super().__init__(env)
        self._check_wrapper_params()
        self._modify_spaces()

    def _check_wrapper_params(self):
        if self.change_obs_space_fn is None:
            space = self.observation_space
            assert isinstance(space, Box), "the observation_lambda_wrapper only allows the change_obs_space_fn argument to be optional for Box observation spaces"

    def _modify_spaces(self):
        space = self.observation_space

        if self.change_obs_space_fn is None:
            new_low = self.change_observation_fn(space.low, space)
            new_high = self.change_observation_fn(space.high, space)
            new_space = Box(low=new_low, high=new_high, dtype=new_low.dtype)
        else:
            new_space = self.change_obs_space_fn(space)
            assert isinstance(new_space, Space), "output of change_obs_space_fn to observation_lambda_wrapper must be a gym space"
        self.observation_space = new_space

    def _modify_observation(self, observation):
        return self.change_observation_fn(observation, self.env.observation_space)

    def step(self, action):
        observation, rew, done, info = self.env.step(action)
        observation = self._modify_observation(observation)
        return observation, rew, done, info

    def reset(self):
        observation = self.env.reset()
        observation = self._modify_observation(observation)
        return observation


observation_lambda_v0 = WrapperChooser(aec_wrapper=aec_observation_lambda, gym_wrapper=gym_observation_lambda)
