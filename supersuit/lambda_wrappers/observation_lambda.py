import gym
import numpy as np
from gym.spaces import Box, Space
from supersuit.utils.base_aec_wrapper import BaseWrapper
from supersuit.utils.wrapper_chooser import WrapperChooser


class aec_observation_lambda(BaseWrapper):
    def __init__(self, env, change_observation_fn, change_obs_space_fn=None):
        assert callable(change_observation_fn), "change_observation_fn needs to be a function. It is {}".format(change_observation_fn)
        assert change_obs_space_fn is None or callable(change_obs_space_fn), "change_obs_space_fn needs to be a function. It is {}".format(change_obs_space_fn)

        old_space_fn = change_obs_space_fn
        old_obs_fn = change_observation_fn

        def space_fn_ignore(space, agent):
            return old_space_fn(space)

        def obs_fn_ignore(obs, obs_space, agent):
            return old_obs_fn(obs, obs_space)

        agent0 = env.possible_agents[0]
        agent0_space = env.observation_spaces[agent0]

        if change_obs_space_fn is not None:
            try:
                change_obs_space_fn(agent0_space, agent0)
            except TypeError:
                change_obs_space_fn = space_fn_ignore

        try:
            change_observation_fn(agent0_space.sample(), agent0_space, agent0)
        except TypeError:
            change_observation_fn = obs_fn_ignore

        self.change_observation_fn = change_observation_fn
        self.change_obs_space_fn = change_obs_space_fn

        super().__init__(env)

    def _modify_action(self, agent, action):
        return action

    def _check_wrapper_params(self):
        if self.change_obs_space_fn is None:
            spaces = self.observation_spaces.values()
            for space in spaces:
                assert isinstance(space, Box), "the observation_lambda_wrapper only allows the change_obs_space_fn argument to be optional for Box observation spaces"

    def _modify_spaces(self):
        new_spaces = {}
        for agent, space in self.observation_spaces.items():
            if self.change_obs_space_fn is None:
                trans_low = self.change_observation_fn(space.low, space, agent)
                trans_high = self.change_observation_fn(space.high, space, agent)
                new_low = np.minimum(trans_low, trans_high)
                new_high = np.maximum(trans_low, trans_high)

                new_spaces[agent] = Box(low=new_low, high=new_high, dtype=new_low.dtype)
            else:
                new_space = self.change_obs_space_fn(space, agent)
                assert isinstance(new_space, Space), "output of change_obs_space_fn to observation_lambda_wrapper must be a gym space"
                new_spaces[agent] = new_space
        self.observation_spaces = new_spaces

    def _modify_observation(self, agent, observation):
        return self.change_observation_fn(observation, self.env.observation_spaces[agent], agent)


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
