import gym
import numpy as np
from gym.spaces import Box, Space, Discrete
from pettingzoo.utils.wrappers import OrderEnforcingWrapper as PettingzooWrap
from supersuit.utils.wrapper_chooser import WrapperChooser


class shared_wrapper_aec(PettingzooWrap):
    def __init__(self, env, modifier_class):
        super().__init__(env)

        self.modifiers = {}
        for agent in self.env.possible_agents:
            self.modifiers[agent] = modifier_class()
            self.observation_spaces[agent] = self.modifiers[agent].modify_obs_space(self.observation_spaces[agent])
            self.action_spaces[agent] = self.modifiers[agent].modify_action_space(self.action_spaces[agent])

    def reset(self):
        for mod in self.modifiers.values():
            mod.reset()
        super().reset()
        self.modifiers[self.agent_selection].modify_obs(super().observe(self.agent_selection))

    def step(self, action):
        mod = self.modifiers[self.agent_selection]
        action = mod.modify_action(action)
        super().step(action)

        self.modifiers[self.agent_selection].modify_obs(super().observe(self.agent_selection))

    def observe(self, agent):
        return self.modifiers[agent].get_last_obs()

class shared_wrapper_gym(gym.Wrapper):
    def __init__(self, env, modifier_class):
        super().__init__(env)
        self.modifier = modifier_class()
        self.observation_space = self.modifier.modify_obs_space(self.observation_space)
        self.action_space = self.modifier.modify_action_space(self.action_space)

    def reset(self):
        self.modifier.reset()
        obs = super().reset()
        obs = self.modifier.modify_obs(obs, self.observation_space)
        return obs

    def step(self, action):
        obs, rew, done, info = super().step(self.modifier.modify_action(action))
        obs = self.modifier.modify_obs(obs)
        return obs, rew, done, info


shared_wrapper = WrapperChooser(aec_wrapper=shared_wrapper_aec, gym_wrapper=shared_wrapper_gym)
