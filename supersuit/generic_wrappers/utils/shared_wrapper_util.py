import functools
import gym
from pettingzoo.utils.wrappers import OrderEnforcingWrapper as PettingzooWrap
from supersuit.utils.wrapper_chooser import WrapperChooser
from pettingzoo.utils import BaseParallelWraper


class shared_wrapper_aec(PettingzooWrap):
    def __init__(self, env, modifier_class):
        super().__init__(env)
        self.modifier_class = modifier_class

        self.modifiers = {}
        if hasattr(self.env, 'possible_agents'):
            self.add_modifiers(self.env.possible_agents)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.modifiers[agent].modify_obs_space(self.env.observation_space(agent))

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.modifiers[agent].modify_action_space(self.env.action_space(agent))

    def seed(self, seed=None):
        super().seed(seed)
        for agent, mod in sorted(self.modifiers.items()):
            mod.seed(seed)
            if seed is not None:
                seed += 1

    def add_modifiers(self, agents_list):
        for agent in agents_list:
            if agent not in self.modifiers:
                self.modifiers[agent] = self.modifier_class()
                # populate modifier spaces
                self.observation_space(agent)
                self.action_space(agent)
                self.modifiers[agent].reset()

    def reset(self):
        for mod in self.modifiers.values():
            mod.reset()
        super().reset()
        self.add_modifiers(self.agents)
        self.modifiers[self.agent_selection].modify_obs(super().observe(self.agent_selection))

    def step(self, action):
        mod = self.modifiers[self.agent_selection]
        action = mod.modify_action(action)
        if self.dones[self.agent_selection]:
            action = None
        super().step(action)
        self.add_modifiers(self.agents)
        self.modifiers[self.agent_selection].modify_obs(super().observe(self.agent_selection))

    def observe(self, agent):
        return self.modifiers[agent].get_last_obs()


class shared_wrapper_parr(BaseParallelWraper):
    def __init__(self, env, modifier_class):
        super().__init__(env)

        self.modifier_class = modifier_class
        self.modifiers = {}

        if hasattr(self.env, 'possible_agents'):
            self.add_modifiers(self.env.possible_agents)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.modifiers[agent].modify_obs_space(self.env.observation_space(agent))

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.modifiers[agent].modify_action_space(self.env.action_space(agent))

    def add_modifiers(self, agents_list):
        for agent in agents_list:
            if agent not in self.modifiers:
                self.modifiers[agent] = self.modifier_class()
                # populate modifier spaces
                self.observation_space(agent)
                self.action_space(agent)
                self.modifiers[agent].reset()

    def seed(self, seed=None):
        super().seed(seed)
        for agent, mod in sorted(self.modifiers.items()):
            mod.seed(seed)
            if seed is not None:
                seed += 1

    def reset(self):
        observations = super().reset()
        self.add_modifiers(self.agents)
        for agent, mod in self.modifiers.items():
            mod.reset()
        observations = {agent: self.modifiers[agent].modify_obs(obs) for agent, obs in observations.items()}
        return observations

    def step(self, actions):
        actions = {agent: self.modifiers[agent].modify_action(action) for agent, action in actions.items()}
        observations, rewards, dones, infos = super().step(actions)
        self.add_modifiers(self.agents)
        observations = {agent: self.modifiers[agent].modify_obs(obs) for agent, obs in observations.items()}
        return observations, rewards, dones, infos


class shared_wrapper_gym(gym.Wrapper):
    def __init__(self, env, modifier_class):
        super().__init__(env)
        self.modifier = modifier_class()
        self.observation_space = self.modifier.modify_obs_space(self.observation_space)
        self.action_space = self.modifier.modify_action_space(self.action_space)

    def seed(self, seed=None):
        super().seed(seed)
        self.modifier.seed(seed)

    def reset(self):
        self.modifier.reset()
        obs = super().reset()
        obs = self.modifier.modify_obs(obs)
        return obs

    def step(self, action):
        obs, rew, done, info = super().step(self.modifier.modify_action(action))
        obs = self.modifier.modify_obs(obs)
        return obs, rew, done, info


shared_wrapper = WrapperChooser(aec_wrapper=shared_wrapper_aec, gym_wrapper=shared_wrapper_gym, parallel_wrapper=shared_wrapper_parr)
