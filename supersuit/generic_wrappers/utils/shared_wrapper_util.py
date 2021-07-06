import gym
from pettingzoo.utils.wrappers import OrderEnforcingWrapper as PettingzooWrap
from supersuit.utils.wrapper_chooser import WrapperChooser
from supersuit.utils.base_parallel_wrapper import ParallelWraper


class shared_wrapper_aec(PettingzooWrap):
    def __init__(self, env, modifier_class):
        super().__init__(env)

        self.modifiers = {}
        self.observation_spaces = {}
        self.action_spaces = {}
        for agent in self.env.possible_agents:
            self.modifiers[agent] = modifier_class()
            self.observation_spaces[agent] = self.modifiers[agent].modify_obs_space(self.env.observation_spaces[agent])
            self.action_spaces[agent] = self.modifiers[agent].modify_action_space(self.env.action_spaces[agent])

    def seed(self, seed=None):
        super().seed(seed)
        for agent, mod in sorted(self.modifiers.items()):
            mod.seed(seed)
            if seed is not None:
                seed += 1

    def reset(self):
        for mod in self.modifiers.values():
            mod.reset()
        super().reset()
        self.modifiers[self.agent_selection].modify_obs(super().observe(self.agent_selection))

    def step(self, action):
        mod = self.modifiers[self.agent_selection]
        action = mod.modify_action(action)
        if self.dones[self.agent_selection]:
            action = None
        super().step(action)
        self.modifiers[self.agent_selection].modify_obs(super().observe(self.agent_selection))

    def observe(self, agent):
        return self.modifiers[agent].get_last_obs()


class shared_wrapper_parr(ParallelWraper):
    def __init__(self, env, modifier_class):
        super().__init__(env)

        self.modifiers = {}
        self.observation_spaces = {}
        self.action_spaces = {}
        for agent in self.env.possible_agents:
            self.modifiers[agent] = modifier_class()
            self.observation_spaces[agent] = self.modifiers[agent].modify_obs_space(self.env.observation_spaces[agent])
            self.action_spaces[agent] = self.modifiers[agent].modify_action_space(self.env.action_spaces[agent])

    def seed(self, seed=None):
        super().seed(seed)
        for agent, mod in sorted(self.modifiers.items()):
            mod.seed(seed)
            if seed is not None:
                seed += 1

    def reset(self):
        observations = super().reset()
        for agent, mod in self.modifiers.items():
            mod.reset()
            observations[agent] = mod.modify_obs(observations[agent])
        return observations

    def step(self, actions):
        actions = {agent: mod.modify_action(actions[agent]) for agent, mod in self.modifiers.items()}
        observations, rewards, dones, infos = super().step(actions)
        observations = {agent: mod.modify_obs(observations[agent]) for agent, mod in self.modifiers.items()}
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
