import functools
from supersuit.utils.base_aec_wrapper import BaseWrapper
from gym.spaces import Box
import numpy as np
import gym
from supersuit.utils.wrapper_chooser import WrapperChooser


class ObservationWrapper(BaseWrapper):
    def _modify_action(self, agent, action):
        return action


class black_death_aec(ObservationWrapper):
    def _check_wrapper_params(self):
        if not hasattr(self, 'possible_agents'):
            raise AssertionError("environment passed to black death wrapper must have the possible_agents attribute.")

        for agent in self.possible_agents:
            space = self.observation_space(agent)
            assert isinstance(space, gym.spaces.Box), f"observation sapces for black death must be Box spaces, is {space}"

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        old_obs_space = self.env.observation_space(agent)
        return Box(low=np.minimum(0, old_obs_space.low), high=np.maximum(0, old_obs_space.high), dtype=old_obs_space.dtype)

    def observe(self, agent):
        return np.zeros_like(self.observation_space(agent).low) if agent not in self.env.dones else self.env.observe(agent)

    def reset(self):
        super().reset()
        self._agent_idx = 0
        self.agent_selection = self.possible_agents[self._agent_idx]
        self.agents = self.possible_agents[:]
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self._update_items()

    def _update_items(self):
        self.dones = {}
        self.infos = {}
        self.rewards = {}

        _env_finishing = self._agent_idx == len(self.possible_agents) - 1 and all(self.env.dones.values())

        for agent in self.agents:
            self.dones[agent] = _env_finishing
            self.rewards[agent] = self.env.rewards.get(agent, 0)
            self.infos[agent] = self.env.infos.get(agent, {})
            self._cumulative_rewards[agent] += self.env.rewards.get(agent, 0)

            # self._cumulative_rewards[agent] = self.env._cumulative_rewards.get(agent, 0)

    def step(self, action):
        self._has_updated = True
        if self.dones[self.agent_selection]:
            return self._was_done_step(action)

        cur_agent = self.agent_selection
        self._cumulative_rewards[cur_agent] = 0
        if cur_agent == self.env.agent_selection:
            assert cur_agent in self.env.dones
            if self.env.dones[cur_agent]:
                action = None

            self.env.step(action)

        self._update_items()

        self._agent_idx = (1 + self._agent_idx) % len(self.possible_agents)
        self.agent_selection = self.possible_agents[self._agent_idx]

        self._dones_step_first()


black_death_v2 = WrapperChooser(aec_wrapper=black_death_aec)
