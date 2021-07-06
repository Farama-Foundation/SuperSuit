from supersuit.utils.base_aec_wrapper import BaseWrapper
from gym.spaces import Box
import numpy as np
import gym
from supersuit.utils.wrapper_chooser import WrapperChooser
from supersuit.utils.deprecated import Deprecated


class ObservationWrapper(BaseWrapper):
    def _modify_action(self, agent, action):
        return action


class black_death_aec(ObservationWrapper):
    def _check_wrapper_params(self):
        for space in self.observation_spaces.values():
            assert isinstance(space, gym.spaces.Box), f"observation sapces for black death must be Box spaces, is {space}"

    def _modify_spaces(self):
        self.observation_spaces = {agent: Box(low=np.minimum(0, obs.low), high=np.maximum(0, obs.high), dtype=obs.dtype) for agent, obs in self.observation_spaces.items()}

    def observe(self, agent):
        return np.zeros_like(self.observation_spaces[agent].low) if agent not in self.env.dones else self.env.observe(agent)

    def reset(self):
        super().reset()
        self._agent_idx = 0
        self.agent_selection = self.possible_agents[self._agent_idx]
        self.agents = self.possible_agents[:]
        self._update_items()

    def _update_items(self):
        self.dones = {}
        self.infos = {}
        self.rewards = {}
        self._cumulative_rewards = {}

        _env_finishing = self._agent_idx == len(self.possible_agents) - 1 and all(self.env.dones.values())

        for agent in self.agents:
            self.dones[agent] = _env_finishing
            self.rewards[agent] = self.env.rewards.get(agent, 0)
            self.infos[agent] = self.env.infos.get(agent, {})
            self._cumulative_rewards[agent] = self.env._cumulative_rewards.get(agent, 0)

    def step(self, action):
        self._has_updated = True
        if self.dones[self.agent_selection]:
            return self._was_done_step(action)

        cur_agent = self.agent_selection
        if cur_agent == self.env.agent_selection:
            assert cur_agent in self.env.dones
            if self.env.dones[cur_agent]:
                action = None
            self.env.step(action)

        self._update_items()

        self._agent_idx = (1 + self._agent_idx) % len(self.possible_agents)
        self.agent_selection = self.possible_agents[self._agent_idx]

        self._dones_step_first()


black_death_v0 = Deprecated("black_death", "v0", "v1")
black_death_v1 = WrapperChooser(aec_wrapper=black_death_aec)
