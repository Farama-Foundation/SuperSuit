from .utils.base_aec_wrapper import BaseWrapper, PettingzooWrap
from gym.spaces import Box, Space, Discrete
from .utils.frame_skip import check_transform_frameskip
import numpy as np
import gym


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


class StepAltWrapper(BaseWrapper):
    def _modify_action(self, agent, action):
        return action

    def _modify_observation(self, agent, observation):
        return observation


class frame_skip_aec(StepAltWrapper):
    def __init__(self, env, num_frames):
        super().__init__(env)
        assert isinstance(num_frames, int), "multi-agent frame skip only takes in an integer"
        assert num_frames > 0
        check_transform_frameskip(num_frames)
        self.num_frames = num_frames

    def reset(self):
        super().reset()
        self.agents = self.env.agents[:]
        self.dones = {agent: False for agent in self.agents}
        self.rewards = {agent: 0. for agent in self.agents}
        self._cumulative_rewards = {agent: 0. for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.skip_num = {agent: 0 for agent in self.agents}
        self.old_actions = {agent: None for agent in self.agents}
        self._final_observations = {agent: None for agent in self.agents}

    def observe(self, agent):
        fin_observe = self._final_observations[agent]
        return fin_observe if fin_observe is not None else super().observe(agent)

    def step(self, action):
        self._has_updated = True
        if self.dones[self.agent_selection]:
            if self.env.agents and self.agent_selection == self.env.agent_selection:
                self.env.step(None)
            self._was_done_step(action)
            return
        cur_agent = self.agent_selection
        self._cumulative_rewards[cur_agent] = 0
        self.rewards = {a: 0. for a in self.agents}
        self.skip_num[cur_agent] = self.num_frames
        self.old_actions[cur_agent] = action
        while self.old_actions[self.env.agent_selection] is not None:
            step_agent = self.env.agent_selection
            if step_agent in self.env.dones:
                # reward = self.env.rewards[step_agent]
                # done = self.env.dones[step_agent]
                # info = self.env.infos[step_agent]
                observe, reward, done, info = self.env.last(observe=False)
                action = self.old_actions[step_agent]
                self.env.step(action)

                for agent in self.env.agents:
                    self.rewards[agent] += self.env.rewards[agent]
                self.infos[self.env.agent_selection] = info
                while self.env.agents and self.env.dones[self.env.agent_selection]:
                    done_agent = self.env.agent_selection
                    self.dones[done_agent] = True
                    self._final_observations[done_agent] = self.env.observe(done_agent)
                    self.env.step(None)
                step_agent = self.env.agent_selection

            self.skip_num[step_agent] -= 1
            if self.skip_num[step_agent] == 0:
                self.old_actions[step_agent] = None

        for agent in self.env.agents:
            self.dones[agent] = self.env.dones[agent]
            self.infos[agent] = self.env.infos[agent]
        self.agent_selection = self.env.agent_selection
        self._accumulate_rewards()
        self._dones_step_first()
