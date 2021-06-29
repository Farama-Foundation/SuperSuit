from .utils.base_aec_wrapper import BaseWrapper, PettingzooWrap
from gym.spaces import Box, Space, Discrete
from .utils import basic_transforms
from .utils.frame_stack import stack_obs_space, stack_init, stack_obs
from .utils.action_transforms import homogenize_ops
from .utils import agent_indicator as agent_ider
from .utils.frame_skip import check_transform_frameskip
from .utils.obs_delay import Delayer
from .utils.accumulator import Accumulator
from .utils.wrapper_chooser import WrapperChooser
import numpy as np
import gym


class ObservationWrapper(BaseWrapper):
    def _modify_action(self, agent, action):
        return action


class agent_indicator_aec(ObservationWrapper):
    def __init__(self, env, type_only=False):
        self.type_only = type_only
        self.indicator_map = agent_ider.get_indicator_map(env.possible_agents, type_only)
        self.num_indicators = len(set(self.indicator_map.values()))
        super().__init__(env)

    def _check_wrapper_params(self):
        agent_ider.check_params(self.observation_spaces.values())

    def _modify_spaces(self):
        self.observation_spaces = {agent: agent_ider.change_obs_space(space, self.num_indicators) for agent, space in self.observation_spaces.items()}

    def _modify_observation(self, agent, observation):
        new_obs = agent_ider.change_observation(
            observation,
            self.env.observation_spaces[agent],
            (self.indicator_map[agent], self.num_indicators),
        )
        return new_obs


class pad_observations_aec(ObservationWrapper):
    def _check_wrapper_params(self):
        spaces = list(self.observation_spaces.values())
        homogenize_ops.check_homogenize_spaces(spaces)

    def _modify_spaces(self):
        spaces = list(self.observation_spaces.values())

        self._obs_space = homogenize_ops.homogenize_spaces(spaces)
        self.observation_spaces = {agent: self._obs_space for agent in self.observation_spaces}

    def _modify_observation(self, agent, observation):
        new_obs = homogenize_ops.homogenize_observations(self._obs_space, observation)
        return new_obs


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


black_death_v1 = WrapperChooser(aec_wrapper=black_death_aec)


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


class pad_action_space_aec(BaseWrapper):
    def _modify_observation(self, agent, obs):
        return obs

    def _check_wrapper_params(self):
        homogenize_ops.check_homogenize_spaces(list(self.env.action_spaces.values()))

    def _modify_spaces(self):
        space = homogenize_ops.homogenize_spaces(list(self.env.action_spaces.values()))

        self.action_spaces = {agent: space for agent in self.action_spaces}

    def _modify_action(self, agent, action):
        new_action = homogenize_ops.dehomogenize_actions(self.env.action_spaces[agent], action)
        return new_action
