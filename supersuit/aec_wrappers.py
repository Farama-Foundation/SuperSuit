from .base_aec_wrapper import BaseWrapper, PettingzooWrap
from gym.spaces import Box, Space, Discrete
from . import basic_transforms
from .utils.frame_stack import stack_obs_space, stack_init, stack_obs
from .action_transforms import homogenize_ops
from .utils import agent_indicator as agent_ider
from .utils.frame_skip import check_transform_frameskip
from .utils.obs_delay import Delayer
from .utils.accumulator import Accumulator
import numpy as np
import gym


class ObservationWrapper(BaseWrapper):
    def _modify_action(self, agent, action):
        return action

    def _update_step(self, agent):
        pass


class observation_lambda(ObservationWrapper):
    def __init__(self, env, change_observation_fn, change_obs_space_fn=None):
        assert callable(change_observation_fn), "change_observation_fn needs to be a function. It is {}".format(change_observation_fn)
        assert change_obs_space_fn is None or callable(change_obs_space_fn), "change_obs_space_fn needs to be a function. It is {}".format(change_obs_space_fn)

        old_space_fn = change_obs_space_fn
        old_obs_fn = change_observation_fn

        def space_fn_ignore(space, agent):
            return old_space_fn(space)

        def obs_fn_ignore(obs, agent):
            return old_obs_fn(obs)

        agent0 = env.possible_agents[0]
        agent0_space = env.observation_spaces[agent0]

        if change_obs_space_fn is not None:
            try:
                change_obs_space_fn(agent0_space, agent0)
            except TypeError:
                change_obs_space_fn = space_fn_ignore

        try:
            change_observation_fn(agent0_space.sample(), agent0)
        except TypeError:
            change_observation_fn = obs_fn_ignore

        self.change_observation_fn = change_observation_fn
        self.change_obs_space_fn = change_obs_space_fn

        super().__init__(env)

    def _check_wrapper_params(self):
        if self.change_obs_space_fn is None:
            spaces = self.observation_spaces.values()
            for space in spaces:
                assert isinstance(space, Box), "the observation_lambda_wrapper only allows the change_obs_space_fn argument to be optional for Box observation spaces"

    def _modify_spaces(self):
        new_spaces = {}
        for agent, space in self.observation_spaces.items():
            if self.change_obs_space_fn is None:
                trans_low = self.change_observation_fn(space.low, agent)
                trans_high = self.change_observation_fn(space.high, agent)
                new_low = np.minimum(trans_low, trans_high)
                new_high = np.maximum(trans_low, trans_high)

                new_spaces[agent] = Box(low=new_low, high=new_high, dtype=new_low.dtype)
            else:
                new_space = self.change_obs_space_fn(space, agent)
                assert isinstance(new_space, Space), "output of change_obs_space_fn to observation_lambda_wrapper must be a gym space"
                new_spaces[agent] = new_space
        self.observation_spaces = new_spaces

    def _modify_observation(self, agent, observation):
        return self.change_observation_fn(observation, agent)


class BasicObservationWrapper(ObservationWrapper):
    """
    For internal use only
    """

    def __init__(self, env, module, param):
        self.check_param = module.check_param
        self.change_obs_space = module.change_obs_space
        self.change_observation = module.change_observation
        self.param = param
        super().__init__(env)

    def _check_wrapper_params(self):
        assert all([isinstance(obs_space, Box) for obs_space in self.observation_spaces.values()]), "All agents' observation spaces are not Box, they are: {}.".format(self.observation_spaces)
        for obs_space in self.env.observation_spaces.values():
            self.check_param(obs_space, self.param)

    def _modify_spaces(self):
        new_spaces = {}
        for agent, space in self.observation_spaces.items():
            new_spaces[agent] = self.change_obs_space(space, self.param)
        self.observation_spaces = new_spaces

    def _modify_observation(self, agent, observation):
        obs_space = self.env.observation_spaces[agent]
        return self.change_observation(observation, obs_space, self.param)


class color_reduction(BasicObservationWrapper):
    def __init__(self, env, mode="full"):
        super().__init__(env, basic_transforms.color_reduction, mode)


class resize(BasicObservationWrapper):
    def __init__(self, env, x_size, y_size, linear_interp=False):
        scale_tuple = (x_size, y_size, linear_interp)
        super().__init__(env, basic_transforms.resize, scale_tuple)


class dtype(BasicObservationWrapper):
    def __init__(self, env, dtype):
        super().__init__(env, basic_transforms.dtype, dtype)


class flatten(BasicObservationWrapper):
    def __init__(self, env):
        super().__init__(env, basic_transforms.flatten, True)


class reshape(BasicObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env, basic_transforms.reshape, shape)


class normalize_obs(BasicObservationWrapper):
    def __init__(self, env, env_min=0.0, env_max=1.0):
        shape = (env_min, env_max)
        super().__init__(env, basic_transforms.normalize_obs, shape)


class agent_indicator(ObservationWrapper):
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


class pad_observations(ObservationWrapper):
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


class black_death(ObservationWrapper):
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


class max_observation(ObservationWrapper):
    def __init__(self, env, memory):
        self.memory = memory
        self.reduction = np.maximum
        super().__init__(env)

    def _check_wrapper_params(self):
        int(self.memory)  # delay must be an int

    def _modify_spaces(self):
        return

    def reset(self):
        self._accumulators = {agent: Accumulator(obs_space, self.memory, self.reduction) for agent, obs_space in self.observation_spaces.items()}
        super().reset()
        for agent, accum in self._accumulators.items():
            accum.add(self.env.observe(agent))

    def _update_step(self, agent):
        observation = self.env.observe(agent)
        self._accumulators[agent].add(observation)

    def _modify_observation(self, agent, observation):
        return self._accumulators[agent].get()


class delay_observations(ObservationWrapper):
    def __init__(self, env, delay):
        self.delay = delay
        super().__init__(env)

    def _check_wrapper_params(self):
        int(self.delay)  # delay must be an int

    def _modify_spaces(self):
        return

    def reset(self):
        self._delayers = {agent: Delayer(obs_space, self.delay) for agent, obs_space in self.observation_spaces.items()}
        self._observes = {agent: None for agent in self.possible_agents}
        super().reset()

    def _update_step(self, agent):
        observation = self.env.observe(agent)
        self._observes[agent] = self._delayers[agent].add(observation)

    def _modify_observation(self, agent, observation):
        return self._observes[agent]


class frame_stack(BaseWrapper):
    def __init__(self, env, num_frames=4):
        self.stack_size = num_frames
        super().__init__(env)

    def _check_wrapper_params(self):
        assert isinstance(self.stack_size, int), "stack size of frame_stack must be an int"
        for space in self.observation_spaces.values():
            if isinstance(space, Box):
                assert 1 <= len(space.shape) <= 3, "frame_stack only works for 1, 2 or 3 dimensional observations"
            elif isinstance(space, Discrete):
                pass
            else:
                assert False, "Stacking is currently only allowed for Box and Discrete observation spaces. The given observation space is {}".format(space)

    def reset(self):
        self.stacks = {agent: stack_init(space, self.stack_size) for agent, space in self.env.observation_spaces.items()}
        super().reset()

    def _modify_spaces(self):
        self.observation_spaces = {agent: stack_obs_space(space, self.stack_size) for agent, space in self.observation_spaces.items()}

    def _modify_action(self, agent, action):
        return action

    def _modify_observation(self, agent, observation):
        return self.stacks[agent]

    def _update_step(self, agent):
        observation = self.env.observe(agent)
        self.stacks[agent] = stack_obs(
            self.stacks[agent],
            observation,
            self.env.observation_spaces[agent],
            self.stack_size,
        )


class StepAltWrapper(BaseWrapper):
    def _check_wrapper_params(self):
        pass

    def _modify_spaces(self):
        pass

    def _update_step(self, agent):
        pass

    def _modify_action(self, agent, action):
        return action

    def _modify_observation(self, agent, observation):
        return observation


class frame_skip(StepAltWrapper):
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


class sticky_actions(StepAltWrapper):
    def __init__(self, env, repeat_action_probability):
        super().__init__(env)
        assert 0 <= repeat_action_probability < 1
        self.repeat_action_probability = repeat_action_probability
        self.np_random, seed = gym.utils.seeding.np_random(None)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        super().seed(seed)

    def reset(self):
        self.old_action = None
        super().reset()

    def step(self, action):
        if self.old_action is not None and self.np_random.uniform() < self.repeat_action_probability:
            action = self.old_action

        super().step(action)


class ActionWrapper(BaseWrapper):
    def __init__(self, env):
        super().__init__(env)

    def _modify_observation(self, agent, observation):
        return observation

    def _update_step(self, agent):
        pass


class action_lambda(ActionWrapper):
    def __init__(self, env, change_action_fn, change_space_fn):
        assert callable(change_action_fn), "change_action_fn needs to be a function. It is {}".format(change_action_fn)
        assert callable(change_space_fn), "change_space_fn needs to be a function. It is {}".format(change_space_fn)

        old_space_fn = change_space_fn
        old_action_fn = change_action_fn

        def space_fn_ignore(space, agent):
            return old_space_fn(space)

        def action_fn_ignore(action, space, agent):
            return old_action_fn(action, space)

        agent0 = env.possible_agents[0]
        agent0_space = env.action_spaces[agent0]

        try:
            new_space0 = change_space_fn(agent0_space, agent0)
        except TypeError:
            change_space_fn = space_fn_ignore
            new_space0 = change_space_fn(agent0_space, agent0)

        try:
            change_action_fn(new_space0.sample(), agent0_space, agent0)
        except TypeError:
            change_action_fn = action_fn_ignore

        self.change_action_fn = change_action_fn
        self.change_space_fn = change_space_fn

        super().__init__(env)

    def _check_wrapper_params(self):
        pass

    def _modify_spaces(self):
        new_spaces = {}
        for agent, space in self.action_spaces.items():
            new_spaces[agent] = self.change_space_fn(space, agent)
            assert isinstance(new_spaces[agent], Space), "output of change_space_fn argument to action_lambda_wrapper must be a gym space"

        self.action_spaces = new_spaces

    def _modify_action(self, agent, action):
        return self.change_action_fn(action, self.env.action_spaces[agent], agent)


class pad_action_space(ActionWrapper):
    def _check_wrapper_params(self):
        homogenize_ops.check_homogenize_spaces(list(self.env.action_spaces.values()))

    def _modify_spaces(self):
        space = homogenize_ops.homogenize_spaces(list(self.env.action_spaces.values()))

        self.action_spaces = {agent: space for agent in self.action_spaces}

    def _modify_action(self, agent, action):
        new_action = homogenize_ops.dehomogenize_actions(self.env.action_spaces[agent], action)
        return new_action


class clip_actions(ActionWrapper):
    def _check_wrapper_params(self):
        for space in self.env.action_spaces.values():
            assert isinstance(space, Box), "clip_actions only works for Box action spaces"

    def _modify_spaces(self):
        pass

    def _modify_action(self, agent, action):
        act_space = self.action_spaces[agent]
        action = np.clip(action, act_space.low, act_space.high)
        return action


class RewardWrapper(PettingzooWrap):
    def _check_wrapper_params(self):
        pass

    def _modify_spaces(self):
        pass

    def reset(self):
        super().reset()
        self.rewards = {agent: self._change_reward_fn(reward) for agent, reward in self.rewards.items()}
        self.__cumulative_rewards = {a: 0 for a in self.agents}
        self._accumulate_rewards()

    def step(self, action):
        agent = self.env.agent_selection
        super().step(action)
        self.rewards = {agent: self._change_reward_fn(reward) for agent, reward in self.rewards.items()}
        self.__cumulative_rewards[agent] = 0
        self._cumulative_rewards = self.__cumulative_rewards
        self._accumulate_rewards()


class reward_lambda(RewardWrapper):
    def __init__(self, env, change_reward_fn):
        assert callable(change_reward_fn), "change_reward_fn needs to be a function. It is {}".format(change_reward_fn)
        self._change_reward_fn = change_reward_fn

        super().__init__(env)


class clip_reward(RewardWrapper):
    def __init__(self, env, lower_bound=-1, upper_bound=1):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        super().__init__(env)

    def _change_reward_fn(self, rew):
        return max(min(rew, self.upper_bound), self.lower_bound)
