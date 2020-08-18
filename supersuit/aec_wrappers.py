from .base_aec_wrapper import BaseWrapper
from gym.spaces import Box,Space,Discrete
from . import basic_transforms
from .adv_transforms.frame_stack import stack_obs_space,stack_init,stack_obs
from .action_transforms import homogenize_ops
from .adv_transforms import agent_indicator as agent_ider
from .adv_transforms.frame_skip import check_transform_frameskip
from .adv_transforms.obs_delay import Delayer
import numpy as np
import gym


class ObservationWrapper(BaseWrapper):
    def _modify_action(self, agent, action):
        return action

    def _update_step(self, agent, observation):
        pass

class observation_lambda(ObservationWrapper):
    def __init__(self, env, change_observation_fn, change_obs_space_fn=None):
        assert callable(change_observation_fn), "change_observation_fn needs to be a function. It is {}".format(change_observation_fn)
        assert change_obs_space_fn is None or callable(change_obs_space_fn), "change_obs_space_fn needs to be a function. It is {}".format(change_obs_space_fn)
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
        for agent,space in self.observation_spaces.items():
            if self.change_obs_space_fn is None:
                new_low = self.change_observation_fn(space.low)
                new_high = self.change_observation_fn(space.high)
                new_spaces[agent] = Box(low=new_low, high=new_high, dtype=new_low.dtype)
            else:
                new_space = self.change_obs_space_fn(space)
                assert isinstance(new_space, Space), "output of change_obs_space_fn to observation_lambda_wrapper must be a gym space"
                new_spaces[agent] = new_space
        self.observation_spaces = new_spaces

    def _modify_observation(self, agent, observation):
        return self.change_observation_fn(observation)


class BasicObservationWrapper(ObservationWrapper):
    '''
    For internal use only
    '''
    def __init__(self,env,module,param):
        self.module = module
        self.param = param
        super().__init__(env)

    def _check_wrapper_params(self):
        assert all([isinstance(obs_space, Box) for obs_space in self.observation_spaces.values()]), \
            "All agents' observation spaces are not Box, they are: {}.".format(self.observation_spaces)
        for obs_space in self.env.observation_spaces.values():
            self.module.check_param(obs_space,self.param)

    def _modify_spaces(self):
        new_spaces = {}
        for agent,space in self.observation_spaces.items():
            new_spaces[agent] = self.module.change_obs_space(space, self.param)
        self.observation_spaces = new_spaces

    def _modify_observation(self, agent, observation):
        obs_space = self.env.observation_spaces[agent]
        return self.module.change_observation(observation, obs_space, self.param)


class color_reduction(BasicObservationWrapper):
    def __init__(self,env,mode='full'):
        super().__init__(env,basic_transforms.color_reduction,mode)

class resize(BasicObservationWrapper):
    def __init__(self,env,x_size,y_size,linear_interp=False):
        scale_tuple = (x_size, y_size, linear_interp)
        super().__init__(env,basic_transforms.resize,scale_tuple)

class dtype(BasicObservationWrapper):
    def __init__(self,env,dtype):
        super().__init__(env,basic_transforms.dtype,dtype)

class flatten(BasicObservationWrapper):
    def __init__(self,env):
        super().__init__(env,basic_transforms.flatten,True)

class reshape(BasicObservationWrapper):
    def __init__(self,env,shape):
        super().__init__(env,basic_transforms.reshape,shape)

class normalize_obs(BasicObservationWrapper):
    def __init__(self,env,env_min=0.,env_max=1.):
        shape = (env_min, env_max)
        super().__init__(env,basic_transforms.normalize_obs,shape)

class agent_indicator(ObservationWrapper):
    def __init__(self,env,type_only=False):
        self.type_only = type_only
        self.indicator_map = agent_ider.get_indicator_map(env.agents,type_only)
        self.num_indicators = len(set(self.indicator_map.values()))
        super().__init__(env)

    def _check_wrapper_params(self):
        agent_ider.check_params(self.observation_spaces.values())

    def _modify_spaces(self):
        self.observation_spaces = {agent:agent_ider.change_obs_space(space,self.num_indicators) for agent,space in self.observation_spaces.items()}

    def _modify_observation(self, agent, observation):
        new_obs = agent_ider.change_observation(observation, self.env.observation_spaces[agent], (self.indicator_map[agent], self.num_indicators))
        return new_obs

class pad_observations(ObservationWrapper):
    def _check_wrapper_params(self):
        spaces = list(self.observation_spaces.values())
        homogenize_ops.check_homogenize_spaces(spaces)

    def _modify_spaces(self):
        spaces = list(self.observation_spaces.values())

        self._obs_space = homogenize_ops.homogenize_spaces(spaces)
        self.observation_spaces = {agent:self._obs_space for agent in self.observation_spaces}

    def _modify_observation(self, agent, observation):
        new_obs = homogenize_ops.homogenize_observations(self._obs_space,observation)
        return new_obs

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
        self._observes = {agent: None for agent in self.agents}
        return super().reset()

    def _update_step(self, agent, observation):
        self._observes[agent] = self._delayers[agent].add(observation)

    def _modify_observation(self, agent, observation):
        return self._observes[agent]

class frame_stack(BaseWrapper):
    def __init__(self,env,num_frames=4):
        self.stack_size = num_frames
        super().__init__(env)

    def _check_wrapper_params(self):
        assert isinstance(self.stack_size, int), "stack size of frame_stack must be an int"
        for space in self.observation_spaces.values():
            if isinstance(space, Box):
                assert 1 <= len(space.shape) <= 3, "frame_stack only works for 1,2 or 3 dimentional observations"
            elif isinstance(space, Discrete):
                pass
            else:
                assert False, "Stacking is currently only allowed for Box and Discrete observation spaces. The given observation space is {}".format(obs_space)

    def reset(self, observe=True):
        self.stacks = {agent: stack_init(space, self.stack_size) for agent,space in self.env.observation_spaces.items()}
        return super().reset(observe)

    def _modify_spaces(self):
        self.observation_spaces = {agent: stack_obs_space(space, self.stack_size)  for agent,space in self.observation_spaces.items()}

    def _modify_action(self, agent, action):
        return action

    def _modify_observation(self, agent, observation):
        return self.stacks[agent]

    def _update_step(self, agent, observation):
        if observation is None:
            observation = self.observe(agent)
        self.stacks[agent] = stack_obs(self.stacks[agent], observation, self.env.observation_spaces[agent], self.stack_size)

class StepAltWrapper(BaseWrapper):
    def _check_wrapper_params(self):
        pass

    def _modify_spaces(self):
        pass

    def _update_step(self, agent, obs):
        pass

    def _modify_action(self, agent, action):
        return action

    def _modify_observation(self, agent, observation):
        return observation

class frame_skip(StepAltWrapper):
    def __init__(self, env, frame_skip, seed=None):
        super().__init__(env)
        assert isinstance(frame_skip, int), "multi-agent frame skip only takes in an integer"
        check_transform_frameskip(frame_skip)
        self.frame_skip = frame_skip

    def reset(self, observe=True):
        self.skip_num = {agent: 0 for agent in self.agents}
        self.old_actions = {agent: None for agent in self.agents}
        #self.skips = {agent: 0 for agent,space in self.env.observation_spaces.items()}
        return super().reset(observe)

    def step(self, action, observe=True):
        cur_agent = self.agent_selection
        super().step(action, observe=False)
        self.skip_num[cur_agent] = self.frame_skip-1
        if self.skip_num[cur_agent] != 0:
            self.old_actions[cur_agent] = action
        while self.old_actions[self.agent_selection] is not None:
            step_agent = self.agent_selection
            super().step(self.old_actions[step_agent], observe=False)

            self.skip_num[step_agent] -= 1
            if self.skip_num[step_agent] == 0:
                self.old_actions[step_agent] = None

        return self.observe(self.agent_selection) if observe else None

class sticky_actions(StepAltWrapper):
    def __init__(self, env, repeat_action_probability, seed=None):
        super().__init__(env)
        assert 0 <= repeat_action_probability < 1
        self.repeat_action_probability = repeat_action_probability
        self.np_random, seed = gym.utils.seeding.np_random(seed)

    def reset(self, observe=True):
        self.old_action = None
        return super().reset(observe)

    def step(self, action, observe=True):
        if self.old_action is not None and self.np_random.uniform() < self.repeat_action_probability:
            action = self.old_action

        return super().step(action, observe)

class ActionWrapper(BaseWrapper):
    def __init__(self, env):
        super().__init__(env)

    def _modify_observation(self, agent, observation):
        return observation

    def _update_step(self, agent, observation):
        pass

class action_lambda(ActionWrapper):
    def __init__(self, env, change_action_fn, change_space_fn):
        assert callable(change_action_fn), "change_action_fn needs to be a function. It is {}".format(change_action_fn)
        assert callable(change_space_fn), "change_space_fn needs to be a function. It is {}".format(change_space_fn)
        self.change_action_fn = change_action_fn
        self.change_space_fn = change_space_fn

        super().__init__(env)

    def _check_wrapper_params(self):
        pass

    def _modify_spaces(self):
        new_spaces = {}
        for agent,space in self.action_spaces.items():
            new_spaces[agent] = self.change_space_fn(space)
            assert isinstance(new_spaces[agent], Space), "output of change_space_fn argument to action_lambda_wrapper must be a gym space"

        self.action_spaces = new_spaces

    def _modify_action(self, agent, action):
        return self.change_action_fn(action, self.env.action_spaces[agent])

class pad_action_space(ActionWrapper):
    def _check_wrapper_params(self):
        homogenize_ops.check_homogenize_spaces(list(self.env.action_spaces.values()))

    def _modify_spaces(self):
        space = homogenize_ops.homogenize_spaces(list(self.env.action_spaces.values()))

        self.action_spaces = {agent:space for agent in self.action_spaces}

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

class RewardWrapper(ActionWrapper,ObservationWrapper):
    def _check_wrapper_params(self):
        pass

    def _modify_spaces(self):
        pass

    def _update_step(self, agent, obs):
        self.rewards = {agent: self._change_reward_fn(reward) for agent,reward in self.rewards.items()}

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
