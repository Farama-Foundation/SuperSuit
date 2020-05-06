from .base_aec_wrapper import BaseWrapper
from gym.spaces import Box,Space,Discrete
from . import basic_transforms
from .frame_stack import stack_obs_space,stack_init,stack_obs
from .action_transforms import homogenize_ops
from .action_transforms import continuous_action_ops
import numpy as np


class ObservationWrapper(BaseWrapper):
    def _modify_action(self, agent, action):
        return action

    def _update_step(self, agent, observation):
        pass

class observation_lambda_wrapper(ObservationWrapper):
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
            "All agents' observation spaces are not Box: {}, and as such the observation spaces are not modified.".format(self.observation_spaces)
        for obs_space in self.env.observation_spaces.values():
            self.module.check_param(obs_space,self.param)

    def _modify_spaces(self):
        new_spaces = {}
        for agent,space in self.observation_spaces.items():
            new_spaces[agent] = self.module.change_obs_space(space, self.param)
        self.observation_spaces = new_spaces

    def _modify_observation(self, agent, observation):
        obs_space = self.observation_spaces[agent]
        return self.module.change_observation(observation, obs_space, self.param)


class color_reduction(BasicObservationWrapper):
    def __init__(self,env,mode='full'):
        super().__init__(env,basic_transforms.color_reduction,mode)

class down_scale(BasicObservationWrapper):
    def __init__(self,env,x_scale=1,y_scale=1):
        old_obs_shape = list(env.observation_spaces.values())[0].shape
        scale_list = [1]*len(old_obs_shape)
        scale_list[0] = y_scale
        scale_list[0] = x_scale
        scale_tuple = tuple(scale_list)
        super().__init__(env,basic_transforms.down_scale,scale_tuple)

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

class homogenize_obs(ObservationWrapper):
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

class ActionWrapper(BaseWrapper):
    def __init__(self, env):
        super().__init__(env)

    def _modify_observation(self, agent, observation):
        return observation

    def _update_step(self, agent, observation):
        pass

class action_lambda_wrapper(ActionWrapper):
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
        return self.change_action_fn(action, self.action_spaces[agent])

class homogenize_actions(ActionWrapper):
    def _check_wrapper_params(self):
        homogenize_ops.check_homogenize_spaces(list(self.env.action_spaces.values()))

    def _modify_spaces(self):
        space = homogenize_ops.homogenize_spaces(list(self.env.action_spaces.values()))

        self.action_spaces = {agent:space for agent in self.action_spaces}

    def _modify_action(self, agent, action):
        new_action = homogenize_ops.dehomogenize_actions(self.env.action_spaces[agent], action)
        return new_action

class continuous_actions(ActionWrapper):
    def __init__(self, env):
        SEED = 0x601326ad
        self.np_random = np.random.RandomState(SEED)
        super().__init__(env)

    def _check_wrapper_params(self):
        for space in self.action_spaces.values():
            continuous_action_ops.check_action_space(space)

    def _modify_spaces(self):
        spaces = {agent: continuous_action_ops.change_action_space(act_space) for agent,act_space in self.env.action_spaces.items()}

        self.action_spaces = spaces

    def _modify_action(self, agent, action):
        act_space = self.env.action_spaces[agent]
        new_action = continuous_action_ops.modify_action(act_space, action, self.np_random)
        return new_action

    def _remove_infos(self):
        self.infos = {agent: {key:value for key,value in info_dict.items() if key != "legal_moves"}
            for agent,info_dict in self.infos.items()}

    def reset(self, observe=True):
        res = super().reset(observe)
        self._remove_infos()
        return res

    def step(self, action, observe=True):
        res = super().step(action, observe)
        self._remove_infos()
        return res
