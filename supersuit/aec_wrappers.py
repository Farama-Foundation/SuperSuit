from .base_aec_wrapper import BaseWrapper
from gym.spaces import Box
from . import basic_transforms
from .frame_stack import stack_obs_space,stack_init,stack_obs

class lambda_wrapper(BaseWrapper):
    '''
    Parameters
    ----------
    env : AECEnv
        PettingZoo compatable environment.
    change_observation_fn : callable
        callable function which takes in an observation, outputs a
    '''
    def __init__(self, env, change_observation_fn, check_space_fn=None):
        assert callable(change_observation_fn), "change_observation_fn needs to be a function. It is {}".format(change_observation_fn)
        assert check_space_fn is None or callable(check_space_fn), "change_observation_fn needs to be a function. It is {}".format(check_space_fn)
        self.change_observation_fn = change_observation_fn
        self.check_space_fn = check_space_fn

        super().__init__(env)

    def _check_wrapper_params(self):
        if self.check_space_fn is not None:
            for space in self.env.observation_spaces.values():
                self.check_space_fn(space)

    def _modify_spaces(self):
        new_spaces = {}
        for agent,space in self.observation_spaces.items():
            new_low = self.change_observation_fn(space.low)
            new_high = self.change_observation_fn(space.high)
            new_spaces[agent] = Box(low=new_low, high=new_high, dtype=new_low.dtype)
        return new_spaces

    def _modify_action(self, agent, action):
        return action

    def _modify_observation(self, agent, observation):
        return self.change_observation_fn(observation)

    def _update_step(self, agent, observation, action):
        pass

class BasicObservationWrapper(BaseWrapper):
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
            new_low = self.module.change_observation(space.low, self.param)
            new_high = self.module.change_observation(space.high, self.param)
            new_spaces[agent] = Box(low=new_low, high=new_high, dtype=new_low.dtype)
        return new_spaces

    def _modify_action(self, agent, action):
        return action

    def _modify_observation(self, agent, observation):
        return self.module.change_observation(observation, self.param)

    def _update_step(self, agent, observation, action):
        pass

class color_reduction(BasicObservationWrapper):
    def __init__(self,env,mode='full'):
        super().__init__(env,basic_transforms.color_reduction,mode)

class down_scale(BasicObservationWrapper):
    def __init__(self,env,scale_tuple):
        super().__init__(env,basic_transforms.down_scale,scale_tuple)

class dtype(BasicObservationWrapper):
    def __init__(self,env,dtype):
        super().__init__(env,basic_transforms.dtype,new_dtype)

class flatten(BasicObservationWrapper):
    def __init__(self,env):
        super().__init__(env,basic_transforms.flatten,True)

class reshape(BasicObservationWrapper):
    def __init__(self,env,shape):
        super().__init__(env,basic_transforms.reshape,shape)

class frame_stack(BaseWrapper):
    def __init__(self,env,stack_size):
        self.stack_size = stack_size
        super().__init__(env)

    def _check_wrapper_params(self):
        assert isinstance(self.stack_size, int), "stack size of frame_stack must be an int"
        for space in self.observation_spaces.values():
            assert 1 <= len(space.shape) <= 3, "frame_stack only works for 1,2 or 3 dimentional observations"

    def reset(self, observe=True):
        self.stacks = {agent: stack_init(space, self.stack_size) for agent,space in self.env.observation_spaces.items()}
        return super().reset(observe)

    def _modify_spaces(self):
        self.observation_spaces = {agent: stack_obs_space(space, self.stack_size)  for agent,space in self.observation_spaces.items()}

    def _modify_action(self, agent, action):
        return action

    def _modify_observation(self, agent, observation):
        stack_obs(self.stacks[agent], observation, self.stack_size)
        return self.stacks[agent]

    def _update_step(self, agent, observation, action):
        pass
