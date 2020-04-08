from wrapper import BaseWrapper
from gym.spaces import Box
from .basic_transforms import color_reduction,down_scale,dtype,flatten,reshape

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
        if self.check_observation_fn is not None:
            for space in self.env.observation_spaces.values():
                self.check_observation_fn(space)

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
        return= self.module.change_observation_fn(observation)


class BasicObservationWrapper(lambda_wrapper):
    '''
    For internal use only
    '''
    def __init__(self,env,module,param):
        super().__init__(env)
        def change_obs_fn(obs):
            return module.change_observation(obs,param)
        def check_obs_fn(obs_space):
            return module.check_param(obs_space,param)
        self.module = module
        self.param = param

    def _check_wrapper_params(self):
        assert all([isinstance(obs_space, Box) for obs_space in self.observation_spaces.values()]), \
            "All agents' observation spaces are not Box: {}, and as such the observation spaces are not modified.".format(self.observation_spaces)
        super()._check_wrapper_params()

class color_reduction(BasicObservationWrapper):
    def __init__(self,env,mode='full'):
        super().__init__(env,color_reduction,mode)

class down_scale(BasicObservationWrapper):
    def __init__(self,env,scale_tuple):
        super().__init__(env,down_scale,scale_tuple)

class dtype(BasicObservationWrapper):
    def __init__(self,env,dtype):
        super().__init__(dtype,new_dtype)

class flatten(BasicObservationWrapper):
    def __init__(self,env):
        super().__init__(flatten,True)

class reshape(BasicObservationWrapper):
    def __init__(self,env,shape):
        super().__init__(reshape,shape)
