from wrapper import BaseWrapper
from gym.spaces import Box

class BasicObservationWrapper(BaseWrapper):
    def __init__(self,module,param):
        self.module = module
        self.param = param

    def _check_wrapper_params(self):
        assert all([isinstance(obs_space, Box) for obs_space in self.observation_spaces.values()]), \
            "All agents' observation spaces are not Box: {}, and as such the observation spaces are not modified.".format(self.observation_spaces)
        self.module.check_param(self.param)

    def _modify_spaces(self):
        self.observation_spaces = self.module.change_space(self.observation_spaces,self.param)

    def _modify_action(self, agent, action):
        return action

    def _modify_observation(self, agent, observation):
        return= self.module.change_observation(observation,self.param)
