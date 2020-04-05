from wrapper import BaseWrapper
class BasicObservationWrapper(BaseWrapper):
    def __init__(self,module,param):
        self.module = module
        self.param = param

    def _check_wrapper_params(self):
        self.module.check_param(self.param)

    def _modify_spaces(self):
        self.observation_spaces = self.module.change_space(self.observation_spaces,self.param)

    def _modify_action(self, agent, action):
        return action

    def _modify_observation(self, agent, observation):
        return= self.module.change_observation(observation,self.param)
