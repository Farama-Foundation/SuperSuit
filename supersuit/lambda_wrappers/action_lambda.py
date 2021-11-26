import functools
from gym.spaces import Space
from supersuit.utils.base_aec_wrapper import BaseWrapper
from supersuit.utils.wrapper_chooser import WrapperChooser
import gym


class aec_action_lambda(BaseWrapper):
    def __init__(self, env, change_action_fn, change_space_fn):
        assert callable(change_action_fn), "change_action_fn needs to be a function. It is {}".format(change_action_fn)
        assert callable(change_space_fn), "change_space_fn needs to be a function. It is {}".format(change_space_fn)

        self.change_action_fn = change_action_fn
        self.change_space_fn = change_space_fn

        super().__init__(env)
        if hasattr(self, 'possible_agents'):
            for agent in self.possible_agents:
                # call any validation logic in this function
                self.action_space(agent)

    def _modify_observation(self, agent, observation):
        return observation

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        old_act_space = self.env.action_space(agent)
        try:
            return self.change_space_fn(old_act_space, agent)
        except TypeError:
            return self.change_space_fn(old_act_space)

    def _modify_action(self, agent, action):
        old_act_space = self.env.action_space(agent)
        try:
            return self.change_action_fn(action, old_act_space, agent)
        except TypeError:
            return self.change_action_fn(action, old_act_space)


class gym_action_lambda(gym.Wrapper):
    def __init__(self, env, change_action_fn, change_space_fn):
        assert callable(change_action_fn), "change_action_fn needs to be a function. It is {}".format(change_action_fn)
        assert callable(change_space_fn), "change_space_fn needs to be a function. It is {}".format(change_space_fn)
        self.change_action_fn = change_action_fn
        self.change_space_fn = change_space_fn

        super().__init__(env)
        self._modify_spaces()

    def _modify_spaces(self):
        new_space = self.change_space_fn(self.action_space)
        assert isinstance(new_space, Space), "output of change_space_fn argument to action_lambda_wrapper must be a gym space"
        self.action_space = new_space

    def _modify_action(self, action):
        return self.change_action_fn(action, self.env.action_space)

    def step(self, action):
        return super().step(self._modify_action(action))


action_lambda_v1 = WrapperChooser(aec_wrapper=aec_action_lambda, gym_wrapper=gym_action_lambda)
