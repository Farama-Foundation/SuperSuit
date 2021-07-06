from gym.spaces import Space
from supersuit.utils.base_aec_wrapper import BaseWrapper
from supersuit.utils.wrapper_chooser import WrapperChooser
from supersuit.utils.deprecated import Deprecated
import gym


class aec_action_lambda(BaseWrapper):
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

    def _modify_observation(self, agent, observation):
        return observation

    def _modify_spaces(self):
        new_spaces = {}
        for agent, space in self.action_spaces.items():
            new_spaces[agent] = self.change_space_fn(space, agent)
            assert isinstance(new_spaces[agent], Space), "output of change_space_fn argument to action_lambda_wrapper must be a gym space"

        self.action_spaces = new_spaces

    def _modify_action(self, agent, action):
        return self.change_action_fn(action, self.env.action_spaces[agent], agent)


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


action_lambda_v0 = Deprecated("action_lambda", "v0", "v1")
action_lambda_v1 = WrapperChooser(aec_wrapper=aec_action_lambda, gym_wrapper=gym_action_lambda)
