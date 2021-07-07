from .utils.base_modifier import BaseModifier
from .utils.shared_wrapper_util import shared_wrapper
import warnings
from supersuit.lambda_wrappers import action_lambda_v1
import numpy as np
import gym


def nan_random_v0(env):
    class NanRandomModifier(BaseModifier):
        def __init__(self):
            super().__init__()
            self.seed()

        def modify_action(self, action):
            if action is not None and np.isnan(action).any():
                obs = self.cur_obs
                if isinstance(obs, dict) and 'action mask' in obs:
                    warnings.warn("[WARNING]: Step received an NaN action {}. Environment is {}. Taking a random action from 'action mask'.".format(action, self))
                    action = self.np_random.choice(np.flatnonzero(obs['action_mask']))
                else:
                    warnings.warn("[WARNING]: Step received an NaN action {}. Environment is {}. Taking a random action.".format(action, self))
                    action = self.action_space.sample()
            return action

        def seed(self, seed=None):
            self.np_random, seed = gym.utils.seeding.np_random(seed)

    return shared_wrapper(env, NanRandomModifier)


def nan_noop_v0(env, no_op_action):
    def on_action(action, action_space):
        if np.isnan(action).any():
            warnings.warn("[WARNING]: Step received an NaN action {}. Evironment is {}. Taking no operation action.".format(action, env))
            action = no_op_action
        return action
    return action_lambda_v1(env, on_action, lambda act_space: act_space)


def nan_zeros_v0(env):
    def on_action(action, action_space):
        if np.isnan(action).any():
            warnings.warn("[WARNING]: Step received an NaN action {}. Environment is {}. Taking the all zeroes action.".format(action, env))
            action = np.zeros_like(action)
        return action
    return action_lambda_v1(env, on_action, lambda act_space: act_space)
