import warnings

import gymnasium
import numpy as np

from supersuit.lambda_wrappers import action_lambda_v1

from .utils.base_modifier import BaseModifier
from .utils.shared_wrapper_util import shared_wrapper


def nan_random_v0(env):
    class NanRandomModifier(BaseModifier):
        def __init__(self):
            super().__init__()

        def reset(self, seed=None, options=None):
            self.np_random, seed = gymnasium.utils.seeding.np_random(seed)

            return super().reset(seed, options=options)

        def modify_action(self, action):
            if action is not None and np.isnan(action).any():
                obs = self.cur_obs
                if isinstance(obs, dict) and "action mask" in obs:
                    warnings.warn(
                        "[WARNING]: Step received an NaN action {}. Environment is {}. Taking a random action from 'action mask'.".format(
                            action, self
                        )
                    )
                    action = self.np_random.choice(np.flatnonzero(obs["action_mask"]))
                else:
                    warnings.warn(
                        "[WARNING]: Step received an NaN action {}. Environment is {}. Taking a random action.".format(
                            action, self
                        )
                    )
                    action = self.action_space.sample()

            return action

    return shared_wrapper(env, NanRandomModifier)


def nan_noop_v0(env, no_op_action):
    def on_action(action, action_space):
        if action is None:
            warnings.warn(
                "[WARNING]: Step received an None action {}. Environment is {}. Taking no operation action.".format(
                    action, env
                )
            )
            return None
        if np.isnan(action).any():
            warnings.warn(
                "[WARNING]: Step received an NaN action {}. Environment is {}. Taking no operation action.".format(
                    action, env
                )
            )
            return no_op_action
        return action

    return action_lambda_v1(env, on_action, lambda act_space: act_space)


def nan_zeros_v0(env):
    def on_action(action, action_space):
        if action is None:
            warnings.warn(
                "[WARNING]: Step received an None action {}. Environment is {}. Taking the all zeroes action.".format(
                    action, env
                )
            )
            return None
        if np.isnan(action).any():
            warnings.warn(
                "[WARNING]: Step received an NaN action {}. Environment is {}. Taking the all zeroes action.".format(
                    action, env
                )
            )
            return np.zeros_like(action)
        return action

    return action_lambda_v1(env, on_action, lambda act_space: act_space)
