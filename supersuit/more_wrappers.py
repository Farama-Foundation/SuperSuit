from .utils.base_modifier import BaseModifier
from .utils.shared_wrapper_util import shared_wrapper
from .utils.obs_delay import Delayer
from gym.spaces import Box, Discrete
from .utils.frame_stack import stack_obs_space, stack_init, stack_obs
from .utils.accumulator import Accumulator
import numpy as np
import gym


def delay_observations_v0(env, delay):
    class DelayObsModifier(BaseModifier):
        def reset(self):
            self.delayer = Delayer(self.observation_space, delay)

        def modify_obs(self, obs):
            obs = self.delayer.add(obs)
            return BaseModifier.modify_obs(self, obs)

    return shared_wrapper(env, DelayObsModifier)


def frame_stack_v1(env, stack_size=4):
    assert isinstance(stack_size, int), "stack size of frame_stack must be an int"
    class FrameStackModifier(BaseModifier):
        def modify_obs_space(self, obs_space):
            if isinstance(obs_space, Box):
                assert 1 <= len(obs_space.shape) <= 3, "frame_stack only works for 1, 2 or 3 dimensional observations"
            elif isinstance(obs_space, Discrete):
                pass
            else:
                assert False, "Stacking is currently only allowed for Box and Discrete observation spaces. The given observation space is {}".format(obs_space)

            self.old_obs_space = obs_space
            self.observation_space = stack_obs_space(obs_space, stack_size)
            return self.observation_space

        def reset(self):
            self.stack = stack_init(self.old_obs_space, stack_size)

        def modify_obs(self, obs):
            self.stack = stack_obs(
                self.stack,
                obs,
                self.old_obs_space,
                stack_size,
            )
            return self.stack

        def get_last_obs(self):
            return self.stack

    return shared_wrapper(env, FrameStackModifier)


def max_observation_v0(env, memory):
    int(memory)  # delay must be an int

    class MaxObsModifier(BaseModifier):
        def reset(self):
            self.accumulator = Accumulator(self.observation_space, memory, np.maximum)

        def modify_obs(self, obs):
            self.accumulator.add(obs)

        def get_last_obs(self):
            return self.accumulator.get()

    return shared_wrapper(env, MaxObsModifier)


def sticky_actions_v0(env, repeat_action_probability):
    assert 0 <= repeat_action_probability < 1

    np_random, _ = gym.utils.seeding.np_random(None)

    class StickyActionsModifier(BaseModifier):
        def reset(self):
            self.old_action = None

        def seed(self, seed):
            np_random.seed(seed)

        def modify_action(self, action):
            if self.old_action is not None and np_random.uniform() < repeat_action_probability:
                action = self.old_action
            return action

    return shared_wrapper(env, StickyActionsModifier)
