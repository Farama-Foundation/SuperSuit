from .utils.base_modifier import BaseModifier
from .utils.shared_wrapper_util import shared_wrapper
from gym.spaces import Box, Discrete
from supersuit.utils.frame_stack import stack_obs_space, stack_init, stack_obs


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
