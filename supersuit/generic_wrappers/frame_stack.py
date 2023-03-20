from gymnasium.spaces import Box, Discrete

from supersuit.utils.frame_stack import stack_init, stack_obs, stack_obs_space

from .utils.base_modifier import BaseModifier
from .utils.shared_wrapper_util import shared_wrapper


def frame_stack_v1(env, stack_size=4, stack_dim=-1):
    assert isinstance(stack_size, int), "stack size of frame_stack must be an int"

    class FrameStackModifier(BaseModifier):
        def modify_obs_space(self, obs_space):
            if isinstance(obs_space, Box):
                assert (
                    1 <= len(obs_space.shape) <= 3
                ), "frame_stack only works for 1, 2 or 3 dimensional observations"
            elif isinstance(obs_space, Discrete):
                pass
            else:
                assert (
                    False
                ), "Stacking is currently only allowed for Box and Discrete observation spaces. The given observation space is {}".format(
                    obs_space
                )

            self.old_obs_space = obs_space
            self.observation_space = stack_obs_space(obs_space, stack_size, stack_dim)
            return self.observation_space

        def reset(self, seed=None, options=None):
            self.stack = stack_init(self.old_obs_space, stack_size, stack_dim)

        def modify_obs(self, obs):
            self.stack = stack_obs(
                self.stack, obs, self.old_obs_space, stack_size, stack_dim
            )

            return self.stack

        def get_last_obs(self):
            return self.stack

    return shared_wrapper(env, FrameStackModifier)


def frame_stack_v2(env, stack_size=4, stack_dim=-1):
    assert isinstance(stack_size, int), "stack size of frame_stack must be an int"
    assert f"stack_dim should be 0 or -1, not {stack_dim}"

    class FrameStackModifier(BaseModifier):
        def modify_obs_space(self, obs_space):
            if isinstance(obs_space, Box):
                assert (
                    1 <= len(obs_space.shape) <= 3
                ), "frame_stack only works for 1, 2 or 3 dimensional observations"
            elif isinstance(obs_space, Discrete):
                pass
            else:
                assert (
                    False
                ), "Stacking is currently only allowed for Box and Discrete observation spaces. The given observation space is {}".format(
                    obs_space
                )

            self.old_obs_space = obs_space
            self.observation_space = stack_obs_space(obs_space, stack_size, stack_dim)
            return self.observation_space

        def reset(self, seed=None, options=None):
            self.stack = stack_init(self.old_obs_space, stack_size, stack_dim)
            self.reset_flag = True

        def modify_obs(self, obs):
            if self.reset_flag:
                for _ in range(stack_size):
                    self.stack = stack_obs(
                        self.stack, obs, self.old_obs_space, stack_size, stack_dim
                    )
                self.reset_flag = False
            else:
                self.stack = stack_obs(
                    self.stack, obs, self.old_obs_space, stack_size, stack_dim
                )

            return self.stack

        def get_last_obs(self):
            return self.stack

    return shared_wrapper(env, FrameStackModifier)
