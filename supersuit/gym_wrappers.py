from gym.spaces import Box, Space, Discrete
from .utils import basic_transforms
from .utils.frame_stack import stack_obs_space, stack_init, stack_obs
from .utils.frame_skip import check_transform_frameskip
from .utils.obs_delay import Delayer
from .utils.accumulator import Accumulator
import numpy as np
import gym


class frame_skip_gym(gym.Wrapper):
    def __init__(self, env, num_frames):
        super().__init__(env)
        self.num_frames = check_transform_frameskip(num_frames)
        self.np_random, seed = gym.utils.seeding.np_random(None)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        super().seed(seed)

    def step(self, action):
        low, high = self.num_frames
        num_skips = int(self.np_random.randint(low, high + 1))
        total_reward = 0.0

        for x in range(num_skips):
            obs, rew, done, info = super().step(action)
            total_reward += rew
            if done:
                break

        return obs, total_reward, done, info
