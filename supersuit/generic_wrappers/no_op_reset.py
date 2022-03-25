from supersuit.utils.frame_skip import check_transform_frameskip
from supersuit.utils.wrapper_chooser import WrapperChooser
from pettingzoo.utils.wrappers import BaseWrapper, BaseParallelWraper
import gym
from supersuit.utils.make_defaultdict import make_defaultdict
import numpy as np
    
class noop_reset_gym(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert noops > 0

        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

class noop_reset_par(BaseParallelWraper):
    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = {} 
        for agent in self.possible_agents: # Check
            self.noop_action[agent] = 0

    def reset(self):
        obs = super().reset()
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert noops > 0

        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset()
        return obs


noop_reset_v0 = WrapperChooser(gym_wrapper=noop_reset_gym, parallel_wrapper=noop_reset_par)
