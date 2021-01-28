import gym
import numpy as np


class SpaceWrapper:
    def __init__(self, space):
        if isinstance(space, gym.spaces.Discrete):
            self.shape = ()
            self.dtype = np.dtype(np.int64)
        elif isinstance(space, gym.spaces.Box):
            self.shape = space.shape
            self.dtype = np.dtype(space.dtype)
        else:
            assert False, "ProcVectorEnv only support Box and Discrete types"
