from collections import deque
import numpy as np


class Accumulator:
    def __init__(self, obs_space, memory, reduction):
        self.memory = memory
        self.reduction = reduction
        self._obs_buffer = np.zeros((memory,) + obs_space.shape, dtype=obs_space.dtype)
        self.index = 0

    def add(self, in_obs):
        self._obs_buffer[self.index] = in_obs
        self.index = (self.index + 1) % self.memory

    def get(self):
        return self.reduction.reduce(self._obs_buffer, axis=0)
