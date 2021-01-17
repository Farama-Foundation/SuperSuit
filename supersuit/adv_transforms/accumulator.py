from collections import deque
import numpy as np
from functools import reduce


class Accumulator:
    def __init__(self, obs_space, memory, reduction):
        self.memory = memory
        self._obs_buffer = deque()
        self.reduction = reduction
        self.maxed_val = None

    def add(self, in_obs):
        self._obs_buffer.append(np.copy(in_obs))
        if len(self._obs_buffer) > self.memory:
            self._obs_buffer.popleft()
        self.maxed_val = None

    def get(self):
        if self.maxed_val is None:
            self.maxed_val = reduce(self.reduction, (self._obs_buffer))
        return self.maxed_val
