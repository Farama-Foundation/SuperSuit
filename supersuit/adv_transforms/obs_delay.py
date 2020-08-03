from collections import deque
import numpy as np

class Delayer:
    def __init__(self, obs_space, delay):
        self.delay = delay
        self.obs_queue = deque()
        self.obs_space = obs_space

    def add(self, in_obs):
        self.obs_queue.append(in_obs)
        if len(self.obs_queue) > self.delay:
            return self.obs_queue.popleft()
        else:
            return np.zeros_like(in_obs)
