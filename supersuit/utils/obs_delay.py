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
            if isinstance(in_obs, np.ndarray):
                return np.zeros_like(in_obs)
            elif (
                isinstance(in_obs, dict)
                and "observation" in in_obs.keys()
                and "action_mask" in in_obs.keys()
            ):
                return {
                    "observation": np.zeros_like(in_obs["observation"]),
                    "action_mask": np.ones_like(in_obs["action_mask"]),
                }
            else:
                raise TypeError(
                    "Observation must be of type np.ndarray or dictionary with keys 'observation' and 'action_mask'"
                )
