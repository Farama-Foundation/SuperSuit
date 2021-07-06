from .utils.base_modifier import BaseModifier
from .utils.shared_wrapper_util import shared_wrapper
from supersuit.utils.accumulator import Accumulator
import numpy as np


def max_observation_v0(env, memory):
    int(memory)  # delay must be an int

    class MaxObsModifier(BaseModifier):
        def reset(self):
            self.accumulator = Accumulator(self.observation_space, memory, np.maximum)

        def modify_obs(self, obs):
            self.accumulator.add(obs)
            return super().modify_obs(self.accumulator.get())

    return shared_wrapper(env, MaxObsModifier)
