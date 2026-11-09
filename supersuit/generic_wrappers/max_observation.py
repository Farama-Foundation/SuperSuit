import numpy as np

from supersuit.utils.accumulator import Accumulator

from .utils.base_modifier import BaseModifier
from .utils.shared_wrapper_util import shared_wrapper


def max_observation_v0(env, memory):
    assert isinstance(memory, int), f"memory must be an int, not {memory}"
    assert memory >= 1, f"memory must be at least 1, not {memory}"

    class MaxObsModifier(BaseModifier):
        def reset(self, seed=None, options=None):
            self.accumulator = Accumulator(self.observation_space, memory, np.maximum)

        def modify_obs(self, obs):
            self.accumulator.add(obs)
            return super().modify_obs(self.accumulator.get())

    return shared_wrapper(env, MaxObsModifier)
