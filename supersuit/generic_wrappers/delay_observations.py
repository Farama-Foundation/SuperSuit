from supersuit.utils.obs_delay import Delayer

from .utils.base_modifier import BaseModifier
from .utils.shared_wrapper_util import shared_wrapper


def delay_observations_v0(env, delay):
    assert isinstance(delay, int), f"delay must be an int, not {delay}"
    assert delay >= 0, f"delay must be at least 0, not {delay}"

    class DelayObsModifier(BaseModifier):
        def reset(self, seed=None, options=None):
            self.delayer = Delayer(self.observation_space, delay)

        def modify_obs(self, obs):
            obs = self.delayer.add(obs)
            return BaseModifier.modify_obs(self, obs)

    return shared_wrapper(env, DelayObsModifier)
