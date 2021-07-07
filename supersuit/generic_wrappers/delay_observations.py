from .utils.base_modifier import BaseModifier
from .utils.shared_wrapper_util import shared_wrapper
from supersuit.utils.obs_delay import Delayer


def delay_observations_v0(env, delay):
    class DelayObsModifier(BaseModifier):
        def reset(self):
            self.delayer = Delayer(self.observation_space, delay)

        def modify_obs(self, obs):
            obs = self.delayer.add(obs)
            return BaseModifier.modify_obs(self, obs)

    return shared_wrapper(env, DelayObsModifier)
