from .utils.base_modifier import BaseModifier
from .utils.shared_wrapper_util import shared_wrapper
import gym


def sticky_actions_v0(env, repeat_action_probability):
    assert 0 <= repeat_action_probability < 1

    class StickyActionsModifier(BaseModifier):
        def __init__(self):
            super().__init__()

        def reset(self, seed=None, return_info=False, options=None):
            self.np_random, _ = gym.utils.seeding.np_random(seed)
            self.old_action = None

        def modify_action(self, action):
            if (
                self.old_action is not None
                and self.np_random.uniform() < repeat_action_probability
            ):
                action = self.old_action
            self.old_action = action
            return action

    return shared_wrapper(env, StickyActionsModifier)
