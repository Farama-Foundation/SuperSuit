import gym
import importlib

__all__ = [
    "color_reduction",
    "continuous_actions",
    "down_scale",
    "dtype",
    "flatten",
    "frame_stack",
    "normalize_obs",
    "reshape",
    "agent_indicator",
    "pad_action_space",
    "pad_observations",
    "clip_reward",
    "action_lambda",
    "observation_lambda",
    "reward_lambda"
]

class MetaWrapper:
    def __init__(self, wrapper_name):
        self.wrapper_name = wrapper_name

    def __call__(self, env, *args, **kwargs):
        print(type(env))
        if isinstance(env, gym.Env):
            from . import gym_wrappers
            wrap_class = getattr(gym_wrappers, self.wrapper_name)
            return wrap_class(env, *args, **kwargs)
        else:
            from . import aec_wrappers
            from pettingzoo import AECEnv
            assert isinstance(env, AECEnv), "environment must either be a gym environment or a pettingzoo environment"
            wrap_class = getattr(aec_wrappers, self.wrapper_name)
            return wrap_class(env, *args, **kwargs)


def __getattr__(wrapper_name):
    if wrapper_name in __all__:
        return MetaWrapper(wrapper_name)
    else:
        raise ImportError(f"cannot import name '{wrapper_name}' from 'supersuit'")
