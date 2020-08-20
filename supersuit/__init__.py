import gym
import importlib

__version__ = "1.1.1"

class WrapperFactory:
    def __init__(self, wrapper_name):
        self.wrapper_name = wrapper_name

    def __call__(self, env, *args, **kwargs):
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

color_reduction = WrapperFactory("color_reduction")
resize = WrapperFactory("resize")
dtype = WrapperFactory("dtype")
flatten = WrapperFactory("flatten")
frame_stack = WrapperFactory("frame_stack")
normalize_obs = WrapperFactory("normalize_obs")
reshape = WrapperFactory("reshape")
clip_reward = WrapperFactory("clip_reward")
action_lambda = WrapperFactory("action_lambda")
clip_actions = WrapperFactory("clip_actions")
observation_lambda = WrapperFactory("observation_lambda")
reward_lambda = WrapperFactory("reward_lambda")
frame_skip = WrapperFactory("frame_skip")
sticky_actions = WrapperFactory("sticky_actions")
delay_observations = WrapperFactory("delay_observations")

from .aec_wrappers import agent_indicator, pad_action_space, pad_observations
from .vector_constructors import gym_vec_env, stable_baselines_vec_env, stable_baselines3_vec_env
