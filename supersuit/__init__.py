import gym
import importlib

class WrapperFactory:
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

color_reduction = WrapperFactory("color_reduction")
down_scale = WrapperFactory("down_scale")
dtype = WrapperFactory("dtype")
flatten = WrapperFactory("flatten")
frame_stack = WrapperFactory("frame_stack")
normalize_obs = WrapperFactory("normalize_obs")
reshape = WrapperFactory("reshape")
agent_indicator = WrapperFactory("agent_indicator")
pad_action_space = WrapperFactory("pad_action_space")
pad_observations = WrapperFactory("pad_observations")
clip_reward = WrapperFactory("clip_reward")
action_lambda = WrapperFactory("action_lambda")
observation_lambda = WrapperFactory("observation_lambda")
reward_lambda = WrapperFactory("reward_lambda")

from .vector_constructors import gym_vec_env
from .vector_constructors import stable_baselines_vec_env
from .vector_constructors import stable_baselines3_vec_env
