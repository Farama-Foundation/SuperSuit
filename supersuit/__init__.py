import gym
import importlib
from pettingzoo.utils.to_parallel import to_parallel, ParallelEnv, from_parallel
from pettingzoo.utils.env import AECEnv
from . import aec_wrappers
from . import gym_wrappers
from . import parallel_wrappers

__version__ = "1.2.0" 

class WrapperFactory:
    def __init__(self, wrapper_name, gym_avaliable=True):
        self.wrapper_name = wrapper_name
        self.gym_avaliable = gym_avaliable

    def __call__(self, env, *args, **kwargs):
        if isinstance(env, gym.Env):
            if not self.gym_avaliable:
                raise ValueError(f"{self.wrapper_name} does not apply to gym environments, pettingzoo environments only")
            wrap_class = getattr(gym_wrappers, self.wrapper_name)
            return wrap_class(env, *args, **kwargs)
        elif isinstance(env, AECEnv):
            wrap_class = getattr(aec_wrappers, self.wrapper_name)
            return wrap_class(env, *args, **kwargs)
        elif isinstance(env, ParallelEnv):
            wrap_class = getattr(parallel_wrappers, self.wrapper_name, None)
            if wrap_class is not None:
                return wrap_class(env, *args, **kwargs)
            else:
                wrap_class = getattr(aec_wrappers, self.wrapper_name)
                return to_parallel(wrap_class(from_parallel(env), *args, **kwargs))
        else:
            raise ValueError("environment passed to supersuit wrapper must either be a gym environment or a pettingzoo environment")

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

agent_indicator = WrapperFactory("agent_indicator", False)
pad_action_space = WrapperFactory("pad_action_space", False)
pad_observations = WrapperFactory("pad_observations", False)

from .vector_constructors import gym_vec_env, stable_baselines_vec_env, stable_baselines3_vec_env
