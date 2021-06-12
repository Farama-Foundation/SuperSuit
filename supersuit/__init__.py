import gym
from pettingzoo.utils.conversions import to_parallel, ParallelEnv, from_parallel
from pettingzoo.utils.env import AECEnv
from . import aec_wrappers
from . import gym_wrappers
from . import parallel_wrappers
from . import vector_constructors
from . import aec_vector

__version__ = "2.6.6"


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
            wrap_class = getattr(aec_wrappers, self.wrapper_name, None)

            if wrap_class is not None:
                return wrap_class(env, *args, **kwargs)
            else:
                wrap_class = getattr(parallel_wrappers, self.wrapper_name)
                return from_parallel(wrap_class(to_parallel(env), *args, **kwargs))
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


class DeprecatedWrapper(ImportError):
    pass


class Deprecated:
    def __init__(self, wrapper_name, orig_version, new_version):
        self.name = wrapper_name
        self.old_version, self.new_version = orig_version, new_version

    def __call__(self, env, *args, **kwargs):
        raise DeprecatedWrapper(f"{self.name}_{self.old_version} is now Deprecated, use {self.name}_{self.new_version} instead")


color_reduction_v0 = WrapperFactory("color_reduction")
resize_v0 = WrapperFactory("resize")
dtype_v0 = WrapperFactory("dtype")
flatten_v0 = WrapperFactory("flatten")
frame_stack_v0 = Deprecated("frame_stack", "v0", "v1")
frame_stack_v1 = WrapperFactory("frame_stack")
normalize_obs_v0 = WrapperFactory("normalize_obs")
reshape_v0 = WrapperFactory("reshape")
clip_reward_v0 = WrapperFactory("clip_reward")
action_lambda_v1 = Deprecated("action_lambda", "v0", "v1")
action_lambda_v1 = WrapperFactory("action_lambda")
clip_actions_v0 = WrapperFactory("clip_actions")
observation_lambda_v0 = WrapperFactory("observation_lambda")
reward_lambda_v0 = WrapperFactory("reward_lambda")
frame_skip_v0 = WrapperFactory("frame_skip")
sticky_actions_v0 = WrapperFactory("sticky_actions")
delay_observations_v0 = WrapperFactory("delay_observations")
max_observation_v0 = WrapperFactory("max_observation")

black_death_v0 = Deprecated("black_death", "v0", "v1")
black_death_v1 = WrapperFactory("black_death", False)
agent_indicator_v0 = WrapperFactory("agent_indicator", False)
pad_action_space_v0 = WrapperFactory("pad_action_space", False)
pad_observations_v0 = WrapperFactory("pad_observations", False)

gym_vec_env_v0 = vector_constructors.gym_vec_env
stable_baselines_vec_env_v0 = vector_constructors.stable_baselines_vec_env
stable_baselines3_vec_env_v0 = vector_constructors.stable_baselines3_vec_env
vectorize_aec_env_v0 = aec_vector.vectorize_aec_env
concat_vec_envs_v0 = vector_constructors.concat_vec_envs
pettingzoo_env_to_vec_env_v0 = vector_constructors.pettingzoo_env_to_vec_env
