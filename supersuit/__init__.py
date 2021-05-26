import gym
from pettingzoo.utils.conversions import to_parallel, ParallelEnv, from_parallel
from pettingzoo.utils.env import AECEnv
# from . import aec_wrappers
# from . import gym_wrappers
# from . import parallel_wrappers
from . import vector_constructors
from . import aec_vector

__version__ = "2.6.5"


class WrapperChooser:
    def __init__(self, aec_wrapper=None, gym_wrapper=None, parallel_wrapper=None):
        assert aec_wrapper is not None or parallel_wrapper is not None, "either the aec wrapper or the parallel wrapper must be defined for all supersuit environments"
        self.aec_wrapper = aec_wrapper
        self.gym_wrapper = gym_wrapper
        self.parallel_wrapper = parallel_wrapper

    def __call__(self, env, *args, **kwargs):
        if isinstance(env, gym.Env):
            if self.gym_wrapper is None:
                raise ValueError(f"{self.wrapper_name} does not apply to gym environments, pettingzoo environments only")
            return self.gym_wrapper(env, *args, **kwargs)
        elif isinstance(env, AECEnv):
            if self.aec_wrapper is not None:
                return self.aec_wrapper(env, *args, **kwargs)
            else:
                return from_parallel(self.parallel_wrapper(to_parallel(env), *args, **kwargs))
        elif isinstance(env, ParallelEnv):
            if self.parallel_wrapper is not None:
                return self.parallel_wrapper(env, *args, **kwargs)
            else:
                return to_parallel(self.aec_wrapper(from_parallel(env), *args, **kwargs))
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

from .lambda_wrappers.action_lambda import gym_action_lambda, aec_action_lambda
from .lambda_wrappers.observation_lambda import gym_observation_lambda, aec_observation_lambda
from .lambda_wrappers.reward_lambda import gym_reward_lambda, aec_reward_lambda

observation_lambda_v0 = WrapperChooser(aec_wrapper=aec_observation_lambda, gym_wrapper=gym_observation_lambda)
action_lambda_v0 = WrapperChooser(aec_wrapper=aec_action_lambda, gym_wrapper=gym_action_lambda)
reward_lambda_v0 = WrapperChooser(aec_wrapper=aec_reward_lambda, gym_wrapper=gym_reward_lambda)


# black_death_v0 = Deprecated("black_death", "v0", "v1")
# black_death_v1 = WrapperFactory("black_death", False)
# agent_indicator_v0 = WrapperFactory("agent_indicator", False)
# pad_action_space_v0 = WrapperFactory("pad_action_space", False)
# pad_observations_v0 = WrapperFactory("pad_observations", False)

gym_vec_env_v0 = vector_constructors.gym_vec_env
stable_baselines_vec_env_v0 = vector_constructors.stable_baselines_vec_env
stable_baselines3_vec_env_v0 = vector_constructors.stable_baselines3_vec_env
vectorize_aec_env_v0 = aec_vector.vectorize_aec_env
concat_vec_envs_v0 = vector_constructors.concat_vec_envs
pettingzoo_env_to_vec_env_v0 = vector_constructors.pettingzoo_env_to_vec_env
