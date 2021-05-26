import gym
from pettingzoo.utils.conversions import to_parallel, ParallelEnv, from_parallel
from pettingzoo.utils.env import AECEnv
# from . import aec_wrappers
# from . import gym_wrappers
# from . import parallel_wrappers
from . import vector_constructors
from . import aec_vector

__version__ = "2.6.5"




class DeprecatedWrapper(ImportError):
    pass


class Deprecated:
    def __init__(self, wrapper_name, orig_version, new_version):
        self.name = wrapper_name
        self.old_version, self.new_version = orig_version, new_version

    def __call__(self, env, *args, **kwargs):
        raise DeprecatedWrapper(f"{self.name}_{self.old_version} is now Deprecated, use {self.name}_{self.new_version} instead")

from .lambda_wrappers import action_lambda_v0, observation_lambda_v0, reward_lambda_v0



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
