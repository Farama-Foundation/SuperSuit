from .async_vector_env import AsyncAECVectorEnv
from .vector_env import SyncAECVectorEnv
from pettingzoo import AECEnv
import cloudpickle


def vectorize_aec_env_v0(aec_env, num_envs, num_cpus=0):
    assert isinstance(aec_env, AECEnv), "pettingzoo_env_to_vec_env takes in a pettingzoo AECEnv."
    assert hasattr(aec_env, 'possible_agents'), "environment passed to vectorize_aec_env must have possible_agents attribute."

    def env_fn():
        return cloudpickle.loads(cloudpickle.dumps(aec_env))

    env_list = [env_fn] * num_envs

    if num_cpus == 0 or num_cpus == 1:
        return SyncAECVectorEnv(env_list)
    else:
        return AsyncAECVectorEnv(env_list, num_cpus)
