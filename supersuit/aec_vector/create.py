from .async_vector_env import AsyncAECVectorEnv
from .vector_env import SyncAECVectorEnv
import cloudpickle


def vectorize_aec_env(aec_env, num_envs, num_cpus=0):
    def env_fn():
        return cloudpickle.loads(cloudpickle.dumps(aec_env))

    env_list = [env_fn] * num_envs

    if num_cpus == 0 or num_cpus == 1:
        return SyncAECVectorEnv(env_list)
    else:
        return AsyncAECVectorEnv(env_list, num_cpus)
