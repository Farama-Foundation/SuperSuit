import gym
import cloudpickle
from . import MakeCPUAsyncConstructor, MarkovVectorEnv
from pettingzoo.utils.env import ParallelEnv
import warnings


def vec_env_args(env, num_envs):
    def env_fn():
        env_copy = cloudpickle.loads(cloudpickle.dumps(env))
        env.seed(None)
        return env_copy

    return [env_fn] * num_envs, env.observation_space, env.action_space


def warn_not_gym_env(env, fn_name):
    if not isinstance(env, gym.Env):
        warnings.warn(f"{fn_name} took in an environment which does not inherit from gym.Env. Note that gym_vec_env only takes in gym-style environments, not pettingzoo environments.")


def gym_vec_env_v0(env, num_envs, multiprocessing=False):
    warn_not_gym_env(env, "gym_vec_env")
    args = vec_env_args(env, num_envs)
    constructor = gym.vector.AsyncVectorEnv if multiprocessing else gym.vector.SyncVectorEnv
    return constructor(*args)


def stable_baselines_vec_env_v0(env, num_envs, multiprocessing=False):
    import stable_baselines

    warn_not_gym_env(env, "stable_baselines_vec_env")
    args = vec_env_args(env, num_envs)[:1]
    constructor = stable_baselines.common.vec_env.SubprocVecEnv if multiprocessing else stable_baselines.common.vec_env.DummyVecEnv
    return constructor(*args)


def stable_baselines3_vec_env_v0(env, num_envs, multiprocessing=False):
    import stable_baselines3

    warn_not_gym_env(env, "stable_baselines3_vec_env")
    args = vec_env_args(env, num_envs)[:1]
    constructor = stable_baselines3.common.vec_env.SubprocVecEnv if multiprocessing else stable_baselines3.common.vec_env.DummyVecEnv
    return constructor(*args)


def concat_vec_envs_v1(vec_env, num_vec_envs, num_cpus=0, base_class='gym'):
    num_cpus = min(num_cpus, num_vec_envs)
    vec_env = MakeCPUAsyncConstructor(num_cpus)(*vec_env_args(vec_env, num_vec_envs))

    if base_class == "gym":
        return vec_env
    elif base_class == "stable_baselines":
        from .sb_vector_wrapper import SBVecEnvWrapper
        return SBVecEnvWrapper(vec_env)
    elif base_class == "stable_baselines3":
        from .sb3_vector_wrapper import SB3VecEnvWrapper
        return SB3VecEnvWrapper(vec_env)
    else:
        raise ValueError("supersuit_vec_env only supports 'gym', 'stable_baselines', and 'stable_baselines3' for its base_class")


def pettingzoo_env_to_vec_env_v1(parallel_env):
    assert isinstance(parallel_env, ParallelEnv), "pettingzoo_env_to_vec_env takes in a pettingzoo ParallelEnv. Can create a parallel_env with pistonball.parallel_env() or convert it from an AEC env with `from pettingzoo.utils.conversions import aec_to_parallel; aec_to_parallel(env)``"
    assert hasattr(parallel_env, 'possible_agents'), "environment passed to pettingzoo_env_to_vec_env must have possible_agents attribute."
    return MarkovVectorEnv(parallel_env)
