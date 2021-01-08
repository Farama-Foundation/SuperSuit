import gym
import pickle
from .vector import MakeCPUAsyncConstructor, MarkovVectorEnv, SB3VecEnvWrapper
from pettingzoo.utils.env import AECEnv, ParallelEnv


def vec_env_args(env, num_envs):
    def env_fn():
        return pickle.loads(pickle.dumps(env))

    return [env_fn] * num_envs, env.observation_space, env.action_space


def gym_vec_env(env, num_envs, multiprocessing=False):
    args = vec_env_args(env, num_envs)
    constructor = gym.vector.AsyncVectorEnv if multiprocessing else gym.vector.SyncVectorEnv
    return constructor(*args)


def stable_baselines_vec_env(env, num_envs, multiprocessing=False):
    import stable_baselines

    args = vec_env_args(env, num_envs)[:1]
    constructor = stable_baselines.common.vec_env.SubprocVecEnv if multiprocessing else stable_baselines.common.vec_env.DummyVecEnv
    return constructor(*args)


def stable_baselines3_vec_env(env, num_envs, multiprocessing=False):
    import stable_baselines3

    args = vec_env_args(env, num_envs)[:1]
    constructor = stable_baselines3.common.vec_env.SubprocVecEnv if multiprocessing else stable_baselines3.common.vec_env.DummyVecEnv
    return constructor(*args)


def supersuit_vec_env(env, num_envs, num_cpus=0, base_class='gym'):
    if isinstance(env, AECEnv):
        raise ValueError("supersuit_vec_env only supports PettingZoo ParallelEnv environments and gym environments. You can import pettingzoo.utils.to_parallel and convert the AEC env to a parallel env with the to_parallel(env)")
    if isinstance(env, ParallelEnv):
        markov_env = MarkovVectorEnv(env)
        vec_env = MakeCPUAsyncConstructor(num_cpus)(*vec_env_args(markov_env, num_envs))
    else:
        vec_env = MakeCPUAsyncConstructor(num_cpus)(*vec_env_args(env, num_envs))

    if base_class == "gym":
        return vec_env
    elif base_class == "stable_baselines3":
        return SB3VecEnvWrapper(vec_env)
    else:
        raise ValueError("supersuit_vec_env only supports 'gym' and 'stable_baselines3' for its base_class")
