import gym
import numpy as np
import pickle

def vec_env_args(env, num_envs):
    def env_fn():
        return pickle.loads(pickle.dumps(env))
    return [env_fn]*num_envs, env.observation_space, env.action_space

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
