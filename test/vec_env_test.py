from supersuit import gym_vec_env
import gym
import numpy as np

def test_vec_env_args():
    env = gym.make("Acrobot-v1")
    num_envs = 8
    vec_env = gym_vec_env(env, num_envs)
    vec_env.reset()
    obs, rew, dones, infos = vec_env.step([0]+[1]*(vec_env.num_envs-1))
    assert not np.any(np.equal(obs[0], obs[1]))

def test_all_vec_env_fns():
    num_envs = 8
    env = gym.make("Acrobot-v1")
    vec_env = gym_vec_env(env, num_envs, False)
    vec_env = gym_vec_env(env, num_envs, True)
