from supersuit import gym_vec_env_v0, concat_vec_envs_v0
import gym
import numpy as np


def test_vector_render_multiproc():
    env = gym.make("LunarLander-v2")
    num_envs = 3
    venv = concat_vec_envs_v0(env, num_envs, num_cpus=num_envs,base_class='stable_baselines3')
    venv.reset()
    arr = venv.render(mode="rgb_array")
    assert len(arr.shape) == 3 and arr.shape[2] == 3
    venv.close()

def test_vector_render_multiproc_human():
    env = gym.make("LunarLander-v2")
    num_envs = 3
    venv = concat_vec_envs_v0(env, num_envs, num_cpus=num_envs,base_class='stable_baselines3')
    venv.reset()
    arr = venv.render(mode="human")
    venv.close()

def test_vector_render_single_proc():
    env = gym.make("Pendulum-v0")
    num_envs = 1
    venv = concat_vec_envs_v0(env, num_envs, num_cpus=num_envs)
    venv.reset()
    venv.render(mode="rgb_array")
    venv.close()

def test_vector_render_single_proc_human():
    env = gym.make("Pendulum-v0")
    num_envs = 1
    venv = concat_vec_envs_v0(env, num_envs, num_cpus=num_envs)
    venv.reset()
    venv.render(mode="human")
    venv.close()
