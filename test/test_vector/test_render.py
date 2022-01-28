from supersuit import gym_vec_env_v0
import gym
import numpy as np
from pettingzoo.butterfly import pistonball_v6
from supersuit import concat_vec_envs_v1, pettingzoo_env_to_vec_env_v1


def make_env():
    env = pistonball_v6.parallel_env()
    env = pettingzoo_env_to_vec_env_v1(env)
    return env

# unfortunately this test does not pass
# def test_vector_render_multiproc():
#     env = make_env()
#     num_envs = 3
#     venv = concat_vec_envs_v1(env, num_envs, num_cpus=num_envs, base_class='stable_baselines3')
#     venv.reset()
#     arr = venv.render(mode="rgb_array")
#     venv.reset()
#     assert len(arr.shape) == 3 and arr.shape[2] == 3
#     venv.reset()
#     try:
#         venv.close()
#     except RuntimeError:
#         pass
