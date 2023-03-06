import gymnasium
import numpy as np

from supersuit import gym_vec_env_v0


def test_vec_env_args():
    env = gymnasium.make("Acrobot-v1")
    num_envs = 8
    vec_env = gym_vec_env_v0(env, num_envs)
    vec_env.reset()
    obs, rew, terminations, truncations, infos = vec_env.step(
        [0] + [1] * (vec_env.num_envs - 1)
    )
    assert not np.any(np.equal(obs[0], obs[1]))


def test_all_vec_env_fns():
    num_envs = 8
    env = gymnasium.make("Acrobot-v1")
    gym_vec_env_v0(env, num_envs, False)
    gym_vec_env_v0(env, num_envs, True)
