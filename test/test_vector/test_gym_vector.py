from supersuit import supersuit_vec_env, gym_vec_env, stable_baselines3_vec_env
from pettingzoo.mpe import simple_spread_v2
import gym
import numpy as np


def check_vec_env_equivalency(venv1, venv2, check_info=True):
    # assert venv1.observation_space == venv2.observation_space
    # assert venv1.action_space == venv2.action_space

    venv1.seed(51)
    venv2.seed(51)

    obs1 = venv1.reset()
    obs2 = venv2.reset()

    for i in range(1000):
        action = [venv1.action_space.sample() for env in range(venv1.num_envs)]
        assert np.all(np.equal(obs1, obs2))

        obs1, rew1, done1, info1 = venv1.step(action)
        obs2, rew2, done2, info2 = venv2.step(action)

        # uses close rather than equal due to inconsistency in reporting rewards as float32 or float64
        assert np.allclose(rew1, rew2)
        assert np.all(np.equal(done1, done2))
        assert info1 == info2 or not check_info


def test_gym_supersuit_equivalency():
    env = gym.make("Pendulum-v0")
    num_envs = 3
    venv1 = supersuit_vec_env(env, num_envs)
    venv2 = gym_vec_env(env, num_envs)
    check_vec_env_equivalency(venv1, venv2)


def test_stable_baselines_supersuit_equivalency():
    env = gym.make("Pendulum-v0")
    num_envs = 3
    venv1 = supersuit_vec_env(env, num_envs, base_class='stable_baselines3')
    venv2 = stable_baselines3_vec_env(env, num_envs)
    check_vec_env_equivalency(venv1, venv2, check_info=False)  # stable baselines does not implement info correctly


def test_mutliproc_single_proc_equivalency():
    env = gym.make("Pendulum-v0")
    num_envs = 3
    venv1 = supersuit_vec_env(env, num_envs, num_cpus=0)  # uses single threaded vector environment
    venv2 = supersuit_vec_env(env, num_envs, num_cpus=4)  # uses multiprocessing vector environment
    check_vec_env_equivalency(venv1, venv2)


def test_multiagent_mutliproc_single_proc_equivalency():
    env = simple_spread_v2.parallel_env()
    num_envs = 3
    venv1 = supersuit_vec_env(env, num_envs, num_cpus=0)  # uses single threaded vector environment
    venv2 = supersuit_vec_env(env, num_envs, num_cpus=4)  # uses multiprocessing vector environment
    check_vec_env_equivalency(venv1, venv2)
