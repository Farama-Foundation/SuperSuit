import copy

from supersuit import gym_vec_env_v0, stable_baselines3_vec_env_v0, concat_vec_envs_v1, pettingzoo_env_to_vec_env_v1
from pettingzoo.mpe import simple_spread_v2
import gym
import numpy as np


def recursive_equal(info1, info2):
    try:
        if info1 == info2:
            return True
    except ValueError:
        if isinstance(info1, np.ndarray) and isinstance(info2, np.ndarray):
            return np.all(np.equal(info1, info2))
        elif isinstance(info1, dict) and isinstance(info2, dict):
            return all((set(info1.keys()) == set(info2.keys()) and recursive_equal(info1[i], info2[i])) for i in info1.keys())
        elif isinstance(info1, list) and isinstance(info2, list):
            return all(recursive_equal(i1, i2) for i1, i2 in zip(info1, info2))
    return False


def check_vec_env_equivalency(venv1, venv2, check_info=True):
    # assert venv1.observation_space == venv2.observation_space
    # assert venv1.action_space == venv2.action_space

    venv1.seed(51)
    venv2.seed(51)

    obs1 = venv1.reset()
    obs2 = venv2.reset()

    for i in range(400):
        action = [venv1.action_space.sample() for env in range(venv1.num_envs)]
        assert np.all(np.equal(obs1, obs2))

        obs1, rew1, done1, info1 = venv1.step(action)
        obs2, rew2, done2, info2 = venv2.step(action)

        # uses close rather than equal due to inconsistency in reporting rewards as float32 or float64
        assert np.allclose(rew1, rew2)
        assert np.all(np.equal(done1, done2))
        assert recursive_equal(info1, info2) or not check_info


def test_gym_supersuit_equivalency():
    env = gym.make("MountainCarContinuous-v0")
    num_envs = 3
    venv1 = concat_vec_envs_v1(env, num_envs)
    venv2 = gym_vec_env_v0(env, num_envs)
    check_vec_env_equivalency(venv1, venv2)


def test_inital_state_dissimilarity():
    env = gym.make("CartPole-v0")
    venv = concat_vec_envs_v1(env, 2)
    observations = venv.reset()
    assert not np.equal(observations[0], observations[1]).all()


# we really don't want to have a stable baselines dependency even in tests
# def test_stable_baselines_supersuit_equivalency():
#     env = gym.make("MountainCarContinuous-v0")
#     num_envs = 3
#     venv1 = supersuit_vec_env(env, num_envs, base_class='stable_baselines3')
#     venv2 = stable_baselines3_vec_env(env, num_envs)
#     check_vec_env_equivalency(venv1, venv2, check_info=False)  # stable baselines does not implement info correctly

def test_mutliproc_single_proc_equivalency():
    env = gym.make("CartPole-v0")
    num_envs = 3
    venv1 = concat_vec_envs_v1(env, num_envs, num_cpus=0)  # uses single threaded vector environment
    venv2 = concat_vec_envs_v1(env, num_envs, num_cpus=4)  # uses multiprocessing vector environment
    check_vec_env_equivalency(venv1, venv2)


def test_multiagent_mutliproc_single_proc_equivalency():
    env = simple_spread_v2.parallel_env(max_cycles=10)
    env = pettingzoo_env_to_vec_env_v1(env)
    num_envs = 3
    venv1 = concat_vec_envs_v1(env, num_envs, num_cpus=0)  # uses single threaded vector environment
    venv2 = concat_vec_envs_v1(env, num_envs, num_cpus=4)  # uses multiprocessing vector environment
    check_vec_env_equivalency(venv1, venv2)


def test_multiproc_buffer():
    num_envs = 2
    env = gym.make("CartPole-v0")
    env = concat_vec_envs_v1(env, num_envs, num_cpus=2)

    obss = env.reset()
    for i in range(55):
        actions = [env.action_space.sample() for i in range(env.num_envs)]

        # Check we're not passing a thing that gets mutated
        keep_obs = copy.deepcopy(obss)
        new_obss, rews, dones, infos = env.step(actions)

        assert hash(str(keep_obs)) == hash(str(obss))

        obss = new_obss
