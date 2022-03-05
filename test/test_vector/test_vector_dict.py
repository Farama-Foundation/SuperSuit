from ..dummy_aec_env import DummyEnv
from gym.spaces import Box, Discrete, Dict, Tuple
from gym.vector.utils import concatenate, create_empty_array
import numpy as np
from pettingzoo.utils.conversions import aec_to_parallel
from supersuit import pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1

n_agents = 5


def make_env():
    test_env = DummyEnv(
        {
            str(i): {
                "feature": i * np.ones((5,), dtype=np.float32),
                "id": (i * np.ones((7,), dtype=np.float32), i * np.ones((8,), dtype=np.float32))
            }
            for i in range(n_agents)
        },
        {
            str(i): Dict(
                {
                    "feature": Box(low=0, high=10, shape=(5,)),
                    "id": Tuple([Box(low=0, high=10, shape=(7,)), Box(low=0, high=10, shape=(8,))]),
                }
            )
            for i in range(n_agents)
        },
        {
            str(i): Dict(
                {
                    "obs": Box(low=0, high=10, shape=(5,)),
                    "mask": Discrete(10),
                }
            )
            for i in range(n_agents)
        },
    )
    test_env.metadata["is_parallelizable"] = True
    return aec_to_parallel(test_env)


def dict_vec_env_test(env):
    # tests that environment really is a vectorized
    # version of the environment returned by make_env

    obss = env.reset()
    for i in range(55):
        actions = [env.action_space.sample() for i in range(env.num_envs)]
        actions = concatenate(
            env.action_space,
            actions,
            create_empty_array(env.action_space, env.num_envs),
        )
        obss, rews, dones, infos = env.step(actions)
        assert obss["feature"][1][0] == 1
        assert {
            "feature": obss['feature'][1][:],
            "id": [o[1] for o in obss['id']]
        } in env.observation_space
        # no agent death, only env death
        if any(dones):
            assert all(dones)


def test_pettingzoo_vec_env():
    env = make_env()
    env = pettingzoo_env_to_vec_env_v1(env)
    dict_vec_env_test(env)


def test_single_threaded_concatenate():
    env = make_env()
    env = pettingzoo_env_to_vec_env_v1(env)
    env = concat_vec_envs_v1(env, 2, num_cpus=1)
    dict_vec_env_test(env)


def test_multi_threaded_concatenate():
    env = make_env()
    env = pettingzoo_env_to_vec_env_v1(env)
    env = concat_vec_envs_v1(env, 2, num_cpus=2)
    dict_vec_env_test(env)


if __name__ == "__main__":
    from pettingzoo.test import parallel_api_test

    test_multi_threaded_concatenate()
    exit(0)
