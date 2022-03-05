import copy

from pettingzoo.mpe import simple_spread_v2, simple_world_comm_v2
from pettingzoo.butterfly import knights_archers_zombies_v9
from supersuit import pettingzoo_env_to_vec_env_v1, black_death_v3, concat_vec_envs_v1
import pytest


def test_good_env():
    env = simple_spread_v2.parallel_env()
    max_num_agents = len(env.possible_agents)
    env = pettingzoo_env_to_vec_env_v1(env)
    assert env.num_envs == max_num_agents

    obss = env.reset()
    for i in range(55):
        actions = [env.action_space.sample() for i in range(env.num_envs)]

        # Check we're not passing a thing that gets mutated
        keep_obs = copy.deepcopy(obss)
        new_obss, rews, dones, infos = env.step(actions)

        assert hash(str(keep_obs)) == hash(str(obss))
        assert len(new_obss) == max_num_agents
        assert len(rews) == max_num_agents
        assert len(dones) == max_num_agents
        assert len(infos) == max_num_agents
        # no agent death, only env death
        if any(dones):
            assert all(dones)
        obss = new_obss


def test_good_vecenv():
    num_envs = 2
    env = simple_spread_v2.parallel_env()
    max_num_agents = len(env.possible_agents) * num_envs
    env = pettingzoo_env_to_vec_env_v1(env)
    env = concat_vec_envs_v1(env, num_envs)

    obss = env.reset()
    for i in range(55):
        actions = [env.action_space.sample() for i in range(env.num_envs)]

        # Check we're not passing a thing that gets mutated
        keep_obs = copy.deepcopy(obss)
        new_obss, rews, dones, infos = env.step(actions)

        assert hash(str(keep_obs)) == hash(str(obss))
        assert len(new_obss) == max_num_agents
        assert len(rews) == max_num_agents
        assert len(dones) == max_num_agents
        assert len(infos) == max_num_agents
        # no agent death, only env death
        if any(dones):
            assert all(dones)
        obss = new_obss


def test_bad_action_spaces_env():
    env = simple_world_comm_v2.parallel_env()
    with pytest.raises(AssertionError):
        env = pettingzoo_env_to_vec_env_v1(env)


def test_env_black_death_assertion():
    env = knights_archers_zombies_v9.parallel_env(spawn_rate=50, max_cycles=2000)
    env = pettingzoo_env_to_vec_env_v1(env)
    with pytest.raises(AssertionError):
        for i in range(100):
            env.reset()
            for i in range(2000):
                actions = [env.action_space.sample() for i in range(env.num_envs)]
                obss, rews, dones, infos = env.step(actions)


def test_env_black_death_wrapper():
    env = knights_archers_zombies_v9.parallel_env(spawn_rate=50, max_cycles=300)
    env = black_death_v3(env)
    env = pettingzoo_env_to_vec_env_v1(env)
    env.reset()
    for i in range(300):
        actions = [env.action_space.sample() for i in range(env.num_envs)]
        obss, rews, dones, infos = env.step(actions)
