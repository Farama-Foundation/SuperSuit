from pettingzoo.mpe import simple_spread_v2, simple_world_comm_v2
from pettingzoo.butterfly import knights_archers_zombies_v5
from supersuit import pettingzoo_env_to_vec_env_v0
import pytest


def test_good_env():
    env = simple_spread_v2.parallel_env()
    max_num_agents = len(env.possible_agents)
    env = pettingzoo_env_to_vec_env_v0(env)
    assert env.num_envs == max_num_agents

    env.reset()
    for i in range(55):
        actions = [env.action_space.sample() for i in range(env.num_envs)]
        obss, rews, dones, infos = env.step(actions)
        assert len(obss) == max_num_agents
        assert len(rews) == max_num_agents
        assert len(dones) == max_num_agents
        assert len(infos) == max_num_agents
        # no agent death, only env death
        if any(dones):
            assert all(dones)


def test_bad_action_spaces_env():
    env = simple_world_comm_v2.parallel_env()
    with pytest.raises(AssertionError):
        env = pettingzoo_env_to_vec_env_v0(env)


def test_env_black_death_assertion():
    env = knights_archers_zombies_v5.parallel_env(spawn_rate=50, max_cycles=2000)
    env = pettingzoo_env_to_vec_env_v0(env)
    with pytest.raises(AssertionError):
        env.reset()
        for i in range(2000):
            actions = [env.action_space.sample() for i in range(env.num_envs)]
            obss, rews, dones, infos = env.step(actions)


def test_env_black_death_option():
    env = knights_archers_zombies_v5.parallel_env(spawn_rate=50, max_cycles=300)
    env = pettingzoo_env_to_vec_env_v0(env, black_death=True)
    env.reset()
    for i in range(300):
        actions = [env.action_space.sample() for i in range(env.num_envs)]
        obss, rews, dones, infos = env.step(actions)
