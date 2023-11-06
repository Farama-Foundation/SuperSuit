import copy

import numpy as np
import pytest
from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.mpe import simple_spread_v3, simple_world_comm_v3

from supersuit import black_death_v3, concat_vec_envs_v1, pettingzoo_env_to_vec_env_v1


def test_good_env():
    env = simple_spread_v3.parallel_env()
    max_num_agents = len(env.possible_agents)
    env = pettingzoo_env_to_vec_env_v1(env)
    assert env.num_envs == max_num_agents

    obss, infos = env.reset()
    for i in range(55):
        actions = [env.action_space.sample() for i in range(env.num_envs)]

        # Check we're not passing a thing that gets mutated
        keep_obs = copy.deepcopy(obss)
        new_obss, rews, terms, truncs, infos = env.step(actions)

        assert hash(str(keep_obs)) == hash(str(obss))
        assert len(new_obss) == max_num_agents
        assert len(rews) == max_num_agents
        assert len(terms) == max_num_agents
        assert len(truncs) == max_num_agents
        assert len(infos) == max_num_agents
        # no agent death, only env death
        if any(terms):
            assert all(terms)
        if any(truncs):
            assert all(truncs)
        obss = new_obss


def test_good_vecenv():
    num_envs = 2
    env = simple_spread_v3.parallel_env()
    max_num_agents = len(env.possible_agents) * num_envs
    env = pettingzoo_env_to_vec_env_v1(env)
    env = concat_vec_envs_v1(env, num_envs)

    obss, infos = env.reset()
    for i in range(55):
        actions = [env.action_space.sample() for i in range(env.num_envs)]

        # Check we're not passing a thing that gets mutated
        keep_obs = copy.deepcopy(obss)
        new_obss, rews, terms, truncs, infos = env.step(actions)

        assert hash(str(keep_obs)) == hash(str(obss))
        assert len(new_obss) == max_num_agents
        assert len(rews) == max_num_agents
        assert len(terms) == max_num_agents
        assert len(truncs) == max_num_agents
        assert len(infos) == max_num_agents
        # no agent death, only env death
        if any(terms):
            assert all(terms)
        if any(truncs):
            assert all(truncs)
        obss = new_obss


def test_bad_action_spaces_env():
    env = simple_world_comm_v3.parallel_env()
    with pytest.raises(AssertionError):
        env = pettingzoo_env_to_vec_env_v1(env)


def test_env_black_death_assertion():
    env = knights_archers_zombies_v10.parallel_env(spawn_rate=50, max_cycles=2000)
    env = pettingzoo_env_to_vec_env_v1(env)
    with pytest.raises(AssertionError):
        for i in range(100):
            env.reset()
            for i in range(2000):
                actions = [env.action_space.sample() for i in range(env.num_envs)]
                obss, rews, terms, truncs, infos = env.step(actions)


def test_env_black_death_wrapper():
    env = knights_archers_zombies_v10.parallel_env(spawn_rate=50, max_cycles=300)
    env = black_death_v3(env)
    env = pettingzoo_env_to_vec_env_v1(env)
    env.reset()
    for i in range(300):
        actions = [env.action_space.sample() for i in range(env.num_envs)]
        obss, rews, terms, truncs, infos = env.step(actions)


def test_terminal_obs_are_returned():
    """
    If we reach (and pass) the end of the episode, the last observation is returned in the info dict.
    """
    max_cycles = 300
    env = knights_archers_zombies_v10.parallel_env(spawn_rate=50, max_cycles=300)
    env = pettingzoo_env_to_vec_env_v1(env)
    env.reset(seed=42)

    # run past max_cycles or until terminated - causing the env to reset and continue
    for _ in range(0, max_cycles + 10):
        actions = [env.action_space.sample() for i in range(env.num_envs)]
        _, _, terms, truncs, infos = env.step(actions)

        env_done = (np.array(terms) | np.array(truncs)).all()

        if env_done:
            # check we have infos for all agents
            assert len(infos) == len(env.par_env.possible_agents)
            # check infos contain terminal_observation
            for info in infos:
                assert "terminal_observation" in info
