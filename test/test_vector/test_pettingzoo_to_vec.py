import copy

import numpy as np
import pytest
from mpe2 import simple_spread_v3, simple_world_comm_v3
from pettingzoo.butterfly import knights_archers_zombies_v10

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
        assert isinstance(infos, dict)
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
        assert isinstance(infos, dict)
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
    env = black_death_v3(env)
    env = pettingzoo_env_to_vec_env_v1(env)
    env.reset(seed=42)

    # run past max_cycles or until terminated - causing the env to reset and continue
    for _ in range(0, max_cycles + 10):
        actions = [env.action_space.sample() for i in range(env.num_envs)]
        _, _, terms, truncs, infos = env.step(actions)

        env_done = (np.array(terms) | np.array(truncs)).all()

        if env_done:
            # Gymnasium vector infos: dict of arrays with boolean masks
            assert isinstance(infos, dict)
            assert "terminal_observation" in infos
            assert "_terminal_observation" in infos
            assert len(infos["terminal_observation"]) == len(
                env.par_env.possible_agents
            )
            assert infos["_terminal_observation"].all()


def test_infos_are_dict_not_list():
    """Gymnasium vector envs must return infos as a dict (issue #249)."""
    env = simple_spread_v3.parallel_env()
    env = pettingzoo_env_to_vec_env_v1(env)

    obss, infos = env.reset(seed=0)
    assert isinstance(infos, dict), f"reset infos should be dict, got {type(infos)}"

    actions = [env.action_space.sample() for _ in range(env.num_envs)]
    obss, rews, terms, truncs, infos = env.step(actions)
    assert isinstance(infos, dict), f"step infos should be dict, got {type(infos)}"


def test_record_episode_statistics_compatible():
    """RecordEpisodeStatistics requires dict infos (issue #249 / CleanRL)."""
    from gymnasium.wrappers.vector import RecordEpisodeStatistics

    env = simple_spread_v3.parallel_env(max_cycles=5)
    env = pettingzoo_env_to_vec_env_v1(env)
    env = RecordEpisodeStatistics(env)

    obss, infos = env.reset(seed=0)
    assert isinstance(infos, dict)

    saw_episode = False
    for _ in range(20):
        actions = env.action_space.sample()
        # MarkovVectorEnv still exposes the unbatched action space historically;
        # sample one action per sub-env if needed.
        if (
            not hasattr(actions, "__len__")
            or len(np.atleast_1d(actions)) != env.num_envs
        ):
            actions = np.array([env.action_space.sample() for _ in range(env.num_envs)])
        obss, rews, terms, truncs, infos = env.step(actions)
        assert isinstance(infos, dict)
        if "episode" in infos:
            saw_episode = True
            assert "_episode" in infos
            assert infos["episode"]["r"].shape == (env.num_envs,)
            break

    assert saw_episode, "expected episode stats after max_cycles autoreset"


def test_infos_dict_through_concat_vec_envs():
    """concat_vec_envs_v1 must also return dict infos for Gymnasium wrappers."""
    from gymnasium.wrappers.vector import RecordEpisodeStatistics

    env = simple_spread_v3.parallel_env(max_cycles=5)
    env = pettingzoo_env_to_vec_env_v1(env)
    env = concat_vec_envs_v1(env, 2)
    assert isinstance(env.reset(seed=0)[1], dict)

    env = simple_spread_v3.parallel_env(max_cycles=5)
    env = pettingzoo_env_to_vec_env_v1(env)
    env = concat_vec_envs_v1(env, 2)
    env = RecordEpisodeStatistics(env)
    env.reset(seed=0)
    for _ in range(20):
        actions = np.array([env.action_space.sample() for _ in range(env.num_envs)])
        _, _, terms, truncs, infos = env.step(actions)
        assert isinstance(infos, dict)
        if "episode" in infos:
            assert infos["_episode"].shape == (env.num_envs,)
            break
    else:
        raise AssertionError("expected episode stats through concat_vec_envs")
