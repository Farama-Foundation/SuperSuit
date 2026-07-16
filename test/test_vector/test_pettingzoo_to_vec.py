import copy

import numpy as np
import pytest
from gymnasium.spaces import Box, Discrete
from mpe2 import simple_spread_v3, simple_world_comm_v3
from pettingzoo import ParallelEnv
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


class GlobalInfoParallelEnv(ParallelEnv):
    """Minimal parallel env whose ``infos`` carries a non-agent-keyed global key."""

    metadata = {"render_modes": [], "name": "global_info_v0", "is_parallelizable": True}

    def __init__(self):
        self.possible_agents = ["player_0", "player_1"]
        self.agents = self.possible_agents[:]
        self.render_mode = None
        self._obs_space = Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        self._act_space = Discrete(2)

    def observation_space(self, agent):
        return self._obs_space

    def action_space(self, agent):
        return self._act_space

    def _obs(self):
        return {a: np.zeros(3, dtype=np.float32) for a in self.agents}

    def _infos(self, tag):
        infos = {a: {"agent_local": a} for a in self.agents}
        infos["common_log"] = [tag]  # global, non-agent-keyed entry
        return infos

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        return self._obs(), self._infos("reset_event")

    def step(self, actions):
        rewards = {a: 0.0 for a in self.agents}
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        return (
            self._obs(),
            rewards,
            terminations,
            truncations,
            self._infos("step_event"),
        )

    def render(self):
        pass

    def close(self):
        pass


class TerminatingGlobalInfoParallelEnv(GlobalInfoParallelEnv):
    """Global-info env that terminates all agents on a chosen step.

    Keeps every agent present (as a black-death-padded env would) so
    MarkovVectorEnv's possible_agents assertion holds at the boundary, while
    still driving the ``env_done`` branch of ``MarkovVectorEnv.step()``.
    """

    def __init__(self, terminate_on=3):
        super().__init__()
        self._terminate_on = terminate_on
        self._steps = 0

    def reset(self, seed=None, options=None):
        self._steps = 0
        return super().reset(seed=seed, options=options)

    def step(self, actions):
        self._steps += 1
        done = self._steps >= self._terminate_on
        rewards = {a: 0.0 for a in self.agents}
        terminations = {a: done for a in self.agents}
        truncations = {a: False for a in self.agents}
        return (
            self._obs(),
            rewards,
            terminations,
            truncations,
            self._infos("step_event"),
        )


def test_global_info_keys_are_preserved():
    """Non-agent-keyed (global) info entries must survive reset() and step().

    Regression test for https://github.com/Farama-Foundation/SuperSuit/issues/261
    """
    env = pettingzoo_env_to_vec_env_v1(GlobalInfoParallelEnv())

    _, reset_infos = env.reset()
    assert len(reset_infos) == env.num_envs
    # global key propagated to every sub-env, per-agent info still present
    for info in reset_infos:
        assert info["common_log"] == ["reset_event"]
        assert "agent_local" in info

    actions = [env.action_space.sample() for _ in range(env.num_envs)]
    _, _, _, _, step_infos = env.step(actions)
    assert len(step_infos) == env.num_envs
    for info in step_infos:
        assert info["common_log"] == ["step_event"]
        assert "agent_local" in info


def test_global_info_keys_are_preserved_vecenv():
    """Global info entries also survive when stacked via concat_vec_envs_v1."""
    num_envs = 2
    env = pettingzoo_env_to_vec_env_v1(GlobalInfoParallelEnv())
    env = concat_vec_envs_v1(env, num_envs)

    _, reset_infos = env.reset()
    assert len(reset_infos) == env.num_envs
    assert all(info.get("common_log") == ["reset_event"] for info in reset_infos)

    actions = [env.action_space.sample() for _ in range(env.num_envs)]
    _, _, _, _, step_infos = env.step(actions)
    assert len(step_infos) == env.num_envs
    assert all(info.get("common_log") == ["step_event"] for info in step_infos)


def test_global_info_keys_survive_terminal_step():
    """Global info keys must survive the ``env_done`` auto-reset boundary.

    On a terminal step ``MarkovVectorEnv.step()`` internally calls ``reset()``
    and merges ``{**terminal_info, **reset_info}``, so a same-named global key
    from the fresh reset wins -- mirroring how ``terminal_observation`` and
    per-agent keys already behave. The key must not be dropped at the boundary.
    Regression test for https://github.com/Farama-Foundation/SuperSuit/issues/261
    """
    terminate_on = 3
    env = pettingzoo_env_to_vec_env_v1(
        TerminatingGlobalInfoParallelEnv(terminate_on=terminate_on)
    )
    env.reset()

    terms = infos = None
    for _ in range(terminate_on):
        actions = [env.action_space.sample() for _ in range(env.num_envs)]
        _, _, terms, _, infos = env.step(actions)

    # the chosen step terminated the episode -> env_done branch was exercised
    assert all(terms)
    assert len(infos) == env.num_envs
    for info in infos:
        # global key preserved across the auto-reset boundary ...
        assert "common_log" in info
        # ... and the fresh reset's value wins the {**inf, **reset_inf} merge
        assert info["common_log"] == ["reset_event"]
        # terminal observation is still surfaced, unaffected by the fix
        assert "terminal_observation" in info


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
            # check we have infos for all agents
            assert len(infos) == len(env.par_env.possible_agents)
            # check infos contain terminal_observation
            for info in infos:
                assert "terminal_observation" in info
