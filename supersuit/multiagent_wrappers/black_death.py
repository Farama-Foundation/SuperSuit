import gymnasium
import numpy as np
from pettingzoo.utils.wrappers import BaseParallelWrapper

from supersuit.utils.wrapper_chooser import WrapperChooser


def _assert_black_deathable(space):
    """Black death needs a well-defined zero observation.

    Box has one. Dict has one exactly when all of its subspaces do, so it is
    checked recursively.
    """
    if isinstance(space, gymnasium.spaces.Dict):
        for subspace in space.spaces.values():
            _assert_black_deathable(subspace)
    else:
        assert isinstance(
            space, gymnasium.spaces.Box
        ), f"observation spaces for black death must be Box or Dict spaces, is {space}"


def _zero_obs(space):
    """Zero observation matching the structure of space."""
    if isinstance(space, gymnasium.spaces.Dict):
        return {name: _zero_obs(subspace) for name, subspace in space.spaces.items()}
    return np.zeros_like(space.low)


class black_death_par(BaseParallelWrapper):
    def __init__(self, env):
        super().__init__(env)

    def _check_valid_for_black_death(self):
        for agent in self.agents:
            _assert_black_deathable(self.observation_space(agent))

    def reset(self, seed=None, options=None):
        obss, infos = self.env.reset(seed=seed, options=options)

        self.agents = self.env.agents[:]
        self._check_valid_for_black_death()
        black_obs = {
            agent: _zero_obs(self.observation_space(agent))
            for agent in self.agents
            if agent not in obss
        }
        return {**obss, **black_obs}, infos

    def step(self, actions):
        active_actions = {agent: actions[agent] for agent in self.env.agents}
        obss, rews, terms, truncs, infos = self.env.step(active_actions)
        black_obs = {
            agent: _zero_obs(self.observation_space(agent))
            for agent in self.agents
            if agent not in obss
        }
        black_rews = {agent: 0.0 for agent in self.agents if agent not in obss}
        black_infos = {agent: {} for agent in self.agents if agent not in obss}
        terminations = np.fromiter(terms.values(), dtype=bool)
        truncations = np.fromiter(truncs.values(), dtype=bool)
        env_is_done = (terminations | truncations).all()
        total_obs = {**black_obs, **obss}
        total_rews = {**black_rews, **rews}
        total_infos = {**black_infos, **infos}
        total_dones = {agent: env_is_done for agent in self.agents}
        if env_is_done:
            self.agents.clear()
        return total_obs, total_rews, total_dones, total_dones, total_infos


black_death_v3 = WrapperChooser(parallel_wrapper=black_death_par)
