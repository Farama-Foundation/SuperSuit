import numpy as np
from pettingzoo.utils.agent_selector import agent_selector

from .base_aec_vec_env import VectorAECEnv


class SyncAECVectorEnv(VectorAECEnv):
    def __init__(self, env_constructors):
        assert len(env_constructors) >= 1
        assert callable(
            env_constructors[0]
        ), "env_constructor must be a callable object (i.e function) that create an environment"

        self.envs = [env_constructor() for env_constructor in env_constructors]
        self.num_envs = len(env_constructors)
        self.env = self.envs[0]
        self.max_num_agents = self.env.max_num_agents
        self.possible_agents = self.env.possible_agents
        self._agent_selector = agent_selector(self.possible_agents)

    def action_space(self, agent):
        return self.env.action_space(agent)

    def observation_space(self, agent):
        return self.env.observation_space(agent)

    def _find_active_agent(self):
        cur_selection = self.agent_selection
        while not any(cur_selection == env.agent_selection for env in self.envs):
            cur_selection = self._agent_selector.next()
        return cur_selection

    def _collect_dicts(self):
        self.rewards = {
            agent: np.array(
                [
                    env.rewards[agent] if agent in env.rewards else 0
                    for env in self.envs
                ],
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }
        self._cumulative_rewards = {
            agent: np.array(
                [
                    env._cumulative_rewards[agent]
                    if agent in env._cumulative_rewards
                    else 0
                    for env in self.envs
                ],
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }
        self.terminations = {
            agent: np.array(
                [
                    env.terminations[agent] if agent in env.terminations else True
                    for env in self.envs
                ],
                dtype=np.uint8,
            )
            for agent in self.possible_agents
        }
        self.truncations = {
            agent: np.array(
                [
                    env.truncations[agent] if agent in env.truncations else True
                    for env in self.envs
                ],
                dtype=np.uint8,
            )
            for agent in self.possible_agents
        }
        self.infos = {
            agent: [env.infos[agent] if agent in env.infos else {} for env in self.envs]
            for agent in self.possible_agents
        }

    def reset(self, seed=None, options=None):
        """
        returns: list of observations
        """
        if seed is not None:
            for i, env in enumerate(self.envs):
                env.reset(seed=seed + i, options=options)
        else:
            for i, env in enumerate(self.envs):
                env.reset(seed=None, options=options)

        self.agent_selection = self._agent_selector.reset()
        self.agent_selection = self._find_active_agent()

        self._collect_dicts()
        self.envs_terminations = np.zeros(self.num_envs)
        self.envs_truncations = np.zeros(self.num_envs)

    def observe(self, agent):
        observations = []
        for env in self.envs:
            obs = (
                env.observe(agent)
                if (agent in env.terminations) or (agent in env.truncations)
                else np.zeros_like(env.observation_space(agent).low)
            )
            observations.append(obs)
        return np.stack(observations)

    def last(self, observe=True):
        passes = np.array(
            [env.agent_selection != self.agent_selection for env in self.envs],
            dtype=np.uint8,
        )
        last_agent = self.agent_selection
        obs = self.observe(last_agent) if observe else None
        return (
            obs,
            self._cumulative_rewards[last_agent],
            self.terminations[last_agent],
            self.truncations[last_agent],
            self.envs_terminations,
            self.envs_truncations,
            passes,
            self.infos[last_agent],
        )

    def step(self, actions, observe=True):
        assert len(actions) == len(
            self.envs
        ), f"{len(actions)} actions given, but there are {len(self.envs)} environments!"
        old_agent = self.agent_selection

        envs_dones = []
        for i, (act, env) in enumerate(zip(actions, self.envs)):
            # Prior to the truncation API update, the env was reset if env.agents was an empty list
            # After the truncation API update, the env needs to be reset if every agent is terminated OR truncated
            terminations = np.fromiter(env.terminations.values(), dtype=bool)
            truncations = np.fromiter(env.truncations.values(), dtype=bool)
            env_done = (terminations | truncations).all()
            envs_dones.append(env_done)

            if env_done:
                env.reset()
            elif env.agent_selection == old_agent:
                if isinstance(type(act), np.ndarray):
                    act = np.array(act)
                act = (
                    act
                    if not (
                        self.terminations[old_agent][i]
                        or self.truncations[old_agent][i]
                    )
                    else None
                )  # if the agent is dead, set action to None
                env.step(act)

        self.agent_selection = self._agent_selector.next()
        self.agent_selection = self._find_active_agent()

        self.envs_dones = np.array(envs_dones)
        self._collect_dicts()
