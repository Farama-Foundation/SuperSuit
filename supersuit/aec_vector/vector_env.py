import copy
import multiprocessing as mp
from gym.vector.utils import shared_memory
from pettingzoo.utils.agent_selector import agent_selector
import numpy as np
import ctypes
import gym
from .base_aec_vec_env import VectorAECEnv


class SyncAECVectorEnv(VectorAECEnv):
    def __init__(self, env_constructors):
        assert len(env_constructors) >= 1
        assert callable(env_constructors[0]), "env_constructor must be a callable object (i.e function) that create an environment"

        self.envs = [env_constructor() for env_constructor in env_constructors]
        self.num_envs = len(env_constructors)
        self.env = self.envs[0]
        self.max_num_agents = self.env.max_num_agents
        self.possible_agents = self.env.possible_agents
        self.observation_spaces = copy.copy(self.env.observation_spaces)
        self.action_spaces = copy.copy(self.env.action_spaces)
        self._agent_selector = agent_selector(self.possible_agents)

    def _find_active_agent(self):
        cur_selection = self.agent_selection
        while not any(cur_selection == env.agent_selection for env in self.envs):
            cur_selection = self._agent_selector.next()
        return cur_selection

    def _collect_dicts(self):
        self.rewards = {
            agent: np.array([env.rewards[agent] if agent in env.rewards else 0 for env in self.envs], dtype=np.float32)
            for agent in self.possible_agents
        }
        self._cumulative_rewards = {
            agent: np.array([env._cumulative_rewards[agent] if agent in env._cumulative_rewards else 0 for env in self.envs], dtype=np.float32)
            for agent in self.possible_agents
        }
        self.dones = {
            agent: np.array([env.dones[agent] if agent in env.dones else True for env in self.envs], dtype=np.uint8) for agent in self.possible_agents
        }
        self.infos = {agent: [env.infos[agent] if agent in env.infos else {} for env in self.envs] for agent in self.possible_agents}

    def reset(self):
        """
        returns: list of observations
        """
        for env in self.envs:
            env.reset()

        self.agent_selection = self._agent_selector.reset()
        self.agent_selection = self._find_active_agent()

        self._collect_dicts()
        self.envs_dones = np.zeros(self.num_envs)

    def seed(self, seed=None):
        for i, env in enumerate(self.envs):
            env.seed(seed + i)

    def observe(self, agent):
        observations = []
        for env in self.envs:
            obs = env.observe(agent) if agent in env.dones else np.zeros_like(self.observation_spaces[agent].low)
            observations.append(obs)
        return np.stack(observations)

    def last(self, observe=True):
        passes = np.array([env.agent_selection != self.agent_selection for env in self.envs], dtype=np.uint8)
        last_agent = self.agent_selection
        obs = self.observe(last_agent) if observe else None
        return obs, self._cumulative_rewards[last_agent], self.dones[last_agent], self.envs_dones, passes, self.infos[last_agent]

    def step(self, actions, observe=True):
        assert len(actions) == len(self.envs)
        old_agent = self.agent_selection

        envs_dones = []
        for i, (act, env) in enumerate(zip(actions, self.envs)):
            env_done = not env.agents
            envs_dones.append(env_done)
            if env_done:
                env.reset()
            elif env.agent_selection == old_agent:
                env.step(act if not self.dones[old_agent][i] else None)

        self.agent_selection = self._agent_selector.next()
        self.agent_selection = self._find_active_agent()

        self.envs_dones = np.array(envs_dones)
        self._collect_dicts()
