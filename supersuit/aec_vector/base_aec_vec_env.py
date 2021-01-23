import copy
import multiprocessing as mp
from gym.vector.utils import shared_memory
from pettingzoo.utils.agent_selector import agent_selector
import numpy as np
import ctypes
import gym


class VectorAECEnv:
    def reset(self):
        """
        resets all environments
        """

    def seed(self, seed=None):
        """
        seeds all environments
        """

    def observe(self, agent):
        """
        returns observation for agent from all environments (if agent is alive, else all zeros)
        """

    def last(self, observe=True):
        """
        returns list of observations, rewards, dones, env_dones, passes, infos

        each of the following is a list over environments that holds the value for the current agent (env.agent_selection)

        dones: are True when the current agent is done
        env_dones: is True when all agents are done, and the environment will reset
        passes: is true when the agent is not stepping this turn (because it is dead or not currently stepping for some other reason)
        infos: list of infos for the agent
        """

    def step(self, actions, observe=True):
        """
        steps the current agent with the following actions.
        Unlike a regular AECEnv, the actions cannot be None
        """

    def agent_iter(self, max_iter):
        """
        Unlike aec agent_iter, this does not stop on environment done. Instead,
        vector environment resets speciic envs when done.

        Instead, just continues until max_iter is reached.
        """
        return AECIterable(self, max_iter)


class AECIterable:
    def __init__(self, env, max_iter):
        self.env = env
        self.max_iter = max_iter

    def __iter__(self):
        return AECIterator(self.env, self.max_iter)


class AECIterator:
    def __init__(self, env, max_iter):
        self.env = env
        self.iters_til_term = max_iter
        self.env._is_iterating = True

    def __next__(self):
        if self.iters_til_term <= 0:
            raise StopIteration
        self.iters_til_term -= 1
        return self.env.agent_selection
