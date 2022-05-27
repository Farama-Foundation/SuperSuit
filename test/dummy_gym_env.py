import gym
import copy


class DummyEnv(gym.Env):
    def __init__(self, observation, observation_space, action_space):
        super().__init__()
        self._observation = observation
        self.observation_space = observation_space
        self.action_space = action_space

    def step(self, action):
        return self._observation, 1, False, {}

    def reset(self, seed=None, return_info=False, options=None):
        if not return_info:
            return self._observation
        else:
            return self._observation, {}
