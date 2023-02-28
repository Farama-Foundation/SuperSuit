import gymnasium
import copy


class DummyEnv(gymnasium.Env):
    def __init__(self, observation, observation_space, action_space):
        super().__init__()
        self._observation = observation
        self.observation_space = observation_space
        self.action_space = action_space

    def step(self, action):
        return self._observation, 1, False, False, {}

    def reset(self, seed=None, options=None):
        return self._observation
