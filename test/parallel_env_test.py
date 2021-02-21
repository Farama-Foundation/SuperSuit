from pettingzoo.utils import ParallelEnv
from gym.spaces import Box, Discrete
import numpy as np
import supersuit


class DummyParEnv(ParallelEnv):
    metadata = {"render.modes": ["human"]}

    def __init__(self, observations, observation_spaces, action_spaces):
        super().__init__()
        self._observations = observations
        self.observation_spaces = observation_spaces

        self.agents = [x for x in observation_spaces.keys()]
        self.possible_agents = self.agents
        self.agent_selection = self.agents[0]
        self.action_spaces = action_spaces

        self.rewards = {a: 1 for a in self.agents}
        self.dones = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

    def step(self, actions):
        for agent, action in actions.items():
            assert action in self.action_spaces[agent]
        return self._observations, self.rewards, self.dones, self.infos

    def reset(self):
        return self._observations

    def close(self):
        pass


base_obs = {"a{}".format(idx): np.zeros([8, 8, 3], dtype=np.float32) + np.arange(3) + idx for idx in range(2)}
base_obs_space = {"a{}".format(idx): Box(low=np.float32(0.0), high=np.float32(10.0), shape=[8, 8, 3]) for idx in range(2)}
base_act_spaces = {"a{}".format(idx): Discrete(5) for idx in range(2)}


def test_basic():
    env = DummyParEnv(base_obs, base_obs_space, base_act_spaces)
    env = supersuit.delay_observations_v0(env, 4)
    env = supersuit.dtype_v0(env, np.uint8)
    orig_obs = env.reset()
    for i in range(10):
        action = {agent: env.action_spaces[agent].sample() for agent in env.agents}
        obs, rew, done, info = env.step(action)
