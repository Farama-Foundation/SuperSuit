
from pettingzoo import AECEnv
import copy
from pettingzoo.utils.agent_selector import agent_selector

class DummyNaNEnv(AECEnv):
    metadata = {"render.modes": ["human"]}

    def __init__(self, observations, observation_spaces, action_spaces):
        super().__init__()
        self._observations = observations
        self.observation_spaces = observation_spaces

        self.agents = sorted([x for x in observation_spaces.keys()])
        self.possible_agents = self.agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self.action_spaces = action_spaces

        self.steps = 0

    def step(self, action, observe=True):
        assert (not (np.isnan(action))), "Action was a NaN, even though it should not have been."