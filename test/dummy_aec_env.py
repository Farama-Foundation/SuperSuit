from pettingzoo import AECEnv
import copy
from pettingzoo.utils.agent_selector import agent_selector

class DummyEnv(AECEnv):
    def __init__(self,observations,observation_spaces,action_spaces):
        super().__init__()
        self._observations = observations
        self.observation_spaces = observation_spaces

        self.agents = [x for x in observation_spaces.keys()]
        self.num_agents = len(self.agents)
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self.action_spaces = action_spaces

        self.rewards = {a:1 for a in self.agents}
        self.dones = {a:False for a in self.agents}
        self.infos = {a:{} for a in self.agents}

    def observe(self, agent):
        return self._observations[agent]

    def step(self, action, observe=True):
        self.agent_selection = self._agent_selector.next()
        if observe:
            return self._observations[self.agent_selection]
        else:
            return None

    def reset(self, observe=True):
        self.agent_selection = self._agent_selector.reset()
        return self._observations[self.agent_selection]

    def close(self):
        pass
