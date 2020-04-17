from pettingzoo import AECEnv
import copy

class DummyEnv(AECEnv):
    def __init__(self,observations,observation_spaces,action_spaces):
        super().__init__()
        self._observations = observations
        self.observation_spaces = observation_spaces

        self.agents = [x for x in observation_spaces.keys()]
        self.agent_order = list(self.agents)
        self.agent_selection = self.agents[0]
        self.action_spaces = action_spaces

        self.rewards = {a:0 for a in self.agents}
        self.dones = {a:False for a in self.agents}
        self.infos = {a:{} for a in self.agents}

        self.agent_order = self.agents

    def observe(self, agent):
        return self._observations[agent]

    def step(self, action, observe=True):
        old_sel = self.agent_selection
        self.agent_selection = self.agents[(self.agents.index(old_sel)+1)%len(self.agents)]
        if observe:
            return self._observations[self.agent_selection]
        else:
            return None

    def reset(self, observe=True):
        self.agent_selection = self.agents[0]
        return self._observations[self.agent_selection]

    def close(self):
        pass
