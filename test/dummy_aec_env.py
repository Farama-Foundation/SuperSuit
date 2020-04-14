from pettingzoo.env import AECEnv

class DummyEnv(AECEnv):
    def __init__(self,observations,observation_spaces):
        self._observations = observations
        self.observation_spaces = observation_spaces

        self.agents = [x for x in observation_spaces.keys()]
        self.agent_order = list(self.agents)
        self.agent_selection = self.agents[0]
        self.action_spaces = copy.copy(self.env.action_spaces)
        self.orig_action_spaces = self.env.action_spaces

        self.rewards = {a:0 for a in self.agents}
        self.dones = {a:False for a in self.agents}
        self.infos = {a:{} for a in self.agents}

        self.agent_order = self.agents

    def observe(self):
        return self._observations

    def step(self, action, observe=True):
        old_sel = self.agent_selection
        self.agent_selection = self.agents[(self.agents.index(old_sel)+1)%len(self.agents)]
        if observe:
            return self._observations[old_sel]
        else:
            return None

    def reset(self, observe=True):
        self.agent_selection = self.agents[0]
        return self._observations

    def close(self):
        pass
