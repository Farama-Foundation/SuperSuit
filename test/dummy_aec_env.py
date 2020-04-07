from pettingzoo.env import AECEnv

class DummyEnv(AECEnv):
    def __init__(self,observation,observation_spaces):
        self._observation = observation
        self.observation_spaces = observation_spaces

        self.agents = list(observation_spaces.keys())
        self.agent_selection = self.agents[0]
        self.action_spaces = copy.copy(self.env.action_spaces)
        self.orig_action_spaces = self.env.action_spaces

        self.rewards = {a:0 for a in self.agents}
        self.dones = {a:False for a in self.agents}
        self.infos = {a:{} for a in self.agents}

        self.agent_order = self.agents

    def observe(self):
        return self._observation

    def step(self, action, observe=True):
        if observe:
            return self._observation
        else:
            return None

    def reset(self, observe=True):
        return self._observation

    def close(self):
        pass
