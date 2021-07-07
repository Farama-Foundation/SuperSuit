from pettingzoo.utils.env import ParallelEnv


class ParallelWraper(ParallelEnv):
    def __init__(self, env):
        self.env = env
        self.observation_spaces = env.observation_spaces
        self.action_spaces = env.action_spaces
        self.metadata = env.metadata
        self.possible_agents = env.possible_agents

    def reset(self):
        res = self.env.reset()
        self.agents = self.env.agents
        return res

    def step(self, actions):
        res = self.env.step(actions)
        self.agents = self.env.agents
        return res

    def render(self, mode="human"):
        return self.env.render(mode)

    def close(self):
        return self.env.close()
