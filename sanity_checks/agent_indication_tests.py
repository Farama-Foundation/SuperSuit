"""
Just to ensure wrappers run
"""
import numpy as np
from pettingzoo.butterfly import pistonball_v6
from supersuit.multiagent_wrappers import agent_indicator_v0

env = pistonball_v6.env()
env = agent_indicator_v0(env, type_only=False)

env.reset()

for agent in env.agent_iter(1000000):
    obs, rew, term, trunc, info = env.last()
    act = None if (term or trunc) else env.action_space(agent).sample()
    terminations = np.fromiter(env.terminations.values(), dtype=bool)
    truncations = np.fromiter(env.truncations.values(), dtype=bool)
    env_done = (terminations | truncations).all()
    if env_done:
        env.reset()
    env.step(act)


env = pistonball_v6.env()
env = agent_indicator_v0(env, type_only=True)

env.reset()
for agent in env.agent_iter(1000000):
    obs, rew, term, trunc, info = env.last()
    act = None if (term or trunc) else env.action_space(agent).sample()
    if (np.array(term) & np.array(trunc)).all():
        env.reset()
    env.step(act)
