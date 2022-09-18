"""
Just to ensure PZ runs (trunc version)
"""
import numpy as np
from pettingzoo.butterfly import pistonball_v6

env = pistonball_v6.env()

env.reset()
for agent in env.agent_iter():
    obs, reward, trunc, term, info = env.last()
    act = None if (term or trunc) else env.action_space(agent).sample()
    terminations = np.fromiter(env.terminations.values(), dtype=bool)
    truncations = np.fromiter(env.truncations.values(), dtype=bool)
    env_done = (terminations | truncations).all()
    if env_done:
        env.reset()
        break
    env.step(act)
    env.render()
