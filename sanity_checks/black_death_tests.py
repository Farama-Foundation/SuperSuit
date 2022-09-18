"""
Just to ensure wrappers run
"""
import numpy as np
from pettingzoo.butterfly import pistonball_v6
from supersuit.multiagent_wrappers import black_death_v3

env = pistonball_v6.env()
env = black_death_v3(env)

env.reset()
for agent in env.agent_iter(1000000):
    obs, rew, term, trunc, info = env.last()
    act = None if (term or trunc) else env.action_space(agent).sample()
    if (np.array(term) & np.array(trunc)).all():
        env.reset()
        break
    env.step(act)