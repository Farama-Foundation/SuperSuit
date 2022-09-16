"""
Just to ensure PZ runs (trunc version)
"""

from pettingzoo.butterfly import pistonball_v6

env = pistonball_v6.env()

env.reset()
for agent in env.agent_iter():
    obs, reward, trunc, term, info = env.last()
    if trunc or term:
        act = None
    else:
        act = env.action_space(agent).sample()
    env.step(act)
    env.render()
