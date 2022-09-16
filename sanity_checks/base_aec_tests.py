"""
Just to ensure AEC wrappers run
"""

from pettingzoo.butterfly import pistonball_v6
from supersuit.aec_vector import vectorize_aec_env_v0

env = pistonball_v6.env()
env = vectorize_aec_env_v0(env, 1)

env.reset()
for agent in env.agent_iter():
    obs, reward, trunc, term, info = env.last()
    if trunc or term:
        act = None
    else:
        act = env.action_space(agent).sample()
    env.step(act)
    env.render()
