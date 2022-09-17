"""
Just to ensure AEC wrappers run
"""

from pettingzoo.butterfly import pistonball_v6
from supersuit.aec_vector import vectorize_aec_env_v0

env = pistonball_v6.env()
env = vectorize_aec_env_v0(env, 1)

env.reset()
for agent in env.agent_iter(10):
    obs, cum_reward, term, trunc, env_terms, env_truncs, passes, info = env.last()
    if (trunc & term).all():
        act = None
    else:
        act = env.action_space(agent).sample()
    env.step(act)
