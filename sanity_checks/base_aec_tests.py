"""
Just to ensure AEC wrappers run
"""
import numpy as np
from pettingzoo.butterfly import pistonball_v6
from supersuit.aec_vector import vectorize_aec_env_v0

env = pistonball_v6.env()
env = vectorize_aec_env_v0(env, 4)

env.reset()
for agent in env.agent_iter(1000000):
    obs, cum_reward, term, trunc, env_terms, env_truncs, passes, info = env.last()
    act = [env.action_space(agent).sample(), env.action_space(agent).sample(), env.action_space(agent).sample(), env.action_space(agent).sample()]
    if (np.array(term) | np.array(trunc)).all():
        env.reset()
    env.step(act)
