from pettingzoo.tests import api_test
from pettingzoo.classic import chess_v0
#from pettingzoo.sisl import multiwalker
from test.dummy_aec_env import DummyEnv
from supersuit import continuous_actions
import numpy as np

def test_pettinzoo_rps():
    _env = chess_v0.env()
    wrapped_env = continuous_actions(_env)
    # wrapped_env.reset()
    # wrapped_env.step(np.nan*np.ones(4672))
    api_test.api_test(wrapped_env, render=True, manual_control=None, save_obs=False)
test_pettinzoo_rps()
