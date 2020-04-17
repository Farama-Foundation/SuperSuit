from pettingzoo.tests import api_test
from pettingzoo.classic import chess_v0
#from pettingzoo.sisl import multiwalker
from test.dummy_aec_env import DummyEnv
from supersuit import continuous_actions,frame_stack
import numpy as np

def test_pettinzoo_chess_frame_stack():
    _env = chess_v0.env()
    wrapped_env = frame_stack(_env)
    api_test.api_test(wrapped_env, render=False, manual_control=None, save_obs=False)

def test_pettinzoo_chess_continuous_actions():
    _env = chess_v0.env()
    wrapped_env = continuous_actions(_env)
    # wrapped_env.reset()
    # wrapped_env.step(np.nan*np.ones(4672))
    api_test.api_test(wrapped_env, render=False, manual_control=None, save_obs=False)

if __name__ == "__main__":
    test_pettinzoo_chess_continuous_actions()
    test_pettinzoo_chess_frame_stack()
