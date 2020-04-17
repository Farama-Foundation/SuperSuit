from pettingzoo.tests import api_test
from pettingzoo.classic import rps_v0
#from pettingzoo.sisl import multiwalker
from .dummy_aec_env import DummyEnv
from supersuit import continuous_actions

def test_pettinzoo_rps():
    _env = rps_v0.env()
    wrapped_env = continuous_actions(_env)
    api_test.api_test(wrapped_env, render=True, manual_control=None, save_obs=False)
