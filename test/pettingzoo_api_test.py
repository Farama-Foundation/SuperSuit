from pettingzoo.tests import api_test,seed_test,error_tests
from pettingzoo.mpe import simple_push_v0
#from pettingzoo.sisl import multiwalker

from supersuit import continuous_actions,frame_stack
import numpy as np

def test_pettinzoo_frame_stack():
    _env = simple_push_v0.env()
    wrapped_env = frame_stack(_env)
    api_test.api_test(wrapped_env)
    # TODO: uncomment this when error tests is more stable
    # error_tests.error_test(frame_stack(simple_push_v0.env()))

def test_pettinzoo_continuous_actions():
    _env = simple_push_v0.env()
    wrapped_env = continuous_actions(_env)
    # wrapped_env.reset()
    # wrapped_env.step(np.nan*np.ones(4672))
    api_test.api_test(wrapped_env)
    seed_test.seed_test(lambda : continuous_actions(simple_push_v0.env()))
