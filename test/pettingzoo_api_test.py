from pettingzoo.tests import api_test,seed_test,error_tests
from pettingzoo.mpe import simple_push_v0,simple_world_comm_v0
#from pettingzoo.sisl import multiwalker

from supersuit import frame_stack, pad_action_space, frame_skip
import numpy as np

def test_pettinzoo_frame_stack():
    _env = simple_push_v0.env()
    wrapped_env = frame_stack(_env)
    api_test.api_test(wrapped_env)

def test_pettinzoo_frame_skip():
    # this is the actual frame_skip test
    env = simple_push_v0.raw_env()
    env = frame_skip(env, 3)
    env.reset()
    for x in range(10):
        assert env.env.steps == (x//2)*3
        action = env.action_spaces[env.agent_selection].sample()
        next_obs = env.step(action)

def test_pettinzoo_pad_actino_space():
    _env = simple_world_comm_v0.env()
    wrapped_env = pad_action_space(_env)
    # wrapped_env.reset()
    # wrapped_env.step(np.nan*np.ones(4672))
    api_test.api_test(wrapped_env)
    seed_test.seed_test(lambda seed=None: pad_action_space(simple_world_comm_v0.env(seed=seed)))
