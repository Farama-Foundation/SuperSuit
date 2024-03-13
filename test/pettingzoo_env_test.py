from pettingzoo.mpe import simple_spread_v3
from pettingzoo.test import parallel_api_test

from supersuit.multiagent_wrappers import pad_action_space_v0


def test_pad_actuon_space():
    env = simple_spread_v3.parallel_env(max_cycles=25, continuous_actions=True)
    env = pad_action_space_v0(env)

    parallel_api_test(env, num_cycles=100)
