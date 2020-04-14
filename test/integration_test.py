from gym.spaces import Box, Discrete
from .dummy_aec_env import DummyEnv
import numpy as np
from supersuit.aec_wrappers import frame_stack,reshape,lambda_wrapper

base_obs = {"a{}".format(idx): np.zeros([8,8,3]) + np.arange(3) + idx for idx in range(2)}
base_obs_space = {"a{}".format(idx): Box(low=np.float32(0.),high=np.float32(10.),shape=[8,8,3]) for idx in range(2)}
base_act_spaces = {"a{}".format(idx): Discrete(5) for idx in range(2)}

def test_frame_stack():
    base_env = DummyEnv(base_obs, base_obs_space, base_act_spaces)
    env = frame_stack(base_env, 10)
    obs = env.reset()
    assert obs.shape == (8,8,30)
    first_obs = env.observe("a1")
    assert np.all(np.equal(first_obs[:,:,-3:],base_obs["a1"]))
    assert np.all(np.equal(first_obs[:,:,:-3],0))


def test_reshape():
    base_env = DummyEnv(base_obs, base_obs_space, base_act_spaces)
    env = reshape(base_env, (64, 3))
    obs = env.reset()
    assert obs.shape == (64,3)
    first_obs = env.observe("a1")
    assert np.all(np.equal(first_obs,base_obs["a1"].reshape([64,3])))

def test_lambda():
    def add1(obs):
        return obs+1
    base_env = DummyEnv(base_obs, base_obs_space, base_act_spaces)
    env = lambda_wrapper(base_env, add1)
    obs0 = env.reset()
    assert int(obs0[0][0][0]) == 1
    
