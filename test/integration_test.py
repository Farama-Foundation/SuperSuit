from gym.spaces import Box, Discrete
from .dummy_aec_env import DummyEnv
import numpy as np
from supersuit.aec_wrappers import frame_stack,reshape,observation_lambda_wrapper,action_lambda_wrapper,homogenize_actions,continuous_actions,homogenize_obs
from supersuit import aec_wrappers

base_obs = {"a{}".format(idx): np.zeros([8,8,3]) + np.arange(3) + idx for idx in range(2)}
base_obs_space = {"a{}".format(idx): Box(low=np.float32(0.),high=np.float32(10.),shape=[8,8,3]) for idx in range(2)}
base_act_spaces = {"a{}".format(idx): Discrete(5) for idx in range(2)}

def test_frame_stack():
    base_obs_space = {"a{}".format(idx): Box(low=np.float32(0.),high=np.float32(10.),shape=[2,3]) for idx in range(2)}
    base_obs = {"a{}".format(idx): np.zeros([2,3]) + np.arange(3) + idx for idx in range(2)}
    base_env = DummyEnv(base_obs, base_obs_space, base_act_spaces)
    env = frame_stack(base_env, 4)
    obs = env.reset()
    assert obs.shape == (2,3,4)
    first_obs = env.step(2)
    print(first_obs)
    assert np.all(np.equal(first_obs[:,:,-1],base_obs["a1"]))
    assert np.all(np.equal(first_obs[:,:,:-1],0))


def test_reshape():
    base_env = DummyEnv(base_obs, base_obs_space, base_act_spaces)
    env = reshape(base_env, (64, 3))
    obs = env.reset()
    assert obs.shape == (64,3)
    first_obs = env.step(5)
    assert np.all(np.equal(first_obs,base_obs["a1"].reshape([64,3])))

def new_dummy():
    return  DummyEnv(base_obs, base_obs_space, base_act_spaces)

def test_basic_wrappers():
    wrappers = [
        aec_wrappers.color_reduction(new_dummy(),"R"),
        aec_wrappers.down_scale(new_dummy(),x_scale=5,y_scale=10),
        aec_wrappers.dtype(new_dummy(),np.int32),
        aec_wrappers.flatten(new_dummy()),
        aec_wrappers.reshape(new_dummy(),(64,3)),
        aec_wrappers.normalize_obs(new_dummy(),env_min=-1,env_max=5.),
        aec_wrappers.frame_stack(new_dummy(),8),
        aec_wrappers.homogenize_obs(new_dummy()),
    ]
    for env in wrappers:
        obs = env.reset()
        first_obs = env.observe("a1")


def test_lambda():
    def add1(obs):
        return obs+1
    base_env = DummyEnv(base_obs, base_obs_space, base_act_spaces)
    env = observation_lambda_wrapper(base_env, add1)
    obs0 = env.reset()
    assert int(obs0[0][0][0]) == 1
    def check_fn(space):
        assert True
    env = observation_lambda_wrapper(env, add1, check_fn)
    obs0 = env.reset()
    assert int(obs0[0][0][0]) == 2


def test_action_lambda():
    def inc1(x,space):
        return (x + 1)
    def change_space_fn(space):
        return Discrete(space.n+1)
    def check_space(space):
        return isinstance(space,Discrete)
    base_env = DummyEnv(base_obs, base_obs_space, base_act_spaces)
    env = action_lambda_wrapper(base_env, inc1, change_space_fn, check_space)
    env.reset()
    env.step(5)

def test_dehomogenize():
    base_act_spaces = {"a{}".format(idx): Discrete(5+idx) for idx in range(2)}

    base_env = DummyEnv(base_obs, base_obs_space, base_act_spaces)
    env = homogenize_actions(base_env)
    env.reset()
    assert all([s.n == 6 for s in env.action_spaces.values()])
    env.step(5)

def test_continuous_actions():
    base_act_spaces = {"a{}".format(idx): Discrete(5) for idx in range(1)}

    base_env = DummyEnv(base_obs, base_obs_space, base_act_spaces)
    env = continuous_actions(base_env)
    env.reset()
    assert all([s.shape == (5,) for s in env.action_spaces.values()])
    env.step(np.ones(5))
