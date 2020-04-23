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
    assert np.all(np.equal(first_obs[:,:,-1],base_obs["a1"]))
    assert np.all(np.equal(first_obs[:,:,:-1],0))

    base_obs = {"a{}".format(idx): idx+3 for idx in range(2)}
    base_env = DummyEnv(base_obs, base_act_spaces, base_act_spaces)
    env = frame_stack(base_env, 4)
    obs = env.reset()
    assert env.observation_spaces[env.agent_selection].n == 5**4
    first_obs = env.step(2)
    assert first_obs == 4
    second_obs = env.step(2)
    assert second_obs == 3+3*5
    for x in range(100):
        nth_obs = env.step(2)
    assert nth_obs == ((3*5+3)*5+3)*5+3



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
    env = observation_lambda_wrapper(env, add1)
    obs0 = env.reset()
    assert int(obs0[0][0][0]) == 2

    def tile_obs(obs):
        shape_size = len(obs.shape)
        tile_shape = [1]*shape_size
        tile_shape[0] *= 2
        return np.tile(obs,tile_shape)

    env = observation_lambda_wrapper(env, tile_obs)
    obs0 = env.reset()
    assert env.observation_spaces[env.agent_selection].shape == (16,8,3)
    def change_shape_fn(obs_space):
        return Box(low=0,high=1,shape=(32,8,3))
    env = observation_lambda_wrapper(env, tile_obs)
    obs0 = env.reset()
    assert env.observation_spaces[env.agent_selection].shape == (32,8,3)
    assert obs0.shape == (32,8,3)

def test_action_lambda():
    def inc1(x,space):
        return (x + 1)
    def change_space_fn(space):
        return Discrete(space.n+1)
    base_env = DummyEnv(base_obs, base_obs_space, base_act_spaces)
    env = action_lambda_wrapper(base_env, inc1, change_space_fn)
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
    base_act_spaces = {"a{}".format(idx): Discrete(5) for idx in range(2)}

    base_env = DummyEnv(base_obs, base_obs_space, base_act_spaces)
    env = continuous_actions(base_env)
    env.reset()
    assert all([s.shape == (5,) for s in env.action_spaces.values()])
    env.step(np.ones(5))
    res = env.step(np.nan*np.ones(5))
