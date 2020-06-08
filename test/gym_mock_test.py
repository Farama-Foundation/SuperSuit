from .dummy_gym_env import DummyEnv
from gym.spaces import Box, Discrete
import numpy as np
from supersuit.gym_wrappers import frame_stack,reshape,observation_lambda,action_lambda,continuous_actions
from supersuit import gym_wrappers
import pytest

base_obs = (np.zeros([8,8,3])+ np.arange(3)).astype(np.float32)
base_obs_space = Box(low=np.float32(0.),high=np.float32(10.),shape=[8,8,3])
base_act_spaces =  Discrete(5)

def test_reshape():
    base_env = DummyEnv(base_obs, base_obs_space, base_act_spaces)
    env = reshape(base_env, (64, 3))
    obs = env.reset()
    assert obs.shape == (64,3)
    first_obs,_,_,_ = env.step(5)
    assert np.all(np.equal(first_obs,base_obs.reshape([64,3])))

def new_dummy():
    return  DummyEnv(base_obs, base_obs_space, base_act_spaces)

wrappers = [
    gym_wrappers.color_reduction(new_dummy(),"R"),
    gym_wrappers.down_scale(new_dummy(),x_scale=5,y_scale=10),
    gym_wrappers.dtype(new_dummy(),np.int32),
    gym_wrappers.flatten(new_dummy()),
    gym_wrappers.reshape(new_dummy(),(64,3)),
    gym_wrappers.normalize_obs(new_dummy(),env_min=-1,env_max=5.),
    gym_wrappers.frame_stack(new_dummy(),8),
    gym_wrappers.continuous_actions(new_dummy()),
    #gym_wrappers.normalize_reward(new_dummy()),
    gym_wrappers.reward_lambda(new_dummy(),lambda x: x/10),
    gym_wrappers.clip_reward(new_dummy()),
]
@pytest.mark.parametrize("env", wrappers)
def test_basic_wrappers(env):
    obs = env.reset()
    act_space = env.action_space
    obs_space = env.observation_space
    assert obs_space.contains(obs)
    assert obs.dtype == obs_space.dtype
    env.step(act_space.sample())

def test_lambda():
    def add1(obs):
        return obs+1
    base_env = DummyEnv(base_obs, base_obs_space, base_act_spaces)
    env = observation_lambda(base_env, add1)
    obs0 = env.reset()
    assert int(obs0[0][0][0]) == 1
    env = observation_lambda(env, add1)
    obs0 = env.reset()
    assert int(obs0[0][0][0]) == 2

    def tile_obs(obs):
        shape_size = len(obs.shape)
        tile_shape = [1]*shape_size
        tile_shape[0] *= 2
        return np.tile(obs,tile_shape)

    env = observation_lambda(env, tile_obs)
    obs0 = env.reset()
    assert env.observation_space.shape == (16,8,3)
    def change_shape_fn(obs_space):
        return Box(low=0,high=1,shape=(32,8,3))
    env = observation_lambda(env, tile_obs)
    obs0 = env.reset()
    assert env.observation_space.shape == (32,8,3)
    assert obs0.shape == (32,8,3)

def test_action_lambda():
    def inc1(x,space):
        return (x + 1)
    def change_space_fn(space):
        return Discrete(space.n+1)
    base_env = DummyEnv(base_obs, base_obs_space, base_act_spaces)
    env = action_lambda(base_env, inc1, change_space_fn)
    env.reset()
    env.step(5)

    def one_hot(x,n):
        v = np.zeros(n)
        v[x] = 1
        return v

    act_spaces = Box(low=0,high=1,shape=(15,))
    base_env = DummyEnv(base_obs, base_obs_space, act_spaces)
    env = action_lambda(base_env,
        lambda action, act_space : one_hot(action, act_space.shape[0]),
        lambda act_space : Discrete(act_space.shape[0]))

    env.reset()
    env.step(2)


def test_rew_lambda():
    env = gym_wrappers.reward_lambda(new_dummy(), lambda x:x/10)
    env.reset()
    obs,rew,done,info = env.step(0)
    assert rew == 1./10

def test_continuous_actions():
    base_act_spaces = Discrete(5)

    base_env = DummyEnv(base_obs, base_obs_space, base_act_spaces)
    env = continuous_actions(base_env)
    env.reset()
    assert env.action_space.shape == (5,)
    env.step(np.ones(5))
    env.step(np.nan*np.ones(5))
