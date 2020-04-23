from supersuit.frame_stack import stack_obs_space,stack_init,stack_obs
from gym.spaces import Box, Discrete
import numpy as np
import pytest

stack_obs_space_3d = Box(low=np.float32(0.),high=np.float32(1.),shape=(4,4,3))
stack_obs_space_2d = Box(low=np.float32(0.),high=np.float32(1.),shape=(4,3))
stack_obs_space_1d = Box(low=np.float32(0.),high=np.float32(1.),shape=(3,))

stack_discrete = Discrete(3)

STACK_SIZE = 11

def test_obs_space():
    assert stack_obs_space(stack_obs_space_1d, STACK_SIZE).shape == (3*STACK_SIZE,)
    assert stack_obs_space(stack_obs_space_2d, STACK_SIZE).shape == (4,3,STACK_SIZE)
    assert stack_obs_space(stack_obs_space_3d, STACK_SIZE).shape ==  (4,4,3*STACK_SIZE)
    assert stack_obs_space(stack_discrete, STACK_SIZE).n == 3**STACK_SIZE

def stack_obs_helper(frame_stack_list, obs_space, stack_size):
    stack = stack_init(obs_space, stack_size)#stack_reset_obs(frame_stack_list[0], stack_size)
    for obs in frame_stack_list:
        stack = stack_obs(stack,obs,obs_space,stack_size)
    return stack

def test_change_observation():
    assert stack_obs_helper([stack_obs_space_1d.low],stack_obs_space_1d,STACK_SIZE).shape == (3*STACK_SIZE,)
    assert stack_obs_helper([stack_obs_space_1d.low,stack_obs_space_1d.high],stack_obs_space_1d,STACK_SIZE).shape == (3*STACK_SIZE,)
    assert stack_obs_helper([stack_obs_space_2d.low],stack_obs_space_2d,STACK_SIZE).shape == (4,3,STACK_SIZE)
    assert stack_obs_helper([stack_obs_space_2d.low,stack_obs_space_2d.high],stack_obs_space_2d,STACK_SIZE).shape == (4,3,STACK_SIZE)
    assert stack_obs_helper([stack_obs_space_3d.low],stack_obs_space_3d,STACK_SIZE).shape == (4,4,3*STACK_SIZE)

    assert stack_obs_helper([1,2],stack_discrete,STACK_SIZE) == 2 + 1 * 3

    stacked = stack_obs_helper([stack_obs_space_2d.low,stack_obs_space_2d.high],stack_obs_space_2d,3)
    raw = np.stack([np.zeros_like(stack_obs_space_2d.high),stack_obs_space_2d.low,stack_obs_space_2d.high],axis=2)
    assert np.all(np.equal(stacked, raw))

    stacked = stack_obs_helper([stack_obs_space_3d.low,stack_obs_space_3d.high],stack_obs_space_3d,3)
    raw = np.concatenate([np.zeros_like(stack_obs_space_3d.high),stack_obs_space_3d.low,stack_obs_space_3d.high],axis=2)
    assert np.all(np.equal(stacked, raw))
