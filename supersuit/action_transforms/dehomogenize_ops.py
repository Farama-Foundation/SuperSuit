import warnings
import numpy as np
from gym.spaces import Box,Discrete
from gym import spaces

def check_dehomogenize_actions(act_spaces):
    assert len(act_spaces) > 0
    space1 = act_spaces[0]
    assert all(isinstance(space,space1.__class__) for space in act_spaces), "all spaces to homogenize must be of same general shape"

    if isinstance(space1, spaces.Box):
        for space in act_spaces:
            assert len(space1.shape) == len(space.shape), "all spaces to homogenize must be of same shape"
            assert space1.dtype == space.dtype, "all spaces to homogenize must be of same dtype"
    elif isinstance(space1, spaces.Discrete):
        pass
    else:
        assert False, "homogenize_actions only supports Discrete and Box spaces"

def pad_to(arr,new_shape):
    old_shape = arr.shape
    pad_size = [ns-os for ns,os in zip(new_shape,old_shape)]
    pad_tuples = [(0,ps) for ps in pad_size]
    return np.pad(arr,pad_tuples)

def homogenize_action_spaces(act_spaces):
    space1 = act_spaces[0]
    print(space1.dtype)
    if isinstance(space1, spaces.Box):
        all_dims = np.array([space.shape for space in act_spaces],dtype=np.int32)
        max_dims = np.max(all_dims,axis=0)
        new_shape = tuple(max_dims)
        all_lows = np.stack([np.zeros(new_shape,dtype=space1.dtype)]+[pad_to(space.low,new_shape) for space in act_spaces])
        all_highs = np.stack([np.ones(new_shape,dtype=space1.dtype)]+[pad_to(space.high,new_shape) for space in act_spaces])
        new_low = np.min(all_lows,axis=0)
        new_high = np.max(all_highs,axis=0)
        assert new_shape == new_low.shape
        return Box(low=new_low,high=new_high,dtype=space1.dtype)
    elif isinstance(space1, spaces.Discrete):
        max_n = max([space.n for space in act_spaces])
        return Discrete(max_n)
    else:
        assert False

def dehomogenize_actions(orig_action_space, action):
    if isinstance(orig_action_space, spaces.Box):
        # choose only the relevant action values
        cur_shape = action.shape
        new_shape = orig_action_space.shape
        assert len(cur_shape) == len(new_shape)
        slices = [slice(0,i) for i in new_shape]
        new_action = action[tuple(slices)]

        return new_action

    elif isinstance(orig_action_space, spaces.Discrete):
        # extra action values refer to action value 0
        n = orig_action_space.n
        if action > n - 1:
            action = 0
        return action
    else:
        assert False
