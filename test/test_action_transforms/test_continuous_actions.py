from gym.spaces import Box,Discrete
import numpy as np
import pytest
from supersuit.action_transforms.continuous_action_ops import check_action_space,change_action_space,modify_action
from collections import Counter

box_spaces = Box(low=np.float32(0),high=np.float32(1),shape=(5,4))
discrete_spaces = Discrete(5)

def test_param_check():
    with pytest.raises(ValueError):
        check_action_space(box_spaces, ("bob", 2))
    with pytest.raises(AssertionError):
        check_action_space(box_spaces, 2)
    bounds = (-5,5)
    check_action_space(box_spaces, bounds)
    check_action_space(discrete_spaces, bounds)

def test_continuous_space_transform():
    bounds = (-5,5)
    old_box = change_action_space(box_spaces, bounds)
    new_box = change_action_space(discrete_spaces, bounds)
    assert old_box.shape == (5,4)
    assert new_box.shape == (5,)

def one_hot(size):
    x = np.zeros(size)
    x[1] = 10.
    return x

def test_discritize_actions():
    action = np.ones([5,4])
    assert modify_action(box_spaces,action,np.random.RandomState()).shape == (5,4)
    acts = [modify_action(discrete_spaces,one_hot(5),np.random.RandomState()) for _ in range(10)]
    res = sorted(list(Counter(acts).items()),key=lambda x:x[1])
    assert res[0][0] == 1

    assert np.isnan(modify_action(discrete_spaces,np.ones(5)*np.nan,np.random.RandomState()))
