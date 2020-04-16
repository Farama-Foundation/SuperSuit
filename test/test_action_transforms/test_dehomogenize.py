from gym.spaces import Box,Discrete
import numpy as np
import pytest
from supersuit.action_transforms.dehomogenize_ops import check_dehomogenize_actions,homogenize_action_spaces,dehomogenize_actions

box_spaces = [Box(low=np.float32(0),high=np.float32(1),shape=(5,4)),Box(low=np.float32(0),high=np.float32(1),shape=(10,2))]
discrete_spaces = [Discrete(5),Discrete(7)]

def test_param_check():
    check_dehomogenize_actions(box_spaces)
    check_dehomogenize_actions(discrete_spaces)

def test_homogenize_spaces():
    hom_space_box = homogenize_action_spaces(box_spaces)
    hom_space_discrete = homogenize_action_spaces(discrete_spaces)
    assert hom_space_box.shape == (10,4)
    assert hom_space_discrete.n == 7

def test_dehomogenize_actions():
    action = np.ones([10,4])
    assert dehomogenize_actions(box_spaces[0],action).shape == (5,4)
    assert dehomogenize_actions(discrete_spaces[0],5) == 0
    assert dehomogenize_actions(discrete_spaces[0],4) == 4
