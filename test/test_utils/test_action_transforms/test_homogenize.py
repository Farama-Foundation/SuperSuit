import numpy as np
from gymnasium.spaces import Box, Discrete

from supersuit.utils.action_transforms.homogenize_ops import (
    check_homogenize_spaces,
    dehomogenize_actions,
    homogenize_observations,
    homogenize_spaces,
)


box_spaces = [
    Box(low=np.float32(0), high=np.float32(1), shape=(5, 4)),
    Box(low=np.float32(0), high=np.float32(1), shape=(10, 2)),
]
discrete_spaces = [Discrete(5), Discrete(7)]


def test_param_check():
    check_homogenize_spaces(box_spaces)
    check_homogenize_spaces(discrete_spaces)


def test_homogenize_spaces():
    hom_space_box = homogenize_spaces(box_spaces)
    hom_space_discrete = homogenize_spaces(discrete_spaces)
    assert hom_space_box.shape == (10, 4)
    assert hom_space_discrete.n == 7


def test_dehomogenize_actions():
    action = np.ones([10, 4])
    assert dehomogenize_actions(box_spaces[0], action).shape == (5, 4)
    assert dehomogenize_actions(discrete_spaces[0], 5) == 0
    assert dehomogenize_actions(discrete_spaces[0], 4) == 4


def test_homogenize_observations():
    obs = np.zeros([5, 4])
    hom_space_box = homogenize_spaces(box_spaces)
    assert homogenize_observations(hom_space_box, obs).shape == (10, 4)
