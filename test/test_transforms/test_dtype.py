from supersuit.basic_transforms.dtype import check_param,change_space,change_observation
from gym.spaces import Box
import numpy as np
import pytest

test_obs_space = Box(low=np.float32(0.),high=np.float32(1.),shape=(4,4,3),dtype=np.float32)
test_obs = np.zeros([4,4,3],dtype=np.float64)+np.arange(3)


def test_param_check():
    check_param(test_obs_space, np.uint8)
    check_param(test_obs_space, np.dtype("uint8"))
    with pytest.raises(AssertionError):
        check_param(test_obs_space, 6)

def test_change_space():
    new_space = change_space(test_obs_space,np.dtype("uint8"))
    assert new_space.low.dtype == np.uint8
    assert new_space.dtype == np.uint8
    with pytest.warns(UserWarning):
        new_space = change_space(new_space,np.float32)
    assert new_space.low.dtype == np.float32
    assert new_space.dtype == np.float32

def test_change_observation():
    new_obs = change_observation(test_obs,np.float32)
    assert new_obs.dtype == np.float32
