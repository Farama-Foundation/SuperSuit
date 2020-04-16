from supersuit.basic_transforms.color_reduction import check_param,change_observation
from gym.spaces import Box
import numpy as np
import pytest

test_obs_space = Box(low=np.float32(0.),high=np.float32(1.),shape=(4,4,3))
bad_test_obs_space = Box(low=np.float32(0.),high=np.float32(1.),shape=(4,4,4))
test_obs = np.zeros([4,4,3])+np.arange(3)

def test_param_check():
    with pytest.raises(AssertionError):
        check_param(test_obs_space,"bob")
    with pytest.raises(AssertionError):
        check_param(bad_test_obs_space,"R")
    check_param(test_obs_space,"G")

def test_change_observation():
    new_obs = change_observation(test_obs,test_obs_space,"B")
    print(new_obs)
    assert np.all(np.equal(new_obs,2*np.ones([4,4])))
