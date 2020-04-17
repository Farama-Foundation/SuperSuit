from supersuit.basic_transforms.normalize_obs import check_param,change_obs_space,change_observation
from gym.spaces import Box
import numpy as np
import pytest

high_val = np.array([1,2,4])
test_val = np.array([1,1,1])
test_obs_space = Box(low=np.zeros(3,dtype=np.float32),high=high_val.astype(np.float32))
bad_test_obs_space = Box(low=np.zeros(3,dtype=np.int32),high=high_val.astype(np.int32),dtype=np.int32)
bad_test_obs_space2 = Box(low=np.zeros(3,dtype=np.float32),high=np.inf*high_val.astype(np.float32))


def test_param_check():
    check_param(test_obs_space, (2,3))
    with pytest.raises(AssertionError):
        check_param(test_obs_space, (2,2))
    with pytest.raises(AssertionError):
        check_param(test_obs_space, ("bob",2))
    with pytest.raises(AssertionError):
        check_param(bad_test_obs_space, (2,3))
    with pytest.raises(AssertionError):
        check_param(bad_test_obs_space2, (2,3))

def test_change_obs_space():
    assert np.all(np.equal(change_obs_space(test_obs_space,(1,2)).high, np.array([2,2,2])))

def test_change_observation():
    print(change_observation(test_val,test_obs_space,(1,2)))
    assert np.all(np.equal(change_observation(test_val,test_obs_space,(1,2)), np.array([2,1.5,1.25])))
