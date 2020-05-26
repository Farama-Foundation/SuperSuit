from supersuit.basic_transforms.down_scale import check_param,change_observation
from gym.spaces import Box
import numpy as np
import pytest

test_shape = (6,4,3)
high_val = (np.ones(test_shape) + np.arange(4).reshape(1,4,1)).astype(np.float32)
test_obs_space = Box(low=high_val-1,high=high_val)
test_val = high_val-0.5

def test_param_check():
    check_param(test_obs_space, (2,2,2))
    with pytest.raises(AssertionError):
        check_param(test_obs_space, (2,2))
    with pytest.raises(AssertionError):
        check_param(test_obs_space, (-2,2,2))
    with pytest.raises(AssertionError):
        check_param(test_obs_space, (2.,2,2))
    with pytest.raises(AssertionError):
        check_param(test_obs_space, 2)
    with pytest.raises(AssertionError):
        check_param(test_obs_space, ("B","C","D"))

def test_change_observation():
    new_obs = change_observation(test_val,test_obs_space,(3,2,2))

    test_obs = np.array([
        [[1.,  0.5],
        [3.,  1.5]],

        [[1.,  0.5],
        [3.,  1.5]]
    ])
    assert np.all(np.equal(new_obs,test_obs))
