from supersuit.basic_transforms.down_scale import check_param,change_space,change_observation
from gym.spaces import Box
import numpy as np
import pytest

test_shape = (6,4,3)
high_val = (np.ones(test_shape) + np.arange(4).reshape(1,4,1)).astype(np.float32)
test_obs_space = {"agent":Box(low=high_val-1,high=high_val)}

def test_param_check():
    check_param(test_obs_space, (2,2,2))
    with pytest.raises(AssertionError):
        check_param(test_obs_space, (2,2))
    with pytest.raises(AssertionError):
        check_param(test_obs_space, 2)
    with pytest.raises(AssertionError):
        check_param(test_obs_space, ("B","C","D"))

def test_change_space():
    new_space = change_space(test_obs_space, (3,2,2))
    assert new_space['agent'].shape == (2,2,1)
    print(new_space['agent'].high)
    assert False

def test_change_observation():
    pass
