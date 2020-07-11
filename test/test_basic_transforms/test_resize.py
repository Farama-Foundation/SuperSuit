from supersuit.basic_transforms.resize import check_param,change_observation
from gym.spaces import Box
import numpy as np
import pytest

test_shape = (6,4,3)
high_val = (np.ones(test_shape) + np.arange(4).reshape(1,4,1))
test_obs_space = Box(low=high_val-1,high=high_val,dtype=np.uint8)
test_val = high_val-0.5

def test_param_check():
    check_param(test_obs_space, (2,2,False))
    with pytest.raises(AssertionError):
        check_param(test_obs_space, (-2,2,2))
    with pytest.raises(AssertionError):
        check_param(test_obs_space, (2.,2,2))

def test_change_observation():
    cur_val = test_val
    #print(cur_val)
    cur_val = cur_val.astype(np.uint8)
    #print(cur_val)
    new_obs = change_observation(cur_val,test_obs_space,(3,2,False))
    new_obs = change_observation(cur_val,test_obs_space,(3,2,True))
    test_obs = np.array(
    [[[0.6666667, 0.6666667, 0.6666667],
      [2.       , 2.       , 2.       ],
      [3.3333335, 3.3333335, 3.3333335]],

     [[0.6666667, 0.6666667, 0.6666667],
      [2.       , 2.       , 2.       ],
      [3.3333335, 3.3333335, 3.3333335]]]
    ).astype(np.uint8)
    assert new_obs.dtype == np.uint8
    assert np.all(np.equal(new_obs,test_obs))

    test_shape = (6,4)
    high_val = np.ones(test_shape).astype(np.float64)
    obs_spae = Box(low=high_val-1,high=high_val)
    new_obs = change_observation(high_val-0.5,obs_spae,(3,2,False))
    assert new_obs.shape == (2,3)
    assert new_obs.dtype == np.float64

    test_shape = (6,5,4)
    high_val = np.ones(test_shape).astype(np.uint8)
    obs_spae = Box(low=high_val-1,high=high_val,dtype=np.uint8)
    new_obs = change_observation(high_val,obs_spae,(5,2,False))
    assert new_obs.shape == (2,5,4)
    assert new_obs.dtype == np.uint8
