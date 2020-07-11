from gym.spaces import Box
import numpy as np

def check_param(obs_space, min_max_pair):
    assert np.dtype(obs_space.dtype) == np.dtype("float32") or np.dtype(obs_space.dtype) == np.dtype("float64")
    assert isinstance(min_max_pair, tuple) and len(min_max_pair) == 2, "range_scale must be tuple of size 2. It is {}".format(min_max_pair)
    try:
        min_res = float(min_max_pair[0])
        max_res = float(min_max_pair[1])
    except ValueError:
        assert False, "normalize_obs inputs must be numbers. They are {}".format(min_max_pair)
    assert max_res > min_res, "maximum must be greater than minimum value in normalize_obs"
    assert np.all(np.isfinite(obs_space.low)) and np.all(np.isfinite(obs_space.high))

def change_obs_space(obs_space, min_max_pair):
    min = np.float64(min_max_pair[0]).astype(obs_space.dtype)
    max = np.float64(min_max_pair[1]).astype(obs_space.dtype)
    return Box(low=min,high=max,shape=obs_space.shape,dtype=obs_space.dtype)

def change_observation(obs, obs_space,  min_max_pair):
    min_res,max_res = [float(x) for x in min_max_pair]
    old_size = obs_space.high - obs_space.low
    new_size = max_res - min_res
    result = (obs - obs_space.low) / old_size * new_size + min_res
    return result
