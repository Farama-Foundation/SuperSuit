import numpy as np
from . import convert_box


def check_param(obs_space, new_dtype):
    np.dtype(new_dtype)  # type argument must be convertable to a numpy dtype


def change_obs_space(obs_space, param):
    return convert_box(lambda obs: change_observation(obs, obs_space, param), obs_space)


def change_observation(obs, obs_space, new_dtype):
    obs = obs.astype(new_dtype)
    return obs
