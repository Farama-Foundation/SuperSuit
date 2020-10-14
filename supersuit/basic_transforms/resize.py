from . import convert_box
import cv2


def check_param(obs_space, resize):
    xsize, ysize, linear_interp = resize
    assert all(isinstance(ds, int) and ds > 0 for ds in [xsize, ysize]), "resize x and y sizes must be integers greater than zero."
    assert isinstance(linear_interp, bool), "resize linear_interp parameter must be bool."
    assert len(obs_space.shape) == 3 or len(obs_space.shape) == 2


def change_obs_space(obs_space, param):
    return convert_box(lambda obs: change_observation(obs, obs_space, param), obs_space)


def change_observation(obs, obs_space, resize):
    xsize, ysize, linear_interp = resize
    if len(obs.shape) == 2:
        obs = obs.reshape(obs.shape + (1,))
    interp_method = cv2.INTER_LINEAR if linear_interp else cv2.INTER_AREA
    obs = cv2.resize(obs, (xsize, ysize), interpolation=interp_method)
    if len(obs_space.shape) == 2:
        obs = obs.reshape(obs.shape[:2])
    return obs
