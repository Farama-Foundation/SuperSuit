from skimage import transform
from gym.spaces import Box
import numpy as np
from . import convert_box
from PIL import Image


def check_param(obs_space, resize):
    xsize,ysize,linear_interp = resize
    assert all(isinstance(ds, int) and ds > 0 for ds in [xsize,ysize]), "resize x and y sizes must be integers greater than zero."
    assert isinstance(linear_interp, bool), "resize linear_interp parameter must be bool."
    assert obs_space.dtype == np.uint8, "resize must take in an image of type uint8. If use the normalize_obs wrapper to scale to 0-255 and the `dtype` wrapper to convert to uint8. Then, after resizing, you can use the normalize_obs and dtype wrappers to convert back to the original scale and size"
    assert len(obs_space.shape) == 3 and (obs_space.shape[2] == 1 or obs_space.shape[2] == 3) or len(obs_space.shape) == 2

def change_obs_space(obs_space, param):
    return convert_box(lambda obs:change_observation(obs, obs_space, param), obs_space)

def change_observation(obs, obs_space, resize):
    xsize,ysize,linear_interp = resize
    if len(obs.shape) == 3 and obs.shape[2] == 1:
        obs = obs.reshape((obs.shape[0],obs.shape[1]))
    mode = "RGB" if len(obs.shape) == 3 else "L"
    im = Image.fromarray(obs,mode=mode)
    interp_method = Image.NEAREST if not linear_interp else Image.BILINEAR
    im = im.resize((xsize, ysize), interp_method)
    obs = np.asarray(im)
    if len(obs_space.shape) == 3 and obs_space.shape[2] == 1:
        obs = obs.reshape(obs.shape + (1,))
    return obs
