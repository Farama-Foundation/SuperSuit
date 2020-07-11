from gym.spaces import Box

def convert_box(convert_obs_fn, old_box):
    new_low = convert_obs_fn(old_box.low)
    new_high = convert_obs_fn(old_box.high)
    return Box(low=new_low, high=new_high, dtype=new_low.dtype)

from . import color_reduction
from . import resize
from . import dtype
from . import flatten
from . import normalize_obs
from . import reshape
