from gym.spaces import Box


def convert_box(convert_obs_fn, old_box):
    new_low = convert_obs_fn(old_box.low)
    new_high = convert_obs_fn(old_box.high)
    return Box(low=new_low, high=new_high, dtype=new_low.dtype)


from . import color_reduction # NOQA
from . import resize # NOQA
from . import dtype # NOQA
from . import flatten # NOQA
from . import normalize_obs # NOQA
from . import reshape # NOQA
