import warnings
import numpy as np
from gym.spaces import Box,Discrete
from gym import spaces

def check_action_space(act_space, bounds):
    assert isinstance(bounds, tuple) and len(bounds) == 2, "`bounds` parameter must be a tuple of length 2"
    float(bounds[0]) # bounds needs to be convertible to float
    float(bounds[1])
    assert isinstance(act_space, spaces.Discrete) or isinstance(act_space, spaces.Box),"space {} is not supported by the continuous_actions option of the wrapper".format(act_space)

def change_action_space(act_space, bounds):
    if isinstance(act_space, spaces.Discrete):
        low, high = bounds
        new_act_space = spaces.Box(low=low, high=high, shape=(act_space.n,))
    elif isinstance(act_space, spaces.Box):
        new_act_space = act_space

    return new_act_space

def modify_action(act_space, action, np_random):
    new_action = action

    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def sample_softmax(vec):
        vec = vec.astype(np.float64)
        return np.argmax(np_random.multinomial(1, softmax(vec)))

    if isinstance(act_space, spaces.Discrete):
        if np.any(np.isnan(action)):
            new_action = np.nan
        else:
            new_action = int(sample_softmax(action))
    elif isinstance(act_space, spaces.Box):
        new_action = action

    return new_action
