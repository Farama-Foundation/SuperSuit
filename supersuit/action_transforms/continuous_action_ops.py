import warnings
import numpy as np
from gym.spaces import Box,Discrete
from gym import spaces

def check_action_space(act_space):
    assert isinstance(act_space, spaces.Discrete) or isinstance(act_space, spaces.Box),"space {} is not supported by the continuous_actions option of the wrapper".format(act_space)

def change_action_space(act_space):
    if isinstance(act_space, spaces.Discrete):
        new_act_space = spaces.Box(low=-np.float32(np.inf), high=np.float32(np.inf), shape=(act_space.n,))
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
            new_action = sample_softmax(action)
    elif isinstance(act_space, spaces.Box):
        new_action = action

    return new_action
