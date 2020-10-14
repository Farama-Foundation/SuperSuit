import numpy as np
from gym.spaces import Box, Discrete


def get_tile_shape(shape, stack_size):
    obs_dim = len(shape)

    if obs_dim == 1:
        tile_shape = (stack_size,)
        new_shape = shape
    elif obs_dim == 3:
        tile_shape = (1, 1, stack_size)
        new_shape = shape
    # stack 2-D frames
    elif obs_dim == 2:
        tile_shape = (1, 1, stack_size)
        new_shape = shape + (1,)
    else:
        assert False, "Stacking is only avaliable for 1,2 or 3 dimentional arrays"

    return tile_shape, new_shape


def stack_obs_space(obs_space, stack_size):
    """
    obs_space_dict: Dictionary of observations spaces of agents
    stack_size: Number of frames in the observation stack
    Returns:
        New obs_space_dict
    """
    if isinstance(obs_space, Box):
        dtype = obs_space.dtype
        # stack 1-D frames and 3-D frames
        tile_shape, new_shape = get_tile_shape(obs_space.low.shape, stack_size)

        low = np.tile(obs_space.low.reshape(new_shape), tile_shape)
        high = np.tile(obs_space.high.reshape(new_shape), tile_shape)
        new_obs_space = Box(low=low, high=high, dtype=dtype)
        return new_obs_space
    elif isinstance(obs_space, Discrete):
        return Discrete(obs_space.n ** stack_size)
    else:
        assert False, "Stacking is currently only allowed for Box and Discrete observation spaces. The given observation space is {}".format(obs_space)


def stack_init(obs_space, stack_size):
    if isinstance(obs_space, Box):
        tile_shape, new_shape = get_tile_shape(obs_space.low.shape, stack_size)
        return np.tile(np.zeros(new_shape, dtype=obs_space.dtype), tile_shape)
    else:
        return 0


def stack_obs(frame_stack, obs, obs_space, stack_size):
    """
    Parameters
    ----------
    frame_stack : if not None, it is the stack of frames
    obs : new observation
        Rearranges frame_stack. Appends the new observation at the end.
        Throws away the oldest observation.
    stack_size : needed for stacking reset observations
    """
    if isinstance(obs_space, Box):
        obs_shape = obs.shape
        agent_fs = frame_stack

        if len(obs_shape) == 1:
            size = obs_shape[0]
            agent_fs[:-size] = agent_fs[size:]
            agent_fs[-size:] = obs
        elif len(obs_shape) == 2:
            agent_fs[:, :, :-1] = agent_fs[:, :, 1:]
            agent_fs[:, :, -1] = obs
        elif len(obs_shape) == 3:
            nchannels = obs_shape[-1]
            agent_fs[:, :, :-nchannels] = agent_fs[:, :, nchannels:]
            agent_fs[:, :, -nchannels:] = obs
        return agent_fs
    elif isinstance(obs_space, Discrete):
        return (frame_stack * obs_space.n + obs) % (obs_space.n ** stack_size)
