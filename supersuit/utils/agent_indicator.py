import re
import numpy as np
from gym.spaces import Box, Discrete
import warnings


def change_obs_space(space, num_indicators):
    if isinstance(space, Box):
        ndims = len(space.shape)
        if ndims == 1:
            pad_space = np.ones((num_indicators,), dtype=space.dtype)
            new_low = np.concatenate([space.low, pad_space * 0], axis=0)
            new_high = np.concatenate([space.high, pad_space], axis=0)
            new_space = Box(low=new_low, high=new_high, dtype=space.dtype)
            return new_space
        elif ndims == 3 or ndims == 2:
            orig_low = space.low if ndims == 3 else np.expand_dims(space.low, 2)
            orig_high = space.high if ndims == 3 else np.expand_dims(space.high, 2)
            pad_space = np.ones(orig_low.shape[:2] + (num_indicators,), dtype=space.dtype)
            new_low = np.concatenate([orig_low, pad_space * 0], axis=2)
            new_high = np.concatenate([orig_high, pad_space], axis=2)
            new_space = Box(low=new_low, high=new_high, dtype=space.dtype)
            return new_space
    elif isinstance(space, Discrete):
        return Discrete(space.n * num_indicators)

    assert False, "agent_indicator space must be 1d, 2d, or 3d Box or Discrete, was {}".format(space)


def get_indicator_map(agents, type_only):
    if type_only:
        assert all(re.match("[a-z]+_[0-9]+", agent) for agent in agents), "when the `type_only` parameter is True to agent_indicator, the agent names must follow the `<type>_<n>` format"
        agent_id_map = {}
        type_idx_map = {}
        idx_num = 0
        for agent in agents:
            type = agent.split("_")[0]
            if type not in type_idx_map:
                type_idx_map[type] = idx_num
                idx_num += 1
            agent_id_map[agent] = type_idx_map[type]
        if idx_num == 1:
            warnings.warn("agent_indicator wrapper is degenerate, only one agent type; doing nothing")
        return agent_id_map
    else:
        return {agent: i for i, agent in enumerate(agents)}


def check_params(spaces):
    spaces = list(spaces)
    first_space = spaces[0]
    for space in spaces:
        assert repr(space) == repr(first_space), "spaces need to be the same shape to add an indicator. Try using the `pad_observations` wrapper before agent_indicator."
        change_obs_space(space, 1)


def change_observation(obs, space, indicator_data):
    indicator_num, num_indicators = indicator_data
    assert 0 <= indicator_num < num_indicators
    if isinstance(space, Box):
        ndims = len(space.shape)
        if ndims == 1:
            old_len = len(obs)
            new_obs = np.pad(obs, (0, num_indicators))
            new_obs[indicator_num + old_len] = 1.0
            return new_obs
        elif ndims == 3 or ndims == 2:
            obs = obs if ndims == 3 else np.expand_dims(obs, 2)
            old_shaped3 = obs.shape[2]
            new_obs = np.pad(obs, [(0, 0), (0, 0), (0, num_indicators)])
            new_obs[:, :, old_shaped3 + indicator_num] = 1.0
            return new_obs
    elif isinstance(space, Discrete):
        return obs * num_indicators + indicator_num

    assert False, "agent_indicator space must be 1d, 2d, or 3d Box or Discrete, was {}".format(space)
