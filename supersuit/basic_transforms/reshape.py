def check_param(param):
    assert self.reshape in OBS_RESHAPE_LIST, "reshape must be in {}".format(OBS_RESHAPE_LIST)


def change_space(observation_spaces,param):
    for agent in self.agents:
        obs_space = self.observation_spaces[agent]
        reshape = self.reshape
        dtype = obs_space.dtype
        if reshape is OBS_RESHAPE_LIST[0]:
            # expand dim by 1
            low = np.expand_dims(obs_space.low, axis=-1)
            high = np.expand_dims(obs_space.high, axis=-1)
        elif reshape is OBS_RESHAPE_LIST[1]:
            # flatten
            low = obs_space.low.flatten()
            high = obs_space.high.flatten()
        self.observation_spaces[agent] = Box(low=low, high=high, dtype=dtype)
    print("Mod obs space: reshape", self.observation_spaces)

def change_observation(obs,param):
    reshape = self.reshape
    if reshape is OBS_RESHAPE_LIST[0]:
        # expand dim by 1
        obs = np.expand_dims(obs, axis=-1)
    elif reshape is OBS_RESHAPE_LIST[1]:
        # flatten
        obs = obs.flatten()
    return obs
