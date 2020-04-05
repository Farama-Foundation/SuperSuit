def check_param(param):
    assert isinstance(self.down_scale, tuple) or isinstance(self.down_scale, dict), "down_scale must be tuple or dict. It is {}".format(self.down_scale)
    if isinstance(self.down_scale, tuple):
        self.down_scale = dict(zip(self.agents, [self.down_scale for _ in enumerate(self.agents)]))
    if isinstance(self.down_scale, dict):
        for agent in self.agents:
            assert agent in self.down_scale.keys(), "Agent id {} is not a key of down_scale {}".format(agent, self.down_scale)

def change_space(observation_spaces,down_scale):
    for agent in self.agents:
        obs_space = self.observation_spaces[agent]
        dtype = obs_space.dtype
        down_scale = self.down_scale[agent]
        shape = obs_space.shape
        new_shape = tuple([int(shape[i] / down_scale[i]) for i in range(len(shape))])
        low = obs_space.low.flatten()[:np.product(new_shape)].reshape(new_shape)
        high = obs_space.high.flatten()[:np.product(new_shape)].reshape(new_shape)
        self.observation_spaces[agent] = Box(low=low, high=high, dtype=dtype)
    print("Mod obs space: down_scale", self.observation_spaces)

def change_observation(obs,down_scale):
    down_scale = self.down_scale[agent]
    mean = lambda x, axis: np.mean(x, axis=axis, dtype=np.uint8)
    obs = measure.block_reduce(obs, block_size=down_scale, func=mean)
    return obs
