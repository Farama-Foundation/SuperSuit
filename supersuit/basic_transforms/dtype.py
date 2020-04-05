def check_param(param):
    assert isinstance(self.new_dtype, type) or isinstance(self.new_dtype, dict), "new_dtype must be type or dict. It is {}".format(self.new_dtype)
    if isinstance(self.new_dtype, type):
        self.new_dtype = dict(zip(self.agents, [self.new_dtype for _ in enumerate(self.agents)]))
    if isinstance(self.new_dtype, dict):
        for agent in self.agents:
            assert agent in self.new_dtype.keys(), "Agent id {} is not a key of new_dtype {}".format(agent, self.new_dtype)
            assert isinstance(self.new_dtype[agent], type), "new_dtype[agent] must be a dict of types. It is {}".format(self.new_dtype[agent])


def change_space(observation_spaces,new_dtype):
    for agent in self.agents:
        obs_space = self.observation_spaces[agent]
        dtype = self.new_dtype[agent]
        low = obs_space.low
        high = obs_space.high
        self.observation_spaces[agent] = Box(low=low, high=high, dtype=dtype)
    print("Mod obs space: new_dtype", self.observation_spaces)

def change_observation(obs,new_dtype):
    dtype = self.new_dtype[agent]
    obs = obs.astype(dtype)
    return obs
