from pettingzoo.utils.wrappers import OrderEnforcingWrapper as PZBaseWrapper


class BaseWrapper(PZBaseWrapper):
    def __init__(self, env):
        """
        Creates a wrapper around `env`. Extend this class to create changes to the space.
        """
        super().__init__(env)

        self._check_wrapper_params()

        self._modify_spaces()

    def _check_wrapper_params(self):
        pass

    def _modify_spaces(self):
        pass

    def _modify_action(self, agent, action):
        raise NotImplementedError()

    def _modify_observation(self, agent, observation):
        raise NotImplementedError()

    def _update_step(self, agent):
        pass

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._update_step(self.agent_selection)

    def observe(self, agent):
        obs = super().observe(
            agent
        )  # problem is in this line, the obs is sometimes a different size from the obs space
        observation = self._modify_observation(agent, obs)
        return observation

    def step(self, action):
        agent = self.env.agent_selection
        if not (self.terminations[agent] or self.truncations[agent]):
            action = self._modify_action(agent, action)

        super().step(action)

        self._update_step(self.agent_selection)
