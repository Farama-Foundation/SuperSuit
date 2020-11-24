from gym.spaces import Box
from pettingzoo.utils.wrappers import OrderEnforcingWrapper as PettingzooWrap


class BaseWrapper(PettingzooWrap):
    def __init__(self, env):
        """
        Creates a wrapper around `env`. Extend this class to create changes to the space.
        """
        super().__init__(env)

        self._check_wrapper_params()

        self._modify_spaces()

    def _check_wrapper_params(self):
        raise NotImplementedError()

    def _modify_spaces(self):
        raise NotImplementedError()

    def _modify_action(self, agent, action):
        raise NotImplementedError()

    def _modify_observation(self, agent, observation):
        raise NotImplementedError()

    def _update_step(self, agent):
        raise NotImplementedError()

    def reset(self):
        super().reset()
        self._update_step(self.agent_selection)

    def observe(self, agent):
        obs = super().observe(agent)
        observation = self._modify_observation(agent, obs)
        return observation

    def step(self, action):
        agent = self.env.agent_selection
        if not self.dones[agent]:
            action = self._modify_action(agent, action)

        super().step(action)

        self._update_step(self.agent_selection)
