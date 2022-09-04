import numpy as np
import gym.vector
from gym.vector.utils import concatenate, iterate, create_empty_array


class MarkovVectorEnv(gym.vector.VectorEnv):
    def __init__(self, par_env, black_death=False):
        """
        parameters:
            - par_env: the pettingzoo Parallel environment that will be converted to a gym vector environment
            - black_death: whether to give zero valued observations and 0 rewards when an agent is done, allowing for environments with multiple numbers of agents.
                            Is equivalent to adding the black death wrapper, but somewhat more efficient.

        The resulting object will be a valid vector environment that has a num_envs
        parameter equal to the max number of agents, will return an array of observations,
        rewards, dones, etc, and will reset environment automatically when it finishes
        """
        self.par_env = par_env
        self.metadata = par_env.metadata
        self.observation_space = par_env.observation_space(par_env.possible_agents[0])
        self.action_space = par_env.action_space(par_env.possible_agents[0])
        assert all(
            self.observation_space == par_env.observation_space(agent)
            for agent in par_env.possible_agents
        ), "observation spaces not consistent. Perhaps you should wrap with `supersuit.multiagent_wrappers.pad_observations_v0`?"
        assert all(
            self.action_space == par_env.action_space(agent)
            for agent in par_env.possible_agents
        ), "action spaces not consistent. Perhaps you should wrap with `supersuit.multiagent_wrappers.pad_action_space_v0`?"
        self.num_envs = len(par_env.possible_agents)
        self.black_death = black_death

    def concat_obs(self, obs_dict):
        obs_list = []
        for i, agent in enumerate(self.par_env.possible_agents):
            if agent not in obs_dict:
                raise AssertionError(
                    "environment has agent death. Not allowed for pettingzoo_env_to_vec_env_v1 unless black_death is True"
                )
            obs_list.append(obs_dict[agent])

        return concatenate(
            self.observation_space,
            obs_list,
            create_empty_array(self.observation_space, self.num_envs),
        )

    def step_async(self, actions):
        self._saved_actions = actions

    def step_wait(self):
        return self.step(self._saved_actions)

    def reset(self, seed=None, return_info=False, options=None):
        if not return_info:
            _observations = self.par_env.reset(seed=seed, options=options)
            observations = self.concat_obs(_observations)
            return observations
        else:
            _observations, infos = self.par_env.reset(
                seed=seed, return_info=return_info, options=options
            )
            infs = [infos.get(agent, {}) for agent in self.par_env.possible_agents]
            observations = self.concat_obs(_observations)
            return observations, infs

    def step(self, actions):
        actions = list(iterate(self.action_space, actions))
        agent_set = set(self.par_env.agents)
        act_dict = {
            agent: actions[i]
            for i, agent in enumerate(self.par_env.possible_agents)
            if agent in agent_set
        }
        observations, rewards, dones, infos = self.par_env.step(act_dict)

        # adds last observation to info where user can get it
        if all(dones.values()):
            for agent, obs in observations.items():
                infos[agent]["terminal_observation"] = obs

        rews = np.array(
            [rewards.get(agent, 0) for agent in self.par_env.possible_agents],
            dtype=np.float32,
        )
        dns = np.array(
            [dones.get(agent, False) for agent in self.par_env.possible_agents],
            dtype=np.uint8,
        )
        infs = [infos.get(agent, {}) for agent in self.par_env.possible_agents]

        if all(dones.values()):
            observations = self.reset()
        else:
            observations = self.concat_obs(observations)
        assert (
            self.black_death or self.par_env.agents == self.par_env.possible_agents
        ), "MarkovVectorEnv does not support environments with varying numbers of active agents unless black_death is set to True"
        return observations, rews, dns, infs

    def render(self, mode="human"):
        return self.par_env.render(mode)

    def close(self):
        return self.par_env.close()

    def env_is_wrapped(self, wrapper_class):
        """
        env_is_wrapped only suppors vector and gym environments
        currently, not pettingzoo environments
        """
        return [False] * self.num_envs
