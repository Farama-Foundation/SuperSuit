import gymnasium.vector
import numpy as np
from gymnasium.vector.utils import concatenate, create_empty_array, iterate
from gymnasium.vector.vector_env import AutoresetMode

from .utils.info_dict import list_infos_to_dict


def dict_infos_to_list(vector_infos: dict, num_envs: int) -> list:
    """Convert Gymnasium vector infos dict back to a list of per-env dicts."""
    list_info = [{} for _ in range(num_envs)]
    if not vector_infos:
        return list_info

    for key, value in vector_infos.items():
        if key.startswith("_"):
            continue

        binary_key = f"_{key}"
        if isinstance(value, dict):
            value_list_info = dict_infos_to_list(value, num_envs)
            if binary_key in vector_infos:
                for env_num, (env_info, has_info) in enumerate(
                    zip(value_list_info, vector_infos[binary_key])
                ):
                    if has_info:
                        list_info[env_num][key] = env_info
            else:
                for env_num, sub_value in enumerate(value_list_info):
                    list_info[env_num][key] = sub_value
        else:
            if binary_key in vector_infos:
                for env_num, has_info in enumerate(vector_infos[binary_key]):
                    if has_info:
                        list_info[env_num][key] = value[env_num]
            else:
                for env_num, sub_value in enumerate(value):
                    list_info[env_num][key] = sub_value
    return list_info


class MarkovVectorEnv(gymnasium.vector.VectorEnv):
    def __init__(self, par_env, black_death=False):
        """
        parameters:
            - par_env: the pettingzoo Parallel environment that will be converted to a gymnasium vector environment
            - black_death: whether to give zero valued observations and 0 rewards when an agent is done, allowing for environments with multiple numbers of agents.
                            Is equivalent to adding the black death wrapper, but somewhat more efficient.

        The resulting object will be a valid vector environment that has a num_envs
        parameter equal to the max number of agents, will return an array of observations,
        rewards, dones, etc, and will reset environment automatically when it finishes
        """
        self.par_env = par_env
        self.metadata = dict(par_env.metadata) if par_env.metadata is not None else {}
        # MarkovVectorEnv autoresets all sub-envs in the same step when the
        # underlying PettingZoo env finishes (all agents done).
        self.metadata.setdefault("autoreset_mode", AutoresetMode.SAME_STEP)
        self.render_mode = par_env.unwrapped.render_mode
        # Gymnasium VectorEnv distinguishes single_* (per sub-env) from the
        # batched observation_space / action_space. SuperSuit historically
        # exposed the per-agent spaces on observation_space / action_space so
        # that concatenate()/iterate() and downstream SB3 code keep working.
        self.single_observation_space = par_env.observation_space(
            par_env.possible_agents[0]
        )
        self.single_action_space = par_env.action_space(par_env.possible_agents[0])
        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space
        assert all(
            self.single_observation_space == par_env.observation_space(agent)
            for agent in par_env.possible_agents
        ), "observation spaces not consistent. Perhaps you should wrap with `supersuit.multiagent_wrappers.pad_observations_v0`?"
        assert all(
            self.single_action_space == par_env.action_space(agent)
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

    def _agent_info_list(self, infos):
        """Build a per-agent list of info dicts from a PettingZoo infos mapping.

        Non-agent (global) keys are propagated into every agent's info dict so
        they are not silently dropped. Per-agent keys take precedence.
        """
        agent_set = set(self.par_env.possible_agents)
        global_infos = {
            key: value for key, value in infos.items() if key not in agent_set
        }
        return [
            {**global_infos, **infos.get(agent, {})}
            for agent in self.par_env.possible_agents
        ]

    def _vectorize_infos(self, infos):
        """Return Gymnasium vector-style infos (dict of arrays + masks)."""
        return list_infos_to_dict(self._agent_info_list(infos))

    def reset(self, seed=None, options=None):
        _observations, infos = self.par_env.reset(seed=seed, options=options)
        observations = self.concat_obs(_observations)
        return observations, self._vectorize_infos(infos)

    def step(self, actions):
        actions = list(iterate(self.action_space, actions))
        agent_set = set(self.par_env.agents)
        act_dict = {
            agent: actions[i]
            for i, agent in enumerate(self.par_env.possible_agents)
            if agent in agent_set
        }
        observations, rewards, terms, truncs, infos = self.par_env.step(act_dict)

        # adds last observation to info where user can get it
        terminations = np.fromiter(terms.values(), dtype=bool)
        truncations = np.fromiter(truncs.values(), dtype=bool)
        env_done = (terminations | truncations).all()
        if env_done:
            for agent, obs in observations.items():
                infos[agent]["terminal_observation"] = obs

        rews = np.array(
            [rewards.get(agent, 0) for agent in self.par_env.possible_agents],
            dtype=np.float32,
        )
        tms = np.array(
            [terms.get(agent, False) for agent in self.par_env.possible_agents],
            dtype=np.uint8,
        )
        tcs = np.array(
            [truncs.get(agent, False) for agent in self.par_env.possible_agents],
            dtype=np.uint8,
        )
        # Per-agent info dicts for the step (before any auto-reset)
        step_info_list = self._agent_info_list(infos)

        if env_done:
            observations, reset_infos = self.reset()
            # reset_infos is already vector-dict form; convert to list so we can
            # merge per-env with step infos (reset keys win on collision).
            reset_list = dict_infos_to_list(reset_infos, self.num_envs)
        else:
            observations = self.concat_obs(observations)
            reset_list = [{} for _ in range(self.num_envs)]

        # combine standard infos and reset infos (reset keys win on collision)
        merged_list = [
            {**inf, **reset_inf} for inf, reset_inf in zip(step_info_list, reset_list)
        ]
        infs = list_infos_to_dict(merged_list)

        assert (
            self.black_death or self.par_env.agents == self.par_env.possible_agents
        ), "MarkovVectorEnv does not support environments with varying numbers of active agents unless black_death is set to True"
        return observations, rews, tms, tcs, infs

    def render(self):
        return self.par_env.render()

    def close(self):
        return self.par_env.close()

    def env_is_wrapped(self, wrapper_class):
        """
        env_is_wrapped only suppors vector and gymnasium environments
        currently, not pettingzoo environments
        """
        return [False] * self.num_envs
