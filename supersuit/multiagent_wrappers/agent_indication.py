from supersuit.utils import agent_indicator as agent_ider
from pettingzoo.utils.env import AECEnv, ParallelEnv
from supersuit import observation_lambda_v0


def agent_indicator_v0(env, type_only=False):
    assert isinstance(env, AECEnv) or isinstance(env, ParallelEnv), "agent_indicator_v0 only accepts an AECEnv or ParallelEnv"
    assert hasattr(env, 'possible_agents'), "environment passed to agent indicator wrapper must have the possible_agents attribute."

    indicator_map = agent_ider.get_indicator_map(env.possible_agents, type_only)
    num_indicators = len(set(indicator_map.values()))

    obs_spaces = [env.observation_space(agent) for agent in env.possible_agents]
    agent_ider.check_params(obs_spaces)

    return observation_lambda_v0(env,
        lambda obs, obs_space, agent: agent_ider.change_observation(
            obs,
            obs_space,
            (indicator_map[agent], num_indicators),
        ),
        lambda obs_space: agent_ider.change_obs_space(obs_space, num_indicators))
