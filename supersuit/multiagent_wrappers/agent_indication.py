from supersuit.utils import agent_indicator as agent_ider
from pettingzoo.utils.env import AECEnv, ParallelEnv
from supersuit import observation_lambda_v0


def agent_indicator_v0(env, type_only=False):
    assert isinstance(env, AECEnv) or isinstance(env, ParallelEnv), "agent_indicator_v0 only accepts an AECEnv or ParallelEnv"
    if not hasattr(env, 'observation_spaces') or not hasattr(env, 'possible_agents'):
        raise AssertionError("environment passed to agent indicator wrapper must have the possible_agents and observation_spaces attributes.")

    indicator_map = agent_ider.get_indicator_map(env.possible_agents, type_only)
    num_indicators = len(set(indicator_map.values()))

    if hasattr(env, 'observation_spaces'):
        agent_ider.check_params(env.observation_spaces.values())

    return observation_lambda_v0(env,
        lambda obs, obs_space, agent: agent_ider.change_observation(
            obs,
            obs_space,
            (indicator_map[agent], num_indicators),
        ),
        lambda obs_space: agent_ider.change_obs_space(obs_space, num_indicators))
