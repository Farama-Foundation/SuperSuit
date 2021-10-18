from pettingzoo.utils.env import AECEnv, ParallelEnv
from supersuit.utils.action_transforms import homogenize_ops
from supersuit import observation_lambda_v0, action_lambda_v1


def pad_action_space_v0(env):
    assert isinstance(env, AECEnv) or isinstance(env, ParallelEnv), "pad_action_space_v0 only accepts an AECEnv or ParallelEnv"
    assert hasattr(env, 'possible_agents'), "environment passed to pad_observations must have a possible_agents list."
    spaces = [env.action_space(agent) for agent in env.possible_agents]
    homogenize_ops.check_homogenize_spaces(spaces)
    padded_space = homogenize_ops.homogenize_spaces(spaces)
    return action_lambda_v1(env,
        lambda action, act_space: homogenize_ops.dehomogenize_actions(act_space, action),
        lambda act_space: padded_space)


def pad_observations_v0(env):
    assert isinstance(env, AECEnv) or isinstance(env, ParallelEnv), "pad_observations_v0 only accepts an AECEnv or ParallelEnv"
    assert hasattr(env, 'possible_agents'), "environment passed to pad_observations must have a possible_agents list."
    spaces = [env.observation_space(agent) for agent in env.possible_agents]
    homogenize_ops.check_homogenize_spaces(spaces)
    padded_space = homogenize_ops.homogenize_spaces(spaces)
    return observation_lambda_v0(env,
        lambda obs, obs_space: homogenize_ops.homogenize_observations(padded_space, obs),
        lambda obs_space: padded_space)
