from pettingzoo.utils.env import AECEnv, ParallelEnv
from supersuit.utils.action_transforms import homogenize_ops
from supersuit import observation_lambda_v0, action_lambda_v1


def pad_action_space_v0(env):
    assert isinstance(env, AECEnv) or isinstance(env, ParallelEnv), "pad_action_space_v0 only accepts an AECEnv or ParallelEnv"
    homogenize_ops.check_homogenize_spaces(list(env.action_spaces.values()))
    padded_space = homogenize_ops.homogenize_spaces(list(env.action_spaces.values()))
    return action_lambda_v1(env,
        lambda action, act_space: homogenize_ops.dehomogenize_actions(act_space, action),
        lambda act_space: padded_space)


def pad_observations_v0(env):
    assert isinstance(env, AECEnv) or isinstance(env, ParallelEnv), "pad_observations_v0 only accepts an AECEnv or ParallelEnv"
    spaces = list(env.observation_spaces.values())
    homogenize_ops.check_homogenize_spaces(spaces)
    padded_space = homogenize_ops.homogenize_spaces(spaces)
    return observation_lambda_v0(env,
        lambda obs, obs_space: homogenize_ops.homogenize_observations(obs_space, obs),
        lambda obs_space: padded_space)
