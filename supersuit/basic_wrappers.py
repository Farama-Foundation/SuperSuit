from .utils import basic_transforms
from .lambda_wrappers import observation_lambda_v0, action_lambda_v1, reward_lambda_v0
import numpy as np
from .utils.action_transforms import homogenize_ops
from .utils import agent_indicator as agent_ider
from pettingzoo.utils.env import AECEnv, ParallelEnv


def basic_obs_wrapper(env, module, param):
    def change_space(space):
        module.check_param(space, param)
        space = module.change_obs_space(space, param)
        return space

    def change_obs(obs, obs_space):
        return module.change_observation(obs, obs_space, param)
    return observation_lambda_v0(env, change_obs, change_space)


def color_reduction_v0(env, mode="full"):
    return basic_obs_wrapper(env, basic_transforms.color_reduction, mode)


def resize_v0(env, x_size, y_size, linear_interp=False):
    scale_tuple = (x_size, y_size, linear_interp)
    return basic_obs_wrapper(env, basic_transforms.resize, scale_tuple)


def dtype_v0(env, dtype):
    return basic_obs_wrapper(env, basic_transforms.dtype, dtype)


def flatten_v0(env):
    return basic_obs_wrapper(env, basic_transforms.flatten, True)


def reshape_v0(env, shape):
    return basic_obs_wrapper(env, basic_transforms.reshape, shape)


def normalize_obs_v0(env, env_min=0.0, env_max=1.0):
    return basic_obs_wrapper(env, basic_transforms.normalize_obs, (env_min, env_max))


def clip_actions_v0(env):
    return action_lambda_v1(env,
        lambda action, act_space: np.clip(action, act_space.low, act_space.high),
        lambda act_space: act_space)


def clip_reward_v0(env, lower_bound=-1, upper_bound=1):
    return reward_lambda_v0(env, lambda rew: max(min(rew, upper_bound), lower_bound))


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


def agent_indicator_v0(env, type_only=False):
    assert isinstance(env, AECEnv) or isinstance(env, ParallelEnv), "agent_indicator_v0 only accepts an AECEnv or ParallelEnv"

    indicator_map = agent_ider.get_indicator_map(env.possible_agents, type_only)
    num_indicators = len(set(indicator_map.values()))

    agent_ider.check_params(env.observation_spaces.values())

    return observation_lambda_v0(env,
        lambda obs, obs_space, agent: agent_ider.change_observation(
            obs,
            obs_space,
            (indicator_map[agent], num_indicators),
        ),
        lambda obs_space: agent_ider.change_obs_space(obs_space, num_indicators))
