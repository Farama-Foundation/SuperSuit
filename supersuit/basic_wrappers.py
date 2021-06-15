from .utils import basic_transforms
from .lambda_wrappers import observation_lambda_v0, action_lambda_v1, reward_lambda_v0

def basic_obs_wrapper(env, module, param):
    def change_space(space):
        module.check_param(space, param)
        return module.change_obs_space(space, param)

    def change_obs(obs, obs_space):
        return module.change_observation(obs, obs_space, param),
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
    shape = (env_min, env_max)
    return basic_obs_wrapper(env, basic_transforms.normalize_obs, (env_min, env_max))

def clip_actions_v0(env):
    return action_lambda_v1(env, lambda action, act_space: np.clip(action, act_space.low, act_space.high))

def clip_reward_v0(env, lower_bound=-1, upper_bound=1):
    return reward_lambda_v0(env, lambda rew: max(min(rew, upper_bound), lower_bound))
