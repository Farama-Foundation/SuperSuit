from .generic_wrappers import * # NOQA
from .lambda_wrappers import action_lambda_v1, observation_lambda_v0, reward_lambda_v0 # NOQA
from .multiagent_wrappers import agent_indicator_v0, black_death_v2, \
    pad_action_space_v0, pad_observations_v0 # NOQA
from supersuit.generic_wrappers import frame_skip_v0, color_reduction_v0, resize_v0, dtype_v0, \
    flatten_v0, reshape_v0, normalize_obs_v0, clip_actions_v0, clip_reward_v0, \
    delay_observations_v0, frame_stack_v1, max_observation_v0, \
    sticky_actions_v0 # NOQA

from .vector.vector_constructors import gym_vec_env_v0, stable_baselines_vec_env_v0, \
    stable_baselines3_vec_env_v0, concat_vec_envs_v1, pettingzoo_env_to_vec_env_v1 # NOQA

from .aec_vector import vectorize_aec_env_v0 # NOQA


class DeprecatedWrapper(ImportError):
    pass


def __getattr__(wrapper_name):
    """
    Gives error that looks like this when trying to import old version of wrapper:
    File "./supersuit/__init__.py", line 38, in __getattr__
    raise DeprecatedWrapper(f"{base}{version_num} is now deprecated, use {base}{act_version_num} instead")
    supersuit.DeprecatedWrapper: concat_vec_envs_v0 is now deprecated, use concat_vec_envs_v1 instead
    """
    start_v = wrapper_name.rfind("_v") + 2
    version = wrapper_name[start_v:]
    base = wrapper_name[:start_v]
    try:
        version_num = int(version)
        is_valid_version = True
    except ValueError:
        is_valid_version = False

    globs = globals()
    if is_valid_version:
        for act_version_num in range(1000):
            if f"{base}{act_version_num}" in globs:
                if version_num < act_version_num:
                    raise DeprecatedWrapper(f"{base}{version_num} is now deprecated, use {base}{act_version_num} instead")

    raise ImportError(f"cannot import name '{wrapper_name}' from 'supersuit'")


__version__ = "3.3.2"
