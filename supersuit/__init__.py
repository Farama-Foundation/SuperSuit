from .generic_wrappers import * # NOQA
from .lambda_wrappers import action_lambda_v1, observation_lambda_v0, reward_lambda_v0 # NOQA
from .multiagent_wrappers import agent_indicator_v0, black_death_v1, black_death_v0, \
    pad_action_space_v0, pad_observations_v0 # NOQA
from supersuit.generic_wrappers import frame_skip_v0, color_reduction_v0, resize_v0, dtype_v0, \
    flatten_v0, reshape_v0, normalize_obs_v0, clip_actions_v0, clip_reward_v0, \
    delay_observations_v0, frame_stack_v1, frame_stack_v0, max_observation_v0, \
    sticky_actions_v0 # NOQA

from .vector.vector_constructors import gym_vec_env_v0, stable_baselines_vec_env_v0, \
    stable_baselines3_vec_env_v0, concat_vec_envs_v0, pettingzoo_env_to_vec_env_v0 # NOQA

from .aec_vector import vectorize_aec_env_v0 # NOQA


__version__ = "3.0.0"


import warnings
warnings.warn("You're using SuperSuit 3.0, released 7/7/21. The entire codebase has been rewritten or refactored as part of this release. While we've tested it thoroughly, please ensure everything you're doing still works properly and report any issues at https://github.com/PettingZoo-Team/SuperSuit. This warning will be removed 2 months after release.")
