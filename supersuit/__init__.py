import gym
from . import vector_constructors
from . import aec_vector
from .utils.wrapper_chooser import WrapperChooser

__version__ = "2.6.5"


class DeprecatedWrapper(ImportError):
    pass


class Deprecated:
    def __init__(self, wrapper_name, orig_version, new_version):
        self.name = wrapper_name
        self.old_version, self.new_version = orig_version, new_version

    def __call__(self, env, *args, **kwargs):
        raise DeprecatedWrapper(f"{self.name}_{self.old_version} is now Deprecated, use {self.name}_{self.new_version} instead")

from .lambda_wrappers import action_lambda_v1, observation_lambda_v0, reward_lambda_v0
from .basic_wrappers import color_reduction_v0, resize_v0, dtype_v0, \
        flatten_v0, reshape_v0, normalize_obs_v0, clip_actions_v0, clip_reward_v0
from .more_wrappers import delay_observations_v0, frame_stack_v1, max_observation_v0, sticky_actions_v0
from .aec_wrappers import agent_indicator_aec, pad_observations_aec, black_death_aec, pad_action_space_aec, frame_skip_aec
from .gym_wrappers import frame_skip_gym

frame_skip_v0 = WrapperChooser(aec_wrapper=frame_skip_aec, gym_wrapper=frame_skip_gym)
agent_indicator_v0 = WrapperChooser(aec_wrapper=agent_indicator_aec)
pad_observations_v0 = WrapperChooser(aec_wrapper=pad_observations_aec)
black_death_v1 = WrapperChooser(aec_wrapper=black_death_aec)
pad_action_space_v0 = WrapperChooser(aec_wrapper=pad_action_space_aec)

black_death_v0 = Deprecated("black_death", "v0", "v1")
frame_stack_v0 = Deprecated("frame_stack", "v0", "v1")
action_lambda_v0 = Deprecated("action_lambda", "v0", "v1")

gym_vec_env_v0 = vector_constructors.gym_vec_env
stable_baselines_vec_env_v0 = vector_constructors.stable_baselines_vec_env
stable_baselines3_vec_env_v0 = vector_constructors.stable_baselines3_vec_env
vectorize_aec_env_v0 = aec_vector.vectorize_aec_env
concat_vec_envs_v0 = vector_constructors.concat_vec_envs
pettingzoo_env_to_vec_env_v0 = vector_constructors.pettingzoo_env_to_vec_env
