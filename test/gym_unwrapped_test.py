import numpy as np
from gym import spaces

from supersuit import (
    clip_actions_v0,
    clip_reward_v0,
    color_reduction_v0,
    delay_observations_v0,
    dtype_v0,
    flatten_v0,
    frame_skip_v0,
    frame_stack_v1,
    max_observation_v0,
    nan_random_v0,
    nan_zeros_v0,
    normalize_obs_v0,
    scale_actions_v0,
    sticky_actions_v0,
)

from .dummy_gym_env import DummyEnv


def unwrapped_check(env):
    # image observations
    if isinstance(env.observation_space, spaces.Box):
        if (
            (env.observation_space.low.shape == 3)
            and (env.observation_space.low == 0).all()
            and (len(env.observation_space.shape[2]) == 3)
            and (env.observation_space.high == 255).all()
        ):
            env = max_observation_v0(env, 2)
            env = color_reduction_v0(env, mode="full")
            env = normalize_obs_v0(env)

    # box action spaces
    if isinstance(env.action_space, spaces.Box):
        env = clip_actions_v0(env)
        env = scale_actions_v0(env, 0.5)

    # stackable observations
    if isinstance(env.observation_space, spaces.Box) or isinstance(
        env.observation_space, spaces.Discrete
    ):
        env = frame_stack_v1(env, 2)

    # not discrete and not multibinary observations
    if not isinstance(env.observation_space, spaces.Discrete) and not isinstance(
        env.observation_space, spaces.MultiBinary
    ):
        env = dtype_v0(env, np.float16)
        env = flatten_v0(env)
        env = frame_skip_v0(env, 2)

    # everything else
    env = clip_reward_v0(env, lower_bound=-1, upper_bound=1)
    env = delay_observations_v0(env, 2)
    env = sticky_actions_v0(env, 0.5)
    env = nan_random_v0(env)
    env = nan_zeros_v0(env)

    assert env.unwrapped.__class__ == DummyEnv, f"Failed to unwrap {env}"


def test_unwrapped():
    observation_spaces = []
    observation_spaces.append(
        spaces.Box(low=-1.0, high=1.0, shape=[2], dtype=np.float32)
    )
    observation_spaces.append(
        spaces.Box(low=0, high=255, shape=[64, 64, 3], dtype=np.int8)
    )
    observation_spaces.append(spaces.Discrete(5))
    observation_spaces.append(spaces.MultiBinary([3, 4]))

    action_spaces = []
    action_spaces.append(spaces.Box(-3.0, 3.0, [3], np.float32))
    action_spaces.append(spaces.Discrete(5))
    action_spaces.append(spaces.MultiDiscrete([3, 5]))

    for obs_space in observation_spaces:
        for act_space in action_spaces:
            env = DummyEnv(obs_space.sample(), obs_space, act_space)
            unwrapped_check(env)
