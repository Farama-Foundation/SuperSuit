import numpy as np
from gym import spaces

from supersuit import (
    agent_indicator_v0,
    black_death_v3,
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
    pad_action_space_v0,
    pad_observations_v0,
    scale_actions_v0,
    sticky_actions_v0,
)

from .dummy_aec_env import DummyEnv


def observation_homogenizable(env, agents):
    homogenizable = True
    for agent in agents:
        homogenizable = homogenizable and (
            isinstance(env.observation_space(agent), spaces.Box)
            or isinstance(env.observation_space(agent), spaces.Discrete)
        )
    return homogenizable


def action_homogenizable(env, agents):
    homogenizable = True
    for agent in agents:
        homogenizable = homogenizable and (
            isinstance(env.action_space(agent), spaces.Box)
            or isinstance(env.action_space(agent), spaces.Discrete)
        )
    return homogenizable


def image_observation(env, agents):
    imagable = True
    for agent in agents:
        if isinstance(env.observation_space(agent), spaces.Box):
            imagable = imagable and (env.observation_space(agent).low.shape == 3)
            imagable = imagable and (len(env.observation_space(agent).shape[2]) == 3)
            imagable = imagable and (env.observation_space(agent).low == 0).all()
            imagable = imagable and (env.observation_space(agent).high == 255).all()
        else:
            return False
    return imagable


def box_action(env, agents):
    boxable = True
    for agent in agents:
        boxable = boxable and isinstance(env.action_space(agent), spaces.Box)
    return boxable


def not_dict_observation(env, agents):
    is_dict = True
    for agent in agents:
        is_dict = is_dict and (isinstance(env.observation_space(agent), spaces.Dict))
    return not is_dict


def not_discrete_observation(env, agents):
    is_discrete = True
    for agent in agents:
        is_discrete = is_discrete and (
            isinstance(env.observation_space(agent), spaces.Discrete)
        )
    return not is_discrete


def not_multibinary_observation(env, agents):
    is_discrete = True
    for agent in agents:
        is_discrete = is_discrete and (
            isinstance(env.observation_space(agent), spaces.MultiBinary)
        )
    return not is_discrete


def unwrapped_check(env):
    env.reset()
    agents = env.agents

    if image_observation(env, agents):
        env = max_observation_v0(env, 2)
        env = color_reduction_v0(env, mode="full")
        env = normalize_obs_v0(env)

    if box_action(env, agents):
        env = clip_actions_v0(env)
        env = scale_actions_v0(env, 0.5)

    if observation_homogenizable(env, agents):
        env = pad_observations_v0(env)
        env = frame_stack_v1(env, 2)
        env = agent_indicator_v0(env)
        env = black_death_v3(env)

    if (
        not_dict_observation(env, agents)
        and not_discrete_observation(env, agents)
        and not_multibinary_observation(env, agents)
    ):
        env = dtype_v0(env, np.float16)
        env = flatten_v0(env)
        env = frame_skip_v0(env, 2)

    if action_homogenizable(env, agents):
        env = pad_action_space_v0(env)

    env = clip_reward_v0(env, lower_bound=-1, upper_bound=1)
    env = delay_observations_v0(env, 2)
    env = sticky_actions_v0(env, 0.5)
    env = nan_random_v0(env)
    env = nan_zeros_v0(env)

    assert env.unwrapped.__class__ == DummyEnv, f"Failed to unwrap {env}"


def test_unwrapped():
    observation_spaces = []
    base = spaces.Box(low=-1.0, high=1.0, shape=[2], dtype=np.float32)
    observation_spaces.append({f"a{i}": base for i in range(2)})
    base = spaces.Box(low=0, high=255, shape=[64, 64, 3], dtype=np.int8)
    observation_spaces.append({f"a{i}": base for i in range(2)})
    base = spaces.Discrete(5)
    observation_spaces.append({f"a{i}": base for i in range(2)})
    base = spaces.MultiBinary([3, 4])
    observation_spaces.append({f"a{i}": base for i in range(2)})

    action_spaces = []
    base = spaces.Box(-3.0, 3.0, [3], np.float32)
    action_spaces.append({f"a{i}": base for i in range(2)})
    base = spaces.Discrete(5)
    action_spaces.append({f"a{i}": base for i in range(2)})
    base = spaces.MultiDiscrete([4, 5])
    action_spaces.append({f"a{i}": base for i in range(2)})

    for obs_space in observation_spaces:
        for act_space in action_spaces:
            base_obs = {a: obs_space[a].sample() for a in obs_space}
            env = DummyEnv(base_obs, obs_space, act_space)
            unwrapped_check(env)
