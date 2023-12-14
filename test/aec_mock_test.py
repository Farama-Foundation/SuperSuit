from test.dummy_aec_env import DummyEnv

import numpy as np
import pytest
from gymnasium.spaces import Box, Discrete
from pettingzoo.utils.wrappers import OrderEnforcingWrapper as PettingzooWrap

import supersuit
from supersuit import (
    action_lambda_v1,
    dtype_v0,
    frame_stack_v1,
    frame_stack_v2,
    observation_lambda_v0,
    pad_action_space_v0,
    reshape_v0,
    scale_actions_v0,
)


base_obs = {
    f"a{idx}": np.zeros([8, 8, 3], dtype=np.float32) + np.arange(3) + idx
    for idx in range(2)
}
base_obs_space = {
    f"a{idx}": Box(low=np.float32(0.0), high=np.float32(10.0), shape=[8, 8, 3])
    for idx in range(2)
}
base_act_spaces = {f"a{idx}": Discrete(5) for idx in range(2)}


def test_frame_stack():
    # frame_stack_v0 test
    base_obs_space = {
        f"a{idx}": Box(low=np.float32(0.0), high=np.float32(10.0), shape=[2, 3])
        for idx in range(2)
    }
    base_obs = {
        f"a{idx}": np.float32(np.zeros([2, 3]) + np.arange(3) + idx) for idx in range(2)
    }
    base_env = DummyEnv(base_obs, base_obs_space, base_act_spaces)
    env = frame_stack_v1(base_env, 4)
    obs = env.reset()
    obs, _, _, _, _ = env.last()
    assert obs.shape == (2, 3, 4)
    env.step(2)
    first_obs, _, _, _, _ = env.last()
    assert np.array_equal(first_obs[:, :, -1], base_obs["a1"])

    base_obs = {f"a{idx}": idx + 3 for idx in range(2)}
    base_env = DummyEnv(base_obs, base_act_spaces, base_act_spaces)
    env = frame_stack_v1(base_env, 4)
    obs = env.reset()
    obs, _, _, _, _ = env.last()
    assert env.observation_space(env.agent_selection).n == 5**4
    env.step(2)
    first_obs, _, _, _, _ = env.last()
    assert first_obs == 4
    env.step(2)
    second_obs, _, _, _, _ = env.last()
    assert second_obs == 3 * 5 + 3
    for x in range(8):
        nth_obs = env.step(2)
        nth_obs, _, _, _, _ = env.last()
    assert nth_obs == ((3 * 5 + 3) * 5 + 3) * 5 + 3

    # frame_stack_v1 test
    base_obs_space = {
        f"a{idx}": Box(low=np.float32(0.0), high=np.float32(10.0), shape=[2, 3])
        for idx in range(2)
    }
    base_obs = {
        f"a{idx}": np.float32(np.zeros([2, 3]) + np.arange(3) + idx) for idx in range(2)
    }
    base_env = DummyEnv(base_obs, base_obs_space, base_act_spaces)
    env = frame_stack_v2(base_env, 4)
    obs = env.reset()
    obs, _, _, _, _ = env.last()
    assert obs.shape == (2, 3, 4)
    env.step(2)
    first_obs, _, _, _, _ = env.last()
    assert np.array_equal(first_obs[:, :, -1], base_obs["a1"])

    base_obs = {f"a{idx}": idx + 3 for idx in range(2)}
    base_env = DummyEnv(base_obs, base_act_spaces, base_act_spaces)
    env = frame_stack_v2(base_env, 4)
    obs = env.reset()
    obs, _, _, _, _ = env.last()
    assert env.observation_space(env.agent_selection).n == 5**4
    env.step(2)
    first_obs, _, _, _, _ = env.last()
    assert first_obs == ((4 * 5 + 4) * 5 + 4) * 5 + 4
    env.step(2)
    second_obs, _, _, _, _ = env.last()
    assert second_obs == ((3 * 5 + 3) * 5 + 3) * 5 + 3
    for x in range(8):
        nth_obs = env.step(2)
        nth_obs, _, _, _, _ = env.last()
    assert nth_obs == ((3 * 5 + 3) * 5 + 3) * 5 + 3


def test_frame_skip():
    base_env = DummyEnv(base_obs, base_obs_space, base_act_spaces)
    env = supersuit.frame_skip_v0(base_env, 3)
    env.reset()
    for a in env.agent_iter(8):
        obs, rew, term, trunc, info = env.last()
        env.step(0 if not (term or trunc) else None)


def test_agent_indicator():
    let = ["a", "a", "b"]
    base_obs = {f"{let[idx]}_{idx}": np.zeros([2, 3]) for idx in range(3)}
    base_obs_space = {
        f"{let[idx]}_{idx}": Box(
            low=np.float32(0.0), high=np.float32(10.0), shape=[2, 3]
        )
        for idx in range(3)
    }
    base_act_spaces = {f"{let[idx]}_{idx}": Discrete(5) for idx in range(3)}

    base_env = DummyEnv(base_obs, base_obs_space, base_act_spaces)
    env = supersuit.agent_indicator_v0(base_env, type_only=True)
    env.reset()
    obs, _, _, _, _ = env.last()
    assert obs.shape == (2, 3, 3)
    assert env.observation_space("a_0").shape == (2, 3, 3)
    env.step(2)

    env = supersuit.agent_indicator_v0(base_env, type_only=False)
    env.reset()
    obs, _, _, _, _ = env.last()
    assert obs.shape == (2, 3, 4)
    assert env.observation_space("a_0").shape == (2, 3, 4)
    env.step(2)


def test_agent_indicator_unbounded_box_space():
    """
    Test that if the observation space is unbounded e.g. space.high is inf,
    then the agent indicator wrapper will not return inf as agent indicator.
    """
    let = ["a", "a", "b"]
    base_obs = {f"{let[idx]}_{idx}": np.zeros([2, 3]) for idx in range(3)}
    base_obs_space = {
        f"{let[idx]}_{idx}": Box(low=0, high=np.inf, shape=[2, 3]) for idx in range(3)
    }
    base_act_spaces = {f"{let[idx]}_{idx}": Discrete(5) for idx in range(3)}

    base_env = DummyEnv(base_obs, base_obs_space, base_act_spaces)
    env = supersuit.agent_indicator_v0(base_env, type_only=True)
    env.reset()
    obs, _, _, _, _ = env.last()
    assert obs.shape == (2, 3, 3)
    assert env.observation_space("a_0").shape == (2, 3, 3)
    # check agent indication is not inf
    assert not np.isinf(obs).any()
    # check agent indicator is 1.0
    assert np.all(obs[:, :, 1] == 1.0)


def test_reshape():
    base_env = DummyEnv(base_obs, base_obs_space, base_act_spaces)
    env = reshape_v0(base_env, (64, 3))
    env.reset()
    obs, _, _, _, _ = env.last()
    assert obs.shape == (64, 3)
    env.step(5)
    first_obs, _, _, _, _ = env.last()
    assert np.all(np.equal(first_obs, base_obs["a1"].reshape([64, 3])))


def new_continuous_dummy():
    base_obs = {
        f"a_{idx}": (np.zeros([8, 8, 3], dtype=np.float32) + np.arange(3) + idx).astype(
            np.float32
        )
        for idx in range(2)
    }
    base_obs_space = {
        f"a_{idx}": Box(low=np.float32(0.0), high=np.float32(10.0), shape=[8, 8, 3])
        for idx in range(2)
    }
    base_act_spaces = {
        f"a_{idx}": Box(low=np.float32(0.0), high=np.float32(10.0), shape=[3])
        for idx in range(2)
    }

    return PettingzooWrap(DummyEnv(base_obs, base_obs_space, base_act_spaces))


def new_dummy():
    base_obs = {
        f"a_{idx}": (np.zeros([8, 8, 3], dtype=np.float32) + np.arange(3) + idx).astype(
            np.float32
        )
        for idx in range(2)
    }
    base_obs_space = {
        f"a_{idx}": Box(low=np.float32(0.0), high=np.float32(10.0), shape=[8, 8, 3])
        for idx in range(2)
    }
    base_act_spaces = {f"a_{idx}": Discrete(5) for idx in range(2)}

    return PettingzooWrap(DummyEnv(base_obs, base_obs_space, base_act_spaces))


wrappers = [
    supersuit.color_reduction_v0(new_dummy(), "R"),
    supersuit.resize_v1(dtype_v0(new_dummy(), np.uint8), x_size=5, y_size=10),
    supersuit.resize_v1(
        dtype_v0(new_dummy(), np.uint8), x_size=5, y_size=10, linear_interp=True
    ),
    supersuit.dtype_v0(new_dummy(), np.int32),
    supersuit.flatten_v0(new_dummy()),
    supersuit.reshape_v0(new_dummy(), (64, 3)),
    supersuit.normalize_obs_v0(new_dummy(), env_min=-1, env_max=5.0),
    supersuit.frame_stack_v1(new_dummy(), 8),
    supersuit.pad_observations_v0(new_dummy()),
    supersuit.pad_action_space_v0(new_dummy()),
    supersuit.agent_indicator_v0(new_dummy(), True),
    supersuit.agent_indicator_v0(new_dummy(), False),
    supersuit.reward_lambda_v0(new_dummy(), lambda x: x / 10),
    supersuit.clip_reward_v0(new_dummy()),
    supersuit.clip_actions_v0(new_continuous_dummy()),
    supersuit.scale_actions_v0(new_continuous_dummy(), 0.3),
    supersuit.frame_skip_v0(new_dummy(), 4),
    supersuit.sticky_actions_v0(new_dummy(), 0.75),
    supersuit.delay_observations_v0(new_dummy(), 3),
    supersuit.max_observation_v0(new_dummy(), 3),
    supersuit.nan_noop_v0(new_dummy(), 0),
    supersuit.nan_zeros_v0(new_dummy()),
    supersuit.nan_random_v0(new_dummy()),
]


@pytest.mark.parametrize("env", wrappers)
def test_basic_wrappers(env):
    env.reset(seed=5)
    obs, _, _, _, _ = env.last()
    act_space = env.action_space(env.agent_selection)
    obs_space = env.observation_space(env.agent_selection)
    first_obs = env.observe("a_0")
    assert obs_space.contains(first_obs)
    assert first_obs.dtype == obs_space.dtype
    env.step(act_space.sample())
    for agent in env.agent_iter():
        act_space = env.action_space(env.agent_selection)
        env.step(
            act_space.sample()
            if not (env.truncations[agent] or env.terminations[agent])
            else None
        )


def test_rew_lambda():
    env = supersuit.reward_lambda_v0(new_dummy(), lambda x: x / 10)
    env.reset()
    assert env.rewards[env.agent_selection] == 1.0 / 10


def test_observation_lambda():
    def add1(obs, obs_space):
        return obs + 1

    base_env = DummyEnv(base_obs, base_obs_space, base_act_spaces)
    env = observation_lambda_v0(base_env, add1)
    env.reset()
    obs0, _, _, _, _ = env.last()
    assert int(obs0[0][0][0]) == 1
    env = observation_lambda_v0(env, add1)
    env.reset()
    obs0, _, _, _, _ = env.last()
    assert int(obs0[0][0][0]) == 2

    def tile_obs(obs, obs_space):
        shape_size = len(obs.shape)
        tile_shape = [1] * shape_size
        tile_shape[0] *= 2
        return np.tile(obs, tile_shape)

    env = observation_lambda_v0(env, tile_obs)
    env.reset()
    obs0, _, _, _, _ = env.last()
    assert env.observation_space(env.agent_selection).shape == (16, 8, 3)

    def change_shape_fn(obs_space):
        return Box(low=0, high=1, shape=(32, 8, 3))

    env = observation_lambda_v0(env, tile_obs)
    env.reset()
    obs0, _, _, _, _ = env.last()
    assert env.observation_space(env.agent_selection).shape == (32, 8, 3)
    assert obs0.shape == (32, 8, 3)

    base_env = DummyEnv(base_obs, base_obs_space, base_act_spaces)
    env = observation_lambda_v0(
        base_env,
        lambda obs, obs_space, agent: obs + base_env.possible_agents.index(agent),
    )
    env.reset()
    obs0 = env.observe(env.agents[0])
    obs1 = env.observe(env.agents[1])

    assert int(obs0[0][0][0]) == 0
    assert int(obs1[0][0][0]) == 2
    assert (
        env.observation_space(env.agents[0]).high + 1
        == env.observation_space(env.agents[1]).high
    ).all()

    base_env = DummyEnv(base_obs, base_obs_space, base_act_spaces)
    env = observation_lambda_v0(
        base_env,
        lambda obs, obs_space, agent: obs + base_env.possible_agents.index(agent),
        lambda obs_space, agent: Box(
            obs_space.low, obs_space.high + base_env.possible_agents.index(agent)
        ),
    )
    env.reset()
    obs0 = env.observe(env.agents[0])
    obs1 = env.observe(env.agents[1])

    assert int(obs0[0][0][0]) == 0
    assert int(obs1[0][0][0]) == 2
    assert (
        env.observation_space(env.agents[0]).high + 1
        == env.observation_space(env.agents[1]).high
    ).all()


def test_action_lambda():
    def inc1(x, space):
        return x + 1

    def change_space_fn(space):
        return Discrete(space.n + 1)

    base_env = DummyEnv(base_obs, base_obs_space, base_act_spaces)
    env = action_lambda_v1(base_env, inc1, change_space_fn)
    env.reset()
    env.step(5)

    def one_hot(x, n):
        v = np.zeros(n)
        v[x] = 1
        return v

    act_spaces = {f"a{idx}": Box(low=0, high=1, shape=(15,)) for idx in range(2)}
    base_env = DummyEnv(base_obs, base_obs_space, act_spaces)
    env = action_lambda_v1(
        base_env,
        lambda action, act_space: one_hot(action, act_space.shape[0]),
        lambda act_space: Discrete(act_space.shape[0]),
    )
    assert env.action_space(env.possible_agents[0]).n == 15

    env.reset()
    env.step(2)

    act_spaces = {f"a{idx}": Box(low=0, high=1, shape=(15,)) for idx in range(2)}
    base_env = DummyEnv(base_obs, base_obs_space, act_spaces)
    env = action_lambda_v1(
        base_env,
        lambda action, act_space, agent: one_hot(
            action + base_env.possible_agents.index(agent), act_space.shape[0]
        ),
        lambda act_space, agent: Discrete(
            act_space.shape[0] + base_env.possible_agents.index(agent)
        ),
    )
    assert env.action_space(env.possible_agents[0]).n == 15
    assert env.action_space(env.possible_agents[1]).n == 16
    env.reset()
    env.step(2)


def test_dehomogenize():
    base_act_spaces = {f"a{idx}": Discrete(5 + idx) for idx in range(2)}

    base_env = DummyEnv(base_obs, base_obs_space, base_act_spaces)
    env = pad_action_space_v0(base_env)
    env.reset()
    assert all([env.action_space(agent).n == 6 for agent in env.possible_agents])
    env.step(5)


class DummyNaNEnv(DummyEnv):
    def step(self, action):
        super().step(action)
        assert not (
            np.isnan(action)
        ), "Action was a NaN, even though it should not have been."


def test_NaN_noop():
    base_env = DummyNaNEnv(base_obs, base_obs_space, base_act_spaces)
    wrapped_env = supersuit.nan_noop_v0(base_env, 0)
    wrapped_env.reset()
    wrapped_env.step(np.nan)


def test_NaN_zeros():
    base_env = DummyNaNEnv(base_obs, base_obs_space, base_act_spaces)
    wrapped_env = supersuit.nan_zeros_v0(base_env)
    wrapped_env.reset()
    wrapped_env.step(np.nan)


def test_NaN_random():
    base_env = DummyNaNEnv(base_obs, base_obs_space, base_act_spaces)
    wrapped_env = supersuit.nan_random_v0(base_env)
    wrapped_env.reset()
    wrapped_env.step(np.nan)


class DummyScaleEnv(DummyEnv):
    def step(self, action):
        self._observations[self.agent_selection] = action
        super().step(action)


def test_scale_action_wrapper():
    base_obs = {
        f"a{idx}": np.zeros([8, 8, 3], dtype=np.float32) + np.arange(3) + idx
        for idx in range(2)
    }
    base_obs_space = {
        f"a{idx}": Box(low=np.float32(0.0), high=np.float32(10.0), shape=[8, 8, 3])
        for idx in range(2)
    }
    base_act_spaces = {
        f"a{idx}": Box(low=np.float32(0.0), high=np.float32(10.0), shape=[3])
        for idx in range(2)
    }
    base_env = DummyScaleEnv(base_obs, base_obs_space, base_act_spaces)
    wrapped_env = scale_actions_v0(base_env, 2)
    wrapped_env.reset()
    wrapped_env.step(np.array([2, 1, 3], dtype=np.float32))
    scaled_action = wrapped_env.observe(wrapped_env.agents[0])
    assert (scaled_action == np.array([4, 2, 6], dtype=np.float32)).all()

    with pytest.raises(AssertionError):
        base_act_spaces = {f"a{idx}": Discrete(5) for idx in range(2)}
        base_env = DummyScaleEnv(base_obs, base_obs_space, base_act_spaces)
        wrapped_env = scale_actions_v0(base_env, 2)
