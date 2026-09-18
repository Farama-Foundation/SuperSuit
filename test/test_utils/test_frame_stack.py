import numpy as np
import pytest
from gymnasium.spaces import Box, Dict, Discrete

from supersuit.utils.frame_stack import stack_init, stack_obs, stack_obs_space


stack_obs_space_3d = Box(low=np.float32(0.0), high=np.float32(1.0), shape=(4, 4, 3))
stack_obs_space_2d = Box(low=np.float32(0.0), high=np.float32(1.0), shape=(4, 3))
stack_obs_space_1d = Box(low=np.float32(0.0), high=np.float32(1.0), shape=(3,))

stack_discrete = Discrete(3)

stack_obs_space_dict = Dict(
    {
        "camera": stack_obs_space_3d,
        "position": stack_obs_space_1d,
    }
)

STACK_SIZE = 11


def test_obs_space():
    assert stack_obs_space(stack_obs_space_1d, STACK_SIZE).shape == (3 * STACK_SIZE,)
    assert stack_obs_space(stack_obs_space_2d, STACK_SIZE).shape == (4, 3, STACK_SIZE)
    assert stack_obs_space(stack_obs_space_3d, STACK_SIZE).shape == (
        4,
        4,
        3 * STACK_SIZE,
    )
    assert stack_obs_space(stack_discrete, STACK_SIZE).n == 3**STACK_SIZE


def stack_obs_helper(frame_stack_list, obs_space, stack_size):
    stack = stack_init(
        obs_space, stack_size
    )  # stack_reset_obs(frame_stack_list[0], stack_size)
    for obs in frame_stack_list:
        stack = stack_obs(stack, obs, obs_space, stack_size)
    return stack


def test_change_observation():
    assert stack_obs_helper(
        [stack_obs_space_1d.low], stack_obs_space_1d, STACK_SIZE
    ).shape == (3 * STACK_SIZE,)
    assert stack_obs_helper(
        [stack_obs_space_1d.low, stack_obs_space_1d.high],
        stack_obs_space_1d,
        STACK_SIZE,
    ).shape == (3 * STACK_SIZE,)
    assert stack_obs_helper(
        [stack_obs_space_2d.low], stack_obs_space_2d, STACK_SIZE
    ).shape == (4, 3, STACK_SIZE)
    assert stack_obs_helper(
        [stack_obs_space_2d.low, stack_obs_space_2d.high],
        stack_obs_space_2d,
        STACK_SIZE,
    ).shape == (4, 3, STACK_SIZE)
    assert stack_obs_helper(
        [stack_obs_space_3d.low], stack_obs_space_3d, STACK_SIZE
    ).shape == (4, 4, 3 * STACK_SIZE)

    assert stack_obs_helper([1, 2], stack_discrete, STACK_SIZE) == 2 + 1 * 3

    stacked = stack_obs_helper(
        [stack_obs_space_2d.low, stack_obs_space_2d.high], stack_obs_space_2d, 3
    )
    raw = np.stack(
        [
            np.zeros_like(stack_obs_space_2d.high),
            stack_obs_space_2d.low,
            stack_obs_space_2d.high,
        ],
        axis=2,
    )
    assert np.all(np.equal(stacked, raw))

    stacked = stack_obs_helper(
        [stack_obs_space_3d.low, stack_obs_space_3d.high], stack_obs_space_3d, 3
    )
    raw = np.concatenate(
        [
            np.zeros_like(stack_obs_space_3d.high),
            stack_obs_space_3d.low,
            stack_obs_space_3d.high,
        ],
        axis=2,
    )
    assert np.all(np.equal(stacked, raw))


def test_dict_obs_space():
    stacked_space = stack_obs_space(stack_obs_space_dict, STACK_SIZE)
    assert isinstance(stacked_space, Dict)
    assert set(stacked_space.spaces) == {"camera", "position"}
    assert stacked_space["camera"].shape == (4, 4, 3 * STACK_SIZE)
    assert stacked_space["position"].shape == (3 * STACK_SIZE,)
    assert stacked_space["camera"].dtype == stack_obs_space_3d.dtype


def test_dict_nested_obs_space():
    nested = Dict({"sensors": stack_obs_space_dict, "flat": stack_obs_space_1d})
    stacked_space = stack_obs_space(nested, STACK_SIZE)
    assert stacked_space["sensors"]["camera"].shape == (4, 4, 3 * STACK_SIZE)
    assert stacked_space["flat"].shape == (3 * STACK_SIZE,)


def test_dict_init():
    stack = stack_init(stack_obs_space_dict, STACK_SIZE)
    assert set(stack) == {"camera", "position"}
    assert stack["camera"].shape == (4, 4, 3 * STACK_SIZE)
    assert stack["position"].shape == (3 * STACK_SIZE,)
    assert np.all(stack["camera"] == 0)


def test_dict_change_observation():
    """A Dict stack must match stacking each subspace independently."""
    obs_lo = {"camera": stack_obs_space_3d.low, "position": stack_obs_space_1d.low}
    obs_hi = {"camera": stack_obs_space_3d.high, "position": stack_obs_space_1d.high}

    stacked = stack_obs_helper([obs_lo, obs_hi], stack_obs_space_dict, 3)

    for key, subspace in stack_obs_space_dict.spaces.items():
        expected = stack_obs_helper([obs_lo[key], obs_hi[key]], subspace, 3)
        assert np.all(np.equal(stacked[key], expected))

    assert stacked["camera"].shape == (4, 4, 3 * 3)
    assert stacked["position"].shape == (3 * 3,)


def test_dict_rejects_unstackable_subspace():
    from gymnasium.spaces import MultiBinary

    bad = Dict({"ok": stack_obs_space_1d, "bad": MultiBinary(4)})
    with pytest.raises(AssertionError):
        stack_obs_space(bad, STACK_SIZE)
