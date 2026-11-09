"""Wrappers must reject an out-of-range argument where they are built, not several frames later.

``resize_v1``, ``reshape_v0`` and ``sticky_actions_v0`` already do this. The wrappers below used to
accept the value and fail inside the library afterwards, with an error naming one of its own
locals, which tells the caller nothing about the argument that was wrong.
"""

from test.dummy_gym_env import DummyEnv

import numpy as np
import pytest
from gymnasium.spaces import Box, Discrete

from supersuit import (
    delay_observations_v0,
    frame_skip_v0,
    frame_stack_v1,
    frame_stack_v2,
    max_observation_v0,
)
from supersuit.utils.frame_skip import check_transform_frameskip


base_obs = (np.zeros([8, 8, 3]) + np.arange(3)).astype(np.float32)
base_obs_space = Box(low=np.float32(0.0), high=np.float32(10.0), shape=[8, 8, 3])
base_act_spaces = Discrete(5)


def new_dummy():
    return DummyEnv(base_obs, base_obs_space, base_act_spaces)


@pytest.mark.parametrize("num_frames", [0, -3])
def test_frame_skip_rejects_fewer_than_one_frame(num_frames):
    # The tuple branch already required this; the bare int skipped the check and the skip loop
    # then ran zero times, leaving step's return values unassigned.
    with pytest.raises(AssertionError, match="frame_skip must be"):
        check_transform_frameskip(num_frames)
    with pytest.raises(AssertionError, match="frame_skip must be"):
        frame_skip_v0(new_dummy(), num_frames)


@pytest.mark.parametrize("num_frames", [1, 4, (1, 4)])
def test_frame_skip_keeps_accepting_valid_values(num_frames):
    assert check_transform_frameskip(num_frames) == (
        num_frames if isinstance(num_frames, tuple) else (num_frames, num_frames)
    )


@pytest.mark.parametrize("memory", [0, -2])
def test_max_observation_rejects_fewer_than_one_frame(memory):
    with pytest.raises(AssertionError, match="memory must be at least 1"):
        max_observation_v0(new_dummy(), memory)


def test_max_observation_rejects_a_non_int_memory():
    with pytest.raises(AssertionError, match="memory must be an int"):
        max_observation_v0(new_dummy(), 2.5)


@pytest.mark.parametrize("delay", [-1, -5])
def test_delay_observations_rejects_a_negative_delay(delay):
    with pytest.raises(AssertionError, match="delay must be at least 0"):
        delay_observations_v0(new_dummy(), delay)


@pytest.mark.parametrize("stack", [frame_stack_v1, frame_stack_v2])
@pytest.mark.parametrize("stack_size", [0, -4])
def test_frame_stack_rejects_fewer_than_one_frame(stack, stack_size):
    with pytest.raises(AssertionError, match="stack_size must be at least 1"):
        stack(new_dummy(), stack_size)


@pytest.mark.parametrize("stack", [frame_stack_v1, frame_stack_v2])
@pytest.mark.parametrize("stack_dim", [1, 7, None])
def test_frame_stack_rejects_an_unsupported_stack_dim(stack, stack_dim):
    # get_tile_shape only builds a shape for -1 and 0, so any other value left tile_shape unset.
    with pytest.raises(AssertionError, match="stack_dim should be 0 or -1"):
        stack(new_dummy(), 4, stack_dim=stack_dim)


@pytest.mark.parametrize("stack", [frame_stack_v1, frame_stack_v2])
@pytest.mark.parametrize("stack_dim", [0, -1])
def test_frame_stack_keeps_accepting_both_supported_dims(stack, stack_dim):
    env = stack(new_dummy(), 4, stack_dim=stack_dim)
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape
