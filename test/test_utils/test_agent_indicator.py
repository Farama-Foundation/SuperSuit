from supersuit.utils.agent_indicator import (
    change_obs_space,
    change_observation,
    get_indicator_map,
)
from gym.spaces import Box, Discrete
import numpy as np
import pytest

obs_space_3d = Box(low=np.float32(0.0), high=np.float32(1.0), shape=(4, 4, 3))
obs_space_2d = Box(low=np.float32(0.0), high=np.float32(1.0), shape=(4, 3))
obs_space_1d = Box(low=np.float32(0.0), high=np.float32(1.0), shape=(3,))

discrete_space = Discrete(3)

NUM_INDICATORS = 11


def test_obs_space():
    assert change_obs_space(obs_space_1d, NUM_INDICATORS).shape == (3 + NUM_INDICATORS,)
    assert change_obs_space(obs_space_2d, NUM_INDICATORS).shape == (
        4,
        3,
        1 + NUM_INDICATORS,
    )
    assert change_obs_space(obs_space_3d, NUM_INDICATORS).shape == (
        4,
        4,
        3 + NUM_INDICATORS,
    )
    assert change_obs_space(discrete_space, NUM_INDICATORS).n == 3 * NUM_INDICATORS


def test_change_observation():
    assert change_observation(np.ones((4, 4, 3)), obs_space_3d, (4, NUM_INDICATORS)).shape == (4, 4, 3 + NUM_INDICATORS)
    assert change_observation(np.ones((4, 3)), obs_space_2d, (4, NUM_INDICATORS)).shape == (4, 3, 1 + NUM_INDICATORS)
    assert change_observation(np.ones((41)), obs_space_1d, (4, NUM_INDICATORS)).shape == (41 + NUM_INDICATORS,)

    assert change_observation(np.ones((4, 4, 3)), obs_space_3d, (4, NUM_INDICATORS))[0, 0, 0] == 1.0
    assert change_observation(np.ones((4, 4, 3)), obs_space_3d, (4, NUM_INDICATORS))[0, 0, 4] == 0.0
    assert change_observation(np.ones((4, 4, 3)), obs_space_3d, (4, NUM_INDICATORS))[0, 1, 7] == 1.0
    assert change_observation(np.ones((4, 4, 3)), obs_space_3d, (4, NUM_INDICATORS))[0, 0, 8] == 0.0
    assert change_observation(np.ones((3,)), obs_space_1d, (4, NUM_INDICATORS))[2] == 1.0
    assert change_observation(np.ones((3,)), obs_space_1d, (4, NUM_INDICATORS))[6] == 0.0
    assert change_observation(np.ones((3,)), obs_space_1d, (4, NUM_INDICATORS))[7] == 1.0
    assert change_observation(np.ones((3,)), obs_space_1d, (4, NUM_INDICATORS))[8] == 0.0
    assert change_observation(2, discrete_space, (4, NUM_INDICATORS)) == 2 * NUM_INDICATORS + 4


def test_get_indicator_map():
    assert len(get_indicator_map(["bob", "joe", "fren"], False)) == 3

    with pytest.raises(AssertionError):
        get_indicator_map(["bob", "joe", "fren"], True)

    assert len(set(get_indicator_map(["bob_0", "joe_1", "fren_2", "joe_3"], True).values())) == 3
