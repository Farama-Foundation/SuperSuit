from test.dummy_aec_env import DummyEnv

import numpy as np
import pytest
from gymnasium.spaces import Box, Dict, Discrete

from supersuit import nan_random_v0


@pytest.mark.parametrize("legal_action", [0, 1])
def test_nan_random_respects_action_mask(legal_action):
    mask = np.zeros(3, dtype=np.int8)
    mask[legal_action] = 1
    observation = {"observation": np.zeros(1, dtype=np.float32), "action_mask": mask}
    observation_space = Dict(
        {
            "observation": Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "action_mask": Box(0, 1, shape=(3,), dtype=np.int8),
        }
    )

    class MaskedEnv(DummyEnv):
        def step(self, action):
            assert mask[action], "NaN replacement selected an illegal action"
            super().step(action)

    agents = ["a0", "a1"]
    base_env = MaskedEnv(
        {agent: observation for agent in agents},
        {agent: observation_space for agent in agents},
        # Unmasked sampling returns action 2, which is illegal in both cases.
        {agent: Discrete(3, seed=0) for agent in agents},
    )
    env = nan_random_v0(base_env)
    env.reset(seed=0)
    with pytest.warns(UserWarning, match="NaN action"):
        env.step(np.nan)
