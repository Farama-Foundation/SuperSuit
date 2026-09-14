import numpy as np
import pytest
from gymnasium.spaces import Box, Dict, Discrete
from pettingzoo.utils import ParallelEnv

import supersuit


class DummyParEnv(ParallelEnv):
    metadata = {"render_modes": ["human"]}

    def __init__(self, observations, observation_spaces, action_spaces):
        super().__init__()
        self._observations = observations
        self._observation_spaces = observation_spaces

        self.agents = [x for x in observation_spaces.keys()]
        self.possible_agents = self.agents
        self.agent_selection = self.agents[0]
        self._action_spaces = action_spaces

        self.rewards = {a: 1 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    def step(self, actions):
        for agent, action in actions.items():
            assert action in self.action_space(agent)
        return (
            self._observations,
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos,
        )

    def reset(self, seed=None, options=None):
        return self._observations, self.infos

    def close(self):
        pass


base_obs = {
    f"a{idx}": np.zeros([8, 8, 3], dtype=np.float32) + np.arange(3) + idx
    for idx in range(2)
}
base_obs_space = {
    f"a{idx}": Box(low=np.float32(0.0), high=np.float32(10.0), shape=[8, 8, 3])
    for idx in range(2)
}
base_act_spaces = {f"a{idx}": Discrete(5) for idx in range(2)}

dict_obs = {
    f"a{idx}": {
        "camera": np.zeros([8, 8, 3], dtype=np.float32) + idx,
        "position": np.zeros([3], dtype=np.float32) + idx,
    }
    for idx in range(2)
}
dict_obs_space = {
    f"a{idx}": Dict(
        {
            "camera": Box(low=np.float32(0.0), high=np.float32(10.0), shape=[8, 8, 3]),
            "position": Box(low=np.float32(-1.0), high=np.float32(1.0), shape=[3]),
        }
    )
    for idx in range(2)
}


def test_basic():
    env = DummyParEnv(base_obs, base_obs_space, base_act_spaces)
    env = supersuit.delay_observations_v0(env, 4)
    env = supersuit.dtype_v0(env, np.uint8)
    env.reset()
    for i in range(10):
        action = {agent: env.action_space(agent).sample() for agent in env.agents}
        env.step(action)


@pytest.mark.parametrize(
    "terms_val,truncs_val,expected_done",
    [
        (True, False, True),  # all terminated -> episode done
        (False, True, True),  # all truncated  -> episode done  (main bug case)
        (True, True, True),  # both           -> episode done
        (False, False, False),  # neither         -> not done
    ],
)
def test_black_death_done_semantics(terms_val, truncs_val, expected_done):
    env = DummyParEnv(base_obs, base_obs_space, base_act_spaces)
    env.terminations = {a: terms_val for a in env.agents}
    env.truncations = {a: truncs_val for a in env.agents}
    env = supersuit.black_death_v3(env)
    env.reset()
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    _, _, terms, truncs, _ = env.step(actions)
    # black_death returns the same done-dict for both term and trunc slots
    assert all(terms.values()) == expected_done
    assert all(truncs.values()) == expected_done
    assert (len(env.agents) == 0) == expected_done


def test_black_death_dict_obs():
    """black_death must accept Dict observation spaces and zero-fill per key."""
    env = DummyParEnv(dict_obs, dict_obs_space, base_act_spaces)
    env = supersuit.black_death_v3(env)
    obs, _ = env.reset()

    assert set(obs) == {"a0", "a1"}
    for agent_obs in obs.values():
        assert set(agent_obs) == {"camera", "position"}
        assert agent_obs["camera"].shape == (8, 8, 3)
        assert agent_obs["position"].shape == (3,)

    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, _, _, _, _ = env.step(actions)
    for agent_obs in obs.values():
        assert agent_obs["camera"].shape == (8, 8, 3)


def test_black_death_dict_obs_zero_fills_dropped_agent():
    """An agent missing from the step output gets a structured zero observation."""
    env = DummyParEnv(dict_obs, dict_obs_space, base_act_spaces)
    wrapped = supersuit.black_death_v3(env)
    wrapped.reset()

    # drop a1 from the underlying env's output
    env._observations = {"a0": dict_obs["a0"]}
    env.agents = ["a0"]
    env.rewards = {"a0": 1}
    env.terminations = {"a0": False}
    env.truncations = {"a0": False}
    env.infos = {"a0": {}}

    obs, _, _, _, _ = wrapped.step({"a0": wrapped.action_space("a0").sample()})

    assert set(obs) == {"a0", "a1"}
    assert np.all(obs["a1"]["camera"] == 0)
    assert np.all(obs["a1"]["position"] == 0)
    assert obs["a1"]["camera"].shape == (8, 8, 3)
    assert obs["a1"]["position"].shape == (3,)


def test_black_death_rejects_unsupported_subspace():
    from gymnasium.spaces import MultiBinary

    bad_spaces = {
        f"a{idx}": Dict(
            {"ok": Box(low=0.0, high=1.0, shape=[3]), "bad": MultiBinary(4)}
        )
        for idx in range(2)
    }
    env = DummyParEnv(dict_obs, bad_spaces, base_act_spaces)
    env = supersuit.black_death_v3(env)
    with pytest.raises(AssertionError):
        env.reset()
