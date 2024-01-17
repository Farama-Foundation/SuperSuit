import warnings
from typing import Any, List, Optional

import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import VecEnvIndices


class SB3VecEnvWrapper(VecEnvWrapper):
    def __init__(self, venv):
        self.venv = venv
        self.num_envs = venv.num_envs
        self.observation_space = venv.observation_space
        self.action_space = venv.action_space
        self.render_mode = venv.render_mode
        self.reset_infos = []

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed=seed)
        # Note: SB3's vector envs return only observations on reset, and store infos in `self.reset_infos`
        observations, self.reset_infos = self.venv.reset()
        return observations

    def step_wait(self):
        observations, rewards, terminations, truncations, infos = self.venv.step_wait()
        # Note: SB3 expects dones to be an np.array
        dones = np.array(
            [terminations[i] or truncations[i] for i in range(len(terminations))]
        )
        return observations, rewards, dones, infos

    def env_is_wrapped(self, wrapper_class, indices=None):
        # ignores indices
        return self.venv.env_is_wrapped(wrapper_class)

    def getattr_recursive(self, name):
        raise AttributeError(name)

    def getattr_depth_check(self, name, already_found):
        return None

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        attr = self.venv.get_attr(attr_name)
        # Note: SB3 expects render_mode to be returned as an array, with values for each env
        if attr_name == "render_mode":
            return [attr for _ in range(self.num_envs)]
        else:
            return attr

    def render(self, mode: Optional[str] = None) -> Optional[np.ndarray]:
        warnings.warn(
            "PettingZoo environments do not take the `render(mode)` argument, to change rendering mode, re-initialize the environment using the `render_mode` argument."
        )
        return self.venv.render()
