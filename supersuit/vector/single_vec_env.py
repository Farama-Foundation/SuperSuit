import gymnasium
import numpy as np
from gymnasium.vector.vector_env import AutoresetMode

from .utils.info_dict import list_infos_to_dict


class SingleVecEnv:
    def __init__(self, gym_env_fns, *args):
        assert len(gym_env_fns) == 1
        self.gym_env = gym_env_fns[0]()
        self.render_mode = self.gym_env.render_mode
        self.observation_space = self.gym_env.observation_space
        self.action_space = self.gym_env.action_space
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
        self.num_envs = 1
        self.metadata = dict(getattr(self.gym_env, "metadata", None) or {})
        self.metadata.setdefault("autoreset_mode", AutoresetMode.SAME_STEP)

    def reset(self, seed=None, options=None):
        result = self.gym_env.reset(seed=seed, options=options)
        if isinstance(result, tuple):
            observations, info = result
        else:
            observations, info = result, {}
        return np.expand_dims(observations, 0), list_infos_to_dict([info])

    def step_async(self, actions):
        self._saved_actions = actions

    def step_wait(self):
        return self.step(self._saved_actions)

    def render(self):
        return self.gym_env.render()

    def close(self):
        self.gym_env.close()

    def step(self, actions):
        observations, reward, term, trunc, info = self.gym_env.step(actions[0])
        if term or trunc:
            reset_result = self.gym_env.reset()
            if isinstance(reset_result, tuple):
                observations, reset_info = reset_result
            else:
                observations, reset_info = reset_result, {}
            info = {**info, **reset_info}
        observations = np.expand_dims(observations, 0)
        rewards = np.array([reward], dtype=np.float32)
        terms = np.array([term], dtype=np.uint8)
        truncs = np.array([trunc], dtype=np.uint8)
        infos = list_infos_to_dict([info])
        return observations, rewards, terms, truncs, infos

    def env_is_wrapped(self, wrapper_class):
        env_tmp = self.gym_env
        while isinstance(env_tmp, gymnasium.Wrapper):
            if isinstance(env_tmp, wrapper_class):
                return [True]
            env_tmp = env_tmp.env
        return [False]
