import numpy as np
from .single_vec_env import SingleVecEnv
import gym.vector


def transpose(ll):
    return [[ll[i][j] for i in range(len(ll))] for j in range(len(ll[0]))]


class ConcatVecEnv(gym.vector.VectorEnv):
    def __init__(self, vec_env_fns, obs_space=None, act_space=None):
        self.vec_envs = vec_envs = [vec_env_fn() for vec_env_fn in vec_env_fns]
        for i in range(len(vec_envs)):
            if not hasattr(vec_envs[i], "num_envs"):
                vec_envs[i] = SingleVecEnv([lambda: vec_envs[i]])
        self.metadata = self.vec_envs[0].metadata
        self.observation_space = vec_envs[0].observation_space
        self.action_space = vec_envs[0].action_space
        tot_num_envs = sum(env.num_envs for env in vec_envs)
        self.num_envs = tot_num_envs
        self.obs_buffer = np.empty((self.num_envs,) + self.observation_space.shape, dtype=self.observation_space.dtype)

    def concat_obs(self, obs_list):
        idx = 0
        for venv, obs in zip(self.vec_envs, obs_list):
            endidx = idx + venv.num_envs
            self.obs_buffer[idx:endidx] = obs
            idx = endidx
        return self.obs_buffer

    def seed(self, seed=None):
        if seed is None:
            for env in self.vec_envs:
                env.seed(None)
        else:
            for env in self.vec_envs:
                env.seed(seed)
                seed += env.num_envs

    def reset(self):
        return self.concat_obs([vec_env.reset() for vec_env in self.vec_envs])

    def step_async(self, actions):
        self._saved_actions = actions

    def step_wait(self):
        return self.step(self._saved_actions)

    def step(self, actions):
        data = []
        idx = 0
        for venv in self.vec_envs:
            data.append(venv.step(actions[idx: idx + venv.num_envs]))
            idx += venv.num_envs
        observations, rewards, dones, infos = transpose(data)
        observations = self.concat_obs(observations)
        rewards = np.concatenate(rewards, axis=0)
        dones = np.concatenate(dones, axis=0)
        infos = sum(infos, [])
        return observations, rewards, dones, infos

    def render(self, mode="human"):
        return self.vec_envs[0].render(mode)

    def close(self):
        for vec_env in self.vec_envs:
            vec_env.close()

    def env_is_wrapped(self, wrapper_class):
        return sum([sub_venv.env_is_wrapped(wrapper_class) for sub_venv in self.vec_envs], [])
