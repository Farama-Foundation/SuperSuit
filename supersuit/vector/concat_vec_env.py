import gymnasium.vector
import numpy as np
from gymnasium.spaces import Discrete
from gymnasium.vector.utils import concatenate, create_empty_array, iterate

from .single_vec_env import SingleVecEnv


def transpose(ll):
    return [[ll[i][j] for i in range(len(ll))] for j in range(len(ll[0]))]


@iterate.register(Discrete)
def iterate_discrete(space, items):
    try:
        return iter(items)
    except TypeError:
        raise TypeError(f"Unable to iterate over the following elements: {items}")


class ConcatVecEnv(gymnasium.vector.VectorEnv):
    def __init__(self, vec_env_fns, obs_space=None, act_space=None):
        self.vec_envs = vec_envs = [vec_env_fn() for vec_env_fn in vec_env_fns]
        for i in range(len(vec_envs)):
            if not hasattr(vec_envs[i], "num_envs"):
                vec_envs[i] = SingleVecEnv([lambda: vec_envs[i]])
        self.metadata = self.vec_envs[0].metadata
        self.render_mode = self.vec_envs[0].render_mode
        self.observation_space = vec_envs[0].observation_space
        self.action_space = vec_envs[0].action_space
        tot_num_envs = sum(env.num_envs for env in vec_envs)
        self.num_envs = tot_num_envs

    def reset(self, seed=None, options=None):
        _res_obs = []
        _res_infos = []

        if seed is not None:
            for i in range(len(self.vec_envs)):
                _obs, _info = self.vec_envs[i].reset(seed=seed + i, options=options)
                _res_obs.append(_obs)
                _res_infos.append(_info)
        else:
            for i in range(len(self.vec_envs)):
                _obs, _info = self.vec_envs[i].reset(options=options)
                _res_obs.append(_obs)
                _res_infos.append(_info)

        # flatten infos (also done in step function)
        flattened_infos = [info for sublist in _res_infos for info in sublist]

        return self.concat_obs(_res_obs), flattened_infos

    def concat_obs(self, observations):
        return concatenate(
            self.observation_space,
            [
                item
                for obs in observations
                for item in iterate(self.observation_space, obs)
            ],
            create_empty_array(self.observation_space, n=self.num_envs),
        )

    def concatenate_actions(self, actions, n_actions):
        return concatenate(
            self.action_space,
            actions,
            create_empty_array(self.action_space, n=n_actions),
        )

    def step_async(self, actions):
        self._saved_actions = actions

    def step_wait(self):
        return self.step(self._saved_actions)

    def step(self, actions):
        data = []
        idx = 0
        actions = list(iterate(self.action_space, actions))
        for venv in self.vec_envs:
            data.append(
                venv.step(
                    self.concatenate_actions(
                        actions[idx : idx + venv.num_envs], venv.num_envs
                    )
                )
            )
            idx += venv.num_envs
        observations, rewards, terminations, truncations, infos = transpose(data)
        observations = self.concat_obs(observations)
        rewards = np.concatenate(rewards, axis=0)
        terminations = np.concatenate(terminations, axis=0)
        truncations = np.concatenate(truncations, axis=0)
        infos = [
            info for sublist in infos for info in sublist
        ]  # flatten infos from nested lists
        return observations, rewards, terminations, truncations, infos

    def render(self):
        return self.vec_envs[0].render()

    def close(self):
        for vec_env in self.vec_envs:
            vec_env.close()

    def env_is_wrapped(self, wrapper_class):
        return sum(
            [sub_venv.env_is_wrapped(wrapper_class) for sub_venv in self.vec_envs], []
        )
