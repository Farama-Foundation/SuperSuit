import numpy as np
from .single_vec_env import SingleVecEnv
import gym.vector
from gym.vector.utils import concatenate, iterate, create_empty_array
from gym.spaces import Discrete


def transpose(ll):
    return [[ll[i][j] for i in range(len(ll))] for j in range(len(ll[0]))]


@iterate.register(Discrete)
def iterate_discrete(space, items):
    try:
        return iter(items)
    except TypeError:
        raise TypeError(f"Unable to iterate over the following elements: {items}")


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

    def concat_obs(self, observations):
        return concatenate(
            self.observation_space,
            [item for obs in observations for item in iterate(self.observation_space, obs)],
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
            data.append(venv.step(self.concatenate_actions(actions[idx: idx + venv.num_envs], venv.num_envs)))
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
