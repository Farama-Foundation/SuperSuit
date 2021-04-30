from stable_baselines3.common.vec_env import VecEnvWrapper
import warnings


class SB3VecEnvWrapper(VecEnvWrapper):
    def __init__(self, venv):
        self.venv = venv
        self.num_envs = venv.num_envs
        self.observation_space = venv.observation_space
        self.action_space = venv.action_space

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        return self.venv.step_wait()

    def env_is_wrapped(self, wrapper_class, indices=None):
        # ignores indicies
        return self.venv.env_is_wrapped(wrapper_class)

    def getattr_recursive(self, name):
        raise AttributeError(name)

    def getattr_depth_check(self, name, already_found):
        return None
