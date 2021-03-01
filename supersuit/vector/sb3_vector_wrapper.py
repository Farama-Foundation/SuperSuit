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
