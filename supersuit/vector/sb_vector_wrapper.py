from stable_baselines.common.vec_env.base_vec_env import VecEnv


class SBVecEnvWrapper(VecEnv):
    def __init__(self, venv):
        self.venv = venv
        self.num_envs = venv.num_envs
        self.observation_space = venv.observation_space
        self.action_space = venv.action_space

    def reset(self):
        return self.venv.reset()

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        return self.venv.step_wait()

    def step(self, actions):
        return self.venv.step(actions)

    def close(self):
        del self.venv

    def seed(self, seed=None):
        self.venv.seed(seed)

    def get_attr(self, attr_name, indices=None):
        raise NotImplementedError()

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError()

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError()
