import gym
from pettingzoo.utils.env import AECEnv, ParallelEnv
from pettingzoo.utils.conversions import aec_to_parallel, parallel_to_aec


class WrapperChooser:
    def __init__(self, aec_wrapper=None, gym_wrapper=None, parallel_wrapper=None):
        assert aec_wrapper is not None or parallel_wrapper is not None, "either the aec wrapper or the parallel wrapper must be defined for all supersuit environments"
        self.aec_wrapper = aec_wrapper
        self.gym_wrapper = gym_wrapper
        self.parallel_wrapper = parallel_wrapper

    def __call__(self, env, *args, **kwargs):
        if isinstance(env, gym.Env):
            if self.gym_wrapper is None:
                raise ValueError(f"{self.wrapper_name} does not apply to gym environments, pettingzoo environments only")
            return self.gym_wrapper(env, *args, **kwargs)
        elif isinstance(env, AECEnv):
            if self.aec_wrapper is not None:
                return self.aec_wrapper(env, *args, **kwargs)
            else:
                return parallel_to_aec(self.parallel_wrapper(aec_to_parallel(env), *args, **kwargs))
        elif isinstance(env, ParallelEnv):
            if self.parallel_wrapper is not None:
                return self.parallel_wrapper(env, *args, **kwargs)
            else:
                return aec_to_parallel(self.aec_wrapper(parallel_to_aec(env), *args, **kwargs))
        else:
            raise ValueError("environment passed to supersuit wrapper must either be a gym environment or a pettingzoo environment")
