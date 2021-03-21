from .concat_vec_env import ConcatVecEnv
from .multiproc_vec import ProcConcatVec


class call_wrap:
    def __init__(self, fn, data):
        self.fn = fn
        self.data = data

    def __call__(self, *args):
        return self.fn(self.data)


def MakeCPUAsyncConstructor(max_num_cpus):
    if max_num_cpus == 0 or max_num_cpus == 1:
        return ConcatVecEnv
    else:

        def constructor(env_fn_list, obs_space, act_space):
            example_env = env_fn_list[0]()
            envs_per_env = getattr(example_env, "num_envs", 1)

            num_fns = len(env_fn_list)
            envs_per_cpu = (num_fns + max_num_cpus - 1) // max_num_cpus
            alloced_num_cpus = (num_fns + envs_per_cpu - 1) // envs_per_cpu

            env_cpu_div = []
            num_envs_alloced = 0
            while num_envs_alloced < num_fns:
                start_idx = num_envs_alloced
                end_idx = min(num_fns, start_idx + envs_per_cpu)
                env_cpu_div.append(env_fn_list[start_idx:end_idx])
                num_envs_alloced = end_idx

            assert alloced_num_cpus == len(env_cpu_div)

            cat_env_fns = [call_wrap(ConcatVecEnv, env_fns) for env_fns in env_cpu_div]
            return ProcConcatVec(cat_env_fns, obs_space, act_space, num_fns * envs_per_env, example_env.metadata)

        return constructor
