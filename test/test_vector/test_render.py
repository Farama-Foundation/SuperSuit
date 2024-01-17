from pettingzoo.butterfly import pistonball_v6
import supersuit as ss
from stable_baselines3.common.vec_env import VecVideoRecorder


def schedule(episode_idx):
    print(episode_idx)
    return episode_idx <= 1


def make_record_env():
    env = pistonball_v6.parallel_env(render_mode="rgb_array")
    print(env.render_mode)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    envs = ss.concat_vec_envs_v1(env, 1, num_cpus=0, base_class="stable_baselines3")
    # envs.render_mode = "rgb_array"
    envs = VecVideoRecorder(envs, f".", schedule)
    return envs


def record_video_test():
    envs = make_record_env()
    envs.reset()
    for _ in range(100):
        envs.step([envs.action_space.sample() for _ in range(envs.num_envs)])
    envs.close()


record_video_test()


# def make_env():
#     env = pistonball_v6.parallel_env(render_mode="rgb_array")
#     env = ss.pettingzoo_env_to_vec_env_v1(env)
#     return env


# unfortunately this test does not pass
# def test_vector_render_multiproc():
#     env = make_env()
#     num_envs = 3
#     venv = ss.concat_vec_envs_v1(
#         env, num_envs, num_cpus=num_envs, base_class="stable_baselines3"
#     )
#     venv.reset()
#     arr = venv.render()
#     venv.reset()
#     assert len(arr.shape) == 3 and arr.shape[2] == 3
#     venv.reset()
#     try:
#         venv.close()
#     except RuntimeError:
#         pass
