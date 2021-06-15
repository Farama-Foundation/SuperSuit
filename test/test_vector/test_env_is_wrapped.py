from supersuit import concat_vec_envs_v0, pettingzoo_env_to_vec_env_v0
from supersuit.gym_wrappers import frame_skip_gym
import gym
from pettingzoo.mpe import simple_spread_v2


def test_env_is_wrapped_true():
    env = gym.make("Pendulum-v0")
    env = frame_skip_gym(env)
    num_envs = 3
    venv1 = concat_vec_envs_v0(env, num_envs)
    assert venv1.env_is_wrapped(frame_skip_gym) == [True] * 3


def test_env_is_wrapped_false():
    env = gym.make("Pendulum-v0")
    num_envs = 3
    venv1 = concat_vec_envs_v0(env, num_envs)
    assert venv1.env_is_wrapped(frame_skip_gym) == [False] * 3


def test_env_is_wrapped_cpu():
    env = gym.make("Pendulum-v0")
    env = frame_skip_gym(env)
    num_envs = 3
    venv1 = concat_vec_envs_v0(env, num_envs, num_cpus=2)
    assert venv1.env_is_wrapped(frame_skip_gym) == [True] * 3


def test_env_is_wrapped_pettingzoo():
    env = simple_spread_v2.parallel_env()
    venv1 = pettingzoo_env_to_vec_env_v0(env)
    num_envs = 3
    venv1 = concat_vec_envs_v0(venv1, num_envs)
    assert venv1.env_is_wrapped(frame_skip_gym) == [False] * 9
