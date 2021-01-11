from supersuit.aec_vector import VectorAECWrapper,ProcVectorEnv
from pettingzoo.classic import rps_v1
from pettingzoo.classic import mahjong_v2, hanabi_v3
from pettingzoo.butterfly import knights_archers_zombies_v5
from pettingzoo.mpe import simple_world_comm_v2
from pettingzoo.classic import chess_v0
from pettingzoo.sisl import multiwalker_v6
from pettingzoo.atari import warlords_v2
import numpy as np
import random
import time
import supersuit

def test_gym_env():
    import gym
    env = lambda: gym.make("SpaceInvaders-v0")
    env1 = gym.vector.Async([env]*3)
    env1.reset()
    env1.step([0,0,0])

def mahjong_maker():
    env = mahjong_v2.env()
    env = supersuit.observation_lambda_v0(env, lambda obs: obs['observation'], lambda obs_space: obs_space['observation'])
    return env

def hanabi_maker():
    env = hanabi_v3.env()
    env = supersuit.observation_lambda_v0(env, lambda obs: obs['observation'], lambda obs_space: obs_space['observation'])
    return env


#test_gym_env()

NUM_ENVS = 5
NUM_CPUS = 2
def test_vec_env(vec_env):
    vec_env.reset()
    obs, rew, agent_done, env_done, agent_passes, infos = vec_env.last()
    print(np.asarray(obs).shape)
    assert len(obs) == NUM_ENVS
    act_space = vec_env.action_spaces[vec_env.agent_selection]
    assert np.all(np.equal(obs, vec_env.observe(vec_env.agent_selection)))
    assert len(vec_env.observe(vec_env.agent_selection)) == NUM_ENVS
    vec_env.step([act_space.sample() for _ in range(NUM_ENVS)])
    obs, rew, agent_done, env_done, agent_passes, infos = vec_env.last(observe=False)
    assert obs is None

def test_infos(vec_env):
    vec_env.reset()
    infos = vec_env.infos[vec_env.agent_selection]
    assert infos[1]['legal_moves']

def test_some_done(vec_env):
    vec_env.reset()
    act_space = vec_env.action_spaces[vec_env.agent_selection]
    assert not any(done for dones in vec_env.dones.values() for done in dones)
    vec_env.step([act_space.sample() for _ in range(NUM_ENVS)])
    assert any(done for dones in vec_env.dones.values() for done in dones)
    assert any(rew != 0 for rews in vec_env.rewards.values() for rew in rews)

def select_action(vec_env,passes,i):
    my_info = vec_env.infos[vec_env.agent_selection][i]
    if False and not passes[i] and 'legal_moves' in my_info:
        return random.choice(my_info['legal_moves'])
    else:
        act_space = vec_env.action_spaces[vec_env.agent_selection]
        return act_space.sample()

def test_performance(vec_env):
    print("perf?")
    start = time.time()
    vec_env.reset()
    for x in range(2000):
        obs, rew, agent_done, env_done, agent_passes, infos = vec_env.last(observe=False)
        vec_env.step([select_action(vec_env,agent_passes,i) for i in range(vec_env.num_envs)])
        #vec_env.reset()
    end = time.time()
    print(end-start)
    return end - start

test_vec_env(VectorAECWrapper([rps_v1.env]*NUM_ENVS))
test_vec_env(VectorAECWrapper([lambda :mahjong_maker() for i in range(NUM_ENVS)]))
test_infos(VectorAECWrapper([hanabi_maker]*NUM_ENVS))
test_some_done(VectorAECWrapper([mahjong_maker]*NUM_ENVS))
test_vec_env(VectorAECWrapper([multiwalker_v6.env]*NUM_ENVS))
test_vec_env(VectorAECWrapper([simple_world_comm_v2.env]*NUM_ENVS))

test_vec_env(ProcVectorEnv([rps_v1.env]*NUM_ENVS, NUM_CPUS))
test_vec_env(ProcVectorEnv([lambda : mahjong_maker() for i in range(NUM_ENVS)], NUM_CPUS))
test_infos(ProcVectorEnv([hanabi_maker]*NUM_ENVS, NUM_CPUS))
test_some_done(ProcVectorEnv([mahjong_maker]*NUM_ENVS, NUM_CPUS))
test_vec_env(ProcVectorEnv([multiwalker_v6.env]*NUM_ENVS, NUM_CPUS))
test_vec_env(ProcVectorEnv([simple_world_comm_v2.env]*NUM_ENVS, NUM_CPUS))

PERF_NUM_ENVS = 12*4
PERF_NUM_CPUS = 24
print("warlords_v2")
print(test_performance(ProcVectorEnv([warlords_v2.env]*PERF_NUM_ENVS))
    / test_performance(VectorAECWrapper([warlords_v2.env]*PERF_NUM_ENVS)))
print("simple_world_comm_v2")
print(test_performance(ProcVectorEnv(simple_world_comm_v2.env, PERF_NUM_ENVS, PERF_NUM_CPUS))
    / test_performance(VectorAECWrapper(simple_world_comm_v2.env, PERF_NUM_ENVS)))
# print("mahjong_v2")
# print(test_performance(ProcVectorEnv(mahjong_maker, NUM_ENVS, NUM_CPUS))
#     - test_performance(VectorAECWrapper(mahjong_maker, NUM_ENVS)))
print("multiwalker_v6")
print(test_performance(ProcVectorEnv(multiwalker_v6.env, PERF_NUM_ENVS, PERF_NUM_CPUS))
    / test_performance(VectorAECWrapper(multiwalker_v6.env, PERF_NUM_ENVS)))
print("rps_v1")
print(test_performance(ProcVectorEnv(rps_v1.env, PERF_NUM_ENVS, PERF_NUM_CPUS))
    / test_performance(VectorAECWrapper(rps_v1.env, PERF_NUM_ENVS)))
print("chess_v0")
print(test_performance(ProcVectorEnv(chess_v0.env, PERF_NUM_ENVS, PERF_NUM_CPUS))
    / test_performance(VectorAECWrapper(chess_v0.env, PERF_NUM_ENVS)))
