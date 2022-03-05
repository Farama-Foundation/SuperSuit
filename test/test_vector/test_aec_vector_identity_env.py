from pettingzoo.butterfly import knights_archers_zombies_v9
from pettingzoo.mpe import simple_push_v2
from supersuit import frame_skip_v0, vectorize_aec_env_v0
import numpy as np


def recursive_equal(info1, info2):
    if info1 == info2:
        return True
    return False


def test_identical():
    def env_fn():
        return knights_archers_zombies_v9.env()  # ,20)

    n_envs = 2
    # single threaded
    env1 = vectorize_aec_env_v0(knights_archers_zombies_v9.env(), n_envs)
    env2 = vectorize_aec_env_v0(knights_archers_zombies_v9.env(), n_envs, num_cpus=1)
    env1.seed(42)
    env2.seed(42)
    env1.reset()
    env2.reset()

    def policy(obs, agent):
        return [env1.action_space(agent).sample() for i in range(env1.num_envs)]

    envs_done = 0
    for agent in env1.agent_iter(200000):
        assert env1.agent_selection == env2.agent_selection
        agent = env1.agent_selection
        obs1, rew1, agent_done1, env_done1, agent_passes1, infos1 = env1.last()
        obs2, rew2, agent_done2, env_done2, agent_passes2, infos2 = env2.last()
        assert np.all(np.equal(obs1, obs2))
        assert np.all(np.equal(agent_done1, agent_done2))
        assert np.all(np.equal(agent_passes1, agent_passes2))
        assert np.all(np.equal(env_done1, env_done2))
        assert np.all(np.equal(obs1, obs2))
        assert all(np.all(np.equal(r1, r2)) for r1, r2 in zip(env1.rewards.values(), env2.rewards.values()))
        assert recursive_equal(infos1, infos2)
        actions = policy(obs1, agent)
        env1.step(actions)
        env2.step(actions)
        # env.envs[0].render()
        for j in range(2):
            # if agent_passes[j]:
            #     print("pass")
            if rew1[j] != 0:
                print(j, agent, rew1, agent_done1[j])
            if env_done1[j]:
                print(j, "done")
                envs_done += 1
                if envs_done == n_envs + 1:
                    print("test passed")
                    return
