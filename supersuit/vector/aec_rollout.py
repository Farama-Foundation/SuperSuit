import numpy as np
import random

class RolloutBuilder:
    def __init__(self, vec_env):
        self.vec_env = vec_env
        self.num_envs = self.vec_env.num_envs
        self.states = None
        self.prev_observes = {agent:None for agent in vec_env.agents}
        self.obs_buffer = {agent:None for agent in vec_env.agents}
        self.prev_infos = {agent:None for agent in vec_env.agents}

    def restart(self, policies):
        observations = self.vec_env.reset()
        self.prev_observes[agent] = observations
        self.prev_infos[agent] = self.vec_env.infos[agent]
        self.states = {agent: policies[agent].start_state() for agent in self.vec_env.agents}

    def rollout(self, policy, state, n_steps, deterministic=False):
        assert self.prev_observes is not None, "must call restart()  before rollout()"
        num_envs = self.vec_env.num_envs
        rews = np.empty((n_steps,self.num_envs),dtype=np.float64)
        dones = np.empty((n_steps,self.num_envs),dtype=np.bool)
        infos = []
        for x in range(n_steps):
            actions,states = policy.rollout_step(self.prev_observes, self.prev_infos, self.states)
            print(actions)
            obs, rew, done, info = self.vec_env.step(actions)

            if self.obs_buffer is None or len(self.obs_buffer) != n_steps:
                # cache observation buffer between rollout so it doesn't have to always reallocate
                self.obs_buffer = np.empty((n_steps,self.num_envs)+obs.shape,dtype=obs.dtype)

            self.obs_buffer[x] = obs
            rews[x] = rew
            dones[x] = done
            infos.append(info)

            self.states = states
            self.prev_observes = obs
            self.prev_infos = info

        return self.obs_buffer, rews, dones, infos

def transpose_rollout(obss, rews, dones, infos):
    obss = np.asarray(obss)
    obss = obss.transpose((1,0)+tuple(range(2,len(obss.shape))))
    rews = np.asarray(rews,dtype=np.float64).T
    dones = np.asarray(dones,dtype=np.bool).T
    infos = [[infos[i][j] for i in range(len(infos))] for j in range(len(infos[0]))]
    return obss, rews, dones, infos

def split_rollouts_on_dones(batch_obs,batch_rews,batch_dones, batch_infos):
    assert len(batch_obs) == len(batch_rews) == len(batch_dones) == len(batch_infos)
    assert len(batch_obs) > 0
    n_steps = len(batch_obs[0])
    assert n_steps > 0

    result_ranges = []
    for i in range(len(batch_obs)):
        dones = np.asarray(batch_dones[i])
        sidx = 0
        fidx = np.argmax(dones)
        while dones[fidx]:
            if sidx != fidx:
                result_ranges.append((i,(sidx,fidx)))
            dones[fidx] = False
            fidx = np.argmax(dones)

        result_ranges.append((i,(sidx,n_steps)))

    observes = [batch_obs[i,s:f] for i, (s,f) in result_ranges]
    rewards = [batch_rews[i,s:f] for i, (s,f) in result_ranges]
    infos = [batch_infos[i][s:f] for i, (s,f) in result_ranges]

    return observes,rewards,infos
