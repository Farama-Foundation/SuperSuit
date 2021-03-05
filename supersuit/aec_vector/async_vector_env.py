import copy
import multiprocessing as mp
from gym.vector.utils import shared_memory
from pettingzoo.utils.agent_selector import agent_selector
import numpy as np
import ctypes
import gym
from .base_aec_vec_env import VectorAECEnv
import warnings
import signal
import traceback


class SpaceWrapper:
    def __init__(self, space):
        if isinstance(space, gym.spaces.Discrete):
            self.shape = ()
            self.dtype = np.dtype(np.int64)
            self.low = 0
        elif isinstance(space, gym.spaces.Box):
            self.shape = space.shape
            self.dtype = np.dtype(space.dtype)
            self.low = space.low
        else:
            assert False, "ProcVectorEnv only support Box and Discrete types"


class SharedData:
    def __init__(self, array, num_envs, shape, dtype):
        self.array = array
        self.num_envs = num_envs
        self.np_arr = np.frombuffer(array, dtype=dtype).reshape((num_envs,) + shape)


def create_shared_data(num_envs, obs_space, act_space):
    obs_array = mp.Array(np.ctypeslib.as_ctypes_type(obs_space.dtype), int(num_envs * np.prod(obs_space.shape)), lock=False)
    act_array = mp.Array(np.ctypeslib.as_ctypes_type(act_space.dtype), int(num_envs * np.prod(act_space.shape)), lock=False)
    rew_array = mp.Array(ctypes.c_float, num_envs, lock=False)
    cum_rew_array = mp.Array(ctypes.c_float, num_envs, lock=False)
    done_array = mp.Array(ctypes.c_bool, num_envs, lock=False)
    data = (obs_array, act_array, rew_array, cum_rew_array, done_array)
    return data


def create_env_data(num_envs):
    env_dones = mp.Array(ctypes.c_bool, num_envs, lock=False)
    agent_sel_idx = mp.Array(ctypes.c_uint, num_envs, lock=False)
    return env_dones, agent_sel_idx


class AgentSharedData:
    def __init__(self, num_envs, obs_space, act_space, data):
        self.num_envs = num_envs

        obs_array, act_array, rew_array, cum_rew_array, done_array = data
        self.obs = SharedData(obs_array, num_envs, obs_space.shape, obs_space.dtype)
        self.act = SharedData(act_array, num_envs, act_space.shape, act_space.dtype)
        self.rewards = SharedData(rew_array, num_envs, (), np.float32)
        self._cumulative_rewards = SharedData(cum_rew_array, num_envs, (), np.float32)
        self.dones = SharedData(done_array, num_envs, (), np.uint8)


class EnvSharedData:
    def __init__(self, num_envs, data):
        env_dones, agent_idx_array = data
        self.env_dones = SharedData(env_dones, num_envs, (), np.uint8)
        self.agent_sel_idx = SharedData(agent_idx_array, num_envs, (), np.uint32)


class _SeperableAECWrapper:
    def __init__(self, env_constructors, num_envs):
        self.envs = [env_constructor() for env_constructor in env_constructors]
        self.env = self.envs[0]
        self.possible_agents = self.env.possible_agents
        self.agent_indexes = {agent: i for i, agent in enumerate(self.env.possible_agents)}
        self.dead_obss = {agent: np.zeros_like(SpaceWrapper(obs_space).low) for agent, obs_space in self.env.observation_spaces.items()}

    def reset(self):
        for env in self.envs:
            env.reset()

        self.rewards = {agent: [env.rewards.get(agent, 0) for env in self.envs] for agent in self.possible_agents}
        self._cumulative_rewards = {agent: [env._cumulative_rewards.get(agent, 0) for env in self.envs] for agent in self.possible_agents}
        self.dones = {agent: [env.dones.get(agent, True) for env in self.envs] for agent in self.possible_agents}
        self.infos = {agent: [env.infos.get(agent, {}) for env in self.envs] for agent in self.possible_agents}

    def observe(self, agent):
        observations = []
        for env in self.envs:
            observations.append(env.observe(agent) if agent in env.dones else self.dead_obss[agent])
        return observations

    def seed(self, seed=None):
        for i, env in enumerate(self.envs):
            env.seed(seed + i)

    def step(self, agent_step, actions):
        assert len(actions) == len(self.envs)

        env_dones = []
        for act, env in zip(actions, self.envs):
            env_done = not env.agents
            env_dones.append(env_done)
            if env_done:
                env.reset()
            elif env.agent_selection == agent_step:
                if env.dones[agent_step]:
                    act = None
                env.step(act)

        self.rewards = {agent: [env.rewards.get(agent, 0) for env in self.envs] for agent in self.possible_agents}
        self._cumulative_rewards = {agent: [env._cumulative_rewards.get(agent, 0) for env in self.envs] for agent in self.possible_agents}
        self.dones = {agent: [env.dones.get(agent, True) for env in self.envs] for agent in self.possible_agents}
        self.infos = {agent: [env.infos.get(agent, {}) for env in self.envs] for agent in self.possible_agents}

        return env_dones

    def get_agent_indexes(self):
        return [self.agent_indexes[env.agent_selection] for env in self.envs]


def sig_handle(signal_object, argvar):
    raise RuntimeError()


def init_parallel_env():
    signal.signal(signal.SIGINT, sig_handle)
    signal.signal(signal.SIGTERM, sig_handle)


def write_out_data(rewards, cumulative_rews, dones, num_envs, start_index, shared_data):
    for agent in shared_data:
        rews = np.asarray(rewards[agent], dtype=np.float32)
        cum_rews = np.asarray(cumulative_rews[agent], dtype=np.float32)
        dns = np.asarray(dones[agent], dtype=np.uint8)
        cur_data = shared_data[agent]
        cur_data.rewards.np_arr[start_index : start_index + num_envs] = rews
        cur_data._cumulative_rewards.np_arr[start_index : start_index + num_envs] = cum_rews
        cur_data.dones.np_arr[start_index : start_index + num_envs] = dns


def write_env_data(env_dones, indexes, num_envs, start_index, shared_data):
    shared_data.env_dones.np_arr[start_index : start_index + num_envs] = env_dones
    agent_indexes = np.asarray(indexes, dtype=np.uint32)
    shared_data.agent_sel_idx.np_arr[start_index : start_index + num_envs] = agent_indexes


def write_obs(obs, num_env, start_index, shared_data):
    for i, o in enumerate(obs):
        if o is not None:
            shared_data.obs.np_arr[start_index + i] = o


def compress_info(infos):
    all_infos = {}
    for agent, infs in infos.items():
        non_empty_infs = [(i, info) for i, info in enumerate(infs) if info]
        if non_empty_infs:
            all_infos[agent] = non_empty_infs
    return all_infos


def decompress_info(agents, num_envs, idx_starts, comp_infos):
    all_info = {agent: [{}] * num_envs for agent in agents}
    for idx_start, comp_info in zip(idx_starts, comp_infos):
        for agent, inf_data in comp_info.items():
            agent_info = all_info[agent]
            for i, info in inf_data:
                agent_info[idx_start + i] = info
    return all_info


def env_worker(env_constructors, total_num_envs, idx_start, my_num_envs, agent_arrays, env_arrays, pipe):
    try:
        env = _SeperableAECWrapper(env_constructors, my_num_envs)
        shared_datas = {
            agent: AgentSharedData(
                total_num_envs, SpaceWrapper(env.env.observation_spaces[agent]), SpaceWrapper(env.env.action_spaces[agent]), agent_arrays[agent]
            )
            for agent in env.possible_agents
        }

        env_datas = EnvSharedData(total_num_envs, env_arrays)

        while True:
            instruction, data = pipe.recv()
            if instruction == "reset":
                env.reset()
                write_out_data(env.rewards, env._cumulative_rewards, env.dones, my_num_envs, idx_start, shared_datas)
                env_dones = np.zeros(my_num_envs, dtype=np.uint8)
                write_env_data(env_dones, env.get_agent_indexes(), my_num_envs, idx_start, env_datas)

                pipe.send(compress_info(env.infos))

            elif instruction == "observe":
                agent_observe = data
                obs = env.observe(agent_observe)
                write_obs(obs, my_num_envs, idx_start, shared_datas[agent_observe])

                pipe.send(True)
            elif instruction == "step":
                step_agent, do_observe = data

                actions = shared_datas[step_agent].act.np_arr[idx_start : idx_start + my_num_envs]

                env_dones = env.step(step_agent, actions)
                write_out_data(env.rewards, env._cumulative_rewards, env.dones, my_num_envs, idx_start, shared_datas)
                write_env_data(env_dones, env.get_agent_indexes(), my_num_envs, idx_start, env_datas)

                pipe.send(compress_info(env.infos))
            elif instruction == "seed":
                env.seed(data)
                pipe.send(True)
            elif instruction == "terminate":
                return
            else:
                assert False, "Bad instruction sent to ProcVectorEnv worker"
    except Exception as e:
        tb = traceback.format_exc()
        pipe.send((e, tb))


class AsyncAECVectorEnv(VectorAECEnv):
    def __init__(self, env_constructors, num_cpus=None, return_copy=True):
        # set signaling so that crashing is handled gracefully
        init_parallel_env()

        num_envs = len(env_constructors)

        if num_cpus is None:
            num_cpus = mp.cpu_count()

        num_cpus = min(num_cpus, num_envs)
        assert num_envs > 0

        assert num_envs >= 1
        assert callable(env_constructors[0]), "env_constructor must be a callable object (i.e function) that create an environment"
        # self.envs = [env_constructor() for _ in range(num_envs)]
        self.env = env = env_constructors[0]()
        self.max_num_agents = len(self.env.possible_agents)
        self.possible_agents = self.env.possible_agents
        self.observation_spaces = copy.copy(self.env.observation_spaces)
        self.action_spaces = copy.copy(self.env.action_spaces)
        self.order_is_nondeterministic = False
        self.num_envs = num_envs

        self.agent_indexes = {agent: i for i, agent in enumerate(self.env.possible_agents)}

        self._agent_selector = agent_selector(self.possible_agents)

        all_arrays = {
            agent: create_shared_data(
                num_envs,
                SpaceWrapper(self.observation_spaces[agent]),
                SpaceWrapper(self.action_spaces[agent]),
            )
            for agent in self.possible_agents
        }

        self.shared_datas = {
            agent: AgentSharedData(num_envs, SpaceWrapper(env.observation_spaces[agent]), SpaceWrapper(env.action_spaces[agent]), all_arrays[agent])
            for agent in env.possible_agents
        }

        env_arrays = create_env_data(num_envs)

        self.env_datas = EnvSharedData(num_envs, env_arrays)
        self.return_copy = return_copy

        self.procs = []
        self.pipes = [mp.Pipe() for _ in range(num_cpus)]
        self.con_ins = [con_in for con_in, con_out in self.pipes]
        self.con_outs = [con_out for con_in, con_out in self.pipes]
        self.env_starts = []
        env_counter = 0
        for pidx in range(num_cpus):
            envs_left = num_envs - env_counter
            allocated_envs = min(envs_left, (num_envs + num_cpus - 1) // num_cpus)
            proc_constructors = env_constructors[env_counter : env_counter + allocated_envs]
            proc = mp.Process(
                target=env_worker, args=(proc_constructors, num_envs, env_counter, allocated_envs, all_arrays, env_arrays, self.con_outs[pidx])
            )
            self.procs.append(proc)
            self.env_starts.append(env_counter)

            proc.start()
            env_counter += allocated_envs

    def _find_active_agent(self):
        cur_selection = self.agent_selection
        while not np.any(np.equal(self.agent_indexes[cur_selection], self.env_datas.agent_sel_idx.np_arr)):
            cur_selection = self._agent_selector.next()
        return cur_selection

    def _receive_info(self):
        all_data = []
        for cin in self.con_ins:
            data = cin.recv()
            if isinstance(data, tuple):
                err, tb = data
                print(tb)
                raise RuntimeError("ProcVectorEnv received error from subprocess (look up there ^^^ for error)")
            all_data.append(data)
        return all_data

    def copy(self, data):
        if self.return_copy:
            return np.copy(data)
        else:
            return data

    def _load_next_data(self, reset):
        all_compressed_info = self._receive_info()

        all_info = decompress_info(self.possible_agents, self.num_envs, self.env_starts, all_compressed_info)

        self.agent_selection = self._agent_selector.reset() if reset else self._agent_selector.next()
        self.agent_selection = self._find_active_agent()

        passes = np.not_equal(self.env_datas.agent_sel_idx.np_arr, self.agent_indexes[self.agent_selection])

        assert not np.all(passes), "something went wrong with finding agent"
        if np.any(passes) or self.order_is_nondeterministic:
            warnings.warn("The agent order of sub-environments of ProcVectorEnv differs, likely due to agent death. The ProcVectorEnv only returns one agent at a time, so it will now 'pass' environments where the current agent is not active, taking up to O(n) more time")
            self.order_is_nondeterministic = True

        self.dones = {agent: self.copy(self.shared_datas[agent].dones.np_arr) for agent in self.possible_agents}
        self.rewards = {agent: self.copy(self.shared_datas[agent].rewards.np_arr) for agent in self.possible_agents}
        self._cumulative_rewards = {agent: self.copy(self.shared_datas[agent]._cumulative_rewards.np_arr) for agent in self.possible_agents}
        self.infos = all_info
        env_dones = self.env_datas.env_dones.np_arr
        self.passes = self.copy(passes)
        self.env_dones = self.copy(env_dones)

    def last(self, observe=True):
        last_agent = self.agent_selection
        obs = self.observe(last_agent) if observe else None
        return obs, self._cumulative_rewards[last_agent], self.dones[last_agent], self.env_dones, self.passes, self.infos[last_agent]

    def reset(self, observe=True):
        for cin in self.con_ins:
            cin.send(("reset", observe))

        self._load_next_data(True)

    def step(self, actions, observe=True):
        step_agent = self.agent_selection

        self.shared_datas[self.agent_selection].act.np_arr[:] = actions
        for cin in self.con_ins:
            cin.send(("step", (step_agent, observe)))

        self._load_next_data(False)

    def observe(self, agent):
        for cin in self.con_ins:
            cin.send(("observe", agent))

        # wait until all are finished
        self._receive_info()

        obs = self.copy(self.shared_datas[self.agent_selection].obs.np_arr)
        return obs

    def seed(self, seed):
        for start, cin in zip(self.env_starts, self.con_ins):
            cin.send(("seed", seed + start if seed is not None else None))

        self._receive_info()

    def __del__(self):
        for cin in self.con_ins:
            cin.send(("terminate", None))
        for proc in self.procs:
            proc.join()
