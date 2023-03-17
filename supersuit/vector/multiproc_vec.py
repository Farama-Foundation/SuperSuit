import copy
import multiprocessing as mp
import time
import traceback

import gymnasium.vector
import numpy as np
from gymnasium.vector.utils import (
    concatenate,
    create_empty_array,
    create_shared_memory,
    iterate,
    read_from_shared_memory,
    write_to_shared_memory,
)

from .utils.shared_array import SharedArray


def compress_info(infos):
    non_empty_infs = [(i, info) for i, info in enumerate(infos) if info]
    return non_empty_infs


def decompress_info(num_envs, idx_starts, comp_infos):
    all_info = [{}] * num_envs
    for idx_start, comp_infos in zip(idx_starts, comp_infos):
        for i, info in comp_infos:
            all_info[idx_start + i] = info
    return all_info


def write_observations(vec_env, env_start_idx, shared_obs, obs):
    obs = list(iterate(vec_env.observation_space, obs))
    for i in range(vec_env.num_envs):
        write_to_shared_memory(
            vec_env.observation_space,
            env_start_idx + i,
            obs[i],
            shared_obs,
        )


def numpy_deepcopy(buf):
    if isinstance(buf, dict):
        return {name: numpy_deepcopy(v) for name, v in buf.items()}
    elif isinstance(buf, tuple):
        return tuple(numpy_deepcopy(v) for v in buf)
    elif isinstance(buf, np.ndarray):
        return buf.copy()
    else:
        raise ValueError("numpy_deepcopy ")


def async_loop(
    vec_env_constr, inpt_p, pipe, shared_obs, shared_rews, shared_terms, shared_truncs
):
    inpt_p.close()
    try:
        vec_env = vec_env_constr()

        pipe.send(vec_env.num_envs)
        env_start_idx = pipe.recv()
        env_end_idx = env_start_idx + vec_env.num_envs
        while True:
            instr = pipe.recv()
            comp_infos = []

            if instr == "close":
                vec_env.close()
                return

            elif isinstance(instr, tuple):
                name, data = instr

                if name == "reset":
                    observations = vec_env.reset(seed=data[0], options=data[1])

                    write_observations(vec_env, env_start_idx, shared_obs, observations)
                    shared_terms.np_arr[env_start_idx:env_end_idx] = False
                    shared_truncs.np_arr[env_start_idx:env_end_idx] = False
                    shared_rews.np_arr[env_start_idx:env_end_idx] = 0.0

                elif name == "step":
                    actions = data
                    actions = concatenate(
                        vec_env.action_space,
                        actions,
                        create_empty_array(vec_env.action_space, n=len(actions)),
                    )
                    observations, rewards, terms, truncs, infos = vec_env.step(actions)
                    write_observations(vec_env, env_start_idx, shared_obs, observations)
                    shared_terms.np_arr[env_start_idx:env_end_idx] = terms
                    shared_truncs.np_arr[env_start_idx:env_end_idx] = truncs
                    shared_rews.np_arr[env_start_idx:env_end_idx] = rewards
                    comp_infos = compress_info(infos)

                elif name == "env_is_wrapped":
                    comp_infos = vec_env.env_is_wrapped(data)

                elif name == "render":
                    render_result = vec_env.render(data)
                    if data == "rgb_array":
                        comp_infos = render_result

                else:
                    raise AssertionError("bad tuple instruction name: " + name)
            else:
                raise AssertionError("bad instruction: " + instr)
            pipe.send(comp_infos)
    except BaseException as e:
        tb = traceback.format_exc()
        pipe.send((e, tb))


class ProcConcatVec(gymnasium.vector.VectorEnv):
    def __init__(
        self,
        vec_env_constrs,
        observation_space,
        action_space,
        tot_num_envs,
        metadata,
        graceful_shutdown_timeout=None,
    ):
        raise NotImplementedError(
            "The wrapper ProcConcatVec is temporarily depreciated whilst it is being debugged. "
            "Please refer to https://github.com/Farama-Foundation/SuperSuit/pull/165 for more information, or to contact the devs in regard to this."
        )
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_envs = num_envs = tot_num_envs
        self.metadata = metadata
        self.graceful_shutdown_timeout = graceful_shutdown_timeout

        self.shared_obs = create_shared_memory(self.observation_space, n=self.num_envs)
        self.shared_act = create_shared_memory(self.action_space, n=self.num_envs)
        self.shared_rews = SharedArray((num_envs,), dtype=np.float32)
        self.shared_terms = SharedArray((num_envs,), dtype=np.uint8)
        self.shared_truncs = SharedArray((num_envs,), dtype=np.uint8)

        self.observations_buffers = read_from_shared_memory(
            self.observation_space, self.shared_obs, n=self.num_envs
        )

        pipes = []
        procs = []
        for constr in vec_env_constrs:
            inpt, outpt = mp.Pipe()
            constr = gymnasium.vector.async_vector_env.CloudpickleWrapper(constr)
            proc = mp.Process(
                target=async_loop,
                args=(
                    constr,
                    inpt,
                    outpt,
                    self.shared_obs,
                    self.shared_rews,
                    self.shared_terms,
                    self.shared_truncs,
                ),
            )
            proc.start()
            outpt.close()
            pipes.append(inpt)
            procs.append(proc)

        self.pipes = pipes
        self.procs = procs

        num_envs = 0
        env_nums = self._receive_info()
        idx_starts = []
        for pipe, cnum_env in zip(self.pipes, env_nums):
            cur_env_idx = num_envs
            num_envs += cnum_env
            pipe.send(cur_env_idx)
            idx_starts.append(cur_env_idx)
        idx_starts.append(num_envs)

        assert num_envs == tot_num_envs
        self.idx_starts = idx_starts

    def reset(self, seed=None, options=None):
        for i, pipe in enumerate(self.pipes):
            if seed is not None:
                pipe.send(("reset", (seed + i, options)))
            else:
                pipe.send(("reset", (seed, options)))

        self._receive_info()

        return numpy_deepcopy(self.observations_buffers)

    def step_async(self, actions):
        actions = list(iterate(self.action_space, actions))
        for i, pipe in enumerate(self.pipes):
            start, end = self.idx_starts[i : i + 2]
            pipe.send(("step", actions[start:end]))

    def _receive_info(self):
        all_data = []
        for cin in self.pipes:
            data = cin.recv()
            if isinstance(data, tuple):
                e, tb = data
                print(tb)
                raise e
            all_data.append(data)
        return all_data

    def step_wait(self):
        compressed_infos = self._receive_info()
        infos = decompress_info(self.num_envs, self.idx_starts, compressed_infos)
        rewards = self.shared_rews.np_arr
        terms = self.shared_terms.np_arr
        truncs = self.shared_truncs.np_arr
        return (
            numpy_deepcopy(self.observations_buffers),
            rewards.copy(),
            terms.astype(bool).copy(),
            truncs.astype(bool).copy(),
            copy.deepcopy(infos),
        )

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def render(self):
        self.pipes[0].send("render")
        render_result = self.pipes[0].recv()

        if isinstance(render_result, tuple):
            e, tb = render_result
            print(tb)
            raise e

        return render_result

    def close(self):
        try:
            for pipe, proc in zip(self.pipes, self.procs):
                if proc.is_alive():
                    pipe.send("close")
        except OSError:
            pass
        else:
            deadline = (
                None
                if self.graceful_shutdown_timeout is None
                else time.monotonic() + self.graceful_shutdown_timeout
            )
            for proc in self.procs:
                timeout = None if deadline is None else deadline - time.monotonic()
                if timeout is not None and timeout <= 0:
                    break
                proc.join(timeout)
        for pipe, proc in zip(self.pipes, self.procs):
            if proc.is_alive():
                proc.kill()
            pipe.close()

    def env_is_wrapped(self, wrapper_class, indices=None):
        for i, pipe in enumerate(self.pipes):
            pipe.send(("env_is_wrapped", wrapper_class))

        results = self._receive_info()
        return sum(results, [])
