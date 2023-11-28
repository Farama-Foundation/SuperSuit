import gymnasium
import numpy as np
from pettingzoo.utils.wrappers import BaseParallelWrapper, BaseWrapper

from supersuit.utils.frame_skip import check_transform_frameskip
from supersuit.utils.make_defaultdict import make_defaultdict
from supersuit.utils.wrapper_chooser import WrapperChooser


class frame_skip_gym(gymnasium.Wrapper):
    def __init__(self, env, num_frames):
        super().__init__(env)
        self.num_frames = check_transform_frameskip(num_frames)

    def step(self, action):
        low, high = self.num_frames
        num_skips = int(self.env.unwrapped.np_random.integers(low, high + 1))
        total_reward = 0.0

        for x in range(num_skips):
            obs, rew, term, trunc, info = super().step(action)
            total_reward += rew
            if term or trunc:
                break

        return obs, total_reward, term, trunc, info


class StepAltWrapper(BaseWrapper):
    def _modify_action(self, agent, action):
        return action

    def _modify_observation(self, agent, observation):
        return observation


class frame_skip_aec(StepAltWrapper):
    def __init__(self, env, num_frames):
        super().__init__(env)
        assert isinstance(
            num_frames, int
        ), "multi-agent frame skip only takes in an integer"
        assert num_frames > 0
        check_transform_frameskip(num_frames)
        self.num_frames = num_frames

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.agents = self.env.agents[:]
        self.terminations = make_defaultdict({agent: False for agent in self.agents})
        self.truncations = make_defaultdict({agent: False for agent in self.agents})
        self.rewards = make_defaultdict({agent: 0.0 for agent in self.agents})
        self._cumulative_rewards = make_defaultdict(
            {agent: 0.0 for agent in self.agents}
        )
        self.infos = make_defaultdict({agent: {} for agent in self.agents})
        self.skip_num = make_defaultdict({agent: 0 for agent in self.agents})
        self.old_actions = make_defaultdict({agent: None for agent in self.agents})
        self._final_observations = make_defaultdict(
            {agent: None for agent in self.agents}
        )

    def observe(self, agent):
        fin_observe = self._final_observations[agent]
        return fin_observe if fin_observe is not None else super().observe(agent)

    def step(self, action):
        self._has_updated = True

        # if agent is dead, perform a None step
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            if self.env.agents and self.agent_selection == self.env.agent_selection:
                self.env.step(None)
            self._was_dead_step(action)
            return

        cur_agent = self.agent_selection
        self._cumulative_rewards[cur_agent] = 0
        self.rewards = make_defaultdict({a: 0.0 for a in self.agents})
        self.skip_num[
            cur_agent
        ] = (
            self.num_frames
        )  # set the skip num to the param inputted in the frame_skip wrapper
        self.old_actions[cur_agent] = action

        while (
            self.old_actions[self.env.agent_selection] is not None
        ):  # this is like `for x in range(num_skips):` (L18)
            step_agent = self.env.agent_selection

            # if agent is dead, perform a None step
            if (step_agent in self.env.terminations) or (
                step_agent in self.env.truncations
            ):
                # reward = self.env.rewards[step_agent]
                # done = self.env.dones[step_agent]
                # info = self.env.infos[step_agent]
                observe, reward, termination, truncation, info = self.env.last(
                    observe=False
                )
                action = self.old_actions[step_agent]
                self.env.step(action)
                for agent in self.env.agents:
                    self.rewards[agent] += self.env.rewards[agent]
                self.infos[self.env.agent_selection] = info

                while self.env.agents and (
                    self.env.terminations[self.env.agent_selection]
                    or self.env.truncations[self.env.agent_selection]
                ):
                    dead_agent = self.env.agent_selection
                    self.terminations[dead_agent] = self.env.terminations[dead_agent]
                    self.truncations[dead_agent] = self.env.truncations[dead_agent]
                    self._final_observations[dead_agent] = self.env.observe(dead_agent)
                    self.env.step(None)
                step_agent = self.env.agent_selection

            self.skip_num[step_agent] -= 1
            if self.skip_num[step_agent] == 0:
                self.old_actions[
                    step_agent
                ] = None  # if it is time to skip, set action to None, effectively breaking the while loop

        my_agent_set = set(self.agents)
        for agent in self.env.agents:
            self.terminations[agent] = self.env.terminations[agent]
            self.truncations[agent] = self.env.truncations[agent]
            self.infos[agent] = self.env.infos[agent]
            if agent not in my_agent_set:
                self.agents.append(agent)
        self.agent_selection = self.env.agent_selection
        self._accumulate_rewards()
        self._deads_step_first()


class frame_skip_par(BaseParallelWrapper):
    def __init__(self, env, num_frames, default_action=None):
        super().__init__(env)
        self.num_frames = check_transform_frameskip(num_frames)
        self.default_action = default_action

    def step(self, action):
        action = {**action}
        low, high = self.num_frames
        num_skips = int(self.env.unwrapped.np_random.integers(low, high + 1))
        orig_agents = set(action.keys())

        total_reward = make_defaultdict({agent: 0.0 for agent in self.agents})
        total_terminations = {}
        total_truncations = {}
        total_infos = {}
        total_obs = {}

        for x in range(num_skips):
            obs, rews, term, trunc, info = super().step(action)

            for agent, rew in rews.items():
                total_reward[agent] += rew
                total_terminations[agent] = term[agent]
                total_truncations[agent] = trunc[agent]
                total_infos[agent] = info[agent]
                total_obs[agent] = obs[agent]

            for agent in self.env.agents:
                if agent not in action:
                    assert (
                        self.default_action is not None
                    ), "parallel environments that use frame_skip_v0 must provide a `default_action` argument for steps between an agent being generated and an agent taking its first step"
                    action[agent] = self.default_action

            if (
                np.fromiter(term.values(), dtype=bool)
                | np.fromiter(trunc.values(), dtype=bool)
            ).all():
                break

        # delete any values created by agents which were
        # generated and deleted before they took any actions
        final_agents = set(self.agents)
        for agent in list(total_reward):
            if agent not in final_agents and agent not in orig_agents:
                del total_reward[agent]
                del total_terminations[agent]
                del total_truncations[agent]
                del total_infos[agent]
                del total_obs[agent]

        return (
            total_obs,
            total_reward,
            total_terminations,
            total_truncations,
            total_infos,
        )


frame_skip_v0 = WrapperChooser(
    aec_wrapper=frame_skip_aec,
    gym_wrapper=frame_skip_gym,
    parallel_wrapper=frame_skip_par,
)
