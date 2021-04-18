from pettingzoo.utils.conversions import ParallelEnv
import gym
from gym.spaces import Box, Discrete
from .utils.frame_stack import stack_obs_space, stack_init, stack_obs
from .utils.frame_skip import check_transform_frameskip
from .utils.obs_delay import Delayer


class ParallelWraper(ParallelEnv):
    def __init__(self, env):
        self.env = env
        self.observation_spaces = env.observation_spaces
        self.action_spaces = env.action_spaces
        self.metadata = env.metadata
        self.possible_agents = env.possible_agents

    def reset(self):
        res = self.env.reset()
        self.agents = self.env.agents
        return res

    def step(self, actions):
        res = self.env.step(actions)
        self.agents = self.env.agents
        return res

    def render(self, mode="human"):
        return self.env.render(mode)

    def close(self):
        return self.env.close()


class ObservationWrapper(ParallelWraper):
    def reset(self):
        obss = super().reset()
        return {agent: self._modify_observation(agent, obs) for agent, obs in obss.items()}

    def step(self, actions):
        obss, rew, done, info = super().step(actions)
        obss = {agent: self._modify_observation(agent, obs) for agent, obs in obss.items()}
        return obss, rew, done, info


class frame_stack(ObservationWrapper):
    def __init__(self, env, num_frames=4):
        self.stack_size = num_frames
        super().__init__(env)
        self._check_wrapper_params()
        self.observation_spaces = {agent: stack_obs_space(space, self.stack_size) for agent, space in self.observation_spaces.items()}

    def _check_wrapper_params(self):
        assert isinstance(self.stack_size, int), "stack size of frame_stack must be an int"
        for space in self.observation_spaces.values():
            if isinstance(space, Box):
                assert 1 <= len(space.shape) <= 3, "frame_stack only works for 1,2 or 3 dimensional observations"
            elif isinstance(space, Discrete):
                pass
            else:
                assert False, "Stacking is currently only allowed for Box and Discrete observation spaces. The given observation space is {}".format(space)

    def reset(self):
        self.stack = {agent: stack_init(space, self.stack_size) for agent, space in self.env.observation_spaces.items()}
        return super().reset()

    def _modify_observation(self, agent, observation):
        space = self.env.observation_spaces[agent]
        self.stack[agent] = stack_obs(self.stack[agent], observation, space, self.stack_size)
        return self.stack[agent]


class delay_observations(ObservationWrapper):
    def __init__(self, env, delay):
        super().__init__(env)
        self.delay = delay

    def _modify_observation(self, agent, obs):
        return self.delayers[agent].add(obs)

    def reset(self):
        self.delayers = {agent: Delayer(space, self.delay) for agent, space in self.observation_spaces.items()}
        return super().reset()


class frame_skip(ParallelWraper):
    def __init__(self, env, num_frames):
        super().__init__(env)
        self.num_frames = check_transform_frameskip(num_frames)
        self.np_random, seed = gym.utils.seeding.np_random(None)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        super().seed(seed)

    def step(self, action):
        low, high = self.num_frames
        num_skips = int(self.np_random.randint(low, high + 1))

        for x in range(num_skips):
            obs, rews, done, info = super().step(action)
            if x == 0:
                next_agents = self.env.agents[:]
                total_reward = {agent: 0.0 for agent in self.env.agents}
                total_dones = {}
                total_infos = {}
                total_obs = {}

            for agent, rew in rews.items():
                total_reward[agent] += rew
                total_dones[agent] = done[agent]
                total_infos[agent] = info[agent]
                total_obs[agent] = obs[agent]
            if all(done.values()):
                break
        self.agents = next_agents
        return total_obs, total_reward, total_dones, total_infos
