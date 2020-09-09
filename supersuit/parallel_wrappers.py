from pettingzoo.utils.to_parallel import ParallelEnv
import gym
from gym.spaces import Box,Space,Discrete
from .adv_transforms.frame_stack import stack_obs_space,stack_init,stack_obs
from .adv_transforms.frame_skip import check_transform_frameskip
from .adv_transforms.obs_delay import Delayer
import numpy as np

class ParallelWraper(ParallelEnv):
    def __init__(self, env):
        self.env = env
        self.observation_spaces = env.observation_spaces
        self.action_spaces = env.action_spaces
        self.agents = env.agents
        self.num_agents = env.num_agents

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        return self.env.step(actions)

    def render(self, mode="human"):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

class ObservationWrapper(ParallelWraper):
    def reset(self):
        obss = self.env.reset()
        return {agent: self._modify_observation(agent, obs) for agent, obs in obss.items()}

    def step(self, actions):
        obss, rew, done, info = self.env.step(actions)
        obss = {agent: self._modify_observation(agent, obs) for agent, obs in obss.items()}
        return obss, rew, done, info

class frame_stack(ObservationWrapper):
    def __init__(self,env,num_frames=4):
        self.stack_size = num_frames
        super().__init__(env)
        self._check_wrapper_params()
        self.observation_spaces = {agent: stack_obs_space(space, self.stack_size) for agent, space in self.observation_spaces.items()}

    def _check_wrapper_params(self):
        assert isinstance(self.stack_size, int), "stack size of frame_stack must be an int"
        for space in self.observation_spaces.values():
            if isinstance(space, Box):
                assert 1 <= len(space.shape) <= 3, "frame_stack only works for 1,2 or 3 dimentional observations"
            elif isinstance(space, Discrete):
                pass
            else:
                assert False, "Stacking is currently only allowed for Box and Discrete observation spaces. The given observation space is {}".format(obs_space)

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
    def __init__(self, env, frame_skip):
        super().__init__(env)
        self.frame_skip = check_transform_frameskip(frame_skip)
        self.np_random, seed = gym.utils.seeding.np_random(None)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        super().seed(seed)

    def step(self, action):
        low, high = self.frame_skip
        num_skips = int(self.np_random.randint(low, high+1))
        total_reward = {agent: 0. for agent in self.agents}

        for x in range(num_skips):
            obs, rews, done, info = super().step(action)
            for agent, rew in rews.items():
                total_reward[agent] += rew
            if all(done.values()):
                break

        return obs, total_reward, done, info
