from gym.spaces import Box, Space, Discrete
from .utils import basic_transforms
from .utils.frame_stack import stack_obs_space, stack_init, stack_obs
from .utils.frame_skip import check_transform_frameskip
from .utils.obs_delay import Delayer
from .utils.accumulator import Accumulator
import numpy as np
import gym



class BasicObservationWrapper(ObservationWrapper):
    """
    For internal use only
    """

    def __init__(self, env, module, param):
        self._module = module
        self._param = param
        super().__init__(env)
        assert isinstance(self.env.observation_space, Box), "Observation space is not Box, is {}.".format(self.observation_space)
        module.check_param(self.env.observation_space, param)
        self.observation_space = module.change_obs_space(self.env.observation_space, param)

    def _modify_observation(self, observation):
        obs_space = self.env.observation_space
        observation = self._module.change_observation(observation, obs_space, self._param)
        return observation


class color_reduction(BasicObservationWrapper):
    def __init__(self, env, mode="full"):
        super().__init__(env, basic_transforms.color_reduction, mode)


class resize(BasicObservationWrapper):
    def __init__(self, env, x_size, y_size, linear_interp=False):
        scale_tuple = (x_size, y_size, linear_interp)
        super().__init__(env, basic_transforms.resize, scale_tuple)


class dtype(BasicObservationWrapper):
    def __init__(self, env, dtype):
        super().__init__(env, basic_transforms.dtype, dtype)


class flatten(BasicObservationWrapper):
    def __init__(self, env):
        super().__init__(env, basic_transforms.flatten, True)


class reshape(BasicObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env, basic_transforms.reshape, shape)


class normalize_obs(BasicObservationWrapper):
    def __init__(self, env, env_min=0.0, env_max=1.0):
        shape = (env_min, env_max)
        super().__init__(env, basic_transforms.normalize_obs, shape)


class frame_stack(ObservationWrapper):
    def __init__(self, env, num_frames=4):
        self.stack_size = num_frames
        super().__init__(env)
        self._check_wrapper_params()
        self.observation_space = stack_obs_space(self.env.observation_space, self.stack_size)

    def _check_wrapper_params(self):
        assert isinstance(self.stack_size, int), "stack size of frame_stack must be an int"
        space = self.env.observation_space
        if isinstance(space, Box):
            assert 1 <= len(space.shape) <= 3, "frame_stack only works for 1,2 or 3 dimentional observations"
        elif isinstance(space, Discrete):
            pass
        else:
            assert False, "Stacking is currently only allowed for Box and Discrete observation spaces. The given observation space is {}".format(space)

    def reset(self):
        space = self.env.observation_space
        self.stack = stack_init(space, self.stack_size)
        return super().reset()

    def _modify_observation(self, observation):
        self.stack = stack_obs(self.stack, observation, self.env.observation_space, self.stack_size)
        return self.stack


class delay_observations(ObservationWrapper):
    def __init__(self, env, delay):
        super().__init__(env)
        self.delay = delay

    def _modify_observation(self, obs):
        return self.delayer.add(obs)

    def reset(self):
        self.delayer = Delayer(self.observation_space, self.delay)
        return super().reset()


class frame_skip(gym.Wrapper):
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
        total_reward = 0.0

        for x in range(num_skips):
            obs, rew, done, info = super().step(action)
            total_reward += rew
            if done:
                break

        return obs, total_reward, done, info


class max_observation(ObservationWrapper):
    def __init__(self, env, memory):
        super().__init__(env)
        self.memory = memory
        self.reduction = np.maximum

    def _modify_observation(self, obs):
        self.accumulator.add(obs)
        return self.accumulator.get()

    def reset(self):
        self.accumulator = Accumulator(self.observation_space, self.memory, self.reduction)
        return super().reset()


class sticky_actions(gym.Wrapper):
    def __init__(self, env, repeat_action_probability):
        super().__init__(env)
        assert 0 <= repeat_action_probability < 1
        self.repeat_action_probability = repeat_action_probability
        self.np_random, seed = gym.utils.seeding.np_random(None)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        super().seed(seed)

    def reset(self):
        self.old_action = None
        return super().reset()

    def step(self, action):
        if self.old_action is not None and self.np_random.uniform() < self.repeat_action_probability:
            action = self.old_action

        return super().step(action)


class clip_actions(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(self.action_space, Box), "clip_actions only works for Box action spaces"

    def _modify_action(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)


class clip_reward(RewardWrapper):
    def __init__(self, env, lower_bound=-1, upper_bound=1):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        super().__init__(env)

    def _change_reward_fn(self, rew):
        return max(min(rew, self.upper_bound), self.lower_bound)
