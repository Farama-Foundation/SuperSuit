from supersuit.utils.base_aec_wrapper import PettingzooWrap
from supersuit.utils.wrapper_chooser import WrapperChooser
import gym
from supersuit.utils.make_defaultdict import make_defaultdict


class aec_reward_lambda(PettingzooWrap):
    def __init__(self, env, change_reward_fn):
        assert callable(change_reward_fn), "change_reward_fn needs to be a function. It is {}".format(change_reward_fn)
        self._change_reward_fn = change_reward_fn

        super().__init__(env)

    def _check_wrapper_params(self):
        pass

    def _modify_spaces(self):
        pass

    def reset(self):
        super().reset()
        self.rewards = {agent: self._change_reward_fn(reward) for agent, reward in self.rewards.items()}
        self.__cumulative_rewards = make_defaultdict({a: 0 for a in self.agents})
        self._accumulate_rewards()

    def step(self, action):
        agent = self.env.agent_selection
        super().step(action)
        self.rewards = {agent: self._change_reward_fn(reward) for agent, reward in self.rewards.items()}
        self.__cumulative_rewards[agent] = 0
        self._cumulative_rewards = self.__cumulative_rewards
        self._accumulate_rewards()


class gym_reward_lambda(gym.Wrapper):
    def __init__(self, env, change_reward_fn):
        assert callable(change_reward_fn), "change_reward_fn needs to be a function. It is {}".format(change_reward_fn)
        self._change_reward_fn = change_reward_fn

        super().__init__(env)

    def step(self, action):
        obs, rew, done, info = super().step(action)
        return obs, self._change_reward_fn(rew), done, info


reward_lambda_v0 = WrapperChooser(aec_wrapper=aec_reward_lambda, gym_wrapper=gym_reward_lambda)
