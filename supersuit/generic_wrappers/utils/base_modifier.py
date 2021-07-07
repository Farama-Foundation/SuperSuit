
class BaseModifier:
    def __init__(self):
        pass

    def reset(self):
        pass

    def modify_obs(self, obs):
        self.cur_obs = obs
        return obs

    def seed(self, seed):
        pass

    def get_last_obs(self):
        return self.cur_obs

    def modify_obs_space(self, obs_space):
        self.observation_space = obs_space
        return obs_space

    def modify_action(self, act):
        return act

    def modify_action_space(self, act_space):
        self.action_space = act_space
        return act_space
