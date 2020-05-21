import aec_wrappers
import re

def agent_indicator(env):
    agent_names = env.agents
    fits_format = re.fullmatch("[a-z]+_[0-9]+", agent)
    env_type_mapper = agent_names
