import numpy as np
from agents.deepracer_base_agent import DeepracerAgent

class RandomDeepracerAgent(DeepracerAgent):
    def __init__(self):
        pass

    def register_reset(self, observations):
        action = np.random.randint(5)
        return action

    def compute_action(self, observations, info):
        action = np.random.randint(5)
        return action
