from typing import Deque
import numpy as np
#from .deepracer_base_agent import DeepracerAgent
from .deepracer_base_agent import DeepracerAgent
from stable_baselines3 import PPO
import os
from pathlib import Path
from collections import deque

DEFAULT_MODEL = "submit_models/best_model"

class BaselineHSVRacerAgent(DeepracerAgent):
    def __init__(self, model=DEFAULT_MODEL, stack_size=4):
        #filename = os.path.join(dirname, 'weights/all_actor_net.pt')
        file_path = Path(model)
        self.model = PPO.load(str(file_path))
        #print(self.model.policy)
        self.stacked_frames = None
        self.stack_size = stack_size
        if stack_size is None:
            self.stack_size = model.policy.feature_extraction.stack_size
        

    def register_reset(self, observations):
        self.stacked_frames = None
        return self._get_action(observations)

    @staticmethod
    def _reshape(obs):
        obs = obs['STEREO_CAMERAS']
        obs = np.moveaxis(obs, 2, 0)
        obs = obs / 255.0
        return obs

    def _get_action(self, observation):
        observation = self._reshape(observation)

        if self.stacked_frames is None:
            self.stacked_frames = deque(
                [observation] * self.stack_size, 
                maxlen=self.stack_size)
        self.stacked_frames.append(observation)

        state = np.stack(self.stacked_frames)
        action, _states = self.model.predict(state, deterministic=True)

        return action

    def compute_action(self, observations, info):
        return self._get_action(observations)


if __name__ == '__main__':
    agent = BaselineHSVRacerAgent()
    agent.register_reset()
