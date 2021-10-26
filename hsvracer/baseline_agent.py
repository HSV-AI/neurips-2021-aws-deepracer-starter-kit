from typing import Deque
import numpy as np
#from .deepracer_base_agent import DeepracerAgent
from .deepracer_base_agent import DeepracerAgent
from stable_baselines3 import PPO
import os
from collections import deque

class BaselineHSVRacerAgent(DeepracerAgent):
    def __init__(self):
        #filename = os.path.join(dirname, 'weights/all_actor_net.pt')
        file_path = os.path.join('models/best_model.zip')
        self.model = PPO.load(file_path)
        self.stacked_frames = None

    def register_reset(self, observations):
        self.stacked_frames = None
        return self._get_action(observations)

    @staticmethod
    def _reshape(obs):
        obs = obs['STEREO_CAMERAS']
        obs = np.swapaxes(obs, 0, 2)
        obs = np.swapaxes(obs, 2, 1)
        obs = obs[1:]
        return obs

    def _get_action(self, observation):
        observation = self._reshape(observation)

        if self.stacked_frames is None:
            self.stacked_frames = deque([observation, observation, observation, observation], maxlen=4)
        self.stacked_frames.append(observation)

        state = np.stack(self.stacked_frames)
        state = np.squeeze(state)
        action, _states = self.model.predict(state, deterministic=True)

        return action

    def compute_action(self, observations, info):
        return self._get_action(observations)


if __name__ == '__main__':
    agent = BaselineHSVRacerAgent()
    agent.register_reset()
