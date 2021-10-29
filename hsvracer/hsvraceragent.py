import numpy as np
import torch
from torch.distributions import Categorical

from .deepracer_base_agent import DeepracerAgent
from hsvracer import RacerNet
import os

class HSVRacerAgent(DeepracerAgent):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'weights/all_actor_net.pt')
        self.model = torch.load(filename)
        self.model.to(self.device)
        self.model.eval()
        pass

    def register_reset(self, observations):
        observation = observations['STEREO_CAMERAS']
        state = torch.FloatTensor(observation).to(self.device)
        with torch.no_grad():
            logits = self.model(state)
            action = torch.argmax(logits).item()
            # pi = Categorical(logits=logits)
            # action = pi.sample()
        return action

    def compute_action(self, observations, info):
        observation = observations['STEREO_CAMERAS']
        state = torch.FloatTensor(observation).to(self.device)
        with torch.no_grad():
            logits = self.model(state)
            action = torch.argmax(logits).item()
            # pi = Categorical(logits=logits)
            # action = pi.sample()
        return action
