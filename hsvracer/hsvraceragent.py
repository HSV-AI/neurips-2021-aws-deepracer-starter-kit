import numpy as np
import torch
from torch.distributions import Categorical
import pytorch_lightning as pl

from .deepracer_base_agent import DeepracerAgent
from hsvracer import RacerNet
import os
import cv2

class HSVRacerAgent(DeepracerAgent):
    def __init__(self):
        pl.seed_everything(0)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'weights/all_actor_net.pt')
        self.model = torch.load(filename)
        self.model.to(self.device)
        self.model.eval()
        pass

    def register_reset(self, observations):
        observation = observations['STEREO_CAMERAS']
        observation = np.swapaxes(observation, 0, 2)
        left = observation[0][:,40:] / 255
        right = observation[1][:,40:] / 255
        state = torch.FloatTensor([left, right]).to(self.device)

        with torch.no_grad():
            # logits = self.model(state)
            # pi = Categorical(logits=logits)
            # action = pi.sample().cpu().numpy()

            action = self.model(state)
            action = int(torch.argmax(action).item())

        return action

    def compute_action(self, observations, info):
        observation = observations['STEREO_CAMERAS']
        observation = np.swapaxes(observation, 0, 2)
        left = observation[0][:,40:] / 255
        right = observation[1][:,40:] / 255
        state = torch.FloatTensor([left, right]).to(self.device)
        with torch.no_grad():
            # logits = self.model(state)
            # pi = Categorical(logits=logits)
            # action = pi.sample().cpu().numpy()

            action = self.model(state)
            action = int(torch.argmax(action).item())
        return action
