import numpy as np
import torch
from torch.distributions import Categorical
import pytorch_lightning as pl

from .deepracer_base_agent import DeepracerAgent
from hsvracer import RacerNet
import os
import cv2

class HSVRacerAgent(DeepracerAgent):
    def __init__(self, record = False):
        pl.seed_everything(0)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'weights/all_actor_net.pt')
        self.model = torch.load(filename)
        self.model.to(self.device)
        self.model.eval()
        self.observations = []
        self.record = record
        pass

    def register_reset(self, observations):
        observation = observations['STEREO_CAMERAS']
        observation = np.swapaxes(observation, 0, 2)
        left = observation[0][:,40:]
        left = left + np.random.normal(128, 64, left.shape)
        right = observation[1][:,40:]
        right = right + np.random.normal(128, 64, right.shape)
        if self.record:
            total = [left, right]
            self.observations.append(total)
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
        left = observation[0][:,40:]
        left = left + np.random.normal(128, 64, left.shape)
        right = observation[1][:,40:]
        right = right + np.random.normal(128, 64, right.shape)
        if self.record:
            total = [left, right]
            self.observations.append(total)
        state = torch.FloatTensor([left, right]).to(self.device)
        with torch.no_grad():
            # logits = self.model(state)
            # pi = Categorical(logits=logits)
            # action = pi.sample().cpu().numpy()

            action = self.model(state)
            action = int(torch.argmax(action).item())
        return action

    def save_obeservations(self, filename):
        if self.record:
            size = 320, 80
            fps=15
            out = cv2.VideoWriter(filename, 0, fps, (size[0], size[1]))

            for idx, state in enumerate(self.observations):

                left = state[0]
                right = state[1]
                state = np.vstack((left, right))

                state = np.rot90(state, 3)

                state = np.dstack([state, state, state])

                print(state.shape)
                out.write(state)
           
            out.release()
            #closing all open windows 
            cv2.destroyAllWindows()

            self.observations.clear()