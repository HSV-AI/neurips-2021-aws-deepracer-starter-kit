import gym
import numpy as np
from torch import nn
from torch.distributions import Categorical, Normal
import torch
from typing import Callable, Iterator, List, Tuple

import deepracer_gym
import pytorch_lightning as pl

pl.seed_everything(0)
class RacerNet(nn.Module):
    
    def __init__(self, input_shape: Tuple[int], n_actions: int, hidden_size: int = 256):
        
        super().__init__()

        self.conv_net = nn.Sequential(
            
            nn.Conv2d(in_channels=input_shape[2], out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=4)
        )

        self.fc_block = nn.Sequential(
            nn.Linear(4256, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):

        try:
            if len(x.shape) < 4:
                x = np.swapaxes(x, 0, 2)
                x.unsqueeze_(0)
            elif len(x.shape) == 4:
                x = np.swapaxes(x, 1, 3)
                x.squeeze_(1)

            x = self.conv_net(x)
            x = torch.flatten(x, start_dim=-3)
            x = self.fc_block(x)

        except RuntimeError as e:
            ic(x.shape)
            ic(x)
            ic(e)
            raise e

        return x.squeeze(0)

device = "cpu"
model = torch.load('all_actor_net.pt')
model.to(device)
model.eval()

print(model)

env = gym.make('deepracer_gym:deepracer-v0')
state = torch.FloatTensor(env.reset()).to(device)
with torch.no_grad():
    logits = model(state)
    pi = Categorical(logits=logits)
    action = pi.sample()
    action = action.cpu().numpy()


# action = torch.argmax(logits).item()

print("Deepracer Environment Connected succesfully")

steps_completed = 0
episodes_completed = 0
total_reward = 0

for _ in range(1500):

    observation, reward, done, info = env.step(action)
    env.render()
    steps_completed += 1 
    total_reward += reward
  
    if done:
        episodes_completed += 1
        print("Episodes Completed:", episodes_completed, "Steps:", steps_completed, "Reward", total_reward)
        steps_completed = 0
        total_reward = 0
        observation = env.reset()

    state = torch.FloatTensor(observation).to(device)
    with torch.no_grad():
        logits = model(state)
        pi = Categorical(logits=logits)
        action = pi.sample()
        action = action.cpu().numpy()

    # action = torch.argmax(logits).item()
