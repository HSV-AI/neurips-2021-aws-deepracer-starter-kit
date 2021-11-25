import argparse

import gym
import icecream

import pytorch_lightning as pl
import torch
from icecream import ic
import numpy as np
from torch.distributions import Categorical

from hsvracer import DQNLightning

pl.seed_everything(0)
device = "cuda"

# env = gym.make('deepracer_gym:deepracer-v0', port=8888)

model = DQNLightning.load_from_checkpoint("dqn.ckpt", env = 'deepracer_gym:deepracer-v0')
model.eval()
model.to(device)

env = model.env

observations = env.reset()
observation = observations['STEREO_CAMERAS']
observation = observation / 255.
observation = np.swapaxes(observation, 0, 2)
state = torch.FloatTensor(observation).to(device)

with torch.no_grad():
    action = model(state)
    action = torch.argmax(action).item()

print("Deepracer Environment Connected succesfully")

steps_completed = 0
episodes_completed = 0
total_reward = 0

while episodes_completed < 5:

    observations, reward, done, info = env.step(action)
    observation = observations['STEREO_CAMERAS']

    steps_completed += 1 
    total_reward += reward

    if done:
        episodes_completed += 1
        print("Episodes Completed:", episodes_completed, "Steps:", steps_completed, "Total Reward", total_reward, "Current Reward", reward)
        steps_completed = 0
        total_reward = 0
        observations = env.reset()
        observation = observations['STEREO_CAMERAS']

    observation = observation / 255.
    observation = np.swapaxes(observation, 0, 2)
    state = torch.FloatTensor(observation).to(device)
    with torch.no_grad():
        action = model(state)
        action = torch.argmax(action).item()

