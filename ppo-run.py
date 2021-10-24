import argparse

import gym
import icecream

import pytorch_lightning as pl
import torch
from icecream import ic
import numpy as np
from torch.distributions import Categorical

from hsvracer import PPOLightning

pl.seed_everything(0)
device = "cuda"

model = PPOLightning.load_from_checkpoint("deepracer-gym/max_length_checkpoints/epoch=42-step=343.ckpt")
model.eval()
model.to(device)

env = gym.make('deepracer_gym:deepracer-v0')

state = torch.FloatTensor(env.reset()).to(device)
with torch.no_grad():
    pi, action, value = model(state)

print("Deepracer Environment Connected succesfully")

steps_completed = 0
episodes_completed = 0
total_reward = 0

while episodes_completed < 5:

    observation, reward, done, info = env.step(action.cpu().numpy())
  
    steps_completed += 1 
    total_reward += reward

    if done:
        episodes_completed += 1
        print("Episodes Completed:", episodes_completed, "Steps:", steps_completed, "Total Reward", total_reward, "Current Reward", reward)
        steps_completed = 0
        total_reward = 0
        observation = torch.FloatTensor(env.reset())

    state = torch.FloatTensor(observation).to(device)
    with torch.no_grad():
        pi, action, value = model(state)
