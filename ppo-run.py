import argparse

import gym
import icecream

import pytorch_lightning as pl
import torch
from icecream import ic
import numpy as np
from torch.distributions import Categorical

import cv2

from hsvracer import PPOLightning

pl.seed_everything(0)
device = "cuda"

model = PPOLightning.load_from_checkpoint("max_reward_checkpoints/epoch=571-step=4575.ckpt")
model.eval()
model.to(device)

env = gym.make('deepracer_gym:deepracer-v0', port=8888)

observations = env.reset()
observation = observations['STEREO_CAMERAS']
observation = np.swapaxes(observation, 0, 2)
left = observation[0][:,40:] / 255
right = observation[1][:,40:] / 255
state = torch.FloatTensor([left, right]).to(device)

with torch.no_grad():
    action = model.actor.actor_net(state)
    action = int(torch.argmax(action).item())
    # pi, action, value = model(state)

print("Deepracer Environment Connected succesfully")

steps_completed = 0
episodes_completed = 0
total_reward = 0

while episodes_completed < 5:

    observations, reward, done, info = env.step(action)
    # observations, reward, done, info = env.step(torch.argmax(action).item())
    observation = observations['STEREO_CAMERAS']
    observation = np.swapaxes(observation, 0, 2)
    left = observation[0][:,40:] / 255
    right = observation[1][:,40:] / 255
    state = torch.FloatTensor([left, right]).to(device)

    steps_completed += 1 
    total_reward += reward

    if done:
        episodes_completed += 1
        print("Episodes Completed:", episodes_completed, "Steps:", steps_completed, "Total Reward", total_reward, "Current Reward", reward, "Average Reward", total_reward / steps_completed)
        steps_completed = 0
        total_reward = 0
        observations = env.reset()
        observation = observations['STEREO_CAMERAS']
        observation = np.swapaxes(observation, 0, 2)
        left = observation[0][:,40:] / 255
        right = observation[1][:,40:] / 255
        state = torch.FloatTensor([left, right]).to(device)

    with torch.no_grad():
        action = model.actor.actor_net(state)
        action = int(torch.argmax(action).item())

        # pi, action, value = model(state)
