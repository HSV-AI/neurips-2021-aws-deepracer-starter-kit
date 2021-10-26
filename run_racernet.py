import gym
import numpy as np
from torch import nn
from torch.distributions import Categorical, Normal
import torch
from typing import Callable, Iterator, List, Tuple

from hsvracer import RacerNet
import deepracer_gym
from icecream import ic

device = "cuda"
model = torch.load('hsvracer/weights/all_actor_net.pt')
model.to(device)
model.eval()

env = gym.make('deepracer_gym:deepracer-v0')
observation = env.reset()
observation = observation['_next_state']['STEREO_CAMERAS']
state = torch.FloatTensor(observation).to(device)
with torch.no_grad():
    logits = model(state)
    pi = Categorical(logits=logits)
    action = pi.sample()

print("Deepracer Environment Connected succesfully")

steps_completed = 0
episodes_completed = 0
total_reward = 0

while episodes_completed < 5:

    observation, reward, done, info = env.step(action.cpu().numpy())

    observation = observation['STEREO_CAMERAS']

    steps_completed += 1 
    total_reward += reward
  
    if done:
        episodes_completed += 1
        print("Episodes Completed:", episodes_completed, "Steps:", steps_completed, "Total Reward", total_reward, "Current Reward", reward)
        steps_completed = 0
        total_reward = 0
        observation = env.reset()
        observation = observation['_next_state']['STEREO_CAMERAS']

    state = torch.FloatTensor(observation).to(device)
    with torch.no_grad():
        logits = model(state)
        pi = Categorical(logits=logits)
        action = pi.sample()
