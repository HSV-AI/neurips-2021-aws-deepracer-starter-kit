import gym
import numpy as np
from torch import nn
from torch.distributions import Categorical, Normal
import torch
from typing import Callable, Iterator, List, Tuple

#from hsvracer import HSVRacerAgent
from hsvracer import BaselineHSVRacerAgent
import deepracer_gym

agent = BaselineHSVRacerAgent()

env = gym.make('deepracer_gym:deepracer-v0')

state = env.reset()
action = agent.register_reset(state)
print("Deepracer Environment Connected succesfully")

steps_completed = 0
episodes_completed = 0
total_reward = 0

while episodes_completed < 5:

    observation, reward, done, info = env.step(action)
  
    steps_completed += 1 
    total_reward += reward
  
    if done:
        episodes_completed += 1
        print("Episodes Completed:", episodes_completed, "Steps:", steps_completed, "Total Reward", total_reward, "Current Reward", reward)
        steps_completed = 0
        total_reward = 0
        observation = env.reset()
        action = agent.register_reset(observation)
    else:
        action  = agent.compute_action(observation, info)
