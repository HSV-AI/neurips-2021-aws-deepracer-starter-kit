import gym
import numpy as np

import deepracer_gym

env = gym.make('deepracer_gym:deepracer-v0')

obs = env.reset()

print("Deepracer Environment Connected succesfully")

steps_completed = 0
episodes_completed = 0
total_reward = 0

for _ in range(500):
    observation, reward, done, info = env.step(np.random.randint(5))
  
    steps_completed += 1 
    total_reward += reward
  
    if done:
        episodes_completed += 1
        print("Episodes Completed:", episodes_completed, "Steps:", steps_completed, "Reward", total_reward)
        steps_completed = 0
        total_reward = 0