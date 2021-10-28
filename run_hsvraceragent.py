import gym
import numpy as np
from torch import nn
from torch.distributions import Categorical, Normal
import torch
from typing import Callable, Iterator, List, Tuple

#from hsvracer import HSVRacerAgent
from hsvracer import BaselineHSVRacerAgent
import deepracer_gym
from wrappers import *
import click



@click.command()
@click.option('--model', help='Model file to use')
# NOTE no default port to keep from messing up training by 
#      connecting to a running server
@click.option('--port', default=6000, help='Port to connect to env on')
@click.option('--stack-size', default=4, help='Port to connect to env on')
def main(model, port, stack_size):

    agent = BaselineHSVRacerAgent(model, stack_size)

    env = gym.make('deepracer_gym:deepracer-v0', port=port)
    #env = Extractor(env)
    #env = BasicRewardRework(env)
    #env = Stacker(env, stack_size=4)

    print("Deepracer Environment Connected succesfully")

    steps_completed = 0
    episodes_completed = 0
    total_reward = 0
    total_total_reward = 0

    episode_count = 5
    for _ in range(episode_count):
        done = False
        state = env.reset()
        action = agent.register_reset(state)
        while not done:
            observation, reward, done, info = env.step(action)
        
            steps_completed += 1 
            total_reward += reward
        
            if done:
                episodes_completed += 1
                print(
                    "Episodes Completed:", episodes_completed, 
                    "Steps:", steps_completed, 
                    "Total Reward", total_reward, 
                    "Current Reward", reward)

                steps_completed = 0
                total_total_reward += total_reward
                total_reward = 0
                observation = env.reset()
                action = agent.register_reset(observation)
            else:
                action  = agent.compute_action(observation, info)

    print("Avg Total Reward:", total_total_reward/episode_count)

if __name__ == '__main__':
    main()