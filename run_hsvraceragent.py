import gym
import numpy as np
from torch import nn
from torch.distributions import Categorical, Normal
import torch
from typing import Callable, Iterator, List, Tuple

from hsvracer import HSVRacerAgent
import deepracer_gym
from wrappers import *
import click


# import pytorch_lightning as pl


@click.command()
@click.option('--model', default='submit_models/best_model', help='Model file to use')
# NOTE no default port to keep from messing up training by 
#      connecting to a running server
@click.option('--port', default=6000, help='Port to connect to env on')
#@click.option('--stack-size', default=2, help='Port to connect to env on')
def main(model, port):

    # pl.seed_everything(0)
    agent = HSVRacerAgent(record=True)

    env = gym.make('deepracer_gym:deepracer-v0', port=port)
    #env = Extractor(env)
    #env = BasicRewardRework(env)
    #env = Stacker(env, stack_size=4)

    print("Deepracer Environment Connected succesfully")

    steps_completed = 0
    episodes_completed = 0
    total_reward = 0
    total_total_reward = 0

    while episodes_completed < 5:

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

if __name__ == '__main__':
    main()