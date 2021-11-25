import argparse

import gym
import icecream

import pytorch_lightning as pl

from icecream import ic
import numpy as np
import torch
from hsvracer import PPOLightning


pl.seed_everything(0)

parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser = pl.Trainer.add_argparse_args(parent_parser)

parser = PPOLightning.add_model_specific_args(parent_parser)
args = parser.parse_args()


model = PPOLightning(**vars(args)).load_from_checkpoint("max_reward_checkpoints/epoch=596-step=4775.ckpt")
torch.save(model.actor.state_dict(), "actor.pt")
torch.save(model.actor, "all_actor.pt")
torch.save(model.actor.actor_net, "all_actor_net.pt")
