import argparse

import gym
import icecream

import pytorch_lightning as pl

from icecream import ic
import numpy as np

from hsvracer import PPOLightning


pl.seed_everything(0)

parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser = pl.Trainer.add_argparse_args(parent_parser)

parser = PPOLightning.add_model_specific_args(parent_parser)
args = parser.parse_args()


model = PPOLightning(**vars(args))

loss_checkpoint = pl.callbacks.ModelCheckpoint(dirpath="min_loss_checkpoints_211021", mode='min', monitor='loss_actor')
length_checkpoint = pl.callbacks.ModelCheckpoint(dirpath="max_length_checkpoints_211021", mode='max', monitor='avg_ep_len')
reward_checkpoint = pl.callbacks.ModelCheckpoint(dirpath="max_reward_checkpoints_211021", mode='max', monitor='avg_reward')
trainer = pl.Trainer(resume_from_checkpoint="deepracer-gym/max_length_checkpoints/epoch=42-step=343.ckpt", 
    callbacks=[loss_checkpoint,length_checkpoint, reward_checkpoint], gpus=1)
trainer.fit(model)
