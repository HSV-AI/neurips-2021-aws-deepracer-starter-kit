import argparse

import gym
import icecream

import pytorch_lightning as pl

from icecream import ic
import numpy as np

from hsvracer import DQNLightning
from hsvracer import EvaluationCallback

pl.seed_everything(0)

parent_parser = argparse.ArgumentParser(add_help=False)
# parent_parser = pl.Trainer.add_argparse_args(parent_parser)

parser = DQNLightning.add_model_specific_args(parent_parser)
args = parser.parse_args()

model = DQNLightning(**vars(args))

# eval_env = gym.make('deepracer_gym:deepracer-v0', port=8889)
# eval_callback = EvaluationCallback(env=eval_env)
# eval_checkpoint = pl.callbacks.ModelCheckpoint(dirpath="dqn_eval_reward_checkpoints", mode='max', monitor='eval_avg_reward')

trainer = pl.Trainer(default_root_dir="./dqn_checkpoints",
    gpus=1, val_check_interval=100)
    # callbacks=[eval_callback, eval_checkpoint], gpus=1, val_check_interval=100)
trainer.fit(model)
trainer.save_checkpoint("dqn.ckpt")
