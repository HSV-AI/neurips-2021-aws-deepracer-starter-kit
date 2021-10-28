import gym
from stable_baselines3.common import callbacks
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_checker import check_env
from torchsummary import summary

from wrappers import *
from modules import *

from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.vec_env.stacked_observations import StackedObservations
from icecream import ic

def create_env_fn(port, stack_size):
    ic(port)
    def env_fn():
        nonlocal port
        env = gym.make("deepracer_gym:deepracer-v0", port=port)
        env = Extractor(env)
        #env = RewardRework(env)
        env = gym.wrappers.FrameStack(env, stack_size)
        env = Monitor(env)
        return env
    return env_fn

import numpy as np
import time as t


N_ENVS = 16
EVAL_N_ENVS = 4
PORT = 8888
STACK_SIZE = 2
from time import sleep
#env = gym.make("deepracer_gym:deepracer-v0")
def main():
    # TODO make_env seems to preserve rollout info
    #env_checker = check_env(gym.make("deepracer_gym:deepracer-v0"))
    #env_fn = create_env_fn()
    env = SubprocVecEnv(
        [create_env_fn(PORT+idx, stack_size=STACK_SIZE) for idx in range(N_ENVS)])
    ic(env.observation_space.shape)
    # TODO pass through info for stacker

    # NOTE OFFICIAL GUIANDANCE 1 AGENT = 1 ENV
    eval_env = SubprocVecEnv(
        [create_env_fn(PORT+idx+N_ENVS, stack_size=STACK_SIZE) for idx in range(EVAL_N_ENVS)])

    ic(eval_env.observation_space.shape)

    rollout = (2048*8) // N_ENVS

    ev_call = EvalCallback(
        eval_env,
        #n_eval_episodes=max(10, EVAL_N_ENVS*4),
        #n_eval_episodes=EVAL_N_ENVS*1,
        # NOTE doesn't need to be equal to envs since may fail a lot
        n_eval_episodes=12,
        eval_freq=(rollout*2),
        #log_path="./logs/",
        best_model_save_path=f"models/{t.time()}",
        deterministic=True,
    )
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=512),
        # TODO no shared feature extractor?
        net_arch=[dict(pi=[512, 512], vf=[512, 512])]
        #features_extractor_kwargs=dict(
            #net_arch=[dict(
                #pi=[]
            #)]
        #),
    )  

    model = PPO(
        "MlpPolicy", 
        #"CnnPolicy", 
        env, 
        #n_steps=8192//N_ENVS,
        n_steps=rollout,
        gamma=0.999,
        #ent_coef=0.01,
        #batch_size=256, #TODO is is batch size causing problems?
        batch_size=128,
        learning_rate=1e-4,
        n_epochs=8,
        policy_kwargs=policy_kwargs,
        tensorboard_log="tensorboard_logs/baseline_eval",
        verbose=1)
    # NOTES
    # 1. decreasing learning rate causes a stale out with large batch size
    # 2. larger batch size alllows a larger learning rate
    # 3. Big steps seem to make first eval WAY better
    ic(model.policy)
    summary(model.policy, env.observation_space.shape)
    model.learn(
        total_timesteps=200000000,
        callback=[ev_call],
        #callback=[ev_call, ev2_call],
    )

if __name__ == '__main__':
    main()