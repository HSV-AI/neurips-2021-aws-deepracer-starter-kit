import gym
from stable_baselines3.common import callbacks
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_checker import check_env
from torchsummary import summary

from wrappers import *

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.stack_size = observation_space.shape[0]
        sub_n_input_channels = observation_space.shape[1]
        ic(observation_space.shape)

        self.cnn = nn.Sequential(
            # nn.Conv3d(n_input_channels, 64, kernel_size=(2, 3, 3), stride=1),
            # nn.LeakyReLU(),

            nn.Conv3d(n_input_channels, 64, kernel_size=(2, 5, 5), stride=(1, 2, 2)),
            nn.LeakyReLU(),

            # NOTE don't need padding and two gos on 3d conv
            #      having more channels should accomplish the same thing

            # nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=1),
            # nn.LeakyReLU(),

            nn.Flatten(1, 2),

            nn.Conv2d(64, 128, kernel_size=5, stride=2),
            nn.LeakyReLU(),
            # nn.BatchNorm2d(64),

            nn.Conv2d(128, 256, kernel_size=5, stride=2),
            nn.LeakyReLU(),

            nn.Conv2d(256, 256, kernel_size=5, stride=2),
            nn.LeakyReLU(),

            #nn.Conv2d(128, 256, kernel_size=5, stride=2),
            #nn.LeakyReLU(),

            nn.Flatten(),

        )
        # TODO take in stero shape and stack after processing indv channels

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.LeakyReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = observations
        x = self.cnn(x)
        x = self.linear(x)
        
        #x = th.gather(x, 1, [0, 2, 4, 8, 16, 32, 48, 64, 80, 96, 112, 128])
        return x


from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.vec_env.stacked_observations import StackedObservations
from icecream import ic

def create_env_fn(port, stack_size=4):
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
EVAL_N_ENVS = 16
PORT = 8888
STACK_SIZE = 4
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
        n_eval_episodes=100,
        eval_freq=(rollout*4),
        #log_path="./logs/",
        best_model_save_path=f"models/{t.time()}",
        deterministic=True,
    )
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=512),
        # TODO no shared feature extractor?
        net_arch=[dict(pi=[256, 256], vf=[256, 256])]
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
        #batch_size=256, #TODO is is batch size causing problems?
        batch_size=512,
        n_epochs=8,
        policy_kwargs=policy_kwargs,
        tensorboard_log="tensorboard_logs/baseline_eval",
        verbose=1)
    ic(model.policy)
    summary(model.policy, env.observation_space.shape)
    model.learn(
        total_timesteps=200000000,
        callback=[ev_call],
        #callback=[ev_call, ev2_call],
    )

if __name__ == '__main__':
    main()