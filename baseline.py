from typing import Deque
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

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        sub_n_input_channels = observation_space.shape[1]
        ic(observation_space.shape)

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=5, stride=2),
            nn.LeakyReLU(),
            # nn.BatchNorm2d(64),

            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.LeakyReLU(),

            nn.Conv2d(64, 128, kernel_size=5, stride=2),
            nn.LeakyReLU(),

            nn.Conv2d(128, 256, kernel_size=5, stride=2),
            nn.LeakyReLU(),

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
        #ic()
        x = observations
        #ic(x.shape)
        #x = self.sub_cnn(x)
        #ic(x.shape)
        #x = self.cnn(x)
        #ic(x.shape)
        #x = self.linear(x)
        #ic(x.shape)
        #return x
        #ic(x.shape)
        #x = x.view(x.shape[0], x.shape[2], x.shape[1], x.shape[3:])
        #ic(x.shape)
        x = self.cnn(x)
        x = self.linear(x)
        #ic(x.shape)
        
        #x = th.gather(x, 1, [0, 2, 4, 8, 16, 32, 48, 64, 80, 96, 112, 128])
        #ic(x.shape)
        return x

        #return self.linear(self.cnn(observations))


from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.vec_env.stacked_observations import StackedObservations
from icecream import ic

def create_env_fn(port):
    ic(port)
    def env_fn():
        nonlocal port
        env = gym.make("deepracer_gym:deepracer-v0", port=port)
        return Monitor(env)
    return env_fn

import numpy as np

class stacker(gym.Wrapper):
    def __init__(self, env, stack_size=4):
        super(stacker, self).__init__(env)
        ic(env.observation_space.shape)
        self.frames = None
        self.observation_space = gym.spaces.Box(low=0, high=255., shape=(4,120,160), dtype=np.uint8)

    def reset(self):
        obs = self.env.reset() 
        self.frames = Deque([obs] * 4, maxlen=4)
        state = np.squeeze(np.stack(self.frames), axis=1)
        return state

    def step(self, action):
        obs = self.env.step(action)
        self.frames.append(obs)
        state = np.squeeze(np.stack(self.frames), axis=1)
        ic(state.shape)
        return state


N_ENVS = 28
EVAL_N_ENVS = 8
PORT = 8888
STACK_SIZE = 4
from time import sleep
#env = gym.make("deepracer_gym:deepracer-v0")
def main():
    # TODO make_env seems to preserve rollout info
    #env_checker = check_env(gym.make("deepracer_gym:deepracer-v0"))
    #env_fn = create_env_fn()
    env = SubprocVecEnv(
        [create_env_fn(PORT+idx) for idx in range(N_ENVS)])
    sleep(10)
    #env = SubprocVecEnv([env_fn for _ in range(N_ENVS)])
    ic(env.observation_space.shape)
    #env = VecTransposeImage(env)
    ic(env.observation_space.shape)
    env = VecFrameStack(env, n_stack=STACK_SIZE, channels_order='first')
    ic(env.observation_space.shape)
    #env = VecNormalize(env, norm_obs=True, norm_reward=False)
    #env = stacked_observations(env, n_stack=3)
    ic(env.observation_space.shape)
    # TODO pass through info for stacker

    eval_env = SubprocVecEnv(
        [create_env_fn(PORT+idx+N_ENVS) for idx in range(EVAL_N_ENVS)])
    #eval_env = SubprocVecEnv([env_fn for _ in range(EVAL_N_ENVS)])

    ic(eval_env.observation_space.shape)
    eval_env = VecFrameStack(eval_env, n_stack=STACK_SIZE, channels_order='first')
    ic(eval_env.observation_space.shape)
    #eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
    #eval_env = VecTransposeImage(eval_env)
    ic(eval_env.observation_space.shape)
    #rollout = (8192 * 1) // N_ENVS
    rollout = (8192*2) // N_ENVS
    #rollout = 8192 // N_ENVS
    #rollout = 256
    ev_call = EvalCallback(
        eval_env,
        #n_eval_episodes=max(10, EVAL_N_ENVS*4),
        n_eval_episodes=EVAL_N_ENVS*4,
        eval_freq=(rollout*4),
        #log_path="./logs/",
        best_model_save_path="37/models/",
        deterministic=True,
    )
    '''
    ev2_call = EvalCallback(
        eval_env,
        n_eval_episodes=10,
        eval_freq=5000,
        log_path="./logs/",
        best_model_save_path="./logs/",
        deterministic=False,
    )
    '''
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=[dict(pi=[128, 128], vf=[128, 128])]
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
        gamma=0.99,
        #batch_size=256, #TODO is is batch size causing problems?
        batch_size=512,
        n_epochs=4,
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