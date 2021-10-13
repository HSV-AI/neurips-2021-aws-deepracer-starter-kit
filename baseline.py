import gym
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_checker import check_env

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
        self.sub_cnn = nn.Sequential(
            #nn.Conv2d(sub_n_input_channels, 32, kernel_size=(5, 5), stride=(2, 2), 
                    #padding=0, dilation=1, bias=False),
            nn.Conv3d(n_input_channels, 32, kernel_size=(1, 5, 5), stride=(1, 1, 1), 
                    padding=(0, 0, 0), dilation=(1,1,1), bias=False),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            nn.Conv3d(32, 64, kernel_size=(2, 5, 5), stride=(1, 1, 1), 
                    padding='same', dilation=(1,1,1), bias=False),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(1, 2, 2),),
            nn.Flatten(1, 2)
        )

        self.cnn = nn.Sequential(
            #nn.Conv3d(sub_n_input_channels, 32, kernel_size=5, stride=(1, 2, 2), padding=0, bias=False),
            #nn.BatchNorm3d(32),
            #nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            # Conv2d *should* make kernel 3d automatically over stero channels 
            #nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=0, bias=False),
            #nn.BatchNorm3d(64),
            #nn.LeakyReLU(),
            #nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=0, bias=False),
            #nn.BatchNorm3d(128),
            #nn.LeakyReLU(),
            #nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=0, bias=False),
            #nn.BatchNorm3d(256),
            #nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        #with th.no_grad():
            #n_flatten = self.cnn(self.sub_cnn(
                #th.as_tensor(observation_space.sample()[None]).float()
            #).shape[1])

        self.linear = nn.Sequential(nn.Linear(128, features_dim), nn.LeakyReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        ic()
        x = observations
        ic(x.shape)
        x = self.sub_cnn(x)
        ic(x.shape)
        x = self.cnn(x)
        ic(x.shape)
        x = self.linear(x)
        ic(x.shape)
        return x

        #return self.linear(self.cnn(observations))


from icecream import ic
#env = gym.make("deepracer_gym:deepracer-v0")
def main():
    #env_checker = check_env(gym.make("deepracer_gym:deepracer-v0"))
    env = make_vec_env(
        "deepracer_gym:deepracer-v0", 
        n_envs=2,
        vec_env_cls=SubprocVecEnv)
    ic(env.observation_space.shape)
    env = VecFrameStack(env, n_stack=4, channels_order='first')
    ic(env.observation_space.shape)
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    ic(env.observation_space.shape)

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )

    model = PPO(
        "MlpPolicy", 
        env, 
        policy_kwargs=policy_kwargs,
        tensorboard_log="tensorboard_logs/baseline",
        verbose=1)
    model.learn(total_timesteps=2000000)

if __name__ == '__main__':
    main()