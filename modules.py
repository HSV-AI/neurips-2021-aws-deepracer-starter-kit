from torch import nn
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gym
from torch.nn.modules.batchnorm import BatchNorm3d
from res_modules import *


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
        #ic(observation_space.shape)

        self.cnn = nn.Sequential(
            # TODO try long kernels to see wider since vertical is less important
            nn.Conv3d(n_input_channels, 64, kernel_size=(sub_n_input_channels, 5, 5),
                    padding=(0, 2, 2), stride=(1, 2, 2)),
            BatchNorm3d(64),
            nn.LeakyReLU(),

            # nn.Conv3d(n_input_channels, 16, kernel_size=(2, 7, 7),
            #         padding=(0, 3, 3), stride=(1, 3, 3)),
            # nn.LeakyReLU(),

            # NOTE don't need padding and two gos on 3d conv
            #      having more channels should accomplish the same thing

            # nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=1),
            # nn.LeakyReLU(),

            nn.Flatten(1, 2),

            ResNetBasicBlock(64, 64),
            ResNetBasicBlock(64, 128, downsampling=2),
            ResNetBasicBlock(128, 128),
            ResNetBasicBlock(128, 256, downsampling=2),
            #ResNetBasicBlock(256, 256),
            ResNetBasicBlock(256, 512, downsampling=2),
            # ResNetBasicBlock(512, 512),

            nn.AdaptiveAvgPool2d(1),

            # nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            # nn.LeakyReLU(),

            # nn.Conv2d(32, 64, kernel_size=5, padding=2, stride=2),
            # nn.LeakyReLU(),

            # nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            # nn.LeakyReLU(),

            # nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=2),
            # nn.LeakyReLU(),

            # nn.Conv2d(128, 256, kernel_size=5, padding=2, stride=2),
            # nn.LeakyReLU(),

            # nn.Conv2d(256, 512, kernel_size=5, padding=2, stride=2),
            # nn.LeakyReLU(),

            # nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
            # nn.LeakyReLU(),

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

