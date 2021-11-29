import torch
from torch import nn
from typing import Callable, Iterator, List, Tuple
from icecream import ic
import numpy as np
from torch.nn.modules.linear import Identity

class RacerNet(nn.Module):
    
    def __init__(self, input_shape: Tuple[int], n_actions: int, hidden_size: int = 256):
        
        super().__init__()

        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[2], out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
        )

        self.res_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        )

        self.relu = nn.ReLU()

        self.fc_block = nn.Sequential(
            # nn.Linear(4256, hidden_size),
            nn.Linear(51200, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        try:
            if len(x.shape) < 4:
                # x = np.swapaxes(x, 0, 2)
                x.unsqueeze_(0)
            elif len(x.shape) == 4:
                # x = np.swapaxes(x, 1, 3)
                x.squeeze_(1)
            elif len(x.shape) == 5:
                x.squeeze_(1)
                x.squeeze_(2)
                # x = np.swapaxes(x, 1, 3)

            x = self.conv_net(x)
            # identity = x
            x = self.res_block(x)
            # ic(identity.shape)
            # ic(x.shape)
            # x += identity
            x = self.relu(x)
            x = torch.flatten(x, start_dim=-3)
            x = self.fc_block(x)

        except RuntimeError as e:
            ic(x.shape)
            ic(x)
            ic(e)
            raise e

        return x.squeeze(0)
