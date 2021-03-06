import torch
from torch import nn
from typing import Callable, Iterator, List, Tuple
from icecream import ic
import numpy as np

class RacerNet(nn.Module):
    
    def __init__(self, input_shape: Tuple[int], n_actions: int, hidden_size: int = 256):
        
        super().__init__()

        self.conv_net = nn.Sequential(
            
            nn.Conv2d(in_channels=input_shape[2], out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=4)
        )

        self.fc_block = nn.Sequential(
            nn.Linear(4256, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):

        try:
            if len(x.shape) < 4:
                x = np.swapaxes(x, 0, 2)
                x.unsqueeze_(0)
            elif len(x.shape) == 4:
                x = np.swapaxes(x, 1, 3)
                x.squeeze_(1)

            x = self.conv_net(x)
            x = torch.flatten(x, start_dim=-3)
            x = self.fc_block(x)

        except RuntimeError as e:
            ic(x.shape)
            ic(x)
            ic(e)
            raise e

        return x.squeeze(0)
