# -*- coding: utf-8 -*-

__author__ = 'Xiaoguang Zhu'

''''''

from osim.env import ProstheticsEnv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


''' Model Definition '''


class SimpleNetwork(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(SimpleNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.out = nn.Linear(512, output_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return self.out(x)

