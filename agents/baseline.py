# -*- coding: utf-8 -*-

import os

import torch
import torch.nn as nn
import torch.optim as optim

from .model import SimpleNetwork

from tensorboardX import SummaryWriter


''' Class Definitions '''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BaselineNetwork:

    def __init__(self, observation_space, action_space,
                 param_path='model/baseline.param',
                 baseline_lr=1e4,
                 summary_dir='summary/baseline',
                 device=device):
        '''
        Args:
            `observation_space`:
            `action_space`:
            `param_path`: pretrained net parameters
            `device`: cuda or cpu
        '''
        observation_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]
        self.param_path = param_path
        self.model = SimpleNetwork(observation_dim + action_dim, 1)
        self.model = self.model.to(device=device)
        if os.path.exists(param_path):
            self.model.load_state_dict(torch.load(param_path))
        else:
            print('No baseline net parameter file found. Starting from scratch!')
        self.lossfn = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=baseline_lr)
        self.summary_dir = summary_dir
        self.writer = SummaryWriter(summary_dir)
        self.device = device

    def get_baselines(self, obs, act):
        '''
        Args:
            `obs`: observations
            `act`: last action

        Return:
            list of baseline scalars
        '''
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        act = torch.tensor(act, dtype=torch.float32, device=self.device)
        feats = torch.cat([obs, act], dim=1)
        return self.model(feats).reshape([-1])

    def fit(self, predict, target, episode=None):
        '''
        Args:
            `predict`: baseline predicted by this `BaselineNetwork` (attached
                       to graph)
            `target`: ground-truth Q-value given by environment
            `episode`: episode number, used for summary
        '''
        self.optimizer.zero_grad()
        loss = self.lossfn(predict, target)
        print('baseline loss:', loss)
        loss.backward(retain_graph=True)
        self.optimizer.step()
        if all([self.writer, episode]):
            self.writer.add_scalar('train/baseline_loss', loss, episode)

    def save(self, path=None):
        if path is not None:
            self.param_path = path
        open(self.param_path, 'w').close()  # empty / touch file
        torch.save(self.model.state_dict(), self.param_path)

