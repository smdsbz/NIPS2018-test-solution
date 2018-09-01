# -*- coding: utf-8 -*-

__author__ = 'Xiaoguang Zhu'

''''''

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter

# import random

from model import SimpleNetwork
from utils import ReplayMemory

from osim.env import ProstheticsEnv

import os


''' Configurations '''


SUMMARY_DIR     = 'summary/policy_gradient'


''' Hyperparameters '''


EPISODE         = int(1e5)
REPLAY_SIZE     = int(1e4)
MIN_BATCH_SIZE  = 2 ** 8

GAMMA           = 1 - 1e-3
POLICY_LR       = 1e-2
BASELINE_LR     = 1e-3


''' Module Initalizations '''


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

env = ProstheticsEnv(visualize=False)
state_dim = len(env.reset())
action_dim = env.action_space.sample().shape[0]

# policy network
policy_net = SimpleNetwork(state_dim, action_dim).to(device=device)
action_logdev = torch.ones(action_dim, device=device, requires_grad=True)
get_mean_actions = lambda obs: torch.sigmoid(policy_net(obs))   # action \in [0, 1]^{19}

# baseline network
baseline_net = SimpleNetwork(state_dim, 1).to(device=device)
get_baselines = lambda obs: baseline_net(obs)


''' Training Ops '''


# policy network
policy_loss = nn.L1Loss()
policy_optimizer = optim.Adam(policy_net.parameters(), lr=POLICY_LR)

# baseline network
baseline_loss = nn.SmoothL1Loss()
baseline_optimizer = optim.Adam(baseline_net.parameters(), lr=BASELINE_LR)


''' Helper Functions '''


def get_action(obs, deterministic=False):
    '''
    get action from policy net

    Args:
        `obs`: observation vector
        `deterministic`: whether to apply randomizer

    Return:
        While `deterministic` is `True`, returns action vector;
        otherwise returns randomized action and its scores w.r.t. randomizer
    '''
    mean_action = get_mean_actions(
        torch.tensor([obs], dtype=torch.float32, device=device)
    )[0]
    if deterministic:
        return mean_action
    randomizer = torch.distributions.MultivariateNormal(
        loc=mean_action,
        covariance_matrix=torch.diag(action_logdev)
    )
    stocastic_action = randomizer.sample()
    scores = -randomizer.log_prob(stocastic_action)
    return stocastic_action, scores


def get_q(rewards, gamma=GAMMA):
    '''
    calculate discounted Q-values

    Args:
        `rewards`: flat step rewards returned from env
        `gamma`: discount factor

    Return:
        `np.ndarray()`: list of discounted Q-values
    '''
    total = len(rewards)
    return np.array([
        np.sum(
            rewards[t:]
            * np.power(gamma, np.arange(total - t))
        )
        for t in range(total)
    ])


def interact_once():
    '''
    interact with environment using __current policy__

    Args:
        None

    Return:
        `dict`: new trajectory
    '''
    last_obs = env.reset()
    done = False
    trajectory = {
        'obs': [],
        'act': [],
        'act_score': [],
        'rew': [],
        'done': [],
        'q': []
    }
    while not done:
        # take step in environment
        act, act_scores = get_action(last_obs)
        act = act.detach().cpu().numpy()
        obs, rew, done, _ = env.step(act)
        # add to memory
        trajectory['obs'].append(last_obs)
        trajectory['act'].append(act)
        trajectory['act_score'].append(act_scores)
        trajectory['rew'].append(rew)
        trajectory['done'].append(done)
        # prepare for next step
        last_obs = obs
    trajectory['q'] = get_q(trajectory['rew'])
    return trajectory


def test_run_reward():
    reward_sum = 0.0
    last_obs = env.reset()
    done = False
    while not done:
        act = get_action(last_obs, deterministic=True)
        act = act.detach().cpu().numpy()
        obs, rew, done, _ = env.step(act)
        reward_sum += rew
        last_obs = obs
    return reward_sum


''' Build Pipeline '''


def train():
    global action_logdev

    writer = SummaryWriter(log_dir=SUMMARY_DIR + '-' +
                           'bs={}_gamma={}_policylr={}_baselinelr={}'
                           .format(MIN_BATCH_SIZE, GAMMA, POLICY_LR,
                                   BASELINE_LR))

    # fill replay buffer with sufficient data
    replay_buffer = ReplayMemory(REPLAY_SIZE)
    while len(replay_buffer) < MIN_BATCH_SIZE:
        print('\rCollecting first run: {:.2f}%'
              .format(len(replay_buffer) / MIN_BATCH_SIZE * 100.0),
              end='')
        trajectory = interact_once()
        replay_buffer.storemany(trajectory)
    print('\rCollecting first run done!' + ' ' * 10)    # new line

    for episode in range(EPISODE):
        print('======== Episode {} ========'.format(episode))

        # insert new trajectory
        trajectory = interact_once()
        replay_buffer.storemany(trajectory)

        # get sample trajectories
        sample = replay_buffer.sample(MIN_BATCH_SIZE)

        # get Q values (flattened)
        q_values = torch.tensor(
            sample.q,
            device=device
        )

        # get observations (flattened)
        observations = torch.tensor(
            sample.obs,
            dtype=torch.float32,
            device=device
        )

        # get baselines
        baselines_raw = baseline_net(observations).reshape(-1)
        baselines = (
            baselines_raw * q_values.std(dim=0)
            + q_values.mean(dim=0)
        )

        # update baseline net
        baseline_optimizer.zero_grad()
        loss = baseline_loss(
            baselines_raw,
            (q_values - q_values.mean(dim=0)) / (q_values.std(dim=0) + 1e-8)
        )
        writer.add_scalar('baseline_loss', loss, episode)
        print('baseline loss:', loss)
        loss.backward(retain_graph=True)
        baseline_optimizer.step()

        # get scores (flattened)
        scores = torch.stack(sample.act_score)

        # get advantages
        advantages = q_values - baselines.detach()
        # - normalize advantage
        advantages = advantages / (advantages.std(dim=0) + 1e-8)

        # update policy net
        policy_optimizer.zero_grad()
        if action_logdev.grad is not None:
            action_logdev.grad.zero_()
        loss = policy_loss(scores, advantages)
        writer.add_scalar('policy_loss', loss, episode)
        print('policy loss:', loss)
        loss.backward(retain_graph=True)
        policy_optimizer.step()
        if action_logdev.grad is not None:
            with torch.no_grad():
                action_logdev -= POLICY_LR * action_logdev.grad
        writer.add_histogram('action_logdev',
                             action_logdev.detach().cpu().numpy(),
                             episode)
        # print('action_logdev:', action_logdev)

        # test and eval
        if episode % 5 == 0:
            test_reward = test_run_reward()
            writer.add_scalar('test_reward', test_reward, episode)
            print('==> test reward:', test_reward)
    # end for


''' Main '''


if __name__ == '__main__':
    train()
