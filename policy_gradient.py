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
import time
import json


''' Configurations '''

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', nargs=1)
parser.add_argument('-s', '--seed', nargs=1)
parser.add_argument('-d', '--device', nargs=1)
args = parser.parse_args()
args = vars(args)
if args['name'] is None:
    raise ValueError('option `name` cannot be `None`!')
if args['seed'] is not None:
    torch.manual_seed(args['seed'][0])
# else:
#     torch.manual_seed(int(time.time()))
if args['device'] is None:
    pass
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = args['device'][0]
    print('Only GPU{} is visible!'.format(os.environ['CUDA_VISIBLE_DEVICES']))

SUMMARY_DIR     = 'summary/pg_run{}'.format(args['name'][0])
MODEL_SAVE_PATH = 'model/run{}.actor'.format(args['name'][0])
if os.path.isdir(SUMMARY_DIR) or os.path.exists(MODEL_SAVE_PATH):
    raise ValueError('actor with name {} already exists!'
                     .format(args['name'][0]))
print('Summary file will be saved at {}'.format(SUMMARY_DIR))
print('Actor model parameter will be saved at {}'.format(MODEL_SAVE_PATH))


''' Hyperparameters '''


EPISODE         = int(1e5)
REPLAY_SIZE     = int(5 * 2 * 1e4)
MIN_BATCH_SIZE  = 2 ** 7

GAMMA           = 0.95
POLICY_LR       = 1e-5
BASELINE_LR     = 1e-4

START_DEVIATION = 10.0
MIN_DEVIATION   = 1.7
DEVIATION_DECAY = 0.997


''' Module Initalizations '''


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

env = ProstheticsEnv(visualize=False)
state_dim = len(env.reset())
action_dim = env.action_space.sample().shape[0]

# policy network
policy_net = SimpleNetwork(state_dim + action_dim, action_dim).to(device=device)
# action_dev = torch.ones(action_dim, device=device, requires_grad=False)
action_dev = START_DEVIATION
get_mean_actions = lambda feat: torch.sigmoid(policy_net(feat))     # action \in [0, 1]^{19}

# baseline network
baseline_net = SimpleNetwork(state_dim + action_dim, 1).to(device=device)
get_baselines = lambda feat: baseline_net(feat)


''' Training Ops '''


# policy network
# policy_loss = nn.L1Loss(reduce=True)
policy_optimizer = optim.Adam(policy_net.parameters(),
                              lr=POLICY_LR)

# baseline network
baseline_loss = nn.SmoothL1Loss()
baseline_optimizer = optim.Adam(baseline_net.parameters(),
                                lr=BASELINE_LR)


''' Helper Functions '''


def get_action(obs, last_act, deterministic=False):
    '''
    get action from policy net

    Args:
        `obs`: observation vector
        `last_act`: action taken in last step
        `deterministic`: whether to apply randomizer

    Return:
        When `deterministic` is `True`, returns action vector only;
        else returns randomized action and its scores w.r.t. randomizer
    '''
    obs = torch.tensor(obs, dtype=torch.float32, device=device)
    last_act = torch.tensor(last_act, dtype=torch.float32, device=device)
    feature = torch.cat([obs, last_act])
    feature = torch.stack([feature])
    mean_action = get_mean_actions(feature)[0]
    if deterministic:
        return mean_action
    randomizer = torch.distributions.MultivariateNormal(
        loc=mean_action,
        covariance_matrix=torch.eye(action_dim,
                                    dtype=torch.float32,
                                    device=device) * action_dev
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


def interact_once(env=env, smooth_action=False):
    '''
    interact with environment using __current policy__

    Args:
        None

    Return:
        `dict`: new trajectory
    '''
    # regularize parameter `smooth_action`
    if not smooth_action:
        smooth_action = 1
    last_obs = env.reset()
    # TODO: YOLO the 0-th step, come back with something better !
    last_act = env.action_space.sample()
    done = False
    trajectory = {
        'obs': [],
        'last_act': [],
        'act': [],
        'act_score': [],
        'rew': [],
        'done': [],
        'q': []
    }
    while not done:
        # take step in environment
        act, act_scores = get_action(last_obs, last_act)
        act = act.detach().cpu().numpy()
        for _ in range(smooth_action):
            if done:
                break
            obs, rew, done, _ = env.step(act)
            # add to memory
            trajectory['obs'].append(last_obs)
            trajectory['last_act'].append(last_act)
            trajectory['act'].append(act)
            trajectory['act_score'].append(act_scores)
            trajectory['rew'].append(rew)
            trajectory['done'].append(done)
            # prepare for next step
            last_obs, last_act = obs, act
    # make summary
    trajectory['q'] = get_q(trajectory['rew'])
    return trajectory


def test_run_reward(smooth_action=5):
    reward_sum = 0.0
    last_obs = env.reset()
    last_act = env.action_space.sample()
    done = False
    while not done:
        act = get_action(last_obs, last_act, deterministic=True)    # only `act` is returned
        act = act.detach().cpu().numpy()
        for _ in range(smooth_action):
            if done:
                break
            obs, rew, done, _ = env.step(act)
            reward_sum += rew
            last_obs, last_act = obs, act
    return reward_sum


''' Build Pipeline '''


def train():
    global action_dev

    writer = SummaryWriter(log_dir=SUMMARY_DIR)
    with open(os.path.join(SUMMARY_DIR, 'hyperparams.txt'), 'w') as f:
        # TODO: jsonify this !
        f.write(
            'bs={}\ngamma={}\npolicylr={}\nbaselinelr={}'
            .format(MIN_BATCH_SIZE, GAMMA, POLICY_LR, BASELINE_LR)
        )

    # fill replay buffer with sufficient data
    replay_buffer = ReplayMemory(REPLAY_SIZE)
    while len(replay_buffer) < MIN_BATCH_SIZE:
        print('\rCollecting first run: {:.2f}%'
              .format(len(replay_buffer) / MIN_BATCH_SIZE * 100.0),
              end='')
        trajectory = interact_once(smooth_action=5)
        replay_buffer.storemany(trajectory)
    print('\rCollecting first run done!' + ' ' * 10)    # new line

    last_test_reward = -1e4     # HACK: value small enough

    for episode in range(EPISODE):
        print('======== Episode {} ========'.format(episode))

        # insert new trajectory
        for _ in range(1):
            trajectory = interact_once(smooth_action=5)
            replay_buffer.storemany(trajectory)
        torch.cuda.empty_cache()    # NOTE: let's hope this will work!
        # TODO: insert best run

        # get sample trajectories
        sample = replay_buffer.sample(MIN_BATCH_SIZE)

        # get Q values (flattened)
        q_values = torch.tensor(
            sample.q,
            device=device
        )

        # get observations
        observations = torch.tensor(
            sample.obs,
            dtype=torch.float32,
            device=device
        )
        # get actions
        actions = torch.tensor(
            sample.act,
            dtype=torch.float32,
            device=device
        )
        baseline_feature = torch.cat([observations, actions], dim=1)

        # get baselines
        # NOTE: using state-only baseline
        baselines_raw = get_baselines(baseline_feature).reshape([-1])
        # baselines = (
        #     baselines_raw * q_values.std(dim=0)
        #     + q_values.mean(dim=0)
        # )
        baselines = baselines_raw

        # update baseline net
        baseline_optimizer.zero_grad()
        # loss = baseline_loss(
        #     baselines_raw,
        #     (q_values - q_values.mean(dim=0)) / (q_values.std(dim=0) + 1e-8)
        # )
        loss = baseline_loss(baselines, q_values)
        writer.add_scalar('baseline_loss', loss, episode)
        print('baseline loss:', loss)
        loss.backward(retain_graph=True)
        baseline_optimizer.step()

        # get scores (flattened)
        scores = torch.stack(sample.act_score)

        # get advantages
        advantages = q_values - baselines.detach()
        # # - normalize advantage
        # advantages = advantages / (advantages.std(dim=0) + 1e-8)

        # update policy net
        policy_optimizer.zero_grad()
        # if action_dev.grad is not None:
        #     action_dev.grad.zero_()
        # loss = policy_loss(scores, advantages)
        loss = torch.sum(scores * advantages)
        writer.add_scalar('policy_loss', loss, episode)
        print('policy loss:', loss)
        loss.backward(retain_graph=True)
        policy_optimizer.step()
        # if action_dev.grad is not None:
        #     with torch.no_grad():
        #         action_dev -= POLICY_LR * action_dev.grad
        # writer.add_histogram('action_dev',
        #                      # action_dev.detach().cpu().numpy(),
        #                      action_dev.cpu().numpy(),
        #                      episode)
        writer.add_scalar('action_dev', action_dev, episode)
        # print('action_dev:', action_dev)
        if action_dev > MIN_DEVIATION:
            action_dev *= DEVIATION_DECAY

        # test and eval
        if episode % 5 == 0:
            test_reward = test_run_reward()
            writer.add_scalar('test_reward', test_reward, episode)
            print('==> test reward:', test_reward)
            # save good model
            if last_test_reward < test_reward:
                last_test_reward = test_reward
                open(MODEL_SAVE_PATH, 'w').close()   # HACK: `touch` an empty file
                torch.save(
                    policy_net.state_dict(),
                    MODEL_SAVE_PATH
                )
    # end for


''' Main '''


if __name__ == '__main__':
    train()
