# -*- coding: utf-8 -*-

__author__ = 'Xiaoguang Zhu'

''''''

import numpy as np
# import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

import random

from utils import ReplayMemory, LinearDecayImpulse
from model import SimpleNetwork

from osim.env import ProstheticsEnv


''' Hyperparameters '''

MAX_PATH_LEN    = 301
GAMMA           = 1 - 1e-3

START_PROB      = 0.9
END_PROB        = 0.05
PROB_SLOPE      = int(1e5)

UPDATE_RATE     = 10

EPISODE         = int(1e4)
BATCH_SIZE      = 128

LEARNING_RATE   = 1e-1


''' Module Initializations '''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

env = ProstheticsEnv(visualize=False)
state_dim = len(env.reset())    # `list` of length 158
action_dim = env.action_space.sample().shape[0]     # `np.ndarray` of length 19

replay = ReplayMemory(int(1 * 1e5))
explore_prob = LinearDecayImpulse(START_PROB, END_PROB, PROB_SLOPE)

explore_q = SimpleNetwork(state_dim, action_dim).to(device)
# HACK: having action in [0, 1]^{19}, use `sigmoid` to convert logits to
#       degrees of forces applied directly, neural net will sort out the rest
get_action_from_explore_q = lambda obs: torch.sigmoid(explore_q(obs))
# NOTE: use *Huber Loss* to smooth high variance in Q value
explore_loss = lambda input, target: F.smooth_l1_loss(input, target)
explore_optimizer = torch.optim.SGD(explore_q.parameters(),
                                    lr=LEARNING_RATE)

target_q = SimpleNetwork(state_dim, action_dim).to(device)
get_action_from_target_q = lambda obs: torch.sigmoid(target_q(obs))
sync_target_q = lambda: target_q.load_state_dict(explore_q.state_dict())
sync_target_q()
target_q.eval()     # required by `nn.BatchNorm1d`??


''' Helper Functions '''

def choose_action(obs, follow_policy=False):
    _sample_logit = random.random()
    _sample_valve = explore_prob.value_and_step()
    if follow_policy:
        with torch.no_grad():
            return get_action_from_target_q(
                torch.tensor([obs],
                             dtype=torch.float32,
                             device=device)
            )[0]
    elif _sample_logit < _sample_valve:
        return torch.tensor(env.action_space.sample(),
                            device=device)
    else:
        with torch.no_grad():
            return get_action_from_explore_q(
                torch.tensor([obs],
                             dtype=torch.float32,
                             device=device)
            )[0]


def update_replay_memory():
    obs_t = np.array(env.reset())
    done = False
    while not done:
        act_t = choose_action(obs_t)
        obs_tp1, rew_t, done, _ = env.step(act_t)
        replay.store(obs_t, act_t, rew_t, obs_tp1, done)
        obs_t = np.array(obs_tp1)


def test_run():
    obs_t = np.array(env.reset())
    done = False
    reward_sum = 0.0
    while not done:
        act_t = choose_action(obs_t, follow_policy=True)
        obs_tp1, rew_t, done, _ = env.step(act_t)
        reward_sum += rew_t
        obs_t = np.array(obs_tp1)
    return reward_sum


def replay_and_learn():
    obs_t, act_t, rew_t, obs_tp1, done_mask = \
        replay.sample_unpacked(BATCH_SIZE)
    q = target_q(
        torch.tensor(obs_t,
                    dtype=torch.float32,
                    device=device)
    )
    q_next = explore_q(
        torch.tensor(obs_tp1,
                     dtype=torch.float32,
                     device=device)
    ).detach()
    # TODO: use one-on-one loss
    rew_t, done_mask = (
        torch.tensor(np.stack([rew_t,] * action_dim, axis=-1),
                     dtype=torch.float32,
                     device=device),
        torch.tensor(np.stack([done_mask,] * action_dim, axis=-1),
                     dtype=torch.float32,
                     device=device)
    )
    explore_score = rew_t + (1 - done_mask) * GAMMA * q_next
    average_score = q
    loss = explore_loss(average_score, explore_score)
    print('loss:', loss.data)
    explore_optimizer.zero_grad()
    loss.backward()
    explore_optimizer.step()


''' Pipeline Building '''

def main(
        plot_reward=False,
        visualize_end_policy=False):

    reward_recorder = []

    # collect enough training samples
    while len(replay) < BATCH_SIZE:
        update_replay_memory()

    # do training steps
    for episode in range(EPISODE):

        update_replay_memory()

        replay_and_learn()

        # update / test target policy periodically
        if episode % UPDATE_RATE == 0:
            sync_target_q()
            reward_recorder.append(test_run())
            print('last reward:', reward_recorder[-1])
    # end for

    # if plot_reward:
    #     plt.plot(reward_recorder)
    #     plt.show()


''' Main '''

if __name__ == '__main__':
    main()
