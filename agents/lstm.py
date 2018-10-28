# -*- coding: utf-8 -*-

__author__ = 'Xiaoguang Zhu'

''''''

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from .utils import ReplayMemory
from .model import LSTMNetwork

from osim.env import ProstheticsEnv

from tensorboardX import SummaryWriter

import os
# import time
# import json


''' Module Definitions '''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LSTMAgent:

    def __init__(self, observation_space, action_space,
                 param_path, policy_lr=1e-6,
                 std_start=3.0, std_floor=0.1, std_slope=0.998,
                 device=device):
        self.device = device
        self.observation_dim = observation_space.shape[0]
        self.action_dim = action_space.shape[0]
        self.param_path = param_path
        self.model = LSTMNetwork(self.observation_dim, self.action_dim)
        self.model = self.model.to(device=device)
        if param_path and os.path.exists(param_path):
            print('Loading previously trained actor net parameters...')
            self.model.load_state_dict(torch.load(param_path))
        else:
            print('No trained actor net found. Starting from scratch!')
        # self.lossfn = nn.L1Loss()
        self.lossfn = lambda x: torch.mean(x)
        self.optimizer = optim.Adam(self.model.parameters(), lr=policy_lr)
        # training-time stotastic action noise control parameters
        self.std_start = std_start
        self.std_floor = std_floor
        self.std_slope = std_slope
        self.action_dev = std_start
        self.policy_lr = policy_lr

    def _get_mean_action(self, obs, last_h=None, last_c=None):
        obs = torch.tensor([[obs]], dtype=torch.float32, device=self.device)
        mean_action, (curr_h, curr_c) = self.model(obs, h=last_h, c=last_c)
        mean_action = torch.sigmoid(mean_action[0][0])
        return mean_action, (curr_h, curr_c)

    def get_action(self, obs, last_h=None, last_c=None,
                   deterministic=True, numpy=True):
        '''
        get action from policy net

        Args:
            `obs`: observation vector
            `last_h, last_c`: LSTM states
            `deterministic`: whether to apply randomizer

        Return:
            When `deterministic` is `True`, returns action vector only;
            else returns randomized action and its scores w.r.t. randomizer
        '''
        mean_action, (curr_h, curr_c) = \
            self._get_mean_action(obs, last_h=last_h, last_c=last_c)
        if deterministic:
            mean_action = mean_action.detach()
            if numpy:
                mean_action = mean_action.cpu().numpy()
            return mean_action, (curr_h, curr_c)
        randomizer = torch.distributions.MultivariateNormal(
            loc=mean_action,
            covariance_matrix=torch.eye(self.action_dim,
                                        dtype=torch.float32,
                                        device=self.device) * self.action_dev
        )
        stocastic_action = randomizer.sample()
        # cutoff into [0, 1]
        stocastic_action = torch.min(
            stocastic_action,
            torch.ones(self.action_dim, dtype=torch.float32, device=self.device)
        )
        stocastic_action = torch.max(
            stocastic_action,
            torch.zeros(self.action_dim, dtype=torch.float32, device=self.device)
        )
        # get usable action scores
        scores = - randomizer.log_prob(stocastic_action)    # `score` is connected to graph
        stocastic_action = stocastic_action.detach()
        if numpy:
            stocastic_action = stocastic_action.cpu().numpy()
        return stocastic_action, scores, (curr_h, curr_c)

    @staticmethod
    def get_q(rewards, gamma):
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

    def _interact_once(self, env, gamma, smooth_factor=None,
                       fixed_reward=False, shaped_reward=False):
        '''
        interact with environment using __current policy__

        Args:
            `env`: gym environment
            `smooth_factor`: an action will run for this many rounds before
                updated

        Return:
            `dict`: new trajectory
        '''
        # regularize parameter `smooth_action`
        if not smooth_factor:   # or left as default value of `None`
            smooth_factor = 1
        last_obs = env.reset()
        # NOTE: YOLO-ed the 0-th step, maybe come back with something better!
        done = False
        trajectory = {
            'obs': [],
            'act': [],
            'act_score': [],
            'rew': [],
            'done': [],
            'q': []
        }
        last_h, last_c = None, None
        while not done:
            # take step in environment
            act, act_scores, (last_h, last_c) = \
                self.get_action(last_obs, last_h=last_h, last_c=last_c,
                                deterministic=False)
            # NOTE: `act` are detached from graph, but `act_scores` are not!
            #       gradients should flow into `act_scores`!
            for _ in range(smooth_factor):
                if done:
                    break
                obs, rew, done, _ = env.step(act)
                if fixed_reward:
                    rew = 10.0
                    # rew -= 3.0 * (obs[-6] - 1.0) ** 2
                if shaped_reward:
                    # HACK: reward shaping: don't go side-ways!!!
                    # rew -= 1.0 * (obs[-4] ** 2)
                    # HACK: reward shaping: stay away from the ground!!!
                    # rew -= 1.0 * max(0.83 - obs[-8], 0.0)
                    # HACK: reward shaping: stay up right!!!
                    # rew -= 1.0 * (obs[57] ** 2)
                    # HACK: reward shaping: face forward!!!
                    # rew -= 1.0 * (obs[58] ** 2)
                    # HACK: reward shaping: don't back off!!!
                    rew -= 8.0 * (max(obs[59], 0.0) ** 2)
                    # # HACK: reward shaping: don't cross your legs!!!
                    # rew -= 1.0 * (min(obs[45], 0.0) ** 2)
                    # rew -= 1.0 * (max(obs[48], 0.0) ** 2)
                # add to memory
                trajectory['obs'].append(last_obs)
                trajectory['act'].append(act)
                trajectory['act_score'].append(act_scores)
                trajectory['rew'].append(rew)
                trajectory['done'].append(done)
                # prepare for next step
                last_obs = obs
        # make summary
        trajectory['q'] = self.get_q(trajectory['rew'], gamma)
        return trajectory

    def _test_run(self, env, smooth_factor=None, fixed_reward=False):
        '''
        Args:
            `env`: gym environment
            `smooth_factor`: action will be applied this much time before
                updated

        Return:
            sum of reward
        '''
        if not smooth_factor:
            smooth_factor = 1
        reward_sum = 0.0
        last_obs = env.reset()
        done = False
        last_h, last_c = None, None
        while not done:
            act, (last_h, last_c) = \
                self.get_action(last_obs, last_h=last_h, last_c=last_c,
                                deterministic=True)
            for _ in range(smooth_factor):
                if done:
                    break
                last_obs, rew, done, _ = env.step(act)
                if fixed_reward:
                    rew = 1.0
                reward_sum += rew
        return reward_sum

    def test(self, env, smooth_factor=1):
        retrew = self._test_run(env,
                                smooth_factor=smooth_factor,
                                fixed_reward=False)
        print('Test reward: {}'.format(retrew))

    def submit(self, env, smooth_factor=1):
        obs = env.reset()
        last_act = env.action_space.sample()
        reward_sum = 0.0
        try:
            while True:
                act = self.get_action(obs, last_act,
                                      deterministic=True, numpy=True).tolist()
                obs, rew, done, _ = env.step(act)
                reward_sum += rew
                if done or obs is None:
                    print('Last submit: reward={}'.format(reward_sum))
                    reward_sum = 0.0
                    obs = env.reset()
                    if obs is None:     # all runs done
                        break
        except TypeError:   # HACK: simple condition for "its done"!
            pass
        env.submit()

    def train(self, env, baseline_model, summary_dir='summary/lstm',
              episode=int(1e5), batch_size=2**7, replay_size=int(1e5),
              train_smooth_factor=1, gamma=0.997,
              fixed_reward=False, shaped_reward=False):
        '''
        Args:
            `env`: gym environment (list style)
            `baseline_model`: baseline model
            `summary_dir`: `str`: directory to put summaries
            `episode`: `int`: total episode count
            `batch_size`: `int`: batch size
            `replay_size`: `int`: replay memory size, affects VRAM usage
            `train_smooth_factor`: `int`: training-time action smooth factor
                (training-time only)
            `gamma`: gamma in Bellman equation (in [0, 1])
            `fixed_reward`: fix reward of every step to 1, the learning goal
                then changes to stand as long as possible
        '''
        # prepare summary directory
        if os.path.isdir(summary_dir):  # clear former summaries
            for file in os.listdir(summary_dir):
                os.remove(os.path.join(summary_dir, file))
        writer = SummaryWriter(summary_dir)
        with open(os.path.join(summary_dir, 'hyperparams.txt'), 'w') as f:
            f.write('bs={}\ngamma={}\npolicylr={}'
                    .format(batch_size, gamma, self.policy_lr))
        # # prepare training utilities
        # _policy_loss_target = - 1e8 * torch.ones([batch_size], dtype=torch.float32, device=self.device)
        replay_buffer = ReplayMemory(replay_size, store_last_act=False)
        # fill replay with one batch of data before first training step
        while len(replay_buffer) < batch_size:
            print('\rCollecting trajectories for first run: {:.2f}%'
                  .format(len(replay_buffer) / batch_size * 100.0),
                  end='')
            traj = self._interact_once(env, gamma,
                                       smooth_factor=train_smooth_factor,
                                       fixed_reward=fixed_reward,
                                       shaped_reward=shaped_reward)
            replay_buffer.storemany(traj)
        print('\rFinished data collection for first run!' + ' ' * 10)
        # start training
        last_test_reward = -1e8     # HACK: approx. negative infinity
        for ep in range(episode):
            print('======== Episode {} ========'.format(ep))
            # update replay every new training step
            for _ in range(1):      # NOTE: only insert one trajectory
                traj = self._interact_once(env, gamma,
                                           smooth_factor=train_smooth_factor,
                                           fixed_reward=fixed_reward,
                                           shaped_reward=shaped_reward)
                writer.add_scalar('train/train_reward', np.sum(traj['rew']), ep)
                replay_buffer.storemany(traj)
            if torch.cuda.is_available():   # clear memory cache
                torch.cuda.empty_cache()
            # sample training data from replay
            sample = replay_buffer.sample(batch_size)
            # get baselines, and update baseline model
            baselines = baseline_model.get_baselines(sample.obs, sample.act)
            q_values = torch.tensor(sample.q, dtype=torch.float32, device=self.device)
            baseline_model.fit(baselines, q_values, episode=ep)
            # get advantages
            advantages = q_values - baselines.detach()
            writer.add_histogram('debug/advantages', advantages, ep)
            # update policy model
            self.optimizer.zero_grad()
            # loss = self.lossfn(torch.stack(sample.act_score) * advantages,
            #                    _policy_loss_target)
            loss = self.lossfn(- torch.stack(sample.act_score) * advantages)
            print('policy loss:', loss)
            loss.backward(retain_graph=True)
            self.optimizer.step()   # minimize
            if writer is not None:
                writer.add_scalar('debug/action_dev', self.action_dev, ep)
                writer.add_scalar('train/policy_loss', loss, ep)
            if self.action_dev > self.std_floor:
                self.action_dev *= self.std_slope
            if ep % 5 == 0:
                test_reward = self._test_run(env,
                                             smooth_factor=train_smooth_factor,
                                             fixed_reward=fixed_reward)
                print('==> test reward:', test_reward)
                baseline_model.save()
                if last_test_reward < test_reward:
                    last_test_reward = test_reward
                    open(self.param_path, 'w').close()
                    torch.save(self.model.state_dict(), self.param_path)
                curr_param_path = self.param_path + '.last'
                open(curr_param_path, 'w').close()
                torch.save(self.model.state_dict(), curr_param_path)
                if writer is not None:
                    writer.add_scalar('test/test_reward', test_reward, ep)

