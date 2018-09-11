# -*- coding: utf-8 -*-

__author__ = 'Xiaoguang Zhu'

''''''

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from .utils import ReplayMemory
from .model import SimpleNetwork

from osim.env import ProstheticsEnv

from tensorboardX import SummaryWriter

import os
# import time
# import json


''' Module Definitions '''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PolicyGradientAgent:

    def __init__(self, observation_space, action_space,
                 param_path, policy_lr=1e-6,
                 std_start=10.0, std_floor=1.7, std_slope=0.997,
                 device=device):
        self.observation_dim = observation_space.shape[0]
        self.action_dim = action_space.shape[0]
        self.param_path = param_path
        self.model = SimpleNetwork(self.observation_dim + self.action_dim,
                                   self.action_dim)
        self.model = self.model.to(device=device)
        if param_path and os.path.exists(param_path):
            print('Loading previously trained actor net parameters...')
            self.model.load_state_dict(torch.load(param_path))
        else:
            print('No trained actor net found. Starting from scratch!')
        self.lossfn = nn.L1Loss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=policy_lr)
        # training-time stotastic action noise control parameters
        self.std_start = std_start
        self.std_floor = std_floor
        self.std_slope = std_slope
        self.action_dev = std_start
        self.device = device
        self.policy_lr = policy_lr

    def _get_mean_action(self, obs, last_act):
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        last_act = torch.tensor(last_act, dtype=torch.float32, device=self.device)
        feature = torch.cat([obs, last_act])
        feature = torch.stack([feature])
        mean_action = self.model(feature)[0]
        return mean_action

    def get_action(self, obs, last_act, deterministic=True, numpy=True):
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
        mean_action = self._get_mean_action(obs, last_act)
        if deterministic:
            if numpy:
                mean_action = mean_action.detach().cpu().numpy()
            return mean_action
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
        scores = - randomizer.log_prob(stocastic_action)
        if numpy:
            stocastic_action = stocastic_action.detach().cpu().numpy()
        return stocastic_action, scores

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

    def _interact_once(self, env, gamma,
                       smooth_factor=None,
                       fixed_reward=False):
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
            act, act_scores = self.get_action(last_obs, last_act, deterministic=False)
            # NOTE: `act` are detached from graph, but `act_scores` are not!
            #       gradients should flow into `act_scores`!
            for _ in range(smooth_factor):
                if done:
                    break
                obs, rew, done, _ = env.step(act)
                if fixed_reward:
                    rew = 1.0
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
        trajectory['q'] = self.get_q(trajectory['rew'], gamma)
        return trajectory

    def _test_run(self, env, smooth_factor=None, fixed_reward=True):
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
        last_act = env.action_space.sample()
        done = False
        while not done:
            act = self.get_action(last_obs, last_act, deterministic=True)
            for _ in range(smooth_factor):
                if done:
                    break
                last_obs, rew, done, _ = env.step(act)
                if fixed_reward:
                    rew = 1.0
                reward_sum += rew
                last_act = act
        return reward_sum

    def test(self, env, smooth_factor=1):
        retrew = self._test_run(env, smooth_factor=smooth_factor)
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

    def train(self, env, baseline_model, summary_dir='summary/pg',
              episode=int(1e5), batch_size=2**7, replay_size=int(1e5),
              train_smooth_factor=1, gamma=0.997, fixed_reward=True):
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
        # prepare training utilities
        _policy_loss_target = torch.zeros([batch_size], dtype=torch.float32, device=self.device)
        replay_buffer = ReplayMemory(replay_size)
        # fill replay with one batch of data before first training step
        while len(replay_buffer) < batch_size:
            print('\rCollecting trajectories for first run: {:.2f}%'
                  .format(len(replay_buffer) / batch_size * 100.0),
                  end='')
            traj = self._interact_once(env, gamma,
                                       smooth_factor=train_smooth_factor,
                                       fixed_reward=fixed_reward)
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
                                           fixed_reward=fixed_reward)
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
            loss = self.lossfn(torch.stack(sample.act_score) * advantages,
                               _policy_loss_target)
            print('policy loss:', loss)
            loss.backward(retain_graph=True)
            self.optimizer.step()
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
                if writer is not None:
                    writer.add_scalar('test/test_reward', test_reward, ep)

