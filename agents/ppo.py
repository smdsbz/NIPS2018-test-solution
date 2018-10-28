# -*- coding: utf-8 -*-

__author__ = 'Xiaoguang Zhu'
__citation__ = [
    'https://github.com/higgsfield/RL-Adventure-2',
]

''''''

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

# from .utils import ReplayMemory
# from .model import DiffableStdNetwork

# from osim.env import ProstheticsEnv

from tensorboardX import SummaryWriter

import os


''' Network Structure '''

class ActorCriticNetwork(nn.Module):

    @staticmethod
    def init_global_vars(submod):
        if isinstance(submod, nn.Linear):
            nn.init.normal_(submod.weight, mean=0.0, std=0.1)
            nn.init.constant_(submod.bias, 0.1)

    def __init__(self, observation_dim, action_dim, initial_std=3.0):
        super(ActorCriticNetwork, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(observation_dim, 512), nn.SELU(),
            nn.Linear(512, 512), nn.SELU(),
            nn.Linear(512, 512), nn.SELU(),
            nn.Linear(512, action_dim)
        )
        self.logstd = nn.Parameter(torch.ones(1, action_dim) * initial_std)

        self.critic = nn.Sequential(
            nn.Linear(observation_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, action_dim)
        )

        self.apply(self.init_global_vars)

    def forward(self, obs):
        act_mean = torch.sigmoid(self.actor(obs))
        act_std = self.logstd.exp().expand_as(act_mean)
        act_dist = torch.distributions.Normal(act_mean, act_std)
        value = self.critic(obs)
        return act_dist, value


''' Module Definitions '''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PPOAgent:

    def __init__(self, observation_space, action_space,
                 param_path, learning_rate=1e-7,
                 device=device):
        self.device = device
        self.observation_dim = observation_space.shape[0]
        self.action_dim = action_space.shape[0]
        self.param_path = param_path
        self.model = ActorCriticNetwork(self.observation_dim, self.action_dim)
        self.model = self.model.to(device=device)
        has_model_loaded = False
        if param_path and os.path.exists(param_path):
            if os.path.exists(param_path + '.last'):
                if input('Load last minute model [y/n] ') in 'yY':      # NOTE: a directed <CR> would count as well!
                    print('Loading last minute model...', end='\r')
                    self.model.load_state_dict(torch.load(param_path + '.last'))
                    has_model_loaded = True
                    print('Last minute model loaded successfully!')
            if not has_model_loaded:
                print('Loading best performance model from last run...', end='\r')
                self.model.load_state_dict(torch.load(param_path))
                has_model_loaded = True
                print('Latest best performance model loaded!          ')
        else:
            print('No trained actor net found. Starting from scratch!')
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    @staticmethod
    def get_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
        values_ = values + [next_value]
        gae = 0.0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = (
                rewards[step]
                + masks[step] * gamma * values_[step + 1]
                - values_[step]
            )
            gae = (
                delta
                + masks[step] * gamma * tau * gae
            )
            returns.insert(0, gae + values_[step])
        return returns

    @staticmethod
    def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
        batch_size = states.size(0)
        for _ in range(batch_size // mini_batch_size + 1):
            rand_ids = np.random.randint(0, batch_size, mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]

    def ppo_update(self, ppo_epochs, mini_batch_size,
                   states, actions, log_probs, returns, advantages,
                   clip_param=0.2, writer=None):
        for ep in range(ppo_epochs):
            print('Fitting model {:.2f}%                   '
                  .format(ep / ppo_epochs * 100.0), end='\r')
            for state, action, old_log_probs, return_, advantage in self.ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
                dist, value = self.model(state)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

                actor_loss  = - torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()

                loss = 0.5 * critic_loss + actor_loss - 0.01 * entropy
                # print('actor loss:', actor_loss)
                # print('critic loss:', critic_loss)
                # print('surrogate loss:', loss)

                if writer is not None:
                    writer.add_histogram('actor/unclipped_ratio', ratio, self.__m_train_writer_cnt_update)
                    writer.add_scalar('actor/loss', actor_loss, self.__m_train_writer_cnt_update)
                    writer.add_scalar('critic/loss', critic_loss, self.__m_train_writer_cnt_update)
                    writer.add_scalar('train/surrogate_loss', loss, self.__m_train_writer_cnt_update)
                    writer.add_scalar('train/entropy', entropy, self.__m_train_writer_cnt_update)
                    writer.add_histogram('train/exploration_rate', self.model.logstd, self.__m_train_writer_cnt_update)
                    self.__m_train_writer_cnt_update += 1

                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()
        print('Training step done!' + (' ' * 20))

    def test(self, env, deterministic=True,
             fixed_reward=False):
        state = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            state = torch.tensor([state], dtype=torch.float32, device=self.device)
            dist, _ = self.model(state)
            if deterministic:
                action = dist.mean
            else:
                action = dist.sample()
                action.clamp_(0.0, 1.0)
            next_state, reward, done, _ = env.step(action.detach().cpu().numpy()[0])
            if fixed_reward:
                reward = 10.0
            state = next_state
            total_reward += reward
        return total_reward

    def submit(self, env):
        state = env.reset()
        sum_reward = 0.0
        try:
            while True:
                state = torch.tensor([state], dtype=torch.float32, device=self.device)
                dist, _ = self.model(state)
                action = dist.mean[0]
                next_state, reward, done, _ = env.step(action.detach().cpu().numpy())
                sum_reward += reward
                if done or next_state is None:
                    print('Last submit: reward =', sum_reward)
                    sum_reward = 0.0
                    next_state = env.reset()
                    if next_state is None:
                        break
                state = next_state
        except TypeError:
            pass
        env.submit()

    def train(self, env, summary_dir='summary/ppo',
              num_episodes=None, mini_batch_size=2**3, ppo_epochs=4, num_steps=500,
              fixed_reward=False):

        if os.path.isdir(summary_dir):
            for filename in os.listdir(summary_dir):
                os.remove(os.path.join(summary_dir, filename))
        writer_comment = 'PPOAgent'
        if fixed_reward:
            writer_comment += '_fixed-reward'
        writer = SummaryWriter(log_dir=summary_dir, comment=writer_comment)
        max_test_reward = -1e8

        self.__m_train_writer_cnt_update = 1
        self.__m_train_writer_cnt_test = 1

        episode_iterator = None
        if num_episodes:
            episode_iterator = range(num_episodes)
        else:
            from itertools import count
            episode_iterator = count()

        for ep in episode_iterator:
            print('======== Episode No.{} ========'.format(ep))
            state = env.reset()

            log_probs = []
            values    = []
            states    = []
            actions   = []
            rewards   = []
            masks     = []
            entropy   = 0.0

            __curr_traj_len = 0
            for st in range(num_steps):
                print('Collecting trajectory {:.2f}%          '
                      .format(st / num_steps * 100.0), end='\r')
                state = torch.tensor([state], dtype=torch.float32, device=self.device)
                dist, value = self.model(state)

                action = dist.sample().clamp_(0.0, 1.0)
                next_state, reward, done, _ = env.step(action.detach().cpu().numpy()[0])
                if fixed_reward:
                    reward = 10.0

                log_prob = dist.log_prob(action)
                entropy += dist.entropy().mean()

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(torch.tensor(reward, dtype=torch.float32, device=self.device))
                masks.append(torch.tensor(1 - done, dtype=torch.float32, device=self.device))

                states.append(state)
                actions.append(action)

                if done:
                    state = env.reset()
                    print('\n    resetting env, interaction lasted {} steps.'
                          .format(__curr_traj_len))
                    __curr_traj_len = 0
                else:
                    state = next_state
                    __curr_traj_len += 1
            print('Trajectory collection complete!')

            next_state = torch.tensor([next_state], dtype=torch.float32, device=self.device)
            _, next_value = self.model(next_state)
            returns = self.get_gae(next_value, rewards, masks, values)

            returns   = torch.cat(returns).detach()
            log_probs = torch.cat(log_probs).detach()
            values    = torch.cat(values)
            states    = torch.cat(states)
            actions   = torch.cat(actions)
            advantage = returns - values

            self.ppo_update(ppo_epochs, mini_batch_size,
                            states, actions, log_probs, returns, advantage,
                            writer=writer)

            if ep % 5 == 0:
                print('Running test run...', end='\r')
                test_reward = np.mean([self.test(env, fixed_reward=fixed_reward) for _ in range(1)])    # NOTE: only test once
                print('==> test reward: {}'.format(test_reward))
                writer.add_scalar('test/sum_reward', test_reward, self.__m_train_writer_cnt_test)
                self.__m_train_writer_cnt_test += 1
                if test_reward > max_test_reward:
                    max_test_reward = test_reward
                    torch.save(self.model.state_dict(), self.param_path)
                open(self.param_path+'.last', 'w').close()
                torch.save(self.model.state_dict(), self.param_path+'.last')

            torch.cuda.empty_cache()

            # end episode
