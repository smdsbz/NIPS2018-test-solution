# -*- coding: utf-8 -*-

__author__ = 'Xiaoguang Zhu'

''''''

from collections import namedtuple
import random


''' Utilities - Classes '''


class ReplayMemory:
    ''' Overwriting Replay Memory '''

    def __init__(self, size):
        '''
        `size`: maximum capacity of memory
        '''
        self._size = size
        self.__record_class = namedtuple(
            'ReplayRecord',
            'obs, last_act,  act, act_score, rew, done, q'
        )
        self._memory = self.__record_class(
            obs=[], last_act=[], act=[], act_score=[], rew=[], done=[], q=[]
        )
        self._pointer = 0

    def __len__(self):
        return len(self._memory.obs)

    def __getitem__(self, key):
        return self.__record_class(
            obs=self._memory.obs[key % self._size],
            last_act=self._memory.last_act[key % self._size],
            act=self._memory.act[key % self._size],
            act_score=self._memory.act_score[key % self._size],
            rew=self._memory.rew[key % self._size],
            done=self._memory.done[key % self._size],
            q=self._memory.q[key % self._size]
        )

    def store(self, contents):
        '''
        stores stuff into memory

        Args:
            `contents`: `dict` of objects to be stored
        '''
        # alloc space if still can
        if len(self) < self._size:
            self._memory.obs.append(None)
            self._memory.last_act.append(None)
            self._memory.act.append(None)
            self._memory.act_score.append(None)
            self._memory.rew.append(None)
            self._memory.done.append(None)
            self._memory.q.append(None)
        # else:
        #     print('utils.py:ReplayMemory: storage reached rooftop, replacing older records!')
        # store values
        self._memory.obs[self._pointer] = contents['obs']
        self._memory.last_act[self._pointer] = contents['last_act']
        self._memory.act[self._pointer] = contents['act']
        self._memory.act_score[self._pointer] = contents['act_score']
        self._memory.rew[self._pointer] = contents['rew']
        self._memory.done[self._pointer] = contents['done']
        self._memory.q[self._pointer] = contents['q']
        # pointer to next location
        self._pointer = (self._pointer + 1) % self._size
        return

    def storemany(self, contents):
        if len(contents['obs']) != len(contents['last_act'])            \
                or len(contents['obs']) != len(contents['act'])         \
                or len(contents['obs']) != len(contents['act_score'])   \
                or len(contents['obs']) != len(contents['rew'])         \
                or len(contents['obs']) != len(contents['done'])        \
                or len(contents['q']) != len(contents['q']):
            raise ValueError('Every entery of dict should have same length!')
        for idx in range(len(contents['obs'])):
            self.store({
                'obs': contents['obs'][idx],
                'last_act': contents['last_act'][idx],
                'act': contents['act'][idx],
                'act_score': contents['act_score'][idx],
                'rew': contents['rew'][idx],
                'done': contents['done'][idx],
                'q': contents['q'][idx]
            })
        return

    def sample(self, batch_size):
        '''
        returns random samples from memory

        Args:
            `batch_size`: count of samples to return

        Return:
            `namedtuple`: `namedtuple` of `list`s
        '''
        idxes = random.sample(range(len(self._memory.obs)), batch_size)
        retsample = self.__record_class(
            obs=[], last_act=[], act=[], act_score=[], rew=[], done=[], q=[]
        )
        for idx in idxes:
            retsample.obs.append(self._memory.obs[idx])
            retsample.last_act.append(self._memory.last_act[idx])
            retsample.act.append(self._memory.act[idx])
            retsample.act_score.append(self._memory.act_score[idx])
            retsample.rew.append(self._memory.rew[idx])
            retsample.done.append(self._memory.done[idx])
            retsample.q.append(self._memory.q[idx])
        return retsample

    # def sample_unpacked(self, batch_size):
    #     packed = random.sample(self._memory, batch_size)
    #     obs, act, rew, obsp, done = [], [], [], [], []
    #     for obs_t, act_t, rew_t, obs_tp1, done_t in packed:
    #         obs.append(obs_t)
    #         act.append(act_t)
    #         rew.append(rew_t)
    #         obsp.append(obs_tp1)
    #         done.append(1 if done_t else 0)
    #     return obs, act, rew, obsp, done


class LinearDecayImpulse:
    ''' Linearly Decay Baseline '''

    def __init__(self, start, end, slope_len):
        self._top       = start
        self._bottom    = end
        self._slope     = 1.0 * (end - start) / slope_len
        self._step      = 0

    def step(self):
        self._step += 1

    def value(self, at=None):
        x = at or self._step
        return max(x * self._slope + self._top,
                   self._bottom)

    def value_and_step(self):
        ret = self.value()
        self.step()
        return ret

