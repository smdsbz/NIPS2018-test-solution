# -*- coding: utf-8 -*-

__author__ = 'Xiaoguang Zhu'

''''''

# from multiprocessing import Process, Pipe, Queue
from multiprocess import Process, Pipe, Queue

import numpy as np
import torch
from osim.env import ProstheticsEnv
from wrappers import DictToListFull, ForceDictObservation, JSONable


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


''' Pool Classes '''

class EnvironmentPool:
    '''
    the Environment Pool - making parallelable agent-environment interaction
    possible

    # After instantiation and service is started, you can **ONLY** interact
    # with the environment with *messages* via the read/write `Queue`s.
    # Messges are sent in `tuple`s, `(message, content)`, where `message` is the
    # message string. List of legal options for messages are listed below:

    #     - `reset`: calls :func:`env.reset`, `content` should be a tuple containing
    #         environment ID only
    #     - `step`: calls :func:`env.step`, `content` should be a tuple of
    #         environment ID and action vector
    #     - `stop`: stops the service

    # Environments will return their infos via `Queue` as well, `(ID, content)`.
    '''

    @staticmethod
    def _factory_env(difficulty=1):
        env = ProstheticsEnv(visualize=False, difficulty=difficulty)
        env = ForceDictObservation(env)
        env = DictToListFull(env)
        env = JSONable(env)
        return env

    def __init__(self, difficulty=1,
                 pool_size=4, queue_size=16):   # NOTE: `queue_size` must be larger than 2x `pool_size`!!!
        self._args_queue = Queue(queue_size)
        # pipes for data dispatching
        self._args_pipes = [
            Pipe()
            for _ in range(pool_size)
        ]
        self._return_queue = Queue(queue_size)
        self._envs = [
            self._factory_env(difficulty=difficulty)
            for _ in range(pool_size)
        ]
        self._service_process = Process(
            target=self._service_job,
            args=(
                self._envs, self._args_pipes, self._args_queue,
                self._return_queue
            )
        )
        self._service_process.start()
        print('[LOG] environment service started!')

    def __len__(self):
        return len(self._envs)

    @property
    def args_queue(self):
        return self._args_queue

    @property
    def return_queue(self):
        return self._return_queue

    @staticmethod
    def _service_job(_envs, _args_pipes, _args_queue, _return_queue):
        '''
        Queue In:
        - `__stop_all__`
        - `reset_all`
        - `(env_id, env_args)` (for refs on `env_args`, see :func:`__instance_job` doc)

        Queue Out:
        - `(env_id, (obs, rew, done, info))`
        '''

        def __instance_job(env, env_id, args_pipe, return_queue):
            '''
            Pipe In: (`env_args`)
            - `'__stop__'`
            - `'reset'`
            - `numpy.ndarray` of `(19,)`

            Queue Out:
            see :func:`_service_job`
            '''
            while True:
                action = args_pipe.recv()
                if isinstance(action, str):
                    if action == 'reset':
                        env.reset()
                    elif action == '__stop__':
                        args_pipe.stop()
                        break
                else:
                    retval = env.step(action)
                    return_queue.put((env_id, retval))

        _pool = []

        # initialize workers
        for idx, (_, pipeout) in enumerate(_args_pipes):
            _pool.append(
                Process(
                    target=__instance_job,
                    args=(_envs[idx], idx, pipeout, _return_queue)
                )
            )
            _pool[-1].start()

        # start service loop
        # NOTE: This service loop produces data only!!!
        #       Make sure you consume data in `return_queue` fase enough, or
        #       you'll break the Queue!!!
        while True:
            data = _args_queue.get()
            if isinstance(data, str):
                if data == '__stop_all__':
                    for pipein, _ in _args_pipes:
                        pipein.send('__stop__')
                    for proc in _pool:
                        proc.join()
                    break
                elif data == 'reset_all':
                    for pipein, _ in _args_pipes:
                        pipein.send('reset')
                    # NOTE: data returned via return Queue
                else:
                    raise ValueError('unrecognized message:', data)
            elif isinstance(data, tuple):       # need for data dispatching
                target_env_id = data[0]
                target_pipein = _args_pipes[target_env_id][0]
                target_pipein.send(data[1])
                # NOTE: data returned via return Queue
            else:
                raise TypeError('data must be of type `str` or `tuple`')

    def _send_action(self, actionfn, env_id, obs):
        '''
        Return:
        - `act`
        - `val`
        - `log_prob`
        - `delta_entropy`
        - `dist` (reserved for future extensibility)

        Send:
        - `(env_id, act)` to `self._args_queue`
        '''
        dist, val = actionfn(obs)
        act = dist.sample().clamp_(0.0, 1.0)
        act = act.detach().cpu().numpy()[0]
        self._args_queue.put((env_id, act))
        log_prob = dist.log_prob(act)
        delta_entropy = dist.entropy().mean()
        return act, val, log_prob, delta_entropy, dist

    def get_trajectories_one_round(self, actionfn):
        '''
        Return:
        Un-concatenated trajectories
        '''
        target_rounds = len(self)
        dones_seen = [ False for _ in range(target_rounds) ]

        # returned via `_send_action`
        actions   = [ [] for _ in range(target_rounds) ]
        values    = [ [] for _ in range(target_rounds) ]
        states    = [ [] for _ in range(target_rounds) ]    # one more
        log_probs = [ [] for _ in range(target_rounds) ]
        entropy   = [ 0.0 for _ in range(target_rounds) ]
        # returned via Queue
        rewards   = [ [] for _ in range(target_rounds) ]
        masks     = [ [] for _ in range(target_rounds) ]

        def __send_and_save(env_id, obs):
            obs = torch.tensor([obs], dtype=torch.flaot32, device=device)
            act, val, log_prob, dlt_ent, _ = \
                self._send_action(actionfn, env_id, obs)
            log_probs[env_id].append(log_prob)
            values[env_id].append(val)
            states[env_id].append(obs)
            actions[env_id].append(act)
            entropy[env_id] += dlt_ent

        # run the very first step (sync-ed)
        self.args_queue.put('reset_all')
        initial_send_buffer = []
        for _ in range(target_rounds):
            env_id, first_obs = self.return_queue.get()
            initial_send_buffer.append((env_id, first_obs))
        for env_id, first_obs in initial_send_buffer:
            __send_and_save(env_id, first_obs)

        # receive and return (parallel)
        while not all(dones_seen):
            env_id, (next_obs, rew, done, _) = self.return_queue.get()
            # save returns: `rewards`, `masks`
            rewards[env_id].append(torch.tensor(rew, dtype=torch.float32, device=device))
            masks[env_id].append(torch.tensor(1 - done, dtype=torch.float32, device=device))
            if not done:
                __send_and_save(env_id, states[env_id][-1])
            else:
                dones_seen[env_id] = True

        _, next_val = actionfn(next_obs)

        return (
            actions, values, states, log_probs, entropy,
            rewards, masks, next_val
        )

    @staticmethod
    def concatenate_trajectories(
            actions, values, states, log_probs, entropy, rewards, masks):
        num_runs = len(actions)
        order = [ idx for idx in range(num_runs) ]
        cat_actions = []
        cat_values = []
        cat_states = []
        cat_log_probs = []
        cat_rewards = []
        cat_masks = []
        for idx in order:
            cat_actions.extend(actions[idx])
            cat_values.extend(values[idx])
            cat_states.extend(states[idx])
            cat_log_probs.extend(log_probs[idx])
            cat_rewards.extend(rewards[idx])
            cat_masks.extend(masks[idx])
        return (
            cat_actions, cat_values, cat_states, cat_log_probs,
            np.mean(entropy), cat_rewards, cat_masks
        )

    def get_trajectories(self, actionfn, approx_min_batchsize=500):
        print("[LOG] started collection trajectory...")
        *traj_data, next_value = self.get_trajectories_one_round(actionfn)
        traj_data = self.concatenate_trajectories(*traj_data)
        while len(traj_data[0]) < approx_min_batchsize * (2 / 3):
            *new_traj_data, next_value = self.get_trajectories_one_round(actionfn)
            new_traj_data = self.concatenate_trajectories(*new_traj_data)
            for item_idx in range(len(traj_data)):
                traj_data[item_idx].extend(new_traj_data[item_idx])
        return traj_data, next_value


''' Unit Test '''

if __name__ == '__main__':
    envpool = EnvironmentPool(pool_size=4)
    traj_data, next_value = envpool.get_trajectories(lambda: [0,] * 19)
    print(len(traj_data[0]))
    from pprint import pprint
    pprint(traj_data)
