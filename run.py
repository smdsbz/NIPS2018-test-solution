#!/usr/bin/env python3
"""
You can run RandomAgent locally with following command:

    ./run.py RandomAgent

You can run FixedActionAgent with visuals with following command:

    ./run.py FixedActionAgent -v

You can submit FixedActionAgent with visuals with following command:

    ./run.py FixedActionAgent -s
"""
import argparse

from osim.env import ProstheticsEnv
from osim.http.client import Client

from wrappers import ClientToEnv, DictToListFull, ForceDictObservation, JSONable
from agents import *

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run or submit agent.')
    parser.add_argument('agent', help='specify agent\'s class name.')
    parser.add_argument('-t', '--train', action='store', dest='nb_steps',
                        help='train agent locally')
    parser.add_argument('-s', '--submit', action='store_true', default=False,
                        help='submit agent to crowdAI server')
    parser.add_argument('-v', '--visualize', action='store_true', default=False,
                        help='render the environment locally')
    args = parser.parse_args()

    if args.agent not in globals():
        raise ValueError('[run] Agent {} not found.'.format(args.agent))
    SpecifiedAgent = globals()[args.agent]

    if args.submit and args.nb_steps:
        raise ValueError('[run] Cannot train and submit agent at same time.')

    if args.submit and args.visualize:
        raise ValueError('[run] Cannot visualize agent while submitting.')

    if args.submit:
        remote_base = None      # TODO
        crowdai_token = None    # TODO
        # Submit agent
        client = Client(remote_base)
        client.env_create(crowdai_token, env_id='ProstheticsEnv')
        client_env = ClientToEnv(client)
        client_env = DictToListFull(client_env)
        client_env = JSONable(client_env)
        agent = SpecifiedAgent(client_env.observation_space,
                               client_env.action_space)
        agent.submit(client_env)
    elif args.nb_steps:
        # Train agent locally
        env = ProstheticsEnv(visualize=args.visualize)
        env = ForceDictObservation(env)
        env = DictToListFull(env)
        env = JSONable(env)
        print('state space =', env.observation_space.shape)
        print('action space =', env.action_space.sample().shape)
        baseline = BaselineNetwork(
            env.observation_space, env.action_space,
            param_path='model/baseline.param', baseline_lr=1e-4,
            device=device
        )
        agent = SpecifiedAgent(
            env.observation_space, env.action_space, 'model/pg.actor',
            policy_lr=1e-6,
            device=device
       )
        agent.train(
            env, baseline,
            episode=int(args.nb_steps), batch_size=2**7, replay_size=int(1e5),
            train_smooth_factor=5, gamma=0.997
        )
    else:
        # Test agent locally
        env = ProstheticsEnv(visualize=args.visualize)
        env = ForceDictObservation(env)
        env = DictToListFull(env)
        env = JSONable(env)
        agent = SpecifiedAgent(env.observation_space, env.action_space)
        agent.test(env)
