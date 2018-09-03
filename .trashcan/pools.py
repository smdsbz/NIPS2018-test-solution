# -*- coding: utf-8 -*-

__author__ = 'Xiaoguang Zhu'

''''''

from multiprocessing import Process, Pipe, Queue

from osim.env import ProstheticsEnv


''' Pool Classes '''


# def env_read_service(read_queue, write_queue):
#     _env = ProstheticsEnv(visualize=False)
#     while True:
#         income = read_queue.get()
#         if income[0] == 'reset':
#             write_queue.put(_env.reset())
#         elif income[0] == 'step':
#             write_queue.put(_env.step(income[1]))
#         else:
#             pass


# class EnvironmentPool:
# 
#     def __init__(self, pool_size=4):
#         self._pool_size = pool_size
#         # self._args_queues = [
#         #     Queue(1) for _ in range(pool_size)
#         # ]
#         # self._return_queues = [
#         #     Queue(1) for _ in range(pool_size)
#         # ]
#         # self._pool = [
#         #     Process(target=env_read_service,
#         #             args=(
#         #                 self._args_queues[id],
#         #                 self._return_queues[id]
#         #             ))
#         #     for id in range(pool_size)
#         # ]
#         # self._occupie = [ False, ] * pool_size
#         self._envs = [
#             ProstheticsEnv(visualize=False)
#             for _ in range(pool_size)
#         ]
# 
#     def _step_on_one(self, envidx, act):
#         return self._envs[envidx].step(act)
# 
#     def run(self, controller):
#         trajectories = [ [], ] * self._pool_size
#         dones = [False,] * self._pool_size
#         # TODO

