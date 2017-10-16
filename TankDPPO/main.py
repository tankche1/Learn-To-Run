import argparse
import sys
import math
import os
from collections import namedtuple
from itertools import count

import numpy as np
import numpy

#import scipy.optimize

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.autograd import Variable
import torch.multiprocessing as mp

import models

from osim.env import RunEnv
import time

from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import namedtuple
import multiprocessing

# from utils import *

#torch.set_default_tensor_type('torch.DoubleTensor')
#PI = torch.DoubleTensor([3.1415926])
torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.98, metavar='G',
                    help='discount factor (default: 0.995)')
parser.add_argument('--lr', type=float, default=3e-4, 
                    help='learning rate (default: 1e-3)')
parser.add_argument('--env-name', default="Reacher-v1", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--tau', type=float, default=0.98, metavar='G',
                    help='gae (default: 0.98)')
parser.add_argument('--seed', type=int, default=543, 
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=200, 
                    help='batch size (default: 200)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, 
                    help='interval between training status logs (default: 10)')
parser.add_argument('--entropy-coeff', type=float, default=0.0, 
                    help='coefficient for entropy cost')
parser.add_argument('--clip-epsilon', type=float, default=0.2, 
                    help='Clipping for PPO grad')
parser.add_argument('--use-sep-pol-val', action='store_true',
                    help='whether to use combined policy and value nets')
parser.add_argument('--bh',default='origin',
                        help='bh')
parser.add_argument('--resume', action='store_true',
                    help='loading the model')
parser.add_argument('--num-processes', type=int, default=4, 
                    help='how many training processes to use (default: 4)')
parser.add_argument('--skip', action='store_true',
                    help='execute an action three times')
parser.add_argument('--feature', type=int, default=91, 
                    help='features num')
parser.add_argument('--start-epoch', type=int, default=0, 
                    help='start-epoch')
parser.add_argument('--dif', type=int, default=2, 
                    help='difficulty')

args = parser.parse_args()
PATH_TO_MODEL = '../models/'+str(args.bh)

Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state',
                                       'reward'))

class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, state, action, mask, next_state, reward):
        """Saves a transition."""
        self.memory.append(Transition(state, action, mask, next_state, reward))

    def sample(self):
        return Transition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)

class Plot_reward:

     def __init__(self):
          self.plot_epoch = []
          self.plot_reward = []
          self.n = 0

     def push(self,reward):
          self.n = self.n+1
          self.plot_epoch.append(self.n)
          self.plot_reward.append(reward)

     def show(self):
          fig = plt.figure(1)
          plt.plot(self.plot_epoch, self.plot_reward)
          plt.xlabel('epoch')
          plt.ylabel('score')
          #plt.show()
          fig.savefig(PATH_TO_MODEL+'/plot.pdf')

class PPO:

     def __init__(self):
          self.Actor = models.create_actor(args.feature,18)
          self.Critic = models.create_critic(args.feature)
          self.Actor_optimizer = optim.Adam(self.Actor.parameters(), lr = 1e-4)
          self.Critic_optimizer = optim.Adam(self.Critic.parameters(), lr = 3e-4)


ROLLING_EVENT = multiprocessing.Event()
UPDATE_EVENT = multiprocessing.Event()
GLOBAL_UPDATE_COUNTER = multiprocessing.Event()
FILL = multiprocessing.Event()

def control():
     num = 0
     while(True):
          GLOBAL_UPDATE_COUNTER.wait()
          num = num+1
          if num == 64:
               FILL.set()
               num = 0

class Worker:

     def __init__(self,wid,diff):
          self.wid = wid
          self.env = RunEnv(visualize=False)
          self.dif = diff
          self.Actor = models.create_actor(args.feature,18)

     def choose_action(self,state):
          state = torch.from_numpy(state).unsqueeze(0)
          action_mean, _, action_std = self.Actor(Variable(state))
          action = torch.normal(action_mean, action_std)
          return action

     def work(self,globalPPO):
          self.Actor.load_state_dict(globalPPO.state_dict())
          while True:
               ep_r = 0
               step_count = 0
               state1,state2,state3,state = [0]*60, [0]*60, [0]*60, [0]*60
               balls = []
               state = self.env.reset(difficulty = self.dif)

               state1, state2, state3, state=process_observation(state1, state2, state3, state,balls)
               state = numpy.array(state)
               buffer_s, buffer_a, buffer_r = [], [], []
               while True:
                    if not ROLLING_EVENT.is_set():
                         ROLLING_EVENT.wait()
                         self.Actor.load_state_dict(globalPPO.state_dict())
                         buffer_s, buffer_a, buffer_r = [], [], []
                    a = choose_action(state)

                    r = 0
                    _,_r,_,_ = env.step(a)
                    r += _r
                    _,_r,_,_ = env.step(a)
                    r += _r
                    next_state, _r, done, _ = env.step(a)
                    r += _r

                    buffer_s.append(state)
                    buffer_a.append(a)
                    buffer_r.append(r)

                    addball_if_new(next_state,balls)
                    state1, state2, state3, next_state=process_observation(state1, state2, state3, next_state,balls)
                    next_state = numpy.array(next_state)
                    state = next_state
                    ep_r = ep_r + r

                    GLOBAL_UPDATE_COUNTER.set()
                    if done == True or FILL.is_set() :





