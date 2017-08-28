'''
from osim.env import RunEnv

env = RunEnv(visualize=False)
observation = env.reset(difficulty = 0)
for i in range(200):
    observation, reward, done, info = env.step(env.action_space.sample())
    print(type(observation))
    print(len(observation))
    print(type(reward))
    print(reward)
    print(type(done))
    print(done)
    print(type(info))
    print(info)
    print(type(env.action_space.sample()))
    print(env.action_space.sample())
    print('-----------------------')
'''

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

for i in count(1):
    print(i)
print(count(1))