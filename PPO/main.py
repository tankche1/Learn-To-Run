
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

from models import Policy, Value, ActorCritic
from replay_memory import Memory
from running_state import ZFilter,Shared_obs_stats
from train import train,test

from osim.env import RunEnv
import my_optim
import time

# from utils import *

#torch.set_default_tensor_type('torch.DoubleTensor')
#PI = torch.DoubleTensor([3.1415926])

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                    help='discount factor (default: 0.995)')
parser.add_argument('--lr', type=float, default=1e-3, 
                    help='learning rate (default: 1e-3)')
parser.add_argument('--env-name', default="Reacher-v1", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                    help='gae (default: 0.97)')
# parser.add_argument('--l2_reg', type=float, default=1e-3, metavar='G',
#                     help='l2 regularization regression (default: 1e-3)')
# parser.add_argument('--max_kl', type=float, default=1e-2, metavar='G',
#                     help='max kl value (default: 1e-2)')
# parser.add_argument('--damping', type=float, default=1e-1, metavar='G',
#                     help='damping (default: 1e-1)')
parser.add_argument('--seed', type=int, default=543, 
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=500, 
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
parser.add_argument('--test', action='store_true',
                    help='test ')
parser.add_argument('--feature', type=int, default=96, 
                    help='features num')


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.manual_seed(args.seed)

    num_inputs = args.feature
    num_actions = 9

    ac_net = ActorCritic(num_inputs, num_actions)
    opt_ac = my_optim.SharedAdam(ac_net.parameters(), lr=args.lr)

    if args.resume:
        print("=> loading checkpoint ")
        checkpoint = torch.load('../models/kankan/best.t7')
        #args.start_epoch = checkpoint['epoch']
        #best_prec1 = checkpoint['best_prec1']
        ac_net.load_state_dict(checkpoint['state_dict'])
        #opt_ac.load_state_dict(checkpoint['optimizer'])
        print(ac_net)
        print("=> loaded checkpoint  (epoch {})"
                .format(checkpoint['epoch']))

    ac_net.share_memory()
    #opt_ac = my_optim.SharedAdam(ac_net.parameters(), lr=args.lr)
    opt_ac.share_memory()
    #shared_grad_buffers = Shared_grad_buffers(shared_model)
    
    shared_obs_stats = Shared_obs_stats(num_inputs)
    processes = []

    if args.test:
        p = mp.Process(target=test, args=(args.num_processes, args, ac_net, opt_ac))
        p.start()
        processes.append(p)

    for rank in range(0, args.num_processes):
        can_save = False
        if rank==0:
            can_save = True

        p = mp.Process(target=train, args=(rank, args, ac_net, opt_ac, can_save,shared_obs_stats))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
