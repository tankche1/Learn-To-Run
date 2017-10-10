
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
from running_state import Shared_obs_stats
from train import train,test

from osim.env import RunEnv
import time
from utils import TrafficLight, Counter
from chief import chief

from models import Shared_grad_buffers
from collections import defaultdict

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
parser.add_argument('--test', action='store_true',
                    help='test ')
parser.add_argument('--feature', type=int, default=153, 
                    help='features num')
parser.add_argument('--force', action='store_true',
                    help='force two leg together')
parser.add_argument('--start-epoch', type=int, default=0, 
                    help='start-epoch')


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['OMP_NUM_THREADS'] = '1' 
    torch.manual_seed(args.seed)

    num_inputs = args.feature
    num_actions = 18

    traffic_light = TrafficLight()
    counter = Counter()

    ac_net = ActorCritic(num_inputs, num_actions)
    opt_ac = optim.Adam(ac_net.parameters(), lr=args.lr)

    shared_grad_buffers = Shared_grad_buffers(ac_net)
    shared_obs_stats = Shared_obs_stats(num_inputs)

    if args.resume:
        print("=> loading checkpoint ")
        checkpoint = torch.load('../../7.87.t7')
        #checkpoint = torch.load('../../best.t7')
        args.start_epoch = checkpoint['epoch']
        #best_prec1 = checkpoint['best_prec1']
        ac_net.load_state_dict(checkpoint['state_dict'])
        opt_ac.load_state_dict(checkpoint['optimizer'])
        opt_ac.state = defaultdict(dict, opt_ac.state)
        #print(opt_ac)
        shared_obs_stats = checkpoint['obs']

        print(ac_net)
        print("=> loaded checkpoint  (epoch {})"
                .format(checkpoint['epoch']))

    ac_net.share_memory()

    
    #opt_ac.share_memory()
    #running_state = ZFilter((num_inputs,), clip=5)
    

    processes = []

    if args.test:
        p = mp.Process(target=test, args=(args.num_processes, args, ac_net, shared_obs_stats, opt_ac))
        p.start()
        processes.append(p)


    p = mp.Process(target=chief, args=(args, args.num_processes+1, traffic_light, counter, ac_net,shared_grad_buffers, opt_ac))
    p.start()
    processes.append(p)

    for rank in range(0, args.num_processes):
        p = mp.Process(target=train, args=(rank, args, traffic_light, counter,  ac_net, shared_grad_buffers, shared_obs_stats,opt_ac))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
