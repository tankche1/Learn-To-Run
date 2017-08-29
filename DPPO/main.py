import argparse
import os
import sys

import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

from model import Model, Shared_grad_buffers, Shared_obs_stats
from train import train
from test import test
from chief import chief
from utils import TrafficLight, Counter

from osim.env import RunEnv
import argparse

class Params():
    def __init__(self,args):
        self.batch_size = 1000
        #self.lr = 3e-4
        self.lr = args.lr
        #self.gamma = 0.99
        self.gamma = args.gamma
        #self.gae_param = 0.95
        self.gae_param = args.tau

        self.clip = args.clip_epsilon

        self.ent_coeff = 0.
        self.num_epoch = 10
        self.num_steps = 1000
        self.exploration_size = 1000
        self.num_processes = args.num_processes
        self.update_treshold = self.num_processes - 1
        self.max_episode_length = 10000
        self.seed = 1
        self.env_name = 'InvertedPendulum-v1'
        self.render = args.render
        self.num_inputs = 41
        self.num_outputs = 18
        self.bh = args.bh
        self.resume = args.resume
        self.skip = args.skip
        #self.env_name = 'Reacher-v1'
        #self.env_name = 'Pendulum-v0'
        #self.env_name = 'Hopper-v1'
        #self.env_name = 'Ant-v1'
        #self.env_name = 'Humanoid-v1'
        #self.env_name = 'HalfCheetah-v1'

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                    help='discount factor (default: 0.995)')
parser.add_argument('--env-name', default="Reacher-v1", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--lr', type=float, default=3e-4, 
                    help='learning rate (default: 3e-4)')
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
    params = Params(args)
    torch.manual_seed(params.seed)
    #env = gym.make(params.env_name)
    #num_inputs = env.observation_space.shape[0]
    #num_outputs = env.action_space.shape[0]
    num_inputs = params.num_inputs
    num_outputs = params.num_outputs

    traffic_light = TrafficLight()
    counter = Counter()

    shared_model = Model(num_inputs, num_outputs)
    shared_model.share_memory()
    shared_grad_buffers = Shared_grad_buffers(shared_model)
    #shared_grad_buffers.share_memory()
    shared_obs_stats = Shared_obs_stats(num_inputs)
    #shared_obs_stats.share_memory()
    optimizer = optim.Adam(shared_model.parameters(), lr=params.lr)
    test_n = torch.Tensor([0])
    test_n.share_memory_()

    processes = []
    p = mp.Process(target=test, args=(params.num_processes, params, shared_model, shared_obs_stats, test_n))
    p.start()
    processes.append(p)
    p = mp.Process(target=chief, args=(params.num_processes, params, traffic_light, counter, shared_model, shared_grad_buffers, optimizer))
    p.start()
    processes.append(p)
    for rank in range(0, params.num_processes):
        p = mp.Process(target=train, args=(rank, params, traffic_light, counter, shared_model, shared_grad_buffers, shared_obs_stats, test_n))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
