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
import threading

# from utils import *

#torch.set_default_tensor_type('torch.DoubleTensor')
#PI = torch.DoubleTensor([3.1415926])
torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.98, metavar='G',
                    help='discount factor (default: 0.995)')
parser.add_argument('--lr', type=float, default=3e-4, 
                    help='learning rate (default: 1e-3)')
parser.add_argument('--actorlr', type=float, default=1e-4, 
                    help='learning rate (default: 1e-3)')
parser.add_argument('--criticlr', type=float, default=3e-4, 
                    help='learning rate (default: 1e-3)')
parser.add_argument('--env-name', default="Reacher-v1", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--tau', type=float, default=0.98, metavar='G',
                    help='gae (default: 0.98)')
parser.add_argument('--seed', type=int, default=543, 
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=256, 
                    help='batch size (default: 256)')
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

from observation_processor import processed_dims
args.feature = processed_dims

import Queue
ROLLING_EVENT = threading.Event()
UPDATE_EVENT = threading.Event()
QUEUE = Queue.Queue()
ROLLING_EVENT.set()
UPDATE_EVENT.clear()

class nnagent:

    def __init__(self,args):
        self.actor = models.create_actor(args.feature,18).cuda()
        self.critic = models.create_critic(args.feature).cuda()
        self.old_actor = models.create_actor(args.feature,18).cuda()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = args.actorlr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = args.criticlr)
        self.batch_size = args.batch_size

        self.epoch = 0
        self.plot_epoch = []
        self.plot_reward = []
        self.best = -1000
        import threading as th
        self.lock = th.Lock()
        self.msize = 0
        self.GAMMA = args.gamma
        self.S_DIM = args.feature
        self.A_DIM = 18

    def normal_log_density(self, x, mean, log_std, std):
        var = std.pow(2)
        log_density = -(x - mean).pow(2) / (
            2 * var) - 0.5 * math.log(2 * math.pi) - log_std
        return log_density.sum(1)


    def update(self):
        while True:
            UPDATE_EVENT.wait()

            time.sleep(1)
            data = [QUEUE.get() for _ in range(QUEUE.qsize())]
            print(self.msize)
            self.msize = 0
            data = np.vstack(data)

            #memory.push(state, np.array([action]), mask, next_state, reward, discounted_reward)
            
            states, actions, masks, next_states, rewards, discounted_rewards = data[:, :self.S_DIM], data[:,self.S_DIM:self.S_DIM+self.A_DIM], data[:,self.S_DIM+self.A_DIM:self.S_DIM+self.A_DIM+1], data[:,self.S_DIM+self.A_DIM+1:self.S_DIM+self.A_DIM+1+self.S_DIM], data[:,self.S_DIM+self.A_DIM+1+self.S_DIM:self.S_DIM+self.A_DIM+1+self.S_DIM+1], data[:,self.S_DIM+self.A_DIM+1+self.S_DIM+1:self.S_DIM+self.A_DIM+1+self.S_DIM+1+1]

            print(states.shape)
            #print(states.shape,actions.shape,masks.shape,next_states.shape,rewards.shape,discounted_rewards.shape)
            rewards = torch.Tensor(rewards).cuda()
            masks = torch.Tensor(masks).cuda()
            #actions = torch.Tensor(np.concatenate(actions, 0))
            actions = torch.Tensor(actions).cuda()
            states = torch.Tensor(states).cuda()
            discounted_rewards = torch.Tensor(discounted_rewards).cuda()
            values = self.critic(Variable(states))
            next_states = torch.Tensor(next_states).cuda()
            next_state_values = self.critic(Variable(next_states))

            returns = torch.Tensor(actions.size(0),1).cuda()
            deltas = torch.Tensor(actions.size(0),1).cuda()
            advantages = torch.Tensor(actions.size(0),1).cuda()

            #print(rewards.size(),masks.size(),actions.size())



            prev_return = 0
            prev_value = 0
            prev_advantage = 0
            #returns = discounted_rewards[i]
            #deltas = rewards + args.gamma * next_state_values * 
            for i in range(rewards.size(0)):
                #returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
                returns[i] = discounted_rewards[i]
                #deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values.data[i]
                deltas[i] = rewards[i] + args.gamma * next_state_values.data[i][0] * masks[i] - values.data[i]
                advantages[i] = deltas[i] #+ args.gamma * args.tau * prev_advantage * masks[i]
                #prev_return = returns[i, 0]
                #prev_value = values.data[i, 0]
                #prev_advantage = advantages[i, 0]

            targets = Variable(returns)

            self.critic_optimizer.zero_grad()
            #print(values.data.size(),targets.data.size())
            value_loss = (values - targets).pow(2.).mean()
            value_loss.backward()
            self.critic_optimizer.step()

            action_var = Variable(actions)

            action_means, action_log_stds, action_stds = self.actor(Variable(states))
            log_prob_cur = self.normal_log_density(action_var, action_means, action_log_stds, action_stds)

            #action_means_old, action_log_stds_old, action_stds_old = self.actor(Variable(states), old=True)
            action_means_old, action_log_stds_old, action_stds_old = self.old_actor(Variable(states))
            
            log_prob_old = self.normal_log_density(action_var, action_means_old, action_log_stds_old, action_stds_old)

            # backup params after computing probs but before updating new params
            #policy_net.backup()
            #self.actor.backup()
            self.old_actor.load_state_dict(self.actor.state_dict())

            advantages = (advantages - advantages.mean()) / advantages.std()
            advantages_var = Variable(advantages)

            #opt_policy.zero_grad()
            self.actor_optimizer.zero_grad()
            ratio = torch.exp(log_prob_cur - log_prob_old) # pnew / pold
            surr1 = ratio * advantages_var[:,0]
            surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages_var[:,0]
            policy_surr = -torch.min(surr1, surr2).mean()
            policy_surr.backward()
            torch.nn.utils.clip_grad_norm(self.actor.parameters(), 40)
            self.actor_optimizer.step()

            UPDATE_EVENT.clear()
            ROLLING_EVENT.set()



    def select_action(self,state):
        state = torch.from_numpy(state).unsqueeze(0).cuda()
        action_mean, _, action_std = self.actor(Variable(state))
        action = torch.normal(action_mean, action_std)
        action = action.data[0].cpu().numpy()
        return action

    def get_value(self,state):
        state = torch.from_numpy(state).unsqueeze(0).cuda()
        #print(state.size())
        value = self.critic(Variable(state))
        return value.data[0].cpu().numpy()

    def play(self,env):

        timer = time.time()
        steps = 0
        total_reward = 0
        episode_memory = []
        buffer_s, buffer_a, buffer_r, buffer_ns, buffer_mask = [], [], [], [], []

        observation = env.reset()
        while True :
            if not ROLLING_EVENT.is_set():
                ROLLING_EVENT.wait()
                buffer_s, buffer_a, buffer_r, buffer_ns, buffer_mask = [], [], [], [], []

            steps += 1
            #print(steps)

            observation_before_action = observation

            action = self.select_action(observation_before_action)

            observation, reward, done, _info = env.step(action)

            mask = 1
            if done:
                mask = 0

            total_reward += reward

            buffer_s.append(observation_before_action)
            buffer_a.append(np.array([action]))
            buffer_r.append(reward)
            buffer_ns.append(observation)
            buffer_mask.append(mask)
            #memory.push(state, np.array([action]), mask, next_state, reward, discounted_reward)

            self.msize = self.msize + 1
            if done or self.msize >= self.batch_size:

                value = self.get_value(observation)
                discounted_r = []
                for r in buffer_r[::-1]:
                    value = r + value * self.GAMMA
                    discounted_r.append(value)
                discounted_r.reverse()

                bs, ba, bdr = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)#[:, np.newaxis]
                bmask, br, bns = np.vstack(buffer_mask), np.vstack(buffer_r), np.vstack(buffer_ns)
                buffer_s, buffer_a, buffer_r, buffer_ns, buffer_mask = [], [], [], [], []

                #print(bs.shape,ba.shape,bmask.shape,bns.shape,br.shape,bdr.shape)
                QUEUE.put(np.hstack((bs,ba,bmask,bns,br,bdr)))

                if self.msize >= self.batch_size:
                    ROLLING_EVENT.clear()
                    UPDATE_EVENT.set()
                    

            if done :
                break

        totaltime = time.time() - timer
        self.best = max(self.best,total_reward)
        print('episode done in {} steps in {:.2f} sec, {:.4f} sec/step, got reward :{:.2f} Best : {:.2f}'.format(
        steps,totaltime,totaltime/steps,total_reward,self.best
        ))

        self.epoch = self.epoch+1
        self.plot_epoch.append(self.epoch)
        self.plot_reward.append(total_reward)
        #epoch = range(0,3000)
        #rewards = range(0,6000,2)

        return

    def plot(self):
        fig = plt.figure(1)
        plt.plot(self.plot_epoch, self.plot_reward)
        plt.xlabel('epoch')
        plt.ylabel('score')
            #plt.show()

        fig.savefig(PATH_TO_MODEL+'/plot.pdf')

        '''
        if curr_noise is not None:
            disp_actions = (actions-self.actions_bias) / self.action_multiplier
            disp_actions = disp_actions * 5 + np.arange(self.outputdims) * 12.0 + 30

            noise = curr_noise * 5 - np.arange(self.outputdims) * 12.0 - 30
        '''
        return actions

    def save_weights(self,name):
        save_model({
                'actor': self.actor.state_dict(),
                'critic': self.critic.state_dict(),
                'actor_optimizer': self.actor_optimizer.state_dict(),
                'critic_optimizer' : self.critic_optimizer.state_dict(),
            },PATH_TO_MODEL,str(name))

    def load_weights(self,name):
        print("=> loading checkpoint ")
        #checkpoint = torch.load('../models/best.t7')
        checkpoint = torch.load(str(name)+'.t7')
        
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.actor_optimizer.state = defaultdict(dict, self.actor_optimizer.state)
        self.critic_optimizer.state = defaultdict(dict, self.critic_optimizer.state)
        print("done..")



if __name__ == '__main__':
    
    agent = nnagent(args)

    import farm
    Farm = farm.new_farm(args.num_processes)

    def save(name):
        agent.save_weights(name)
        #agent.rpm.save('rpm'+name+'.pickle')

    def load(name):
        agent.load_weights(name)
        #agent.rpm.load('rpm'+name+'.pickle')

    def playonce(remote_env):
        from multi import fastenv

        fenv = fastenv(remote_env,2)
        #print(type(agent))
        agent.play(fenv)
        remote_env.rel()
        del fenv

    def open_update():
        import threading as th
        t = th.Thread(target=agent.update)
        t.daemon = True
        t.start()

    def play_ignore(remote_env):
        import threading as th
        t = th.Thread(target=playonce,args=(remote_env,))
        t.daemon = True
        t.start()

    def playifavailable():
        while True:
            envid = Farm.acq(args.num_processes)
            if envid == False:
                time.sleep(0.01)
                pass
            else:
                remote_env = farm.new_remote_env(Farm,envid)
                play_ignore(remote_env)
                break
            
            

    def r(ep,times=1):

        for i in range(ep):
            #print np.random.uniform()
            # playtwice(times)
            print('epoch : ' + str(i) +' !!!')
            playifavailable()

            time.sleep(0.05)

            if (i+1) % 1000 == 0:
                # save the training result.
                # save(str(i+1))
                agent.plot()
                #return
    open_update()
    r(1000000)
