import numpy as np

from math import *
import random
import time
import argparse

from rpm import rpm # replay memory implementation

from noise import one_fsq_noise

from observation_processor import process_observation as po
from observation_processor import generate_observation as go
import models

import copy

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from collections import defaultdict

torch.set_default_tensor_type('torch.FloatTensor')

parser = argparse.ArgumentParser(description='PyTorch DDPG')
parser.add_argument('--gamma', type=float, default=0.985, metavar='G',
                    help='discount factor (default: 0.985)')
parser.add_argument('--lr', type=float, default=1e-3, 
                    help='learning rate (default: 1e-3)')
parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                    help='gae (default: 0.97)')
parser.add_argument('--seed', type=int, default=543, 
                    help='random seed (default: 1)')
parser.add_argument('--bh',default='origin',
                        help='bh')
parser.add_argument('--resume', action='store_true',
                    help='loading the model')
parser.add_argument('--num-processes', type=int, default=1, 
                    help='how many training processes to use (default: 4)')
parser.add_argument('--observation_space_dims', type=int, default=90, 
                    help='observation_space_dims')
parser.add_argument('--action_space', type=int, default=18, 
                    help='action_space')
parser.add_argument('--dif', type=int, default=0, 
                    help='difficulty')
parser.add_argument('--train_multiplier', type=int, default=1, 
                    help='train_multiplier')

'''
python: ../nptl/pthread_mutex_lock.c:117: __pthread_mutex_lock: Assertion `mutex->__data.__owner == 0' failed.
'''

args = parser.parse_args()
PATH_TO_MODEL = '../models/'+str(args.bh)
from observation_processor import processed_dims
args.observation_space_dims = processed_dims

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    ex = np.exp(x)
    return ex / np.sum(ex, axis=0)

def save_model(model,PATH_TO_MODEL,epoch):
    print('saving the model ...')
    if not os.path.exists(PATH_TO_MODEL):
        os.mkdir(PATH_TO_MODEL)

    torch.save(model,PATH_TO_MODEL+'/'+str(epoch)+'.t7')
    print('done.')

class nnagent(object):
    def __init__(self,args):
        self.rpm = rpm(1000000)
        self.render = True
        self.training = True
        self.noise_source = one_fsq_noise()

        self.train_multiplier = args.train_multiplier
        self.inputdims = args.observation_space_dims

        low = 0.0
        high = 1.0
        num_of_actions = args.action_space
        self.action_bias = high/2.0 + low/2.0
        self.action_multiplier = high - self.action_bias

        def clamper(actions):
            return np.clip(actions,a_max=high,a_min=low)

        self.clamper = clamper

        self.outputdims = args.action_space
        self.discount_factor = args.gamma
        ids, ods = self.inputdims, self.outputdims
        print('inputdims:{}, outputdims:{}'.format(ids,ods))

        self.actor = models.create_actor_network(ids,ods).cuda()
        self.critic = models.create_critic_network(ids,ods).cuda()
        self.actor_target = models.create_actor_network(ids,ods).cuda()
        self.critic_target = models.create_critic_network(ids,ods).cuda()
        self.critic_criterion = nn.MSELoss().cuda()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

        self.plot_epoch = [0]
        self.plot_reward = [0]
        import threading as th
        self.lock = th.Lock()

        # print(self.actor.get_weights())
        # print(self.critic.get_weights())

        #self.feed,self.joint_inference,sync_target = self.train_step_gen()

        #sess = ct.get_session()
        #sess.run(tf.global_variables_initializer())

        #sync_target()

    def update_target(self, target_model, model):
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def mse_loss(input, target):
        return torch.sum((input - target)^2) / input.data.nelement()

    def update(self,tup):
        [s1,a1,r1,isdone,s2] = tup

        s1 = Variable(torch.FloatTensor(s1).cuda(),requires_grad = True)
        a1 = Variable(torch.FloatTensor(a1).cuda(),requires_grad = True)
        s2 = Variable(torch.FloatTensor(s2).cuda(),requires_grad = True)
        

        a2 = self.actor_target(s2)
        q2 = self.critic_target([s2,a2])
        #print(type(r1))
        #print(type(isdone))
        #print(float(r1))
        #print(type(q2))

        #return 
        q1_target = Variable(torch.FloatTensor(r1).cuda()) + Variable(torch.FloatTensor(1-isdone).cuda())*self.discount_factor*q2
        q1_target = Variable(q1_target.data,requires_grad = False)
        #q1_target.requires_grad = False
        #print(q1_target.data)
        q1_predict = self.critic([s1,a1])
        #print(q1_target.data.size(),q1_predict.data.size())
        #critic_loss = F.mse_loss(q1_target,q1_predict)
        critic_loss = self.critic_criterion(q1_predict,q1_target)
        #print(type(critic_loss))

        


        a1_predict = self.actor(s1)
        q1_predict = self.critic_target([s1,a1_predict])
        actor_loss = -q1_predict.mean()

       
        self.lock.acquire()
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
         # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.lock.release()

        self.tau = 1e-3
        # Update the target networks
        self.update_target(self.critic_target, self.critic)
        self.update_target(self.actor_target, self.actor)

    def train(self):
        memory = self.rpm
        batch_size = 64
        total_size = batch_size
        epochs = 1

        if memory.size() > total_size*128:
        #if memory.size() > 128:
            for i in range(self.train_multiplier):
                [s1,a1,r1,isdone,s2] = memory.sample_batch(batch_size)
                #self.lock.acquire()
                self.update([s1,a1,r1,isdone,s2]) 
                #self.lock.release()

    def feed_one(self,tup):
        self.rpm.add(tup)

    def play(self,env,max_steps=-1,realtime=False,noise_Level=0.):
        timer = time.time()
        noise_source = one_fsq_noise()

        for j in range(200):
            noise_source.one((self.outputdims,),noise_Level)

        max_steps = max_steps if max_steps > 0 else 50000
        steps = 0
        total_reward = 0
        episode_memory = []

        observation = env.reset()
        while True and steps <= max_steps:
            steps += 1

            observation_before_action = observation

            exploration_noise = noise_source.one((self.outputdims,),noise_Level)

            action = self.act(observation_before_action, exploration_noise)

            exploration_noise *= self.action_multiplier
            action = self.clamper(action)
            action_out = action

            observation, reward, done, _info = env.step(action_out)

            isdone = 1 if done else 0 
            total_reward += reward

            if self.training == True:
                episode_memory.append((
                    observation_before_action,action,reward,isdone,observation
                ))

                self.train()

            if done :
                break

        totaltime = time.time() - timer
        print('episode done in {} steps in {:.2f} sec, {:.4f} sec/step, got reward :{:.2f}'.format(
        steps,totaltime,totaltime/steps,total_reward
        ))

        self.plot_epoch.append(self.plot_epoch[-1]+1)
        self.plot_reward.append(total_reward)
        #epoch = range(0,3000)
        #rewards = range(0,6000,2)
            
        
        self.lock.acquire()

        for t in episode_memory:
            self.feed_one(t)

        #self.plotter.pushys([total_reward,noise_level,(time.time()%3600)/3600-2])
        # self.noiseplotter.pushy(noise_level)
        self.lock.release()

        return

    def plot(self):
        fig = plt.figure(1)
        plt.plot(self.plot_epoch, self.plot_reward)
        plt.xlabel('epoch')
        plt.ylabel('score')
            #plt.show()

        fig.savefig(PATH_TO_MODEL+'/plot.pdf')

    def act(self,observation,curr_noise=None):
        actor, critic = self.actor,self.critic
        obs = np.reshape(observation,(1,len(observation)))

        obs = Variable(torch.FloatTensor(obs).cuda())

        actions = self.actor(obs)

        actions = actions.data[0].cpu().numpy()


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
                'actor_target': self.actor_target.state_dict(),
                'critic_target': self.critic_target.state_dict(),
                'actor_optimizer': self.actor_optimizer.state_dict(),
                'critic_optimizer' : self.critic_optimizer.state_dict(),
            },PATH_TO_MODEL,str(name))

    def load_weights(self,name):
        print("=> loading checkpoint ")
        #checkpoint = torch.load('../models/best.t7')
        checkpoint = torch.load(str(name)+'.t7')
        
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.actor_optimizer.state = defaultdict(dict, self.actor_optimizer.state)
        self.critic_optimizer.state = defaultdict(dict, self.critic_optimizer.state)
        print("done..")

from osim.env import RunEnv


if __name__ == '__main__':

    agent = nnagent(args)

    if args.resume:
        agent.load_weights('../models/40coreddpg/3000')
    
    noise_level = 2.
    noise_decay_rate = 0.001
    noise_floor = 0.05
    noiseless = 0.01

    import farm
    Farm = farm.new_farm(args.num_processes)

    def save(name):
        agent.save_weights(name)
        agent.rpm.save('rpm'+name+'.pickle')

    def load(name):
        agent.load_weights(name)
        agent.rpm.load('rpm'+name+'.pickle')

    def playonce(nl,remote_env):
        from multi import fastenv

        fenv = fastenv(remote_env,2)
        #print(type(agent))
        agent.play(fenv,realtime=False,max_steps=-1,noise_Level=nl)
        remote_env.rel()
        del fenv


    def play_ignore(nl,remote_env):
        import threading as th
        t = th.Thread(target=playonce,args=(nl,remote_env))
        t.daemon = True
        t.start()

    def playifavailable(nl):
        while True:
            envid = Farm.acq(args.num_processes)
            if envid == False:
                time.sleep(0.01)
                pass
            else:
                remote_env = farm.new_remote_env(Farm,envid)
                play_ignore(nl,remote_env)
                break
            
            

    def r(ep,times=1):
        global noise_level,noiseless

        for i in range(ep):

            noise_level *= (1-noise_decay_rate)
            noise_level = max(noise_floor, noise_level)
            
            #print np.random.uniform()
            nl = noise_level if np.random.uniform()>0.05 else noiseless

            print('ep',i+1,'/',ep,'times:',times,'noise_level',nl)
            # playtwice(times)
            playifavailable(nl)

            time.sleep(0.05)

            if (i+1) % 1000 == 0:
                # save the training result.
                save(str(i+1))
                agent.plot()
                #return


    r(100000)
