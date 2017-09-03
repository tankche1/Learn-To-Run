# pip install http://download.pytorch.org/whl/cu75/torch-0.1.12.post2-cp27-none-linux_x86_64.whl
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

from models import Policy, Value, ActorCritic
from replay_memory import Memory
from running_state import ZFilter

from osim.env import RunEnv
import math
import time
from models import Shared_grad_buffers

# from utils import *

PI = torch.DoubleTensor([3.1415926])

def save_model(model,PATH_TO_MODEL,epoch):
    print('saving the model ...')
    if not os.path.exists(PATH_TO_MODEL):
        os.mkdir(PATH_TO_MODEL)

    torch.save(model,PATH_TO_MODEL+'/'+str(epoch)+'.t7')
    print('done.')


def select_action(state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action

def select_action_actor_critic(state,ac_net):
    state = torch.from_numpy(state).unsqueeze(0)
    #print(state)
    action_mean, _, action_std, v = ac_net(Variable(state))

    action = torch.normal(action_mean, action_std)
    #print(action_mean,action_std)

    #action = torch.clamp(action,min= 0.0,max = 1.0)
    return action

def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (
        2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1)

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            #print('FUCK')
            return
        shared_param._grad = param.grad

def update_params_actor_critic(batch,args,ac_net,opt_ac):
    ac_net.zero_grad()
    rewards = torch.Tensor(batch.reward)
    masks = torch.Tensor(batch.mask)
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(batch.state)
    action_means, action_log_stds, action_stds, values = ac_net(Variable(states))

    returns = torch.Tensor(actions.size(0),1)
    deltas = torch.Tensor(actions.size(0),1)
    advantages = torch.Tensor(actions.size(0),1)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + args.gamma * args.tau * prev_advantage * masks[i]
        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    targets = Variable(returns)

    # kloldnew = policy_net.kl_old_new() # oldpi.pd.kl(pi.pd)
    # ent = policy_net.entropy() #pi.pd.entropy()
    # meankl = torch.reduce_mean(kloldnew)
    # meanent = torch.reduce_mean(ent)
    # pol_entpen = (-args.entropy_coeff) * meanent

    action_var = Variable(actions)
    # compute probs from actions above
    log_prob_cur = normal_log_density(action_var, action_means, action_log_stds, action_stds)

    action_means_old, action_log_stds_old, action_stds_old, values_old = ac_net(Variable(states), old=True)
    log_prob_old = normal_log_density(action_var, action_means_old, action_log_stds_old, action_stds_old)

    # backup params after computing probs but before updating new params
    ac_net.backup()

    advantages = (advantages - advantages.mean()) / advantages.std()
    advantages_var = Variable(advantages)

    
    ratio = torch.exp(log_prob_cur - log_prob_old) # pnew / pold
    surr1 = ratio * advantages_var[:,0]
    surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages_var[:,0]
    policy_surr = -torch.min(surr1, surr2).mean()

    vf_loss1 = (values - targets).pow(2.)
    vpredclipped = values_old + torch.clamp(values - values_old, -args.clip_epsilon, args.clip_epsilon)
    vf_loss2 = (vpredclipped - targets).pow(2.)
    vf_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()

    #opt_ac.zero_grad()

    total_loss = policy_surr + vf_loss
    total_loss.backward(retain_variables=True)
    torch.nn.utils.clip_grad_norm(ac_net.parameters(), 40)

    #ensure_shared_grads(ac_net, shared_model)
    #opt_ac.step()

def process_observation(observation):
    o = list(observation) # an array

    pr = o[0]
    o[0]/=4

    px = o[1]
    py = o[2]

    pvr = o[3]
    o[3] /=4
    pvx = o[4]
    pvy = o[5]

    for i in range(6,18):
        o[i]/=4

    o = o + [o[22+i*2+1]-0.5 for i in range(7)] # a copy of original y, not relative y.

    # x and y relative to pelvis
    for i in range(7): # head pelvis torso, toes and taluses
        o[22+i*2+0] -= px
        o[22+i*2+1] -= py

    o[18] -= px # mass pos xy made relative
    o[19] -= py
    o[20] -= pvx
    o[21] -= pvy

    o[38]= min(4,o[38])/3 # ball info are included later in the stage
    # o[39]/=5
    # o[40]/=5

    o[1]=0 # abs value of pel x is not relevant
    o[2]-= 0.5

    o[4]/=2
    o[5]/=2

    return o

def transform_observation(last_state,observation):
    last_state = [(observation[i] - last_state[i])/0.01 for i in range(0,48)]
    #print(len(observation))
    #print(len(last_state))
    return observation,observation + last_state

def train(rank,args,traffic_light, counter, shared_model, shared_grad_buffers, shared_obs_stats ,opt_ac):
    best_result =-1000 
    torch.manual_seed(args.seed+rank)
    torch.set_default_tensor_type('torch.DoubleTensor')
    num_inputs = args.feature
    num_actions = 18
    last_state = [1]*48
    #last_state = numpy.zeros(48)

    env = RunEnv(visualize=False)

    #running_state = ZFilter((num_inputs,), clip=5)
    #running_reward = ZFilter((1,), demean=False, clip=10)
    episode_lengths = []

    PATH_TO_MODEL = '../models/'+str(args.bh)

    ac_net = ActorCritic(num_inputs, num_actions)

    #running_state = ZFilter((num_inputs,), clip=5)

    start_time = time.time()

    for i_episode in count(1):
        #print(shared_obs_stats.n[0])
        #print('hei')
        #if rank == 0:
        #    print(running_state.rs._n)

        signal_init = traffic_light.get()
        memory = Memory()
        ac_net.load_state_dict(shared_model.state_dict())

        num_steps = 0
        reward_batch = 0
        num_episodes = 0
        #Tot_loss = 0
        #Tot_num =
        while num_steps < args.batch_size:
            #state = env.reset()
            #print(num_steps)
            state = env.reset(difficulty = 0)
            #state = numpy.array(state)

            last_state = process_observation(state)
            state = process_observation(state)
            last_state ,state = transform_observation(last_state,state)

            state = numpy.array(state)

            #state = running_state(state)

            state = Variable(torch.Tensor(state).unsqueeze(0))
            shared_obs_stats.observes(state)
            state = shared_obs_stats.normalize(state)
            state = state.data[0].numpy()

            #print(state)
            #return

            #print(AA)

            #print(type(AA))
            #print(type(state))
            #print(AA.shape)
            #print(state.shape)

            reward_sum = 0
            #timer = time.time()
            for t in range(10000): # Don't infinite loop while learning
                #print(t)
                if args.use_sep_pol_val:
                    action = select_action(state)
                else:
                    action = select_action_actor_critic(state,ac_net)
                #print(action)
                action = action.data[0].numpy()
                if numpy.any(numpy.isnan(action)):
                    print(state)
                    print(action)
                    print(ac_net.affine1.weight)
                    print(ac_net.affine1.weight.data)
                    print('ERROR')
                    #action = select_action_actor_critic(state,ac_net)
                    #action = action.data[0].numpy()
                    #state = state + numpy.random.rand(args.feature)*0.001

                    raise RuntimeError('action NaN problem')
                #print(action)
                #print("------------------------")
                #timer = time.time()
                reward = 0
                if args.skip:
                    #env.step(action)
                    _,A,_,_ = env.step(action)
                    reward += A
                    _,A,_,_ = env.step(action)
                    reward += A
                next_state, A, done, _ = env.step(action)
                reward += A
                #print(next_state)
                #last_state = process_observation(state)
                next_state = process_observation(next_state)
                last_state ,next_state = transform_observation(last_state,next_state)

                next_state = numpy.array(next_state)
                #print(next_state)
                #print(next_state.shape)
                #return
                reward_sum += reward
                #print('env:')
                #print(time.time()-timer)

                #last_state ,next_state = update_observation(last_state,next_state)

                #next_state = running_state(next_state)

                next_state = Variable(torch.Tensor(next_state).unsqueeze(0))
                shared_obs_stats.observes(next_state)
                next_state = shared_obs_stats.normalize(next_state)
                next_state = next_state.data[0].numpy()

                #print(next_state[41:82])

                mask = 1
                if done:
                    mask = 0

                memory.push(state, np.array([action]), mask, next_state, reward)

                #if args.render:
                #    env.render()
                if done:
                    break

                state = next_state
            num_steps += (t-1)
            num_episodes += 1

            reward_batch += reward_sum


        reward_batch /= num_episodes
        batch = memory.sample()
        
        #print('env:')
        #print(time.time()-timer)

        #timer = time.time()
        update_params_actor_critic(batch,args,ac_net,opt_ac)
        shared_grad_buffers.add_gradient(ac_net)

        counter.increment()

        epoch = i_episode
        if (i_episode % args.log_interval == 0) and (rank == 0):

            print('TrainEpisode {}\tLast reward: {}\tAverage reward {:.2f}'.format(
                i_episode, reward_sum, reward_batch))

            epoch = i_episode
            if reward_batch > best_result:
                best_result = reward_batch
                save_model({
                        'epoch': epoch ,
                        'bh': args.bh,
                        'state_dict': shared_model.state_dict(),
                        'optimizer' : opt_ac.state_dict(),
                        'obs' : shared_obs_stats,
                    },PATH_TO_MODEL,'best')

            if epoch%30==1:
                save_model({
                        'epoch': epoch ,
                        'bh': args.bh,
                        'state_dict': shared_model.state_dict(),
                        'optimizer' : opt_ac.state_dict(),
                        'obs' :shared_obs_stats,
                    },PATH_TO_MODEL,epoch)
        # wait for a new signal to continue
        while traffic_light.get() == signal_init:
            pass

        
        
def test(rank, args,shared_model, shared_obs_stats, opt_ac):
    best_result =-1000 
    torch.manual_seed(args.seed+rank)
    torch.set_default_tensor_type('torch.DoubleTensor')
    num_inputs = args.feature
    num_actions = 18
    #last_state = numpy.zeros(41)

    if args.render:
        env = RunEnv(visualize=True)
    else:
        env = RunEnv(visualize=False)

    running_state = ZFilter((num_inputs,), clip=5)
    #running_reward = ZFilter((1,), demean=False, clip=10)
    episode_lengths = []

    PATH_TO_MODEL = '../models/'+str(args.bh)

    if not os.path.exists(PATH_TO_MODEL):
        os.mkdir(PATH_TO_MODEL)

    ac_net = ActorCritic(num_inputs, num_actions)

    start_time = time.time()

    for i_episode in count(1):
        memory = Memory()
        ac_net.load_state_dict(shared_model.state_dict())

        num_steps = 0
        reward_batch = 0
        num_episodes = 0
        while num_steps < args.batch_size:
            #state = env.reset()
            #print(num_steps)
            state = env.reset(difficulty = 0)
            state = numpy.array(state)
            #global last_state
            #last_state = state
            #last_state,_ = update_observation(last_state,state)
            #last_state,state = update_observation(last_state,state)
            #print(state.shape[0])
            #print(state[41])
            state = running_state(state)
            #state = Variable(torch.Tensor(state).unsqueeze(0))
            #shared_obs_stats.observes(state)
            #state = shared_obs_stats.normalize(state)
            #state = state.data[0].numpy()

            reward_sum = 0
            for t in range(10000): # Don't infinite loop while learning
                #print(t)
                #timer = time.time()
                if args.use_sep_pol_val:
                    action = select_action(state)
                else:
                    action = select_action_actor_critic(state,ac_net)

                #print(action)
                action = action.data[0].numpy()
                if numpy.any(numpy.isnan(action)):
                    print(action)
                    puts('ERROR')
                    return
                #print('NN take:')
                #print(time.time()-timer)
                #print(action)
                #print("------------------------")

                #timer = time.time()
                if args.skip:
                    #env.step(action)
                    _,reward,_,_ = env.step(action)
                    reward_sum += reward
                next_state, reward, done, _ = env.step(action)
                next_state = numpy.array(next_state)
                reward_sum += reward

                #print('env take:')
                #print(time.time()-timer)

                #timer = time.time()

                #last_state ,next_state = update_observation(last_state,next_state)
                next_state = running_state(next_state)
                #next_state = Variable(torch.Tensor(next_state).unsqueeze(0))
                #shared_obs_stats.observes(next_state)
                #next_state = shared_obs_stats.normalize(next_state)
                #next_state = next_state.data[0].numpy()
                #print(next_state[41:82])

                mask = 1
                if done:
                    mask = 0

                #print('update take:')
                #print(time.time()-timer)

                #timer = time.time()

                memory.push(state, np.array([action]), mask, next_state, reward)

                #print('memory take:')
                #print(time.time()-timer)

                #if args.render:
                #    env.render()
                if done:
                    break

                state = next_state
                
            num_steps += (t-1)
            num_episodes += 1
            #print(num_episodes)
            reward_batch += reward_sum

        #print(num_episodes)
        reward_batch /= num_episodes
        batch = memory.sample()
        
        #update_params_actor_critic(batch,args,shared_model,ac_net,opt_ac)
        time.sleep(60)

        if i_episode % args.log_interval == 0:
            File = open(PATH_TO_MODEL + '/record.txt', 'a+')
            File.write("Time {}, episode reward {}, Average reward {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                reward_sum, reward_batch))
            File.close()
            #print('TestEpisode {}\tLast reward: {}\tAverage reward {:.2f}'.format(
            #    i_episode, reward_sum, reward_batch))
            print("Time {}, episode reward {}, Average reward {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                reward_sum, reward_batch))
            #print('!!!!')

        epoch = i_episode
        if reward_batch > best_result:
            best_result = reward_batch
            save_model({
                    'epoch': epoch ,
                    'bh': args.bh,
                    'state_dict': shared_model.state_dict(),
                    'optimizer' : opt_ac.state_dict(),
                    #'obs' : shared_obs_stats
                },PATH_TO_MODEL,'best')

        if epoch%30==1:
            save_model({
                    'epoch': epoch ,
                    'bh': args.bh,
                    'state_dict': shared_model.state_dict(),
                    'optimizer' : opt_ac.state_dict(),
                    #'obs' :shared_obs_stats
                },PATH_TO_MODEL,epoch)

