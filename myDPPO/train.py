# pip install http://download.pytorch.org/whl/cu75/torch-0.1.12.post2-cp27-none-linux_x86_64.whl
import argparse
import sys
import math
import os
from collections import namedtuple
from itertools import count
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
import random
import torch.multiprocessing as mp

pq = mp.Queue()

def listen():
    tot = []
    while(True):
        msg = pq.get()
        if msg[0]=='Time':
            tot.append(msg[1])
            if len(tot) == 16:
                sorted(tot)
                print('min: '+str(tot[0]) +' max: ' + str(tot[15]) + ' middle: ' +str(tot[7]))
                tot = []


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
        # discounted reward
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

    
    ratio = torch.exp(log_prob_cur - log_prob_old) #     pnew / pold
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

## 189
def process_observation(state1,state2,state3,observation,balls):

    o = list(observation) # an array
    l1 = list(state1)
    l2 = list(state2)
    l3 = list(state3)

    foot_touch_indicators = []
    fly0,fly1,fly2,fly3 = 1,1,1,1
    for i in [29,31,33,35]: # y of toes and taluses
        touch_ind = 1 if o[i] < 0.05 else 0
        touch_ind2 = 1 if o[i] < 0.1 else 0
        if o[i]<0.1:
            fly0 = 0
        if l1[i]<0.1:
            fly1 = 0
        if l2[i]<0.1:
            fly2 = 0
        if l3[i]<0.1:
            fly3 = 0
        
        foot_touch_indicators.append(touch_ind)
        foot_touch_indicators.append(touch_ind2)
    fly_air = [fly0]

    px = o[1]
    py = o[2]
    pvx = o[4]
    pvy = o[5]

    no_obstacle = [0]
    if px >6.0:
        no_obstacle = [1]

    o[18] -= px
    o[19] -= py

    o[20] -= pvx
    o[21] -= pvy

    for i in range(7):
        o[22+2*i] -= px
        o[22+2*i+1] -= py

    bodies = ['head', 'pelvis', 'torso', 'toes_l', 'toes_r', 'talus_l', 'talus_r']
    pelvis_pos = [o[0],o[1],o[2]]
    pelvis_vel = [o[3],o[4],o[5]]

    jnts = ['hip_r','knee_r','ankle_r','hip_l','knee_l','ankle_l']
    joint_angles = [o[6],o[7],o[8],o[9],o[10],o[11]]
    joint_vel = [o[12],o[13],o[14],o[15],o[16],o[17]]

    mass_pos = [o[18],o[19]]
    mass_vel = [o[20],o[21]]

    bodypart_pos = [o[22],o[23],o[24],o[25],o[26],o[27],o[28],o[29],o[30],o[31],o[32],o[33],o[34],o[35]]
    muscles = [o[36],o[37]]
    obstacle = [o[38],o[39],o[40]]

    v = [0]*14 # past 3 frames bodypart v

    for i in range(14):
        v[i] = (o[22+i] - l1[22+i])#*100.0

    '''
    for i in range(14):
        v[14+i] = (l1[22+i]-l2[22+i])

    for i in range(14):
        v[28+i] = (l2[22+i]-l3[22+i])
    '''

    pelvis_past = [px-l1[0],py-l1[1],l1[2],l1[3],l1[4],l1[5],px-l2[0],py-l2[1],l2[2],l2[3],l2[4],l2[5]] 

    av = [0]*17 # past 2 frames bodypart av 
    #pelvis av
    for i in range(3):
        av[i] = (o[3+i] - l1[3+i])
    '''
    for i in range(3):
        av[3+i] = (l1[3+i] - l2[3+i])
    '''
    for i in range(14):
        av[3+i] = (o[22+i] - l1[22+i]) - (l1[22+i] - l2[22+i])
    '''
    for i in range(14):
        av[20+i] = (l1[22+i] - l2[22+i]) - (l2[22+i] - l3[22+i])
    '''

    action_past = [l1[i] for i in range(41,59)] + [l2[i] for i in range(41,59)]
    reward_past = [l1[59],l2[59],l3[59]]
    #av = av*100
    

    current_pelvis = o[1]
    current_ball_relative = o[38]
    current_ball_height = o[39]       
    current_ball_radius = o[40]
    absolute_ball_pos = current_ball_relative + current_pelvis

    ball_vectors = []
    for i in range(3):
        if i<len(balls):
            rel = balls[i][0] - current_pelvis
            falloff = 0
            if abs(rel) < 3:
                falloff = 1
            #falloff = min(1,max(0,3-abs(rel))) # when ball is closer than 3 falloff become 1
            ball_vectors.append(min(4,max(-3, rel))/3) # ball pos relative to current pos
            ball_vectors.append(balls[i][1] * falloff) # radius
            ball_vectors.append(balls[i][2] * falloff) # height
        else:
            ball_vectors.append(0)
            ball_vectors.append(0)
            ball_vectors.append(0)
    #print(len(o),len(v),len(av))

    # 41 + 42 + 34 +12 + 36 + 3 + 8 + 4 + 9 = 189
    #return o,l1,l2,o + v + av + pelvis_past + action_past + reward_past + foot_touch_indicators + fly_air + ball_vectors 
    # 41 + 14 + 17  + 8 + 1 + 9 + 1= 91
    return o,l1,l2,o + v + av   + foot_touch_indicators + fly_air + ball_vectors + no_obstacle

def addball_if_new(new,balls):
    current_pelvis = new[1]
    current_ball_relative = new[38]
    current_ball_height = new[39]       
    current_ball_radius = new[40]

    absolute_ball_pos = current_ball_relative + current_pelvis

    if current_ball_radius == 0: # no balls ahead
       return

    compare_result = [abs(b[0] - absolute_ball_pos) < 1e-9 for b in balls]
    # [False, False, False, False] if is different ball

    got_new = sum([(1 if r==True else 0)for r in compare_result]) == 0

    if got_new:
        # for every ball there is
        '''
        for b in balls:
            # if this new ball is smaller in x than any ball there is
            if absolute_ball_pos < (b[0] - 1e-9):
                print(absolute_ball_pos,balls)
                print('(@ step )'+')Damn! new ball closer than existing balls.')
                #q.dump(reason='ballcloser')
                raise Exception('new ball closer than the old ones.')
        '''

        balls.append([
            absolute_ball_pos,
            current_ball_height,
            current_ball_radius,
        ])
        if len(balls)>3:
            print(balls)
            print('(@ step '+')What the fuck you just did! Why num of balls became greater than 3!!!')
            #q.dump(reason='ballgt3')
            raise Exception('ball number greater than 3.')
    else:
        pass # we already met this ball before.       

def train(rank,args,traffic_light, counter, shared_model, shared_grad_buffers, shared_obs_stats ,opt_ac):
    best_result =-1000 
    torch.manual_seed(args.seed+rank)
    torch.set_default_tensor_type('torch.DoubleTensor')
    num_inputs = args.feature
    num_actions = 18
    plot_epoch = []
    plot_reward = []
    #last_state = numpy.zeros(48)

    env = RunEnv(visualize=False)
    balls = []

    #running_state = ZFilter((num_inputs,), clip=5)
    #running_reward = ZFilter((1,), demean=False, clip=10)
    episode_lengths = []

    PATH_TO_MODEL = '../models/'+str(args.bh)

    ac_net = ActorCritic(num_inputs, num_actions)

    #running_state = ZFilter((num_inputs,), clip=5)

    start_time = time.time()

    for i_episode in range(args.start_epoch+1,999999):
        #print(shared_obs_stats.n[0])
        #print('hei')
        #if rank == 0:
        #    print(running_state.rs._n)

        epoch_start_time = time.time()
        
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
            state1,state2,state3,state = [0]*60, [0]*60, [0]*60, [0]*60
            #state = env.reset(difficulty = 2,seed = (random.randint(0, 10000)))
            #if (i_episode+rank)%3 == 0:
            #    state = env.reset(difficulty = 0)
            state = env.reset(difficulty = args.dif)
            balls = []
            tot_frame = 0
            #state = numpy.array(state)


            #print(len(state1),len(state2),len(state3),len(state))
            state1, state2, state3, state=process_observation(state1, state2, state3, state,balls)

            #print(state)
            #print(type(state))
            #print(len(state))
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
                tot_frame = tot_frame + 1



                reward += A

                state1 = state1 + list(action) + [reward]
                #print(next_state)
                #last_state = process_observation(state)
                addball_if_new(next_state,balls)
                #print(len(state1),len(state2),len(state3),len(next_state))
                state1, state2, state3, next_state=process_observation(state1, state2, state3, next_state,balls)
                #print(next_state)

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
        epoch = i_episode

        if (reward_batch > best_result) and (rank == 0):
            best_result = reward_batch
            save_model({
                    'epoch': epoch ,
                    'bh': args.bh,
                    'state_dict': shared_model.state_dict(),
                    'optimizer' : opt_ac.state_dict(),
                    'obs' : shared_obs_stats,
                },PATH_TO_MODEL,'best')
        
        
        #print('env:')
        #print(time.time()-timer)

        #timer = time.time()
        update_params_actor_critic(batch,args,ac_net,opt_ac)
        shared_grad_buffers.add_gradient(ac_net)

        counter.increment()

        pq.put(('Time',time.time()-epoch_start_time))
        if (i_episode % args.log_interval == 0) and (rank == 0):

            print('TrainEpisode {}\tTime{}\tBest reward: {}\tAverage reward {:.2f} frames {}'.format(
                i_episode,
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                best_result, reward_batch,tot_frame))

            plot_epoch.append(i_episode)
            plot_reward.append(reward_batch)
            #epoch = range(0,3000)
            #rewards = range(0,6000,2)
            
            fig = plt.figure(1)
            plt.plot(plot_epoch, plot_reward)
            plt.xlabel('epoch')
            plt.ylabel('score')
            #plt.show()

            fig.savefig(PATH_TO_MODEL+'/plot.pdf')

            epoch = i_episode
            
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
