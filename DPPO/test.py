import os
import sys
import time
from collections import deque
import numpy as np
import numpy

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from osim.env import RunEnv

from model import Model

def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def save_model(model,PATH_TO_MODEL,epoch):
    print('saving the model ...')
    if not os.path.exists(PATH_TO_MODEL):
        os.mkdir(PATH_TO_MODEL)

    torch.save(model,PATH_TO_MODEL+'/'+str(epoch)+'.t7')
    print('done.')


# 41 to 41+11+14=66
def process_observation(last_state,observation):

    o = list(observation) # an array
    l = list(last_state)

    px = o[1]
    py = o[2]
    pvx = o[4]
    pvy = o[5]

    o[18] -= px
    o[19] -= py

    o[20] -= pvx
    o[21] -= pvy

    for i in range(7):
        o[22+2*i] -= px
        o[22+2*i+1] -= py

    av = [0]*11
    v = [0]*14
    if len(l)>0:
        for i in range(3):
            av[i] = (o[3+i] - l[3+i])*100
        for i in range(6):
            av[3+i] = (o[12+i] - l[12+i])*100
        av[9] = (o[20] - l[20])*100
        av[10] = (o[21] - l[21])*100

        for i in range(14):
            v[i] = o[22+i] - l[22+i]

    #av = av*100

    #print(len(o),len(v),len(av))

    return o,o + v + av


def test(rank, params, shared_model, shared_obs_stats, test_n):
    PATH_TO_MODEL = '../models/'+params.bh
    torch.manual_seed(params.seed + rank)
    best_result = -1000
    work_dir = mkdir('exp', 'ppo')
    monitor_dir = mkdir(work_dir, 'monitor')
    last_state = []
    #env = gym.make(params.env_name)
    if params.render:
        env = RunEnv(visualize=True)
    else:
        env = RunEnv(visualize=False)
    #env = wrappers.Monitor(env, monitor_dir, force=True)
    #num_inputs = env.observation_space.shape[0]
    #num_outputs = env.action_space.shape[0]
    num_inputs = params.num_inputs
    num_outputs = params.num_outputs
    model = Model(num_inputs, num_outputs)

    #state = env.reset()
    state = env.reset(difficulty=0)

    last_state ,state = process_observation(last_state,state)
    state = numpy.array(state)

    state = Variable(torch.Tensor(state).unsqueeze(0))
    reward_sum = 0
    done = True

    start_time = time.time()

    episode_length = 0
    epoch = 0
    while True:
        #print(episode_length)
        episode_length += 1
        model.load_state_dict(shared_model.state_dict())
        shared_obs_stats.observes(state)
        #print(shared_obs_stats.n[0])
        state = shared_obs_stats.normalize(state)
        mu,sigma_sq,_ = model(state)
        eps = torch.randn(mu.size())
        action = mu + sigma_sq.sqrt()*Variable(eps)
        env_action = action.data.squeeze().numpy()
        state, reward, done, _ = env.step(env_action)

        last_state ,state = process_observation(last_state,state)
        state = numpy.array(state)

        reward_sum += reward

        if done:
            print("Time {}, epoch {} ,episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                epoch,reward_sum, episode_length))
            epoch = epoch + 1
            if reward_sum > best_result:
                best_result = reward_sum

                save_model({
                            'epoch': epoch ,
                            'bh': params.bh,
                            'state_dict': model.state_dict(),
                            #'optimizer' : shared_obs_stats.state_dict(),
                        },PATH_TO_MODEL,'best')

            if epoch%100 == 1:
                save_model({
                            'epoch': epoch ,
                            'bh': params.bh,
                            'state_dict': model.state_dict(),
                            #'optimizer' : shared_obs_stats.state_dict(),
                        },PATH_TO_MODEL,epoch)

            reward_sum = 0
            episode_length = 0
            state = env.reset(difficulty=0)

            last_state = []
            last_state ,state = process_observation(last_state,state)
            state = numpy.array(state)
            time.sleep(10)

        state = Variable(torch.Tensor(state).unsqueeze(0))
