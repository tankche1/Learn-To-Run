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

def test(rank, params, shared_model, shared_obs_stats, test_n):
    PATH_TO_MODEL = '../models/'+params.bh
    torch.manual_seed(params.seed + rank)
    best_result = -1000
    work_dir = mkdir('exp', 'ppo')
    monitor_dir = mkdir(work_dir, 'monitor')
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
                            'state_dict': shared_model.state_dict(),
                            #'optimizer' : shared_obs_stats.state_dict(),
                        },PATH_TO_MODEL,'best')

            if epoch%100 == 1:
                save_model({
                            'epoch': epoch ,
                            'bh': params.bh,
                            'state_dict': shared_model.state_dict(),
                            #'optimizer' : shared_obs_stats.state_dict(),
                        },PATH_TO_MODEL,epoch)

            reward_sum = 0
            episode_length = 0
            state = env.reset(difficulty=0)
            state = numpy.array(state)
            time.sleep(10)

        state = Variable(torch.Tensor(state).unsqueeze(0))
