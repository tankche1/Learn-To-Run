import numpy as np
import numpy
from collections import defaultdict
from itertools import count
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from utils.replay_memory import ReplayMemory
from utils import plotting

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

def action_map(action):
                act_k = (1.0 - 0.0)/ 2.
                act_b = (1.0 + 0.0)/ 2.
                return act_k * action + act_b

def ddpg_learning(
    env,
    random_process,
    agent,
    num_episodes,
    gamma=1.0,
    log_every_n_eps=10,
    ):

    """The Deep Deterministic Policy Gradient algorithm.
    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    random_process: Defined in utils.random_process
        The process that add noise for exploration in deterministic policy.
    agent:
        a DDPG agent consists of a actor and critic.
    num_episodes:
        Number of episodes to run for.
    gamma: float
        Discount Factor
    log_every_n_eps: int
        Log and plot training info every n episodes.
    """
    ###############
    # RUN ENV     #
    ###############
    stats = plotting.EpisodeStats(
        episode_lengths=[],
        episode_rewards=[],
        mean_rewards=[])
    total_timestep = 0

    last_state = [1]*48

    for i_episode in range(num_episodes):
        state = env.reset(difficulty = 0)

        last_state = process_observation(state)
        state = process_observation(state)
        last_state ,state = transform_observation(last_state,state)
        state = numpy.array(state)

        random_process.reset_states()

        episode_reward = 0
        episode_length = 0
        for t in count(1):
            action = agent.select_action(state)\

            # Add noise for exploration
            noise = random_process.sample()[0]
            action += noise

            #print(noise)
            action = np.clip(action, -1.0, 1.0)
            action = action_map(action)


            #print(action.shape)
            #print(state.shape)
            reward = 0 
            next_state, A, done, _ = env.step(action)
            reward += A

            next_state = process_observation(next_state)
            last_state ,next_state = transform_observation(last_state,next_state)
            next_state = numpy.array(next_state)

            # Update statistics
            total_timestep += 1
            episode_reward += reward
            episode_length = t
            # Store transition in replay memory
            agent.replay_memory.push(state, action, reward, next_state, done)
            # Update
            agent.update(gamma)
            if done:
                stats.episode_lengths.append(episode_length)
                stats.episode_rewards.append(episode_reward)
                mean_reward = np.mean(stats.episode_rewards[-100:])
                stats.mean_rewards.append(mean_reward)
                break
            else:
                state = next_state

        if i_episode % 10 == 0:
            pass
            print("### EPISODE %d ### TAKES %d TIMESTEPS" % (i_episode + 1, stats.episode_lengths[i_episode]))
            print("MEAN REWARD (100 episodes): " + "%.3f" % (mean_reward))
            print("TOTAL TIMESTEPS SO FAR: %d" % (total_timestep))
            #plotting.plot_episode_stats(stats)

    return stats
