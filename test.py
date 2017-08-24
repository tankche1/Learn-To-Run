'''
from osim.env import RunEnv

env = RunEnv(visualize=False)
observation = env.reset(difficulty = 0)
for i in range(200):
    observation, reward, done, info = env.step(env.action_space.sample())
    print(type(observation))
    print(len(observation))
    print(type(reward))
    print(reward)
    print(type(done))
    print(done)
    print(type(info))
    print(info)
    print(type(env.action_space.sample()))
    print(env.action_space.sample())
    print('-----------------------')
'''
import numpy
A = [0.1]*60
print(A)