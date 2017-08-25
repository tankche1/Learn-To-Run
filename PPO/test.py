'''
from osim.env import RunEnv
import scipy

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
import torch
import math
from torch.autograd import Variable
torch.set_default_tensor_type('torch.DoubleTensor')
x = torch.ones(212,18)
mean = torch.ones(212,18)
var = torch.ones(212,18)
log_std = torch.ones(212,18)
PI = torch.DoubleTensor([3.1415926])
x = Variable(x)
mean = Variable(mean)
var = Variable(var)
log_std = Variable(log_std)
PI = Variable(PI)
A = -(x - mean).pow(2) / (2 * var) 
#B = - 0.5 * torch.log(2 * PI)
B = -0.5*math.log(2*3.1415926)
print(A,B,log_std)
log_density =  A + B - log_std
print(log_density)