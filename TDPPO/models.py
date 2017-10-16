import copy
import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Policy(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 128)
        self.affine2 = nn.Linear(128, 300)
        self.affine3 = nn.Linear(300, 300)
        self.affine4 = nn.Linear(300, 128)
        self.action_mean = nn.Linear(128, num_outputs)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)
        self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))
        self.module_list_current = [self.affine1, self.affine2, self.affine3, self.affine4, self.action_mean, self.action_log_std]

        self.module_list_old = [None]*len(self.module_list_current) #self.affine1_old, self.affine2_old, self.action_mean_old, self.action_log_std_old]
        self.backup()

    def backup(self):
        for i in range(len(self.module_list_current)):
            self.module_list_old[i] = copy.deepcopy(self.module_list_current[i])

    def kl_div_p_q(self, p_mean, p_std, q_mean, q_std):
        """KL divergence D_{KL}[p(x)||q(x)] for a fully factorized Gaussian"""
        # print (type(p_mean), type(p_std), type(q_mean), type(q_std))
        # q_mean = Variable(torch.DoubleTensor([q_mean])).expand_as(p_mean)
        # q_std = Variable(torch.DoubleTensor([q_std])).expand_as(p_std)
        numerator = square(p_mean - q_mean) + \
            square(p_std) - square(q_std) #.expand_as(p_std)
        denominator = 2. * square(q_std) + eps
        return torch.sum(numerator / denominator + torch.log(q_std) - torch.log(p_std))

    def kl_old_new(self):
        """Gives kld from old params to new params"""
        kl_div = self.kl_div_p_q(self.module_list_old[-2], self.module_list_old[-1], self.action_mean, self.action_log_std)
        return kl_div

    def entropy(self):
        """Gives entropy of current defined prob dist"""
        ent = torch.sum(self.action_log_std + .5 * torch.log(2.0 * np.pi * np.e))
        return ent

    def forward(self, x, old=False):
        if old:
            x = F.tanh(self.module_list_old[0](x))
            x = F.tanh(self.module_list_old[1](x))
            x = F.tanh(self.module_list_old[2](x))
            x = F.tanh(self.module_list_old[3](x))

            action_mean = self.module_list_old[4](x)
            action_log_std = self.module_list_old[5].expand_as(action_mean)
            action_std = torch.exp(action_log_std)
        else:
            x = F.tanh(self.affine1(x))
            x = F.tanh(self.affine2(x))
            x = F.tanh(self.affine3(x))
            x = F.tanh(self.affine4(x))

            action_mean = self.action_mean(x)
            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std


class Value(nn.Module):
    def __init__(self, num_inputs):
        super(Value, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 128)
        self.affine2 = nn.Linear(128, 300)
        self.affine3 = nn.Linear(300, 300)
        self.affine4 = nn.Linear(300, 128)
        self.value_head = nn.Linear(128, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.tanh(self.affine1(x))
        x = F.tanh(self.affine2(x))
        x = F.tanh(self.affine3(x))
        x = F.tanh(self.affine4(x))

        state_values = self.value_head(x)
        return state_values

def create_actor(dim_i,dim_o):
    Actor = Policy(dim_i,dim_o)
    return Actor

def create_critic(dim_i):
    Critic = Value(dim_i)
    return Critic


if __name__ == '__main__':
    #for ep in range(100):
        #print(ep)
    Actor = create_actor(135,18)
    Critic = create_critic(135)
    States = torch.randn(64,135)
        #Actions = torch.randn(64,18)
        #ans = Variable(torch.randn(64,1))

        #actor_optimizer = optim.Adam(Actor.parameters(), lr=1e-4)
        #critic_optimizer = optim.Adam(Critic.parameters(), lr=3e-4)

    ans_actor = Actor(Variable(States))
    ans_critic = Critic(Variable(States))

        #critic_loss = F.mse_loss(ans_critic,ans)

        #critic_optimizer.zero_grad()
        #critic_loss.backward()
        #critic_optimizer.step()

    print(ans_actor)
    print(ans_critic)
