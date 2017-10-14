import copy
import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


class ActorNN(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(ActorNN, self).__init__()
        self.Linear1 = nn.Linear(num_inputs,256)
        self.Linear2 = nn.Linear(256,128)
        self.Linear3 = nn.Linear(128,128)
        self.Linear4 = nn.Linear(128,128)
        self.Linear5 = nn.Linear(128,num_outputs)

    def forward(self,x):
        x = F.relu(self.Linear1(x))
        x = F.relu(self.Linear2(x))
        x = F.relu(self.Linear3(x))
        x = F.relu(self.Linear4(x))
        x = F.tanh(self.Linear5(x))
        return x

class CriticNN(nn.Module):

    def __init__(self, num_states, num_actions):
        super(CriticNN, self).__init__()
        self.Linear1 = nn.Linear(num_states,256)
        self.Linear2 = nn.Linear(256,128)
        self.Linear3 = nn.Linear(128,128)

        self.Linear4 = nn.Linear(128+num_actions,128)
        self.Linear5 = nn.Linear(128,128)
        self.Linear6 = nn.Linear(128,48)
        self.Linear7 = nn.Linear(48,1)

    def forward(self,A):
        [state,action] = A
        x = F.relu(self.Linear1(state))
        x = F.relu(self.Linear2(x))
        x = F.relu(self.Linear3(x))

        x = torch.cat((x,action),1)
        x = F.relu(self.Linear4(x))
        x = F.relu(self.Linear5(x))
        x = F.relu(self.Linear6(x))
        x = self.Linear7(x)

        return x


def create_actor_network(ids,ods):
    actor_network = ActorNN(ids,ods)
    return actor_network
def create_critic_network(ids,ods):
    critic_network = CriticNN(ids,ods)
    return critic_network

if __name__ == '__main__':
    for ep in range(100):
        print(ep)
        Actor = create_actor_network(41,18)
        Critic = create_critic_network(41,18)
        States = torch.randn(64,41)
        Actions = torch.randn(64,18)
        ans = Variable(torch.randn(64,1))

        actor_optimizer = optim.Adam(Actor.parameters(), lr=1e-4)
        critic_optimizer = optim.Adam(Critic.parameters(), lr=3e-4)

        ans_actor = Actor(Variable(States))
        ans_critic = Critic([Variable(States),Variable(Actions)])

        critic_loss = F.mse_loss(ans_critic,ans)

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

    #print(ans_actor)
    #print(ans_critic)

