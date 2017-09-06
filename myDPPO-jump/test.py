
import scipy
import torch
from models import Policy, Value, ActorCritic
from running_state import Shared_obs_stats

#A = Shared_obs_stats(3)
#torch.save({'obs':A},'haha.t7')
A =torch.load('haha.t7')['obs']
print(A.n)
