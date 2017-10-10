
import scipy
import torch
from models import Policy, Value, ActorCritic
from running_state import Shared_obs_stats

#A = Shared_obs_stats(3)
#torch.save({'obs':A},'haha.t7')
def haha(A):
    A.append(1)
A = [2]
haha(A)
print(A)
