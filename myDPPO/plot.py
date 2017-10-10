import os
import re
import string


epoch = range(0,3000)
rewards = range(0,6000,2)
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure(1)
plt.plot(epoch, rewards)
plt.xlabel('epoch')
plt.ylabel('score')
#plt.show()

fig.savefig('plot.pdf')
