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
o = [1]*3
o[0]=2
o[1]=3
print(o/0.5)