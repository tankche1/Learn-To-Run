
import gym
env = gym.make('InvertedPendulum-v1')
observation = env.reset()
for _ in range(1000):
  #env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)
  print(type(observation))
  print(type(reward))
  print(type(done))