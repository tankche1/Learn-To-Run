# TODO LIST

- [ ] calculate velocities of all the body parts: v = dx/dt = (x_now - x_last)/0.01 and add them to your observation vector, then feed to your learning algorithm.
- [ ] 3 frames execute an action
- [ ] change network topology 1.wide and shallow 2.leakyrelu 3.RNN?
- [ ] early stop in training (avoid memory leaks)
- [ ] 16 threads



# Brief Introduction

Please select the right branch.

PPO3 is the original PPO working on original observation.(Reach 12points in 4 days)

PPO is the asychronous PPO (like A3C).Didn't fine-tuned well.

myDPPO is the sychronous PPO with observation pre-processed and frame skip.

currently working on DDPG.

