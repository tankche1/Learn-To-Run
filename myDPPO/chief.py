import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torch.autograd import Variable
import time
from running_state import ZFilter

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 50 epochs"""
    lr = args.lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def chief(args, rank, traffic_light, counter, shared_model, shared_grad_buffers, optimizer):
    epoch = 0
    while True:
        time.sleep(1)
        epoch = epoch + 1
        # workers will wait after last loss computation
        if counter.get() > args.num_processes-1:
            #print(shared_grad_buffers.grads['mu.weight_grad'])
            for n,p in shared_model.named_parameters():
                p._grad = Variable(shared_grad_buffers.grads[n+'_grad']) / args.num_processes
            optimizer.step()
            counter.reset()
            shared_grad_buffers.reset()
            if epoch%30 == 0:
                adjust_learning_rate(optimizer,epoch)
            #running_state = ZFilter((args.feature,), clip=5)
            traffic_light.switch() # workers start new loss computation
            #print('update')
    