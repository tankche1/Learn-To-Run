import multiprocessing
cq = multiprocessing.Queue()
Lock = multiprocessing.Lock()

def sub(bh):
    Lock.acquire()
    for i in range(bh,bh+5):
        cq.put(('number',i))
    Lock.release()

def listen():
    while(True):
        msg = cq.get()
        print(msg)
p = multiprocessing.Process(target=listen)
p.start()

for bh in range(0,30,5):
    p = multiprocessing.Process(target=sub, args=(bh,))
    p.start()



