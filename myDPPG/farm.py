import multiprocessing,time,random,threading
from multiprocessing import Process, Pipe, Queue

ncpu = 16

def standalone_headless_isolated(pq,cq,plock):
    plock.acquire()
    print('starting headless...',pq,cq)
    from osim.env import RunEnv
    e = RunEnv(visualize=False)
    plock.release()

    def report(e):
        cq.put(('error',e))

    def floatify(np):
        return [float(np[i]) for i in range(len(np))]

    while True:
        msg = pq.get()

        if msg[0] == 'reset':
            o = e.reset(difficulty=2)
            cq.put(floatify(o))
        elif msg[0] == 'step':
            o,r,d,i = e.step(msg[1])
            o = floatify(o)
            cq.put((o,r,d,i))
        else:   # exit
            cq.close()
            pq.close()
            del e
            break
    return

plock = multiprocessing.Lock()

tlock = threading.Lock()

eid = int(random.random()*100000)

def get_eid():
    global eid,tlock
    tlock.acquire()
    i = eid
    eid += 1
    tlock.release()
    return i

class ei:
    def __init__(self):
        self.occupied = False
        self.id = get_eid()
        self.pretty('instance creating')

        self.newproc()
        import threading as th
        self.lock = th.Lock()

    def timer_update(self):
        self.last_interaction = time.time()

    def is_occupied(self):
        if self.occupied == False:
            return False
        else:
            if time.time() - self.last_interaction > 20*60 :
                self.pretty('no interaction for long')

                self.id = get_eid()
                self.occupied = False

                self.pretty('self-released.')
                return False
            else:
                return True  

    def occupy(self):
        self.lock.acquire()
        if self.is_occupied() == False:
            self.occupied = True
            self.id = get_eid()
            self.lock.release()
            return True
        else:
            self.lock.release()
            return False

    def release(self):
        self.lock.acquire()
        self.occupied = False
        self.id = get_eid()
        self.lock.release()


    def newproc(self):
        global plock
        self.timer_update()

        self.pq, self.cq = Queue(1),Queue(1)

        self.p = Process(
            target = standalone_headless_isolated,
            args = (self.pq,self.cq, plock)
            )

        self.p.daemon = True # child process end when father process end
        self.p.start()

        self.reset_count = 0
        self.step_count = 0

        self.timer_update()
        return

    def send(self,x):
        return self.pq.put(x)

    def recv(self):
        r = self.cq.get()

        if r[0] == 'error':
            e = r[1]
            self.pretty('got exception')
            self.pretty(e)

            raise Exception(e)
        return r

    def reset(self):
        self.timer_update()
        if not self.is_alive():
            self.pretty('process not alive.reloading..')
            self.kill()
            self.newproc()

        if self.reset_count>50 or self.step_count>10000:
            self.pretty('environment has been reset too much renewing..')

            self.kill()
            self.newproc()

        self.reset_count += 1
        self.send(('reset',))
        r = self.recv()
        self.timer_update()
        return r

    def step(self,actions):
        self.timer_update()
        self.send(('step',actions,))
        r = self.recv()
        self.timer_update()
        self.step_count += 1
        return r

    def kill(self):
        if not self.is_alive():
            self.pretty('already died before kill')
        else:
            self.send(('exit',))
            self.pretty('waiting for join()...')

            while 1:
                self.p.join(timeout=5)
                if not self.is_alive():
                    break
                else:
                    self.pretty('waiting to kill(5s)...')

            self.pretty('process joined.')

    def __del__(self):
        self.pretty('__del__')
        self.kill()
        self.pretty('__del__ done.')

    def is_alive(self):
        return self.p.is_alive()

    def pretty(self,s):
        print(('(ei) {} ').format(self.id) + str(s))

class eipool:
    def pretty(self,s):
        print('(eipool) '+str(s))

    def __init__(self,n=1):
        import threading as th
        self.pretty('starting ' +str(n) + '  instance(s)..')
        self.pool = [ei() for i in range(n)]
        self.lock = th.Lock()

    def acq_env(self,n=1):
        self.lock.acquire()
        for e in self.pool:
            if e.occupy() == True:
                self.lock.release()
                return e

        self.lock.release()
        return False

    def rel_env(self,ei):
        self.lock.acquire()
        for e in self.pool:
            if e == ei:
                e.release()
        self.lock.release()

    def get_env_by_id(self,id):
        for e in self.pool:
            if e.id == id:
                return e

        return False

    def __del__(self):
        for e in self.pool:
            del e

class farm:
    def pretty(self,s):
        print '(farm) ' + str(s)

    def __init__(self):
        import threading as th
        self.lock = th.Lock()

    def acq(self,n=None):
        self.renew_if_needed(n)
        result = self.eip.acq_env()
        if result == False:
            ret = False
        else:
            self.pretty('acq' + str(result.id))
            ret = result.id
        return ret

    def rel(self,id):
        e = self.eip.get_env_by_id(id)
        if e == False:
            self.pretty(str(id) + ' not found on rel(),might be del .')
        else:
            self.eip.rel_env(e)
            self.pretty('rel' + str(id))

    def step(self,id,actions):
        e = self.eip.get_env_by_id(id)
        if e == False:
            self.pretty(str(id)+'not found on step()')
            return False

        ordi = e.step(actions)
        return ordi

    def reset(self,id):
        e = self.eip.get_env_by_id(id)
        oo = e.reset()
        return oo

    def renew_if_needed(self,n=None):
        self.lock.acquire()
        if not hasattr(self,'eip'):
            self.pretty('renew because no eipool represent')
            self._new(n)
        self.lock.release()

    def forcerenew(self,n=None):
        self.lock.acquire()
        self.pretty('forced pool renew')

        if hasattr(self,'eip'):
            del self.eip
        self._new(n)
        self.lock.release()

    def _new(self,n=None):
        self.eip = eipool(ncpu if n is None else n)

class remoteEnv:
    def pretty(self,s):
        print(('(remoteEnv) {} ').format(self.id)+str(s))

    def __init__(self,fp,id): # fp = farm proxy
        self.fp = fp
        self.id = id

    def reset(self):
        return self.fp.reset(self.id)

    def step(self,actions):
        ret = self.fp.step(self.id, actions)
        if ret == False:
            self.pretty('env not found on farm side, might been released.')
            raise Exception('env not found on farm side, might been released.')
        return ret

    def rel(self):
        while True: # releasing is important, so
            try:
                self.fp.rel(self.id)
                break
            except Exception as e:
                self.pretty('exception caught on rel()')
                self.pretty(e)
                time.sleep(3)
                pass

        #self.fp._pyroRelease()

    def __del__(self):
        self.rel()

def new_farm(n):
    Farm = farm()
    Farm._new(n)
    return Farm

def new_remote_env(fp,id):
    remote_env = remoteEnv(fp,id)
    return remote_env
if __name__ == '__main__':
    farm = farm()
    farm._new()
