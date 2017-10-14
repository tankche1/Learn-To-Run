import multiprocessing,time,random,threading
from multiprocessing import Process, Pipe, Queue

ncpu = 16

def standalone_headless_isolated(pq, cq, plock):
    # locking to prevent mixed-up printing.
    plock.acquire()
    print('starting headless...',pq,cq)
    try:
        import traceback
        from osim.env import RunEnv
        e = RunEnv(visualize=False)
    except Exception as e:
        print('error on start of standalone')
        traceback.print_exc()

        plock.release()
        return
    else:
        plock.release()

    def report(e):
        # a way to report errors ( since you can't just throw them over a pipe )
        # e should be a string
        print('(standalone) got error!!!')
        # conn.send(('error',e))
        # conn.put(('error',e))
        cq.put(('error',e))

    def floatify(np):
        return [float(np[i]) for i in range(len(np))]

    try:
        while True:
            # msg = conn.recv()
            # msg = conn.get()
            msg = pq.get()
            # messages should be tuples,
            # msg[0] should be string

            # isinstance is dangerous, commented out
            # if not isinstance(msg,tuple):
            #     raise Exception('pipe message received by headless is not a tuple')

            if msg[0] == 'reset':
                o = e.reset(difficulty=2)
                # conn.send(floatify(o))
                cq.put(floatify(o))
                # conn.put(floatify(o))
            elif msg[0] == 'step':
                o,r,d,i = e.step(msg[1])
                o = floatify(o) # floatify the observation
                cq.put((o,r,d,i))
                # conn.put(ordi)
                # conn.send(ordi)
            else:
                # conn.close()
                cq.close()
                pq.close()
                del e
                break
    except Exception as e:
        traceback.print_exc()
        report(str(e))

    return # end process

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

class ei: # Environment Instance
    def __init__(self):
        self.occupied = False # is this instance occupied by a remote client
        self.id = get_eid() # what is the id of this environment
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
            if time.time() - self.last_interaction > 20*60:
                # if no interaction for more than 20 minutes
                self.pretty('no interaction for too long, self-releasing now. applying for a new id.')

                self.id = get_eid() # apply for a new id.
                self.occupied == False

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
            return True # on success
        else:
            self.lock.release()
            return False # failed

    def release(self):
        self.lock.acquire()
        self.occupied = False
        self.id = get_eid()
        self.lock.release()

    # create a new RunEnv in a new process.
    def newproc(self):
        global plock
        self.timer_update()

        self.pq, self.cq = Queue(1), Queue(1) # two queue needed
        # self.pc, self.cc = Pipe()

        self.p = Process(
            target = standalone_headless_isolated,
            args=(self.pq, self.cq, plock)
        )
        self.p.daemon = True
        self.p.start()

        self.reset_count = 0 # how many times has this instance been reset() ed
        self.step_count = 0

        self.timer_update()
        return

    # send x to the process
    def send(self,x):
        return self.pq.put(x)
        # return self.pc.send(x)

    # receive from the process.
    def recv(self):
        # receive and detect if we got any errors
        # r = self.pc.recv()
        r = self.cq.get()

        # isinstance is dangerous, commented out
        # if isinstance(r,tuple):
        if r[0] == 'error':
            # read the exception string
            e = r[1]
            self.pretty('got exception')
            self.pretty(e)

            raise Exception(e)
        return r

    def reset(self):
        self.timer_update()
        if not self.is_alive():
            # if our process is dead for some reason
            self.pretty('process found dead on reset(). reloading.')
            self.kill()
            self.newproc()

        if self.reset_count>50 or self.step_count>10000: # if resetted for more than 100 times
            self.pretty('environment has been resetted too much. memory leaks and other problems might present. reloading.')

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
        self.step_count+=1
        return r

    def kill(self):
        if not self.is_alive():
            self.pretty('process already dead, no need for kill.')
        else:
            self.send(('exit',))
            self.pretty('waiting for join()...')

            while 1:
                self.p.join(timeout=5)
                if not self.is_alive():
                    break
                else:
                    self.pretty('process is not joining after 5s, still waiting...')

            self.pretty('process joined.')

    def __del__(self):
        self.pretty('__del__')
        self.kill()
        self.pretty('__del__ accomplished.')

    def is_alive(self):
        return self.p.is_alive()

    # pretty printing
    def pretty(self,s):
        print(('(ei) {} ').format(self.id)+str(s))

class eipool: # Environment Instance Pool
    def pretty(self,s):
        print(('(eipool) ')+str(s))

    def __init__(self,n=1):
        import threading as th
        self.pretty('starting '+str(n)+' instance(s)...')
        self.pool = [ei() for i in range(n)]
        self.lock = th.Lock()

    def acq_env(self):
        self.lock.acquire()
        for e in self.pool:
            if e.occupy() == True: # successfully occupied an environment
                self.lock.release()
                return e # return the envinstance

        self.lock.release()
        return False # no available ei

    def rel_env(self,ei):
        self.lock.acquire()
        for e in self.pool:
            if e == ei:
                e.release() # freed
        self.lock.release()

    # def num_free(self):
    #     return sum([0 if e.is_occupied() else 1 for e in self.pool])
    #
    # def num_total(self):
    #     return len(self.pool)
    #
    # def all_free(self):
    #     return self.num_free()==self.num_total()

    def get_env_by_id(self,id):
        for e in self.pool:
            if e.id == id:
                return e
        return False

    def __del__(self):
        for e in self.pool:
            del e

import traceback
class farm:
    def pretty(self,s):
        print(('(farm) ')+str(s))

    def __init__(self):
        # on init, create a pool
        # self.renew()
        import threading as th
        self.lock = th.Lock()

    def acq(self,n=None):
        self.renew_if_needed(n)
        result = self.eip.acq_env() # thread-safe
        if result == False:
            ret = False
        else:
            self.pretty('acq '+str(result.id))
            ret = result.id
        return ret

    def rel(self,id):
        e = self.eip.get_env_by_id(id)
        if e == False:
            self.pretty(str(id)+' not found on rel(), might already be released')
        else:
            self.eip.rel_env(e)
            self.pretty('rel '+str(id))

    def step(self,id,actions):
        e = self.eip.get_env_by_id(id)
        if e == False:
            self.pretty(str(id)+' not found on step(), might already be released')
            return False

        try:
            ordi = e.step(actions)
            return ordi
        except Exception as e:
            traceback.print_exc()
            raise e

    def reset(self,id):
        e = self.eip.get_env_by_id(id)
        if e == False:
            self.pretty(str(id)+' not found on reset(), might already be released')
            return False

        try:
            oo = e.reset()
            return oo
        except Exception as e:
            traceback.print_exc()
            raise e

    def renew_if_needed(self,n=None):
        self.lock.acquire()
        if not hasattr(self,'eip'):
            self.pretty('renew because no eipool present')
            self._new(n)
        self.lock.release()

    # # recreate the pool
    # def renew(self,n=None):
    #     global ncpu
    #     self.pretty('natural pool renew')
    #
    #     if hasattr(self,'eip'): # if eip exists
    #         while not self.eip.all_free(): # wait until all free
    #             self.pretty('wait until all of self.eip free..')
    #             time.sleep(1)
    #         del self.eip
    #     self._new(n)

    def forcerenew(self,n=None):
        self.lock.acquire()
        self.pretty('forced pool renew')

        if hasattr(self,'eip'): # if eip exists
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

    #def __del__(self):
    #    self.rel()

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
