#!/usr/bin/env python                                                           
import threading, multiprocessing, os, time, datetime, sys, signal, traceback, Queue
from multiprocessing.pool import Pool
from pprint import pprint

from pysurvey import util
# from pysurvey import splog, nicefile

# import calc.util


# from data.makesamples import _internal, _iterator



## http://www.bryceboe.com/2010/08/26/python-multiprocessing-and-keyboardinterrupt/
def _test(a,b=None):
    # time.sleep(1)
    # if a == 5:
    #     raise ValueError
    util.splog('test',a,b)
    time.sleep(10.1)
    
    



#### Some basic multiprocess functions


def do_work(args):
    '''Wrap the function with some nice Keyboard error handling and error
    printing.  Args = (function, *arguments).  '''
    
    # Ignore the SIGINT -- Interrupt signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    # Attempt to run the function with a set of arguments
    try:
        time.sleep(0.1) # noop to prevent spawning after ctrl-c
        fcn, args = args[0], args[1:]
        fcn(*args)
    except KeyboardInterrupt, e:
        raise
    except:
        # something else happened so grab the information and give it to the
        # user since the default error handleing only gives some idea that 
        # there was a python multiprocess error.
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=5)
        raise


def getargs(fcn, args):
    out = []
    for x in args:
        if isinstance(x, str):
            # should fix this to iterables
            out.append([fcn]+[x])
        else:
            out.append([fcn]+[x])
    return out


def multi_process(fcn, args, n=None):
    '''Multiprocess a function with an iterable list of arguments (args).
    Set n>= int for the number of processes to use.   This might blow up, use 
    caution and get coffee.
    '''
    util.SPLOG['multi']=True
    
    # Attempt to all of the CPU except for one.  The documentation suggests     
    # that this might not return a value so default to two processes.
    if n is None:
        try:
            n = multiprocessing.cpu_count()-1
        except:
            n = 2
    util.splog('Starting run with %d processes'%(n), color='green')

    start = util.deltatime()
    pool = multiprocessing.Pool(n)
    # p = pool.map_async(do_work, [[fcn]+list(x) for x in args])
    tmp = getargs(fcn,args)
    print tmp
    p = pool.map_async(do_work, tmp)
    try:
        # This is the timeout to wait for the results.  We default to something
        # really really large.
        results = p.get(0xFFFFFFF)
    except KeyboardInterrupt:
        # This captures the error raised by do_work and raises a different error
        # so that the computation finally ends and we do not attempt to run
        # other problems.
        print 'parent received control-c'
        raise RuntimeError
    finally:
        # Shut down the pool of processes if we are done or failed at our task.
        pool.terminate()
        util.SPLOG['multi'] = False
        
    # look how nice this is
    util.splog("Finished!: %s"%(fcn.__name__), util.deltatime(start),color='green')










##### Threading vs processes
# import logging
# logging.basicConfig(level=logging.DEBUG,
#                     format='(%(threadName)-10s) %(message)s',
#                     )
# exitFlag = 0






class KThread(threading.Thread):
    """A subclass of threading.Thread, with a kill() method."""
    def __init__(self, *args, **keywords):
        threading.Thread.__init__(self, *args, **keywords)
        self.killed = False
        
    def start(self):
        """Start the thread."""
        self.__run_backup = self.run
        self.run = self.__run      # Force the Thread to install our trace.
        threading.Thread.start(self)
        
    def __run(self):
        """Hacked run function, which installs the trace."""
        sys.settrace(self.globaltrace)
        self.__run_backup()
        self.run = self.__run_backup

    def globaltrace(self, frame, why, arg):
        if why == 'call':
            return self.localtrace
        else:
            return None

    def localtrace(self, frame, why, arg):
        if self.killed:
            if why == 'line':
                raise SystemExit()
        return self.localtrace

    def kill(self):
        print "%s KILLED ... You Monster -- you may have to Ctrl-c again"%(self.name)
        self.killed = True








def multi_thread(fcn, args, n=None):
    '''Ok fuck you threads.  Leave all of the variables in here so that 
    things work. '''
    if n is None:
        try:
            n = multiprocessing.cpu_count()-1
        except:
            n = 2
    
    
    
    # Signal for the Worker to be done and the different locks and queues
    exitFlag = 0
    queueLock = threading.Lock()
    workQueue = Queue.Queue(1000)
    
    
    class Worker(KThread):
        '''A little worker thread that runs and runs and runs'''
        def __init__(self, fcn):
            KThread.__init__(self)
            self.fcn = fcn
            # self.queue = queue
            
        def run(self):
            print "Starting " + self.name
            
            while not exitFlag:
                queueLock.acquire()
                if not workQueue.empty():
                    args = workQueue.get()
                    # args = self.queue.get()
                    queueLock.release()
                    
                    start = util.deltatime()
                    util.splog("Starting: %s"%(fcn.__name__), 
                                    'with', repr(args), color='green')
                    self.fcn(*args)
                    util.splog("Finished!: %s"%(fcn.__name__), 
                                    util.deltatime(start),color='green')
                else:
                    queueLock.release()
                time.sleep(0.1)
            print "Exiting " + self.name
    
    
    def _print_spinner(t):
        ''' A simple little printer that ensures that future messages
        are preserved and that waits 0.2 sec'''
        if   t == 4: t = 0
        if   t == 0: sys.stdout.write("\r/")
        elif t == 1: sys.stdout.write("\r-")
        elif t == 2: sys.stdout.write("\r\\")
        elif t == 3: sys.stdout.write("\r|")
        
        sys.stdout.flush()
        sys.stdout.write('\r')
        time.sleep(0.2)
        return t+1
    






    # Create new threads that will be doing all of the work
    threads = []
    for i in range(n):
        # thread = Worker(fcn, workQueue)
        thread = Worker(fcn)
        thread.start()
        threads.append(thread)

    # Fill the queue
    queueLock.acquire()
    for arg in args:
        workQueue.put(list(arg))
    queueLock.release()



    # Ok now as the main thread we wait and wait until the queue is empty and
    # we can close everything down.  If something bombed, lets quickly attempt 
    # to shut it all down and go about our business
    try:
        t = 0
        while not workQueue.empty():
            t = _print_spinner(t)
            pass
    except Exception as e:
        print e
        raise
    finally:
        # Clean up any alive threads
        exitFlag = 1
        for t in threads:
            # t.join()
            # I will give you 1 hr to time out
            t.join(1*60*60)
            # or lets kill that mf 
            if t.isAlive():
                t.kill()
        print "Exiting Main Thread"








# by default  use the multiprocess 
multi = multi_process


if __name__ == "__main__":
    
    def _big(a):
        calc.splog('aaaa', a)
        import numpy as np
        x = np.zeros( (10000,10000) ) # ~ 1 GB vert ~0.7GB res 
        time.sleep(10)
        
    
    def _gen():
        for i in range(5):
            yield (i,)
    
    # multi_thread(_test, _gen())
    multi_thread(_big, _gen())
    
    
    
    
    # args = zip(range(10),range(10))
    # multi(_test, args)
    # multi(_test, args)




def signal_example():
    # http://stackoverflow.com/questions/366682/how-to-limit-execution-time-of-a-function-call-in-python
    def signal_handler(signum, frame):
        raise Exception("Timed out!")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(2)   # Ten seconds
    try:
        long_function_call()
        
        
    except Exception, msg:
        print "Timed out!"















# # http://stackoverflow.com/questions/6728236/exception-thrown-in-multiprocessing-pool-not-detected
# # Shortcut to multiprocessing's logger
# def error(msg, *args):
#     return multiprocessing.get_logger().error(msg, *args)
# 
# class LogExceptions(object):
#     def __init__(self, callable):
#         self.__callable = callable
#         return
# 
#     def __call__(self, *args, **kwargs):
#         try:
#             result = self.__callable(*args, **kwargs)
# 
#         except Exception as e:
#             # Here we add some debugging help. If multiprocessing's
#             # debugging is on, it will arrange to log the traceback
#             error(traceback.format_exc())
#             # Re-raise the original exception so the Pool worker can
#             # clean up
#             raise
# 
#         # It was fine, give a normal answer
#         return result
#     pass
# 
# 
# class LoggingPool(Pool):
#     def apply_async(self, func, args=(), kwds={}, callback=None):
#         return Pool.apply_async(self, LogExceptions(func), args, kwds, callback)









# def do_work(i):
#     try:
#         print 'Work Started: %d %d' % (os.getpid(), i)
#         time.sleep(2)
#         return 'Success'
#     except KeyboardInterrupt, e:
#         pass
#  
# def main():
#     pool = multiprocessing.Pool(3)
#     p = pool.map_async(do_work, range(6))
#     try:
#         results = p.get(0xFFFF)
#     except KeyboardInterrupt:
#         print 'parent received control-c'
#         return
#  
#     for i in results:
#         print i
#  
# if __name__ == "__main__":
#     main()


# #!/usr/bin/env python
# import multiprocessing, os, signal, time, Queue
#  
# def do_work():
#     print 'Work Started: %d' % os.getpid()
#     time.sleep(2)
#     return 'Success'
#  
# def manual_function(job_queue, result_queue, fcn):
#     signal.signal(signal.SIGINT, signal.SIG_IGN)
#     while not job_queue.empty():
#         try:
#             job = job_queue.get(block=False)
#             result_queue.put(fcn())
#         except Queue.Empty:
#             pass
#         #except KeyboardInterrupt: pass
#  
# def main(fcn):
#     job_queue = multiprocessing.Queue()
#     result_queue = multiprocessing.Queue()
#  
#     for i in range(6):
#         job_queue.put(None)
#  
#     workers = []
#     for i in range(3):
#         tmp = multiprocessing.Process(target=manual_function,
#                                       args=(job_queue, result_queue, fcn))
#         tmp.start()
#         workers.append(tmp)
#  
#     try:
#         for worker in workers:
#             worker.join()
#     except KeyboardInterrupt:
#         print 'parent received ctrl-c'
#         for worker in workers:
#             worker.terminate()
#             worker.join()
#  
#     while not result_queue.empty():
#         print result_queue.get(block=False)
#  
# if __name__ == "__main__":
#     main(do_work)