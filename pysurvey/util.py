# plot.py : some nice plotting tools
# Mendez 05.2013

# System Libraries
import os
import sys
import socket
import datetime
import inspect
import traceback
import collections

# Installed Libraries
from clint.textui import colored, puts

# Package Imports

# nice variables
SPLOG = {'multi':False, # multiprocessing setup
         'outfile':None, # save to disk
        }

class stop(Exception):
    pass

def setup_stop():
    '''Give a nice user hook
    or use 
    # import pdb; pdb.set_trace()
    # import ipdb; ipdb.set_trace()
    # from IPython import embed; embed() 
    
    '''
    
    import numpy as np
    np.set_printoptions(precision=3, suppress=True)
    
    
    
    def excepthook(ec, ei, tb):
        '''the Exception Hook'''
        traceback.print_exception(ec, ei, tb)
        # splog('Variables', color='red', stack=-1)
        # for item,value in vars().items():
        #     v = '{}'.format(value)
        #     if 'module' not in v:
        #         splog('{}: {}'.format(item, v), color='red', stack=-1)
        try:
            import ipdb # is slow, so kee[ it here.]
            ipdb.pm()
        except:
            import pdb
            pdb.pm()
    sys.excepthook = excepthook

def edit(item):
    '''a simple edit function'''
    from subprocess import call
    
    try:
        if isinstance(item, str):
            call(['open', '-a', 'TextMate', item])
        else:
            filename = item.__file__.replace('.pyc','.py')
            call(['open', '-a', 'TextMate', filename])
        return
    except:
        pass
    print item


def gethostname():
    '''Get the hostname of the current machine.  This is a wrapper around socket.gethostname
    since it is nice to attempt to get the FQDN rather than just the hostname.  Sometimes
    the FDQN fails on osx, so then just return the default hostname'''
    return socket.gethostname()
    # try:
    #     return socket.gethostbyaddr(socket.gethostname())[0]
    # except:
    #     return socket.gethostname()

def ishostname(host):
    hostname = gethostname()
    if isinstance(host, str):
        return host in hostname
    elif isinstance(host, (tuple, list)):
        return any([h in hostname for h in host])
    else:
        raise NotImplementedError('Failed to process: %s'%host)
    


def settitle(name):
    '''Set the window title by hit or miss'''
    if os.name == 'posix':
        sys.stdout.write("\x1b]2;{}\x07".format(name))
    else:
        ### Windows: -- might be too much of an assumption
        # platform.system() might be better to determine
        os.system("title {}".format(name))






def start_deltatime():
    '''Return the date time for use with deltatime'''
    return datetime.datetime.now()


def print_deltatime(seconds):
    '''formats a nice set of strings'''
    out = []
    epocs = [ [86400, 2, 'day'],
              [3600,  2, 'hour'],
              [60,    2, 'minute'],
              [1,     2, 'second'],
              [0.001, 3, 'millisecond'] ]
    factors = zip(*epocs)[0]
    while seconds > min(factors):
      for factor, d, noun in epocs:
        if seconds >= factor:
          delta = int(seconds/factor)
          if delta > 1: noun +='s'
          s = '' if delta > 1 else ' '
          out.append('%{:d}d %s%s'.format(d)%(delta,s,noun))
          seconds -= factor*(delta)
  
    n = 2 if len(out) > 2 else len(out)
    return ', '.join(out[:n])


def deltatime(start=None, gettime=False):
  '''From a start datetime object get a nice string of 'x hours, y minutes, z seconds'
  of time passed since the start. 
  
  example:
  start = deltatime()
  # do calculations
  splog('Finished:', deltatime(datetime.datetime.now()))
  
  '''
  if start is None:
    return datetime.datetime.now()
      
  diff = datetime.datetime.now() - start
  # seconds = diff.seconds # limited to 1 second, so lets get fractions
  seconds = diff.days*86400 + diff.seconds + diff.microseconds/float(10**6)
  if gettime:
      return seconds
  return print_deltatime(seconds)

def projecttime(start=None, index=None, number=None):
    '''Return a nicely formated string of the time it will take to finish this.'''
    if start is None:
        return deltatime()
    seconds = deltatime(start, True)
    projected = (number-index) * (seconds*1.0/index)
    return print_deltatime(projected)
    





def uniqify(x, idfun=None, order=False, sort=True):
    ''' Uniqify a list :: http://www.peterbe.com/plog/uniqifiers-benchmark
    
    idfun=None    -- pass in a function with returns a sorting key for each element of x
    order=[False] -- preserve the relative ordering of original list.  
    '''
    if order:
        # return list(Set(seq))
        if sort:
            return sorted({}.fromkeys(x).keys())
        return {}.fromkeys(x).keys()
    else:  # order preserving
       if idfun is None:
           idfun = lambda x: x
       seen = {}
       result = []
       for item in x:
           marker = idfun(item)
           if marker in seen: 
               continue
           seen[marker] = 1
           result.append(item)
       return result
uniq = uniqify # to make it similar to IDL -- yeah, I know...


def uniq_dict(x, **kwargs):
    '''Get a nice dictionary which we can loop over'''
    import numpy as np
    toscreen = kwargs.pop('print',False)
    u = uniq(x, **kwargs)
    out = {}
    for item in u:
        n = len(np.where(x == item)[0])
        if toscreen:
            splog('{:>10d} : {!r}'.format(n, item))
        out[item] = n
    return out


def print_uniq(x, **kwargs):
    '''print the unit elements with number of total elements'''
    kwargs.setdefault('print',True)
    uniq_dict(x, **kwargs)








def  _name():
    try:
        import re
        frame,_,_,_,code,_ = inspect.getouterframes(inspect.currentframe())[-1]
        ## This fails for multi line code.  ugh
        
        # pdb.set_trace()
        # ccmodule = inspect.getmodule(frame)
        # slines, start = inspect.getsourcelines(ccmodule)
        # clen = len(slines)
        # finfo = inspect.getframeinfo(frame, clen)
        # 
        # theindex = finfo[4]
        # lines = finfo[3]
        # code = lines
        
        outer = re.compile("\((.+)\)").search(' '.join(code)).group(1)
        outer = re.sub(r'\[.*?\]', '', outer)
        names = [x.strip() for x in outer.split(',') if ('=' not in x)]
        return names
    except Exception as e:
        # raise
        # print 'FAILED: ',e
        return None
        # pdb.set_trace()

def _flatten(x):
    if isinstance(x, collections.Iterable) and not isinstance(x,str):
        return [a for i in x for a in _flatten(i)]
    else:
        return [x]


def for_print(*args, **kwargs):
    '''Print a set of arrays'''
    kwargs.setdefault('n',100)
    kwargs.setdefault('width',14)
    kwargs.setdefault('precision',2)
    fmt = kwargs.setdefault('fmt',lambda x: x)
    
    names = _name()
    if names is not None:
        print(', '.join(['{0:>{width}}'.format(fmt(name), width=(kwargs['width']+2)*(len(arg[0]))-2)
                         if isinstance(arg[0], collections.Iterable) and not isinstance(arg[0], str) else
                         '{0:>{width}}'.format(fmt(name), width=kwargs['width'])
                         for name,arg in zip(names,args)]))
    
    
    # import pdb; pdb.set_trace()
    for i,group in enumerate(zip(*args)):
        group = _flatten(group)
        
        print(', '.join(['{0: {w}.{p}f}'.format(item, w=kwargs['width'], p=kwargs['precision']) 
                         if not isinstance(item,(str,bool)) else 
                         '{0:>{width}}'.format(str(item), width=kwargs['width'])
                         for item in group ]))
        if kwargs['n'] is not None and i > kwargs['n']:
            print(' ... [> %i rows]'%(kwargs['n']))
            return

forprint = for_print






def splog(*args, **kwargs):
    ''' Mimic splog in idl but now with more colors and fancy module names.
    This is not that fast so ensure that you are only running this function
    at a low rate.
    *args -- you can pass as many arguments into the function and it will be 
             printed as a space separated list.
    color='blue' -- by setting this keyword, it attempts to load one of the 
                    color function from the clint.textui.colored package
                    ('red', 'green', 'yellow', 'blue', 'black', 'magenta', 'cyan')
    name = defined name to use rather than determine it from the stack
    stack=1 -- go up the stack some number of values
    
    getstr=True -- return the string without printing to stdout
    
    
    if the first letter of the first argument is one of ('\n','\t','!') it will be 
    prepended before the line. The '!' adds a nice red exclamation point for somewhat
    important debugs that can be found.
    
    
    '''
    
    # colordict = {'red': colored.red,
    #              'green': colored.green,
    #              'blue': colored.blue,
    #              'yellow':colored.red}
    # f = colordict.get(kwargs.get('color','blue'), colored.blue)
    f = getattr(colored, kwargs.get('color','blue'), colored.blue)
    multiline = kwargs.pop('multiline',False)
    
    
    
    s = inspect.stack()
    try:
        i = kwargs.pop('stack',1)
        module = inspect.getmodule(s[i][0]).__name__
        # import pdb; pdb.set_trace()
    except AttributeError:
        module = '__main__'
    
    module = '<main>' if module == '__main__' else module
    name = kwargs.pop('name', s[1+kwargs.get('stack',0)][3])
    prefix = ' '*(len(s)-2)
    
    if (len(prefix) > 11) and SPLOG['multi']:
        # There are a bunch of indirection levels that are not
        # that important for multi process things, so drop them out
        # of the heiarchy.
        prefix = prefix[11:]
    
    # attempt to prefix the entire line with a specific keyword set
    try:
        if (len(args) > 0) and ( args[0].startswith('\n') or 
                                 args[0].startswith('\t') or
                                 args[0].startswith('!') ):
            p = colored.red('! ') if args[0][0] == '!' else args[0][0]
            prefix = p + prefix
            args = [args[0][1:].strip()] + list(args[1:])
    except:
        pass
    
    
    if SPLOG['multi']:
        try:
            if len(prefix) > 10:
                prefix = prefix[:10]
            # while len(prefix) > 10:
            #     prefix = '>' + prefix.split()[0] + prefix[10:]
        except:
            pass
        prefix = '%s>'%(os.getpid())+prefix
    
    if multiline:
        p = '%s'%(prefix) + f('%s.%s'%(module,name))
        output = '\n'.join('%s %s'%(p, s) for s in args)
    else:
        output = ('%s'%(prefix) + 
                  f('%s.%s '%(module,name))+
                  ' '.join('%s'%s for s in args))
    
    if SPLOG['outfile'] is not None:
        # There is probably a better way of keeping this file open
        # in the SPLOG dict, but since this is not suppose to be run
        # at high frequency lets just keep open and closing the file.
        with open(SPLOG['outfile'], 'a+') as f:
            f.write('[%s]: %s\n'%(datetime.datetime.now(), output))
    
    if kwargs.get('getstr',False):
        return output 
    

            
    
    puts(output)




def getargs(fcn, **kwargs):
    args, vargs, keywords, _ = inspect.getargspec(fcn)
    if keywords is not None:
        raise NotImplementedError('kwargs are not setup... yeah fix that')
    out = {}
    for arg in args:
        tmp = kwargs.pop(arg, None)
        if tmp is not None:
            out[arg] = tmp
    return out, kwargs

    





