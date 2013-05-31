# plot.py : some nice plotting tools
# Mendez 05.2013

# System Libraries
import os
import datetime
import inspect
import collections

# Installed Libraries
import numpy as np
from clint.textui import colored, puts

# Package Imports


# nice variables
SPLOG = {'multi':False, # multiprocessing setup
        }







def minmax(x, nan=True):
    '''Returns both the minimum and maximum'''
    if nan:
        return np.min(x), np.max(x)
    else:
        return np.nanmin(x), np.nanmax(x)

def embiggen(r, p=0.05, mode='both'):
    '''Returns a larger range from an input range (r) and percentage p.
    p=[0.05]  -- 5 percent increase in size
    mode=['both'] -- increase both sides of the range ('both','upper','lower')
    '''
    xmin, xmax = r
    sign = 1.0
    if xmin > xmax:
        sign = -1.0
    delta = abs(xmax-xmin)
    d = delta*p*sign
    if mode == 'both':
        return xmin-d, xmax+d
    elif mode == 'upper':
        return xmin, xmax+d
    elif mode == 'lower':
        return xmin-d, xmax

def match(X,Y):
    '''(unoptimized) -- Returns the matched index so that:
        ii,jj = match(X,Y)
        X[ii] == Y[jj]
    
    '''
    ii = np.in1d(X, Y).nonzero()[0]
    jj = [np.where(Y == x)[0][0] for x in X[ii]]
    return ii, jj




def start_deltatime():
    '''Return the date time for use with deltatime'''
    return datetime.datetime.now()


def deltatime(start=None):
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
  out = []
  epocs = [ [3600,'hour'],
            [60,'minute'],
            [1,'second'],
            [0.001,'milisecond'] ]
  factors,tmp = zip(*epocs)
  while seconds > min(factors):
    for factor, noun in epocs:
      if seconds >= factor:
        delta = int(seconds/factor)
        if delta > 1: noun +='s'
        out.append('%d %s'%(delta,noun))
        seconds -= factor*(delta)
  
  return ', '.join(out)




def uniqify(x, idfun=None, order=False):
    ''' Uniqify a list :: http://www.peterbe.com/plog/uniqifiers-benchmark
    
    idfun=None    -- pass in a function with returns a sorting key for each element of x
    order=[False] -- preserve the relative ordering of original list.  
    '''
    if order:
        # return list(Set(seq))
        if sort:
            return sorted({}.fromkeys(seq).keys())
        return {}.fromkeys(seq).keys()
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
        print ', '.join(['{0:>{width}}'.format(fmt(name), width=(kwargs['width']+2)*(len(arg[0]))-2)
                         if isinstance(arg[0], collections.Iterable) and not isinstance(arg[0], str) else
                         '{0:>{width}}'.format(fmt(name), width=kwargs['width'])
                         for name,arg in zip(names,args)])
    
    
    # import pdb; pdb.set_trace()
    for i,group in enumerate(zip(*args)):
        group = _flatten(group)
        
        print ', '.join(['{0: {w}.{p}f}'.format(item, w=kwargs['width'], p=kwargs['precision']) 
                         if not isinstance(item,(str,bool)) else 
                         '{0:>{width}}'.format(str(item), width=kwargs['width'])
                         for item in group ])
        if kwargs['n'] is not None and i > kwargs['n']:
            print ' ... [> %i rows]'%(kwargs['n'])
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
    
    getstr=True -- return the string without printing to stdout
    stack=1 -- go up the stack some number of values
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
        
    s = inspect.stack()
    try:
        module = inspect.getmodule(s[1][0]).__name__
    except AttributeError:
        module = '__main__'
    
    module = '<main>' if module == '__main__' else module
    name = s[1+kwargs.get('stack',0)][3]
    prefix = ' '*(len(s)-2)
    
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
            while len(prefix) > 10:
                prefix = '>' + prefix.split()[0] + prefix[10:]
        except:
            pass
        prefix = '%s>'%(os.getpid())+prefix
    
    output = ('%s'%(prefix) + 
              f('%s.%s '%(module,name))+
              ' '.join('%s'%s for s in args))
    
    if kwargs.get('getstr',False):
        return output 
    
    puts(output)






