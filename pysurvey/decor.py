# some nice decorations
from functools import wraps
from pysurvey import util


def timeit(fcn):
    '''print out a nice length of time.'''
    
    @wraps(fcn)
    def wrapper(*args, **kwargs):
        # get a nice list of arguments -- defaults to nothing if none
        name = fcn.func_name
        tmp = ', '.join(map(str, args))
        if len(tmp) > 0: tmp = '[{}]'.format(tmp)
        # Arguments for splog making things look all nice.
        kw = {'color':'green', 'name':name, 'stack':2}
        
        start = util.deltatime()
        util.splog('Starting:', name, tmp, **kw)
        results = fcn(*args, **kwargs)
        util.splog('Finished running:',tmp, util.deltatime(start), **kw)
        
        return results 
        
    return wrapper


def filterkeys(fcn):
    '''Filter out unneeded kwargs
    is fcn is well defined (not using kwargs then it will attempt to just take the right bits from kwargs)'''
    @wraps(fcn)
    def wrapper(*args, **kwargs):
        out, tmp = util.getargs(fcn, **kwargs)
        return fcn(*args, **out)
    
    return wrapper





## {{{ http://code.activestate.com/recipes/284742/ (r4)
def expecting():
    """
    Call within a function to find out how many output arguments the caller of the function
    is expecting.  This is to emulate matlab's nargout.
    """
    import inspect,dis
    f = inspect.currentframe()
    f = f.f_back.f_back.f_back
    c = f.f_code
    i = f.f_lasti
    bytecode = c.co_code
    instruction = ord(bytecode[i+3])
    if instruction == dis.opmap['UNPACK_SEQUENCE']:
        howmany = ord(bytecode[i+4])
        return howmany
    elif instruction == dis.opmap['POP_TOP']:
        return 0
    return -1
## end of http://code.activestate.com/recipes/284742/ }}}


def filteroutput(fcn):
    '''Returns the first n items to squash it down.
    x = fcn() -- returns everything
    x,y = fcn() -- returns first two
    x,y,z = fcn() -- returns first three and so forth
    '''
    
    @wraps(fcn)
    def wrapper(*args, **kwargs):
        nout = expecting()
        results = fcn(*args, **kwargs)
        if nout > 0:
            return results[:nout]
        else:
            return results
        
    return wrapper
    
        
        
        