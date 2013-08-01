# math.py -- some nice math utilites

# [System]

# [Installed]
import numpy as np
import scipy
import scipy.signal
import scipy.optimize 
from scipy.interpolate import interp1d

# [Package]

# [Constants]


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




#http://www.scipy.org/Cookbook/SignalSmooth?action=AttachFile&do=get&target=cookb_signalsmooth.py
def gauss_kern(size, sizey=None):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = np.mgrid[-size:size+1, -sizey:sizey+1]
    g = np.exp(-(x**2/float(size) + y**2/float(sizey)))
    return g / g.sum()

def blur_image(im, n, ny=None) :
    """ blurs the image by convolving with a gaussian kernel of typical
        size n. The optional keyword argument ny allows for a different
        size in the y direction.
    """
    g = gauss_kern(n, sizey=ny)
    improc = scipy.signal.convolve(im, g, mode='same')
    return improc




class Parameter(object):
    '''A Parameter class for the fit() function. This is from 
    http://www.scipy.org/Cookbook/FittingData'''
    def __init__(self, value):
            self.value = value

    def set(self, value):
            self.value = value

    def __call__(self):
            return self.value

def fit(function, parameters, x, y, yerr=None):
    '''A simple fitting function to make it easier on the commandline / notebook.
    from http://www.scipy.org/Cookbook/FittingData
    
    example:
    # giving initial parameters
    mu = Parameter(7)
    sigma = Parameter(3)
    height = Parameter(5)

    # define your function:
    def f(x): return height() * exp(-((x-mu())/sigma())**2)

    # fit! (given that data is an array with the data to fit)
    fit(f, [mu, sigma, height], data)
    
    '''
    def f(params):
        i = 0
        for p in parameters:
            p.set(params[i])
            i += 1
        if yerr is not None:
            return (y - function(x))/yerr
        else:
            return y - function(x)
    
    p = [param() for param in parameters]
    
    xx, cov_x, infodict, mesg, ier = scipy.optimize.leastsq(f, p, full_output=True)
    if ier not in [1,2,3,4]:
        return [-999.0 for param in parameters]
    else:
        return [param() for param in parameters]








def extrap1d(interpolator):
    ''' http://stackoverflow.com/questions/2745329/how-to-make-scipy-interpolate-give-an-extrapolated-result-beyond-the-input-range'''
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)

    def ufunclike(xs):
        return scipy.array(map(pointwise, scipy.array(xs)))

    return ufunclike



