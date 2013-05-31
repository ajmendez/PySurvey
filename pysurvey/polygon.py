# polygon.py -- some nice polygon / windowfunction tools

# [System]

# [Installed]
import numpy as np

# [Package]
from location import radec2xy
from mangle import Mangle

# [Constants]







def inwindow(ra, dec, header, window):
    '''looks through the RA and Dec to see if they are within
    the window function. The window function is an array with associated
    header that specifies the values at each point.'''
    iswindow = np.zeros(len(ra))
    
    xx,yy = radec2xy(header, ra,dec)
    for i, (x,y) in enumerate(zip(xx,yy)):
        try:
            iswindow[i] = window[np.floor(y),np.floor(x)]
        except IndexError as e:
            pass
    return (iswindow > 0)


def inpolygon(ra, dec, polygon):
    '''Returns a boolean array if the ra,dec is in the polygon'''
    return (polygon.get_weights(ra,dec) > 0)


def loadpolygon(filename):
    return Mangle(filename)
        