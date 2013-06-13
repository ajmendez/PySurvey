# location.py -- some nice location handling utilities
# [System]

# [Installed]
import numpy as np
import pywcsgrid2
import cosmolopy

# [Package]

# [Constants]

def _cosmology():
    '''returns the values that match kCorrect and idl'''
    # p = cosmolopy.fidcosmo # default values from cosmolpy
    p  = {
        'h': 0.7,
        'omega_M_0':0.3, 
        'omega_lambda_0':0.7,
        'sigma_8':0.8,
    }
    return p

def radec2xy(header, ra, dec):
    '''From a header, and radec position, return the x,y position relative to an image'''
    
    t = pywcsgrid2.wcs_transforms.WcsSky2PixelTransform(header)
    ll = np.vstack((ra,dec)).T    
    xy = t.transform(ll)
    return xy[:,0], xy[:,1]


def xy2radec(header, xx, yy=None):
    '''Returns (ra, dec) for a set of x and y values that match a wcs header.
    you can pass in xy = np.zeros((n,2)) array as xx and leave yy as None'''
    if yy is None:
        xy = xx
    else:
        xy = np.vstack((xx,yy)).T
    
    t = pywcsgrid2.wcs_transforms.WcsPixel2SkyTransform(header)
    radec = t.transform(xy)
    return radec[:,0], radec[:,1]



def convert2distance(ra, dec, z, center):
    '''Convert ra,dec,z into Mpc/h units
    
    '''    
    p = _cosmology()
    dt = cosmolopy.distance.comoving_distance_transverse(z, **p)*p['h']
    rx = dt*np.radians(ra - center[0])
    ry = dt*np.radians(dec - center[1])
    rz = cosmolopy.distance.comoving_distance(z, **p)*p['h']
    return rx,ry,rz