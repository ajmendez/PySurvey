# bins.py -- nice binning functions and the sort

# [System]

# [Installed]
import scipy
import numpy as np


# [Package]

# [Constants]


# TODO clean up.  a bunch of this is not being used or redundant.



class Bins(object):
    def __init__(self, name, xmin,xmax,dx,log):
        '''The Bin object is just a nice little container for using the histogram functions
        by default it stores the left edge as bin.array with .min, .max, .delta, and .log 
        properties.  If log is set it updates the .rmin, .rmax and .rarray to the real (not log)
        space sizes of these things.  It stores the centers of the bins under .center
        len(bin) returns the proper length.'''
        self.name = name
        self.min = xmin
        self.max = xmax
        self.delta = dx 
        self.log = log
        self.uniform = True # Uniformly spaced bins
        
        self.array = np.arange(self.min,self.max,self.delta)  # left edge
        self._update()
    
    
    def _update(self):
        ''' Modular the nice functions so that they can be run afterthefact'''
        self.edges = np.concatenate((self.array,[self.max]))
        self.diff = np.diff(self.edges)
        
        self.centers = self.array + self.delta/2.0 # center edge
        self.rmin, self.rmax = self.min, self.max# some nice things
        self.rarray, self.redges = self.array, self.edges
        if self.log:
            self.rmin = 10**self.min
            self.rmax = 10**self.max
            self.rarray = 10**self.array
            self.redges = 10**self.redges
            self.centers = 10**self.centers
            self.diff = np.diff(10**self.edges)
    
    
    def __len__(self):
        '''Make sure that we can just take the len() of a bin object'''
        return len(self.array)
        
        
    def __str__(self):
        return ', '.join(['% .2f'%t for t in self.array])
    
    
    def __getitem__(self,items):
        '''Get the centers by default -- this might break things!!!!!!'''
        # print items
        # print self.log, self.centers
        return self.centers[items]
    
    
    @classmethod
    def _fromarray(self, name, array, log=False, xmin=None):
        '''If we need to generate a Bin object from an array here is the way.'''
        dx = array[1] - array[0]
        # xmin, xmax = array[0], array[-1]#+dx # failed on even/oddness
        if xmin is None:
            xmin = array[0]
        else:
            xmin = np.array([xmin])
        xmax = np.floor((array[-1]-array[0])/dx + 1.0)*dx + array[0]
        # There was somthing fucked up without rounding, fuck.
        return Bins(name, xmin.round(3),xmax.round(3),dx.round(3), log)
    
    @classmethod
    def fromdata(self, item, log=None, xmin=None):
        array = item.data
        if log is None:
            # do not expect 10^5 as a real MPC length
            log = (np.max(array) < 5) 
        if array[1]-array[0] != array[-1]-array[-2]:
            # not uniform
            # splog('not uniform: %s'%item.name.lower())
            
            return self.fromcenters(item.name.lower(), 10**array)
        else:
            # splog('uniform: %s'%item.name.lower())
            return self._fromarray(item.name.lower(), array, log, xmin=xmin)
    
    @classmethod
    def fromcenters(self, name, array):
        x = self._fromarray(name,array)
        x.min = np.log10(array[0] - abs(array[1]-array[0])/2.0)
        x.max = np.log10(array[-1] + abs(array[-1]-array[-2])/2.0)
        x.log = True
        x.uniform = False
        x.array = np.concatenate( ([x.min], 
                                   [np.log10(np.mean(array[i:i+2])) for i in range(len(array)-1)]) )
        x._update()
        x.delta = None
        x.centers = array
        return x
        




### Specialized histograming to make things faster. These are 

def hist(x, weight, xbin):
    ''' Get the histogram of locations quickly without too much memory or time.  This is a
    1d version of the hist2d program below.  it defaults to uniform '''
    ii = np.where((x > xbin.rmin) & (x < xbin.rmax) & (weight > 0))
    xx = np.log10(x[ii]) if xbin.log else x[ii]
    ww = weight[ii]
    
    if not xbin.uniform:
        value, xedge = np.histogram(xx, xbin.edges, weights=ww)
        return value.T
    
    xx -= xbin.min
    xx /= xbin.delta
    xx = np.floor(xx)
    xx = np.vstack((xx, np.zeros(xx.size)))
    
    grid = scipy.sparse.coo_matrix((ww,xx), shape=(len(bin),1)).toarray()
    return grid.T[0,:] # Flatten the matrix



def hist2d(x, y, weight, xbin, ybin):
    '''http://stackoverflow.com/questions/8805601/efficiently-create-2d-histograms-from-large-datasets
    Efficient hist binning of the data.  Defaults to using np.histogram2d if the bins are not 
    uniform
    
    '''
    ii = np.where( (x > xbin.rmin) & (x < xbin.rmax) & 
                   (y > ybin.rmin) & (y < ybin.rmax) & 
                   (weight > 0) )
    if len(ii[0]) == 0:
        raise ValueError('No Sources found in within bins')
    
    ww = weight[ii]
    xx = np.log10(x[ii]) if xbin.log else x[ii]
    yy = np.log10(y[ii]) if ybin.log else y[ii]
    
    if not xbin.uniform or not ybin.uniform:
        value, xedge, yedge = np.histogram2d(xx,yy, (xbin.edges, ybin.edges), weights=ww)
        return value.T
    
    # Basically, this is just doing what np.digitize does with one less copy
    xyi = np.vstack((xx,yy)).T
    xyi -= [xbin.min, ybin.min]
    xyi /= [xbin.delta, ybin.delta]
    xyi = np.floor(xyi, xyi).T
    
    # Now, we'll exploit a sparse coo_matrix to build the 2D histogram...
    grid = scipy.sparse.coo_matrix((ww, xyi), shape=(len(xbin), len(ybin))).toarray()
    
    # return grid, np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
    return grid.T