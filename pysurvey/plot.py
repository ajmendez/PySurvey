# plot.py : some nice plotting tools
# Mendez 05.2013

# System Libraries
import os
import copy

# Installed Libraries
import pylab
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
import mpl_toolkits.axes_grid1 as axes_grid1

# Package Imports
from util import splog, embiggen, minmax
from file import nicefile
from math import blur_image

OUTDIR = nicefile('$PYSURVEY_FIGURE')


def saveplot(filename, clear=True, ext='.png', paper=False):
    '''Save the current figure to a specific filename.  If the
    filename is not an absolute path it uses the $PYSURVEY_FIGURE
    directory'''
    if not os.path.isabs(filename):
        if paper:
            outname = (OUTDIR, 'paper', filename+ext)
        else:
            outname = (OUTDIR, filename+ext)
        filename = os.path.join(*outname)
    
    pylab.savefig(filename, dpi=150)
    splog('Saved Figure: ', filename+ext)
    if clear:
        pylab.clf()



class PDF(object):
    def __init__(self, filename, ext='.pdf', **kwargs):
        '''Makes a nice little pdf'''
        if not os.path.isabs(filename):
            filename = os.path.join(OUTDIR,filename+ext)
        self.pdf = PdfPages(filename)
        pylab.figure('pdf_fig', **kwargs)
            
    def __enter__(self):
        '''Start with with constructor.'''
        return self
    
    def __exit__(self, type, value, traceback):
        ''' With Constructor is done'''
        self.close()
    
    def save(self, clear=True):
        '''Save the current figure as a new page, and ready
        the figure for the next page'''
        self.pdf.savefig()
        if clear:
            pylab.clf()
    
    def close(self):
        '''Close out the pdf and any additional final items'''
        self.pdf.close()
    


def legend(textsize=9, **kwargs):
    '''Set a better legend
    zorder=int -- layer ordering
    box = T/F -- draw the box or not'''
    zorder = kwargs.pop('zorder',None)
    box = kwargs.pop('box', None)
    kwargs.setdefault('numpoints',1)
    kwargs.setdefault('prop',{'size':textsize})
    
    l = pylab.legend(**kwargs) 
    
    
    if zorder is not None:
        l.set_zorder(zorder)
    
    if box is not None:
        l.draw_frame(box)
    return l
    
    
    

def setup(subplt=None, 
          xr=None, yr=None,
          xlog=False, ylog=False,
          xlabel=None, ylabel=None, 
          suptitle=None,
          subtitle=None, subtitle_prop=None, subtitleloc=1,
          title=None,
          xticks=True, yticks=True, autoticks=False,
          subplot=None,
          grid=True, tickmarks=True, font=True,
          adjust=True, hspace=0.1, wspace=0.1, aspect=None
          ):
    '''Setup some nice defaults so that we are all fancy like
    
    xr,yr -- xrange and yrange.  by setting this it turns off the autoscale feature in that axis
    xlog,ylog -- T/F I do love me my log log plots
    
    xlabel,ylabel,title -- Set some nice text for the different axis
    
    xticks, yticks -- set to False if you want to hide the tickmarks and labels
    
    grid -- Turn on the grid in a nice way
    
    tickmarks -- make me some nick minor and major ticks
    '''
    
    
    ## Handle subplot being an int `223`, tuple `(2,2,3)` or gridspec 
    if subplt is None:
        ax = pylab.gca()
    elif (isinstance(subplt, int) or 
          isinstance(subplt, gridspec.GridSpec) ):
        ax = pylab.subplot(subplt)
    else:
        ax = pylab.subplot(*subplt)
    
    # Ranges -- Setting either xr,yr stops the auto ranging
    if xr is not None:
        ax.set_xlim(xr)
        pylab.autoscale(False, 'x', True)
    if yr is not None:
        ax.set_ylim(yr)
        pylab.autoscale(False, 'y', True)
    
    # Log stuff -- do this afterwards to ensure the minor tick marks are updated
    # can set the specific ticks using subsx, subsy -- 
    #   ax.set_xscale('log', subsx=[2, 3, 4, 5, 6, 7, 8, 9])
    # Look here: http://www.ianhuston.net/2011/02/minor-tick-labels-in-matplotlib
    # clip ensures that lines that go off the edge of the plot are shown but clipped
    if xlog:
        ax.set_xscale('log', nonposx='clip')
    if ylog:
        ax.set_yscale('log', nonposy='clip')

    
    # Labels
    if xlabel is not None:
        pylab.xlabel(xlabel)
    if ylabel is not None:
        pylab.ylabel(ylabel)
    if title is not None:
        pylab.title(title)
    if suptitle is not None:
        pylab.suptitle(suptitle)
    if subtitle is not None:
        if subtitleloc == 1:
            prop = {'location':(0.95,0.95),
                    'horizontalalignment':'right',
                    'verticalalignment':'top',
                    'transform':ax.transAxes}
        elif subtitleloc == 3:
            prop = {'location':(0.05,0.05),
                    'horizontalalignment':'left',
                    'verticalalignment':'bottom',
                    'transform':ax.transAxes}
        else: 
            raise NotImplementedError('Get to work adding the following subtitle location: %d'%(subtitleloc))
        if subtitle_prop is not None:
            prop.update(subtitle_prop)
        loc = prop.pop('location')
        pylab.text(loc[0], loc[1], subtitle, **prop)
    
    
    
    
    # Axis hiding
    if autoticks is True:
        if not (isinstance(subplt, tuple) and len(subplt) == 3):
            splog('Cannot setup auto ticks without a proper subplot')
        else:
            if isinstance(subplt, gridspec.SubplotSpec):
                rows,cols,i, _ = subplt.get_geometry()
                i += 1 # i is 0 indexed.
            else:
                rows,cols,i = subplt
            
            if ( (i%cols) != 1 ):
                yticks = False
            if ( i < (cols*(rows-1) + 1) ):
                xticks = False
    
    
    # Tickmark hiding -- used by autoticks as well.
    if xticks is False:
        # ax.set_xticklabels([])
        ax.set_xlabel('') 
        pylab.setp(ax.get_xticklabels(), visible=False)
        
    if yticks is False:
        # ax.set_yticklabels([])
        ax.set_ylabel('')
        pylab.setp(ax.get_yticklabels(), visible=False)
    
    
    
    # some nice defaults
    if grid:
        pylab.grid(b=True, which='major', linestyle='solid', color='0.3', alpha=0.5)
        ax.set_axisbelow(True)
    
    if tickmarks:
        ax.tick_params('both', which='major', length=5, width=2)
        ax.tick_params('both', which='minor', length=3, width=1)
        ax.minorticks_on()
    
    if adjust:
        pylab.subplots_adjust(hspace=hspace, wspace=wspace)
    
    if font:
        # this in theory should work, but fails becuase of the stupidify known as `_`
        # pylab.rc('font', **{'family':'sans-serif', 'sans-serif':['Helvetica']})
        # prop = {'family' : 'normal', 'weight' : 'bold', 'size'   : 22}
        # pylab.rc('font', **{'family':'serif', 'serif':['Computer Modern Roman']})
        # pylab.rc('text', usetex=True)
        pass
    
    if aspect is not None:
        # 'auto', 'equal'
        ax.set_aspect(aspect)
        print aspect
    
    
    # temp
    return ax












### Spatial things


def _getextent(extent, X,Y):
    '''Return the extent :: extent = (xmin, xmax, ymin, ymax) '''
    if extent is None:
        extent = minmax(X,nan=False) + minmax(Y, nan=False)
    return extent
    
def _getvrange(vrange, Z, p=0.05):
    '''Return the vertical / value range'''
    if vrange is None:
        vrange = embiggen(minmax(Z, nan=False), p, mode='upper')
    return vrange

def _getcmap(cmap):
    '''Build a nice colormap'''
    if cmap is None:
        cmap = copy.copy(pylab.cm.gray_r)
        cmap.set_bad('r',1.)
    return cmap

def colorbar(a, b=None, clabel=None, 
             levels=None, size='2%', pad=0.02, ):
    '''Builds a nice colorbar a is an image or contour or scalable,
    b is another scaleable -- eg contour lines'''
    ax = pylab.gca()
    divider = axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)
    
    cb = pylab.colorbar(a, cax=cax)
    if b is not None:
        cb.add_lines(b)
        if levels is not None:
            cb.set_ticks(levels)
            cb.set_ticklabels(['%2.2f'%x for x in levels])
    if clabel is not None:
        cb.set_label(clabel)
    pylab.sca(ax)
    return cb
    


def image(X,Y,Z, origin='lower', interpolation='nearest',
          vrange=None, extent=None, 
          cmap=None, addcolorbar=True, clabel=None):
    '''A nice wrapper around imshow.
    vrange = (vmin, vmax) -- Z value range
    cmap = colormap
    
    '''
    
    # calculate the spatial location and vertical / value range
    extent = _getextent(extent, X, Y)
    vrange = _getvrange(vrange, Z)
    
    # get a nice color map with nans set to be
    cmap = _getcmap(cmap)
    
    # Make the masked array hiding the nans
    MZ = np.ma.array(Z, mask=(np.isfinite(Z) is False) )
    
    out = []
    
    im = pylab.imshow(MZ, origin=origin, extent=extent,
                      vmin=vrange[0], vmax=vrange[1],
                      cmap=cmap, interpolation='nearest')
    
    # setup a colorbar and return it out for modifications
    if addcolorbar:
        cb = colorbar(im, clabel=clabel)
        return im, cb
    
    return im



def _getlevels(levels, vrange):
    if levels is None:
        if vrange[0] < 0:
            levels = np.linspace(vrange[0],vrange[1], 15)
        else:
            levels = np.logspace(np.log10(vrange[0]), np.log10(vrange[1]), 7)
    return levels

def _smooth(X,Y,Z, smoothlen):
    if smoothlen is None:
        smoothlen = 1.0
    
    if len(X.shape) == 1 and len(Y.shape) == 1:
        X, Y = np.meshgrid(X,Y)
    out = []
    for A in (Z, ):
        out.append(blur_image(A, smoothlen))
    # print len(X), len(Z), len(out[0])
    # return out
    return X,Y, out[0]
        


def contour(X,Y,Z, 
           extent=None, vrange=None, levels=None, extend='both', 
           cmap=None, addcolorbar=True, clabel=None,
           smooth=True, smoothlen=None):
    '''Build a super fancy contour image'''
    
    # Build up some nice ranges and levels to be plotted 
    extent = _getextent(extent, X, Y)
    vrange = _getvrange(vrange, Z)
    levels = _getlevels(levels, vrange)
    cmap   = _getcmap(cmap)
    
    # Smooth if needed
    if smooth:
        X,Y,Z = _smooth(X,Y,Z, smoothlen)
    
    cs = pylab.contourf(X, Y, Z, levels,
                       vmin=vrange[0], vmax=vrange[1],
                       extent=extent, extend='both',
                       cmap=cmap)
    ccs = pylab.contour(X, Y, Z, levels, vmin=vrange[0], vmax=vrange[1],
                        cmap=cmap)
    
    # setup a colorbar, add in the lines, and then return it all out.
    if addcolorbar:
        cb = colorbar(cs, ccs, levels=levels, clabel=clabel)
        return cs, ccs, cb
    
    return cs, ccs



### Helper function things



def line(x=None, y=None, **kwargs):
    '''X,Y Arrays of lines to plot'''
    xmin,xmax,ymin,ymax = pylab.axis()
    kwargs.setdefault('color','orange')
    kwargs.setdefault('linestyle','-')
    kwargs.setdefault('linewidth', 2)
    
    if x is not None:
        if isinstance(x, (float, int)):
            x = [x]
        for a in x:
            pylab.plot(np.ones(2)*a, [ymin, ymax], **kwargs)
            print np.ones(2)*a, [ymin, ymax]
    if y is not None:
        if isinstance(y, (float, int)):
            y = [y]
        for a in y:
            pylab.plot([xmin, xmax], np.ones(2)*a, **kwargs)







### sky plots
def sky(ra, dec, **kwargs):
    ''' Basemap setup'''
    
    






