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


def saveplot(filename, clear=True, ext=None):
    '''Save the current figure to a specific filename.  If the
    filename is not an absolute path it uses the $PYSURVEY_FIGURE
    directory'''
    if ext is None:
        tmp = os.path.splitext(filename)
        if len(tmp[1]) == 0:
            ext = '.png'
        else:
            filename, ext = tmp

    if not os.path.isabs(filename):
        filename = os.path.join(OUTDIR, filename)
    filename += '.'+ext.replace('.','')
    
    
    pylab.savefig(filename, dpi=150)
    splog('Saved Figure:', filename)
    if clear:
        pylab.clf()



class PDF(object):
    def __init__(self, filename, ext='.pdf', **kwargs):
        '''Makes a nice little pdf'''
        if not os.path.isabs(filename):
            filename = os.path.join(OUTDIR,filename+ext)
        self.filename = filename
        self.pdf = PdfPages(filename)
        pylab.figure('pdf_fig', **kwargs)
            
    def __enter__(self):
        '''Start with with constructor.'''
        return self
    
    def __exit__(self, type, value, traceback):
        ''' With Constructor is done'''
        self.close()
    
    def save(self, clear=True, quiet=False):
        '''Save the current figure as a new page, and ready
        the figure for the next page'''
        self.pdf.savefig()
        if clear:
            pylab.clf()
        if not quiet:
            splog('Added Page:',self.filename)
            
    
    def close(self):
        '''Close out the pdf and any additional final items'''
        self.pdf.close()
        splog('Finished Figure:',self.filename)
    


def legend(handles=None, labels=None, 
           textsize=9, zorder=None, box=None, 
           reverse=False, **kwargs):
    '''Set a better legend
    zorder=int -- layer ordering
    box = T/F -- dra`w the box or not
    
    http://matplotlib.org/users/legend_guide.html
    '''
    kwargs.setdefault('numpoints',1)
    kwargs.setdefault('prop',{'size':textsize})
    
    args = []
    if handles is not None:
        args.append(handles)
    if labels is not None:
        args.append(labels)
    
    l = pylab.legend(*args, **kwargs) 
    
    if reverse:
        ax = pylab.gca()
        handles, labels = ax.get_legend_handles_labels()
        
        return legend(handles[::-1], labels[::-1], zorder=zorder, box=box, **kwargs)
    
    if l is None:
        return l
    
    if zorder is not None:
        l.set_zorder(zorder)
    
    if box is not None:
        l.draw_frame(box)
    
    return l
    
    
    

def setup(subplt=None, 
          xr=None, yr=None,
          xlog=False, ylog=False,
          xlabel=None, ylabel=None, 
          suptitle=None, suptitle_prop=None, 
          subtitle=None, subtitle_prop=None, subtitleloc=1,
          title=None,
          xticks=True, yticks=True, autoticks=False,
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
    
    you can pass in a gridspec
    '''
    
    
    ## Handle subplot being an int `223`, tuple `(2,2,3)` or gridspec 
    if subplt is None:
        ax = pylab.gca()
    elif isinstance(subplt, (int, gridspec.GridSpec, gridspec.SubplotSpec) ):
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
        if suptitle_prop is None:
            suptitle_prop = {}
        pylab.suptitle(suptitle, **suptitle_prop)
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
        if not ( ( (isinstance(subplt, tuple) and len(subplt) == 3) ) or 
                 ( (isinstance(subplt, gridspec.SubplotSpec)) ) ):
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
             levels=None, levellabels=None,
             size='2%', pad=0.02, ):
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
            if levellabels is None:
                levellabels = ['%2.2f'%x for x in levels]
            cb.set_ticklabels(levellabels)
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


def _crange(xmin, xmax, nbin):
    if xmin > xmax: xmin, xmax = xmax, xmin
    xmin, xmax = embiggen([xmin,xmax], 0.1)
    bins = np.linspace(xmin, xmax, nbin+1)
    delta = (bins[1] - bins[0])/2.0
    return bins,delta



def scontour(x,y, levels=None, nbin=20,
             frac_contour=False,
             fill_contour=True, 
             add_bar=False, 
             smooth=False, smoothlen=None,
             **kwargs):
    '''contour a scatter plot'''
    '''Contour the data for the bulk bits
    returns the outermost polygon
    '''
    
    tmp = {
        'color': '0.6',
        'alpha': 0.8,
        'cmap': pylab.cm.gray_r,
    }
    tmp.update(kwargs)
    
    # make sure that we have good bounds for the 2d contouring
    xmin,xmax,ymin,ymax = pylab.axis()
    xbin, xdelta = _crange(xmin, xmax, nbin)
    ybin, ydelta = _crange(ymin, ymax, nbin)

    # find the height map for the points
    H, _, _ = np.histogram2d(x, y, bins=(xbin,ybin))
    

    
    # sort by the cumulative number for each point in the Height map
    if frac_contour:
        if levels is None: 
            levels = np.linspace(0.1,1.0, 5)
        t = np.reshape(H,-1)
        ii = np.argsort(t)
        t = np.cumsum(t[ii]) / np.sum(t)
        H[np.unravel_index(ii,H.shape)] = t
    
    if levels is None:
        # levels = np.logspace(np.log10(0.2),np.log10(1.0), 5)
        levels = np.linspace(0.3*np.nanmax(H), np.nanmax(H)*1.05, 6)
    
    # plot the resulting contours X,Y are centers rather than edges
    X, Y = np.meshgrid(xbin[:-1]+xdelta, ybin[1:]-ydelta)
    if smooth:
        X,Y,H = _smooth(X,Y,H, smoothlen)
    
    if fill_contour:
        con = pylab.contourf(X,Y,H.T,levels, **tmp)
        conl = pylab.contour(X,Y,H.T,levels, colors='0.8', linewidths=1.1)
    else:
        con = pylab.contour(X,Y,H.T,levels, **tmp)
        conl = None
    
    if add_bar:
        label = kwargs.get('label', None)
        if frac_contour:
            # I hope you dont want to use levels below here becuase I am adjusting it
            levels = levels[1:-1]
            labels = ['%0.1f'%(1-t) for t in levels]
            if label is None:
                label = 'Fraction of Sample'
            
        cb = colorbar(con, conl, clabel=label, levels=levels, levellabels=labels)
        
        if frac_contour:
            cb.ax.invert_yaxis()
    else:
        cb = None
    
    return con, cb












### Helper function things



def line(x=None, y=None, r=None, **kwargs):
    '''X,Y Arrays of lines to plot, r is the range of the line.'''
    xmin,xmax,ymin,ymax = pylab.axis()
    kwargs.setdefault('color','orange')
    kwargs.setdefault('linestyle','-')
    kwargs.setdefault('linewidth', 2)
    
    if x is not None:
        yr = [ymin, ymax] if r is None else r
        if isinstance(x, (float, int)):
            x = [x]
        
        for a in x:
            pylab.plot(np.ones(2)*a, yr, **kwargs)
    
    if y is not None:
        xr = [xmin, xmax] if r is None else r
        if isinstance(y, (float, int)):
            y = [y]
        
        for a in y:
            pylab.plot(xr, np.ones(2)*a, **kwargs)







### sky plots
def sky(ra, dec, **kwargs):
    ''' Basemap setup'''
    
    






