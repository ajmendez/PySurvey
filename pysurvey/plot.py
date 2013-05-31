# plot.py : some nice plotting tools
# Mendez 05.2013

# System Libraries
import os

# Installed Libraries
import pylab
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
import numpy as np

# Package Imports
from util import splog
from file import nicefile

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
    def __init__(self, filename, ext='.pdf'):
        '''Makes a nice little pdf'''
        if not os.path.isabs(filename):
            filename = os.path.join(OUTDIR,filename+ext)
        self.pdf = PdfPages(filename)
    
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
    '''Set a better legend'''
    kwargs.setdefault('numpoints',1)
    kwargs.setdefault('prop',{'size':textsize})
    return pylab.legend(**kwargs)
    
    
    

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
          adjust=True, hspace=0.1, wspace=0.1,
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
          isinstance(subplt, matplotlib.gridspec.GridSpec) ):
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
            if isinstance(subplt, matplotlib.gridspec.SubplotSpec):
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
        # ax.set_xlabel('') 
        pylab.setp(ax.get_xticklabels(), visible=False)
        
    if yticks is False:
        # ax.set_yticklabels([])
        # ax.set_ylabel('')
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
    
    # temp
    return ax



