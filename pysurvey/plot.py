# plot.py : some nice plotting tools
# Mendez 05.2013

# System Libraries
import os
import copy
import warnings
from datetime import datetime

# Installed Libraries
import pylab
import numpy as np
import collections
import matplotlib
import matplotlib.ticker
from matplotlib import ticker
from matplotlib.dates import date2num, num2date
import matplotlib.gridspec as gridspec
import mpl_toolkits.axes_grid1 as axes_grid1
import matplotlib.patheffects as PathEffects
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


from matplotlib.patches import Polygon
from matplotlib.collections import PolyCollection, PatchCollection


# Package Imports
from pysurvey import util, file, math, oldmangle
# from util import splog, embiggen, minmax, getargs
# from file import nicefile
# from math import blur_image
# from oldmangle import Mangle
minmax = math.minmax
embiggen = math.embiggen

OUTDIR = file.nicefile('$PYSURVEY_FIGURE')


def quick(*args, **kwargs):
    '''A simple plot to make things quick. 
    separates out the setup arguments so you can do things like 
    xr=[2,3] and the sort'''
    tmp, kwargs2 = util.getargs(setup, **kwargs)
    tmp2 = { 'marker':',', 'alpha':0.5 }
    tmp2.update(kwargs2)
    
    pylab.clf()
    setup(**tmp)
    pylab.plot(*args, **tmp2)
    pylab.show()

def plothist(*args, **kwargs):
    '''simple little histogram that can be run from python bare.
    TODO: add in delta=[xmin,xmax,dx] to do a different style of binning.'''
    delta = kwargs.pop('delta', None)
    
    if delta is not None:
        kwargs['bins'] = np.arange(*delta)
        
    tmp, kwargs2 = util.getargs(setup, **kwargs)
    tmp2 = {'alpha':0.8}
    tmp2.update(kwargs2)
    
    pylab.clf()
    setup(**tmp)
    pylab.hist(*args, **tmp2)
    pylab.show()

def hist(x,bins, weight=None, weights=None, index=None, 
         norm=None, frac=False, total=False, dist=False,
         cumulative=False, revcumulative=False,
         bottom=None, filled=False, 
         **kwargs):
    '''
    norm == set max value of hist
    frac == return the fraction index is of the entire sample 
    
    '''
    rotate = kwargs.pop('rotate',False)
    noplot = kwargs.pop('noplot',False)
    if bottom is None: bottom=0.0
    if index is not None:
        xx = x[index]
        if weight is not None and weights is not None:
            raise ValueError('Only supply one.  This is just so that you can push weight and weights')
        if weight is not None:
            ww = weight[index]
        elif weights is not None:
            ww = weights[index]
        else:
            ww = None
    else:
        xx = x
        ww = weight
    
    v,l = np.histogram(xx,bins,weights=ww)
    d = np.diff(l)
    l = l[:-1] + d/2.0
    
    if frac:
        vv = np.histogram(x,bins, weight)[0]
        ii = np.where(vv == 0)
        v = np.array(v)*1.0/np.array(vv)
        v[ii] = 0
    
    if cumulative:
        v = np.cumsum(v)
    if revcumulative:
        v = np.cumsum(v[::-1])[::-1] 
    
    if norm is not None:
        v = v/float(np.max(v))*float(norm)
    if total:
        v = v / (1.0*np.sum(v))
    
    if dist:
      v /= d
    
    if bottom is not None:
        v += bottom
    # if rotate:
    #     l,v = v,l
    if not noplot:
        if filled:
            kwargs.setdefault('align', 'center')
            # hack to fix pylab.bar's coloring 
            if 'color' not in kwargs:
                kwargs['color'] = next(pylab.gca()._get_lines.color_cycle)
            if rotate:
                pylab.barh(l,v-bottom, height=d, left=bottom, **kwargs)
            else:
                pylab.bar(l,v-bottom, width=d, bottom=bottom, **kwargs)
        else:
            if rotate:
                pylab.step(v,l, where='mid', **kwargs)
            else:
                pylab.step(l,v, where='mid', **kwargs)
    # if rotate:
    #     l,v = v,l
    return l,v



def saveplot(filename, clear=True, ext=None, nice=False, quiet=False):
    '''Save the current figure to a specific filename.  If the
    filename is not an absolute path it uses the $PYSURVEY_FIGURE
    directory'''
    if nice:
        filename = nicefile(filename)
    
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
    if not quiet:
        util.splog('Saved Figure:', filename)
    if clear:
        pylab.clf()



class PDF(object):
    def __init__(self, filename, ext='.pdf', **kwargs):
        '''Makes a nice little pdf'''
        if not os.path.isabs(filename):
            filename = os.path.join(OUTDIR,filename+ext)
        self.filename = filename
        self.pdf = PdfPages(filename)
        
        # pylab.figure('pdf_fig', **kwargs)
            
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
            util.splog('Added Page:',self.filename)
            
    
    def close(self):
        '''Close out the pdf and any additional final items'''
        self.pdf.close()
        util.splog('Finished Figure:',self.filename)
    


def legend(handles=None, labels=None, 
           textsize=9, zorder=None, box=None, 
           alpha=None,
           reverse=False, ax=None, **kwargs):
    '''Set a better legend
    zorder=int -- layer ordering
    box = T/F -- draw the box or not
    
    http://matplotlib.org/users/legend_guide.html
    
    http://matplotlib.org/users/recipes.html
    '''
    kwargs.setdefault('fancybox', True)
    kwargs.setdefault('numpoints',1)
    kwargs.setdefault('prop',{'size':textsize})
    
    if ax is None:
        ax = pylab.gca()
    
    args = []
    if handles is not None:
        args.append(handles)
    if labels is not None:
        args.append(labels)
    
    l = ax.legend(*args, **kwargs) 
    
    if reverse:
        handles, labels = ax.get_legend_handles_labels()
        
        return legend(handles[::-1], labels[::-1], zorder=zorder, box=box, **kwargs)
    
    if l is None:
        return l
    
    if zorder is not None:
        l.set_zorder(zorder)
    
    if box is not None:
        l.draw_frame(box)
    
    if alpha is not None:
        l.get_frame().set_alpha(alpha)
    
    return l



def icolorbar(mappable, loc=2, orientation='horizontal', borderpad=1,
              width=None, height=None, ax=None, 
              label=None, axes_kwargs=None,
              **kwargs):
    '''An inset colorbar.  Should be a simple replacement for colorbar
    orientation='horizontal', width='30%', height='2%'
    orientation='vertical', width='5%', height='25%'
    
    bbox_to_anchor=(1.05, 0., 1, 1),
    bbox_transform=ax2.transAxes,
    borderpad=0,
    
    http://matplotlib.org/examples/axes_grid/demo_colorbar_with_inset_locator.html
    '''
    
    if ax is None: 
        ax = pylab.gca()
    if width is None:
        width = '30%' if orientation == 'horizontal' else '2%'
    if height is None:
        height = '2%' if orientation == 'horizontal' else '25%'
        
    if label is None:
        try:
            label = mappable.get_label()
            if label.startswith('_'):
                # hide internal names
                label = ''
        except:
            label = ''
    
    
    
    tmp = dict(
        nticks=3,
        ticks=None,
        ticknames=None,
        tickfmt=None,
        tickfcn=None,
        orientation=orientation,
        rotate=None,
        outline=True,
        outline_props=dict(),
        color=None,
    )
    tmp.update(kwargs)
    nticks = tmp.pop('nticks')
    ticks = tmp.pop('ticks')
    tickfmt = tmp.pop('tickfmt')
    tickfcn = tmp.pop('tickfcn')
    ticknames = tmp.pop('ticknames')
    rotate = tmp.pop('rotate')
    outlinetext = tmp.pop('outline')
    outline_props = tmp.pop('outline_props')
    color = tmp.pop('color')
    
    if 'cax' not in tmp:
        if axes_kwargs is None:
            axes_kwargs = {'borderpad':borderpad}
        tmp['cax'] = inset_axes(ax, width=width, height=height, loc=loc, **axes_kwargs)
    
    cb = pylab.colorbar(mappable, **tmp)
    cb.set_label(label)
    
    if ticks is None:
        ticks = np.linspace(cb.vmin, cb.vmax, nticks)
    cb.set_ticks(ticks)
    
    if tickfcn is not None:
        ticks = map(tickfcn, ticks)
    
    if tickfmt is not None:
        cb.set_ticklabels(map(tickfmt.format, ticks))
    
    if ticknames is not None:
        cb.set_ticklabels(ticknames)
    
    if rotate is not None:
        if not isinstance(rotate, (int,float)):
            rotate = 90
            
        pylab.setp(pylab.xticks(axes=cb.ax)[1], rotation=rotate, ha='center')
    
    if color is not None:
        if orientation == 'horizontal':
            pylab.setp(pylab.getp(cb.ax.axes, 'xticklabels'), color=color)
            pylab.setp(pylab.getp(cb.ax.xaxis, 'label'), color=color)
        else:
            pylab.setp(pylab.getp(cb.ax.axes, 'yticklabels'), color=color)
            pylab.setp(pylab.getp(cb.ax.yaxis, 'label'), color=color)
    
    if outlinetext:
        if orientation == 'horizontal':
            items = [cb.ax.xaxis.get_label(),
                     cb.ax.xaxis.get_ticklabels()]
        else:
            items = [cb.ax.yaxis.get_label(),
                     cb.ax.yaxis.get_ticklabels()]
        outline(items, **outline_props)
    
    
    pylab.sca(ax)
    return cb






def hcolorbar(*args, **kwargs):
    '''Arguments, label='', nticks=[3], cticks=[None], axes=[[0.8,0.01,0.1,0.02]] #[left,bottom, width,height], cax=[None] colorbar axis
    returns colorbar'''
    try:
        itemlabel = args[0].get_label()
        if itemlabel.startswith('_'):
            itemlabel = ''
    except:
        itemlabel = ''
    
    label = kwargs.pop('label', itemlabel)
    nticks = kwargs.pop('nticks', 3)
    cticks = kwargs.pop('cticks', None)
    cticknames = kwargs.pop('cticknames', None)
    rotate = kwargs.pop('rotate', None)
    
    tickfmt = kwargs.pop('tickfmt', None)
    axes = kwargs.pop('axes', [0.8,0.01,0.1,0.02])
    ax = pylab.gca()
    
    tmp = dict(
        # cax = pylab.axes(axes),
        orientation='horizontal',
    )
    tmp.update(kwargs)
    if 'cax' not in tmp:
        tmp['cax'] = pylab.axes(axes)
    
    cb = pylab.colorbar(*args, **tmp)
    cb.set_label(label)
    
    if cticks is None:
        cticks = np.linspace(cb.vmin, cb.vmax, nticks)
    cb.set_ticks(cticks)
    
    if tickfmt is not None:
        tmp = map(tickfmt.format, cticks)
        cb.set_ticklabels(tmp)
    
    if cticknames is not None:
        cb.set_ticklabels(cticknames)
    
    if rotate is not None:
        pylab.setp(pylab.xticks(axes=cb.ax)[1], rotation=90, ha='center')
        
    
    pylab.sca(ax)
    return cb

def setup(subplt=None, figsize=None, ax=None,
          xr=None, xmin=None, xmax=None,
          yr=None, ymin=None, ymax=None,
          xlog=False, ylog=False,
          xlabel=None, ylabel=None, 
          xtickv=None, xticknames=None, xtickrotate=None,
          ytickv=None, yticknames=None, ytickrotate=None,
          halfxlog=False,
          suptitle=None, suptitle_prop=None, 
          subtitle=None, subtitle_prop=None, subtitleloc=1, 
          title=None,
          xticks=None, yticks=None, autoticks=False,
          embiggenx=None, embiggeny=None,
          secondx=False, secondx_prop=None,
          secondy=False, secondy_prop=None,
          grid=True, tickmarks=True, font=True,
          adjust=True, hspace=0.1, wspace=0.1, aspect=None,
          rasterized=False,
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
    
    # if notebook:
    #   # http://matplotlib.org/users/customizing.html
    #   # http://damon-is-a-geek.com/publication-ready-the-first-time-beautiful-reproducible-plots-with-matplotlib.html
    #   matplotlib.rcParams['savefig.dpi'] = 144
    #   matplotlib.rcParams.update({'font.size': 12})
    #   # matplotlib.rcParams['font.family'] = 'serif'
    #   # matplotlib.rcParams['font.serif'] = ['Computer Modern Roman']
    #   # matplotlib.rcParams['text.usetex'] = True
    
    
    if figsize is not None:
        fig = pylab.figure(figsize=figsize)
    
    ## Handle subplot being an int `223`, tuple `(2,2,3)` or gridspec 
    try:
        if subplt is None:
            if ax is None:
                ax = pylab.gca()
        elif isinstance(subplt, (int, gridspec.GridSpec, gridspec.SubplotSpec) ):
            ax = pylab.subplot(subplt)
        else:
            ax = pylab.subplot(*subplt)
    except Exception as e:
        raise ValueError('Failed to setup subplt:[{}] -- probably indexing error'.format(subplt))
    
    
    
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
        prop = dict(transform=ax.transAxes)
        if subtitleloc == 1:
            prop.update({'location':(0.95,0.95),
                         'horizontalalignment':'right',
                         'verticalalignment':'top'})
        elif subtitleloc == 3:
            prop.update({'location':(0.05,0.05),
                         'horizontalalignment':'left',
                         'verticalalignment':'bottom'})
        else: 
            raise NotImplementedError('Get to work adding the following subtitle location: %d'%(subtitleloc))
        if subtitle_prop is not None:
            prop.update(subtitle_prop)
        loc = prop.pop('location')
        outline = prop.pop('outline',True)
        outlinewidth = prop.pop('linewidth',3.5)
        txt = pylab.text(loc[0], loc[1], subtitle, **prop)
        if outline:
            # pylab.setp(txt,path_effects=[PathEffects.Stroke(linewidth=outlinewidth, foreground="w")]) # Broken
            txt.set_path_effects(
                [PathEffects.Stroke(linewidth=outlinewidth, foreground="w"),
                 PathEffects.Normal()])
            # raise ValueError()
    
    
    if xtickv is not None:
        ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(xtickv))
        if xticknames is not None:
            ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(xticknames))
    if xtickrotate is not None:
        if xtickrotate == 'vertical':
            pylab.xticks(xtickv, xticknames, rotation='vertical')
        else:
            tmp = dict(rotation=30, ha='right')
            if isinstance(xtickrotate, dict):
                tmp.update(xtickrotate)
            else:
                tmp['rotation'] = xtickrotate
            pylab.setp(pylab.xticks()[1], **tmp)
    
    if ytickv is not None:
        ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ytickv))
        if yticknames is not None:
            ax.yaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(yticknames))
    if ytickrotate is not None:
        pylab.setp(pylab.yticks()[1], rotation=ytickrotate, ha='right')
        
    
    # Axis hiding
    if autoticks:
        if not ( ( (isinstance(subplt, tuple) and len(subplt) == 3) ) or 
                 ( (isinstance(subplt, gridspec.SubplotSpec)) ) ):
            splog('Cannot setup auto ticks without a proper subplot')
        else:
            if isinstance(subplt, gridspec.SubplotSpec):
                rows,cols,i, _ = subplt.get_geometry()
                i += 1 # i is 0 indexed.
            else:
                rows,cols,i = subplt
            
            if ( (i%cols) != 1 ) and (not yticks):
                yticks = False
            # if ( i < (cols*(rows-1) + 1) ) and (not xticks):
            #     xticks = False
            xticks = not ( ( i < (cols*(rows-1) + 1) ) and (not xticks) )
    
    
    
    
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
    if grid is True:
        pylab.grid(b=True, which='major', linestyle='solid', color='0.3', alpha=0.5)
        ax.set_axisbelow(True)
    if grid is False:
        pylab.grid('off')
    
    
    if tickmarks:
        ax.tick_params('both', which='major', length=5, width=2)
        ax.tick_params('both', which='minor', length=3, width=1)
        ax.minorticks_on()
    else:
        ax.minorticks_off()
        pylab.tick_params(axis='both',which='both',
                          bottom='off', top='off',
                          left='off', right='off')
        
    
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
    
    if embiggenx:
        setup(xr=math.embiggen(ax.axis()[:2],embiggenx))
    if embiggeny:
        setup(yr=math.embiggen(ax.axis()[2:],embiggeny))
    
    if (xticks) and (halfxlog):
        xax = ax.xaxis
        xax.set_minor_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
        
        tmp = (10.0**(np.arange(-1,5,1)))*5.0
        for x,label in zip(xax.get_minorticklocs(), xax.get_minorticklabels()):
            if x in tmp:
                label.set_fontsize(8)
            else:
                label.set_fontsize(0)
                pylab.setp(label, visible=False)
    

    if secondx:
        # second x axis
        tmp = dict(xlabel=None,
                   fcn=lambda arr: arr,
                   fmt='{}',
                   xr=ax.axis()[:2],
                   xlog=('log' in ax.get_xscale()),
                   # yr=ax.axis()[2:]
                   xtickv=ax.get_xticks(),
                   ) # defaults
        if secondx_prop:
            tmp.update(secondx_prop)
            
        # adjust some nice things
        f = tmp.pop('fcn')
        fcn = lambda arr: [f(x) for x in arr]
        fmt = tmp.pop('fmt')
        if isinstance(fmt, str): fmt = (fmt,fmt)
        
        if 'xtickv' in tmp:
            tmp['xtickv'] = np.array(tmp['xtickv'])
        
        if 'xticknames' not in tmp:
            arr = np.array(fcn(tmp['xtickv']))
            tmp['xticknames'] = np.array([fmt[0].format(x) for x in arr])
            if tmp['xlog'] and tmp['nicelog']:
                ii = np.where((arr >= 1e1)&
                              (arr < 1e4) )[0]
                tmp['xticknames'][ii] = [fmt[1].format(x) for x in arr[ii]]
                ii = np.where(arr >= 1e4)
                tmp['xticknames'][ii] = [fmt[2].format(np.log10(x)) for x in arr[ii]]
        
        ax2 = ax.twiny() # I have never understood this
        ax2.set_xlabel(tmp['xlabel'])
        ax2.set_xlim(tmp['xr'])
        if tmp['xlog']:
            ax2.set_xscale('log', nonposx='clip')
        if 'xtickv' in tmp:
            ax2.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(tmp['xtickv']))
        if 'xticknames' in tmp:
            ax2.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(tmp['xticknames']))
        
        ax.secondx = ax2
        pylab.sca(ax)

    if secondy:
        # second x axis
        tmp = dict(ylabel=None,
                   fcn=lambda arr: arr,
                   fmt='{}',
                   yr=ax.axis()[2:],
                   ylog=('log' in ax.get_yscale()),
                   ytickv=ax.get_yticks(), )
        if secondy_prop:
            tmp.update(secondy_prop)
            
        # adjust some nice things
        f = tmp.pop('fcn')
        fcn = lambda arr: [f(x) for x in arr]
        fmt = tmp.pop('fmt')
        if isinstance(fmt, str): fmt = (fmt,fmt)
        
        if 'ytickv' in tmp:
            tmp['ytickv'] = np.array(tmp['ytickv'])
        
        if 'yticknames' not in tmp:
            arr = np.array(fcn(tmp['ytickv']))
            tmp['yticknames'] = np.array([fmt[0].format(x) for x in arr])
            if tmp['ylog'] and tmp['nicelog']:
                ii = np.where((arr >= 1e1)&
                              (arr < 1e4) )[0]
                tmp['yticknames'][ii] = [fmt[1].format(x) for x in arr[ii]]
                ii = np.where(arr >= 1e4)
                tmp['yticknames'][ii] = [fmt[2].format(np.log10(x)) for x in arr[ii]]
            
        print tmp
        
        ax2 = ax.twinx() # I have never understood this
        ax2.set_ylabel(tmp['ylabel'])
        ax2.set_ylim(tmp['yr'])
        if tmp['ylog']:
            ax2.set_yscale('log', nonposx='clip')
        if 'ytickv' in tmp:
            ax2.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(tmp['ytickv']))
        if 'yticknames' in tmp:
            ax2.yaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(tmp['yticknames']))
            
        ax.secondy = ax2
        pylab.sca(ax)

    
    
    if rasterized:
        ax.set_rasterized(True)
    # temp
    return ax













### Spatial things

def _getmesh(X,Y,Z):
    if Z.shape != X.shape:
        return np.meshgrid(X,Y)
    else:
        return X,Y


def _getextent(extent, X,Y):
    '''Return the extent :: extent = (xmin, xmax, ymin, ymax) '''
    if extent is None:
        extent = math.minmax(X, nan=False) + math.minmax(Y, nan=False)
    return extent
    
def _getvrange(vrange, XX,YY,Z, inaxis=None, p=0.05):
    '''Return the vertical / value range'''
    if inaxis is None:
        inaxis = False
    
    if vrange is None:
        if inaxis:
            axis = pylab.gca().axis()
            xr = math.minmax(axis[:2])
            yr = math.minmax(axis[2:])
            # XX, YY = np.meshgrid(X,Y)
            ii = np.where( (XX >= xr[0]) &
                           (XX <= xr[1]) &
                           (YY >= yr[0]) &
                           (YY <= yr[1]) )
            vrange = math.embiggen(math.minmax(Z[ii], nan=False), p, mode='upper')
        else:
            vrange = math.embiggen(math.minmax(Z, nan=False), p, mode='upper')
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
          vrange=None, extent=None, inaxis=None,
          cmap=None, addcolorbar=True, clabel=None):
    '''A nice wrapper around imshow.
    vrange = (vmin, vmax) -- Z value range
    cmap = colormap
    
    '''
    
    # calculate the spatial location and vertical / value range
    XX, YY = _getmesh(X,Y,Z)
    extent = _getextent(extent, X, Y)
    vrange = _getvrange(vrange, XX,YY,Z, inaxis=inaxis)
    
    # get a nice color map with nans set to be
    cmap = _getcmap(cmap)
    
    # Make the masked array hiding the nans
    # MZ = np.ma.array(Z, mask=(np.isfinite(Z) is False) )
    MZ = np.ma.array(Z, mask=np.isnan(Z))
    
    out = []
    
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    #     im = pylab.imshow(MZ, origin=origin, extent=extent,
    #                       vmin=vrange[0], vmax=vrange[1],
    #                       cmap=cmap, interpolation='nearest')
    im = pylab.pcolormesh(XX,YY, MZ, 
                          vmin=vrange[0], vmax=vrange[1],
                          cmap=cmap)
    # box(y=[0,0.2], percent=True, color='r')
    # pylab.plot(np.medians(XX,axis=0), np.median(MZ,axis=0)
    # if addmedians:
    #     med = np.median(MZ,axis=0)
    #     box(y=[0,0.1], percent=True)
    
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
        out.append(math.blur_image(A, smoothlen))
    # print len(X), len(Z), len(out[0])
    # return out
    return X,Y, out[0]
        


def contour(X,Y,Z, 
           extent=None, vrange=None, levels=None, extend='both', 
           inaxis=None,
           cmap=None, addcolorbar=True, clabel=None,
           smooth=True, smoothlen=None):
    '''Build a super fancy contour image'''
    
    # Build up some nice ranges and levels to be plotted 
    XX, YY = _getmesh(X,Y,Z)
    extent = _getextent(extent, XX, YY)
    vrange = _getvrange(vrange, XX,YY,Z, inaxis=inaxis)
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
    xmin, xmax = math.embiggen([xmin,xmax], 0.1)
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

def outline(item, color='w', width='3'):
    '''Outlines an object'''
    if isinstance(item, (tuple, list)):
        for it in item:
            outline(it, color=color, width=width)
    else:
        item.set_path_effects(
            [PathEffects.Stroke(linewidth=width, foreground=color),
             PathEffects.Normal()])


def text(*args, **kwargs):
    '''Writes some text to the screen.
    use normal=True to normalize to [0,1], axes units
    outline=True to outline.  properties[color, width] given by outline_prop dict'''
    
    normal = kwargs.pop('normal', False)
    oline = kwargs.pop('outline', True)
    outline_prop = kwargs.pop('outline_prop', {})
    ax = kwargs.pop('ax', pylab.gca())
    
    tmp = {}
    if normal:
        tmp['transform'] = ax.transAxes
    tmp.update(kwargs)
    
    txt = ax.text(*args, **tmp)
    
    if oline:
        outline(txt, **outline_prop)
    return txt





def line(x=None, y=None, r=None, **kwargs):
    '''X,Y Arrays of lines to plot, r is the range of the line.'''
    xmin,xmax,ymin,ymax = pylab.axis()
    oline = kwargs.pop('outline',False)
    outline_prop = kwargs.pop('outline_prop',{})
    kwargs.setdefault('color','orange')
    kwargs.setdefault('linestyle','-')
    kwargs.setdefault('linewidth', 2)
    
    if x is not None:
        yr = [ymin, ymax] if r is None else r
        # if isinstance(x, (np.float32, float, int)):
        if not isinstance(x, collections.Iterable):
            x = [x]
        
        for a in x:
            pl = pylab.plot([a,a], yr, **kwargs)
            if oline:
                outline(pl, **outline_prop)
    
    
    
    if y is not None:
        xr = [xmin, xmax] if r is None else r
        # if isinstance(y, (float, int)):
        if not isinstance(y, collections.Iterable):
            y = [y]
        for a in y:
            pl = pylab.plot(xr, [a,a], **kwargs)
            if oline:
                outline(pl, **outline_prop)



def box(x=None, y=None, normal=False, fill=True, ax=None,**kwargs):
    oline = kwargs.pop('outline',False)
    outline_prop = kwargs.pop('outline_prop',{})
    tmp = dict(color='0.2', alpha=0.4, linewidth=2,)
    
    if ax is None:
        ax = pylab.gca()
    axt = ax.axis()
    if normal:
        tmp['transform'] = ax.transAxes
    else:
        tmp['transform'] = ax.transData
    
    # if x is not None and percent: x = np.diff(axt[:2])*np.array(x) + axt[0]
    # if y is not None and percent: y = np.diff(axt[2:])*np.array(y) + axt[2]
    if x is None: x = axt[:2]
    if y is None: y = axt[2:]
    
    dx = np.diff(x)[0]
    dy = np.diff(y)[0]
    # use date2num
    # if isinstance(x[0], datetime): dx = dx.days
    # if isinstance(y[0], datetime): dy = dy.total_seconds()
    
    
    if oline or (not fill):
        tmp['facecolor'] = 'none'
        tmp['edgecolor'] = kwargs.pop('color', tmp.pop('color'))
    tmp.update(kwargs)
    patch = pylab.Rectangle((x[0],y[0]),dx,dy, **tmp)
    if oline:
        outline(patch, **outline_prop)
    ax.add_patch(patch)
    

    # return patch



def polygon(x,y=None, normal=False, fill=True, **kwargs):
    oline = kwargs.pop('outline',False)
    outline_prop = kwargs.pop('outline_prop',{})
    tmp = dict(color='0.2', alpha=0.4, linewidth=2,)
    
    ax = pylab.gca()
    if normal:
        tmp['transform'] = ax.transAxes
    
    
    if y is not None:
        x = np.array(zip(x,y))
        
    if oline or (not fill):
        tmp['facecolor'] = 'none'
        tmp['edgecolor'] = kwargs.pop('color', tmp.pop('color'))
        
    tmp.update(kwargs)
    patch  = pylab.Polygon(x, **tmp)
    
    if oline:
        outline(oatch, **outline_prop)
        
    ax.add_patch(patch)
    return patch









### sky plots
def setup_sky(header, subplt=111, delta=100, title=None, **kwargs):
    '''Setup a nice sky plot that handles images'''
    import pywcsgrid2 # slow as shit
    ax = pywcsgrid2.subplot(subplt, header=header)
    fig = pylab.gcf()
    fig.add_axes(ax)
    ax.set_display_coord_system('fk5')
    ax.set_ticklabel_type('absdeg', 'absdeg')
    ax.set_aspect('equal')
    ax.locator_params(axis='x', nbins=4)
    ax.grid()
    ax.set_xlim(-delta,delta+header['naxis1'])
    ax.set_ylim(-delta,delta+header['naxis2'])
    if title is not None:
        pylab.title(title)
    return ax
    
def skypoly(window, **kwargs):
    from matplotlib.patches import Polygon
    from matplotlib.collections import PolyCollection, PatchCollection
    
    
    
    cmap = kwargs.pop('cmap',None)
    if cmap is None:
        cmap = copy.copy(pylab.cm.gray_r)
        cmap.set_bad('r',1.0)
    
    try:
        ax = pylab.gca()['fk5']
    except Exception as e:
        # print e
        ax = pylab.gca()
    
    p,w = window.graphics(getweight=True)
    if len(p) == 1: w = np.array([w])
    vmin = kwargs.pop('vmin',np.max(w))
    vmax = kwargs.pop('vmax',np.min(w))
    if vmin==vmax:
        vmin = vmax-1
    # w = np.ones(len(p))
    
    
    patches = []
    for i,poly in enumerate(p):
        ra,dec = poly['ra'], poly['dec']
        patches.append(Polygon(zip(ra,dec), edgecolor='none', lw=0.01) )
    
    tmp = {'cmap':cmap,
           'rasterized': True,
           'edgecolors':'none',
           'antialiaseds':True,
         }
    tmp.update(kwargs)
    
    p = PatchCollection(patches, **tmp)
    p.set_array(w)
    p.set_clim(vmin,vmax)
    ax.add_collection(p)
    ax.set_aspect('equal')
    ax.set_rasterization_zorder(0)
    return p



def skywindow(window, header=None, **kwargs):
    '''Plot a window function'''
    cmap = kwargs.pop('cmap',None)
    if cmap is None:
        cmap = copy.copy(pylab.cm.gray_r)
        cmap.set_bad('r',1.0)
    
    ax = pylab.gca()
        
    vmin, vmax = math.minmax(window)
    if vmin == vmax:
        vmax += 1
    tmp = {'vmin':vmin, 'vmax':vmax, 
           'origin':'low',
           'cmap':cmap}
    tmp.update(kwargs)
    
    if header is None:
        im = ax.imshow(window, **tmp)
    else:
        im = ax[header].imshow_affine(window, **tmp)
    return im
    

        
    
    

def sky(ra, dec, **kwargs):
    ''' Basemap setup'''
    tmp = {'marker':',', 
           'c':'k',
           's':0.5,
           #  'edgecolors':'none',
           'rasterized':True,
           'alpha':0.5}
    tmp.update(kwargs)
    
    ax = pylab.gca()
    ax['fk5'].scatter(ra,dec, **tmp)
    


def skyheader(ra, dec, npixel=None):
    '''Generate a nice header from a set of ra bounds and dec bounds'''
    if npixel is None:
        npixel = [1000,1000]
    import pywcs # meh slow?
    dra,ddec = ra[1]-ra[0], dec[1]-dec[0]
    dummy = pywcs.WCS(naxis=2, minerr=1E-4)
    # dummy.wcs.crpix = [npixel[0]/2, npixel[1]/2]  # pixel position
    # dummy.wcs.crval = [np.mean(ra), np.mean(dec)]   # RA, Dec (degrees)
    # dummy.wcs.crpix = [1,1]
    # dummy.wcs.crval = [ra[0]+dra/npixel[0], dec[0]+ddec/npixel[1]]
    dummy.wcs.crpix = [1,1]
    dummy.wcs.crval = [ra[0], dec[0]]
    
    dummy.wcs.ctype = ["RA", "DEC"]
    # dummy.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    # dummy.wcs.ctype = ["RA---AIR", "DEC--AIR"]
    
    dummy.wcs.cdelt = [dra/npixel[0], ddec/npixel[1]]
    # print [dra/npixel[0],ddec/npixel[1]]
    # 
    # tmp = dummy.wcs_pix2sky(np.array([npixel]), 0)
    # print tmp
    # print np.array([ra[1], dec[1]]) - tmp
    
    # print np.array([ra[1], dec[1]])
    # print dummy.wcs_sky2pix(np.array([[ra[1], dec[1]]]), 0)
    
    
    header = dummy.to_header()
    header.update('NAXIS1',npixel[0])
    header.update('NAXIS2',npixel[1])
    header.update('EQUINOX', 2000.0)
    
    # del header['WCSAXES']
    # del header['RESTFRQ']
    # del header['RESTWAV']
    # del header['LONPOLE']
    # del header['LATPOLE']
    
    return header

def _sky2():
    pass
    # import pylab
    # from kapteyn import maputils
    # from matplotlib import pyplot as plt
    # fitsobj = maputils.FITSimage('/Users/ajmendez/research/data/fields/egs/polygons/windowf.41.fits.gz')
    # ax =  pylab.subplot(111)
    # baseim = fitsobj.Annotatedimage(ax)
    # baseim.Image()
    # graticule1 = baseim.Graticule()
    # 
    # Secondfits = maputils.FITSimage('/Users/ajmendez/research/data/fields/egs/polygons/windowf.42.fits.gz')
    # pars = dict(cval=0.0, order=1)
    # Reprojfits = Secondfits.reproject_to(fitsobj, interpol_dict=pars)
    # overlayim = fitsobj.Annotatedimage(ax, boxdat=Reprojfits.boxdat)
    # overlayim.Image(alpha=0.0)
    # baseim.plot()
    # overlayim.plot()
    # baseim.interact_toolbarinfo()
    # baseim.interact_imagecolors()
    # plt.show()




def dateticks(fmt='%Y-%m', **kwargs):
    '''setup the date ticks'''
    dateticker = ticker.FuncFormatter(lambda numdate, _: num2date(numdate).strftime(fmt))
    pylab.gca().xaxis.set_major_formatter(dateticker)
    # pylab.gcf().autofmt_xdate()
    tmp = dict(rotation=30, ha='right')
    tmp.update(kwargs)
    pylab.setp(pylab.xticks()[1], **tmp)








def density(x,y, z=None, weights=None, 
            bins=None, nbins=None, 
            extent=None, ngcextent=False, radecbins=False, remap_ra=False,
            extent_embiggen=None,
            numbernorm=False,
            vrange=None, logvrange=False,
            sqrt=False,
            areaunit=False, colorbar=True,
            ax = None, trans=None, mesh=False,
            hidezeros=False, maskzeros=False, 
            maskthreshold=False, threshold=None,
            noplot=False,
            **kwargs):
    '''makes nbins if bins is not defined
    radecbins should always be used with mesh=True!!!!
    
    '''
    if ax is None:
        ax = pylab.gca()
    if trans is None:
        trans = ax.transData
    
    if remap_ra:
      x = np.array([a-360.0 if a > 300 else a for a in x])
    
    
    if vrange is None: 
        vrange=[None,None]
    if extent is None:
        extent = [np.min(x), np.max(x), np.min(y), np.max(y)]
    if ngcextent:
        extent = [132.1, 228.9, -1.1, 56.1]
    
    if extent_embiggen is not None:
        extent = embiggen(extent[:2], extent_embiggen) + embiggen(extent[2:], extent_embiggen)
    
    
    if nbins is None:
        nbins = 100
    if isinstance(nbins, int):
        nbins = (nbins,nbins)
    if bins is None:
        bins = ( np.linspace(extent[0], extent[1], nbins[0]+1),
                 np.linspace(extent[2], extent[3], nbins[1]+1) )
    if radecbins:
        tr = np.deg2rad(extent[:2])/(2*np.pi)
        pr  = (np.cos(np.deg2rad(extent[2:])+np.pi/2)+1)/2.0
        rabins = np.rad2deg(2*np.pi*np.linspace(tr[0], tr[1], nbins[0]+1) )
        decbins = np.rad2deg(np.arccos(2*np.linspace(pr[0], pr[1], nbins[1]+1)-1)-np.pi/2)
        bins = (rabins, decbins)
    
    if z is not None:
        numbernorm = True
        w = z 
        w[np.where(~np.isfinite(z))] = 0
    else:
        w = weights
    
    
    H,xx,yy = np.histogram2d(x,y, bins=bins, weights=w)
    if areaunit:
        H /= np.diff(xx)*np.diff(yy)
    
    
    if hidezeros:
        ii = np.where(H==0)
        H[ii] = np.nan
    
    if maskzeros:
        H = np.ma.MaskedArray(H, (H==0))
    
    if maskthreshold:
        if threshold is None:
            threshold = 0
        H = np.ma.MaskedArray(H, (H<threshold))
    
    if numbernorm:
        N = np.histogram2d(x,y, bins=bins, weights=weights)[0]
        ii = np.where(N > 0)
        H[ii] /= N[ii]
    
    if logvrange:
        H = np.log10(H)
    if sqrt:
        H = np.sqrt(H)

    if noplot:
        return bins[0], bins[1], H.T
        

    if mesh:
        tmp = dict(
            edgecolors='none',
            vmin=vrange[0], vmax=vrange[1], 
            clip_on=True, clip_path=ax.patch,
            cmap=pylab.cm.OrRd, transform=trans
        )
        tmp.update(kwargs)
        im = ax.pcolormesh(bins[0], bins[1], H.T, **tmp)
        im.set_clip_on(True) # Broken?!
        im.set_clip_path(ax.patch)
        # im.set_transform(trans)
    else:
        tmp = dict(
            extent=minmax(xx)+minmax(yy),
             origin='lower', interpolation='nearest', aspect='auto',
             cmap=pylab.cm.OrRd, vmin=vrange[0], vmax=vrange[1]
        )
        tmp.update(kwargs)
        im = ax.imshow(H.T, **tmp)

    if colorbar:
        tmp = pylab.gca()
        cb = pylab.colorbar(im, shrink=0.7)
        ticks = np.array(map(int, np.linspace(cb.vmin, cb.vmax, 5)))
        cb.set_ticks(ticks)
        # if areaunit:
            # estimate area of bin and multiply
            # N/bin = N/(area of bin) = N * 
            # area = np.mean(np.diff(bins[0]))*np.mean(np.diff(bins[1]))
            # ticks = ticks*area
        if logvrange: 
            if np.min(ticks) > 1:
                ticks = np.array(map(int, ticks))
                formatter = r'$10^{{{:d}}}$'
            else:
                ticks = 10.0**ticks
                formatter = r'{:0.1f}'
        else:
            formatter = (r'{:0.0f}' if np.min(ticks) > 1 else '{:0.2f}')
        
    #     formater =  if logvrange else 
        cb.set_ticklabels(map(lambda x:formatter.format(x), ticks))
        if 'label' in kwargs: 
            cb.set_label(kwargs['label'])
        pylab.sca(tmp)
    
    im.xbins = bins[0]
    im.ybins = bins[1]
    im.Z = H
    return im
