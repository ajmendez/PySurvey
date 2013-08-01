===========
pySurvey
===========

*pySurvey* provides some nice helper functions for people who work 
on redshift surveys, object catalogs, and visualization.  It was 
built after many years of IDL so there is that.

    #!/usr/bin/env python

    from pysurvey import plot
    from pysurvey import catalog

    for i,a in enumerate(x):
      plot.setup( (2,2,i), # i is assumed to be zero indexed
                  xr=[0,10], xlabel='X Title',
                  yr=[1,1E3], ylog=True, ylabel='Log Y axis',
                  subtitle='subtitle', suptitle='Page title', title='plot title',
                  autoTick=True)


SubPackages
=========

Lists look like this:

* plot -- plotting routines

* polygon -- Mangle Polygons

* catalog -- fits catalog handling and loading

* util -- A nice set of utilities like spherematch and match


* pip install http://www.astro.rug.nl/software/kapteyn/kapteyn-2.2.tar.gz
* pip install pywcs
* github install of pyfits