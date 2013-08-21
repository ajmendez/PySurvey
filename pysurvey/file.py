# fily.py -- nice file handling
# Mendez 05.2013

# System Libraries
import os
import subprocess

# Installed Libraries
import numpy as np
import pyfits

# Package Imports
from util import splog


def nicefile(filename):
    '''Handles all of that os fun like {$HOME, ~, $ENV, etc} so that a absolute path is returned.'''
    niceName = os.path.abspath(os.path.expandvars(os.path.expanduser(os.path.normpath(filename))))
    if os.path.isdir(niceName):
        niceName += os.sep
    return niceName








def _namefmt(a,b,c,d, sep='| '):
    return ' {1:<20}{0}{2:<12}{0}{3:<20}{0}{4:<10}'.format(sep,a,b,c,d)
    
def _arrayfmt(data):
    return ['%s, ...'%(x[0]) if isinstance(x,np.ndarray) else '%s'%(x) for x in data]


def fromHDU(name, hdu):
    '''Return a nice named catobject (Cat) from an hdu'''
    if isinstance(hdu, pyfits.column.ColDefs):
        hdu = pyfits.new_table(hdu)
    cat = Cat('_%s_'%(name), setup=False)
    cat.name = name
    cat.hdu = hdu
    cat.data = hdu.data
    cat.columns = hdu.columns
    cat.header = hdu.header
    cat.names = map(str.lower, cat.columns.names)
    return cat

def join(a, b):
    ''' Join two cat objects. '''
    tmp = pyfits.new_table(a.columns, nrows=len(a)+len(b) )
    for column in b.columns:
        if column.dim is not None:
            # s = tmp.data[column.name][len(a):, :].shape
            # tmp.data[column.name][len(a):, :] = np.reshape(column.array,s)
            tmp.data[column.name][len(a):, :] = b[column.name]
        else:
            tmp.data[column.name][len(a):] = column.array
    return fromHDU(None, tmp)


class Cat(object):
    def __init__(self, filename, index=1, setup=True):
        '''Loads the data. defaults to the first index for the data'''
        self.filename = nicefile(filename)
        self.ind = index
        if setup:
            self._setup()
    
    def _setup(self):
        self.fits = pyfits.open(self.filename)
        self.data = self.fits[self.ind].data
        self.columns = self.data.columns
        self.name = self.fits[self.ind].name
        self.header = self.fits[self.ind].header
        self.names = map(str.lower, self.data.columns.names)
    
    def __repr__(self):
        print self.__str__()
        print
        return "%s.%s('%s', index=%d)"%(self.__class__.__module__,
                                      self.__class__.__name__,
                                      self.filename,
                                      self.ind)
    
    def __str__(self):
        '''Returns a nicely formatted string for the Cat'''
        tmp = [_namefmt('Tag','Format','Example','Unit')]
        tmp += [_namefmt('-'*20,'-'*12,'-'*20,'-'*10,sep='+-')]
        f = [a+b for a,b in zip(self.columns.formats, self.columns.dims)]
        tmp.extend([_namefmt(*x) for x in zip(self.columns.names, 
                                            # self.columns.formats,
                                            f,
                                            _arrayfmt(self.data[0]),
                                            self.data.columns.units)])
        return "%s.%s [%s]: %s \n%s"%(self.__class__.__module__,
                                 self.__class__.__name__,
                                 hex(id(self)),
                                 self.name if self.name is not None else '',
                                 '\n'.join(tmp))
    
    def __len__(self):
        '''Return the number of rows in the data set'''
        return len(self.data)
    
    def __getitem__(self, key):
        '''Returns the value for the key from the data.  assumes that the data
        keys are unique by lower case strings'''
        if (isinstance(key,str) and key.lower() in self.names):
            return self.data[key.lower()]
        else:
            return self._get(key)
    
    def __getattr__(self, key):
        '''Grabs the attribute [CLASS.xxx] for the name by using the 
          attempting to use the __getitem__ value'''
        if key.lower() in self.names:
            return self.__getitem__(key)
        else:
            return getattr(self.data,key)
            
    
    
    def _get(self, key):
        '''Attempts to load the right index and populate a new value'''
        tmp = Cat(self.filename, setup=False)
        # Convert a single integer key into a slice
        if isinstance(key,int):
            if key < 0:
                key += len(self)
            key = slice(key,key+1)
        # tmp.fits = pyfits.new_table(self.data[key].columns)
        tmp.fits = pyfits.BinTableHDU(self.data[key],
                                      header=self.header,
                                      name=self.name)
        tmp.name = self.name
        tmp.data = tmp.fits.data
        tmp.columns = tmp.fits.data.columns
        tmp.header = self.header
        tmp.names = self.names
        
        return tmp
    
    def filtercolumns(self, include=None, exclude=None, rename=None, name=None):
        '''Returns an columndef list of columns'''
        columns = []
        
        if name is None:
            name = 'Cat'
        
        for column in self.columns:
            if ( ( (include is not None) and (column.name in include) ) or 
                 ( (exclude is not None) and (column.name not in exclude) ) ):
                # Before adding go and see if we need to rename
                if rename is not None:
                    for r in rename:
                        t,f = r.split('>')
                        if column.name == t:
                            column.name = f
                # time to add to the columns list
                columns.append(column)
                
        return fromHDU(name, pyfits.ColDefs(columns))
    
    def append(self, columns):
        return fromHDU(self.name, self.columns + columns)
    def prepend(self, columns):
        return fromHDU(self.name, columns+self.columns)










def write(hdu, filename, quiet=False, clobber=True, shortdir=None):
    '''Write a nice fits file.  Filenames with .gz will be compressed using the system
    gzip call.  hdu can be either a column table list or a pyfits.bintable object.
    it notifies the user what was written.
    
    clobber=[True] -- Overwrite files by default
    quiet=[False]  -- Do not announce that we are writing a file.  Why would you not?
    shortdir=['/path/','<Dir>'] -- Shorten the filename that is printed out to the screen
    '''        
    
    # Handle column lists and hdus
    if isinstance(hdu, pyfits.hdu.table.BinTableHDU):
        hdus = hdu
    elif (isinstance(hdu, list) or isinstance(hdu, tuple)) and (isinstance(hdu[0], pyfits.Column)):
        hdus = pyfits.new_table(pyfits.ColDefs(hdu))
    elif isinstance(hdu, Cat):
        hdus = pyfits.new_table(pyfits.ColDefs(hdu.columns))
    else:
        hdus = pyfits.HDUList(hdu)
    
    # Save and compress if needed
    outname = filename.replace('.gz','')
    hdus.writeto(outname, clobber=clobber)
    if '.gz' in filename:
        subprocess.call(['gzip', '-f', outname])
    
    # You should generally tell the user what was saved, but hey you can
    # play with matches if you want.
    if not quiet:
        # print "Saved to: %s"%(filename)
        if shortdir is not None:
            splog('Saved to:', filename.replace(shortdir[0], shortdir[1]))
        else:
            splog('Saved to:', filename)



