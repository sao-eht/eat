from astropy.io import fits

# some easier interactive fits interface
class fitshdulist:

    def __init__(self, hdulist):
        self.hdulist = hdulist

    def __dir__(self):
        return [a.name for a in self.hdulist]

    def __repr__(self):
        from StringIO import StringIO
        out = StringIO()
        self.hdulist.info(output=out)
        return out.getvalue()

    def __getitem__(self, item):
        return fitshdu(self.hdulist[item])

    def __getattr__(self, attr):
        if hasattr(self.hdulist, attr):
            return getattr(self.hdulist, attr)
        else:
            return fitshdu(next((a for a in reversed(self.hdulist) if attr.upper() in a.name)))

# some easier interactive fits interface
class fitshdu:

    def __init__(self, hdu):
        self.hdu = hdu

    def __dir__(self):
        if hasattr(self.hdu, 'columns'):
            return [a.name for a in self.hdu.columns]
        else:
            return dir(self.hdu)

    def __repr__(self):
        if hasattr(self.hdu, 'columns'):
            return repr(self.hdu.columns)
        else:
            return repr(self.hdu)

    def __getitem__(self, item):
        if hasattr(self.hdu, 'data') and item.upper() in self.hdu.data.names:
            return self.hdu.data[item.upper()]

    def __getattr__(self, attr):
        if hasattr(self.hdu, attr):
            return getattr(self.hdu, attr)
        elif hasattr(self.hdu, 'columns'):
            return next(self.hdu.data[a.name] for a in self.hdu.columns if attr.upper() in a.name)

def open(filename):
    # mode='denywrite' here to avoid annoying memmap preallocation problem with mmap.ACCESS_COPY on large files
    # https://github.com/astropy/astropy/issues/1380
    hdulist = fits.open(filename, mode='denywrite')
    return fitshdulist(hdulist)

