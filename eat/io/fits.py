# from future import standard_library
# standard_library.install_aliases()
# from builtins import next
# from builtins import object
from astropy.io import fits

# some easier interactive fits interface
class fitshdulist(object):

    def __init__(self, hdulist):
        self.hdulist = hdulist

    def __dir__(self):
        return [a.name for a in self.hdulist]

    def __repr__(self):
        # from io import StringIO
        # at some point (?) io.StringIO stopped accepting byte strings, need revisit in 3.x
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
            # reversed iterator broken at some point as of astropy 1.3 (debian jessie), possibly related to,
            # https://github.com/astropy/astropy/issues/5585
            # return fitshdu(next((a for a in reversed(self.hdulist) if attr.upper() in a.name)))
            return fitshdu(next((a for a in self.hdulist[::-1] if attr.upper() in a.name)))

# some easier interactive fits interface
class fitshdu(object):

    def __init__(self, hdu):
        self.hdu = hdu

    def __dir__(self):
        if hasattr(self.hdu, 'columns'):
            return [a.name for a in self.hdu.columns] + ['hdu', 'header']
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
    """open fits file into convenient contained class for interactive work

    Args:
        filename: fits filename

    Returns:
        container object (:class:`.fitshdulist`)
    """
    # mode='denywrite' here to avoid annoying memmap preallocation problem with mmap.ACCESS_COPY on large files
    # https://github.com/astropy/astropy/issues/1380
    hdulist = fits.open(filename, mode='denywrite')
    return fitshdulist(hdulist)
