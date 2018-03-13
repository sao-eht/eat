#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This is a sub-module of sparselab handling image fits data.
'''
__author__ = "Sparselab Developer Team"
#-------------------------------------------------------------------------
# Modules
#-------------------------------------------------------------------------
# standard modules
import os
import copy
import datetime as dt

# numerical packages
import numpy as np
import pandas as pd
import scipy.ndimage as sn
import astropy.coordinates as coord
import astropy.io.fits as pyfits
from astropy.convolution import convolve_fft

# matplotlib
import matplotlib.pyplot as plt

# internal
#import fortlib

#-------------------------------------------------------------------------
# IMAGEFITS (Manupulating FITS FILES)
#-------------------------------------------------------------------------
class IMFITS(object):
    ds9 = None
    ds9region = None
    angunit = "mas"

    # Initialization
    def __init__(self, fitsfile=None, uvfitsfile=None, source=None,
                 fov=None, nx=100, ny=None, angunit="mas", **args):
        '''
        This is a class to handle image data, in particular, a standard image
        FITS data sets.

        The order of priority for duplicated parameters is
            1 uvfitsfile (strongest),
            2 source
            3 fitsfile
            4 fov
            5 other parameters (weakest).

        Args:
            fitsfile (string):
                input FITS file
            uvfitsfile (string):
                input uv-fits file
            source (string):
               The source of the image RA and Dec will be obtained from CDS
            fov(array-like):
                Field of View of the image [xmin, xmax, ymin, ymax]
            nx (integer):
                the number of pixels in the RA axis.
            ny (integer):
                the number of pixels in the Dec axis.
                Default value is same to nx.
            angunit (string):
                angular unit for fov, x, y, dx and dy.

            **args: you can also specify other header information.
                x (float):
                    The source RA at the reference pixel.
                nxref (float):
                    The reference pixel in RA direction.
                    "1" will be the left-most pixel.
                dx (float):
                    The grid size of the RA axis.
                    **MUST BE NEGATIVE** for the astronomical image.
                y (float):
                    The source DEC at the reference pixel.
                nyref (float):
                    the reference pixel of the DEC axis.
                    "1" will be the bottom pixel.
                dy (float):
                    the grid size in the DEC axis.
                    MUST BE POSITIVE for the astronomical image.
                f (float):
                    the reference frequency in Hz
                nf (integer):
                    the number of pixels in the Frequency axis
                nfref (float):
                    the reference pixel of the Frequency axis
                df (float):
                    the grid size of the Frequency axis
                s (float):
                    the reference Stokes parameter
                ns (integer):
                    the number of pixels in the Stokes axis
                nsref (float):
                    the reference pixel of the Stokes axis
                ds (float):
                    the grid size of the Stokes axis

                observer (string)
                telescope (string)
                instrument (string)
                object (string)
                dateobs (string)

        Returns:
            imdata.IMFITS object
        '''
        # get conversion factor for angular scale
        angconv = self.angconv(angunit, "deg")
        self.angunit = angunit

        # get keys of arguments
        argkeys = args.keys()

        # Set header and data
        self.init_header()
        self.data = None

        # set pixel size
        if ny is None:
            ny = nx
        self.header["nx"] = nx
        self.header["ny"] = ny

        # read header from arguments
        for argkey in argkeys:
            headerkeys = self.header.keys()
            if argkey in headerkeys:
                self.header[argkey] = self.header_dtype[argkey](args[argkey])

        # Initialize from FOV
        if fov is not None:
            fovarr = np.float64(np.asarray(fov))
            self.header["dx"] = - np.abs(fovarr[0] - fovarr[1]) / (self.header["nx"] - 1)
            self.header["dy"] = np.abs(
                fovarr[2] - fovarr[3]) / (self.header["ny"] - 1)
            self.header["nxref"] = 1. - fovarr[0:2].max() / self.header["dx"]
            self.header["nyref"] = 1. - fovarr[2:4].min() / self.header["dy"]

        self.header["x"] *= angconv
        self.header["y"] *= angconv
        self.header["dx"] *= angconv
        self.header["dy"] *= angconv

        # Initialize from fitsfile
        if fitsfile is not None:
            self.read_fits(fitsfile)

        # Set source coordinates
        if source is not None:
            self.set_source(source)

        # copy headers from uvfits file
        if uvfitsfile is not None:
            self.read_uvfitsheader(uvfitsfile)

        # initiliza image data
        if self.data is None:
            self.data = np.zeros([self.header["ns"], self.header["nf"],
                                  self.header["ny"], self.header["nx"]])

        # initialize fitsdata
        self.update_fits()

    # Definition of Headers and their datatypes
    def init_header(self):
        header = {}
        header_dtype = {}

        # Information
        header["object"] = "NONE"
        header_dtype["object"] = str
        header["telescope"] = "NONE"
        header_dtype["telescope"] = str
        header["instrument"] = "NONE"
        header_dtype["instrument"] = str
        header["observer"] = "NONE"
        header_dtype["observer"] = str
        header["dateobs"] = "NONE"
        header_dtype["dateobs"] = str

        # RA information
        header["x"] = np.float64(0.)
        header_dtype["x"] = np.float64
        header["dx"] = np.float64(-1.)
        header_dtype["dx"] = np.float64
        header["nx"] = np.int64(1)
        header_dtype["nx"] = np.int64
        header["nxref"] = np.float64(1.)
        header_dtype["nxref"] = np.float64

        # Dec information
        header["y"] = np.float64(0.)
        header_dtype["y"] = np.float64
        header["dy"] = np.float64(1.)
        header_dtype["dy"] = np.float64
        header["ny"] = np.int64(1)
        header_dtype["ny"] = np.int64
        header["nyref"] = np.float64(1.)
        header_dtype["nyref"] = np.float64

        # Frequency Information
        header["f"] = np.float64(229.345e9)
        header_dtype["f"] = np.float64
        header["df"] = np.float64(4e9)
        header_dtype["df"] = np.float64
        header["nf"] = np.int64(1)
        header_dtype["nf"] = np.int64
        header["nfref"] = np.float64(1.)
        header_dtype["nfref"] = np.float64

        # Stokes Information
        header["s"] = np.int64(1)
        header_dtype["s"] = np.int64
        header["ds"] = np.int64(1)
        header_dtype["ds"] = np.int64
        header["ns"] = np.int64(1)
        header_dtype["ns"] = np.int64
        header["nsref"] = np.int64(1)
        header_dtype["nsref"] = np.int64

        self.header = header
        self.header_dtype = header_dtype

    # set source name and source coordinates
    def set_source(self, source="SgrA*"):
        srccoord = coord.SkyCoord.from_name(source)

        # Information
        self.header["object"] = source
        self.header["x"] = srccoord.ra.deg
        self.header["y"] = srccoord.dec.deg
        self.update_fits()

    # Read data from an image fits file
    def read_fits(self, fitsfile):
        '''
        Read data from the image FITS file

        Args:
          fitsfile (string): input image FITS file
        '''
        hdulist = pyfits.open(fitsfile)
        self.hdulist = hdulist

        keyname = "OBJECT"
        try:
            self.header["object"] = self.header_dtype["object"](
                hdulist[0].header.get(keyname))
        except:
            print("warning: FITS file doesn't have a header info of '%s'"
                  % (keyname))

        keyname = "TELESCOP"
        try:
            self.header["telescope"] = self.header_dtype["telescope"](
                hdulist[0].header.get(keyname))
        except:
            print("warning: FITS file doesn't have a header info of '%s'"
                  % (keyname))

        keyname = "INSTRUME"
        try:
            self.header["instrument"] = self.header_dtype["instrument"](
                hdulist[0].header.get(keyname))
        except:
            print("warning: FITS file doesn't have a header info of '%s'"
                  % (keyname))

        keyname = "OBSERVER"
        try:
            self.header["observer"] = self.header_dtype["observer"](
                hdulist[0].header.get(keyname))
        except:
            print("warning: FITS file doesn't have a header info of '%s'"
                  % (keyname))

        keyname = "DATE-OBS"
        try:
            self.header["dateobs"] = \
                self.header_dtype["dateobs"](hdulist[0].header.get(keyname))
        except:
            print("warning: FITS file doesn't have a header info of '%s'" % (keyname))

        isx = False
        isy = False
        isf = False
        iss = False
        naxis = hdulist[0].header.get("NAXIS")
        for i in range(naxis):
            ctype = hdulist[0].header.get("CTYPE%d" % (i + 1))
            if ctype is None:
                continue
            if ctype[0:2] == "RA":
                isx = i + 1
            elif ctype[0:3] == "DEC":
                isy = i + 1
            elif ctype[0:4] == "FREQ":
                isf = i + 1
            elif ctype[0:6] == "STOKES":
                iss = i + 1

        if isx != False:
            self.header["nx"] = \
                self.header_dtype["nx"](hdulist[0].header.get("NAXIS%d" % (isx)))
            self.header["x"] = \
                self.header_dtype["x"](hdulist[0].header.get("CRVAL%d" % (isx)))
            self.header["dx"] = \
                self.header_dtype["dx"](hdulist[0].header.get("CDELT%d" % (isx)))
            self.header["nxref"] = \
                self.header_dtype["nxref"](hdulist[0].header.get("CRPIX%d" % (isx)))
        else:
            print("Warning: No image data along RA axis.")

        if isy != False:
            self.header["ny"] = self.header_dtype["ny"](
                hdulist[0].header.get("NAXIS%d" % (isy)))
            self.header["y"] = self.header_dtype["y"](
                hdulist[0].header.get("CRVAL%d" % (isy)))
            self.header["dy"] = self.header_dtype["dy"](
                hdulist[0].header.get("CDELT%d" % (isy)))
            self.header["nyref"] = self.header_dtype["nyref"](
                hdulist[0].header.get("CRPIX%d" % (isy)))
        else:
            print("Warning: No image data along DEC axis.")

        if isf != False:
            self.header["nf"] = self.header_dtype["nf"](
                hdulist[0].header.get("NAXIS%d" % (isf)))
            self.header["f"] = self.header_dtype["f"](
                hdulist[0].header.get("CRVAL%d" % (isf)))
            self.header["df"] = self.header_dtype["df"](
                hdulist[0].header.get("CDELT%d" % (isf)))
            self.header["nfref"] = self.header_dtype["nfref"](
                hdulist[0].header.get("CRPIX%d" % (isf)))
        else:
            print("Warning: No image data along STOKES axis.")

        if iss != False:
            self.header["ns"] = self.header_dtype["ns"](
                hdulist[0].header.get("NAXIS%d" % (iss)))
            self.header["s"] = self.header_dtype["s"](
                hdulist[0].header.get("CRVAL%d" % (iss)))
            self.header["ds"] = self.header_dtype["ds"](
                hdulist[0].header.get("CDELT%d" % (iss)))
            self.header["nsref"] = self.header_dtype["nsref"](
                hdulist[0].header.get("CRPIX%d" % (iss)))
        else:
            print("Warning: No image data along STOKES axis.")

        self.data = hdulist[0].data.reshape(
            [self.header["ns"], self.header["nf"], self.header["ny"], self.header["nx"]])

    def read_uvfitsheader(self, infits):
        '''
        Read header information from uvfits file

        Args:
          infits (string): input uv-fits file
        '''
        hdulist = pyfits.open(infits)
        hduinfos = hdulist.info(output=False)
        for hduinfo in hduinfos:
            idx = hduinfo[0]
            if hduinfo[1] == "PRIMARY":
                grouphdu = hdulist[idx]
        if not 'grouphdu' in locals():
            print("[Error] %s does not contain the Primary HDU" % (infits))

        if 'OBJECT' in grouphdu.header:
            self.header["object"] = self.header_dtype["object"](
                grouphdu.header.get('OBJECT'))
        else:
            self.header["object"] = self.header_dtype["object"]('None')

        if 'TELESCOP' in grouphdu.header:
            self.header["telescope"] = self.header_dtype["telescope"](
#                telescope=grouphdu.header.get('TELESCOP'))
                grouphdu.header.get('TELESCOP'))
        else:
            self.header["telescope"] = self.header_dtype["telescope"]('None')

        if 'INSTRUME' in grouphdu.header:
            self.header["instrument"] = self.header_dtype["instrument"](
                grouphdu.header.get('INSTRUME'))
        else:
            self.header["instrument"] = self.header_dtype["instrument"]('None')

        if 'OBSERVER' in grouphdu.header:
            self.header["observer"] = self.header_dtype["observer"](
                grouphdu.header.get('OBSERVER'))
        else:
            self.header["observer"] = self.header_dtype["observer"]('None')

        if 'DATE-OBS' in grouphdu.header:
            self.header["dateobs"] = self.header_dtype["dateobs"](
                grouphdu.header.get('DATE-OBS'))
        else:
            self.header["dateobs"] = self.header_dtype["dateobs"]('None')

        naxis = grouphdu.header.get("NAXIS")
        for i in range(naxis):
            ctype = grouphdu.header.get("CTYPE%d" % (i + 1))
            if ctype is None:
                continue
            elif ctype[0:2] == "RA":
                isx = i + 1
            elif ctype[0:3] == "DEC":
                isy = i + 1
            elif ctype[0:4] == "FREQ":
                isf = i + 1

        if isx != False:
            self.header["x"] = self.header_dtype["x"](
                hdulist[0].header.get("CRVAL%d" % (isx)))
        else:
            print("Warning: No RA info.")

        if isy != False:
            self.header["y"] = self.header_dtype["y"](
                hdulist[0].header.get("CRVAL%d" % (isy)))
        else:
            print("Warning: No Dec info.")

        if isf != False:
            self.header["f"] = self.header_dtype["f"](
                hdulist[0].header.get("CRVAL%d" % (isf)))
            self.header["df"] = self.header_dtype["df"](hdulist[0].header.get(
                "CDELT%d" % (isf)) * hdulist[0].header.get("NAXIS%d" % (isf)))
            self.header["nf"] = self.header_dtype["nf"](1)
            self.header["nfref"] = self.header_dtype["nfref"](1)
        else:
            print("Warning: No image data along STOKES axis.")

        self.update_fits()

    def update_fits(self,cctab=True,threshold=None, relative=True,
                    istokes=0, ifreq=0):
        '''
        Reflect current self.data / self.header info to the image FITS data.
        Args:
            cctab (boolean): If True, AIPS CC table is attached to fits file.
            istokes (integer): index for Stokes Parameter at which the image will be used for CC table.
            ifreq (integer): index for Frequency at which the image will be used for CC table.
            threshold (float): pixels with the absolute intensity smaller than this value will be ignored in CC table.
            relative (boolean): If true, theshold value will be normalized with the peak intensity of the image.
        '''

        # CREATE HDULIST
        hdu = pyfits.PrimaryHDU(self.data)
        hdulist = pyfits.HDUList([hdu])

        # GET Current Time
        dtnow = dt.datetime.now()

        # FILL HEADER INFO
        hdulist[0].header.set("OBJECT",   self.header["object"])
        hdulist[0].header.set("TELESCOP", self.header["telescope"])
        hdulist[0].header.set("INSTRUME", self.header["instrument"])
        hdulist[0].header.set("OBSERVER", self.header["observer"])
        hdulist[0].header.set("DATE",     "%04d-%02d-%02d" %
                              (dtnow.year, dtnow.month, dtnow.day))
        hdulist[0].header.set("DATE-OBS", self.header["dateobs"])
        hdulist[0].header.set("DATE-MAP", "%04d-%02d-%02d" %
                              (dtnow.year, dtnow.month, dtnow.day))
        hdulist[0].header.set("BSCALE",   np.float64(1.))
        hdulist[0].header.set("BZERO",    np.float64(0.))
        hdulist[0].header.set("BUNIT",    "JY/PIXEL")
        hdulist[0].header.set("EQUINOX",  np.float64(2000.))
        hdulist[0].header.set("OBSRA",    np.float64(self.header["x"]))
        hdulist[0].header.set("OBSDEC",   np.float64(self.header["y"]))
        hdulist[0].header.set("DATAMAX",  self.data.max())
        hdulist[0].header.set("DATAMIN",  self.data.min())
        hdulist[0].header.set("CTYPE1",   "RA---SIN")
        hdulist[0].header.set("CRVAL1",   np.float64(self.header["x"]))
        hdulist[0].header.set("CDELT1",   np.float64(self.header["dx"]))
        hdulist[0].header.set("CRPIX1",   np.float64(self.header["nxref"]))
        hdulist[0].header.set("CROTA1",   np.float64(0.))
        hdulist[0].header.set("CTYPE2",   "DEC---SIN")
        hdulist[0].header.set("CRVAL2",   np.float64(self.header["y"]))
        hdulist[0].header.set("CDELT2",   np.float64(self.header["dy"]))
        hdulist[0].header.set("CRPIX2",   np.float64(self.header["nyref"]))
        hdulist[0].header.set("CROTA2",   np.float64(0.))
        hdulist[0].header.set("CTYPE3",   "FREQ")
        hdulist[0].header.set("CRVAL3",   np.float64(self.header["f"]))
        hdulist[0].header.set("CDELT3",   np.float64(self.header["df"]))
        hdulist[0].header.set("CRPIX3",   np.float64(self.header["nfref"]))
        hdulist[0].header.set("CROTA3",   np.float64(0.))
        hdulist[0].header.set("CTYPE4",   "STOKES")
        hdulist[0].header.set("CRVAL4",   np.int64(self.header["s"]))
        hdulist[0].header.set("CDELT4",   np.int64(self.header["ds"]))
        hdulist[0].header.set("CRPIX4",   np.int64(self.header["nsref"]))
        hdulist[0].header.set("CROTA4",   np.int64(0))

        # Add AIPS CC Table
        if cctab:
            aipscctab = self._aipscc(threshold=threshold, relative=relative,
                    istokes=istokes, ifreq=ifreq)

            hdulist.append(hdu=aipscctab)

            next = len(hdulist)
            hdulist[next-1].name = 'AIPS CC'

        self.hdulist = hdulist

    def _aipscc(self, threshold=None, relative=True,
                    istokes=0, ifreq=0):
        '''
        Make AIPS CC table

        Arguments:
            istokes (integer): index for Stokes Parameter at which the image will be saved
            ifreq (integer): index for Frequency at which the image will be saved
            threshold (float): pixels with the absolute intensity smaller than this value will be ignored.
            relative (boolean): If true, theshold value will be normalized with the peak intensity of the image.
        '''
        Nx = self.header["nx"]
        Ny = self.header["ny"]
        xg, yg = self.get_xygrid(angunit="deg")
        X, Y = np.meshgrid(xg, yg)    
        X = X.reshape(Nx * Ny)
        Y = Y.reshape(Nx * Ny)
        flux = self.data[istokes, ifreq]
        flux = flux.reshape(Nx * Ny)

        # threshold
        if threshold is None:
            thres = np.finfo(np.float32).eps
        else:
            if relative:
                thres = self.peak(istokes=istokes, ifreq=ifreq) * threshold
            else:
                thres = threshold
        thres = np.abs(thres)

        # adopt threshold
        X = X[flux >= thres]
        Y = Y[flux >= thres]
        flux = flux[flux >= thres]

        # make table columns
        c1 = pyfits.Column(name='FLUX', array=flux, format='1E',unit='JY')
        c2 = pyfits.Column(name='DELTAX', array=X, format='1E',unit='DEGREES')
        c3 = pyfits.Column(name='DELTAY', array=Y, format='1E',unit='DEGREES')
        c4 = pyfits.Column(name='MAJOR AX', array=np.zeros(len(flux)), format='1E',unit='DEGREES')
        c5 = pyfits.Column(name='MINOR AX', array=np.zeros(len(flux)), format='1E',unit='DEGREES')
        c6 = pyfits.Column(name='POSANGLE', array=np.zeros(len(flux)), format='1E',unit='DEGREES')
        c7 = pyfits.Column(name='TYPE OBJ', array=np.zeros(len(flux)), format='1E',unit='CODE')

        # make CC table
        tab = pyfits.BinTableHDU.from_columns([c1, c2, c3, c4, c5, c6, c7])
        return tab

    def save_fits(self, outfitsfile, overwrite=True):
        '''
        save the image(s) to the image FITS file.

        Args:
            outfitsfile (string): file name
            overwrite (boolean): It True, an existing file will be overwritten.
        '''
        if os.path.isfile(outfitsfile):
            if overwrite:
                os.system("rm -f %s" % (outfitsfile))
                self.hdulist.writeto(outfitsfile)
            else:
                print("Warning: does not overwrite %s" % (outfitsfile))
        else:
            self.hdulist.writeto(outfitsfile)

    def angconv(self, unit1="deg", unit2="deg"):
        '''
        return a conversion factor from unit1 to unit2
        Available angular units are uas, mas, asec or arcsec, amin or arcmin and degree.
        '''
        if unit1 == unit2:
            return 1

        # Convert from unit1 to "arcsec"
        if unit1 == "deg":
            conv = 3600.
        elif unit1 == "rad":
            conv = 180. * 3600. / np.pi
        elif unit1 == "arcmin" or unit1 == "amin":
            conv = 60.
        elif unit1 == "arcsec" or unit1 == "asec":
            conv = 1.
        elif unit1 == "mas":
            conv = 1e-3
        elif unit1 == "uas":
            conv = 1e-6
        else:
            print("Error: unit1=%s is not supported" % (unit1))
            return -1

        # Convert from "arcsec" to unit2
        if unit2 == "deg":
            conv /= 3600.
        elif unit2 == "rad":
            conv /= (180. * 3600. / np.pi)
        elif unit2 == "arcmin" or unit2 == "amin":
            conv /= 60.
        elif unit2 == "arcsec" or unit2 == "asec":
            pass
        elif unit2 == "mas":
            conv *= 1e3
        elif unit2 == "uas":
            conv *= 1e6
        else:
            print("Error: unit2=%s is not supported" % (unit2))
            return -1

        return conv

    #-------------------------------------------------------------------------
    # Getting Some information about images
    #-------------------------------------------------------------------------
    def get_xygrid(self, twodim=False, angunit=None):
        '''
        calculate the grid of the image

        Args:
          angunit (string): Angular unit (uas, mas, asec or arcsec, amin or arcmin, degree)
          twodim (boolean): It True, the 2D grids will be returned. Otherwise, the 1D arrays will be returned
        '''
        if angunit is None:
            angunit = self.angunit

        dx = self.header["dx"]
        dy = self.header["dy"]
        Nx = self.header["nx"]
        Ny = self.header["ny"]
        Nxref = self.header["nxref"]
        Nyref = self.header["nyref"]
        xg = dx * (np.arange(Nx) - Nxref + 1) * self.angconv("deg", angunit)
        yg = dy * (np.arange(Ny) - Nyref + 1) * self.angconv("deg", angunit)
        if twodim:
            xg, yg = np.meshgrid(xg, yg)
        return xg, yg

    def get_imextent(self, angunit=None):
        '''
        calculate the field of view of the image

        Args:
          angunit (string): Angular unit (uas, mas, asec or arcsec, amin or arcmin, degree)
        '''
        if angunit is None:
            angunit = self.angunit

        dx = self.header["dx"]
        dy = self.header["dy"]
        Nx = self.header["nx"]
        Ny = self.header["ny"]
        Nxref = self.header["nxref"]
        Nyref = self.header["nyref"]
        xmax = (1 - Nxref - 0.5) * dx
        xmin = (Nx - Nxref + 0.5) * dx
        ymax = (Ny - Nyref + 0.5) * dy
        ymin = (1 - Nyref - 0.5) * dy
        return np.asarray([xmax, xmin, ymin, ymax]) * self.angconv("deg", angunit)

    def peak(self, absolute=False, istokes=0, ifreq=0):
        '''
        calculate the peak intensity of the image

        Args:
          istokes (integer): index for Stokes Parameter at which the peak intensity will be calculated
          ifreq (integer): index for Frequency at which the peak intensity will be calculated
        '''
        if absolute:
            t = np.argmax(np.abs(self.data[istokes, ifreq]))
            t = np.unravel_index(t, [self.header["ny"], self.header["nx"]])
            return self.data[istokes, ifreq][t]
        else:
            return self.data[istokes, ifreq].max()

    def totalflux(self, istokes=0, ifreq=0):
        '''
        calculate the total flux of the image

        Args:
          istokes (integer): index for Stokes Parameter at which the total flux will be calculated
          ifreq (integer): index for Frequency at which the total flux will be calculated
        '''
        return self.data[istokes, ifreq].sum()

    def l1norm(self, istokes=0, ifreq=0):
        '''
        calculate l1-norm of the image

        Args:
          istokes (integer): index for Stokes Parameter at which l1-norm will be calculated
          ifreq (integer): index for Frequency at which l1-norm will be calculated
        '''
        return np.abs(self.data[istokes, ifreq]).sum()

    #-------------------------------------------------------------------------
    # Plotting
    #-------------------------------------------------------------------------
    def imshow(self, logscale=False, angunit=None,
               vmin=None, istokes=0, ifreq=0, **imshow_args):
        '''
        plot contours of the image

        Args:
          logscale (boolean):
            If True, the color contour will be on log scales. Otherise,
            the color contour will be on linear scales.
          angunit (string):
            Angular Unit for the axis labels (pixel, uas, mas, asec or arcsec,
            amin or arcmin, degree)
          vmin (string):
            minimum value of the color contour in Jy/pixel
          istokes (integer):
            index for Stokes Parameter at which the image will be plotted
          ifreq (integer):
            index for Frequency at which the image will be plotted
          **imshow_args: Arguments will be input in matplotlib.pyplot.imshow
        '''
        if angunit is None:
            angunit = self.angunit

        # Get Image Axis
        if angunit == "pixel":
            imextent = None
        else:
            imextent = self.get_imextent(angunit)

        if logscale:
            plotdata = np.log(self.data[istokes, ifreq])
            if vmin is None:
                vmin_scaled = np.log(0.001 * self.peak(istokes, ifreq))
            else:
                vmin_scaled = np.log(vmin)
            plotdata[np.where(plotdata < vmin_scaled)] = vmin_scaled
            plt.imshow(plotdata, extent=imextent, origin="lower",
                       vmin=vmin_scaled, **imshow_args)
        else:
            plt.imshow(self.data[istokes, ifreq], extent=imextent, origin="lower",
                       vmin=vmin, **imshow_args)

        # Axis Label
        if angunit.lower().find("pixel") == 0:
            unit = "pixel"
        elif angunit.lower().find("uas") == 0:
            unit = r"$\rm \mu$as"
        elif angunit.lower().find("mas") == 0:
            unit = "mas"
        elif angunit.lower().find("arcsec") * angunit.lower().find("asec") == 0:
            unit = "arcsec"
        elif angunit.lower().find("arcmin") * angunit.lower().find("amin") == 0:
            unit = "arcmin"
        elif angunit.lower().find("deg") == 0:
            unit = "deg"
        else:
            unit = "mas"
        plt.xlabel("Relative RA (%s)" % (unit))
        plt.ylabel("Relative Dec (%s)" % (unit))

    def contour(self, cmul=None, levels=None, angunit=None,
                colors="white", relative=True,
                istokes=0, ifreq=0,
                **contour_args):
        '''
        plot contours of the image

        Arguments:
          istokes (integer): index for Stokes Parameter at which the image will be plotted
          ifreq (integer): index for Frequency at which the image will be plotted
          angunit (string): Angular Unit for the axis labels (pixel, uas, mas, asec or arcsec, amin or arcmin, degree)
          colors (string, array-like): colors of contour levels
          cmul: The lowest contour level. Default value is 1% of the peak intensity.
          levels: contour level. This will be multiplied with cmul.
          **contour_args: Arguments will be input in matplotlib.pyplot.contour
        '''
        if angunit is None:
            angunit = self.angunit

        # Get Image Axis
        if angunit == "pixel":
            imextent = None
        else:
            imextent = self.get_imextent(angunit)

        # Get image
        image = self.data[istokes, ifreq]

        if cmul is None:
            vmin = np.abs(image).max() * 0.01
        else:
            if relative:
                vmin = cmul * np.abs(image).max()
            else:
                vmin = cmul

        if levels is None:
            clevels = np.power(2, np.arange(10))
        else:
            clevels = np.asarray(levels)
        clevels = vmin * np.asarray(clevels)

        plt.contour(image, extent=imextent, origin="lower",
                    colors=colors, levels=clevels, ls="-", **contour_args)
        # plt.contour(image,extent=imextent,origin="lower",
        #            colors=colors,levels=-levels,ls="--",**contour_args)
        # Axis Label
        if angunit.lower().find("pixel") == 0:
            unit = "pixel"
        elif angunit.lower().find("uas") == 0:
            unit = r"$\rm \mu$as"
        elif angunit.lower().find("mas") == 0:
            unit = "mas"
        elif angunit.lower().find("arcsec") * angunit.lower().find("asec") == 0:
            unit = "arcsec"
        elif angunit.lower().find("arcmin") * angunit.lower().find("amin") == 0:
            unit = "arcmin"
        elif angunit.lower().find("deg") == 0:
            unit = "deg"
        else:
            unit = "mas"

        plt.xlabel("Relative RA (%s)" % (unit))
        plt.ylabel("Relative Dec (%s)" % (unit))

    #-------------------------------------------------------------------------
    # DS9
    #-------------------------------------------------------------------------
    def open_ds9(self):
        pass

    def read_ds9reg(self):
        pass
    #-------------------------------------------------------------------------
    # Output some information to files
    #-------------------------------------------------------------------------

    def to_difmapmod(self, outfile, threshold=None, relative=True,
                     istokes=0, ifreq=0):
        '''
        Save an image into a difmap model file

        Arguments:
          istokes (integer): index for Stokes Parameter at which the image will be saved
          ifreq (integer): index for Frequency at which the image will be saved
          threshold (float): pixels with the absolute intensity smaller than this value will be ignored.
          relative (boolean): If true, theshold value will be normalized with the peak intensity of the image.
          save_totalflux (boolean): If true, the total flux of the image will be conserved.
        '''
        Nx = self.header["nx"]
        Ny = self.header["ny"]
        xg, yg = self.get_xygrid(angunit="mas")
        X, Y = np.meshgrid(xg, yg)
        R = np.sqrt(X * X + Y * Y)
        theta = np.rad2deg(np.arctan2(X, Y))
        flux = self.data[istokes, ifreq]

        R = R.reshape(Nx * Ny)
        theta = theta.reshape(Nx * Ny)
        flux = flux.reshape(Nx * Ny)

        if threshold is None:
            thres = np.finfo(np.float32).eps
        else:
            if relative:
                thres = self.peak(istokes=istokes, ifreq=ifreq) * threshold
            else:
                thres = threshold
        thres = np.abs(thres)

        f = open(outfile, "w")
        for i in np.arange(Nx * Ny):
            if np.abs(flux[i]) < thres:
                continue
            line = "%20e %20e %20e\n" % (flux[i], R[i], theta[i])
            f.write(line)
        f.close()

    #-------------------------------------------------------------------------
    # Editing images
    #-------------------------------------------------------------------------
    def cpimage(self, fitsdata, save_totalflux=False, order=3):
        '''
        Copy the first image into the image grid specified in the secondaly input image.

        Arguments:
          fitsdata: input imagefite.imagefits object. This image will be copied into self.
          self: input imagefite.imagefits object specifying the image grid where the orgfits data will be copied.
          save_totalflux (boolean): If true, the total flux of the image will be conserved.
        '''
        # generate output imfits object
        outfits = copy.deepcopy(self)

        dx0 = fitsdata.header["dx"]
        dy0 = fitsdata.header["dy"]
        Nx0 = fitsdata.header["nx"]
        Ny0 = fitsdata.header["ny"]
        Nxr0 = fitsdata.header["nxref"]
        Nyr0 = fitsdata.header["nyref"]

        dx1 = outfits.header["dx"]
        dy1 = outfits.header["dy"]
        Nx1 = outfits.header["nx"]
        Ny1 = outfits.header["ny"]
        Nxr1 = outfits.header["nxref"]
        Nyr1 = outfits.header["nyref"]

        coord = np.zeros([2, Nx1 * Ny1])
        xgrid = (np.arange(Nx1) + 1 - Nxr1) * dx1 / dx0 + Nxr0 - 1
        ygrid = (np.arange(Ny1) + 1 - Nyr1) * dy1 / dy0 + Nyr0 - 1
        x, y = np.meshgrid(xgrid, ygrid)
        coord[0, :] = y.reshape(Nx1 * Ny1)
        coord[1, :] = x.reshape(Nx1 * Ny1)

        for idxs in np.arange(outfits.header["ns"]):
            for idxf in np.arange(outfits.header["nf"]):
                outfits.data[idxs, idxf] = sn.map_coordinates(
                    fitsdata.data[idxs, idxf], coord, order=order,
                    mode='constant', cval=0.0, prefilter=True).reshape([Ny1, Nx1]
                                                                       ) * dx1 * dy1 / dx0 / dy0
                # Flux Scaling
                if save_totalflux:
                    totalflux = fitsdata.totalflux(istokes=idxs, ifreq=idxf)
                    outfits.data[idxs, idxf] *= totalflux / \
                        outfits.totalflux(istokes=idxs, ifreq=idxf)

        outfits.update_fits()
        return outfits

    def gauss_convolve(self, majsize, minsize=None, x0=None, y0=None,
                       pa=0., scale=1., angunit=None, pos="rel", save_totalflux=False):
        '''
        Gaussian Convolution

        Arguments:
          self: input imagefite.imagefits object
          majsize (float): Major Axis Size
          minsize (float): Minor Axis Size. If None, it will be same to the Major Axis Size (Circular Gaussian)
          angunit (string): Angular Unit for the sizes (uas, mas, asec or arcsec, amin or arcmin, degree)
          pa (float): Position Angle of the Gaussian
          scale (float): The sizes will be multiplied by this value.
          save_totalflux (boolean): If true, the total flux of the image will be conserved.
        '''
        if minsize is None:
            minsize = majsize

        if angunit is None:
            angunit = self.angunit

        # Create outputdata
        outfits = copy.deepcopy(self)

        # Create Gaussian
        imextent = outfits.get_imextent(angunit)
        Imxref = (imextent[0] + imextent[1]) / 2.
        Imyref = (imextent[2] + imextent[3]) / 2.
        if x0 is None:
            x0 = 0.
        if y0 is None:
            y0 = 0.
        if pos=="rel":
            x0 += Imxref
            y0 += Imyref
		
        X, Y = outfits.get_xygrid(angunit=angunit, twodim=True)
        cospa = np.cos(np.deg2rad(pa))
        sinpa = np.sin(np.deg2rad(pa))
        X1 = (X - x0) * cospa - (Y - y0) * sinpa
        Y1 = (X - x0) * sinpa + (Y - y0) * cospa
        majsig = majsize / np.sqrt(2 * np.log(2)) / 2 * scale
        minsig = minsize / np.sqrt(2 * np.log(2)) / 2 * scale
        gauss = np.exp(-X1 * X1 / 2 / minsig / minsig -
                       Y1 * Y1 / 2 / majsig / majsig)

        # Replace nan with zero
        gauss[np.isnan(gauss)] = 0

        # Convolusion (except:gauss is zero array)
        if np.any(gauss != 0):
            gauss /= gauss.sum()
            for idxs in np.arange(outfits.header["ns"]):
                for idxf in np.arange(outfits.header["nf"]):
                    orgimage = outfits.data[idxs, idxf]
                    newimage = convolve_fft(orgimage, gauss)
                    outfits.data[idxs, idxf] = newimage
                    # Flux Scaling
                    if save_totalflux:
                        totalflux = self.totalflux(istokes=idxs, ifreq=idxf)
                        outfits.data[idxs, idxf] *= totalflux / \
                            outfits.totalflux(istokes=idxs, ifreq=idxf)

        # Update and Return
        outfits.update_fits()
        return outfits

    def ds9flag(self, regfile, save_totalflux=False):
        '''
        Flagging the image with DS9region file

        Arguments:
          self: input imagefite.imagefits object
          regfile (string): input DS9 region file
          save_totalflux (boolean): If true, the total flux of the image will be conserved.
        '''
        # create output fits
        outfits = copy.deepcopy(self)

        # original file
        xgrid = np.arange(self.header["nx"])
        ygrid = np.arange(self.header["ny"])
        X, Y = np.meshgrid(xgrid, ygrid)

        # Check which grids should be flagged
        pixels = _get_flagpixels(regfile, X, Y)
        pixels = (pixels == False)
        pixels = np.where(pixels)

        for idxs in np.arange(self.header["ns"]):
            for idxf in np.arange(self.header["nf"]):
                image = outfits.data[idxs, idxf]
                image[pixels] = 0.
                outfits.data[idxs, idxf] = image
                # Flux Scaling
                if save_totalflux:
                    totalflux = self.totalflux(istokes=idxs, ifreq=idxf)
                    outfits.data[idxs, idxf] *= totalflux / image.sum()

        # Update and Return
        outfits.update_fits()
        return outfits

    def read_cleanbox(self, regfile):
        # Read DS9-region file
        f = open(regfile)
        lines = f.readlines()
        f.close()

        # original file
        xgrid = np.arange(self.header["nx"])
        ygrid = np.arange(self.header["ny"])
        X, Y = np.meshgrid(xgrid, ygrid)
        area = np.zeros(X.shape, dtype="Bool")

        # Read each line
        for line in lines:
            # Skipping line
            if line[0] == "#":
                continue
            if "image" in line == True:
                continue
            if "(" in line == False:
                continue
            if "global" in line:
                continue

            # Replacing many characters to empty spaces
            line = line.replace("(", " ")
            line = line.replace(")", " ")
            while "," in line:
                line = line.replace(",", " ")

            # split line to elements
            elements = line.split(" ")
            while "" in elements:
                elements.remove("")
            while "\n" in elements:
                elements.remove("\n")

            if len(elements) < 4:
                continue

            # Check whether the box is for "inclusion" or "exclusion"
            if elements[0][0] == "-":
                elements[0] = elements[0][1:]
                exclusion = True
            else:
                exclusion = False

            if elements[0] == "box":
                tmparea = _region_box(X, Y,
                                      x0=np.float64(elements[1]),
                                      y0=np.float64(elements[2]),
                                      width=np.float64(elements[3]),
                                      height=np.float64(elements[4]),
                                      angle=np.float64(elements[5]))
            elif elements[0] == "circle":
                tmparea = _region_circle(X, Y,
                                         x0=np.float64(elements[1]),
                                         y0=np.float64(elements[2]),
                                         radius=np.float64(elements[3]))
            elif elements[0] == "ellipse":
                tmparea = _region_ellipse(X, Y,
                                          x0=np.float64(elements[1]),
                                          y0=np.float64(elements[2]),
                                          radius1=np.float64(elements[3]),
                                          radius2=np.float64(elements[4]),
                                          angle=np.float64(elements[5]))
            else:
                print("[WARNING] The shape %s is not available." %
                      (elements[0]))

            if not exclusion:
                area += tmparea
            else:
                area[np.where(tmparea)] = False

        return area

    def comshift(self, save_totalflux=False, ifreq=0, istokes=0):
        '''
        Shift the image so that its center-of-mass position coincides with the reference pixel.

        Arguments:
          istokes (integer):
            index for Stokes Parameter at which the image will be edited
          ifreq (integer):
            index for Frequency at which the image will be edited
          save_totalflux (boolean):
            If true, the total flux of the image will be conserved.

        Returns:
          imdata.IMFITS object
        '''
        # create output fits
        outfits = copy.deepcopy(self)
        image = outfits.data[istokes, ifreq]
        nxref = outfits.header["nxref"]
        nyref = outfits.header["nyref"]

        # move the center of mass to the actual center of the self
        pix = sn.measurements.center_of_mass(image)
        outfits.data[istokes, ifreq] = sn.interpolation.shift(
            image, np.asarray([nyref - 1, nxref - 1]) - pix)

        # scale total flux
        if save_totalflux:
            totalflux = self.totalflux(istokes=istokes, ifreq=ifreq)
            outfits.data[istokes, ifreq] *= totalflux / \
                outfits.totalflux(istokes=istokes, ifreq=ifreq)

        # update FITS
        outfits.update_fits()
        return outfits

    def peakshift(self, save_totalflux=False, ifreq=0, istokes=0):
        '''
        Shift the image so that its peak position coincides with the reference pixel.

        Arg:
          istokes (integer):
            index for Stokes Parameter at which the image will be edited
          ifreq (integer):
            index for Frequency at which the image will be edited
          save_totalflux (boolean):
            If true, the total flux of the image will be conserved.

        Returns:
          imdata.IMFITS object
        '''
        # create output fits
        outfits = copy.deepcopy(self)
        image = outfits.data[istokes, ifreq]
        nxref = outfits.header["nxref"]
        nyref = outfits.header["nyref"]

        # move the center of mass to the actual center of the self
        pix = np.unravel_index(np.argmax(image), image.shape)
        outfits.data[istokes, ifreq] = sn.interpolation.shift(
            image, np.asarray([nyref - 1, nxref - 1]) - pix)
        # scale total flux
        if save_totalflux:
            totalflux = self.totalflux(istokes=istokes, ifreq=ifreq)
            outfits.data[istokes, ifreq] *= totalflux / \
                outfits.totalflux(istokes=istokes, ifreq=ifreq)
        # update FITS
        outfits.update_fits()
        return outfits

    def zeropad(self, Mx, My):
        '''
        Uniformly pad zero and extend fov of the image to (My, Mx) pixels.

        Args:
          self: input imagefite.imagefits object
          Mx (integer): Number of pixels in RA (x) direction for the padded image.
          My (integer): Number of pixels in Dec(y) direction for the padded image.

        Returns:
          imdata.IMFITS object
        '''
        # create output fits
        outfits = copy.deepcopy(self)
        Nx = outfits.header["nx"]
        Ny = outfits.header["ny"]
        Nf = outfits.header["nf"]
        Ns = outfits.header["ns"]
        if (Nx > Mx):
            print("[Error] please set a pixel size for RA  larger than original one!")
            return -1
        if (Ny > My):
            print("[Error] please set a pixel size for Dec larger than original one!")
            return -1
        newdata = np.zeros([Ns, Nf, My, Mx])
        for istokes in np.arange(Ns):
            for ifreq in np.arange(Nf):
                # update data
                newdata[istokes, ifreq, np.around(My / 2 - Ny / 2):np.around(My / 2 - Ny / 2) + Ny, np.around(
                    Mx / 2 - Nx / 2):np.around(Mx / 2 - Nx / 2) + Nx] = outfits.data[istokes, ifreq]
        outfits.data = newdata

        # update pixel info
        outfits.header["nx"] = Mx
        outfits.header["ny"] = My
        outfits.header["nxref"] += Mx / 2 - Nx / 2
        outfits.header["nyref"] += My / 2 - Ny / 2
        outfits.update_fits()

        return outfits

    def rotate(self, angle=0, deg=True, save_totalflux=False):
        '''
        Rotate the input image

        Arguments:
          self: input imagefite.imagefits object
          angle (float): Rotational Angle. Anti-clockwise direction will be positive (same to the Position Angle).
          deg (boolean): It true, then the unit of angle will be degree. Otherwise, it will be radian.
          save_totalflux (boolean): If true, the total flux of the image will be conserved.
        '''
        # create output fits
        outfits = copy.deepcopy(self)
        if deg:
            degangle = -angle
            radangle = -np.deg2rad(angle)
        else:
            degangle = -np.rad2deg(angle)
            radangle = -angle
        #cosa = np.cos(radangle)
        #sina = np.sin(radangle)
        Nx = outfits.header["nx"]
        Ny = outfits.header["ny"]
        for istokes in np.arange(self.header["ns"]):
            for ifreq in np.arange(self.header["nf"]):
                image = outfits.data[istokes, ifreq]
                # rotate data
                newimage = sn.rotate(image, degangle)
                # get the size of new data
                My = newimage.shape[0]
                Mx = newimage.shape[1]
                # take the center of the rotated image
                outfits.data[istokes, ifreq] = newimage[np.around(My / 2 - Ny / 2):np.around(My / 2 - Ny / 2) + Ny,
                                                        np.around(Mx / 2 - Nx / 2):np.around(Mx / 2 - Nx / 2) + Nx]
                # Flux Scaling
                if save_totalflux:
                    totalflux = self.totalflux(istokes=istokes, ifreq=ifreq)
                    outfits.data[istokes, ifreq] *= totalflux / \
                        outfits.totalflux(istokes=istokes, ifreq=ifreq)
        outfits.update_fits()
        return outfits

    def hard_threshold(self, threshold=0.01, relative=True, save_totalflux=False,
                       istokes=0, ifreq=0):
        '''
        Do hard-threshold the input image

        Arguments:
          istokes (integer): index for Stokes Parameter at which the image will be edited
          ifreq (integer): index for Frequency at which the image will be edited
          threshold (float): threshold
          relative (boolean): If true, theshold value will be normalized with the peak intensity of the image
          save_totalflux (boolean): If true, the total flux of the image will be conserved.
        '''
        # create output fits
        outfits = copy.deepcopy(self)
        if relative:
            thres = np.abs(threshold * self.peak(istokes=istokes, ifreq=ifreq))
        else:
            thres = np.abs(threshold)
        # thresholding
        image = outfits.data[istokes, ifreq]
        t = np.where(np.abs(self.data[istokes, ifreq]) < thres)
        image[t] = 0
        outfits.data[istokes, ifreq] = image
        # flux scaling
        if save_totalflux:
            totalflux = self.totalflux(istokes=istokes, ifreq=ifreq)
            outfits.data[istokes, ifreq] *= totalflux / \
                outfits.totalflux(istokes=istokes, ifreq=ifreq)
        outfits.update_fits()
        return outfits

    def soft_threshold(self, threshold=0.01, relative=True, save_totalflux=False,
                       istokes=0, ifreq=0):
        '''
        Do soft-threshold the input image

        Arguments:
          istokes (integer): index for Stokes Parameter at which the image will be edited
          ifreq (integer): index for Frequency at which the image will be edited
          threshold (float): threshold
          relative (boolean): If true, theshold value will be normalized with the peak intensity of the image
          save_totalflux (boolean): If true, the total flux of the image will be conserved.
        '''
        # create output fits
        outfits = copy.deepcopy(self)
        if relative:
            thres = np.abs(threshold * self.peak(istokes=istokes, ifreq=ifreq))
        else:
            thres = np.abs(threshold)
        # thresholding
        image = outfits.data[istokes, ifreq]
        t = np.where(np.abs(self.data[istokes, ifreq]) < thres)
        image[t] = 0
        t = np.where(self.data[istokes, ifreq] >= thres)
        image[t] -= thres
        t = np.where(self.data[istokes, ifreq] <= -thres)
        image[t] += thres
        outfits.data[istokes, ifreq] = image
        if save_totalflux:
            totalflux = self.totalflux(istokes=istokes, ifreq=ifreq)
            outfits.data[istokes, ifreq] *= totalflux / \
                outfits.totalflux(istokes=istokes, ifreq=ifreq)
        outfits.update_fits()
        return outfits

    def add_gauss(self, x0=0., y0=0., totalflux=1., majsize=1., minsize=None,
                  pa=0., istokes=0, ifreq=0, angunit=None):
        if angunit is None:
            angunit = self.angunit

        # copy self (for output)
        outfits = copy.deepcopy(self)

        # get size
        thmaj = majsize
        if minsize is None:
            thmin = thmaj
        else:
            thmin = minsize

        # Calc X,Y grid
        #yg = (np.arange(self.header["ny"]) - self.header["nyref"] +
        #      1) * self.header["dy"] * 3600e3  # (mas)
        #xg = (np.arange(self.header["nx"]) - self.header["nxref"] +
        #      1) * self.header["dx"] * 3600e3  # (mas)
        #X, Y = np.meshgrid(xg, yg)
        X, Y = self.get_xygrid(twodim=True, angunit=angunit)

        # Calc Gaussian Distribution
        X1 = X - x0
        Y1 = Y - y0
        cospa = np.cos(np.deg2rad(pa))
        sinpa = np.sin(np.deg2rad(pa))
        X2 = X1 * cospa - Y1 * sinpa
        Y2 = X1 * sinpa + Y1 * cospa
        majsig = thmaj / np.sqrt(2 * np.log(2)) / 2
        minsig = thmin / np.sqrt(2 * np.log(2)) / 2
        gauss = np.exp(-X2 * X2 / 2 / minsig / minsig -
                       Y2 * Y2 / 2 / majsig / majsig)
        gauss /= gauss.sum()
        gauss *= totalflux

        # add to original FITS file
        outfits.data[istokes, ifreq] += gauss

        return outfits

    def edge_detect(self, method="prewitt", mask=None, sigma=1,
                    low_threshold=0.1, high_threshold=0.2):
        '''
        Output edge-highlighted images.

        Args:
          method (string, default="prewitt"):
            Type of edge filters to be used.
            Availables are ["prewitt","sobel","scharr","roberts","canny"].
          mask (array):
            array for masking
          sigma (integer):
            index for canny
          low_threshold (float):
            index for canny
          high_threshold (float):
            index for canny

        Returns:
          imdata.IMFITS object
        '''
        from skimage.filters import prewitt, sobel, scharr, roberts
        from skimage.feature import canny

        # copy self (for output)
        outfits = copy.deepcopy(self)

        # get information
        nstokes = outfits.header["ns"]
        nif = outfits.header["nf"]
        # detect edge
        # prewitt
        if method == "prewitt":
            if mask is None:
                for idxs in np.arange(nstokes):
                    for idxf in np.arange(nif):
                        outfits.data[idxs, idxf] = prewitt(
                            outfits.data[idxs, idxf])
            else:
                for idxs in np.arange(nstokes):
                    for idxf in np.arange(nif):
                        outfits.data[idxs, idxf] = prewitt(
                            outfits.data[idxs, idxf], mask=mask)
        # sobel
        if method == "sobel":
            if mask is None:
                for idxs in np.arange(nstokes):
                    for idxf in np.arange(nif):
                        outfits.data[idxs, idxf] = sobel(
                            outfits.data[idxs, idxf])
            else:
                for idxs in np.arange(nstokes):
                    for idxf in np.arange(nif):
                        outfits.data[idxs, idxf] = sobel(
                            outfits.data[idxs, idxf], mask=mask)
        # scharr
        if method == "scharr":
            if mask is None:
                for idxs in np.arange(nstokes):
                    for idxf in np.arange(nif):
                        outfits.data[idxs, idxf] = scharr(
                            outfits.data[idxs, idxf])
            else:
                for idxs in np.arange(nstokes):
                    for idxf in np.arange(nif):
                        outfits.data[idxs, idxf] = scharr(
                            outfits.data[idxs, idxf], mask=mask)
        # roberts
        if method == "roberts":
            if mask is None:
                for idxs in np.arange(nstokes):
                    for idxf in np.arange(nif):
                        outfits.data[idxs, idxf] = roberts(
                            outfits.data[idxs, idxf])
            else:
                for idxs in np.arange(nstokes):
                    for idxf in np.arange(nif):
                        outfits.data[idxs, idxf] = roberts(
                            outfits.data[idxs, idxf], mask=mask)
        # canny
        if method == "canny":
            if mask is None:
                for idxs in np.arange(nstokes):
                    for idxf in np.arange(nif):
                        outfits.data[idxs, idxf] = canny(
                            outfits.data[idxs, idxf], sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold, use_quantiles=True)
            else:
                for idxs in np.arange(nstokes):
                    for idxf in np.arange(nif):
                        outfits.data[idxs, idxf] = canny(outfits.data[idxs, idxf], mask=mask, sigma=sigma,
                                                         low_threshold=low_threshold, high_threshold=high_threshold, use_quantiles=True)

        outfits.update_fits()
        return outfits

    def circle_hough(self, radius, ntheta=360,
                     angunit=None, istokes=0, ifreq=0):
        '''
        A function calculates the circle Hough transform (CHT) of the input image

        Args:
          radius (array):
            array for radii for which the circle Hough transform is
            calculated. The unit of the radius is specified with angunit.
          Ntheta (optional, integer):
            The number of circular shifts to be used in the circle Hough transform.
            For instance, ntheta=360 (default) gives circular shifts of every 1 deg.
          angunit (optional, string):
            The angular unit for radius and also the output peak profile
          istokes (integer): index for Stokes Parameter at which the CHT to be performed
          ifreq (integer): index for Frequency at which the CHT to be performed

        Returns:
          H (ndarray):
            The Circle Hough Accumulator. This is a three dimensional array of which
            shape is [Nx, Ny, Nr] in *Fortran Order*.
          profile (pd.DataFrame):
            The table for the peak profile Hr(r)=max_r(H(x,y,r)).
        '''
        if angunit is None:
            angunit = self.angunit

        Nr = len(radius)
        Nx = self.header["nx"]
        Ny = self.header["ny"]

        # get xy-coordinates
        xgrid, ygrid = self.get_xygrid(angunit=angunit)
        if self.header["dx"] < 0:
            sgnx = -1
        else:
            sgnx = 1
        if self.header["dy"] < 0:
            sgny = -1
        else:
            sgny = 1

        # calculate circle hough transform
        H = fortlib.hough_lib.circle_hough(self.data[istokes, ifreq],
                                           sgnx * xgrid, sgny * ygrid,
                                           radius, np.int32(ntheta))
        isfort = np.isfortran(H)

        # make peak profile
        profile = pd.DataFrame()
        profile["ir"] = np.arange(Nr)
        profile["r"] = radius
        profile["xpeak"] = np.zeros(Nr)
        profile["ypeak"] = np.zeros(Nr)
        profile["ixpeak"] = np.zeros(Nr, dtype=np.int64)
        profile["iypeak"] = np.zeros(Nr, dtype=np.int64)
        profile["hpeak"] = np.zeros(Nr)
        if isfort:
            for i in np.arange(Nr):
                profile.loc[i, "hpeak"] = np.max(H[:, :, i])
                peakxyidx = np.unravel_index(
                    np.argmax(H[:, :, i]), dims=[Ny, Nx])
                profile.loc[i, "xpeak"] = xgrid[peakxyidx[1]]
                profile.loc[i, "ypeak"] = ygrid[peakxyidx[0]]
                profile.loc[i, "ixpeak"] = peakxyidx[1]
                profile.loc[i, "iypeak"] = peakxyidx[0]
        else:
            for i in np.arange(Nr):
                profile.loc[i, "hpeak"] = np.max(H[i, :, :])
                peakxyidx = np.unravel_index(
                    np.argmax(H[i, :, :]), dims=[Ny, Nx])
                profile.loc[i, "xpeak"] = xgrid[peakxyidx[1]]
                profile.loc[i, "ypeak"] = ygrid[peakxyidx[0]]
                profile.loc[i, "ixpeak"] = peakxyidx[1]
                profile.loc[i, "iypeak"] = peakxyidx[0]
        return H, profile


#-------------------------------------------------------------------------
# Calculate Matrix Among Images
#-------------------------------------------------------------------------
def calc_metric(fitsdata, reffitsdata, metric="NRMSE", istokes1=0, ifreq1=0, istokes2=0, ifreq2=0, edgeflag=False):
    '''
    Calculate metrics between two images

    Args:
      fitsdata (imdata.IMFITS object):
        input image

      reffitsdata (imdata.IMFITS object):
        reference image

      metric (string):
        type of a metric to be calculated.
        Availables are ["NRMSE","MSE","SSIM","DSSIM"]

      istokes1 (integer):
        index for the Stokes axis of the input image

      ifreq1 (integer):
        index for the frequency axis of the input image

      istokes2 (integer):
        index for the Stokes axis of the reference image

      ifreq2 (integer):
        index for the frequency axis of the reference image

      edgeflag (boolean):
        calculation of metric on image domain or image gradient domain

    Returns:
      ???
    '''
    from skimage.filters import prewitt

    # adjust resolution and FOV
    fitsdata2 = copy.deepcopy(fitsdata)
    reffitsdata2 = copy.deepcopy(reffitsdata)
    fitsdata2 = cpimage(fitsdata2, reffitsdata2)
    # edge detection
    if edgeflag:
        fitsdata2 = edge_detect(fitsdata2, method="sobel")
        reffitsdata2 = edge_detect(reffitsdata2, method="sobel")
    # get image data
    inpimage = fitsdata2.data[istokes1, ifreq1]
    refimage = reffitsdata2.data[istokes2, ifreq2]
    # calculate metric
    if metric == "NRMSE" or metric == "MSE":
        metrics = np.sum((inpimage - refimage)**2)
        metrics /= np.sum(refimage**2)
    if metric == "SSIM" or metric == "DSSIM":
        meanI = np.mean(inpimage)
        meanK = np.mean(refimage)
        stdI = np.std(inpimage, ddof=1)
        stdK = np.std(refimage, ddof=1)
        cov = np.sum((inpimage - meanI) * (refimage - meanK)) / \
            (inpimage.size - 1)
        metrics = (2 * meanI * meanK / (meanI**2 + meanK**2)) * \
            (2 * stdI * stdK / (stdI**2 + stdK**2)) * (cov / (stdI * stdK))
    if metric == "NRMSE":
        metrics = np.sqrt(metrics)
    if metric == "DSSIM":
        metrics = 1 / abs(metrics) - 1

    return metrics


#-------------------------------------------------------------------------
# Fllowings are subfunctions for ds9flag and read_cleanbox
#-------------------------------------------------------------------------
def _get_flagpixels(regfile, X, Y):
    # Read DS9-region file
    f = open(regfile)
    lines = f.readlines()
    f.close()
    keep = np.zeros(X.shape, dtype="Bool")
    # Read each line
    for line in lines:
        # Skipping line
        if line[0] == "#":
            continue
        if "image" in line == True:
            continue
        if "(" in line == False:
            continue
        if "global" in line:
            continue
        # Replacing many characters to empty spaces
        line = line.replace("(", " ")
        line = line.replace(")", " ")
        while "," in line:
            line = line.replace(",", " ")
        # split line to elements
        elements = line.split(" ")
        while "" in elements:
            elements.remove("")
        while "\n" in elements:
            elements.remove("\n")
        if len(elements) < 4:
            continue
        # Check whether the box is for "inclusion" or "exclusion"
        if elements[0][0] == "-":
            elements[0] = elements[0][1:]
            exclusion = True
        else:
            exclusion = False
        if elements[0] == "box":
            tmpkeep = _region_box(X, Y,
                                  x0=np.float64(elements[1]),
                                  y0=np.float64(elements[2]),
                                  width=np.float64(elements[3]),
                                  height=np.float64(elements[4]),
                                  angle=np.float64(elements[5]))
        elif elements[0] == "circle":
            tmpkeep = _region_circle(X, Y,
                                     x0=np.float64(elements[1]),
                                     y0=np.float64(elements[2]),
                                     radius=np.float64(elements[3]))
        elif elements[0] == "ellipse":
            tmpkeep = _region_ellipse(X, Y,
                                      x0=np.float64(elements[1]),
                                      y0=np.float64(elements[2]),
                                      radius1=np.float64(elements[3]),
                                      radius2=np.float64(elements[4]),
                                      angle=np.float64(elements[5]))
        else:
            print("[WARNING] The shape %s is not available." % (elements[0]))
        if not exclusion:
            keep += tmpkeep
        else:
            keep[np.where(tmpkeep)] = False
    return keep


def _region_box(X, Y, x0, y0, width, height, angle):
    cosa = np.cos(np.deg2rad(angle))
    sina = np.sin(np.deg2rad(angle))
    dX = X - x0
    dY = Y - y0
    X1 = dX * cosa + dY * sina
    Y1 = -dX * sina + dY * cosa
    region = (Y1 >= -np.abs(height) / 2.)
    region *= (Y1 <= np.abs(height) / 2.)
    region *= (X1 >= -np.abs(width) / 2.)
    region *= (X1 <= np.abs(width) / 2.)
    return region


def _region_circle(X, Y, x0, y0, radius):
    return (X - x0) * (X - x0) + (Y - y0) * (Y - y0) <= radius * radius


def _region_ellipse(X, Y, x0, y0, radius1, radius2, angle):
    cosa = np.cos(np.deg2rad(angle))
    sina = np.sin(np.deg2rad(angle))
    dX = X - x0
    dY = Y - y0
    X1 = dX * cosa + dY * sina
    Y1 = -dX * sina + dY * cosa
    return X1 * X1 / radius1 / radius1 + Y1 * Y1 / radius2 / radius2 <= 1
