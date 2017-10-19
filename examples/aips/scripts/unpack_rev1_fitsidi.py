#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This script unpack Rev1-Cal FITS IDI files in a systematic way.

See help documents with options "-h" or "--help" for more details.
'''
__author__      = "Kazunori Akiyama"
#__authors__     =
#__copyright__   = ""
#__credits__     = ""
#__license__     = ""
#__version__     = ""
#__contact__     = ""
#__maintainer__  = "Name"
#__email__       = ""
__date__        = "Oct 19 2017"
#__status__      = ""
#__deprecated__  =
# ------------------------------------------------------------------------------
# load modules
import os
from tqdm import tqdm
import astropy.io.fits as pf
import astropy.time as at
import numpy as np
from optparse import OptionParser
# ------------------------------------------------------------------------------

def main():
    # Option Parser
    usage ="%prog -i datadir [-o fitsdir] [-a obscodes] [-b bands] [--skipna] [-h or --help]"
    parser=OptionParser(usage=usage)
    parser.add_option("-i", type="string",dest="datadir",default=None,
                      metavar="DIR",help=r"Data directory that includes 'Rev1-Cal'[Mandatory]")
    parser.add_option("-o", type="string",dest="fitsdir",default=None,
                      metavar="DIR",help=r"Output Directory Names [Default: datadir/rev1-fitsidi]")
    parser.add_option("-a", type="string",dest="obscodes",default="e17d05,e17b06,e17c07,e17a10,e17e11",
                      metavar="string",help=r"list of obscodes to be read, seperated by ',' [Default: %default]")
    parser.add_option("-b", type="string",dest="bands",default="lo,hi",
                      metavar="string",help=r"list of bands to be read, seperated by ',' [Default: %default]")
    parser.add_option("--skipna", action="store_true", dest="skipna", default=False,
                      help="If specified, skip Non-ALMA data.")
    (opts,args)=parser.parse_args()
    argc=len(args)

    if opts.datadir is None:
        parser.error("Please secify data directory with '-i'")
    if opts.fitsdir is None:
        opts.fitsdir=os.path.join(opts.datadir,"rev1-fitsidi")
    
    unpack_rev1_fitsidi(
        datadir=opts.datadir,
        fitsdir=opts.fitsdir,
        obscodes=opts.obscodes,
        bands=opts.bands,
        skipna=opts.skipna)


def unpack_rev1_fitsidi(
        datadir,fitsdir,
        obscodes="e17d05,e17b06,e17c07,e17a10,e17e11",
        bands="lo,hi",
        skipna=False):
    
    # Check input directories
    rev1caldir = os.path.join(datadir,"Rev1-Cal")
    if not os.path.isdir(rev1caldir):
        raise ValueError("%s does not exist."%(rev1caldir))
    
    
    # List up FITS tarballs
    fitstars = []
    list1 = os.listdir(rev1caldir)
    for comp1 in tqdm(list1, bar_format="Listing up FITS tarballs: "+r'{l_bar}{bar}{r_bar}'):
        # get the relative path
        comp1path = os.path.join(rev1caldir,comp1)
        if os.path.isdir(comp1path) is False:
            continue
        
        list2 = os.listdir(comp1path)
        for comp2 in list2:
            # Check if this is a fits-tarball on the current obcode, rev, band
            if "fits.tar" not in comp2:
                continue
            
            # Check obscodes and band
            obscode, rev, band, tgtsrc = split_fitstarname(comp2)
            if (band not in bands) or (obscode not in obscodes):
                continue
            if skipna and "na-" in tgtsrc.lower():
                continue
            comp2path = os.path.join(comp1path,comp2)
            fitstars.append(comp2path)

    Ntar = len(fitstars)
    if Ntar > 0:
        print("- %d FITS tarballs are found."%(Ntar))
        print("")
    else:
        errmsg("No FITS tarballs are found in %s/*"%(revcaldir))
        raise ValueError(errmsg)

    # Unpack FITS tarballs
    for fitstar in tqdm(fitstars, bar_format="Unpacking FITS tarballs: "+r'{l_bar}{bar}{r_bar}'):
        # Get observationa information
        obscode, rev, band, tgtsrc = split_fitstarname(os.path.split(fitstar)[1])

        # Target Directory
        tgtdir = os.path.join(fitsdir, "%s-%s-%s"%(obscode,rev,band))
        run_command("mkdir -p %s"%(tgtdir))

        # Unpack Files
        tmpdir = os.path.join(fitsdir,"tmp","%s-%s-%s-%s"%(obscode, rev, band, tgtsrc))
        run_command("mkdir -p %s"%(tmpdir))
        run_command("tar xf %s -C %s --strip-components=1"%(fitstar, tmpdir))

        # rename unpacked FITS files
        list1 = os.listdir(tmpdir)
        for comp1 in list1:
            if os.path.splitext(comp1)[1] != ".FITS":
                continue
            comp1path = os.path.join(tmpdir, comp1)
            timestamp = get_timestamp(comp1path)
            tgtfilename = os.path.join(tgtdir,"%s_%s_%s.fits"%(timestamp, band, tgtsrc.lower()))
            run_command("mv -f %s %s"%(comp1path, tgtfilename))
        run_command("rm -rf %s"%(tmpdir))

def split_fitstarname(filename):
    splits = filename.split("-")
    obscode = splits[0]
    rev = splits[1]
    band = splits[2]
    tgtsrc = "-".join(splits[3:-1])
    return obscode, rev, band, tgtsrc 

def run_command(cmd, doprint=False):
    if doprint: print(cmd)
    os.system(cmd)

def get_timestamp(fitsfile):
    hdulist = pf.open(fitsfile)
    uvdata = hdulist["UV_DATA"]
    dates = at.Time(uvdata.data["DATE"], format="jd", scale="utc")
    times = at.TimeDelta(uvdata.data["TIME"], format="jd")
    datetimes = dates+times
    starttime = datetimes.min()
    hdulist.close()
    year,doy,h,m,s = starttime.yday.split(":")
    return "%04s-%03s-%02s%02s%02d"%(year,doy,h,m,np.int64(np.around(np.float64(s))))

if __name__=="__main__":
    main()
