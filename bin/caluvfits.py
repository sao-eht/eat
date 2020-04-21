#!/usr/bin/env python

#hops2uvfits.py
#take data from all fringe files in a directory and put them in a uvfits file
import numpy as np
import mk4 # part of recent HOPS install, need HOPS ENV variables
import datetime
import ctypes
import astropy.io.fits as fits
import astropy.time as at
from argparse import Namespace
import glob
import os, sys
#import eat.hops.util
#from eat.io import util
#from eat.plots import util as putil
#from astropy.time import Time
import numpy.matlib
import scipy.interpolate
import itertools as it
from hops2uvfits import *

# For Andrew:
#DATADIR_DEFAULT = '/home/achael/EHT/hops/data/3554/' #/098-0924/'
DAY=str(3600)
CALDIR_DEFAULT = '/home/achael/Desktop/imaging_workshop/HOPS_Rev1/SEFDs/SEFD_HI/'+DAY
DATADIR_DEFAULT = '/home/achael/Desktop/imaging_workshop/HOPS_Rev1/er1-hops-hi/6.uvfits_new/'+DAY
OUTDIR_DEFAULT = '/home/achael/Desktop/imaging_workshop/HOPS_Rev1/er1-hops-hi/7.apriorical'

#For Katie
#CALDIR_DEFAULT = '/Users/klbouman/Research/vlbi_imaging/software/hops/eat/SEFDs/SEFD_HI/3601'
#DATADIR_DEFAULT =  '/Users/klbouman/Research/vlbi_imaging/software/hops/tmpout2' #'/Users/klbouman/Research/vlbi_imaging/software/hops/er1-hops-hi/6.uvfits/3601'
#OUTDIR_DEFAULT = '/Users/klbouman/Research/vlbi_imaging/software/hops/tmpout'


#conversion factors and data types
station_dic = {'ALMA':'AA', 'APEX':'AP', 'SMTO':'AZ', 'JCMT':'JC', 'LMT':'LM', 'PICOVEL':'PV', 'SMAP':'SM', 'SMAR':'SR', 'SPT':'SP'}
BLTYPE = [('time','f8'),('t1','a32'),('t2','a32')]
DTARR = [('site', 'a32'), ('x','f8'), ('y','f8'), ('z','f8')]
DTCAL = [('time','f8'), ('rscale','c16'), ('lscale','c16')]
DTPOL = [('time','f8'),('freq','f8'),('tint','f8'),
            ('t1','a32'),('t2','a32'),
            ('u','f8'),('v','f8'),
            ('rr','c16'),('ll','c16'),('rl','c16'),('lr','c16'),
            ('rrweight','f8'),('llweight','f8'),('rlweight','f8'),('lrweight','f8')]
EP = 1.e-5
CORRCOEFF = 10000.0
DEGREE = np.pi/180.0
HOUR = 15.0*DEGREE
C = 299792458.0
MHZ2HZ = 1e6
MJD_0 = 2400000.5
RADPERARCSEC = (np.pi / 180.) / 3600.

##################################################################################################
# Caltable object
##################################################################################################
# ANDREW TODO copied from caltable.py in ehtim
# load directly instead?
class Caltable(object):
    """
       Attributes:
    """

    def __init__(self, ra, dec, rf, bw, datatables, tarr, source='NONE', mjd=0, timetype='UTC'):
        """A polarimetric VLBI observation of visibility amplitudes and phases (in Jy).

           Args:

           Returns:
               caltable (Caltable): an Caltable object
        """

        if len(datatables) == 0:
            raise Exception("No data in input table!")

        # Set the various parameters
        self.source = str(source)
        self.ra = float(ra)
        self.dec = float(dec)
        self.rf = float(rf)
        self.bw = float(bw)
        self.mjd = int(mjd)

        if timetype not in ['GMST', 'UTC']:
            raise Exception("timetype must by 'GMST' or 'UTC'")
        self.timetype = timetype
        self.tarr = tarr

        # Dictionary of array indices for site names
        self.tkey = {self.tarr[i]['site']: i for i in range(len(self.tarr))}

        # Save the data
        self.data = datatables

    def copy(self):
        """Copy the caltable object.

           Args:

           Returns:
               (Caltable): a copy of the Caltable object.
        """
        new_caltable = Caltable(self.ra, self.dec, self.rf, self.bw, self.data, self.tarr, source=self.source, mjd=self.mjd, timetype=self.timetype)
        return new_caltable


def load_caltable_ds(datastruct, tabledir, sqrt_gains=False ):
    """Load apriori cal tables
    """

    if datastruct.dtype != "EHTIM":
        raise Exception("datastruct must be in EHTIM format in load_caltable!")
    tarr = datastruct.antenna_info
    source = datastruct.obs_info.src
    mjd = int(np.min(datastruct.data['time'] - MJD_0))
    ra = datastruct.obs_info.ra
    dec = datastruct.obs_info.dec
    rf = datastruct.obs_info.ref_freq
    bw = datastruct.obs_info.ch_bw

    datatables = {}
    for s in range(0, len(tarr)):

        site = tarr[s]['site']
        filename = tabledir + "/" + source + '_' + site + '.txt'
        try:
            data = np.loadtxt(filename, dtype=bytes).astype(str)
        except IOError:
            print "NO FILE: " + filename
            continue

        datatable = []

        # ANDREW HACKY WAY TO MAKE IT WORK WITH ONLY ONE ENTRY
        onerowonly=False
        try: data.shape[1]
        except IndexError:
            data = data.reshape(1,len(data))
            onerowonly = True
        for row in data:

            time = (float(row[0]) - mjd) * 24.0 # time is given in mjd

 #            # Maciek's old convention had a square root
 #           rscale = np.sqrt(float(row[1])) # r
 #           lscale = np.sqrt(float(row[2])) # l

            if len(row) == 3:
                rscale = float(row[1])
                lscale = float(row[2])
            elif len(row) == 5:
                rscale = float(row[1]) + 1j*float(row[2])
                lscale = float(row[3]) + 1j*float(row[4])
            else:
                raise Exception("cannot load caltable -- format unknown!")
            if sqrt_gains:
                rscale = rscale**.5
                lscale = lscale**.5
            #ANDREW THERE ARE ZERO VALS IN THE CALTABLE
            if rscale==0. and lscale==0.:
                continue
            else:
                datatable.append(np.array((time, rscale, lscale), dtype=DTCAL))
            #ANDREW HACKY WAY TO MAKE IT WORK WITH ONLY ONE ENTRY
            if onerowonly:
                datatable.append(np.array((1.1*time, rscale, lscale), dtype=DTCAL))

        datatables[site] = np.array(datatable)

    if len(datatables)>0:
        caltable = Caltable(ra, dec, rf, bw, datatables, tarr, source=source, mjd=mjd, timetype='UTC')
    else:
        caltable=False
    return caltable

def apply_caltable_uvfits(caltable, datastruct, filename_out, interp='linear', extrapolate=None):
    """apply a calibration table to a uvfits file
       Args:
        caltable (Caltable) : a caltable object
        datastruct (Datastruct) :  input data structure in EHTIM format
        filename_out (str) :  uvfits output file name
    """

    if datastruct.dtype != "EHTIM":
        raise Exception("datastruct must be in EHTIM format in apply_caltable_uvfits!")

    if not (caltable.tarr == datastruct.antenna_info).all():
        raise Exception("The telescope array in the Caltable is not the same as in the Datastruct")

    if extrapolate is True: # extrapolate can be a tuple or numpy array
        fill_value = "extrapolate"
    else:
        fill_value = extrapolate

    # interpolate the calibration  table
    rinterp = {}
    linterp = {}
    skipsites = []
    for s in range(0, len(caltable.tarr)):
        site = caltable.tarr[s]['site']
        try:
            caltable.data[site]
        except KeyError:
            skipsites.append(site)
            print ("No Calibration  Data for %s !" % site)
            continue

        time_mjd = caltable.data[site]['time']/24.0 + caltable.mjd

        rinterp[site] = scipy.interpolate.interp1d(time_mjd, caltable.data[site]['rscale'],
                                                   kind=interp, fill_value=fill_value)
        linterp[site] = scipy.interpolate.interp1d(time_mjd, caltable.data[site]['lscale'],
                                                   kind=interp, fill_value=fill_value)

    # sort by baseline
    data =  datastruct.data
    idx = np.lexsort((data['t2'], data['t1']))
    bllist = []
    for key, group in it.groupby(data[idx], lambda x: set((x['t1'], x['t2'])) ):
        bllist.append(np.array([obs for obs in group]))
    bllist = np.array(bllist)

    # apply the  calibration
    datatable = []
    for bl_obs in bllist:
        t1 = bl_obs['t1'][0]
        t2 = bl_obs['t2'][0]
        time_mjd = bl_obs['time'] - MJD_0 #dates are in mjd in Datastruct

        if t1 in skipsites:
            rscale1 = lscale1 = np.array(1.)
        else:
            rscale1 = rinterp[t1](time_mjd)
            lscale1 = linterp[t1](time_mjd)
        if t2 in skipsites:
            rscale2 = lscale2 = np.array(1.)
        else:
            rscale2 = rinterp[t2](time_mjd)
            lscale2 = linterp[t2](time_mjd)

#        if force_singlepol == 'R':
#            lscale1 = rscale1
#            lscale2 = rscale2
#        if force_singlepol == 'L':
#            rscale1 = lscale1
#            rscale2 = lscale2

        rrscale = rscale1 * rscale2.conj()
        llscale = lscale1 * lscale2.conj()
        rlscale = rscale1 * lscale2.conj()
        lrscale = lscale1 * rscale2.conj()

        bl_obs['rr'] = (bl_obs['rr']) * rrscale
        bl_obs['ll'] = (bl_obs['ll']) * llscale
        bl_obs['rl'] = (bl_obs['rl']) * rlscale
        bl_obs['lr'] = (bl_obs['lr']) * lrscale

        bl_obs['rrweight'] = (bl_obs['rrweight']) / (np.abs(rrscale)**2)
        bl_obs['llweight'] = (bl_obs['llweight']) / (np.abs(llscale)**2)
        bl_obs['rlweight'] = (bl_obs['rlweight']) / (np.abs(rlscale)**2)
        bl_obs['lrweight'] = (bl_obs['lrweight']) / (np.abs(lrscale)**2)

        if len(datatable):
            datatable = np.hstack((datatable, bl_obs))
        else:
            datatable = bl_obs

    # put in uvfits format datastruct
    # telescope arrays
    tarr = datastruct.antenna_info
    tkeys = {tarr[i]['site']: i for i in range(len(tarr))}
    tnames = tarr['site']
    tnums = np.arange(1, len(tarr) + 1)
    xyz = np.array([[tarr[i]['x'],tarr[i]['y'],tarr[i]['z']] for i in np.arange(len(tarr))])

    # uvfits format output data table
    bl_list = []
    for i in xrange(len(datatable)):
        entry = datatable[i]
        t1num = entry['t1']
        t2num = entry['t2']

        rl = entry['rl']
        lr = entry['lr']
        if tkeys[entry['t2']] < tkeys[entry['t1']]: # reorder telescopes if necessary
            #print entry['t1'], tkeys[entry['t1']], entry['t2'], tkeys[entry['t2']]
            entry['t1'] = t2num
            entry['t2'] = t1num
            entry['u'] = -entry['u']
            entry['v'] = -entry['v']
            entry['rr'] = np.conj(entry['rr'])
            entry['ll'] = np.conj(entry['ll'])
            entry['rl'] = np.conj(lr)
            entry['lr'] = np.conj(rl)
            datatable[i] = entry

        bl_list.append(np.array((entry['time'],entry['t1'],entry['t2']),dtype=BLTYPE))
    _, unique_idx_anttime, idx_anttime = np.unique(bl_list, return_index=True, return_inverse=True)
    _, unique_idx_freq, idx_freq = np.unique(datatable['freq'], return_index=True, return_inverse=True)

    # random group params
    u = datatable['u'][unique_idx_anttime]
    v = datatable['v'][unique_idx_anttime]
    t1num = [tkeys[scope] + 1 for scope in datatable['t1'][unique_idx_anttime]]
    t2num = [tkeys[scope] + 1 for scope in datatable['t2'][unique_idx_anttime]]
    bls = 256*np.array(t1num) + np.array(t2num)
    jds = datatable['time'][unique_idx_anttime]
    tints = datatable['tint'][unique_idx_anttime]

    # data table
    nap = len(unique_idx_anttime)
    nsubchan = 1
    nstokes = 4
    nchan = datastruct.obs_info.nchan

    outdat = np.zeros((nap, 1, 1, nchan, nsubchan, nstokes, 3))
    outdat[:,:,:,:,:,:,2] = -1.0

    vistypes = ['rr','ll','rl','lr']
    for i in xrange(len(datatable)):
        row_freq_idx = idx_freq[i]
        row_dat_idx = idx_anttime[i]

        for j in range(len(vistypes)):
            outdat[row_dat_idx,0,0,row_freq_idx,0,j,0] = np.real(datatable[i][vistypes[j]])
            outdat[row_dat_idx,0,0,row_freq_idx,0,j,1] = np.imag(datatable[i][vistypes[j]])
            outdat[row_dat_idx,0,0,row_freq_idx,0,j,2] = datatable[i][vistypes[j]+'weight']

    # package data for saving
    obsinfo_out = datastruct.obs_info
    antennainfo_out = Antenna_info(tnames, tnums, xyz)
    uvfitsdata_out = Uvfits_data(u,v,bls,jds, tints, outdat)
    datastruct_out = Datastruct(obsinfo_out, antennainfo_out, uvfitsdata_out)

    # save final file
    save_uvfits(datastruct_out, filename_out)
    return

##################################################################################################################################
##########################  Main FUNCTION ########################################################################################
##################################################################################################################################
def main(datadir=DATADIR_DEFAULT, caldir=CALDIR_DEFAULT, outdir=DATADIR_DEFAULT,
         interp='linear', extrapolate=True, ident='',sqrt_gains=False):

    print "********************************************************"
    print "*********************CALUVFITS**************************"
    print "********************************************************"

    print "Applying calibration tables from directory", caldir
    print "to uvfits files in directory: ", datadir
    print

    uvfitsfiles = glob.glob(datadir + '/*.uvfits')
    for uvfitsfile in uvfitsfiles:
        print
        print "A priori calibrating: ", uvfitsfile

        datastruct_ehtim = load_and_convert_hops_uvfits(uvfitsfile)
        source = datastruct_ehtim.obs_info.src
        tarr = datastruct_ehtim.antenna_info
        caltable = load_caltable_ds(datastruct_ehtim, caldir,sqrt_gains=sqrt_gains)
        if caltable==False:
            print "couldn't find caltable in " + caldir + " for " + source + "!!"
            continue


        outname = outdir + '/hops_' + os.path.basename(os.path.normpath(datadir)) + '_' + source + ident + '.apriori.uvfits'
        apply_caltable_uvfits(caltable, datastruct_ehtim, outname, interp=interp, extrapolate=extrapolate)
        print "Saved calibrated data to ", outname
    print "---------------------------------------------------------"
    print "---------------------------------------------------------"
    print "---------------------------------------------------------"
    print
    return 0

if __name__=='__main__':
    if len(sys.argv) == 1:
        datadir = DATADIR_DEFAULT
    else: datadir = sys.argv[-1]
    if datadir[0] == '-': datadir=DATADIR_DEFAULT

    if ("-h" in sys.argv) or ("--h" in sys.argv):
        print("usage: caluvfits.py datadir \n" +
              "options: \n" +
              "   --caldir caldir : specify directory with cal tables \n" +
              "   --outdir outdir : specifiy output directory for calibrated files \n" +
              "   --ident : specify identifying tag for uvfits files \n"
              "   --interp : specify interpolation order \n" +
              "   --no-extrapolate : specify to not calibrate outside interval in cal tables \n"
              "   --sqrt_gains : specify to take sqrt of gains before applying")
        sys.exit()

    extrapolate = True
    if "--no-extrapolate" in sys.argv: extrapolate = None

    sqrt_gains = False
    if "--sqrt_gains" in sys.argv: sqrt_gains = True

    interp = "linear"
    if "--interp" in sys.argv:
        for a in range(0, len(sys.argv)):
            if(sys.argv[a] == '--interp'):
                interp = int(sys.argv[a+1])
    ident = ""
    if "--ident" in sys.argv:
        for a in range(0, len(sys.argv)):
            if(sys.argv[a] == '--ident'):
                ident = "_" + sys.argv[a+1]

    caldir = CALDIR_DEFAULT
    if "--caldir" in sys.argv:
        for a in range(0, len(sys.argv)):
            if(sys.argv[a] == '--caldir'):
                caldir = sys.argv[a+1]

    outdir = datadir
    if "--outdir" in sys.argv:
        for a in range(0, len(sys.argv)):
            if(sys.argv[a] == '--outdir'):
                outdir = sys.argv[a+1]
    else:
        outdir = OUTDIR_DEFAULT

    main(datadir=datadir, outdir=outdir, caldir=caldir, ident=ident, interp=interp, extrapolate=extrapolate,sqrt_gains=sqrt_gains)
