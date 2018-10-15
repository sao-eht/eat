#!/usr/bin/env python2

#hops2uvfits.py
#take data from all fringe files in a directory and put them in a uvfits file
import numpy as np
import mk4 # part of recent HOPS install, need HOPS ENV variables
import datetime
import ctypes
import astropy.io.fits as fits
import astropy.time as at
from astropy.time import Time
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
import pandas as pd
import datetime
from datetime import timedelta
import pandas as pd

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
#station_dic = {'ALMA':'AA', 'APEX':'AP', 'SMTO':'AZ', 'JCMT':'JC', 'LMT':'LM', 'PICOVEL':'PV', 'SMAP':'SM', 'SMAR':'SR', 'SPT':'SP'}
station_dic = {'ALMA':'AA', 'A':'AA','AA':'AA',
           'APEX':'AP', 'X':'AP','AP': 'AP',
            'LMT':'LM','L':'LM','LM':'LM',
            'PICOVEL':'PV','P':'PV','IRAM30': 'PV','PV':'PV',
            'SMTO':'AZ','Z': 'AZ','SMT':'AZ','AZ':'AZ',
            'SPT':'SP','Y':'SP','SP':'SP',
            'JCMT':'JC','J':'JC','JC':'JC',
            'SMAP':'SM','S':'SM','SMAR':'SM','SMA':'SM','SM':'SM',
            'SMAR':'SR','R':'SR','SMR':'SR','SR':'SR'}

station_frot = {'PV':(1,-1,0),'AZ':(1,1,0),'SM':(1,-1,np.pi/4.),'LM': (1,-1,0),'AA':(1,0,0),'SP':(1,0,0),'AP':(1,1,0),'JC':(1,0,0),'SR':(1,-1,np.pi/4.)}

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

def poly_from_str(strcoeffs):
    coeffs = list(map(float, strcoeffs.split(',')))
    return np.polynomial.polynomial.Polynomial(coeffs)

def apply_caltable_uvfits(gaincaltable, datastruct, filename_out,cal_amp=False):
    """apply a calibration table to a uvfits file
       Args:
        caltable (Caltable) : a gaincaltable object
        datastruct (Datastruct) :  input data structure in EHTIM format
        filename_out (str) :  uvfits output file name
        cal_amp (bool): whether to do amplitude calibration
    """

    if datastruct.dtype != "EHTIM":
        raise Exception("datastruct must be in EHTIM format in apply_caltable_uvfits!")

    gains0 = pd.read_csv(gaincaltable)
    polygain={}
    mjd_start={}
    polyamp={}

    #deterimine which calibration to use when multiple options for multiple periods
    mjd_mean = datastruct.data['time'].mean()  - MJD_0
    gains = gains0[(gains0.mjd_start<=mjd_mean)&(gains0.mjd_stop>=mjd_mean)].reset_index(drop=True).copy()

    for cou, row in gains.iterrows():
        polygain[row.station] = poly_from_str(str(row.ratio_phas))
        mjd_start[row.station] = row.mjd_start
        if cal_amp==True:
            polyamp[row.station] = poly_from_str(str(row.ratio_amp))
        else:
            polyamp[row.station] = poly_from_str('1.0')

    #print(gains0)
    #print(polygain)
    # interpolate the calibration  table
    rinterp = {}
    linterp = {}
    skipsites = []

    #-------------------------------------------
    # sort by baseline
    data =  datastruct.data
    idx = np.lexsort((data['t2'], data['t1']))
    bllist = []
    for key, group in it.groupby(data[idx], lambda x: set((x['t1'], x['t2'])) ):
        bllist.append(np.array([obs for obs in group]))
    bllist = np.array(bllist)

    # apply the  calibration

    datatable = []
    coub=0
    for bl_obs in bllist:
        t1 = bl_obs['t1'][0]
        t2 = bl_obs['t2'][0]
        coub=coub+1
        print('Calibrating {}-{} baseline, {}/{}'.format(t1,t2,coub,len(bllist)))
        time_mjd = bl_obs['time'] - MJD_0 #dates are in mjd in Datastruct

        if t1 in skipsites:
            rscale1 = lscale1 = np.array(1.)
        else:
            try:
                rscale1 = 1./np.sqrt(polyamp[t1](time_mjd))
                lscale1 = np.sqrt(polyamp[t1](time_mjd))*np.exp(1j*polygain[t1](time_mjd - mjd_start[t1])*np.pi/180.)
            except KeyError:
                rscale1 = lscale1 = np.array(1.)

        if t2 in skipsites:
            rscale2 = lscale2 = np.array(1.)
        else:
            try:
                rscale2 = 1./np.sqrt(polyamp[t2](time_mjd))
                lscale2 = np.sqrt(polyamp[t2](time_mjd))*np.exp(1j*polygain[t2](time_mjd - mjd_start[t2])*np.pi/180.)
            except KeyError:
                rscale2 = lscale2 = np.array(1.) 

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
def main(datadir=DATADIR_DEFAULT, calfile=CALDIR_DEFAULT, outdir=DATADIR_DEFAULT, ident='',cal_amp=False):

    print("********************************************************")
    print("********************POLCALUVFITS************************")
    print("********************************************************")

    print("Applying polarimetric gains calibration tables from", calfile)
    print("to uvfits files in directory: ", datadir)
    print(' ')

    uvfitsfiles = glob.glob(datadir + '/*.uvfits')

    for uvfitsfile in sorted(uvfitsfiles):
        print(' ')
        print("Polcal gains calibrating: ", uvfitsfile)

        datastruct_ehtim = load_and_convert_hops_uvfits(uvfitsfile)
        source = datastruct_ehtim.obs_info.src
        tarr = datastruct_ehtim.antenna_info

        outname = outdir + '/hops_' + os.path.basename(os.path.normpath(datadir)) + '_' + source + ident + '.polcal.uvfits'
        apply_caltable_uvfits(calfile, datastruct_ehtim, outname,cal_amp=False)
        print("Saved calibrated data to ", outname)
    print("---------------------------------------------------------")
    print("---------------------------------------------------------")
    print("---------------------------------------------------------")
    print(' ')
    return 0

if __name__=='__main__':
    if len(sys.argv) == 1:
        datadir = DATADIR_DEFAULT
    else: datadir = sys.argv[-1]
    if datadir[0] == '-': datadir=DATADIR_DEFAULT

    if ("-h" in sys.argv) or ("--h" in sys.argv):
        print("usage: caluvfits.py datadir \n" +
              "options: \n" +
              "   --calfile calfile : specify directory with cal tables \n" +
              "   --outdir outdir : specifiy output directory for calibrated files \n" +
              "   --ident : specify identifying tag for uvfits files \n")
        sys.exit()


    cal_amp = False
    if "--cal_amp" in sys.argv: cal_amp = True

    ident = ""
    if "--ident" in sys.argv:
        for a in range(0, len(sys.argv)):
            if(sys.argv[a] == '--ident'):
                ident = "_" + sys.argv[a+1]

    calfile = CALDIR_DEFAULT
    if "--calfile" in sys.argv:
        for a in range(0, len(sys.argv)):
            if(sys.argv[a] == '--calfile'):
                calfile = sys.argv[a+1]

    outdir = datadir
    if "--outdir" in sys.argv:
        for a in range(0, len(sys.argv)):
            if(sys.argv[a] == '--outdir'):
                outdir = sys.argv[a+1]
    else:
        outdir = OUTDIR_DEFAULT

    main(datadir=datadir, calfile=calfile, outdir=outdir, ident=ident,cal_amp=cal_amp)
