# INI: This script contains routines for performing a priori flux calibration
# adapted from previous years' scripts by Maciek, CK, Lindy, and others.

import numpy as np
import datetime
from argparse import Namespace
import glob
import os
import scipy.interpolate
import itertools as it
from astropy.time import Time
from astropy.io import fits
import logging
#from hops2uvfits import *
#import mk4 # part of recent HOPS install, need HOPS ENV variables
#import ctypes
#import astropy.io.fits as fits
#import astropy.time as at
#import sys
#import numpy.matlib
#import pandas as pd

# Configure logging
loglevel = getattr(logging, 'INFO', None)
logging.basicConfig(level=loglevel,
                    format='%(asctime)s %(levelname)s:: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

#conversion factors and data types
'''stationdict = {'ALMA':'AA', 'A':'AA','AA':'AA',
            'APEX':'AX', 'X':'AX','AP': 'AX',
            'LMT':'LM','L':'LM','LM':'LM',
            'PICOVEL':'PV','P':'PV','IRAM30': 'PV','PV':'PV',
            'SMTO':'MG','Z': 'MG','SMT':'MG','AZ':'MG',
            'SPT':'SZ','Y':'SZ','SP':'SZ',
            'JCMT':'MM','J':'MM','JC':'MM',
            'SMAP':'SW','S':'SW','SMAR':'SW','SMA':'SW','SM':'SW',
            'GLT':'GL','G':'GL','THULE':'GL','GL':'GL',
            'NOEMA':'NN','N':'NN','NN':'NN',
            'KITTPEAK':'KT','K':'KT','KT':'KT',   
            'SMAR':'SR','R':'SR','SMR':'SR','SR':'SR',
            'GBT': 'GB','FD': 'FD','PT':'PT','LA':'LA', 
            'KP':'KP', 'MK':'MK', 'BR':'BR', 'NL':'NL',
            'OV':'OV','YS':'YS','EB':'EB'
            }

station_frot_old = {'PV':(1,-1,0),'AZ':(1,1,0),'SM':(1,-1,np.pi/4.),'LM': (1,-1,0),
                'AA':(1,0,0),'SP':(1,0,0),'AP':(1,1,0),'JC':(1,0,0),'SR':(1,-1,np.pi/4.),
                'GB':(1,0,0),'FD':(1,0,0),'PT':(1,0,0),'LA':(1,0,0),'KP':(1,0,0),
                'MK':(1,0,0),'BR':(1,0,0),'NL':(1,0,0),'OV':(1,0,0),'YS':(1,0,0),'EB':(1,0,0)}'''

STATION_FROT = {'PV':(1,-1,0),'MG':(1,1,0),'SW':(1,-1,np.pi/4.),'LM': (1,-1,0),
                'AA':(1,0,0),'SZ':(1,0,0),'AX':(1,1,0),'MM':(1,0,0),'GL': (1,0,0),
                'NN':(1,0,0),'KT':(1,0,0),'SR':(1,-1,np.pi/4.),
                'GB':(1,0,0),'FD':(1,0,0),'PT':(1,0,0),'LA':(1,0,0),'KP':(1,0,0),
                'MK':(1,0,0),'BR':(1,0,0),'NL':(1,0,0),'OV':(1,0,0),'YS':(1,0,0),'EB':(1,0,0),
                'AP':(1,1,0),'AZ':(1,1,0),'JC':(1,0,0),'SM':(1,-1,np.pi/4.),'SP':(1,0,0),'SR':(1,-1,np.pi/4.)
                }

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

# INI: copied from hops2uvfits.py
# TODO these constants should go into ___init__.py or util.py
#reference date
RDATE = '2017-04-04'
rdate_tt = Time(RDATE, format='isot', scale='utc')
RDATE_JD = rdate_tt.jd
RDATE_GSTIA0 = rdate_tt.sidereal_time('apparent','greenwich').degree
RDATE_DEGPERDY = 360.98564497330 # TODO from AIPS, get the actual value?
RDATE_OFFSET = rdate_tt.ut1.datetime.second - rdate_tt.utc.datetime.second
RDATE_OFFSET += 1.e-6*(rdate_tt.ut1.datetime.microsecond - rdate_tt.utc.datetime.microsecond)

# decimal precision for the scan start & stop times (fractional day)
ROUND_SCAN_INT = 20

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

# INI: classes copied from hops2uvfits.py to make this module self-sufficient;
# TODO These class definitions should probably be moved to a common "util.py" / "__init__.py" in this dir
class Uvfits_data(object):
    """data table and random group parameter arrays to save to uvfits"""
    def __init__(self, u, v, bls, jds, tints, datatable):
        self.u = u
        self.v = v
        self.bls = bls
        self.jds = jds
        self.tints = tints
        self.datatable = datatable

class Antenna_info(object):
    """antenna metadata """
    def __init__(self, antnames, antnums, xyz):
        self.antnames =  antnames
        self.antnums = antnums
        self.xyz = xyz

class Datastruct(object):
    """Data and metadata to save to uvfits, in uvfits format
       dtype tells you if the data table is in uvfits or ehtim format
       in ehtim format antenna_info and data are tables,
       in uvfits format they are Antenna_info and Uvfits_data objects
    """

    def __init__(self, obs_info, antenna_info, data, dtype='UVFITS'):
        self.dtype = dtype
        self.obs_info = obs_info
        self.antenna_info = antenna_info
        self.data = data

# INI: function copied from hops2uvfits.py
# TODO these should really be in the io submodule!
def save_uvfits(datastruct, fname):
    """save information already in uvfits format to uvfits file
       Args:
        datastruct (Datastruct) : a datastruct object with type 'UVFITS' for saving
        fname (str) : filename to save to
    """

    # unpack data
    if datastruct.dtype != 'UVFITS':
        raise Exception("datastruct.dtype != 'UVFITS' in save_uvfits()!")

    src = datastruct.obs_info.src
    ra = datastruct.obs_info.ra
    dec = datastruct.obs_info.dec
    ref_freq = datastruct.obs_info.ref_freq
    ch_bw = datastruct.obs_info.ch_bw
    ch_spacing = datastruct.obs_info.ch_spacing
    ch1_freq = datastruct.obs_info.ch_1
    nchan = datastruct.obs_info.nchan
    scan_arr = datastruct.obs_info.scans
    bw = nchan*ch_bw

    antnames = datastruct.antenna_info.antnames
    antnums = datastruct.antenna_info.antnums
    xyz = datastruct.antenna_info.xyz
    nsta = len(antnames)

    u = datastruct.data.u
    v = datastruct.data.v
    bls = datastruct.data.bls
    jds = datastruct.data.jds
    tints = datastruct.data.tints
    outdat = datastruct.data.datatable

    if (len(u) != len(v) != len(bls) != len(jds) != len(tints) != len(outdat)):
        raise Exception("rg parameter shapes and data shape not consistent!")

    ndat = len(u)
    mjd = int(np.min(jds) - MJD_0)
    jd_start = (MJD_0 + mjd)
    fractimes = (jds - jd_start)
    jds_only = np.ones(ndat) * jd_start

    #print "timedur uvfits " , (np.max(jds) - np.min(jds)) * 3600 * 24, (np.max(fractimes) - np.min(fractimes)) * 3600 * 24
    nsubchan = 1
    nstokes = 4

    # Create new HDU
    hdulist = fits.HDUList()
    hdulist.append(fits.GroupsHDU())

    ##################### DATA TABLE ##################################################################################################
    # Data header
    header = hdulist['PRIMARY'].header

    #mandatory
    header['OBJECT'] = src
    header['TELESCOP'] = 'VLBA' # TODO Can we change this field?
    header['INSTRUME'] = 'VLBA'
    header['OBSERVER'] = 'EHT'
    header['BSCALE'] = 1.0
    header['BZERO'] = 0.0
    header['BUNIT'] = 'JY'
    header['EQUINOX'] = 2000
    header['ALTRPIX'] = 1.e0
    header['ALTRVAL'] = 0.e0

    #optional
    header['OBSRA'] = ra * 180./12.
    header['OBSDEC'] = dec
    header['MJD'] = float(mjd)
    # new astropy broke this subfmt for jd for some reason
    # header['DATE-OBS'] = Time(mjd + MJD_0, format='jd', scale='utc', out_subfmt='date').iso
    header['DATE-OBS'] = Time(mjd + MJD_0, format='jd', scale='utc').iso[:10]
    #header['DATE-MAP'] = ??
    #header['VELREF'] = 3

    # DATA AXES #
    header['NAXIS'] = 7
    header['NAXIS1'] = 0

    # real, imag, weight
    header['CTYPE2'] = 'COMPLEX'
    header['NAXIS2'] = 3
    header['CRVAL2'] = 1.e0
    header['CDELT2'] = 1.e0
    header['CRPIX2'] = 1.e0
    header['CROTA2'] = 0.e0
    # RR, LL, RL, LR
    header['CTYPE3'] = 'STOKES'
    header['NAXIS3'] = nstokes
    header['CRVAL3'] = -1.e0 #corresponds to RR LL RL LR
    header['CDELT3'] = -1.e0
    header['CRPIX3'] = 1.e0
    header['CROTA3'] = 0.e0
    # frequencies
    header['CTYPE4'] = 'FREQ'
    header['NAXIS4'] = nsubchan
    header['CRPIX4'] = 1.e0
    # header['CRVAL4'] = ch1_freq # is this the right ref freq? in Hz
    header['CRVAL4'] = ref_freq # is this the right ref freq? in Hz
    header['CDELT4'] = ch_bw
    header['CROTA4'] = 0.e0
    # frequencies
    header['CTYPE5'] = 'IF'
    header['NAXIS5'] = nchan
    header['CRPIX5'] = 1.e0
    header['CRVAL5'] = 1.e0
    header['CDELT5'] = 1.e0
    header['CROTA5'] = 0.e0
    # RA
    header['CTYPE6'] = 'RA'
    header['NAXIS6'] = 1.e0
    header['CRPIX6'] = 1.e0
    header['CRVAL6'] = header['OBSRA']
    header['CDELT6'] = 1.e0
    header['CROTA6'] = 0.e0
    # DEC
    header['CTYPE7'] = 'DEC'
    header['NAXIS7'] = 1.e0
    header['CRPIX7'] = 1.e0
    header['CRVAL7'] = header['OBSDEC']
    header['CDELT7'] = 1.e0
    header['CROTA7'] = 0.e0

    ##RANDOM PARAMS##
    header['PTYPE1'] = 'UU---SIN'
    header['PSCAL1'] = 1/ref_freq
    header['PZERO1'] = 0.e0
    header['PTYPE2'] = 'VV---SIN'
    header['PSCAL2'] = 1.e0/ref_freq
    header['PZERO2'] = 0.e0
    header['PTYPE3'] = 'WW---SIN'
    header['PSCAL3'] = 1.e0/ref_freq
    header['PZERO3'] = 0.e0
    header['PTYPE4'] = 'BASELINE'
    header['PSCAL4'] = 1.e0
    header['PZERO4'] = 0.e0
    header['PTYPE5'] = 'DATE'
    header['PSCAL5'] = 1.e0
    header['PZERO5'] = 0.e0
    header['PTYPE6'] = 'DATE'
    header['PSCAL6'] = 1.e0
    header['PZERO6'] = 0.0
    header['PTYPE7'] = 'INTTIM'
    header['PSCAL7'] = 1.e0
    header['PZERO7'] = 0.e0
    header['history'] = "AIPS SORT ORDER='TB'"

    # Save data
    pars = ['UU---SIN', 'VV---SIN', 'WW---SIN', 'BASELINE', 'DATE', 'DATE', 'INTTIM']
    x = fits.GroupData(outdat, parnames=pars, pardata=[u, v, np.zeros(ndat), np.array(bls).reshape(-1), jds_only, fractimes, tints], bitpix=-32)

    hdulist['PRIMARY'].data = x
    hdulist['PRIMARY'].header = header

    ####################### AIPS AN TABLE ###############################################################################################
    #Antenna Table entries
    col1 = fits.Column(name='ANNAME', format='8A', array=antnames)
    col2 = fits.Column(name='STABXYZ', format='3D', unit='METERS', array=xyz)
    col3= fits.Column(name='ORBPARM', format='D', array=np.zeros(0))
    col4 = fits.Column(name='NOSTA', format='1J', array=antnums)

    #TODO get the actual information for these parameters for each station
    col5 = fits.Column(name='MNTSTA', format='1J', array=np.zeros(nsta)) #zero = alt-az
    col6 = fits.Column(name='STAXOF', format='1E', unit='METERS', array=np.zeros(nsta)) #zero = no axis  offset
    col7 = fits.Column(name='POLTYA', format='1A', array=np.array(['R' for i in range(nsta)], dtype='|S1')) #RCP
    col8 = fits.Column(name='POLAA', format='1E', unit='DEGREES', array=np.zeros(nsta)) #feed orientation A
    col9 = fits.Column(name='POLCALA', format='2E', array=np.zeros((nsta,2))) #zero = no pol cal info TODO should have extra dim for nif
    col10 = fits.Column(name='POLTYB', format='1A', array=np.array(['L' for i in range(nsta)], dtype='|S1')) #LCP
    col11 = fits.Column(name='POLAB', format='1E', unit='DEGREES', array=90*np.ones(nsta)) #feed orientation A
    col12 = fits.Column(name='POLCALB', format='2E', array=np.zeros((nsta,2))) #zero = no pol cal info

    # create table
    tbhdu = fits.BinTableHDU.from_columns(fits.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12]), name='AIPS AN')
    hdulist.append(tbhdu)

    # header information
    head = hdulist['AIPS AN'].header
    head['EXTVER'] = 1
    head['ARRAYX'] = 0.e0
    head['ARRAYY'] = 0.e0
    head['ARRAYZ'] = 0.e0

    # TODO change the reference date
    #rdate_out = RDATE
    #rdate_gstiao_out = RDATE_GSTIA0
    #rdate_offset_out = RDATE_OFFSET

    # new astropy broke this subfmt, it should be a day boundary hopefully
    # rdate_tt_new = Time(mjd + MJD_0, format='jd', scale='utc', out_subfmt='date')
    rdate_tt_new = Time(mjd + MJD_0, format='jd', scale='utc')
    rdate_out = rdate_tt_new.iso[:10]
    rdate_jd_out = rdate_tt_new.jd
    rdate_gstiao_out = rdate_tt_new.sidereal_time('apparent','greenwich').degree
    rdate_offset_out = (rdate_tt_new.ut1.datetime.second - rdate_tt_new.utc.datetime.second)
    rdate_offset_out += 1.e-6*(rdate_tt_new.ut1.datetime.microsecond - rdate_tt_new.utc.datetime.microsecond)

    head['RDATE'] = rdate_out
    head['GSTIA0'] = rdate_gstiao_out
    head['DEGPDY'] = RDATE_DEGPERDY
    head['UT1UTC'] = rdate_offset_out   #difference between UT1 and UTC ?
    head['DATUTC'] = 0.e0
    head['TIMESYS'] = 'UTC'

    head['FREQ']= ref_freq
    head['POLARX'] = 0.e0
    head['POLARY'] = 0.e0

    head['ARRNAM'] = 'VLBA'  # TODO must be recognized by aips/casa
    head['XYZHAND'] = 'RIGHT'
    head['FRAME'] = '????'
    head['NUMORB'] = 0
    head['NO_IF'] = nchan
    head['NOPCAL'] = 0  #TODO add pol cal information
    head['POLTYPE'] = 'VLBI'
    head['FREQID'] = 1

    hdulist['AIPS AN'].header = head

    ##################### AIPS FQ TABLE #####################################################################################################
    # Convert types & columns
    freqid = np.array([1])
    bandfreq = np.array([ch1_freq + ch_spacing*i - ref_freq for i in range(nchan)]).reshape([1,nchan])
    chwidth = np.array([ch_bw for i in range(nchan)]).reshape([1,nchan])
    totbw = np.array([ch_bw for i in range(nchan)]).reshape([1,nchan])
    sideband = np.array([1 for i in range(nchan)]).reshape([1,nchan])

    freqid = fits.Column(name="FRQSEL", format="1J", array=freqid)
    bandfreq = fits.Column(name="IF FREQ", format="%dD"%(nchan), array=bandfreq, unit='HZ')
    chwidth = fits.Column(name="CH WIDTH",format="%dE"%(nchan), array=chwidth, unit='HZ')
    totbw = fits.Column(name="TOTAL BANDWIDTH",format="%dE"%(nchan),array=totbw, unit='HZ')
    sideband = fits.Column(name="SIDEBAND",format="%dJ"%(nchan),array=sideband)
    cols = fits.ColDefs([freqid, bandfreq, chwidth, totbw, sideband])

    # create table
    tbhdu = fits.BinTableHDU.from_columns(cols)

    # header information
    tbhdu.header.append(("NO_IF", nchan, "Number IFs"))
    tbhdu.header.append(("EXTNAME","AIPS FQ"))
    tbhdu.header.append(("EXTVER",1))
    hdulist.append(tbhdu)

    ##################### AIPS NX TABLE #####################################################################################################

    scan_times = []
    scan_time_ints = []
    start_vis = []
    stop_vis = []

    #TODO make sure jds AND scan_info MUST be time sorted!!
    jj = 0
    #print scan_info

    comp_fac = 3600*24*100 # compare to 100th of a second

    for scan in  scan_arr:
        scan_start = round(scan[0], ROUND_SCAN_INT)
        scan_stop = round(scan[1], ROUND_SCAN_INT)
        scan_dur = (scan_stop - scan_start)

        if jj>=len(jds):
            #print start_vis, stop_vis
            break

        # print "%.12f %.12f %.12f" %( jds[jj], scan_start, scan_stop)
        jd = round(jds[jj], ROUND_SCAN_INT)*comp_fac # ANDREW TODO precision??

        if (np.floor(jd) >= np.floor(scan_start*comp_fac)) and (np.ceil(jd) <= np.ceil(comp_fac*scan_stop)):
            start_vis.append(jj)
            # TODO AIPS MEMO 117 says scan_times should be midpoint!, but AIPS data looks likes it's at the start?
            #scan_times.append(scan_start  - rdate_jd_out)
            scan_times.append(scan_start + 0.5*scan_dur - rdate_jd_out)
            scan_time_ints.append(scan_dur)
            while (jj < len(jds) and np.floor(round(jds[jj],ROUND_SCAN_INT)*comp_fac) <= np.ceil(comp_fac*scan_stop)):
                jj += 1
            stop_vis.append(jj-1)
        else:
            continue

    if jj < len(jds):
        if len(scan_arr) == 0:
            print("len(scan_arr) == 0")
        else:
            print(scan_arr[-1])
            print(round(scan_arr[-1][0],ROUND_SCAN_INT),round(scan_arr[-1][1],ROUND_SCAN_INT))
        print(jj, len(jds), round(jds[jj], ROUND_SCAN_INT))
        print("WARNING!!!: in save_uvfits NX table, didn't get to all entries when computing scan start/stop!")
        #raise Exception("in save_uvfits NX table, didn't get to all entries when computing scan start/stop!")

    time_nx = fits.Column(name="TIME", format="1D", unit='DAYS', array=np.array(scan_times))
    timeint_nx = fits.Column(name="TIME INTERVAL", format="1E", unit='DAYS', array=np.array(scan_time_ints))
    sourceid_nx = fits.Column(name="SOURCE ID",format="1J", unit='', array=np.ones(len(scan_times)))
    subarr_nx = fits.Column(name="SUBARRAY",format="1J", unit='', array=np.ones(len(scan_times)))
    freqid_nx = fits.Column(name="FREQ ID",format="1J", unit='', array=np.ones(len(scan_times)))
    startvis_nx = fits.Column(name="START VIS",format="1J", unit='', array=np.array(start_vis)+1)
    endvis_nx = fits.Column(name="END VIS",format="1J", unit='', array=np.array(stop_vis)+1)
    cols = fits.ColDefs([time_nx, timeint_nx, sourceid_nx, subarr_nx, freqid_nx, startvis_nx, endvis_nx])

    tbhdu = fits.BinTableHDU.from_columns(cols)

    # header information
    tbhdu.header.append(("EXTNAME","AIPS NX"))
    tbhdu.header.append(("EXTVER",1))

    hdulist.append(tbhdu)

    # Write final HDUList to file
    #hdulist.writeto(fname, clobber=True)#this is deprecated and changed to overwrite
    hdulist.writeto(fname, overwrite=True)

    return 0

def load_caltable_ds(datastruct, tabledir, sqrt_gains=False, skip_fluxcal=False):
    """Load apriori cal tables
    """

    # load metadata from the UVFITS file pre-converted to EHTIM-type datastruct
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

        site = tarr[s]['site'].decode() # INI: bytes to str to avoid type errors downstream

        if skip_fluxcal: #mocking SEFDs equal to 1.0 spread across the day

            datatable = []
            for time in np.linspace(0.,24.,100):
                datatable.append(np.array((time, 1.0, 1.0), dtype=DTCAL))

        else: # getting SEFDS from files
            # AIPS can only handle 8-character source name so "source" may
            # be truncated.  Therefore, we match a pattern (note the "*")
            # and proceed only if there is one match
            pattern = os.path.join(tabledir, f'{source}*_{site}.txt')
            filenames = glob.glob(pattern)
            if len(filenames) == 1:
                try:
                    data = np.loadtxt(filenames[0], dtype=bytes).astype(str)
                except IOError:
                    print(f'Skipping corrupted file: {filenames[0]}')
                    continue

                filename_source = filenames[0].replace(tabledir+'/', '').replace(f'_{site}.txt', '')
                if source != filename_source:
                    print('WARNING: name of source in filename is different from the one in the EHTIM datastruct')
                    if filename_source.startswith(source):
                        print(f'which is probably due to AIPS source name truncation; using the full name {filename_source} from the filename...')
                        source = filename_source
            elif len(filenames) == 0:
                print(f'No file matching {pattern} exists! Skipping...')
                continue
            else:
                print(f'More than one file matching pattern {pattern}. Skipping...')
                continue

            datatable = []

            # ANDREW HACKY WAY TO MAKE IT WORK WITH ONLY ONE ENTRY
            onerowonly=False
            if data.ndim == 1:
                data = np.expand_dims(data, axis=0)
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

    if (len(datatables)<=0)&(skip_fluxcal==False):#only if no SEFD files available and we don't want just field rotation
        caltable=False
    else: #other cases, either we want flux and we do have SEFDs, or we want to skip fluxcal
        caltable = Caltable(ra, dec, rf, bw, datatables, tarr, source=source, mjd=mjd, timetype='UTC')
    return caltable

def xyz_2_latlong(obsvecs):
    """Compute the (geocentric) latitude and longitude of a site at geocentric position x,y,z
       The output is in radians
    """
    if len(obsvecs.shape)==1:
        obsvecs=np.array([obsvecs])
    out = []
    for obsvec in obsvecs:
        x = obsvec[0]
        y = obsvec[1]
        z = obsvec[2]
        lon = np.array(np.arctan2(y,x))
        lat = np.array(np.arctan2(z, np.sqrt(x**2+y**2)))
        out.append([lat,lon])
    out = np.array(out)
    #if out.shape[0]==1: out = out[0]
    return out

def apply_caltable_uvfits(caltable, datastruct, filename_out, interp='linear', extrapolate=True, frotcal=True, elev_function='astropy', interp_dt=1., \
        elev_interp_kind='cubic', err_scale=1., skip_fluxcal=False, keep_absolute_phase=True, station_frot=STATION_FROT):
    """apply a calibration table to a uvfits file
       Args:
        caltable (Caltable) : a caltable object
        datastruct (Datastruct) :  input data structure in EHTIM format
        filename_out (str) :  uvfits output file name
        interp (str) : kind of interpolation to perform for gain tables
        extrapolate (bool) : toggle extrapolation for gain tables and elevation computation
        frotcal (bool): whether apply field rotation angle correction
        elev_function (str): 'astropy' or 'ehtim' for calculating elevation
        interp_dt (float) : time resolution for interpolation
        elev_interp_kind (str) : kind of interpolation to perform for elevation computation
        err_scale (float) : scaling factor for error
        skip_fluxcal (bool): toggle whether SEFDs should be applied (i.e. flux calibration)
        keep_absolute_phase (bool): toggle whether absolute phase of LL* visibilities should be kept
    """

    if datastruct.dtype != "EHTIM":
        raise Exception("datastruct must be in EHTIM format in apply_caltable_uvfits!")

    if not (caltable.tarr == datastruct.antenna_info).all():
        raise Exception("The telescope array in the Caltable is not the same as in the Datastruct")

    # interpolate the calibration table
    rinterp = {}
    linterp = {}
    skipsites = []

    #PREPARE INTERPOLATION DATA
    xyz={}
    latitude={}
    longitude={}
    ra = caltable.ra*np.pi*2./24.#rad
    dec = caltable.dec*np.pi*2./360.#rad
    sourcevec = np.array([np.cos(dec), 0, np.sin(dec)])
    PAR={}
    ELE={}
    OFF={}
    elevfit={}
    gmst_function= lambda time_mjd: Time(time_mjd, format='mjd').sidereal_time('mean','greenwich').hour*2.*np.pi/24.

    if not station_frot:
        logging.warning("Empty dict passed for station mount information! Using default values...")
        station_frot = STATION_FROT

    #FIND MAX RANGE OF MJD TIMES FOR INTERPOLATION
    if (frotcal==True)&(interp_dt>0):
        dt_mjd = interp_dt*1./24./60./60. #interp_dt in sec
        mjd_max=-1
        mjd_min=1e10
        for s in range(0, len(caltable.tarr)):
            site = caltable.tarr[s]['site'].decode() # INI: bytes to str
            try:
                #sometimes station reported but no calibration
                time_mjd = caltable.data[site]['time']/24.0 + caltable.mjd
                mjd_max_foo = np.max(time_mjd)
                mjd_min_foo = np.min(time_mjd)
                if (mjd_max_foo > mjd_max):
                    mjd_max = mjd_max_foo
                if (mjd_min_foo < mjd_min):
                    mjd_min = mjd_min_foo
            except KeyError: continue
        #MAKE TIME GRIDS FOR INTERPOLATION
        time_mjd_fake = np.arange(mjd_min,mjd_max,dt_mjd)
        gmst_fake = gmst_function(time_mjd_fake)
        datetimes_fake = Time(time_mjd_fake, format='mjd').to_datetime()
        strtime_fake = [str(round_time(x)) for x in datetimes_fake]
        thetas_fake = np.mod((gmst_fake - ra), 2.*np.pi)

    for s in range(0, len(caltable.tarr)):
        site = caltable.tarr[s]['site'].decode() # INI: bytes to str
        xyz_foo = np.asarray((caltable.tarr[s]['x'],caltable.tarr[s]['y'],caltable.tarr[s]['z']))
        xyz[site] = xyz_foo
        latlong = xyz_2_latlong(xyz_foo)
        latitude[site] = latlong[0][0]#rad
        longitude[site] = latlong[0][1]#rad
        PAR[site] = station_frot[site][0]
        ELE[site] = station_frot[site][1]
        OFF[site] = station_frot[site][2]

        # This is only if we interpolate elevation
        if (frotcal==True)&(interp_dt>0):
            if elev_function=='ehtim':
                elev_fake_foo = get_elev_2(earthrot(xyz[site], thetas_fake), sourcevec)#ehtim
            else:
                elev_fake_foo = get_elev(ra, dec, xyz[site], strtime_fake)##astropy

            # INI: extrapolate elevation to values outside the range; NOT IN OLDER VERSIONS OF AMPLITUDE CALIBRATION SCRIPT
            if extrapolate:
                elevfit[site] = scipy.interpolate.interp1d(time_mjd_fake, elev_fake_foo, kind=elev_interp_kind, fill_value='extrapolate')
            else:
                elevfit[site] = scipy.interpolate.interp1d(time_mjd_fake, elev_fake_foo, kind=elev_interp_kind)

        try:
            caltable.data[site]
        except KeyError:
            skipsites.append(site)
            print ("No Calibration  Data for %s !" % site)
            continue

        if skip_fluxcal: #if we don't do flux calibration don't waste time on serious interpolating
            rinterp[site] = scipy.interpolate.interp1d([0],[1],kind='zero',fill_value='extrapolate')
            linterp[site] = scipy.interpolate.interp1d([0],[1],kind='zero',fill_value='extrapolate')

        else: #default option, create interpolating station based SEFD gains
            time_mjd = caltable.data[site]['time']/24.0 + caltable.mjd
           
            if extrapolate:
                rinterp[site] = scipy.interpolate.interp1d(time_mjd, caltable.data[site]['rscale'], kind=interp, fill_value='extrapolate')
                linterp[site] = scipy.interpolate.interp1d(time_mjd, caltable.data[site]['lscale'], kind=interp, fill_value='extrapolate')
            else:
                rinterp[site] = scipy.interpolate.interp1d(time_mjd, caltable.data[site]['rscale'], kind=interp)
                linterp[site] = scipy.interpolate.interp1d(time_mjd, caltable.data[site]['lscale'], kind=interp)


    #-------------------------------------------
    # sort by baseline
    data =  datastruct.data
    idx = np.lexsort((data['t2'], data['t1']))
    bllist = []
    for key, group in it.groupby(data[idx], lambda x: set((x['t1'], x['t2'])) ):
        bllist.append(np.array([obs for obs in group]))

    # apply the calibration

    datatable = []
    coub=0
    for bl_obs in bllist:
        t1 = bl_obs['t1'][0].decode() # INI: bytes to str
        t2 = bl_obs['t2'][0].decode()
        coub=coub+1
        print('Calibrating {}-{} baseline, {}/{}'.format(t1,t2,coub,len(bllist)))
        time_mjd = bl_obs['time'] - MJD_0 #dates are in mjd in Datastruct
        if frotcal==True:
            gmst = gmst_function(time_mjd)
            thetas = np.mod((gmst - ra), 2*np.pi)
            hangle1 = gmst + longitude[t1] - ra #HOUR ANGLE FIRST TELESCOPE
            hangle2 = gmst + longitude[t2] - ra #HOUR ANGLE SECOND TELESCOPE
            par1I_t1 = np.sin(hangle1)
            par1I_t2 = np.sin(hangle2)
            par1R_t1 = np.cos(dec)*np.tan(latitude[t1]) - np.sin(dec)*np.cos(hangle1)
            par1R_t2 = np.cos(dec)*np.tan(latitude[t2]) - np.sin(dec)*np.cos(hangle2)
            parangle1 = np.angle(par1R_t1 + 1j*par1I_t1 ) #PARALACTIC ANGLE T1
            parangle2 = np.angle(par1R_t2 + 1j*par1I_t2 ) #PARALACTIC ANGLE T2
            if interp_dt<=0:
                if elev_function=='ehtim':
                    elev1 = get_elev_2(earthrot(xyz[t1], thetas), sourcevec)
                    elev2 = get_elev_2(earthrot(xyz[t2], thetas), sourcevec)
                else:
                    datetimes = Time(time_mjd, format='mjd').to_datetime()
                    strtime = [str(round_time(x)) for x in datetimes]
                    elev1 = get_elev(ra, dec, xyz[t1], strtime) #ELEVATION T1
                    elev2 = get_elev(ra, dec, xyz[t2], strtime) #ELEVATION T2
            else:
                elev1 = elevfit[t1](time_mjd)
                elev2 = elevfit[t2](time_mjd)

            fran1 = PAR[t1]*parangle1 + ELE[t1]*elev1 + OFF[t1]
            fran2 = PAR[t2]*parangle2 + ELE[t2]*elev2 + OFF[t2]
            
            #Keeping absolute phase of the LL* visibilities
            if keep_absolute_phase:
                shift1 = 1j*fran1
                shift2 = 1j*fran2
                fran_R1 = np.exp(1j*fran1 + shift1)
                fran_L1 = np.exp(-1j*fran1 + shift1)
                fran_R2 = np.exp(1j*fran2 + shift2)
                fran_L2 = np.exp(-1j*fran2 + shift2)
            else:
                fran_R1 = np.exp(1j*fran1)
                fran_L1 = np.exp(-1j*fran1)
                fran_R2 = np.exp(1j*fran2)
                fran_L2 = np.exp(-1j*fran2)
            

        if t1 in skipsites:
            rscale1 = lscale1 = np.array(1.)
        else:
            if frotcal==False:
                rscale1 = rinterp[t1](time_mjd)
                lscale1 = linterp[t1](time_mjd)
            else:
                rscale1 = rinterp[t1](time_mjd)*fran_R1
                lscale1 = linterp[t1](time_mjd)*fran_L1
        if t2 in skipsites:
            rscale2 = lscale2 = np.array(1.)
        else:
            if frotcal==False:
                rscale2 = rinterp[t2](time_mjd)
                lscale2 = linterp[t2](time_mjd)
            else:
                rscale2 = rinterp[t2](time_mjd)*fran_R2
                lscale2 = linterp[t2](time_mjd)*fran_L2


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
    bl_arr = np.empty((len(datatable)), dtype=BLTYPE)
    for i in range(len(datatable)):
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
        bl_arr[i] = np.array((entry['time'],entry['t1'],entry['t2']),dtype=BLTYPE)
    _, unique_idx_anttime, idx_anttime = np.unique(bl_arr, return_index=True, return_inverse=True)
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
    for i in range(len(datatable)):
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

def get_elev(ra_source, dec_source, xyz_antenna, time):
    #this one is by Michael Janssen
    """
    given right ascension and declination of a sky source [ICRS: ra->(deg,arcmin,arcsec) and dec->(hour,min,sec)]
    and given the position of the telescope from the vex file [Geocentric coordinates (m)]
    and the time of the observation (e.g. '2012-7-13 23:00:00') [UTC:yr-m-d],
    returns the elevation of the telescope.
    Note that every parameter can be an array (e.g. the time)
    """
    from astropy import units as u
    from astropy.coordinates import EarthLocation, AltAz, ICRS, Angle
    #angle conversions:
    ra_src      = Angle(ra_source, unit=u.rad)
    dec_src      = Angle(dec_source, unit=u.rad)

    source_position  = ICRS(ra=ra_src, dec=dec_src)
    antenna_position = EarthLocation(x=xyz_antenna[0]*u.m, y=xyz_antenna[1]*u.m, z=xyz_antenna[2]*u.m)
    altaz_system     = AltAz(location=antenna_position, obstime=time)
    trans_to_altaz   = source_position.transform_to(altaz_system)
    elevation        = trans_to_altaz.alt
    return elevation.rad

def round_time(t,round_s=1.):

    """rounding time to given accuracy

    Args:
        t: time
        round_s: delta time to round to in seconds

    Returns:
        round_t: rounded time
    """
    t0 = datetime.datetime(t.year,1,1)
    foo = t - t0
    foo_s = foo.days*24*3600 + foo.seconds + foo.microseconds*(1e-6)
    foo_s = np.round(foo_s/round_s)*round_s
    days = np.floor(foo_s/24/3600)
    seconds = np.floor(foo_s - 24*3600*days)
    microseconds = int(1e6*(foo_s - days*3600*24 - seconds))
    round_t = t0+datetime.timedelta(days,seconds,microseconds)
    return round_t


def earthrot(vecs, thetas):
    """Rotate a vector / array of vectors about the z-direction by theta / array of thetas (radian)
    """
    if len(vecs.shape)==1:
        vecs = np.array([vecs])
    if np.isscalar(thetas):
        thetas = np.array([thetas for i in range(len(vecs))])

    # equal numbers of sites and angles
    if len(thetas) == len(vecs):
        rotvec = np.array([np.dot(np.array(((np.cos(thetas[i]),-np.sin(thetas[i]),0),(np.sin(thetas[i]),np.cos(thetas[i]),0),(0,0,1))), vecs[i])
                       for i in range(len(vecs))])

    # only one rotation angle, many sites
    elif len(thetas) == 1:
        rotvec = np.array([np.dot(np.array(((np.cos(thetas[0]),-np.sin(thetas[0]),0),(np.sin(thetas[0]),np.cos(thetas[0]),0),(0,0,1))), vecs[i])
                       for i in range(len(vecs))])
    # only one site, many angles
    elif len(vecs) == 1:
        rotvec = np.array([np.dot(np.array(((np.cos(thetas[i]),-np.sin(thetas[i]),0),(np.sin(thetas[i]),np.cos(thetas[i]),0),(0,0,1))), vecs[0])
                       for i in range(len(thetas))])
    else:
        raise Exception("Unequal numbers of vectors and angles in earthrot(vecs, thetas)!")

    return rotvec

def get_elev_2(obsvecs, sourcevec):
    """Return the elevation of a source with respect to an observer/observers in radians
       obsvec can be an array of vectors but sourcevec can ONLY be a single vector
    """

    if len(obsvecs.shape)==1:
        obsvecs=np.array([obsvecs])

    anglebtw = np.array([np.dot(obsvec,sourcevec)/np.linalg.norm(obsvec)/np.linalg.norm(sourcevec) for obsvec in obsvecs])
    el = 0.5*np.pi - np.arccos(anglebtw)

    return el