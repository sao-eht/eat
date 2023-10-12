#!/usr/bin/env python

# INI: Adapted from previous years' post-processing calibration scripts
# by Andrew, Lindy, Maciek, Kevin, Katie, and others
# This script performs a priori flux calibration and field angle rotation correction

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
import argparse

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

station_frot = {'PV':(1,-1,0),'MG':(1,1,0),'SW':(1,-1,np.pi/4.),'LM': (1,-1,0),
                'AA':(1,0,0),'SZ':(1,0,0),'AX':(1,1,0),'MM':(1,0,0),'GL': (1,0,0),
                'NN':(1,0,0),'KT':(1,0,0),'SR':(1,-1,np.pi/4.),
                'GB':(1,0,0),'FD':(1,0,0),'PT':(1,0,0),'LA':(1,0,0),'KP':(1,0,0),
                'MK':(1,0,0),'BR':(1,0,0),'NL':(1,0,0),'OV':(1,0,0),'YS':(1,0,0),'EB':(1,0,0)}

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


def load_caltable_ds(datastruct, tabledir, sqrt_gains=False, skip_fluxcal=False):
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
        elev_interp_kind='cubic', err_scale=1., skip_fluxcal=False, keep_absolute_phase=True):
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

            # INI: extrapolate elevation to values outside the range
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
    bllist = np.array(bllist, dtype=object) # INI: avoid VisibleDeprecationWarning

    # apply the  calibration

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
    bl_list = []
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

def create_parser():
    p = argparse.ArgumentParser()

    p.add_argument("datadir", help="Directory containing input UVFITS files")
    p.add_argument("caldir", help="Directory containing input calibration tables")
    p.add_argument("outdir", help="Directory to which calibrated UVFITS files must be written")
    p.add_argument('--identifier', type=str, default='', help="Identifier tag to suffix uvfits filenames with (apart from the automatic identifiers introduced by this script)")
    p.add_argument('--interpkind', type=str, default='linear', help="Kind of interpolation to perform (scipy-compatible)")
    p.add_argument('--extrapolate', action='store_true', help='Toggle whether to extrapolate gain tables')
    p.add_argument('--sqrtgains', action='store_true', help='Toggle whether to take square root of gains before applying')
    p.add_argument('--skipfluxcal', action='store_true', help='Toggle whether to perform a priori flux calibration')
    p.add_argument('--skipfrotcorr', action='store_true', help='Toggle whether to perform field angle rotation correction')
    p.add_argument('--keepllabsphase', action='store_true', help='Toggle whether to keep absolute phase of LL* (i.e. do not rotate)')
    p.add_argument('--elevmodule', type=str, default='astropy', choices=['astropy', 'ehtim'], help="Python module to use to compute elevation")
    p.add_argument('--elevinterpkind', type=str, default='cubic', help="kind of interpolation to perform (scipy-compatible)")
    p.add_argument('--interpolatedt', type=float, default=1., help="Interpolation resolution for integration time dt in seconds")
    p.add_argument('--errorscale', type=float, default=1., help="Error scaling factor")

    return p

def main(args):
    print('**************** Post-processing HOPS results ****************')
    print(f"Applying calibration tables from {args.caldir}\nto uvfits files in {args.datadir}\nand writing results to {args.outdir}")

    if args.skipfluxcal: print('WARNING:: Will skip a priori flux calibration')
    else: print('INFO:: Will perform a priori flux calibration')
    if args.skipfrotcorr: print('WARNING:: Will skip field angle rotation correction')
    else: print('INFO:: Will perform field angle rotation correction')

    uvfitsfiles = sorted(glob.glob(os.path.join(args.datadir, '*.uvfits')))
    print(f'List of files: {uvfitsfiles}')

    # exclude previously averaged uvfits files
    excludepattern = "+avg.uvfits"
    '''reslist = []
    for uvf in uvfitsfiles:
        if excludepattern not in uvf:
            res.append(uvf)
    uvfitsfiles = reslist'''
    uvfitsfiles = [uvf for uvf in uvfitsfiles if excludepattern not in uvf]
    print(f'List of files: {uvfitsfiles}')

    # For each uvfitsfile, perform requested calibration steps and write output to a new uvfits file
    for uvfitsfile in uvfitsfiles:
        print(f"A priori calibrating: {uvfitsfile}")

        tok = uvfitsfile.split('/')[-1].replace('.uvfits', '').split('_', 2)
        print(f"cal pipeline: {tok[0]}")
        print(f"expt no: {tok[1]}")
        print(f"source: {tok[2]}")

        # convert uvfits to ehtim data structure
        datastruct_ehtim = load_and_convert_hops_uvfits(uvfitsfile)

        # ensure source names are consistent between ehtim data structure and the input uvfits filenames
        source = datastruct_ehtim.obs_info.src
        if len(source) <= 8 and source != tok[2]:
            print(f"WARNING:: Source name {source} inside the uvfits file does not match source name {tok[2]} in the filename! Using name {tok[2]} from filename instead...")
            source = tok[2]
            datastruct_ehtim.obs_info.src = tok[2]

        caltable = load_caltable_ds(datastruct_ehtim, args.caldir, sqrt_gains=args.sqrtgains, skip_fluxcal=args.skipfluxcal)
        if not caltable:
            print(f'Could not find caltable in {args.caldir} for {source}! Skipping {uvfitsfile}')
            continue

        outname = os.path.join(args.outdir, os.path.basename(uvfitsfile).replace('.uvfits', args.identifier+'.apriori.uvfits'))

        apply_caltable_uvfits(caltable, datastruct_ehtim, outname, interp=args.interpkind, extrapolate=args.extrapolate, frotcal=not(args.skipfrotcorr), elev_function=args.elevmodule,
                interp_dt=args.interpolatedt, elev_interp_kind=args.elevinterpkind, err_scale=args.errorscale, skip_fluxcal=args.skipfluxcal, keep_absolute_phase=args.keepllabsphase)

        print(f'Saved calibrated data to {outname}')
    
    return 0

if __name__=='__main__':
    args = create_parser().parse_args()
    ret = main(args)
    sys.exit(ret)
