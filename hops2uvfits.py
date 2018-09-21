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
try:
    import eat.hops.util
    from eat.io import util
    #from eat.plots import util as putil
except: #work on Maciek's laptop
    sys.path.append('/Users/mwielgus/Dropbox (Smithsonian External)/EHT/Data/MasterEAT/eat/')
    import eat.hops.util
    from eat.io import util
    #from eat.plots import util as putil

from astropy.time import Time
import numpy.matlib

# For Andrew:
#DATADIR_DEFAULT = '/home/achael/EHT/hops/data/3554/' #/098-0924/'
DATADIR_DEFAULT = '/Users/klbouman/Research/vlbi_imaging/software/hops/er1-hops-hi/5.+close/data/3601'
OUTDIR_DEFAULT = '/Users/klbouman/Research/vlbi_imaging/software/hops/tmpout2'


# For Katie
#DATADIR_DEFAULT = '/Users/klbouman/Downloads/newscans/apr2017s/3601' #3600' #'/Users/klbouman/Downloads/apr2017s/3597' #3598_orig' #'../3554/'# /098-0916/'
# source hops.bash in /Users/klbouman/Research/vlbi_imaging/software/hops/build
# run this from /Users/klbouman/Research/vlbi_imaging/software/hops/eat

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

#conversion factors and data types
BLTYPE = [('time','f8'),('t1','a32'),('t2','a32')]
DTARR = [('site', 'a32'), ('x','f8'), ('y','f8'), ('z','f8')]
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

#######################################################################
##########################  Recompute uv points #######################
#######################################################################
# ANDREW these are copied from ehtim -- maybe integrate them?
def gmst_to_utc(gmst,mjd):
    """Convert gmst times in hours to utc hours using astropy
    """

    mjd=int(mjd)
    time_obj_ref = at.Time(mjd, format='mjd', scale='utc')
    time_sidereal_ref = time_obj.sidereal_time('mean', 'greenwich').hour
    time_utc = (gmst - time_sidereal_ref) * 0.9972695601848
    return time_utc

def utc_to_gmst(utc, mjd):
    """Convert utc times in hours to gmst using astropy
    """
    mjd=int(mjd) #MJD should always be an integer, but was float in older versions of the code
    time_obj = at.Time(utc/24.0 + np.floor(mjd), format='mjd', scale='utc')
    time_sidereal = time_obj.sidereal_time('mean','greenwich').hour
    return time_sidereal

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

def recompute_uv_points(site1vec, site2vec, time, ra, dec, mjd, rf, timetype='UTC'):
    """recompute uv points for a given observation

       Args:
        site1vec (np.ndarray) :  a single earth frame xyz vector of the site1 position or an array of vectors of len(times)
        site2vec (np.ndarray) :  a single earth frame xyz vector of the site2 position or an array of vectors of len(times)
        time (np.ndarray) :  an array of UTC hours for which  to recompute uv points
        ra (float) :  source right ascension
        dec (float) : source declination
        mjd (int) : the  mjd  of the observation  start
        rf  (float) : the observing frequency in  Hz
        timetype (str) : 'UTC' or 'GMST'
        elevmin (float) : minimum elevation cutoff in degrees
        elevmax (float) : maximum elevation cutoff in degrees
       Returns:
        (u, v) : numpy arrays of scan times and corresponding uv points
    """

    if not isinstance(time, np.ndarray): time = np.array([time]).flatten()
    if not isinstance(site1vec, np.ndarray): site1vec = np.array([site1vec])
    if not isinstance(site2vec, np.ndarray): site2vec = np.array([site2vec])

    try:
        if not site1vec.shape[1] == 3: raise Exception("site1vec does not have 3 coords!")
    except IndexError:
        site1vec = site1vec.reshape(1,3)
    try:
        if not site2vec.shape[1] == 3: raise Exception("site2vec does not have 3 coords!")
    except IndexError:
        site2vec = site2vec.reshape(1,3)

    if len(site1vec) == len(site2vec) == 1:
        site1vec = np.array([site1vec[0] for i in range(len(time))])
        site2vec = np.array([site2vec[0] for i in range(len(time))])
    elif not (len(site1vec) == len(site2vec) == len(time)):
        raise Exception("site1vec, site2vec, and time not the same dimension in compute_uv_coordinates!")

    # Source vector
    sourcevec = np.array([np.cos(dec*DEGREE), 0, np.sin(dec*DEGREE)])
    projU = np.cross(np.array([0,0,1]), sourcevec)
    projU = projU/np.linalg.norm(projU)
    projV = -np.cross(projU, sourcevec)

    # Wavelength
    l = C/rf

    if timetype=='GMST':
        time_sidereal = time
        time_utc = gmst_to_utc(time, mjd)
    elif timetype=='UTC':
        time_sidereal = utc_to_gmst(time, mjd)
        time_utc = time
    else: raise Exception("timetype must be UTC or GMST!")

    fracmjd = np.floor(mjd) + time/24.
    dto = (at.Time(fracmjd, format='mjd')).datetime
    theta = np.mod((time_sidereal - ra)*HOUR, 2*np.pi)

    # rotate the station coordinates with the earth
    coord1 = earthrot(site1vec, theta)
    coord2 = earthrot(site2vec, theta)

    # u,v coordinates
    u = np.dot((coord1 - coord2)/l, projU) # u (lambda)
    v = np.dot((coord1 - coord2)/l, projV) # v (lambda)

    # return times and uv points where we have  data
    return (u, v)

#######################################################################
#######################  Data/Metadata  OBJECTS #######################
#######################################################################
#TODO consistency checks for all these classes!
class Uvfits_data(object):
    """data table and random group parameter arrays to save to uvfits"""
    def __init__(self, u, v, bls, jds, tints, datatable):
        self.u = u
        self.v = v
        self.bls = bls
        self.jds = jds
        self.tints = tints
        self.datatable = datatable

class Obs_info(object):
    """observing metadata """
    def __init__(self, src, ra, dec, ref_freq, ch_bw, ch_spacing, ch_1, nchan, scan_array):
        self.src =  src
        self.ra = ra
        self.dec = dec
        self.ref_freq = ref_freq
        self.ch_bw = ch_bw
        self.ch_spacing = ch_spacing
        self.ch_1 = ch_1
        self.nchan = nchan
        self.scans = scan_array

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

#######################################################################
##########################  Load/Save FUNCTIONS #######################
#######################################################################
def convert_bl_fringefiles(datadir=DATADIR_DEFAULT, rot_rate=False, rot_delay=False, recompute_uv=False,
                           sqrt2corr=False, flip_ALMA_pol=False, flip_SPT_pol=False):
    """read all fringe files in a directory and produce single baseline uvfits files

       Args:
        datadir (str) :  directory with fringe files
        rot_rate (bool) : True to apply fringe rate correction to get total phase
        rot_delay (bool) : True to apply fringe delay rate correction to get total phase
        recompute_uv (bool): True to recompute uv points over the track

    """

    baselineNames = []
    for filename in glob.glob(datadir + '*'):
        # remove type 1 and 3 files
        #if ".." not in filename:
        if os.path.basename(filename).count('.')==3:
            baselineNames.append(os.path.basename(filename).split(os.extsep)[0])

    baselineNames = set(baselineNames)

    ###################################  LOAD DATA ###################################
    for baselineName in baselineNames:
        nap = 0
        nchan = 0
        nsubchan = 1
        nstokes= 4
        first_pass_flag = True

        #TODO will currently create an empty uvfits file for every non hops file not caught here
        #TODO make eat throw errors if its reading something that's not a fringe file
        if baselineName.split("_")[-1] == "baseline":
            continue
        if baselineName.split("_")[-1] == "merged":
            continue
        try:
            if baselineName.split("_")[-2] == "hops":
                continue
        except IndexError: pass

        # remove type 3 files containing model spline coefficients, phasecal data, state count information, and tape error statistics
        #if len(baselineName)==1:
        #    continue

        # remove auto correlations
        if baselineName[0] == baselineName[1]:
            continue

        print("Making uvfits for baseline: ", baselineName)
        for filename in glob.glob(datadir + baselineName + '*'):

            # remove type 1 and 2 files and uvfits files with the same basename
            if filename.split(os.extsep)[-1] == "uvfits" or os.path.basename(filename).count('.')!=3:
                continue

            #print "reading hops fringe file: ", filename
            a = mk4.mk4fringe(filename)
            b = eat.hops.util.getfringefile(a)

            if first_pass_flag: #some info we get only once per baseline

                ###################################  SOURCE INFO ###################################
                # name of the source
                srcname = b.t201[0].source

                # get the ra
                ra_hrs = b.t201[0].coord.ra_hrs
                ra_min = b.t201[0].coord.ra_mins
                ra_sec = b.t201[0].coord.ra_secs

                # get the dec
                dec_degs = b.t201[0].coord.dec_degs
                dec_mins = b.t201[0].coord.dec_mins
                dec_secs = b.t201[0].coord.dec_secs

                ra =  ra_hrs+ra_min/60.+ra_sec/3600.
                dec = np.sign(dec_degs)*(np.abs(dec_degs)+(dec_mins/60.+dec_secs/3600.))

                ################################### OBSERVATION INFO ###################################
                ref_freq_hops = b.t205.contents.ref_freq * MHZ2HZ # refrence frequency for all channels
                nchan = b.n212 # number of channels
                nap = b.t212[0].contents.nap # number of data points

                # get the fixed integration time
                totaltime  = (eat.hops.util.mk4time(b.t205.contents.stop) - eat.hops.util.mk4time(b.t205.contents.start)).total_seconds()
                inttime_fixed = totaltime/nap

                ###################################  ANTENNA INFO ###################################
                # names of antennas
                ant1 = b.t202.contents.ref_intl_id.upper()
                ant2 = b.t202.contents.rem_intl_id.upper()

                scalingFac = 1.0
                if sqrt2corr:
                    if ant1 == 'AA' and ant2 == 'AA':
                        scalingFac = 1.0
                    elif ant1== 'AA' or ant2 == 'AA':
                        scalingFac = 1.0 / np.sqrt(2)

                baselineName = b.t202.contents.baseline # first one is ref antenna second one is rem

                # x, y, z coordinate of antenna 1
                ant1_x = b.t202.contents.ref_xpos # meters
                ant1_y = b.t202.contents.ref_ypos # meters
                ant1_z = b.t202.contents.ref_zpos # meters

                # x, y, z coordinate of antenna 2
                ant2_x = b.t202.contents.rem_xpos # meters
                ant2_y = b.t202.contents.rem_ypos # meters
                ant2_z = b.t202.contents.rem_zpos # meters

                # get the elevation for opacity estimate
                ant1_elevation = b.t202.contents.ref_elev # degrees
                ant2_elevation = b.t202.contents.rem_elev # degrees

                antennas = {} # WARNING: DO NOT REINITLIIZE THIS IF WE LOOP OVER FILES
                xyz = []
                antnames = []
                antnums = []

                if ant1 not in antennas.keys():
                    antennas[ant1] = len(antennas.keys()) + 1
                    xyz.append([ant1_x, ant1_y, ant1_z]) #warning: do not take this out of the if statement with setting the key of the dictionary
                    antnames.append(ant1)
                    antnums.append(antennas[ant1])

                if ant2 not in antennas.keys():
                    antennas[ant2] = len(antennas.keys()) + 1
                    xyz.append([ant2_x, ant2_y, ant2_z]) #warning: do not take this out of the if statement with setting the key of the dictionary
                    antnames.append(ant2)
                    antnums.append(antennas[ant2])

                xyz = np.array(xyz)
                antnames = np.array(antnames)
                antnums = np.array(antnums)

                ###################################  MEASUREMENT INFO ###################################

                u_static = (1./RADPERARCSEC) * b.t202.contents.u  #Fringes/arcsec E-W 1GHz
                v_static = (1./RADPERARCSEC) * b.t202.contents.v  #Fringes/arcsec N-S 1GHz

                outdat = np.zeros((nap, 1, 1, nchan, nsubchan, nstokes, 3))
                outdat[:,:,:,:,:,:,2] = -1.0

                # this is a static u/v. we will want to change it with the real time later
                u = u_static * np.ones(outdat.shape[0])
                v = v_static * np.ones(outdat.shape[0])
                w = np.zeros(outdat.shape[0])

                ndat = outdat.shape[0]
                numberofchannels = outdat.shape[4]

                # we believe that this ordeirng of bl has to be the same as the ordering/conjugation in the data
                if antennas[ant1] < antennas[ant2]:
                    bls = (256*antennas[ant1] + antennas[ant2]) * np.ones(outdat.shape[0])
                else:
                    bls = (256*antennas[ant2] + antennas[ant1]) * np.ones(outdat.shape[0])

                ################################### ONLY LOAD ONCE ###################################
                first_pass_flag = False

            # loop through channels and get the frequency, polarization, id, and bandwidth
            clabel = [q.ffit_chan_id for q in b.t205.contents.ffit_chan[:nchan]]
            cidx = [q.channels[0] for q in b.t205.contents.ffit_chan[:nchan]]
            cinfo = [b.t203[0].channels[i] for i in cidx]

            # TODO CHECK THIS IS THE SAME THROUGHOUT THE LOOP EXCEPT CHANNEL POL
            channel_freq_ant1 = np.zeros([len(cinfo)])
            channel_freq_ant2 = np.zeros([len(cinfo)])
            channel_pol_ant1 = []
            channel_pol_ant2 = []
            channel_id_ant1 = []
            channel_id_ant2 = []
            channel_bw_ant1 = np.zeros([len(cinfo)])
            channel_bw_ant2 = np.zeros([len(cinfo)])
            count = 0

            for ch in cinfo:
                # frequency of the channel (in Hz)
                channel_freq_ant1[count] = ch.ref_freq
                channel_freq_ant2[count] = ch.rem_freq

                # bandwidth of the channel (in Hz) - mutliply by 0.5 because it is the sample_rate
                channel_bw_ant1[count] = 0.5*eat.hops.util.short2int(ch.sample_rate)
                channel_bw_ant2[count] = 0.5*eat.hops.util.short2int(ch.sample_rate)

                # the polarization 'L' or 'R'
                channel_pol_ant1.append(ch.refpol)
                channel_pol_ant2.append(ch.rempol)

                # the channel id
                channel_id_ant1.append(ch.ref_chan_id)
                channel_id_ant2.append(ch.rem_chan_id)
                count = count + 1

            channel_pol_ant1 = np.array(channel_pol_ant1)
            channel_id_ant1= np.array(channel_id_ant1)
            channel_pol_ant2 = np.array(channel_pol_ant2)
            channel_id_ant2= np.array(channel_id_ant2)

            if len(set(channel_bw_ant1)) != 1:
                print(channel_bw_ant1)
                raise Exception('the channels on this baseline have different bandwidths')

            if len(set(np.diff(channel_freq_ant1))) > 1:
                raise Exception('the spacing between the channels is different')
            else:
                channel_spacing = channel_freq_ant1[1] - channel_freq_ant1[0]

            channel_bw = channel_bw_ant1[0]
            channel1_freq = channel_freq_ant1[0]
            nsta = len(antennas)
            bw = channel_bw*numberofchannels

            #print 'ERROR: REMOVE THIS!'
            #channel_bw = channel_spacing

            # the fraction of data used to generate each measurement.
            weights = np.zeros([nchan,nap])
            for i in range(0,nchan):
                for j in range(0,nap):
                    weights[i,j] = b.t212[i].contents.data[j].weight

            # the integration time for each measurement
            inttime = inttime_fixed*weights
            tints = np.mean(inttime,axis=0)
            mjd_start = Time(eat.hops.util.mk4time(b.t205.contents.start)).mjd
            mjd_stop = Time(eat.hops.util.mk4time(b.t205.contents.stop)).mjd
            jd_start = MJD_0 + mjd_start
            jd_stop = MJD_0 + mjd_stop

            jds = (MJD_0 + np.floor(mjd_start)) * np.ones(len(outdat))
            obsseconds = np.arange(len(outdat)) * inttime_fixed
            fractimes = (mjd_start - np.floor(mjd_start)) + ( obsseconds / 86400.)
            jds = jds + fractimes
            jds = jds + ( (0.5 * inttime_fixed) / 86400. )

            # get the complex visibilities
            visibilities = eat.hops.util.pop212(a)
            if antennas[ant2] < antennas[ant1]:
                visibilities =  visibilities.conj()

            amplitude = b.t208[0].amplitude * 10000. * scalingFac
            #amplitude = np.abs(np.mean(visibilities))
            #TODO ANDREW is below comment still true?
            #print 'WARNING: the coherent average is currently off by 4 orders of magnitude - check it!'

            snr = b.t208[0].snr * scalingFac
            if snr==0.0:
                sigma_ind = -1 * np.ones(weights.shape)
            else:
                sigma_full = amplitude/snr
                sigma_ind = sigma_full * np.sqrt( np.sum(weights) / weights )
                recover_snr = np.abs(np.mean(visibilities)) / np.sqrt(np.mean(sigma_ind**2)/np.prod(sigma_ind.shape))

            ##########################  to uvfits style visibility table ##########################
            delay = b.t208[0].resid_mbd * 1e-6 # delay in sec
            rate = b.t208[0].resid_rate * 1e-6 # rate in sec/sec^2
            centertime = (eat.hops.util.mk4time(b.t205[0].utc_central) - eat.hops.util.mk4time(b.t205.contents.start)).total_seconds()

            shift = np.zeros((nchan, nap))

            if rot_rate:
                rate_mtx = ref_freq_hops*(np.matlib.repmat( ( inttime_fixed * np.arange(nap) ).reshape(1,nap), nchan, 1 ) - centertime)
                shift = shift + (2 * np.pi *  rate * rate_mtx )

            if rot_delay:
                fedge = np.array([ ch.ref_freq_hops for ch in cinfo])
                flip = np.array([-1 if ch.refsb == 'L' else 1 for ch in cinfo])
                dfvec = fedge + 0.5*flip*np.array([0.5 * short2int(ch.sample_rate) for ch in cinfo]) - ref_freq_hops
                delay_mtx = np.matlib.repmat( dfvec.reshape(dfvec.shape[0], 1), 1, nap )
                shift = shift + (2 * np.pi *  delay * delay_mtx )

            for i in xrange(nchan):

                # visibility  weight corresponds to sigma
                visweight = 1.0 / (sigma_ind[i,:]**2)

                # set weight to 0 if the snr is 0
                visweight[sigma_ind[i,:] == -1 ] = 0.0

                vis_i = visibilities[:,i] * scalingFac
                vis_i = vis_i * np.exp( 1j * shift[i,:] )

                # account for the correlation coefficient
                vis_i = vis_i / CORRCOEFF
                visweight = visweight * (CORRCOEFF**2)
                #visweight = inttime[i,:] * channel_bw_ant1[i] #TODO: WHAT WERE WE DOING?

                if flip_ALMA_pol and (ant1== 'AA' or ant2 == 'AA'):
                    if channel_pol_ant1[i]=='R' and channel_pol_ant2[i]=='L':
                        channel_pol_ant1[i]=='L'
                        channel_pol_ant2[i]=='R'
                    elif channel_pol_ant1[i]=='L' and channel_pol_ant2[i]=='R':
                        channel_pol_ant1[i]=='R'
                        channel_pol_ant2[i]=='L'

                if flip_SPT_pol:
                    swap = {'R': 'L', 'L': 'R'}
                    if ant1 == 'SP':
                        channel_pol_ant1[i]== swap[channel_pol_ant1[i]]
                    elif ant2 == 'SP':
                        channel_pol_ant2[i]== swap[channel_pol_ant2[i]]

                # put visibilities in the data table
                if channel_pol_ant1[i]=='R' and channel_pol_ant2[i]=='R':
                    outdat[:,0,0,i,0,0,0] = np.real(vis_i)
                    outdat[:,0,0,i,0,0,1] = np.imag(vis_i)
                    outdat[:,0,0,i,0,0,2] = visweight
                if channel_pol_ant1[i]=='L' and channel_pol_ant2[i]=='L':
                    outdat[:,0,0,i,0,1,0] = np.real(vis_i)
                    outdat[:,0,0,i,0,1,1] = np.imag(vis_i)
                    outdat[:,0,0,i,0,1,2] = visweight
                if channel_pol_ant1[i]=='R' and channel_pol_ant2[i]=='L':
                    outdat[:,0,0,i,0,2,0] = np.real(vis_i)
                    outdat[:,0,0,i,0,2,1] = np.imag(vis_i)
                    outdat[:,0,0,i,0,2,2] = visweight
                if channel_pol_ant1[i]=='L' and channel_pol_ant2[i]=='R':
                    outdat[:,0,0,i,0,3,0] = np.real(vis_i)
                    outdat[:,0,0,i,0,3,1] = np.imag(vis_i)
                    outdat[:,0,0,i,0,3,2] = visweight

            #print "     ", baselineName, ant1, ant2, ":",channel_pol_ant1[-1], channel_pol_ant2[-1]

        ##########################  save baseline uvfits file ############################

        # pack data
        scan_arr = np.array([[jd_start, jd_stop]]) # single scan info -- in JD!!
        obsinfo = Obs_info(srcname, ra, dec, ref_freq_hops, channel_bw, channel_spacing, channel1_freq, nchan, scan_arr)
        antennainfo = Antenna_info(antnames, antnums, xyz)

        # recompute uv points if necessary
        if recompute_uv:
            print("    recomputing uv points!")
            site1vec = xyz[0]
            site2vec = xyz[1]

            #ANDREW TODO is ref_freq_hops correct?
            times = fractimes * 24 # in hours
            (u,v) = recompute_uv_points(site1vec, site2vec, times, ra, dec, mjd_start, ref_freq_hops, timetype='UTC')

        alldata = Uvfits_data(u,v,bls,jds, tints, outdat)
        outstruct = Datastruct(obsinfo,  antennainfo, alldata)

        #print "Saving baseline uvfits file: ", fname
        fname= datadir + baselineName + '_hops_baseline.uvfits'
        save_uvfits(outstruct, fname)


    return 0

def load_hops_uvfits(filename):
    """read a uvfits file and save the data in a Datastruct

       Args:
        filename (str) :  uvfits file name
       Returns:
        Datastruct : in UVFITS format
    """

    # Read the uvfits file
    hdulist = fits.open(filename)
    header = hdulist[0].header
    data = hdulist[0].data

    # Load the telescope array data
    tnames = hdulist['AIPS AN'].data['ANNAME']
    tnums = hdulist['AIPS AN'].data['NOSTA'] - 1
    xyz = hdulist['AIPS AN'].data['STABXYZ']
    antennainfo = Antenna_info(tnames, tnums, xyz)

    # Load the various observing header parameters
    ra = header['OBSRA'] * 12./180.
    dec = header['OBSDEC']
    src = header['OBJECT']

    rf = hdulist['AIPS AN'].header['FREQ']

    if header['CTYPE4'] == 'FREQ':
        ch1_freq = header['CRVAL4']
        ch_bw = header['CDELT4']
    else: raise Exception('Cannot find observing frequency/bandwidth!')

    if header['CTYPE5'] == 'IF':
        nchan = header['NAXIS5']
    else: raise Exception('Cannot find number of channels!')

    num_ifs = len(hdulist['AIPS FQ'].data['IF FREQ'][0])
    if (num_ifs>1):
        ch_spacing = hdulist['AIPS FQ'].data['IF FREQ'][0][1] - hdulist['AIPS FQ'].data['IF FREQ'][0][0]
    else: raise Exception('Cannot find uvfits channel spacing in AIPS FREQ table!')

    # load the scan information
    refdate_str = hdulist['AIPS AN'].header['RDATE'] # in iso
    refdate = Time(refdate_str, format='isot', scale='utc').jd
    scan_starts = hdulist['AIPS NX'].data['TIME'] #in days since reference date
    scan_durs = hdulist['AIPS NX'].data['TIME INTERVAL']
    scan_arr = []
    for kk in range(len(scan_starts)):
        scan_start = scan_starts[kk]
        scan_dur = scan_durs[kk]
        # TODO AIPS MEMO 117 says scan_times should be midpoint!, but AIPS data looks likes it's at the start?
        #scan_arr.append([scan_start + refdate,
        #                  scan_start + scan_dur + refdate])
        scan_arr.append([scan_start - 0.5*scan_dur + refdate,
                          scan_start + 0.5*scan_dur + refdate])

    scan_arr = np.array(scan_arr)
    obsinfo = Obs_info(src, ra, dec, rf, ch_bw, ch_spacing, ch1_freq, nchan, scan_arr)

    # Load the random group parameters and the visibility data
    # Convert uv in lightsec to lambda by multiplying by rf
    u = data['UU---SIN'] * rf
    v = data['VV---SIN'] * rf
    baselines = data['BASELINE']
    jds = data['DATE'].astype('d') + data['_DATE'].astype('d')
    tints = data['INTTIM']
    obsdata = data['DATA']

    alldata = Uvfits_data(u,v,baselines,jds, tints, obsdata)

    return Datastruct(obsinfo, antennainfo, alldata)

def load_and_convert_hops_uvfits(filename):
    """read a uvfits file and convert data table to an ehtim-type format

       Args:
        filename (str) :  uvfits file name

       Returns:
        Datastruct : output data structure in ehtim format
    """

    # Load the uvfits file to a UVFITS format datasctruct
    datastruct = load_hops_uvfits(filename)

    # get the various necessary header parameters
    ch1_freq = datastruct.obs_info.ch_1
    ch_spacing = datastruct.obs_info.ch_spacing
    nchan = datastruct.obs_info.nchan

    # put the array data in a telescope array format
    tnames = datastruct.antenna_info.antnames
    tnums = datastruct.antenna_info.antnums
    xyz = datastruct.antenna_info.xyz

    tarr = [np.array((tnames[i], xyz[i][0], xyz[i][1], xyz[i][2]),
            dtype=DTARR) for i in range(len(tnames))]
    tarr = np.array(tarr)

    # put the random group data and vis data into a data array
    # Convert uv in lightsec to lambda by multiplying by rf
    u = datastruct.data.u
    v = datastruct.data.v
    baseline = datastruct.data.bls
    jds = datastruct.data.jds
    tints = datastruct.data.tints
    obsdata = datastruct.data.datatable

    # Sites - add names
    t1 = baseline.astype(int)/256
    t2 = baseline.astype(int) - t1*256
    t1 = t1 - 1
    t2 = t2 - 1
    t1 = np.array([tarr[i]['site'] for i in t1])
    t2 = np.array([tarr[i]['site'] for i in t2])

    # Obs Times
    #mjd = int(np.min(jds) - MJD_0)
    #times = (jds - MJD_0 - mjd) * 24.0

    # Get vis data
    rr = obsdata[:,0,0,:,0,0,0] + 1j*obsdata[:,0,0,:,0,0,1]
    ll = obsdata[:,0,0,:,0,1,0] + 1j*obsdata[:,0,0,:,0,1,1]
    rl = obsdata[:,0,0,:,0,2,0] + 1j*obsdata[:,0,0,:,0,2,1]
    lr = obsdata[:,0,0,:,0,3,0] + 1j*obsdata[:,0,0,:,0,3,1]

    rrweight = obsdata[:,0,0,:,0,0,2]
    llweight = obsdata[:,0,0,:,0,1,2]
    rlweight = obsdata[:,0,0,:,0,2,2]
    lrweight = obsdata[:,0,0,:,0,3,2]

    # Make a datatable
    # TODO check that the jd not cut off by precision
    datatable = []
    for i in xrange(len(jds)):
        for j in xrange(nchan):
            freq = j*ch_spacing + ch1_freq
            datatable.append(np.array(
                             (jds[i], freq, tints[i],
                              t1[i], t2[i],
                              u[i], v[i],
                              rr[i,j], ll[i,j], rl[i,j], lr[i,j],
                              rrweight[i,j], llweight[i,j], rlweight[i,j], lrweight[i,j]), dtype=DTPOL
                             )
                            )
    datatable = np.array(datatable)

    datastruct_out = Datastruct(datastruct.obs_info, tarr, datatable, dtype="EHTIM")

    #TODO get flags from uvfits?
    return datastruct_out


def construct_bl_list(datatable_merge, tkeys):
    """
       Args:
        datatable_merge:  merged uvfits data table
        tkeys: dictionary of index -> baseline code
       Returns:
        unique_entries_sorted: unique time-baseline entries
        unique_entry_indexes: data table index of unique entries
        inverse_indexes: index of corresponding unique entry from original table

        Construct a list of unique time-baseline entries from the merged data table
        without explictly loading all of them into memory and sorting them first.
        This function returns a list of unique entries, their indexes in the data
        table and an inverse index list, used to reconstruct the original
        (non-unique) table entries.

        This function replaces a call to numpy.unique of the form:
        _, unique_idx_anttime, idx_anttime = np.unique(bl_list, return_index=True, return_inverse=True)

    """

    bl_dict_table = dict()
    bl_set = set()
    for i in xrange(len(datatable_merge)):
        entry = datatable_merge[i]
        t1num = entry['t1']
        t2num = entry['t2']
        if tkeys[entry['t1']] > tkeys[entry['t2']]: # reorder telescopes if necessary
            #print entry['t1'], tkeys[entry['t1']], entry['t2'], tkeys[entry['t2']]
            entry['t1'] = t2num
            entry['t2'] = t1num
            entry['u'] = -entry['u']
            entry['v'] = -entry['v']
            entry['rr'] = np.conj(entry['rr'])
            entry['ll'] = np.conj(entry['ll'])
            entry['rl'] = np.conj(entry['rl'])
            entry['lr'] = np.conj(entry['lr'])
            datatable_merge[i] = entry
        bl_size = len(bl_set)
        tmp_entry = (entry['time'],entry['t1'],entry['t2'])
        bl_set.add( tmp_entry )
        if bl_size < len(bl_set): #added a unique entry to the set, so expand our lists
            bl_dict_table[ tmp_entry ] = i

    #now sort the dictionary we have constructed by the key entries (time, t1, and t2)
    #and retrieve the sorted unique list of entries
    unique_entries_sorted = sorted( bl_dict_table.keys() )

    #now create the table of indexes with the proper order
    unique_entry_indexes = []
    for entry in unique_entries_sorted:
        index = bl_dict_table[entry]
        unique_entry_indexes.append(index)

    #now create a look up table into the list of unique entries
    bl_dict_table2 = dict()
    for n in xrange(len(unique_entries_sorted)):
        bl_dict_table2[ unique_entries_sorted[n] ] = n

    #now we need to go through and construct the inverse indexes
    inverse_indexes = []
    for i in xrange(len(datatable_merge)):
        entry = datatable_merge[i]
        tmp_entry = (entry['time'],entry['t1'],entry['t2'])
        inverse_indexes.append( bl_dict_table2[tmp_entry] )

    return unique_entries_sorted, unique_entry_indexes, inverse_indexes


def merge_hops_uvfits(fitsFiles):
    """load and merge all uvfits files in a data directory
       Args:
        fitsFiles (list) :  list of uvfits files

       Returns:
        Datastruct : output data structure in uvfits format
    """

    nsubchan = 1
    nstokes = 4

    obsinfo_list = []
    tarr_list = []
    datatable_list = []
    firstfreq_list = []
    lastfreq_list = []

    # Collect all the fits files
    for fitsFile in fitsFiles:
        fname=  fitsFile

        #print "read baseline uvfits file: ", fname
        out = load_and_convert_hops_uvfits(fname)
        if out.dtype != "EHTIM":
            raise Exception("datastruct must be in EHTIM format in merge_hops_uvfits!")

        obsinfo_list.append(out.obs_info)
        tarr_list.append(out.antenna_info)
        datatable_list.append(out.data)
        firstfreq_list.append(np.min(out.data['freq']))
        lastfreq_list.append(np.max(out.data['freq']))

    # Check that observation parameters are all the same
    if (len(set([obsinfo.src for obsinfo in obsinfo_list]) ) >1 ):
        raise Exception('sources not the same!')
    else:
        src = obsinfo_list[0].src

    if (len(set([obsinfo.ra for obsinfo in obsinfo_list]) )>1):
        raise Exception('ra not the same!')
    else:
        ra = obsinfo_list[0].ra

    if (len(set([obsinfo.dec for obsinfo in obsinfo_list]) )>1):
        raise Exception('dec not the same!')
    else:
        dec = obsinfo_list[0].dec

    # NOTE: ref_freq should be equal to the first channel's freq but if the channels are not
    # aligned on every baseline the ref_freqs wont be the same
    if (len(set([obsinfo.ref_freq for obsinfo in obsinfo_list]) )>1):
        print([obsinfo.ref_freq for obsinfo in obsinfo_list])
        raise Exception('rf not the same!')
    else:
        ref_freq = obsinfo_list[0].ref_freq

    if (len(set([obsinfo.ch_bw for obsinfo in obsinfo_list]) )>1):
        raise Exception('channel bandwidth not the same!')
    else:
        ch_bw = float(obsinfo_list[0].ch_bw)

    if (len(set([obsinfo.ch_spacing for obsinfo in obsinfo_list]) )>1):
        raise Exception('channel spacings not the same in merge!')
    else:
        ch_spacing = float(obsinfo_list[0].ch_spacing)

    # Merge bands -- find the channel 1 frequency and number of channels
    firstfreq_min = np.min(np.array(firstfreq_list))
    lastfreq_max = np.max(np.array(lastfreq_list))

    for i in range(len(firstfreq_list)):
        diff = (firstfreq_list[i] - firstfreq_min)/ch_spacing
        if (np.abs(diff - np.round(diff)) > EP):
            raise Exception('all channels not aligned!')

    nchan = (lastfreq_max - firstfreq_min)/ch_spacing + 1
    if (np.abs(nchan - np.round(nchan)) > EP):
        raise Exception('channel number not an integer!')
    nchan = int(np.round(nchan))
    ch1_freq = firstfreq_min

    if not (ch1_freq in [obsinfo.ch_1 for obsinfo in obsinfo_list]):
        raise Exception('ch1_freq determined from merging ehtim style datatable not in any read uvfits header!')

    print("baseline files: ", len(fitsFiles))
    print("source: ", src)
    print("ra: ", ra)
    print("dec: ", dec)
    print("ch1 freq: ", ch1_freq/1.e9, "GHz")
    print("ref freq: ", ref_freq/1.e9, "GHz")
    print("channels: ", nchan)

    #print "Merging data ... "

    # Merge scans
    scan_list = [obsinfo.scans for obsinfo in obsinfo_list]
    scan_arr = np.vstack(scan_list)
    scan_arr = np.vstack({tuple(row) for row in scan_arr})
    scan_arr = np.sort(scan_arr, axis=0)

    start_time_last = 0.

    end_time_last = 0.
    scan_arr2 = []
    for scan in scan_arr:
        start_time =  scan[0]
        end_time =  scan[1]

        #TODO -- get rid of cases with different scan lengths?
        if end_time<=end_time_last:
            continue
        elif start_time==start_time_last:
            continue

        scan_arr2.append(scan)

        end_time_last = end_time
        start_time_last = start_time
    scan_arr = np.array(scan_arr2)

    # Merge telescope arrays
    tarr_merge = np.hstack(tarr_list)
    _, idx = np.unique(tarr_merge, return_index=True)
    tarr_merge = tarr_merge[idx]

    if len(tarr_merge) != len(set(tarr_merge['site'])):
        raise Exception('same name telescopes have different coords!')

    tkeys = {tarr_merge[i]['site']: i for i in range(len(tarr_merge))}
    tnames = tarr_merge['site']
    tnums = np.arange(1, len(tarr_merge) + 1)
    xyz = np.array([[tarr_merge[i]['x'],tarr_merge[i]['y'],tarr_merge[i]['z']] for i in np.arange(len(tarr_merge))])

    # Merge data table
    datatable_merge = np.hstack(datatable_list)
    datatable_merge.sort(order=['time','t1'])

    # get unique time and baseline data
    _, unique_idx_anttime, idx_anttime = construct_bl_list(datatable_merge, tkeys)
    _, unique_idx_freq, idx_freq = np.unique(datatable_merge['freq'], return_index=True, return_inverse=True)
    nap = len(unique_idx_anttime)

    #random group params
    u = datatable_merge['u'][unique_idx_anttime]
    v = datatable_merge['v'][unique_idx_anttime]
    jds = datatable_merge['time'][unique_idx_anttime]

    t1num = [tkeys[scope] + 1 for scope in datatable_merge['t1'][unique_idx_anttime]]
    t2num = [tkeys[scope] + 1 for scope in datatable_merge['t2'][unique_idx_anttime]]
    bls = 256*np.array(t1num) + np.array(t2num)
    tints = datatable_merge['tint'][unique_idx_anttime]


    #merged uvfits format data table
    outdat = np.zeros((nap, 1, 1, nchan, nsubchan, nstokes, 3))
    outdat[:,:,:,:,:,:,2] = -1.0

    vistypes = ['rr','ll','rl','lr']
    for i in xrange(len(datatable_merge)):
        row_freq_idx = idx_freq[i]
        row_dat_idx = idx_anttime[i]

        for j in range(len(vistypes)):
            outdat[row_dat_idx,0,0,row_freq_idx,0,j,0] = np.real(datatable_merge[i][vistypes[j]])
            outdat[row_dat_idx,0,0,row_freq_idx,0,j,1] = np.imag(datatable_merge[i][vistypes[j]])
            outdat[row_dat_idx,0,0,row_freq_idx,0,j,2] = datatable_merge[i][vistypes[j]+'weight']

    # return data for saving
    obsinfo_out = Obs_info(src, ra, dec, ref_freq, ch_bw, ch_spacing, ch1_freq, nchan, scan_arr)
    antennainfo_out = Antenna_info(tnames, tnums, xyz)
    uvfitsdata_out = Uvfits_data(u,v,bls,jds, tints, outdat)
    datastruct_out = Datastruct(obsinfo_out, antennainfo_out, uvfitsdata_out)
    return datastruct_out

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
    header['EQUINOX'] = 'J2000'
    header['ALTRPIX'] = 1.e0
    header['ALTRVAL'] = 0.e0

    #optional
    header['OBSRA'] = ra * 180./12.
    header['OBSDEC'] = dec
    header['MJD'] = float(mjd)
    header['DATE-OBS'] = Time(mjd + MJD_0, format='jd', scale='utc', out_subfmt='date').iso
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
    header['CRVAL4'] = ch1_freq # is this the right ref freq? in Hz
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
    col3= fits.Column(name='ORBPARM', format='0D', array=np.zeros(0))
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

    rdate_tt_new = Time(mjd + MJD_0, format='jd', scale='utc', out_subfmt='date')
    rdate_out = rdate_tt_new.iso
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
    hdulist.writeto(fname, clobber=True)

    return 0

##################################################################################################################################
##########################  Main FUNCTION ########################################################################################
##################################################################################################################################
def main(datadir=DATADIR_DEFAULT, outdir=OUTDIR_DEFAULT, ident='', recompute_bl_fits=True,
                    recompute_uv=False,clean_bl_fits=False, rot_rate=False, rot_delay=False, sqrt2corr=False, flip_ALMA_pol=False):


    print("********************************************************")
    print("*********************HOPS2UVFITS************************")
    print("********************************************************")

    print("Creating merged single-source uvfits files from hops fringe files")
    print("directory: ", datadir)
    print(' ')

    scandirs = [os.path.join(datadir,o) for o in os.listdir(datadir) if os.path.isdir(os.path.join(datadir,o))]

    scan_fitsFiles = []
    scan_sources = []

    i = 1
    N = len(scandirs)
    for scandir in sorted(scandirs):

        # make sure a type 2 file exists in the current directory
        fileflag = 1
        for filename in glob.glob(scandir + '/*'):
            if os.path.basename(filename).count('.')==3:
                fileflag = 0
                break
        if fileflag:
            continue


        scandir = scandir + '/'

        if recompute_bl_fits:
            # convert the finge files to baseline uv files
            print("---------------------------------------------------------")
            print("---------------------------------------------------------")
            print("scan directory %i/%i: %s" % (i,N, scandir))
            # clean up the files in case there were extra ones already there that we no longer want
            if clean_bl_fits:
                print('    REMOVING old uvfits baseline files due to --clean flag')
                for filename in glob.glob(scandir + '*_hops_baseline.uvfits'):
                    os.remove(filename)
            if not recompute_bl_fits:
                print('    WARNING - not recomputing U,V coordinates!')
            print("---------------------------------------------------------")
            print("---------------------------------------------------------")
            convert_bl_fringefiles(datadir=scandir, rot_rate=rot_rate, rot_delay=rot_delay, recompute_uv=recompute_uv, sqrt2corr=sqrt2corr, flip_ALMA_pol=flip_ALMA_pol)

        print(' ')
        print("Merging baseline uvfits files in directory: ", scandir)

        bl_fitsFiles = []
        for filename in glob.glob(scandir + '*_hops_baseline.uvfits'):
            bl_fitsFiles.append(filename)

        i += 1
        if not len(bl_fitsFiles):
            #raise Exception("cannot find any fits files with extension _hops_baseline.uvfits in %s" % scandir)
            print("cannot find any fits files with extension _hops_baseline.uvfits in %s" % scandir)
            continue

        datastruct = merge_hops_uvfits(bl_fitsFiles)
        outname = scandir + "scan_hops_merged.uvfits"
        save_uvfits(datastruct, outname)

        scan_fitsFiles.append(outname)
        scan_sources.append(datastruct.obs_info.src)


        print("Saved scan merged data to ", outname)
        print(' ')

    print(' ')
    print("---------------------------------------------------------")
    print("---------------------------------------------------------")
    print("---------------------------------------------------------")
    #print scan_sources
    print(' ')
    unique_sources = set(scan_sources)
    scan_fitsFiles = np.array(scan_fitsFiles)
    scan_sources = np.array(scan_sources)
    for source in unique_sources:
        print(' ')
        print("Merging all scan uvfits files in directory: ", datadir, "for source: ", source)
        #print 'WARNING - U,V coordinate units unknown!'
        source_scan_fitsFiles = scan_fitsFiles[scan_sources==source]
        datastruct = merge_hops_uvfits(source_scan_fitsFiles)
        outname = outdir + '/hops_' + os.path.basename(os.path.normpath(datadir)) + '_' + source + ident + '.uvfits'
        save_uvfits(datastruct, outname)
        print("Saved full merged data to ", outname)
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
        print("usage: hops2uvfits.py datadir \n" +
              "options: \n" +
              "   --outdir outdir : specifiy output directory for uvfits files \n" +
              "   --ident : specify identifying for uvfits files \n"
              "   --skip_bl : specify to skip step of recomputing individual basline files \n" +
              "   --clean : specify to remove individual baseline files after they are merged \n" +
              "   --rot_rate : specify to remove rate solution in fringe files \n" +
              "   --rot_delay : specify to remove delay rate solution in fringe files \n" +
              "   --uv : specify to recompute uv points \n" +
              "   --sqrt2corr : specify to include the sqrt(2) correction to ALMA baselines"
              "   --flip_ALMA_pol : flip RL and LR for ALMA"
             )
        sys.exit()


    recompute_bl_fits = True
    if "--skip_bl" in sys.argv: recompute_bl_fits = False

    clean_bl_fits = False
    if "--clean" in sys.argv: clean_bl_fits = True

    recompute_uv = False
    if "--uv" in  sys.argv: recompute_uv = True

    rot_rate = False
    if "--rot_rate" in sys.argv: rot_rate = True

    rot_delay = False
    if "--rot_delay" in sys.argv: rot_delay = True

    sqrt2corr = False
    if "--sqrt2corr" in sys.argv: sqrt2corr = True

    flip_ALMA_pol = False
    if "--flip_ALMA_pol" in sys.argv: flip_ALMA_pol = True

    flip_SPT_pol = False
    if "--flip_SPT_pol" in sys.argv: flip_SPT_pol = True

    ident = ""
    if "--ident" in sys.argv:
        for a in range(0, len(sys.argv)):
            if(sys.argv[a] == '--ident'):
                ident = "_" + sys.argv[a+1]

    outdir = datadir
    if "--outdir" in sys.argv:
        for a in range(0, len(sys.argv)):
            if(sys.argv[a] == '--outdir'):
                outdir = sys.argv[a+1]
    else:
        outdir = OUTDIR_DEFAULT

    main(datadir=datadir, outdir=outdir, ident=ident,
         recompute_bl_fits=recompute_bl_fits, clean_bl_fits=clean_bl_fits,
         rot_rate=rot_rate, rot_delay=rot_delay, recompute_uv=recompute_uv,
         sqrt2corr=sqrt2corr, flip_ALMA_pol=flip_ALMA_pol, flip_SPT_pol=flip_SPT_pol)
