# INI: This script contains routines for performing a priori flux calibration
# adapted from previous years' scripts by Maciek, CK, Lindy, and others.

import numpy as np
import datetime
import glob
import os
import scipy.interpolate
import itertools as it
from astropy.time import Time
import logging
from eat.io.datastructures import Caltable, Uvfits_data, Antenna_info, Datastruct
from eat.io.uvfits import save_uvfits
import eat.postproc.constants as const

# Configure logging
loglevel = getattr(logging, 'INFO', None)
logging.basicConfig(level=loglevel,
                    format='%(asctime)s %(levelname)s:: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')



def load_caltable_ds(datastruct, tabledir, sqrt_gains=False, skip_fluxcal=False):
    """Load apriori cal tables
    """

    # load metadata from the UVFITS file pre-converted to EHTIM-type datastruct
    if datastruct.dtype != "EHTIM":
        raise Exception("datastruct must be in EHTIM format in load_caltable!")
    tarr = datastruct.antenna_info
    source = datastruct.obs_info.src
    mjd = int(np.min(datastruct.data['time'] - const.MJD_0))
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
                datatable.append(np.array((time, 1.0, 1.0), dtype=const.DTCAL))

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
                logging.warning(f'No file matching {pattern} exists! Skipping...')
                continue
            else:
                logging.warning(f'More than one file matching pattern {pattern}. Skipping...')
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
                    datatable.append(np.array((time, rscale, lscale), dtype=const.DTCAL))
                #ANDREW HACKY WAY TO MAKE IT WORK WITH ONLY ONE ENTRY
                if onerowonly:
                    datatable.append(np.array((1.1*time, rscale, lscale), dtype=const.DTCAL))

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
        elev_interp_kind='cubic', err_scale=1., skip_fluxcal=False, keep_absolute_phase=True, station_frot=const.STATION_FROT):
    """
    Apply a calibration table to a uvfits file.

    Parameters
    ----------
    caltable : Caltable
        Calibration table object.
    datastruct : Datastruct
        Input data structure in EHTIM format.
    filename_out : str
        Output uvfits file name.
    interp : str, optional
        Kind of interpolation to perform for gain tables. Default is 'linear'.
    extrapolate : bool, optional
        Toggle extrapolation for gain tables and elevation computation. Default is True.
    frotcal : bool, optional
        Toggle whether field rotation angle correction is to be applied. Default is True.
    elev_function : str, optional
        Choose whether to use 'astropy' or 'ehtim' for calculating elevation. Default is 'astropy'.
    interp_dt : float, optional
        Time resolution for interpolation (seconds). Default is 1.
    elev_interp_kind : str, optional
        Kind of interpolation to perform for elevation computation. Default is 'cubic'.
    err_scale : float, optional
        Scaling factor for error. Default is 1.
    skip_fluxcal : bool, optional
        Toggle whether SEFDs should be applied (i.e. apriori amplitude calibration). Default is False.
    keep_absolute_phase : bool, optional
        Toggle whether absolute phase of LL* visibilities should be kept. Default is True.
    station_frot : dict, optional
        Dictionary of station field rotation parameters. Default is STATION_FROT.

    Returns
    -------
    None
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
        station_frot = const.STATION_FROT

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
            logging.warning(f"No SEFD information found for {site}!")
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
        logging.info('Calibrating {}-{} baseline, {}/{}'.format(t1,t2,coub,len(bllist)))
        time_mjd = bl_obs['time'] - const.MJD_0 #dates are in mjd in Datastruct
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
    bl_arr = np.empty((len(datatable)), dtype=const.BLTYPE)
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
        bl_arr[i] = np.array((entry['time'],entry['t1'],entry['t2']),dtype=const.BLTYPE)
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