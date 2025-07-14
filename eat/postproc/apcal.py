"""
This script adapts and updates the routines to generate SEFDs from the original script by Maciek Wielgus.
"""

import re
import pandas as pd
import numpy as np
import os, datetime
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import EarthLocation, AltAz, ICRS, Angle
from astropy.time import Time, TimeDelta
from eat.io import vex
import scipy.interpolate as si
from numpy.polynomial import Polynomial
import logging

# Configure logging
loglevel = getattr(logging, 'INFO', None)
logging.basicConfig(level=loglevel,
                    format='%(asctime)s %(levelname)s:: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

AZ2Z = {}
SMT2Z = {}
track2expt = {}
ant_locat = {}
sourL = []
antL0 = []
exptL0 = []
bandL0 = ['b1', 'b2', 'b3', 'b4']

def compute_elev(ra_source, dec_source, xyz_antenna, time):
    """
    Given right ascension and declination of a sky source [ICRS: ra->(deg,arcmin,arcsec) and dec->(hour,min,sec)]
    and given the position of the telescope from the vex file [Geocentric coordinates (m)]
    and the time of the observation (e.g. '2012-7-13 23:00:00') [UTC:yr-m-d],
    returns the elevation of the telescope.
    Note that every parameter can be an array (e.g. the time)

    Written by Michael Janssen
    """
    #angle conversions:
    ra_src_deg       = Angle(ra_source, unit=u.hour)
    ra_src_deg       = ra_src_deg.degree * u.deg
    dec_src_deg      = Angle(dec_source, unit=u.deg)

    source_position  = ICRS(ra=ra_src_deg, dec=dec_src_deg)
    antenna_position = EarthLocation(x=xyz_antenna[0]*u.m, y=xyz_antenna[1]*u.m, z=xyz_antenna[2]*u.m)
    altaz_system     = AltAz(location=antenna_position, obstime=time)
    trans_to_altaz   = source_position.transform_to(altaz_system)
    elevation        = trans_to_altaz.alt
    
    return elevation.degree

def extract_dpfu_gfit_from_antab(filename, az2z):
    """
    Parses a file to extract DPFU and GFIT values for antennas from ANTAB files for a single band and returns them as dictionaries.

    Parameters
    ----------
    filename : str
        The ANTAB filename with full path.
    az2z : dict
        A dictionary mapping 2-letter station codes to their respective 1-letter codes.

    Returns
    -------
    tuple
        A tuple containing two dictionaries:
        - dpfudict (dict): A dictionary with tuples of (antenna, track, band, polarization) as keys and DPFU values as values.
        - gfitdict (dict): A dictionary with tuples of (antenna, track, band, polarization) as keys and GFIT coefficients as values.

    Notes
    -----
    The ANTAB file is expected to contain DPFU and POLY (gain coefficients) for multiple antennas and polarizations.
    The values for each station are contained in lines starting with 'GAIN' followed by the strings 'DPFU' and 'POLY' with corresponding values.
    The function reads the file line by line, processes the data, and converts it into the required format.
    """
    track, band = os.path.basename(filename).split('_')[:2]
    pol = ['R', 'L']    # Polarizations

    # Regular expressions to match DPFU and POLY values
    float_re = r'[+\-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+\-]?\d+)?'
    dpfu_pattern = re.compile(rf'DPFU\s*=\s*({float_re}(?:\s*,\s*{float_re})*)')
    poly_pattern = re.compile(rf'POLY\s*=\s*({float_re}(?:\s*,\s*{float_re})*)')

    dict_dpfu = {}
    dict_gfit = {}

    # Open the file and process only those lines that start with "GAIN"
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith("GAIN"):
                line_station = line.split(' ')[1]
                # Check if the line contains a valid station code
                if line_station not in az2z:
                    continue
                # Extract the antenna code from the line
                ant = az2z[line_station]
                dpfu_match = dpfu_pattern.search(line)
                poly_match = poly_pattern.search(line)
                if dpfu_match and poly_match:
                    dpfu = [float(value.strip()) for value in dpfu_match.group(1).split(',')]
                    poly = [float(value.strip()) for value in poly_match.group(1).split(',')]

                    # Add the extracted values to the dictionary
                    dict_dpfu[(ant, track, band, pol[0])] = dpfu[0]
                    dict_dpfu[(ant, track, band, pol[1])] = dpfu[-1]
                    dict_gfit[(ant, track, band, pol[0])] = poly
                    dict_gfit[(ant, track, band, pol[1])] = poly

    return dict_dpfu, dict_gfit

def extract_dpfu_gfit_from_all_antab(folder_path, AZ2Z=AZ2Z, bandL=bandL0):
    """
    Reads ANTAB format files in a specified folder and returns dictionaries containing DPFU and GFIT (gain coefficient) values.

    Parameters
    ----------
    folder_path : str
        The path to the folder containing ANTAB format files.
    AZ2Z : dict
        A dictionary mapping 2-letter station codes to 1-letter station codes.
    bandL : list
        A list of bands for which to generate the outputs.

    Returns
    -------
    dict
        A dictionary containing DPFU values from all ANTAB files.
    dict
        A dictionary containing GFIT values from all ANTAB files.
    """
    dict_dpfu = {}; dict_gfit = {}
    list_files = [f for f in os.listdir(folder_path) if f[0] == 'e' and any(f'_{band}_' in f for band in bandL)]

    for f in list_files:
        fpath = os.path.join(folder_path, f)
        logging.info(f"Extracting DPFU and GFIT coeffs from {fpath}")
        dict_dpfu_loc, dict_gfit_loc = extract_dpfu_gfit_from_antab(fpath, AZ2Z)
        dict_dpfu = {**dict_dpfu, **dict_dpfu_loc}
        dict_gfit = {**dict_gfit, **dict_gfit_loc}
    
    return dict_dpfu, dict_gfit

def isfloat(value):
    """
    Check if the given value can be converted to a float.
    Parameters
    ----------
    value : any
        The value to check.
    Returns
    -------
    bool
        True if the value can be converted to a float, False otherwise.
    Examples
    --------
    >>> isfloat('3.14')
    True
    >>> isfloat('abc')
    False
    >>> isfloat(10)
    True
    >>> isfloat(None)
    False
    """

    try:
        float(value)
        return True
    except ValueError:
        return False

def time2datetimeyear(year, day, hour):
    """
    Convert strings representing year, day, and hour to a datetime object.

    Parameters
    ----------
    year : str
        The year as a string.
    day : str
        The day of the year as a string.
    hour : str
        The time in 'HH:MM:SS' format as a string.

    Returns
    -------
    datetime.datetime
        A datetime object representing the specified date and time.
    """
    
    day = int(day)

    hms = hour.split(':')
    h = int(hms[0])%24
    m = int(hms[1])
    s = int(hms[2])
    
    datet = (datetime.datetime(int(year), 1, 1, h, m, s) + datetime.timedelta(days=day-1))    
    
    return datet

def ALMAtime2STANDARDtime(atime):
    """
    Convert ALMA time format to standard time format.
    Parameters
    ----------
    atime : str
        Time in ALMA format (HH:MM.SS).
    Returns
    -------
    datetime.timedelta
        Time in standard format as a timedelta object.
    """
    h = int(atime.split(':')[0])
    m = int(atime.split(':')[1].split('.')[0])
    #s = round(60*(float(atime.split(':')[1].split('.')[1])/100))
    sec = (atime.split(':')[1].split('.')[1])
    frac_min = float(sec)/10**(len(sec))
    sec_with_frac = 60*frac_min
    s = np.floor(sec_with_frac)
    us = int((sec_with_frac - s)*1e6)
    dt = datetime.timedelta(hours = h, minutes=m,seconds = s, microseconds=us)

    return dt

def group_tsys_blocks(filename):
    """
    Groups lines from a file into blocks based on 'TSYS' headers and '/' delimiters.

    Parameters
    ----------
    filename : str
        The file to be read with full path.

    Returns
    -------
    blocks : list of list of str
        A list where each element is a block of lines (as a list of strings) 
        starting with 'TSYS' and ending with the second occurrence of '/'.
    """
    
    with open(filename, 'r') as file:
        lines = file.readlines()

    blocks = []
    current_block = []
    in_block = False
    first_slash_found = False

    for line in lines:
        if line.startswith('TSYS'):
            if in_block:
                blocks.append(current_block)
            current_block = [line.strip()]
            in_block = True
            first_slash_found = False
        elif in_block:
            current_block.append(line.strip())
            if '/' in line:
                if first_slash_found:
                    blocks.append(current_block)
                    current_block = []
                    in_block = False
                else:
                    first_slash_found = True

    if in_block and current_block:
        blocks.append(current_block)

    return blocks

def extract_Tsys_from_antab(antabpath, AZ2Z=AZ2Z, track2expt=track2expt, bandL=bandL0):
    """
    Extracts Tsys values from ANTAB files and returns them as a DataFrame.
    Parameters
    ----------
    antabpath : str
        Path to the directory containing ANTAB files.
    AZ2Z : dict, optional
        Dictionary mapping station codes to their respective identifiers.
    track2expt : dict, optional
        Dictionary mapping track identifiers to experiment numbers.
    bandL : list, optional
        List of bands to filter the ANTAB files.
    Returns
    -------
    pd.DataFrame
        DataFrame containing the extracted Tsys values with columns:
        ['datetime', 'mjd', 'Tsys_star_R', 'Tsys_star_L', 'band', 'station', 'track', 'expt'].
    Notes
    -----
    - The function assumes that the ANTAB files are named in a specific format (*_{band}_*) and contain Tsys blocks.
    - The function handles different formats of Tsys values, including averaging per-channel Tsys values if necessary (this is the case for ALMA).
    """

    list_files = [f for f in os.listdir(antabpath) if f[0] == 'e' and any(f'_{band}_' in f for band in bandL)]

    cols = ['datetime', 'mjd', 'Tsys_star_R', 'Tsys_star_L', 'band', 'station', 'track', 'expt']
    Tsys = pd.DataFrame(columns=cols)

    for f in list_files:
        fname = os.path.join(antabpath, f)
        track, band = os.path.basename(fname).split('_')[:2] # get track and band from the filename
        if track not in track2expt:
            logging.warning(f"Track {track} not found in track2expt. Skipping file {fname}.")
            continue  # Skip to the next file if track is not in track2expt
        expt = track2expt[track]  # Get expt number from track2expt dict
        year = f"20{track[1:3]}"
        logging.info(f"Extracting TSYS from {fname}")

        # get Tsys blocks from the file
        blocks = group_tsys_blocks(fname)

        for block in blocks:
            skip_block = False # to skip the current block if this station does not exist in the input auxiliary metadata (az2z)
            rowdict = {}
            rowdict['track'] = track
            rowdict['band'] = band
            rowdict['expt'] = expt

            first_slash_encountered = False
            for line in block:
                # Skip empty lines
                if not line.strip():
                    continue

                if line.startswith('TSYS'):
                    parts = line.split()
                    # check if the line contains a valid station code
                    if parts[1] not in AZ2Z:
                        logging.warning(f"Station {parts[1]} not found in {AZ2Z}. Skipping SEFD generation for {parts[1]}.")
                        skip_block = True
                        break
                    rowdict['station'] = AZ2Z[parts[1]] # get station code
                    print(rowdict['station'])

                    match = re.search(r'timeoff=\s*([\d.]+)', line)
                    if match:
                        timeoff = float(match.group(1))
                    else:
                        timeoff = 0.0
                    timeoff = datetime.timedelta(seconds = timeoff) # a datetime duration

                    # check if the first slash is at the end of the TSYS line
                    if line.endswith('/'):
                        first_slash_encountered = True

                # move on to the next block if the line starts with '/'
                if line.startswith('/'):
                    if first_slash_encountered:
                        break
                    else:
                        first_slash_encountered = True                    

                if isfloat(line.split()[0]):
                    parts = line.split()
                    parts = [x for x in parts if len(x) > 0]

                    # get datetime_loc
                    if rowdict['station'] == 'A':
                        datetime_loc = time2datetimeyear(year, parts[0], '00:00:00')
                        datetime_loc = datetime_loc + ALMAtime2STANDARDtime(parts[1]) + timeoff
                    else:
                        datetime_loc = time2datetimeyear(year, parts[0], parts[1]) + timeoff

                    rowdict['datetime'] = datetime_loc
                    rowdict['mjd'] = Time(datetime_loc).mjd # get mjd

                    # get Tsys values
                    if len(parts) == 3:
                        # this station has one Tsys value per time, averaged over channels and polarization feeds
                        Tsys_star_R = Tsys_star_L = float(parts[2])
                    elif len(parts) == 4:
                        # this station has Tsys values per time per pol, averaged over channels
                        Tsys_star_R = float(parts[2])
                        Tsys_star_L = float(parts[3])
                    else:
                        # this station has Tsys values per time per channel, averaged over polarization feeds
                        # We channel-average the Tsys values to get a single Tsys value per time
                        Tsysarr = np.asarray(list(map(float,parts[2:])))
                        Tsysarr = Tsysarr[(Tsysarr != 0) & ~np.isnan(Tsysarr)]
                        if Tsysarr.size > 0:
                            Tsys_star_R = Tsys_star_L = (1./np.mean(1./np.sqrt(Tsysarr)))**2
                        else:
                            Tsys_star_R = Tsys_star_L = np.nan

                    rowdict['Tsys_star_R'] = Tsys_star_R
                    rowdict['Tsys_star_L'] = Tsys_star_L

                    rowdf = pd.DataFrame([rowdict], columns=cols)
                    if not rowdf.dropna().empty:
                        if Tsys.empty:
                            Tsys = rowdf
                        else:
                            Tsys = pd.concat([Tsys, rowdf], ignore_index=True)

            if skip_block:
                continue

    Tsys.expt = Tsys.expt.astype(int) # convert expt to integer

    return Tsys

def generate_and_save_sefd_data(Tsys_full, dict_dpfu, sourL=sourL, antL=antL0, exptL=exptL0, bandL=bandL0, expt2track={}, Z2AZ={}, pathSave='SEFD'):
    """
    Generate and save SEFD (System Equivalent Flux Density) data.
    Parameters
    ----------
    Tsys_full : pandas.DataFrame
        DataFrame containing Tsys data with columns 'band', 'source', 'antena', 'track', 'Tsys_star_R', 'Tsys_star_L', 'gainP', 'gainZ', 'gainX'.
    dict_dpfu : dict
        Dictionary containing DPFU (Degrees per Flux Unit) values for different antenna, track, band, and polarization.
    sourL : list, optional
        List of sources to process. Default is sourL.
    antL : list, optional
        List of antennas to process. Default is antL0.
    exptL : list, optional
        List of experiments to process. Default is exptL0.
    bandL : list, optional
        List of bands to process. Default is bandL0.
    pathSave : str, optional
        Path to save the SEFD data. Default is 'SEFD'.

    Returns
    -------
    None
    """
    if not os.path.exists(pathSave):
        os.makedirs(pathSave)

    # loop by band
    for band in bandL:
        banddir = os.path.join(pathSave, f'SEFD_{band}')

        if not os.path.exists(banddir):
            os.makedirs(banddir)

        # loop by expt_no
        for expt in exptL:
            exptdir = os.path.join(banddir, str(int(expt)))

            if not os.path.exists(exptdir):
                os.makedirs(exptdir)
            
            # loop by station
            for ant in antL:
                print('no ad hoc fix')
                for sour in sourL:

                    condB = (Tsys_full['band']==band)
                    condS = (Tsys_full['source']==sour)
                    condA = (Tsys_full['station']==ant)
                    condE = (Tsys_full['track']==expt2track[expt])
                    condPositive = (Tsys_full['Tsys_star_R']>0)&(Tsys_full['Tsys_star_L']>0)
                    Tsys_local = Tsys_full.loc[condB&condS&condA&condE&condPositive].copy()
                    
                    try:
                        Tsys_local.loc[:,'sefd_L'] = np.sqrt(Tsys_local['Tsys_star_L']/dict_dpfu[(ant,expt2track[expt],band,'L')])
                        Tsys_local.loc[:,'sefd_R'] = np.sqrt(Tsys_local['Tsys_star_R']/dict_dpfu[(ant,expt2track[expt],band,'R')])
                        if ant=='P':
                            Tsys_local.loc[:,'sefd_L'] = Tsys_local['sefd_L']/np.sqrt(Tsys_local['gainP'])
                            Tsys_local.loc[:,'sefd_R'] = Tsys_local['sefd_R']/np.sqrt(Tsys_local['gainP'])
                        elif ant=='Z':
                            Tsys_local.loc[:,'sefd_lo_L'] = Tsys_local['sefd_L']/np.sqrt(Tsys_local['gainZ'])
                            Tsys_local.loc[:,'sefd_lo_R'] = Tsys_local['sefd_R']/np.sqrt(Tsys_local['gainZ'])
                        elif ant=='X':
                            Tsys_local.loc[:,'sefd_lo_L'] = Tsys_local['sefd_L']/np.sqrt(Tsys_local['gainX'])
                            Tsys_local.loc[:,'sefd_lo_R'] = Tsys_local['sefd_R']/np.sqrt(Tsys_local['gainX'])
                     
                       
                        
                        try:
                            Tsys_local.loc[:,'foo_Imag_1'] = 0.*Tsys_local['sefd_R']
                            Tsys_local.loc[:,'foo_Imag_2'] = 0.*Tsys_local['sefd_R']
                            SEFDS = Tsys_local.loc[:,['mjd','sefd_R','foo_Imag_1','sefd_L','foo_Imag_2']]
                            SEFDS = SEFDS.sort_values('mjd')
                            ######
                            #RENAME '1921-293' to 'J1924-2914'
                            if sour=='1921-293': sour='J1924-2914'
                            #####################################
                        
                            NameF = exptdir+'/'+sour+'_'+Z2AZ[ant]+'.txt'
                            ###APPLY AD HOC FIXES

                            SEFDS = ad_dummy_values(SEFDS)
                            #SEFDS = ad_hoc_fixes(SEFDS,ant,sour)
                            #CUT out too low SEFDs
                            SEFDS = SEFDS[(SEFDS['sefd_R']>1.)|(SEFDS['sefd_L']>1.)]
                            #####################
                            if SEFDS.shape[0]>0:
                                SEFDS.to_csv(NameF,sep=' ',index=False, header=False)
                            print(sour+'_'+Z2AZ[ant]+'_'+str(int(expt))+'_'+band+' ok')
                        except ValueError:
                            print(sour+'_'+Z2AZ[ant]+'_'+str(int(expt))+'_'+band+' crap, not ok')
                    
                    except KeyError:
                        print(sour+'_'+Z2AZ[ant]+'_'+str(int(expt))+'_'+band+' not available')
                            
def ad_hoc_fixes(df,ant,sour):
    fix_T = [[57854.5,57854.6],[57850.0,57856.2]]
    if (ant=='A')&(sour in ['SGRA','J1924-2914','J1733-1304']):
        for fixt in fix_T:
            cond=(df['mjd']>fixt[0])&(df['mjd']<fixt[1])&(df['sefd_R']<7.)
            df.loc[cond,'sefd_R'] *= np.sqrt(10.)
            df.loc[cond,'sefd_L'] *= np.sqrt(10.)
    return df

def ad_dummy_values(df,dmjd=0.001):
    print('Adding dummy boundary points to SEFDs')
    first = df.head(1).copy()
    last = df.tail(1).copy()
    first['mjd'] = list(map(lambda x: x-dmjd,first['mjd']))
    last['mjd'] = list(map(lambda x: x+dmjd,last['mjd']))
    df = pd.concat([first,df,last],ignore_index=True)
    #df.iloc[-1]
    return df

def extract_scans_from_all_vex(fpath, dict_gfit, year='2021', SMT2Z=SMT2Z, track2expt=track2expt, ant_locat=ant_locat):
    """
    Generate a list of scans from all the VEX files in a given directory.

    Parameters
    ----------
    fpath : str
        Path to the directory containing VEX files.
    dict_gfit : dict
        Dictionary containing gain fit parameters.
    year : str, optional
        Additional processing specific to campaign year. Default is '2021'.
    ant_locat : dict
        Dictionary containing antenna locations.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing scan information including source, time, elevation, antennas, and gain.

    Notes
    -----
    The function processes VEX files to extract scan information, computes elevation for specified antennas,
    and applies gain corrections based on the gain fit parameters provided in the metadata (via dict_gfit).
    """

    # get list of all VEX files in fpath
    list_files = [os.path.join(fpath, fname) for fname in os.listdir(fpath)]

    # initialize DataFrame to store scan information for all tracks
    tracks = pd.DataFrame({'source' : []})

    # extract only those sites that have polynomial coefficients for gains in the ANTAB files
    # convert to set to avoid duplicate sites that show up for different polarizations
    # note that these stations may not be present in all tracks TODO Make the entire code work for one track and loop through tracks in ehthops
    polygain_stations = list(set([key[0] for key, value in dict_gfit.items() if len(value) > 1]))

    # loop over all VEX files in fpath; one VEX file per observing track
    for fi in list_files:
        track_loc = os.path.splitext(os.path.basename(fi))[0] # get track name from vex file name
        if track_loc not in track2expt:
            logging.warning(f"Track {track_loc} not found in track2expt. Skipping this VEX file.")
            continue

        aa = vex.Vex(fi) # read VEX file

        # Create dict_ra with 'source' as keys and the 3-tuple (hours, minutes, seconds) as values
        dict_ra = {d['source']: d['ra'] for d in aa.source}

        # Create dict_dec with 'source' as keys and the 3-tuple (degrees, minutes, seconds) as values
        dict_dec = {d['source']: d['dec'] for d in aa.source}

        # populate dataframe with scan information
        tstart_hr = [aa.sched[x]['start_hr'] for x in range(len(aa.sched))]
        source = [aa.sched[x]['source'] for x in range(len(aa.sched))]
        datet = []
        elev = []
        stations = []
        duration=[]

        # loop over each item in aa.sched to extract scan information
        for scanind in range(len(aa.sched)):
            # extract MJD floor from VEX file and convert to ISO format
            mjd_floor = Time(aa.sched[scanind]['mjd_floor'], format='mjd', scale='utc')
            mjd_floor_iso = Time(mjd_floor, format='iso', scale='utc')
            mjd_floor_iso = mjd_floor_iso + TimeDelta(tstart_hr[scanind]*3600., format='sec')
            datet.append(mjd_floor_iso)

            # Include only those stations in the elevation dict that have polynomial (degree > 1) coefficients for gains.
            # Exclude stations that are absent in SMT2Z dict (derived from ovex files) which contains only those stations
            # that actually observed. The info derived from VEX files may contain stations that were scheduled but ended up not observing.
            stations_in_scan = [value['site'] for value in aa.sched[scanind]['scan'].values()]
            stations_in_scan = [SMT2Z[station] for station in stations_in_scan if station in SMT2Z.keys()]

            # compute elevation for each station with polynomial gain coeffs (POLY in ANTAB files) in the scan and append to elevation list
            elevloc = {}
            Z2SMT = {v: k for k, v in SMT2Z.items()} # to access the keys of ant_locat easily
            for station in stations_in_scan:
                if station in polygain_stations:
                    elevloc[station] = compute_elev(dict_ra[source[scanind]], dict_dec[source[scanind]], ant_locat[Z2SMT[station]], datet[scanind] + TimeDelta(100., format='sec'))
            elev.append(elevloc)

            # append list of stations in the scan to stations list
            if year == '2017' and 'S' in stations_in_scan:
                stations_in_scan = list(set(stations_in_scan) | {'R'})
            stations.append(stations_in_scan)

            # append scan durations to duration list
            scan_sec = max([aa.sched[scanind]['scan'][y]['scan_sec'] for y in range(len(aa.sched[scanind]['scan']))])
            duration.append(scan_sec)

        time_min = [pd.Timestamp(datet[x].datetime) for x in range(len(datet))]
        time_max = [time_min[x] + datetime.timedelta(seconds=duration[x]) for x in range(len(aa.sched))]

        pervexdf = pd.DataFrame(aa.sched)
        pervexdf = pervexdf[['source','mjd_floor','start_hr']]
        pervexdf['time_min'] = time_min
        pervexdf['time_max'] = time_max
        pervexdf['elev'] = elev
        pervexdf['scan_no'] = pervexdf.index
        pervexdf['scan_no'] = list(map(int, pervexdf['scan_no']))
        pervexdf['track'] = [track_loc]*pervexdf.shape[0]
        pervexdf['expt'] = [int(track2expt[track_loc])]*pervexdf.shape[0]
        pervexdf['stations'] = stations
        pervexdf['duration'] = duration

        # concatenate pervexdf to scans DataFrame
        tracks = pd.concat([tracks, pervexdf], ignore_index=True)

    tracks = tracks.reindex(['mjd_floor','expt','track','scan_no','source','time_min','time_max','duration','elev','stations'], axis=1)
    tracks = tracks.sort_values('time_max')
    tracks = tracks.reset_index(drop=True)
    tracks['scan_no_tot'] = tracks.index

    # Compute gain curves and add to dataframe
    # Get the current band from dict_gfit and set pol to 'R' for accessing gfit coeffs.
    # This works because all keys pertain to the same band and both polarizations.
    gfitband = list(dict_gfit.keys())[0][2]
    gfitpol = 'R'
    for station in polygain_stations:
        tracks[f'gain{station}'] = [1.]*tracks.shape[0]
        for index, row in tracks.iterrows():
            if station in row.stations:
                gfitkey = (station, row.track, gfitband, gfitpol)
                if gfitkey in dict_gfit:
                    # Get the polynomial coefficients for the gain curve
                    coeffs = dict_gfit[gfitkey]
                    gainf = Polynomial(coeffs)
                    foo = gainf(tracks.elev[index][station])
                    tracks.loc[index, f'gain{station}'] = float(foo)
                else:
                    tracks.loc[index, f'gain{station}'] = np.nan
                
    return tracks

def match_scans_with_Tsys(Tsys, scans):
    """
    Match system temperature (Tsys) data with scans.
    Parameters
    ----------
    Tsys : pandas.DataFrame
        DataFrame containing Tsys data with a 'datetime' column.
    scans : pandas.DataFrame
        DataFrame containing scan data with 'scan_no_tot', 'time_min', 'time_max', and 'source' columns.
    Returns
    -------
    pandas.DataFrame
        DataFrame with Tsys data matched to scans, including additional columns for 'scan_no_tot', 'source', and 't_scan'.
    Notes
    -----
    - Negative labels are used for Tsys from ANTAB corresponding to timestamps that fall between scans.
    """
    #create scan labels to match Tsys with scans
    bins_labels = [None]*(2*scans.shape[0]-1)
    bins_labels[1::2] = [-x for x in scans['scan_no_tot'][1:]]
    bins_labels[::2] = scans['scan_no_tot'].tolist()

    first_scanno = scans['scan_no_tot'].iloc[0]
    if first_scanno==0:
        first_scanno=10000
    bins_labels = [-first_scanno] + bins_labels

    dtmin = datetime.timedelta(seconds=0.) 
    dtmax = datetime.timedelta(seconds=0.) 
    binsT = [None] * (2*scans.shape[0])
    binsT[::2] = [x - dtmin for x in scans.time_min]
    binsT[1::2] = [x + dtmax for x in scans.time_max]

    #add bin for time before the first scan
    min_time = min(scans.time_min) - datetime.timedelta(seconds = 1600.)
    binsT = [min_time] + binsT

    #add scan indexed label to Tsys 
    ordered_labels = pd.cut(Tsys.datetime, binsT, labels=bins_labels)

    if list(Tsys.station.unique())==['Y']:
        Tsys = Tsys.assign(scan_no_tot=np.abs(np.asarray(list(ordered_labels))))
    else:
        Tsys = Tsys.assign(scan_no_tot=list(ordered_labels))

    DictSource = dict(zip(list(scans.scan_no_tot), list(scans.source)))
    DictTmin = dict(zip(list(scans.scan_no_tot), list(scans.time_min)))


    # Initialize a dictionary to store gain columns
    DictGains = {}

    # Filter columns that match the pattern 'gainX' where X is exactly one capital letter
    gain_columns = [col for col in scans.columns if re.match(r'^gain[A-Z]$', col)]

    # Store the results in dictionaries with the gain_columns as keys
    for col in gain_columns:
        DictGains[col] = dict(zip(list(scans.scan_no_tot), list(scans[col])))

    #select only the data taken during scans, not in between scans
    Tsys = Tsys[Tsys['scan_no_tot'] >= 0]
    Tsys = Tsys.assign(source=Tsys['scan_no_tot'].map(DictSource))
    for col in gain_columns:
        Tsys = Tsys.assign(**{col: Tsys['scan_no_tot'].map(DictGains[col])})

    # Add t_scan col to the DataFrame
    Tsys = Tsys.assign(t_scan=Tsys['scan_no_tot'].map(DictTmin))
    Tsys = Tsys.sort_values('datetime').reset_index(drop=True)
    
    return Tsys

def global_match_scans_with_Tsys(Tsys_full, scans, antL=antL0):

    Tsys_matched = pd.DataFrame({'source' : []})

    for ant in antL:
        for expt in list(Tsys_full.expt.unique()):       
            # select relevant data from Tsys_full and scans dataframes    
            Tsys_loc = Tsys_full.loc[(Tsys_full['station']==ant) & (Tsys_full['expt']==expt)].sort_values('datetime').reset_index(drop=True)
            scans_loc = scans[(scans.expt == expt) & scans.stations.apply(lambda x: ant in x)].sort_values('time_min').reset_index(drop=True)
            if(np.shape(Tsys_loc)[0]>0 and np.shape(scans_loc)[0]>0):
                Tsys_matched_loc = match_scans_with_Tsys(Tsys_loc, scans_loc)
                Tsys_matched= pd.concat([Tsys_matched, Tsys_matched_loc], ignore_index=True)
            else:
                continue

    Tsys_matched = Tsys_matched.sort_values('datetime').reset_index(drop=True)

    return Tsys_matched

def get_sefds_from_antab(antab_path='antab', vex_path='vex', year='2021', sourL=sourL, antL=antL0, AZ2Z=AZ2Z, SMT2Z=SMT2Z, track2expt=track2expt, ant_locat=ant_locat, exptL = exptL0, bandL=bandL0, pathSave = 'SEFD'):
    """
    Compute SEFD values for all sources and antennas from metadata obtained from previous calibration steps, ANTAB files, and observed VEX files.

    Parameters
    ----------
    antab_path : str, optional
        Path to the directory containing ANTAB files. Default is 'antab/'.
    vex_path : str, optional
        Path to the directory containing VEX files. Default is 'VEX/'.
    year : str, optional
        Year of the observation. Default is '2021'.
    sourL : list
        List of sources.
    antL : list
        List of antennas.
    AZ2Z : dict
        Dictionary mapping AZ to Z.
    SMT2Z : dict
        Dictionary mapping SMT to Z.
    track2expt : dict
        Dictionary mapping track to experiment.
    ant_locat : dict
        Dictionary containing antenna locations.
    exptL : list
        List of experiments.
    bandL : list
        List of bands.
    pathSave : str, optional
        Path to save the SEFD data. Default is 'SEFD'.

    Returns
    -------
    None
    """
    logging.info('Extracting DPFU, gain coeffs, and Tsys values from ANTAB files...')
    dict_dpfu, dict_gfit = extract_dpfu_gfit_from_all_antab(antab_path, AZ2Z, bandL)

    # get all Tsys data from ANTAB files
    Tsys_full = extract_Tsys_from_antab(antab_path, AZ2Z, track2expt, bandL)

    logging.info('Extracting scan information from VEX files...')
    #TABLE of SCANS from VEX files, using elevation gain info
    scans = extract_scans_from_all_vex(vex_path, dict_gfit, year=year, SMT2Z=SMT2Z, track2expt=track2expt, ant_locat=ant_locat)

    logging.info('Matching ANTAB-derived information to scans...')
    #MATCH CALIBRATION with SCANS to determine the source and 
    Tsys_matched = global_match_scans_with_Tsys(Tsys_full, scans, antL=antL)

    logging.info('Computing and saving SEFD files...')
    #produce a priori calibration data
    expt2track = {value: key for key, value in track2expt.items()}
    Z2AZ = {value: key for key, value in AZ2Z.items()}
    generate_and_save_sefd_data(Tsys_matched, dict_dpfu, sourL=sourL, antL=antL, exptL=exptL, bandL=bandL, expt2track=expt2track, Z2AZ=Z2AZ, pathSave=pathSave)  

