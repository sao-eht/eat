"""
This script adapts and updates the routines to generate SEFDs from the original script by Maciek Wielgus.
"""

import re
import glob
import pandas as pd
import numpy as np
import os, datetime
import scipy.interpolate
import itertools as it
from astropy.time import Time, TimeDelta
from eat.io import vex
import eat.postproc.constants as const
from eat.io.datastructures import Caltable, Uvfits_data, Antenna_info, Datastruct
from eat.io.uvfits import save_uvfits
from eat.postproc.utils import get_elev_astropy_deg, isfloat, time2datetimeyear, ALMAtime2STANDARDtime
from eat.postproc.utils import get_elev_numpy, get_elev_astropy, convert_xyz_to_latlong, earthrot, round_time
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
    Extracts Tsys values from ANTAB files and returns them as a DataFrame,
    inserting every TSYS block at its measured timestamps.

    Parameters
    ----------
    antabpath : str
        Path to directory containing ANTAB files.
    AZ2Z : dict
        Map of 2-letter station codes to 1-letter codes.
    track2expt : dict
        Map of track identifiers to experiment numbers.
    bandL : list of str
        List of bands to include (e.g. ['b1', 'b2', 'b3', 'b4']).

    Returns
    -------
    pandas.DataFrame
        Columns ['datetime','mjd','Tsys_star_R','Tsys_star_L','band','station','track','expt'],
        with one row per timestamp in every TSYS block.
    """
    # find all ANTAB files for our bands
    list_files = [f for f in os.listdir(antabpath) if f.startswith('e') and any(f'_{band}_' in f for band in bandL)]

    cols = ['datetime','mjd','Tsys_star_R','Tsys_star_L','band','station','track','expt']
    Tsys = pd.DataFrame(columns=cols)

    for fname in list_files:
        path = os.path.join(antabpath, fname)
        track, band = os.path.basename(path).split('_')[:2]
        if track not in track2expt:
            logging.warning(f"Track {track} not in track2expt; skipping {path}")
            continue
        expt = track2expt[track]
        year = f"20{track[1:3]}"
        logging.info(f"Extracting TSYS from {path}")

        # get Tsys blocks from file
        blocks = group_tsys_blocks(path)

        for block in blocks:
            rowdict = dict(track=track, band=band, expt=expt)
            timeoff = datetime.timedelta(0)
            saw_header = False

            # first pass: parse TSYS header for station code & time-offset
            for line in block:
                if line.startswith('TSYS'):
                    parts = line.split()
                    code = parts[1]
                    if code not in AZ2Z:
                        logging.warning(f"Station {code} in {track}/{band}/{expt} not in AZ2Z; skipping this Tsys block.")
                        saw_header = False
                        break
                    rowdict['station'] = AZ2Z[code]
                    logging.info(f"Tsys block found for station {rowdict['station']}")

                    # capture timeoff value
                    match = re.search(r'timeoff=\s*([-+]?\d+(?:\.\d+)?)', line)
                    if match:
                        timeoff = datetime.timedelta(seconds=float(match.group(1)))
                        logging.info(f"  timeoff = {match.group(1)} s")
                    saw_header = True
                    break

            # skip this block if no header was found or if station not in AZ2Z
            if not saw_header:
                continue

            # second pass: extract each timestamp & value line in the block
            for line in block:
                parts = line.split()
                if not parts or not isfloat(parts[0]):
                    continue

                # parse timestamp
                day, hhmm = parts[0], parts[1]
                if rowdict['station'] == 'A':  # ALMA special case
                    dt0 = time2datetimeyear(year, day, '00:00:00')
                    dt0 += ALMAtime2STANDARDtime(hhmm) + timeoff
                else:
                    dt0 = time2datetimeyear(year, day, hhmm) + timeoff
                rowdict['datetime'] = dt0
                rowdict['mjd'] = Time(dt0).mjd

                # parse Tsys values
                vals = list(map(float, parts[2:]))
                if len(vals) == 1:
                    # single Tsys value, averaged over channels and pols
                    R = L = vals[0]
                elif len(vals) == 2:
                    # two Tsys values, per pol, averaged over channels
                    R, L = vals
                else:
                    # multiple Tsys values, per channel, averaged over pols
                    # we channel-average the Tsys values to get a single Tsys value per timestamp
                    arr = np.array(vals)
                    arr = arr[(arr != 0) & ~np.isnan(arr)]
                    R = L = (1.0/np.mean(1.0/np.sqrt(arr)))**2 if arr.size else np.nan

                rowdict['Tsys_star_R'] = R
                rowdict['Tsys_star_L'] = L

                # append this row
                row = pd.DataFrame([rowdict], columns=cols)
                if not row.dropna().empty:
                    if Tsys.empty:
                        Tsys = row
                    else:
                        Tsys = pd.concat([Tsys, row], ignore_index=True)

    # finalize types & sort
    Tsys['expt'] = Tsys['expt'].astype(int)
    Tsys = Tsys.sort_values('datetime').reset_index(drop=True)
    return Tsys

def generate_and_save_sefd_data(Tsys_matched, dict_dpfu, sourL=sourL, antL=antL0, exptL=exptL0, bandL=bandL0, expt2track={}, Z2AZ={}, pathSave='SEFD'):
    """
    Generate and save SEFD (System Equivalent Flux Density) data.
    Parameters
    ----------
    Tsys_matched : pandas.DataFrame
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
                
                # loop by source
                for sour in sourL:
                    logging.info(f"Generating SEFDs for band {band}, expt {expt}, station {ant}, source {sour}")

                    # filter Tsys data for the current band, source, antenna, and experiment, as well as positive Tsys values
                    condB = (Tsys_matched['band']==band)
                    condS = (Tsys_matched['source']==sour)
                    condA = (Tsys_matched['station']==ant)
                    condE = (Tsys_matched['track']==expt2track[expt])
                    condPositive = (Tsys_matched['Tsys_star_R']>0)&(Tsys_matched['Tsys_star_L']>0)
                    Tsys_local = Tsys_matched.loc[condB&condS&condA&condE&condPositive].copy()
                    
                    try:
                        Tsys_local.loc[:,'sefd_L'] = np.sqrt(Tsys_local['Tsys_star_L']/dict_dpfu[(ant,expt2track[expt],band,'L')])
                        Tsys_local.loc[:,'sefd_R'] = np.sqrt(Tsys_local['Tsys_star_R']/dict_dpfu[(ant,expt2track[expt],band,'R')])

                        gaincol = f'gain{ant}'
                        if gaincol in Tsys_local.columns:
                            logging.info(f"Applying additional gain correction to SEFD for antenna {ant}...")
                            Tsys_local.loc[:,'sefd_L'] = Tsys_local['sefd_L']/np.sqrt(Tsys_local[gaincol])
                            Tsys_local.loc[:,'sefd_R'] = Tsys_local['sefd_R']/np.sqrt(Tsys_local[gaincol])
                     
                        try:
                            Tsys_local.loc[:,'foo_Imag_1'] = 0.*Tsys_local['sefd_R']
                            Tsys_local.loc[:,'foo_Imag_2'] = 0.*Tsys_local['sefd_R']
                            SEFDS = Tsys_local.loc[:,['mjd','sefd_R','foo_Imag_1','sefd_L','foo_Imag_2']]
                            SEFDS = SEFDS.sort_values('mjd')

                            SEFDS = ad_dummy_values(SEFDS)

                            # filter out SEFDs with values less than 1
                            SEFDS = SEFDS[(SEFDS['sefd_R']>1.)|(SEFDS['sefd_L']>1.)]

                            # rename '1921-293' to 'J1924-2914'
                            if sour=='1921-293':
                                sour='J1924-2914'
                            
                            fname = f"{exptdir}/{sour}_{Z2AZ[ant]}.txt"
                            if SEFDS.shape[0] > 0:
                                SEFDS.to_csv(fname, sep=' ', index=False, header=False)
                                logging.info(f"{os.path.basename(fname)} generated for band {band} and expt {expt}")
                            else:
                                logging.warning(f"No valid SEFD data for {sour}_{Z2AZ[ant]} in band {band}, expt {expt}. File not created.")

                        except Exception as e:
                            logging.warning(f"Error while generating SEFDs for source {sour}, antenna {ant}, band {band}, expt {expt}: {e}")

                    except Exception as e:
                        logging.warning(f"Error while generating SEFDs for source {sour}, antenna {ant}, band {band}, expt {expt}: {e}")
                            
def ad_hoc_fixes(df,ant,sour):
    fix_T = [[57854.5,57854.6],[57850.0,57856.2]]
    if (ant=='A')&(sour in ['SGRA','J1924-2914','J1733-1304']):
        for fixt in fix_T:
            cond=(df['mjd']>fixt[0])&(df['mjd']<fixt[1])&(df['sefd_R']<7.)
            df.loc[cond,'sefd_R'] *= np.sqrt(10.)
            df.loc[cond,'sefd_L'] *= np.sqrt(10.)
    return df

def ad_dummy_values(df,dmjd=0.001):
    logging.info(f'Adding dummy boundary points to SEFDs at +/- {dmjd} days ({dmjd*const.SECONDS_IN_DAY} s) ...')
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
                    elevloc[station] = get_elev_astropy_deg(dict_ra[source[scanind]], dict_dec[source[scanind]], ant_locat[Z2SMT[station]], datet[scanind] + TimeDelta(100., format='sec'))
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

def match_scans_with_Tsys(Tsys: pd.DataFrame,
                          scans: pd.DataFrame,
                          pad_seconds: float = 0.0) -> pd.DataFrame:
    """
    Match system temperature (Tsys) data with scans.

    Parameters
    ----------
    Tsys : DataFrame
        Must contain columns 'datetime' and 'station'.
    scans : DataFrame
        Must contain 'scan_no_tot', 'time_min', 'time_max', 'source';
        optional gain columns 'gainA', 'gainB', ...
    pad_seconds : float, optional
        Expand each scan window on both sides by this many seconds.

    Returns
    -------
    DataFrame
        Same schema as the original function:
        - all original Tsys rows that fall **inside** a scan
        - extra columns: scan_no_tot, source, t_scan, gainX …
        - chronological order
    """
    # Pre-sort the two input DataFrames
    scans = scans.sort_values('time_min', ignore_index=True)
    Tsys  = Tsys.sort_values('datetime',  ignore_index=True)

    # Convert scan start end times and the Tsys datetime values to nanoseconds
    pad = pd.to_timedelta(pad_seconds, unit='s')
    start_ns = (scans['time_min'] - pad).values.astype('datetime64[ns]').astype('int64')
    end_ns   = (scans['time_max'] + pad).values.astype('datetime64[ns]').astype('int64')
    t_ns = Tsys['datetime'].values.astype('datetime64[ns]').astype('int64')

    # Map Tsys datetimes to scan numbers
    idx = np.searchsorted(start_ns, t_ns, side='right') - 1 # with side='right', subtract 1 so that we get the index of t_ns in start_ns, not the "insertion location"
    valid = (idx >= 0) & (t_ns <= end_ns[idx]) # valid indices where Tsys falls within the scan time range

    # Initialize scan_label with -1 (for Tsys timestamps outside of any scan) and the values from scan_ids for valid indices
    scan_ids = scans['scan_no_tot'].to_numpy()
    scan_label = np.full_like(idx, -1, dtype=int)
    scan_label[valid] = scan_ids[idx[valid]]

    # # Station Y quirk (keep everything, flip to positive)
    # keep_mask = valid | (list(Tsys.station.unique()) == ['Y'])
    # if list(Tsys.station.unique()) == ['Y']:
    #     scan_label = np.abs(scan_label) # emulate old behaviour
    keep_mask = valid

    # Build the output DataFrame with only the rows where Tsys falls within a scan
    out = Tsys.loc[keep_mask].copy()
    out['scan_no_tot'] = scan_label[keep_mask]

    # Add relevant columns from scans DataFrame to the matched Tsys DataFrame
    gain_cols = [c for c in scans.columns if re.fullmatch(r'gain[A-Z]', c)]
    meta_cols = ['scan_no_tot', 'source', 'time_min', *gain_cols]
    meta = scans[meta_cols].rename(columns={'time_min': 't_scan'})

    out = out.merge(meta, on='scan_no_tot', how='left', copy=False)

    # Start filling in missing Tsys values for timestamps/scans that have no Tsys data
    # Expand scans so each row has a single station
    scans_exp = scans[['scan_no_tot', 'time_min', 'time_max', 'stations']].explode('stations', ignore_index=True).rename(columns={'stations': 'station'})

    # Build the full set of expected pairs from scans and the set of pairs we already have in `out`
    pairs_scans = set(map(tuple, scans_exp[['scan_no_tot', 'station']].to_numpy()))
    pairs_have = set(map(tuple, out[['scan_no_tot', 'station']].dropna().to_numpy()))

    # Find missing pairs
    missing_pairs = pairs_scans - pairs_have

    # Choose which Tsys columns to interpolate
    tsys_cols = [c for c in Tsys.columns if c.startswith('Tsys')]

    synthetic_rows = []
    # columns from scans dataframe to copy over to synthetic rows
    scan_meta = scans.set_index('scan_no_tot')
    gaincols = [col for col in scan_meta.columns if re.match(r'gain[A-Z]$', col)] # gain columns to copy from scans dataframe

    for sid, st in missing_pairs:
        # Grab scan window and midpoint time
        s_min  = scan_meta.at[sid, 'time_min']
        s_max  = scan_meta.at[sid, 'time_max']
        t_mid  = s_min + (s_max - s_min) / 2

        # All Tsys rows for that station
        ts_df = Tsys[Tsys['station'] == st].sort_values('datetime')
        if ts_df.empty:
            # No data at all for this station
            continue

        # Convert to numpy ints for safe comparisons
        times_ns = ts_df['datetime'].to_numpy('datetime64[ns]').astype('int64')
        tmid_ns  = np.int64(pd.Timestamp(t_mid).value)

        pos = np.searchsorted(times_ns, tmid_ns)

        # Perform constant extrapolation if outside range
        if pos == 0:
            vals = ts_df.iloc[0][tsys_cols].to_numpy()
        elif pos >= len(times_ns):
            vals = ts_df.iloc[-1][tsys_cols].to_numpy()
        else:
            # Perform linear interpolation between pos-1 and pos
            row0 = ts_df.iloc[pos - 1]
            row1 = ts_df.iloc[pos]
            t0   = row0['datetime'].value
            t1   = row1['datetime'].value
            alpha = (tmid_ns - t0) / (t1 - t0) if t1 != t0 else 0.0
            v0 = row0[tsys_cols].to_numpy(dtype=float)
            v1 = row1[tsys_cols].to_numpy(dtype=float)
            vals = v0 + alpha * (v1 - v0)

        # Build synthetic row
        row_dict = {
            'datetime':   t_mid,
            'mjd': Time(t_mid.to_pydatetime(), scale='utc').mjd,
            'station':    st,
            'scan_no_tot': sid,
            'filled':     True
        }

        # Copy over the Tsys values
        for c, v in zip(tsys_cols, vals):
            row_dict[c] = v

        # Other columns to copy from scans
        row_dict['expt'] = scan_meta.at[sid, 'expt']
        for gain_col in gaincols:
            row_dict[gain_col] = scan_meta.at[sid, gain_col]
        row_dict['source'] = scan_meta.at[sid, 'source']
        if 't_scan' in out.columns:
            row_dict['t_scan'] = scan_meta.at[sid, 'time_min']

        # Fill in band/track information if available
        for extra_col in ['band', 'track']:
            if extra_col in Tsys.columns:
                use_idx = 0 if pos == 0 else (len(ts_df) - 1 if pos >= len(ts_df) else pos-1)
                row_dict[extra_col] = ts_df.iloc[use_idx][extra_col]

        synthetic_rows.append(row_dict)

    # Append synthetic rows to out DataFrame
    if synthetic_rows:
        out = pd.concat([out, pd.DataFrame(synthetic_rows)], ignore_index=True)
    
    return out.sort_values('datetime', ignore_index=True)

def global_match_scans_with_Tsys(Tsys_full, scans, antL=antL0, pad_seconds=0.0):

    Tsys_matched = pd.DataFrame({'source' : []})

    for ant in antL:
        for expt in list(Tsys_full.expt.unique()):       
            # select relevant data from Tsys_full and scans dataframes    
            Tsys_loc = Tsys_full.loc[(Tsys_full['station']==ant) & (Tsys_full['expt']==expt)].sort_values('datetime').reset_index(drop=True)
            scans_loc = scans[(scans.expt == expt) & scans.stations.apply(lambda x: ant in x)].sort_values('time_min').reset_index(drop=True)
            if(np.shape(Tsys_loc)[0]>0 and np.shape(scans_loc)[0]>0):
                Tsys_matched_loc = match_scans_with_Tsys(Tsys_loc, scans_loc, pad_seconds=pad_seconds)
                Tsys_matched= pd.concat([Tsys_matched, Tsys_matched_loc], ignore_index=True)
            else:
                continue

    Tsys_matched = Tsys_matched.sort_values('datetime').reset_index(drop=True)

    return Tsys_matched

def get_sefds_from_antab(antab_path='antab', vex_path='vex', year='2021', sourL=sourL, antL=antL0, AZ2Z=AZ2Z, \
                         SMT2Z=SMT2Z, track2expt=track2expt, ant_locat=ant_locat, exptL = exptL0, bandL=bandL0, 
                         pad_seconds=0.0, pathSave = 'SEFD'):
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
    pad_seconds : float, optional
        Number of seconds to pad the scan start and end times. Default is 0.0 seconds.
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
    Tsys_matched = global_match_scans_with_Tsys(Tsys_full, scans, antL=antL, pad_seconds=pad_seconds)

    logging.info('Computing and saving SEFD files...')
    #produce a priori calibration data
    expt2track = {value: key for key, value in track2expt.items()}
    Z2AZ = {value: key for key, value in AZ2Z.items()}
    generate_and_save_sefd_data(Tsys_matched, dict_dpfu, sourL=sourL, antL=antL, exptL=exptL, bandL=bandL, expt2track=expt2track, Z2AZ=Z2AZ, pathSave=pathSave)  

def load_caltable_ds(datastruct, tabledir, sqrt_gains=False, skip_fluxcal=False):
    """
    Load calibration table data for amplitude and field rotation corrections.

    This function loads calibration data (such as SEFDs) for each antenna site from text files,
    or generates mock SEFDs if `skip_fluxcal` is True. The data are used to construct a `Caltable`
    object for further processing.

    Parameters
    ----------
    datastruct : object
        EHTIM-type data structure containing observation metadata and data.
    tabledir : str
        Directory path containing calibration table files for each antenna site.
    sqrt_gains : bool, optional
        If True, take the square root of the gain values when loading calibration data.
        Default is False.
    skip_fluxcal : bool, optional
        If True, skip loading SEFD files and generate mock SEFDs with value 1.0 for all times.
        Default is False.

    Returns
    -------
    caltable : Caltable or bool
        Returns a `Caltable` object containing calibration data if successful.
        Returns False if no SEFD files are available and `skip_fluxcal` is False.

    Raises
    ------
    Exception
        If `datastruct` is not in EHTIM format or if the calibration table file format is unknown.

    Notes
    -----
    - Handles AIPS source name truncation by matching patterns in filenames.
    - If only one row of calibration data is present, duplicates it with a slightly offset time.
    - Skips corrupted or missing calibration files and logs warnings.
    """
    if datastruct.dtype != "EHTIM":
        raise Exception("datastruct must be in EHTIM format in load_caltable!")
    
    # Load observation info from ehtim-style datastruct
    tarr = datastruct.antenna_info
    source = datastruct.obs_info.src
    mjd = int(np.min(datastruct.data['time'] - const.MJD_0))
    ra = datastruct.obs_info.ra
    dec = datastruct.obs_info.dec
    rf = datastruct.obs_info.ref_freq
    bw = datastruct.obs_info.ch_bw

    datatables = {}
    for s in range(0, len(tarr)):
        site = tarr[s]['site'].decode() # bytes to str

        # If skip_fluxcal is True, generate mock SEFDs equal to unity
        if skip_fluxcal:
            datatable = []
            for time in np.linspace(0.,24.,100):
                datatable.append(np.array((time, 1.0, 1.0), dtype=const.DTCAL))

        else:
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

                sourcename_file = filenames[0].replace(tabledir+'/', '').replace(f'_{site}.txt', '')

                if source != sourcename_file:
                    logging.warning(f'Source name in filename ({sourcename_file}) is different from the one in datastruct ({source}).')
                    if sourcename_file.startswith(source):
                        logging.warning(f'This is likely due to AIPS source name truncation. Using the one in the filename...')
                        source = sourcename_file

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

                # Maciek's old convention had a square root
                # rscale = np.sqrt(float(row[1])) # r
                # lscale = np.sqrt(float(row[2])) # l

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
        Choose whether to use 'astropy' or 'numpy' for calculating elevation. Default is 'astropy'.
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
        raise Exception("datastruct must be in EHTIM format to proceed with apriori calibration! Exiting...")

    if not (caltable.tarr == datastruct.antenna_info).all():
        raise Exception("Mismatch between antenna information in caltable and datastruct! Exiting...")

    # declare variables
    rinterp = {}
    linterp = {}
    skipsites = []

    xyz = {}
    latitude = {}
    longitude = {}

    ra = caltable.ra * 15 * const.DEG2RAD # convert hours to radians
    dec = caltable.dec * const.DEG2RAD # convert degrees to radians
    sourcevec = np.array([np.cos(dec), 0, np.sin(dec)])

    PAR = {}
    ELE = {}
    OFF = {}

    elevfit = {}
    gmst_function = lambda time_mjd: Time(time_mjd, format='mjd').sidereal_time('mean','greenwich').hour * 2.*np.pi/24.

    if not station_frot:
        logging.warning("Empty dict passed for station mount information! Using default values...")
        station_frot = const.STATION_FROT

    # Make fine time grids for interpolation from info obtained from caltable
    if frotcal and interp_dt > 0:
        dt_mjd = interp_dt / const.SECONDS_IN_DAY # convert seconds to days

        mjd_max = -1
        mjd_min = 1e10
        for s in range(0, len(caltable.tarr)):
            site = caltable.tarr[s]['site'].decode() # bytes to str
            try:
                #sometimes station reported but no calibration
                time_mjd = caltable.data[site]['time']/24.0 + caltable.mjd
                mjd_max_foo = np.max(time_mjd)
                mjd_min_foo = np.min(time_mjd)
                if (mjd_max_foo > mjd_max):
                    mjd_max = mjd_max_foo
                if (mjd_min_foo < mjd_min):
                    mjd_min = mjd_min_foo
            except KeyError:
                continue

        time_mjd_vec = np.arange(mjd_min, mjd_max+dt_mjd, dt_mjd) # INI: create a time vector for interpolation with one additional element!!!

        datetimes_vec = Time(time_mjd_vec, format='mjd').to_datetime()
        strtime_vec = [str(round_time(x)) for x in datetimes_vec]

        gmst_vec = gmst_function(time_mjd_vec)
        thetas_vec = np.mod((gmst_vec - ra), 2.*np.pi)

    for s in range(0, len(caltable.tarr)):
        site = caltable.tarr[s]['site'].decode() # bytes to str

        xyz[site] = np.asarray((caltable.tarr[s]['x'],caltable.tarr[s]['y'],caltable.tarr[s]['z']))
        latlong = convert_xyz_to_latlong(xyz[site])
        latitude[site] = latlong[0][0] # rad
        longitude[site] = latlong[0][1] # rad

        PAR[site] = station_frot[site][0]
        ELE[site] = station_frot[site][1]
        OFF[site] = station_frot[site][2]

        # This is only if we interpolate elevation
        if frotcal and interp_dt > 0:
            if elev_function == 'numpy':
                elev_vec = get_elev_numpy(earthrot(xyz[site], thetas_vec), sourcevec)  # numpy
            else:
                elev_vec = get_elev_astropy(ra, dec, xyz[site], strtime_vec)  # astropy

            # Allow extrapolation of elevation angles to values outside the timerange obtained from caltable.
            # This is necessary since the timerange obtained from datastruct may spill over the timerange obtained from caltable.
            if extrapolate:
                elevfit[site] = scipy.interpolate.interp1d(time_mjd_vec, elev_vec, kind=elev_interp_kind, fill_value='extrapolate')
            else:
                elevfit[site] = scipy.interpolate.interp1d(time_mjd_vec, elev_vec, kind=elev_interp_kind)

        try:
            caltable.data[site]
        except KeyError:
            skipsites.append(site)
            logging.warning(f"No SEFD information found for {site}! Skipping this site...")
            continue

        # If skip_fluxcal is True, we do not load SEFDs, but create mock ones; otherwise we load the SEFDs from the caltable
        if skip_fluxcal:
            rinterp[site] = scipy.interpolate.interp1d([0],[1],kind='zero',fill_value='extrapolate')
            linterp[site] = scipy.interpolate.interp1d([0],[1],kind='zero',fill_value='extrapolate')
        else:
            time_mjd = caltable.data[site]['time']/24.0 + caltable.mjd # convert time to mjd           
            if extrapolate:
                rinterp[site] = scipy.interpolate.interp1d(time_mjd, caltable.data[site]['rscale'], kind=interp, fill_value='extrapolate')
                linterp[site] = scipy.interpolate.interp1d(time_mjd, caltable.data[site]['lscale'], kind=interp, fill_value='extrapolate')
            else:
                rinterp[site] = scipy.interpolate.interp1d(time_mjd, caltable.data[site]['rscale'], kind=interp)
                linterp[site] = scipy.interpolate.interp1d(time_mjd, caltable.data[site]['lscale'], kind=interp)


    #------------------------- Get info from datastruct ------------------
    # sort by baseline
    data =  datastruct.data
    idx = np.lexsort((data['t2'], data['t1']))
    bllist = []
    for key, group in it.groupby(data[idx], lambda x: set((x['t1'], x['t2'])) ):
        bllist.append(np.array([obs for obs in group]))

    # Apply the calibration
    datatable = []
    blcount=0
    for bl_obs in bllist:
        t1 = bl_obs['t1'][0].decode() # bytes to str
        t2 = bl_obs['t2'][0].decode()
        blcount=blcount+1
        logging.info(f'Calibrating {t1}-{t2} baseline, {blcount}/{len(bllist)}')

        time_mjd = bl_obs['time'] - const.MJD_0 # dates are in mjd in Datastruct

        if frotcal:
            gmst = gmst_function(time_mjd)
            thetas = np.mod((gmst - ra), 2*np.pi)

            hangle1 = gmst + longitude[t1] - ra # HOUR ANGLE T1
            hangle2 = gmst + longitude[t2] - ra # HOUR ANGLE T2
            
            # Calculate parallactic and elevation angles
            par1I_t1 = np.sin(hangle1) # numerators
            par1I_t2 = np.sin(hangle2)
            par1R_t1 = np.cos(dec)*np.tan(latitude[t1]) - np.sin(dec)*np.cos(hangle1) # denominators
            par1R_t2 = np.cos(dec)*np.tan(latitude[t2]) - np.sin(dec)*np.cos(hangle2)

            parangle1 = np.angle(par1R_t1 + 1j*par1I_t1) # PARALLACTIC ANGLE T1
            parangle2 = np.angle(par1R_t2 + 1j*par1I_t2) # PARALLACTIC ANGLE T2

            if interp_dt <= 0:
                if elev_function == 'numpy':
                    elev1 = get_elev_numpy(earthrot(xyz[t1], thetas), sourcevec)
                    elev2 = get_elev_numpy(earthrot(xyz[t2], thetas), sourcevec)
                else:
                    datetimes = Time(time_mjd, format='mjd').to_datetime()
                    strtime = [str(round_time(x)) for x in datetimes]
                    elev1 = get_elev_astropy(ra, dec, xyz[t1], strtime) # ELEVATION T1
                    elev2 = get_elev_astropy(ra, dec, xyz[t2], strtime) # ELEVATION T2
            else:
                # Apply scipy interpolation for elevation calculation
                elev1 = elevfit[t1](time_mjd)
                elev2 = elevfit[t2](time_mjd)

            # Compute feed rotation angles from the station mount parameters obtained from station_frot
            fran1 = PAR[t1]*parangle1 + ELE[t1]*elev1 + OFF[t1]
            fran2 = PAR[t2]*parangle2 + ELE[t2]*elev2 + OFF[t2]
            
            # Decide whether to keep the absolute phase of the LL* visibilities
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
            
        # The following is done regardless of whether frotcal is True or False
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


        # if force_singlepol == 'R':
        #   lscale1 = rscale1
        #   lscale2 = rscale2
        # if force_singlepol == 'L':
        #   rscale1 = lscale1
        #   rscale2 = lscale2

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
