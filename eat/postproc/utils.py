from typing import Iterator
import os
from typing import Dict, Tuple
import re
import logging
import ast
import numpy as np
import datetime
from numpy.typing import ArrayLike
from typing import Union, Sequence
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import EarthLocation, AltAz, ICRS, Angle

# Configure logging
loglevel = getattr(logging, 'INFO', None)
logging.basicConfig(level=loglevel,
                    format='%(asctime)s %(levelname)s:: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# alias for “name -> (x, y)”
Coordinates = Dict[str, Tuple[float, float]]

def find_uvfits(
    base_path: str,
    extension: str = ".uvfits",
    avg_suffix: str = "+avg",
    include_avg: bool = False,
    recursive: bool = False
) -> Iterator[str]:
    """
    Yield files under `base_path` that end with `extension`, optionally recursing into subdirectories.
    By default, skips those ending in `avg_suffix` before the extension.

    Args:
        base_path: Directory in which to look.
        extension: File extension to match (default: '.uvfits').
                   May be provided with or without leading dot.
        avg_suffix: The string that precedes the extension for "averaged" files (default: '+avg').
        include_avg: If True, include files ending in '{avg_suffix}{extension}'; if False, skip them.
        recursive: If True, walk into subdirectories; otherwise only the top-level directory.

    Yields:
        Full path (as a string) to each matching file.
    """
    # Ensure extension begins with a dot
    ext = extension if extension.startswith('.') else f".{extension}"
    avg_full = f"{avg_suffix}{ext}"

    # Inner function to handle recursion
    def scan_dir(path: str) -> Iterator[str]:
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_dir():
                    if recursive:
                        yield from scan_dir(entry.path)
                    continue
                name = entry.name
                if not name.endswith(ext):
                    continue
                if not include_avg and name.endswith(avg_full):
                    continue
                yield entry.path

    # Start scanning from the base path and all paths 
    yield from scan_dir(base_path)

def read_dict_from_file(fpath: str) -> Dict[str, str]:
    """
    Reads a dictionary from a file containing its string representation.
    Parameters
    ----------
    fpath : str
        The file path to read the dictionary from.
    Returns
    -------
    Dict[str, str]
        A dictionary parsed from the file's contents. If the file does not 
        exist or is empty, an empty dictionary is returned.
    Notes
    -----
    The file is expected to contain a valid Python dictionary in string 
    representation. The function uses `ast.literal_eval` to safely evaluate 
    the string into a dictionary.
    """
    if os.path.exists(fpath):
        with open(fpath, 'r') as f:
            contents = f.read()
            if contents:
                return ast.literal_eval(contents)
            
    return {}

def write_dict_to_file(fpath: str, dictionary: Dict[str, str]) -> None:
    """
    Writes a dictionary to a file as a string.
    Parameters
    ----------
    fpath : str
        The file path where the dictionary will be written.
    dictionary : Dict[str, str]
        The dictionary to be written to the file.
    Returns
    -------
    None
    """
    with open(fpath, 'w') as f:
        f.write(str(dictionary))

    return

def extract_metadata_from_ovex(
    scandirs: list,
    az2z_file: str,
    smt2z_file: str,
    track2expt_file: str
) -> None:
    """
    Extract metadata from OVEX (root) files in the given scan directories and write them to text files.

    Parameters
    ----------
    scandirs : list
        List of directories containing scan data.
    az2z_file : str
        Path to the file where azimuth-to-Z metadata will be stored.
    smt2z_file : str
        Path to the file where station-to-Z metadata will be stored.
    track2expt_file : str
        Path to the file where track-to-experiment metadata will be stored.

    Returns
    -------
    None
    """
    # declare dictionaries for extracting metadata from ovex files
    az2z = {}
    smt2z = {}
    track2expt = {}

    ################### loop over all scan directories and create per-scan uvfits files (two-step process creates per-baseline uvfits first) ###################
    for scandir in (scandirs):
        # process scandir to get station codes and expt numbers
        pattern = r"^[a-zA-Z0-9+-]+\.[a-zA-Z0-9]{6}$"
        rootfilename = next((entry.name for entry in os.scandir(scandir) if entry.is_file() and re.match(pattern, entry.name)), None)
        if rootfilename is None:
            logging.warning(f"No valid root file (ovex) found in scan directory {scandir}. Skipping scan...")
            continue
        else:
            with open(os.path.join(scandir, rootfilename)) as f:
                logging.info(f"Processing root file {rootfilename}...")
                contents = f.read()                
                # extract track name and HOPS expt_no from root file
                pattern = r'def (\w+);.*?(?:exper_num = (\d+);.*?exper_name = \1;|exper_name = \1;.*?exper_num = (\d+);)'
                match = re.search(pattern, contents, re.DOTALL)
                if match and match.group(1) not in track2expt.keys():
                    track2expt[match.group(1)] = match.group(2) or match.group(3)

                # extract station codes from root file
                pattern = r'def (\w+);.*?site_name = \1;.*?site_ID = (\w+);.*?mk4_site_ID = (\w+);'
                matches = re.findall(pattern, contents, re.DOTALL)               
                if matches:
                    for m in matches:
                        if m[0] not in smt2z.keys():
                            smt2z[m[0]] = m[2]
                        if m[1] not in az2z.keys():
                            az2z[m[1]] = m[2]

    # create text files for tracking station metadata information; if they
    # exist, merge contents while preserving pre-existing key-value pairs.
    if os.path.exists(az2z_file):
        az2z = {**az2z, **read_dict_from_file(az2z_file)}
    az2z = {k.upper(): v.upper() for k, v in az2z.items()}
    write_dict_to_file(az2z_file, az2z)
    if os.path.exists(smt2z_file):
        smt2z = {**smt2z, **read_dict_from_file(smt2z_file)}
    smt2z = {k.upper(): v.upper() for k, v in smt2z.items()}
    write_dict_to_file(smt2z_file, smt2z)
    if os.path.exists(track2expt_file):
        track2expt = {**track2expt, **read_dict_from_file(track2expt_file)}
    write_dict_to_file(track2expt_file, track2expt)

    return

def read_station_mounts(
    array_file: str,
    az2zfile: str,
    smt2zfile: str
) -> Dict[str, str]:
    """
    Read station mount information from an EHTIM-style array.txt file.

    Parameters
    ----------
    array_file : str
        Path to the array.txt file.
    az2zfile : str
        Path to the az2z dictionary file.
    smt2zfile : str
        Path to the smt2z dictionary file.

    Returns
    -------
    Dict[str, str]
        Dictionary mapping station names to their mount types.
    """
    station_frot = {}

    # Check if all the input files exist
    if not os.path.exists(array_file):
        logging.warning(f"Array file {array_file} does not exist. Returning empty station_frot dictionary.")
        return station_frot
    if not os.path.exists(az2zfile):
        logging.warning(f"AZ2Z file {az2zfile} does not exist. Returning empty station_frot dictionary.")
        return station_frot
    if not os.path.exists(smt2zfile):
        logging.warning(f"SMT2Z file {smt2zfile} does not exist. Returning empty station_frot dictionary.")
        return station_frot
    
    az2z = read_dict_from_file(az2zfile)
    smt2z = read_dict_from_file(smt2zfile)

    # Build a new mapping from full names to 2-letter codes
    smt2az = {smt: az for smt, z in smt2z.items() for az, z2 in az2z.items() if z == z2}

    # Read the array file and extract station mount types
    with open(array_file) as fh:
        for row in fh:
            row = row.strip()
            if not row or row.startswith("#"):
                continue
            cols = row.split()
            site = cols[0]
            fr_par = float(cols[6]) # FR_PAR
            fr_el  = float(cols[7]) # FR_EL
            fr_off = np.deg2rad(float(cols[8])) # FR_OFF in radians

            if site in smt2z:
                station_frot[smt2az[site]] = (fr_par, fr_el, fr_off)
            
    return station_frot

def isfloat(value: object) -> bool:
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
    
def time2datetimeyear(
    year: str,
    day: str,
    hour: str
) -> datetime.datetime:
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
    h = int(hms[0]) % 24
    m = int(hms[1])
    s = int(hms[2])
    
    datet = (datetime.datetime(int(year), 1, 1, h, m, s) + datetime.timedelta(days=day-1))    
    
    return datet

def ALMAtime2STANDARDtime(atime: str) -> datetime.timedelta:
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
    sec = (atime.split(':')[1].split('.')[1])
    frac_min = float(sec)/10**(len(sec))
    sec_with_frac = 60*frac_min
    s = np.floor(sec_with_frac)
    us = int((sec_with_frac - s)*1e6)
    dt = datetime.timedelta(hours = h, minutes=m,seconds = s, microseconds=us)

    return dt

def get_elev_numpy(
    obsvecs: Union[np.ndarray, list],
    sourcevec: Union[np.ndarray, list]
) -> np.ndarray:
    """
    Compute the elevation angle(s) of a source as seen from one or more observer positions.

    Parameters
    ----------
    obsvecs : array_like, shape (N, 3) or (3,)
        Observer position vectors in a geocentric frame (e.g., ECEF). If a single 
        vector is provided, it will be broadcasted into shape (1, 3).
        
    sourcevec : array_like, shape (3,)
        Unit vector pointing to the source in the same coordinate frame as `obsvecs`.

    Returns
    -------
    el : np.ndarray
        Elevation angles in radians for each observer position, with shape (N,).

    Notes
    -----
    The function assumes all vectors are given in the same Cartesian coordinate system. 
    The result is computed as:
        elevation = π/2 - arccos(dot(obsvec, sourcevec) / (||obsvec|| * ||sourcevec||))

    Examples
    --------
    >>> get_elev_numpy(np.array([1, 0, 0]), np.array([0, 0, 1]))
    array([0.])

    >>> get_elev_numpy(np.array([[0, 0, 1], [0, 0, 0.5]]), np.array([0, 0, 1]))
    array([1.57079633, 1.57079633])
    """
    if len(obsvecs.shape) == 1:
        obsvecs = np.array([obsvecs])

    anglebtw = np.array([
        np.dot(obsvec, sourcevec) / np.linalg.norm(obsvec) / np.linalg.norm(sourcevec)
        for obsvec in obsvecs
    ])
    el = 0.5 * np.pi - np.arccos(anglebtw)

    return el

def get_elev_astropy(
    ra_source: Union[float, Sequence[float], np.ndarray],
    dec_source: Union[float, Sequence[float], np.ndarray],
    xyz_antenna: Union[Sequence[float], np.ndarray],
    time: Union[str, Time, Sequence[str], Sequence[Time]]
) -> np.ndarray:
    """
    Compute the elevation angle(s) of a celestial source at specified time(s) 
    as seen from a given antenna location using Astropy.

    Originally written by Michael Janssen.

    Parameters
    ----------
    ra_source : float or array_like
        Right Ascension(s) of the source in **radians**. Scalar or array.
    dec_source : float or array_like
        Declination(s) of the source in **radians**. Must broadcast with `ra_source`.
    xyz_antenna : array_like, shape (3,)
        Geocentric antenna position coordinates `[x, y, z]` in **meters**. Obtained from VEX files.
    time : str, astropy.time.Time, or array_like
        Observation time(s). Can be:
        - A string or list of strings parseable by `astropy.time.Time`,
        - A `Time` object or list of `Time` objects.
        Must broadcast with `ra_source` and `dec_source`.

    Returns
    -------
    elevation : np.ndarray
        Elevation angle(s) in **radians** above the horizon. The output array
        has the shape resulting from broadcasting `ra_source`, `dec_source`, 
        and `time`.

    Notes
    -----
    - Uses the ICRS frame for the source and transforms to the AltAz frame 
      at the specified EarthLocation and observation time(s).
    - Elevation is the altitude component of the resulting AltAz coordinate.
    - Inputs follow NumPy broadcasting rules.

    Examples
    --------
    >>> from astropy.time import Time
    >>> # Single source, single time
    >>> get_elev_astropy(0.0, 0.0, [6371000, 0, 0], Time('2025-07-20T00:00:00'))
    array([0.])
    
    >>> # Multiple sources and times
    >>> ras = [0.0, np.pi/4]
    >>> decs = [0.0, np.pi/6]
    >>> times = Time(['2025-07-20T00:00:00', '2025-07-20T01:00:00'])
    >>> get_elev_astropy(ras, decs, [6371000, 0, 0], times)
    array([..., ...])
    """
    # Convert inputs to Astropy Angle objects
    ra_src      = Angle(ra_source, unit=u.rad)
    dec_src      = Angle(dec_source, unit=u.rad)
    source_position  = ICRS(ra=ra_src, dec=dec_src)

    antenna_position = EarthLocation(x=xyz_antenna[0]*u.m, y=xyz_antenna[1]*u.m, z=xyz_antenna[2]*u.m)

    # Define the AltAz frame, translate to AltAz, and extract elevation
    altaz_system = AltAz(location=antenna_position, obstime=time)
    trans_to_altaz = source_position.transform_to(altaz_system)
    elevation = trans_to_altaz.alt

    # Return elevation in radians
    return elevation.rad

def get_elev_astropy_deg(
    ra_source: Union[float, Sequence[float], np.ndarray],
    dec_source: Union[float, Sequence[float], np.ndarray],
    xyz_antenna: Union[Sequence[float], np.ndarray],
    time: Union[str, Time, Sequence[str], Sequence[Time]]
) -> np.ndarray:
    """
    Given right ascension and declination of a sky source [ICRS: ra->(deg,arcmin,arcsec) and dec->(hour,min,sec)]
    and given the position of the telescope from the vex file [Geocentric coordinates (m)]
    and the time of the observation (e.g. '2012-7-13 23:00:00') [UTC:yr-m-d],
    returns the elevation of the telescope.
    Note that every parameter can be an array (e.g. the time)

    Written by Michael Janssen
    """
    # angle conversions
    ra_src_deg = Angle(ra_source, unit=u.hour)
    ra_src_deg = ra_src_deg.degree * u.deg
    dec_src_deg = Angle(dec_source, unit=u.deg)
    source_position = ICRS(ra=ra_src_deg, dec=dec_src_deg)

    antenna_position = EarthLocation(x=xyz_antenna[0]*u.m, y=xyz_antenna[1]*u.m, z=xyz_antenna[2]*u.m)
    altaz_system = AltAz(location=antenna_position, obstime=time)
    trans_to_altaz = source_position.transform_to(altaz_system)
    elevation = trans_to_altaz.alt

    return elevation.degree

def convert_xyz_to_latlong(
    obsvecs: ArrayLike
) -> np.ndarray:
    """
    Converts Cartesian coordinates (x, y, z) to latitude and longitude.

    Parameters
    ----------
    obsvecs : array_like
        Array of shape (N, 3) or (3,) representing N vectors or a single vector
        in Cartesian coordinates (x, y, z).

    Returns
    -------
    out : ndarray
        Array of shape (N, 2) or (2,) containing latitude and longitude pairs
        in radians for each input vector. The first column is latitude, the second is longitude.

    Notes
    -----
    Latitude is computed as arctan2(z, sqrt(x^2 + y^2)).
    Longitude is computed as arctan2(y, x).
    """

    if len(obsvecs.shape)==1:
        obsvecs=np.array([obsvecs])

    out = []
    for obsvec in obsvecs:
        x = obsvec[0]
        y = obsvec[1]
        z = obsvec[2]

        lat = np.array(np.arctan2(z, np.sqrt(x**2+y**2)))
        lon = np.array(np.arctan2(y,x))

        out.append([lat,lon])

    out = np.array(out)
    
    #if out.shape[0]==1: out = out[0]
    return out

def round_time(t: datetime.datetime, round_s: float = 1.) -> datetime.datetime:
    """
    Round a datetime object to the nearest specified number of seconds.

    Parameters
    ----------
    t : datetime.datetime
        The datetime object to be rounded.
    round_s : float, optional
        The number of seconds to round to (default is 1.0).

    Returns
    -------
    datetime.datetime
        The rounded datetime object.
    """
    t0 = datetime.datetime(t.year, 1, 1)

    foo = t - t0
    foo_s = foo.days * 24 * 3600 + foo.seconds + foo.microseconds * (1e-6)
    foo_s = np.round(foo_s / round_s) * round_s

    days = np.floor(foo_s / 24 / 3600)
    seconds = np.floor(foo_s - 24 * 3600 * days)
    microseconds = int(1e6 * (foo_s - days * 3600 * 24 - seconds))

    round_t = t0 + datetime.timedelta(days, seconds, microseconds)
    
    return round_t

def earthrot(vecs, thetas):
    """
    Rotate vectors about the z-axis by specified angles.

    Parameters
    ----------
    vecs : array_like
        A vector or an array of vectors to be rotated. Each vector should be of length 3.
    thetas : float or array_like
        Angle(s) in radians by which to rotate the vectors. Can be a single float, or an array of floats.

    Returns
    -------
    rotvec : ndarray
        Array of rotated vectors, each of length 3.

    Raises
    ------
    Exception
        If the number of vectors and angles are unequal and cannot be broadcasted.

    Examples
    --------
    >>> earthrot([1, 0, 0], np.pi/2)
    array([[0., 1., 0.]])

    >>> earthrot([[1, 0, 0], [0, 1, 0]], [np.pi/2, np.pi])
    array([[0., 1., 0.],
           [-1., 0., 0.]])
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