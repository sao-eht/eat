from typing import Iterator
import os
from typing import Dict, Tuple
import re
import logging
import ast
from numpy import deg2rad

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
            fr_off = deg2rad(float(cols[8])) # FR_OFF in radians

            if site in smt2z:
                station_frot[smt2az[site]] = (fr_par, fr_el, fr_off)
            
    return station_frot