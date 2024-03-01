# Auxiliary ASCII tables for EHT
# 2016-10-11 Lindy Blackburn
# 2024-03-01 Updated to use Polars by Iniyan Natarajan

import polars as pl
import datetime

# CNS Clock II GPS time interval log with chosen datetime format
CNS2LOG_FIELDS = (
    ('timestamp', str), # Time Stamp: selectable format
    ('tic', float), # Time interval reading from the Time Interval Counter
    ('gps_clock_err', float), # GPS receiver "sawtooth" clock error data
    ('gps_accuracy', float), # GPS receiver accuracy estimate
    ('pps_offset', float), # PPS offset
    ('tic_corrected', float), # corrected TIC data
    ('pc_vs_utc', float), # PC time error vs UTC
    ('utc_corr_s', float), # UTC correction factor (seconds part)
    ('utc_corr_ns', float), # UTC correction factor (nanosecs part)
)

cns2log_polarsargs = dict(columns=[a[0] for a in CNS2LOG_FIELDS])

def read_cns2log(filename):
    """
    Read a GPS II time log file.

    Parameters
    ----------
    filename : str
        Name of the file to read.

    Returns
    -------
    df : DataFrame
        A Polars DataFrame with the data from the file.
    """
    df = pl.read_csv(filename, columns=cns2log_polarsargs['columns'])
    return df

# X21 USB accelerometer
X21_FIELDS = (
    ('seconds', float),
    ('xacc', int),
    ('yacc', int),
    ('zacc', int),
)

x21_polarsargs = dict(
    has_header = False,
    comment_prefix=';',
    columns=[a[0] for a in X21_FIELDS]
)

def read_x21log(filename):
    """
    Read an X21 USB accelerometer log file.

    Parameters
    ----------
    filename : str
        Name of the file to read.

    Returns
    -------
    df : DataFrame
        A Polars DataFrame with the data from the file.
    """
    df = pl.read_csv(filename, **x21_polarsargs)
    return df

# GISELA Calibrated 3mm output
G3MM_FIELDS = (
    ('number', int),
    ('uu', float),
    ('vv', float),
    ('ww', float),
    ('date', float),
    ('baseline', int),
    ('int_time', float),
    ('gate_id', int),
    ('source', int),
    ('v_real', float),
    ('v_imag', float),
    ('weight', float),
)

g3mm_polarsargs = dict(
    separator=' ',
    skip_rows=2,
    has_header = False,
    columns=[a[0] for a in G3MM_FIELDS]
)

def read_g3mm(filename):
    """
    Read a GISELA Calibrated 3mm output file.

    Parameters
    ----------
    filename : str
        Name of the file to read.

    Returns
    -------
    df : DataFrame
        A Polars DataFrame with the data from the file.
    """
    df = pl.read_csv(filename, **g3mm_polarsargs)
    return df.drop_nulls()

# MICHAEL 3mm debiased closure amplitudes
M3MMCA_FIELDS = [('time', float),] + \
	[("site%d" % (i+1), int) for i in range(8)] + \
	[field for i in range(4) for field in (("u%d" % (i+1), float), ("v%d" % (i+1), float))] + [
	('camp', float),
	('camp_err', float),
]

m3mmca_polarsargs = dict(
    separator=' ',
    skip_rows=0,
    has_header = None,
    columns=[a[0] for a in M3MMCA_FIELDS]
)

def read_m3mmca(filename):
    """
    Read a MICHAEL 3mm debiased closure amplitudes file.

    Parameters
    ----------
    filename : str
        Name of the file to read.

    Returns
    -------
    df : DataFrame
        A Polars DataFrame with the data from the file.
    """
    df = pl.read_csv(filename, **m3mmca_polarsargs)
    return df.dropna()

# ANDRE's blist for closure phase
BLIST_FIELDS = [
    ('year', int),
    ('day', int),
    ('hour', float),
    ('source', str),
    ('triangle', str),
    ('cphase', float),
    ('cphase_err', float),
    ('u0', float),
    ('u1', float),
    ('u2', float),
    ('v0', float),
    ('v1', float),
    ('v2', float),
]

blist_polarsargs = dict(
    separator=' ',
    comment_prefix='#',
    skip_rows=0,
    has_header=None,
    parse_dates={'datetime':[0,1,2]},
    columns=[a[0] for a in BLIST_FIELDS]
)

def read_blist(filename):
    """
    Read an ANDRE's blist for closure phase file.

    Parameters
    ----------
    filename : str
        Name of the file to read.

    Returns
    -------
    df : DataFrame
        A Polars DataFrame with the data from the file.
    """
    df = pl.read_csv(filename, **blist_polarsargs)
    return df

# rusen's format for SGRA 2013 ampcal
"""
 #dd hhmm     src  bl        u                v           uv-distance    sefd1       sefd2   flux    \rou     snr   ele1  ele2 exp. L/H
 80 1204     SGRA FS     462693291.3     -80013877.9     469560754.9   12532.00   19458.36  2.862   1.68     9.17  20.1  26.6 3424 1
"""
RLULIST_FIELDS = "day hhmm source baseline u v uvmag ref_sefd rem_sefd flux rou snr elev1 elev2 expt_no band".split()

rluformat = "%3d %4d %12s %2s %12.1f %12.1f %12.1f %8.2f %8.2f %5.3f %4.2f %4.2f %4.1f %4.1f %4d %1d".split()
rluformatters = {col:lambda x, fmt=fmt: fmt % x for col,fmt in zip(RLULIST_FIELDS, rluformat)}

rlulist_polarsargs = dict(
    separator=' ',
    skip_rows=1,
    has_header=False,
    columns=RLULIST_FIELDS,
)

def read_rlulist(filename):
    """
    Read a rusen's format for SGRA 2013 ampcal file.

    Parameters
    ----------
    filename : str
        Name of the file to read.

    Returns
    -------
    df : DataFrame
        A Polars DataFrame with the data from the file.
    """
    df = pd.read_csv(filename, **rlulist_polarsargs)
    return df

generic_polarsargs = dict(
    separator = ' ',
)

def read_generic(filename):
    """
    Read a generic file.

    Parameters
    ----------
    filename : str
        Name of the file to read.

    Returns
    -------
    df : DataFrame
        A Polars DataFrame with the data from the file.
    """
    df = pl.read_csv(filename, **generic_polarsargs)
    return df
