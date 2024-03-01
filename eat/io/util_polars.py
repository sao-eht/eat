# EHT table misc support utilities
# 2016-10-11 Lindy Blackburn
# 2024-02-26 Updated to use Polars by Iniyan Natarajan

import polars as pl
import numpy as np
import os

def isunique(df, cols=['timetag', 'baseline', 'polarization']):
    """
    Return True if the rows of dataframe are uniquely identified by the given columns.

    e.g. check for single root_id per (timetag, baseline, polarization)

    Parameters
    ----------
    df : Polars.DataFrame 
        Input dataframe containing requested columns
    cols : list 
        List of columns to use for checking uniqueness

    Returns
    -------
    bool
        True if all values in the column *len* are unique, False otherwise.
    """
    return df.group_by(cols).len().select(pl.col("len")).n_unique() == 1

# unwrap the MBD based on the 32 MHz ambiguity in HOPS, choose value closest to SBD
def unwrap_mbd(df, mbd_ambiguity=None):
    """
    Add *mbd_unwrap* to DataFrame based on ambiguity [us], choose value closest to SBD

    Parameters
    ----------
    df : Polars.DataFrame 
        Input dataframe containing columns *ambiguity*, *sbdelay*, *mbdelay*.
    mbd_ambiguity : float
        Ambiguity in mbd in units of us. If None, it is calculated from the dataframe.

    Returns
    -------
    Polars.DataFrame
        DataFrame with *mbd_unwrap* added.
    """

    if mbd_ambiguity is None:
        df = df.with_columns((1./(1./df["ambiguity"]).round(5)).alias("ambiguity")) # improve precision of ambiguity
        mbd_ambiguity = df["ambiguity"]

    offset = np.remainder(df["sbdelay"] - df["mbdelay"] + 1.5*mbd_ambiguity, df["ambiguity"])
    df = df.with_columns((df["sbdelay"] - offset + 0.5*mbd_ambiguity).alias("mbd_unwrap"))

    return df

def rewrap_mbd(df, mbd_ambiguity=None):
    """
    Rewrap in place the MBD based on the ambiguity [us], choose value within +/-ambiguity window.

    Parameters
    ----------
    df : Polars.DataFrame 
        Input dataframe containing columns unwrapped mbdelay (mbd_unwrap) and ambiguity.
    mbd_ambiguity : float
        Ambiguity in mbd in units of us. If None, it is calculated from the dataframe.

    Returns
    -------
    Polars.DataFrame
        DataFrame with rewrapped mbdelay replacing input *mbdelay* column.
    """

    if mbd_ambiguity is None:
        mbd_ambiguity = df["ambiguity"]

    df = df.with_columns((np.remainder(df["mbd_unwrap"] + 0.5*mbd_ambiguity, mbd_ambiguity) - \
                          0.5*mbd_ambiguity).alias("mbdelay"))
    
    return df

def add_delayerr(df, bw=None, bw_factor=1.0, mbd_systematic=0.000002, sbd_systematic=0.000002,
                 rate_systematic=0.001, crosspol_systematic=0.000020):
    """
    Add error columns for delay and rate fit from fourfit to the input dataframe.

    Parameters
    ----------
    bw : float
        bw spread in MHz (not in alist..) [default guess based on ambiguity and freq code].
    bw_factor : float
        use bw*bw_factor ("effective bandwidth") to calculate statistical error on estimate; 
        compensates for non-white data.
    mbd_systematic : float
        added in quadrature to statistical error (us).
    rate_systematic : float
        added in quadrature to statistical error (ps/s).
    crosspol_systematic : float
        added in quadrature to delay error for cross polarization products.

    Returns
    -------
    Polars.DataFrame
        DataFrame with new columns *mbd_err* and *rate_err* added to the input dataframe.

    Notes
    -----
        This is re-derived and close in spirit to the code in fourfit/fill_208.c
        but there are small different factors, not sure what is origin of the fourfit eqns
        add some sytematic errors in quadrature.. (alist precision, linear approx systematics..).
    """
    nchan = df["freq_code"].map_elements(lambda x: int(x[1:]))
    sbw   = 1./df["ambiguity"] # bw of single channel

    if df["version"][0] < 6:
        sbw = sbw.map_elements(round) # account for lack of precision in alist v5 (round to MHz)

    if bw is None:
        bw = nchan * sbw # assume contiguous channels, not always correct

    df = df.with_columns((np.sqrt(12) / (2*np.pi * df["snr"] * bw * bw_factor)).alias("mbd_err")) # us
    df = df.with_columns(np.sqrt(12) / (2*np.pi * df["snr"] * sbw * bw_factor).alias("sbd_err")) # us, for nchan measurements
    df = df.with_columns((1e6 * np.sqrt(12) / (2*np.pi * df["ref_freq"] * df["snr"] * df["duration"])).alias("rate_err")) # us/s -> ps/s

    df = df.with_columns((np.sqrt(df["mbd_err"]**2 + mbd_systematic**2 + crosspol_systematic**2 * \
                                    df["polarization"].map_elements(lambda p: p[0] != p[1]))).alias("mbd_err"))
    df = df.with_columns((np.sqrt(df["sbd_err"]**2 + sbd_systematic**2 + crosspol_systematic**2 * \
                                    df["polarization"].map_elements(lambda p: p[0] != p[1]))).alias("sbd_err"))
    df = df.with_columns((np.sqrt(df["rate_err"]**2 + rate_systematic**2)).alias("rate_err"))

    return df
    
def tt2dt(timetag, year=2018):
    """
    Convert HOPS timetag to polars Datetime.

    Parameters
    ----------
    timetag : str
        HOPS timetag in format DOY-HHMMSS.
    year : int
        Year to use for conversion.

    Returns
    -------
    datetime : datetime.datetime
        Datetime object corresponding to the input timetag.
    """
    return pl.Series([str(year)+timetag]).str.strptime(pl.Datetime, "%Y%j-%H%M%S")[0]

def tt2days(timetag):
    """
    Convert HOPS timetag DOY-HHMMSS to days since Jan 1 00:00:00.

    Parameters
    ----------
    timetag : str
        HOPS timetag in format DOY-HHMMSS.

    Returns
    -------
    days : float
        Decimal days since Jan 1 00:00:00.
    """
    return float(timetag[:3]) - 1. + (float(timetag[4:6]) + (float(timetag[6:8]) + float(timetag[8:10])/60.)/60.)/24.

def dt2tt(dt):
    """
    Convert datetime to HOPS timetag.

    Parameters
    ----------
    dt : datetime.datetime
        Datetime object.

    Returns
    -------
    timetag : str
        HOPS timetag in format DOY-HHMMSS.
    """
    return dt.strftime("%j-%H%M%S")

def add_id(df, col=['timetag', 'baseline', 'polarization']):
    """
    Add unique *id* described by a string generated from input columns.

    Parameters
    ----------
    df : Polars.DataFrame 
        Input dataframe.
    col : list
        List of columns to use for creating the unique id. These columns must be present in the input dataframe.

    Returns
    -------
    Polars.DataFrame
        DataFrame with new column *id* added to the input dataframe.

    Notes
    -----
        In the Pandas version of this function, the id is stored as a tuple of he values of the input columns.
        Since Polars does not support tuples natively and lists cannot be stored as csv, the id is stored as a
        string consisting of the values of the input column list *col* separated by ':'.
    """
    df = df.with_columns(id = pl.Series([f'{l[0]}:{l[1]}:{l[2]}' for l in df.select(pl.col(col)).rows()]))
    return df

def add_scanno(df, unique=True):
    """
    Add *scan_no* based on 2017 scan_id. E.g. No0012 -> 12, or a unique number in increasing order.

    Parameters
    ----------
    df : Polars.DataFrame
        Input dataframe.
    unique : bool
        If True, add a unique number in increasing order as scan_no. If False, add *scan_no* based on existing *scan_id*.

    Returns
    -------
    Polars.DataFrame
        DataFrame with new column *scan_no* added to the input dataframe.
    """
    if unique:
        tts = sorted(sorted(set(zip(df["expt_no"], df["scan_id"]))))
        tt2i = dict(zip(tts, range(len(tts))))
        df = df.with_columns(scan_no = pl.Series([tt2i[s] for s in zip(df["expt_no"], df["scan_id"])]))
    else:
        df = df.with_columns(pl.col("scan_id").map_elements(lambda x: int(x.replace("-",""))).alias("scan_no"))

    return df

def add_path(df, datadir=''):
    """
    Add a *path* to each alist line for easier file access, with optional root datadir

    Parameters
    ----------
    df : Polars.DataFrame
        Input dataframe.
    datadir : str
        Root directory to be added to the path.

    Returns
    -------
    Polars.DataFrame
        DataFrame with new column *path* added to the input dataframe.
    """
    df = df.with_columns(path = pl.Series([os.path.join(datadir, f'{l[0]}/{l[1]}/{l[2]}.{l[3]:.1s}.{l[4]}.{l[5]}') \
                                           for l in dfpl.select(pl.col(cols)).rows()]))

    return df

def add_utime(df):
    """add UNIX time *utime*"""
    #df['utime'] = 1e-9*np.array(df.datetime).astype('float')

    # TODO: Implement this function

    return df

# add a UT hour between [t0, t0+24h]
def add_hour(df, t0=-6):
    """
    Add *hour* if HOPS timetag available. If not, add *hour* based on *hhmm*.

    Parameters
    ----------
    df : Polars.DataFrame
        Input dataframe.
    t0 : float
        Reference hour for the day, defaults to -6.

    Returns
    -------
    Polars.DataFrame
        DataFrame with new column *hour* added to the input dataframe.
    """
    if 'timetag' in df:
        df = df.with_columns(pl.col('timetag').map_elements(lambda x: float(x[4:6]) + float(x[6:8])/60. + float(x[8:10])/3600.).alias('hour'))
    elif 'hhmm' in df:
        df = df.with_columns(pl.col('timetag').map_elements(lambda x: float(x[0:2]) + float(x[2:4])/60.).alias('hour'))
    if t0 < 0:
        t0 = np.fmod(t0+24, 24) - 24
    df = df.with_columns(hour=np.fmod(pl.col('hour') - t0, 24) + t0)

    return df

def add_doy(df):
    """
    Add day-of-year *doy* extracted from time-tag.

    Parameters
    ----------
    df : Polars.DataFrame
        Input dataframe.

    Returns
    -------
    Polars.DataFrame
        DataFrame with new column *doy* added to the input dataframe.
    """
    df = df.with_columns(pl.col('timetag').map_elements(lambda x: int(x[:3])).alias('doy'))
    return df

def add_days(df):
    """
    Decimal *days* since beginning of year = (DOY - 1) + hour/24.

    Parameters
    ----------
    df : Polars.DataFrame
        Input dataframe.
    
    Returns
    -------
    Polars.DataFrame
        DataFrame with new column *days* added to the input dataframe.
    """
    df = df.with_columns(pl.col('timetag').map_elements(lambda x: float(x[0:3])-1. + float(x[4:6])/24. + float(x[6:8])/1440. + \
            float(x[8:10])/86400.).alias('days'))
    
    return df

def add_gmst(df):
    """
    Add *gmst* column to dataframe using *datetime* column using astropy for conversion.
    """
    # TODO: Implement this function

    return df

def add_mjd(df):
    """add *gmst* column to data frame with *datetime* field using astropy for conversion"""

    # TODO: Implement this function

    return df

def noauto(df):
    """
    Remove autocorrelations regardless of polarization.

    Parameters
    ----------
    df : Polars.DataFrame
        Input dataframe.

    Returns
    -------
    Polars.DataFrame
        New DataFrame with autocorrelations removed.
    """

    return df.filter(pl.col('baseline').map_elements(lambda x: x[0] != x[1]))

def debias(df):
    """
    Debias amplitude by subtracting noise expectation from squared amplitude from df.amp. DO NOT run twice!!!

    Parameters
    ----------
    df : Polars.DataFrame
        Input dataframe.

    Returns
    -------
    Polars.DataFrame
        DataFrame with debiased amplitude.
    """
    df = df.with_columns((pl.col('amp')*np.sqrt(np.maximum(0.0, pl.col('snr')**2 - 1.0))/pl.col('snr')).alias('amp'))
    return df

def fix(df):
    """
    A number of polconvert fixes based on rootcode (correlation proc time).

    Parameters
    ----------
    df : Polars.DataFrame
        Input dataframe.

    Returns
    -------
    Polars.DataFrame
        DataFrame with fixes applied.
    """
    # merge source with two different names
    df.with_columns(pl.col('source').str.replace('1921-293','J1924-2914')) # inplace operation; no need to assign to df

    if 'baseline' not in df.columns:
        return df

    # sqrt2 fix er2lo:('zplptp', 'zrmvon') er2hi:('zplscn', 'zrmvoi')
    mask = (df['baseline'].str.count_matches(r"A")==1) & (df['root_id']>'zpaaaa') & (df['root_id']<'zrzzzz')
    df = df.with_columns(pl.when(mask).then(pl.col('snr') / np.sqrt(2.0)).otherwise(pl.col('snr')))
    df = df.with_columns(pl.when(mask).then(pl.col('amp') / np.sqrt(2.0)).otherwise(pl.col('amp')))

    # swap polarization fix er3lo:('zxuerf', 'zyjmiy') er3hi:('zymrse', 'zztobd') er3hiv2:('0036EJ', '00GYUV', 'zzsivx', 'zzzznu')
    mask = (df['baseline'].str.count_matches(r"A")==1) & (df['polarization'] == 'LR') & (df['root_id']>'zxaaaa') & (df['root_id']<'zzzzzz')
    df = df.with_columns(pl.when(mask).then(pl.col('polarization').map_elements(lambda x: 'RL')).otherwise(pl.col('polarization')))

    mask = (df['baseline'].str.count_matches(r"A")==1) & (df['polarization'] == 'RL') & (df['root_id']>'zxaaaa') & (df['root_id']<'zzzzzz')
    df = df.with_columns(pl.when(mask).then(pl.col('polarization').map_elements(lambda x: 'LR')).otherwise(pl.col('polarization')))

    # SMA polarization swap EHT high band D05
    mask = (df['baseline'].str.starts_with('S')) & (df['root_id']>'zxaaaa') & (df['root_id']<'zztzzz') & \
            (df['expt_no'] == 3597) & (df['ref_freq'] > 228100.)
    df = df.with_columns(pl.when(mask).then(pl.col('polarization').map_elements(lambda x: {'LL': 'RL', 'LR': 'RR', 'RL': 'LL', 'RR': 'LR'}.get(x, x))))

    mask = (df['baseline'].str.ends_with('S')) & (df['root_id']>'zxaaaa') & (df['root_id']<'zztzzz') & \
            (df['expt_no'] == 3597) & (df['ref_freq'] > 228100.)
    df = df.with_columns(pl.when(mask).then(pl.col('polarization').map_elements(lambda x: {'LL': 'LR', 'LR': 'LL', 'RL': 'RR', 'RR': 'RL'}.get(x, x))))

    # SPT polarization swap EHT high band D05 for Rev3
    mask = (df['baseline'].str.starts_with('Y')) & (df['root_id']>'zxaaaa') & (df['root_id']<'zztzzz') & \
            (df['expt_no'] == 3597) & (df['ref_freq'] > 228100.)
    df = df.with_columns(pl.when(mask).then(pl.col('polarization').map_elements(lambda x: {'LL': 'RL', 'LR': 'RR', 'RL': 'LL', 'RR': 'LR'}.get(x, x))))

    mask = (df['baseline'].str.ends_with('Y')) & (df['root_id']>'zxaaaa') & (df['root_id']<'zztzzz') & \
            (df['expt_no'] == 3597) & (df['ref_freq'] > 228100.)
    df = df.with_columns(pl.when(mask).then(pl.col('polarization').map_elements(lambda x: {'LL': 'LR', 'LR': 'LL', 'RL': 'RR', 'RR': 'RL'}.get(x, x))))

    # SPT polarization swap EHT high band D05 for Rev5 (and earlier new root code)
    mask = (df['baseline'].str.starts_with('Y')) & (df['root_id']>'000000') & (df['root_id']<'09FRZZ') & \
            (df['expt_no'] == 3597) & (df['ref_freq'] > 228100.)
    df = df.with_columns(pl.when(mask).then(pl.col('polarization').map_elements(lambda x: {'LL': 'RL', 'LR': 'RR', 'RL': 'LL', 'RR': 'LR'}.get(x, x))))

    mask = (df['baseline'].str.ends_with('Y')) & (df['root_id']>'000000') & (df['root_id']<'09FRZZ') & \
            (df['expt_no'] == 3597) & (df['ref_freq'] > 228100.)
    df = df.with_columns(pl.when(mask).then(pl.col('polarization').map_elements(lambda x: {'LL': 'LR', 'LR': 'LL', 'RL': 'RR', 'RR': 'RL'}.get(x, x))))

    return df

def undofix(df):
    """
    Undo a number of polconvert fixes based on rootcode (correlation proc time).

    Parameters
    ----------
    df : Polars.DataFrame
        Input dataframe.

    Returns
    -------
    Polars.DataFrame
        DataFrame with fixes undone.
    """
    # sqrt2 fix er2lo:('zplptp', 'zrmvon') er2hi:('zplscn', 'zrmvoi')
    mask = (df['baseline'].str.count_matches(r"A")==1) & (df['root_id']>'zpaaaa') & (df['root_id']<'zrzzzz')
    df = df.with_columns(pl.when(mask).then(pl.col('snr') * np.sqrt(2.0)).otherwise(pl.col('snr')))
    df = df.with_columns(pl.when(mask).then(pl.col('amp') * np.sqrt(2.0)).otherwise(pl.col('amp')))

    # SMA polarization swap EHT high band D05
    mask = (df['baseline'].str.starts_with('S')) & (df['root_id']>'zxaaaa') & (df['root_id']<'zztzzz') & \
            (df['expt_no'] == 3597) & (df['ref_freq'] > 228100.)
    df = df.with_columns(pl.when(mask).then(pl.col('polarization').map_elements(lambda x: {'LL': 'RL', 'LR': 'RR', 'RL': 'LL', 'RR': 'LR'}.get(x, x))))

    mask = (df['baseline'].str.ends_with('S')) & (df['root_id']>'zxaaaa') & (df['root_id']<'zztzzz') & \
            (df['expt_no'] == 3597) & (df['ref_freq'] > 228100.)
    df = df.with_columns(pl.when(mask).then(pl.col('polarization').map_elements(lambda x: {'LL': 'LR', 'LR': 'LL', 'RL': 'RR', 'RR': 'RL'}.get(x, x))))

    # swap polarization fix er3lo:('zxuerf', 'zyjmiy') er3hi:('zymrse', 'zztobd')
    mask = (df['baseline'].str.count_matches(r"A")==1) & (df['polarization'] == 'LR') & (df['root_id']>'zxaaaa') & (df['root_id']<'zzzzzz')
    df = df.with_columns(pl.when(mask).then(pl.col('polarization').map_elements(lambda x: 'RL')).otherwise(pl.col('polarization')))

    mask = (df['baseline'].str.count_matches(r"A")==1) & (df['polarization'] == 'RL') & (df['root_id']>'zxaaaa') & (df['root_id']<'zzzzzz')
    df = df.with_columns(pl.when(mask).then(pl.col('polarization').map_elements(lambda x: 'LR')).otherwise(pl.col('polarization')))

    return df

def uvdict(filename):
    """
    Take calibration output data frame, and make UV dictionary lookup table.

    Parameters
    ----------
    filename : str
        Name of the calibration output file.

    Returns
    -------
    dict
        Dictionary with keys (day, hhmm, baseline) and values (u, v).
    """
    from . import hops_polars
    df = hops_polars.read_caltable(filename)
    uvdict = {}
    for (day, hhmm, baseline, u, v) in zip(df["day"], df["hhmm"], df["baseline"], df["u"], df["v"]):
        if sort:
            bl = ''.join(sorted(baseline))
        else:
            bl = baseline
        if sf != baseline:
            (u, v) = (-u, -v)
        uvdict[(day, hhmm, bl)] = (u, v)

    return uvdict