# EHT table misc support utilities
# 2016-10-11 Lindy Blackburn

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

# from builtins import zip
from pkg_resources import parse_version
import pandas as pd
try:
    assert(parse_version(pd.__version__) >= parse_version('0.15.1dev'))
except:
    if type(pd).__name__ == "_MockModule":
        print("processed by autodoc; pandas version comparison failed")
    else:
        print("pandas version too old")
import datetime
import numpy as np
import os

# stations grouped according to having related fringe parameters (2013)
sites = ['DEFG', 'JPQ', 'ST', 'A']
locations = ['C', 'H', 'Z', 'A']
dishes = ['DE', 'FG', 'J', 'PQ', 'ST', 'A']
feeds = [l for l in "DEFGJPQSTA"]
# reverse index of lookup for single dish station code
# for now builtins mock module for sphinx doc will not work, remove site2loc since it is not used anywhere
# site2loc = {site:location for (sitelist, location) in zip(sites, locations) for site in sitelist}
isite = {f:i for i, flist in enumerate(sites) for f in flist}
idish = {f:i for i, flist in enumerate(dishes) for f in flist}
ifeed = {f:i for i, f in enumerate(feeds)}

def isunique(df, cols=['timetag', 'baseline', 'polarization']):
    """Return True if data frame rows are uniquely identified by columns

    e.g. check for single root_id per (timetag, baseline, polarization)

    Args:
        df (pandas.DataFrame): input data frame with necessary columns
        cols: list of columns to use for checking uniquness
    """
    count = set(len(rows) for (name, rows) in df.groupby(cols))
    return count == {1,}

# unwrap the MBD based on the 32 MHz ambiguity in HOPS, choose value closest to SBD
# this old version uses some old column names (instead of the HOPS code defined names)
def unwrap_mbd_old(df, mbd_ambiguity=None):
    if mbd_ambiguity is None:      # we may want to set this manually
        mbd_ambiguity = df.mbd_amb # if alist file does not contain sufficient precision
    offset = np.remainder(df.sbd - df.mbd + 1.5*mbd_ambiguity, mbd_ambiguity)
    df['mbd_unwrap'] = df.sbd - offset + 0.5*mbd_ambiguity

def unwrap_mbd(df, mbd_ambiguity=None):
    """Add *mbd_unwrap* to DataFrame based on ambiguity [us], choose value closest to SBD
    """

    if mbd_ambiguity is None:
        df['ambiguity'] = 1./np.round(1./df.ambiguity, 5) # improve precision of ambiguity
        mbd_ambiguity = df.ambiguity
    offset = np.remainder(df.sbdelay - df.mbdelay + 1.5*mbd_ambiguity, df.ambiguity)
    df['mbd_unwrap'] = df.sbdelay - offset + 0.5*mbd_ambiguity

def rewrap_mbd(df, mbd_ambiguity=None):
    """Rewrap in place the MBD based on the ambiguity [us], choose value within +/-ambiguity window"""
    if mbd_ambiguity is None:
        mbd_ambiguity = df.ambiguity
    df['mbdelay'] = np.remainder(df.mbd_unwrap + 0.5*mbd_ambiguity, mbd_ambiguity) - 0.5*mbd_ambiguity

def add_delayerr(df, bw=None, bw_factor=1.0, mbd_systematic=0.000002, sbd_systematic=0.000002,
                 rate_systematic=0.001, crosspol_systematic=0.000020):
    """Add in place error to delay and rate fit from fourfit.

    This is re-derived and close in spirit to the code in fourfit/fill_208.c
    but there are small different factors, not sure what is origin of the fourfit eqns
    add some sytematic errors in quadrature.. (alist precision, linear approx systematics..)

    Args:
        bw: bw spread in MHz (not in alist..) [default guess based on ambiguity and freq code]
        bw_factor: use bw*bw_factor "effective bandwidth" to calculate statistical error on estimate
                   compensates for non-white data
        mbd_systematic, rate_systematic: added in quadrature to statistical error (us, ps/s)
        crosspol_systematic: added in quadrature to delay error for cross polarization products

    Returns:
        additional columns *mbd_err* and *rate_err* added directly to original DataFrame
    """
    nchan = pd.to_numeric(df.freq_code.str[1:])
    sbw   = 1./pd.to_numeric(df.ambiguity) # bw of single channel
    if df.version.iloc[0] < 6:
        sbw = np.round(sbw) # account for lack of precision in alist v5 (round to MHz)
    if bw is None:
        bw = nchan * sbw # assume contiguous channels, not always correct
    df['mbd_err'] = np.sqrt(12) / (2*np.pi * df.snr * bw * bw_factor) # us
    df['sbd_err'] = np.sqrt(12) / (2*np.pi * df.snr * sbw * bw_factor) # us, for nchan measurements
    df['rate_err'] = 1e6 * np.sqrt(12) / (2*np.pi * df.ref_freq * df.snr * df.duration) # us/s -> ps/s
    df['mbd_err'] = np.sqrt(df['mbd_err']**2 + mbd_systematic**2 +
                            crosspol_systematic**2*df.polarization.apply(lambda p: p[0] != p[1]))
    df['sbd_err'] = np.sqrt(df['sbd_err']**2 + sbd_systematic**2 +
                            crosspol_systematic**2*df.polarization.apply(lambda p: p[0] != p[1]))
    df['rate_err'] = np.sqrt(df['rate_err']**2 + rate_systematic**2)

def tt2dt(timetag, year=2017):
    """convert HOPS timetag to pandas Timestamp (np.datetime64)"""
    return pd.to_datetime(str(year) + timetag, format="%Y%j-%H%M%S")

def dt2tt(dt):
    """convert datetime to HOPS timetag"""
    return dt.strftime("%j-%H%M%S")

def add_id(df, col=['timetag', 'baseline', 'polarization']):
    """add unique *id* tuple to data frame based on columns"""
    df['id'] = list(zip(*[df[c] for c in col]))

def add_scanno(df, unique=True):
    """add *scan_no* based on 2017 scan_id e.g. No0012 -> 12, or a unique number in increasing order"""
    if unique:
        tts = sorted(sorted(set(zip(df.expt_no, df.scan_id))))
        tt2i = dict(zip(tts, range(len(tts))))
        df['scan_no'] = [tt2i[s] for s in zip(df.expt_no, df.scan_id)]
    else:
        df['scan_no'] = df.scan_id.str[2:].astype(int)

def add_path(df):
    """add a *path* to each alist line for easier file access"""
    df['path'] = ['%s/%s/%s.%.1s.%s.%s' % par for par in zip(df.expt_no, df.scan_id, df.baseline, df.freq_code, df.extent_no, df.root_id)]

def add_utime(df):
    """add UNIX time *utime*"""
    df['utime'] = 1e-9*np.array(df.datetime).astype('float')

def add_hour(df):
    """add *hour* if HOPS timetag available"""
    if 'timetag' in df:
        df['hour'] = df.timetag.apply(lambda x: float(x[4:6]) + float(x[6:8])/60. + float(x[8:10])/3600.)
    elif 'hhmm' in df:
        df['hour'] = df.hhmm.apply(lambda x: float(x[0:2]) + float(x[2:4])/60.)

def add_doy(df):
    """add day-of-year *doy* extracted from time-tag"""
    df['doy'] = df.timetag.str[:3].astype(int)

def add_days(df):
    """decimal *days* since beginning of year = (DOY - 1) + hour/24."""
    df['days'] = df.timetag.apply(lambda x: float(x[0:3])-1. + float(x[4:6])/24. + float(x[6:8])/1440. + float(x[8:10])/86400.)

def add_gmst(df):
    """add *gmst* column to data frame with *datetime* field using astropy for conversion"""
    from astropy import time
    g = df.groupby('datetime')
    (timestamps, indices) = list(zip(*iter(g.groups.items())))
    # this broke in pandas 0.9 with API changes
    if type(timestamps[0]) is np.datetime64: # pandas < 0.9
        times_unix = 1e-9*np.array(
            timestamps).astype('float') # note one float64 is not [ns] precision
    elif type(timestamps[0]) is pd.Timestamp:
        times_unix = np.array([1e-9 * t.value for t in timestamps]) # will be int64's
    else:
        raise Exception("do not know how to convert timestamp of type " + repr(type(timestamps[0])))
    times_gmst = time.Time(
        times_unix, format='unix').sidereal_time('mean', 'greenwich').hour # vectorized
    df['gmst'] = 0. # initialize new column
    for (gmst, idx) in zip(times_gmst, indices):
        df.ix[idx, 'gmst'] = gmst

def noauto(df):
    """returns new data frame with autocorrelations removed regardless of polarziation"""
    auto = df.baseline.str[0] == df.baseline.str[1]
    return df[~auto].copy()

# a number of polconvert fixes based on rootcode (correlation proc time)
# optionally undo fix
def fix(df):
    # sqrt2 fix er2lo:('zplptp', 'zrmvon') er2hi:('zplscn', 'zrmvoi')
    idx = (df.baseline.str.count('A') == 1) & (df.root_id > 'zpaaaa') & (df.root_id < 'zrzzzz')
    df.loc[idx,'snr'] /= np.sqrt(2.0)
    df.loc[idx,'amp'] /= np.sqrt(2.0)
    # swap polarization fix er3lo:('zxuerf', 'zyjmiy') er3hi:('zymrse', 'zztobd')
    idx1 = df.baseline.str.contains('A') & (df.polarization == 'LR') & (df.root_id > 'zxaaaa') & (df.root_id < 'zzzzzz')
    idx2 = df.baseline.str.contains('A') & (df.polarization == 'RL') & (df.root_id > 'zxaaaa') & (df.root_id < 'zzzzzz')
    df.loc[idx1,'polarization'] = 'RL'
    df.loc[idx2,'polarization'] = 'LR'
    # SMA polarization swap EHT high band D05
    idx1 = (df.baseline.str[0] == 'S') & (df.root_id > 'zxaaaa') & (df.root_id < 'zztzzz') & (df.expt_no == 3597)
    idx2 = (df.baseline.str[1] == 'S') & (df.root_id > 'zxaaaa') & (df.root_id < 'zztzzz') & (df.expt_no == 3597)
    df.loc[idx1,'polarization'] = df.loc[idx1,'polarization'].map({'LL':'RL', 'LR':'RR', 'RL':'LL', 'RR':'LR'})
    df.loc[idx2,'polarization'] = df.loc[idx2,'polarization'].map({'LL':'LR', 'LR':'LL', 'RL':'RR', 'RR':'RL'})
    
def undofix(df):        
# a number of polconvert fixes based on rootcode (correlation proc time)
    # sqrt2 fix er2lo:('zplptp', 'zrmvon') er2hi:('zplscn', 'zrmvoi')
    idx = (df.baseline.str.count('A') == 1) & (df.root_id > 'zpaaaa') & (df.root_id < 'zrzzzz')
    df.loc[idx,'snr'] *= np.sqrt(2.0)
    df.loc[idx,'amp'] *= np.sqrt(2.0)
    # SMA polarization swap EHT high band D05
    idx1 = (df.baseline.str[0] == 'S') & (df.root_id > 'zxaaaa') & (df.root_id < 'zztzzz') & (df.expt_no == 3597)
    idx2 = (df.baseline.str[1] == 'S') & (df.root_id > 'zxaaaa') & (df.root_id < 'zztzzz') & (df.expt_no == 3597)
    df.loc[idx1,'polarization'] = df.loc[idx1,'polarization'].map({'LL':'RL', 'LR':'RR', 'RL':'LL', 'RR':'LR'})
    df.loc[idx2,'polarization'] = df.loc[idx2,'polarization'].map({'LL':'LR', 'LR':'LL', 'RL':'RR', 'RR':'RL'})
    # swap polarization fix er3lo:('zxuerf', 'zyjmiy') er3hi:('zymrse', 'zztobd')
    idx1 = df.baseline.str.contains('A') & (df.polarization == 'LR') & (df.root_id > 'zxaaaa') & (df.root_id < 'zzzzzz')
    idx2 = df.baseline.str.contains('A') & (df.polarization == 'RL') & (df.root_id > 'zxaaaa') & (df.root_id < 'zzzzzz')
    df.loc[idx1,'polarization'] = 'RL'
    df.loc[idx2,'polarization'] = 'LR'

def uvdict(filename):
    """take calibration output data frame, and make UV dictionary lookup table"""
    from . import hops
    df = hops.read_caltable(filename, sort=False)
    uvdict = {}
    for (day, hhmm, baseline, u, v) in zip(df.day, df.hhmm, df.baseline, df.u, df.v):
        if sort:
            bl = ''.join(sorted(baseline))
        else:
            bl = baseline
        if sf != baseline:
            (u, v) = (-u, -v)
        uvdict[(day, hhmm, bl)] = (u, v)
    return uvdict
