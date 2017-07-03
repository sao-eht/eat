# EHT table misc support utilities
# 2016-10-11 Lindy Blackburn

from pkg_resources import parse_version
import pandas as pd
if parse_version(pd.__version__) < parse_version('0.15.1dev'):
    print "pandas version too old and buggy, please update"
import datetime
import numpy as np
import os

# stations grouped according to having related fringe parameters
sites = ['DEFG', 'JPQ', 'ST', 'A']
locations = ['C', 'H', 'Z', 'A']
dishes = ['DE', 'FG', 'J', 'PQ', 'ST', 'A']
feeds = [l for l in "DEFGJPQSTA"]
# reverse index of lookup for single dish station code
site2loc = {site:location for (sitelist, location) in zip(sites, locations) for site in sitelist}
isite = {f:i for i, flist in enumerate(sites) for f in flist}
idish = {f:i for i, flist in enumerate(dishes) for f in flist}
ifeed = {f:i for i, f in enumerate(feeds)}

def istrivial(triangle):
    locs = set((site2loc[s] for s in triangle))
    return len(locs) < 3

# return True if data frame rows are uniquely identified by columns
# e.g. check for single root_id per (timetag, baseline, polarization)
def isunique(df, cols=['timetag', 'baseline', 'polarization']):
	count = set(len(rows) for (name, rows) in df.groupby(cols))
	return count == {1,}

# unwrap the MBD based on the 32 MHz ambiguity in HOPS, choose value closest to SBD
# this old version uses some old column names (instead of the HOPS code defined names)
def unwrap_mbd_old(df, mbd_ambiguity=None):
    if mbd_ambiguity is None:      # we may want to set this manually
        mbd_ambiguity = df.mbd_amb # if alist file does not contain sufficient precision
    offset = np.fmod(df.sbd - df.mbd + 1.5*mbd_ambiguity, mbd_ambiguity)
    df['mbd_unwrap'] = df.sbd - offset + 0.5*mbd_ambiguity

# unwrap the MBD based on the 32 MHz ambiguity, choose value closest to SBD
def unwrap_mbd(df, mbd_ambiguity=None):
    if mbd_ambiguity is None:      # we may want to set this manually
        mbd_ambiguity = df.ambiguity # if alist file does not contain sufficient precision
    offset = np.fmod(df.sbdelay - df.mbdelay + 1.5*mbd_ambiguity, df.ambiguity)
    df['mbd_unwrap'] = df.sbdelay - offset + 0.5*mbd_ambiguity

# rewrap the MBD based on the 32 MHz ambiguity, choose value within +/-ambiguity window
def rewrap_mbd(df):
    mbd_ambiguity = 1. / 32. # us
    df['mbdelay'] = np.fmod(df.mbd_unwrap + 0.5*mbd_ambiguity, mbd_ambiguity) - 0.5*mbd_ambiguity

# add statistical error from fitting straight lines here, note no systematics!
# this is re-derived and close in spirit to the code in fourfit/fill_208.c
# but there are small different factors, not sure what is origin of the fourfit eqns
# add some sytematic errors in quadrature.. (alist precision, linear approx systematics..)
# bw: bw spread in MHz (not in alist..) default: guess based on ambiguity and freq code
# mbd_systematic, rate_systematic: added in quadrature to statistical error (us, ps/s)
def add_delayerr(df, bw=None, mbd_systematic=0.000010, rate_systematic=0.001):
    if bw is None:
        nchan = pd.to_numeric(df.freq_code.str[1:])
        sbw   = 1./pd.to_numeric(df.ambiguity) # bw of single channel
        if df.version.iloc[0] < 6:
            sbw = np.round(sbw) # account for lack of precision in alist v5 (round to MHz)
        bw = nchan * sbw # assume contiguous channels, not always correct
    df['mbd_err'] = np.sqrt(12) / (2*np.pi * df.snr * bw) # us
    df['rate_err'] = 1e6 * np.sqrt(12) / (2*np.pi * df.ref_freq * df.snr * df.duration) # us/s -> ps/s
    df['mbd_err'] = np.sqrt(df['mbd_err']**2 + mbd_systematic**2)
    df['rate_err'] = np.sqrt(df['rate_err']**2 + rate_systematic**2)

# add a path to each alist line for easier finding
def add_path(df):
    df['path'] = ['%s/%s/%s.%.1s.%s.%s' % par for par in zip(df.expt_no, df.scan_id, df.baseline, df.freq_code, df.extent_no, df.root_id)]

# add UNIX time
def add_utime(df):
    df['utime'] = 1e-9*np.array(df.datetime).astype('float')

# add hour if HOPS timetag available
def add_hour(df):
    if 'timetag' in df:
        df['hour'] = df.timetag.apply(lambda x: float(x[4:6]) + float(x[6:8])/60. + float(x[8:10])/3600.)
    elif 'hhmm' in df:
        df['hour'] = df.hhmm.apply(lambda x: float(x[0:2]) + float(x[2:4])/60.)

# day of year tag as integer (from timetag)
def add_doy(df):
    df['doy'] = df.timetag.str[:3].astype(int)

# decimal days since beginning of year = (DOY - 1) + hour/24.
def add_days(df):
    df['days'] = df.timetag.apply(lambda x: float(x[0:3])-1. + float(x[4:6])/24. + float(x[6:8])/1440. + float(x[8:10])/86400.)

# add GMST column to data frame with 'datetime' field
def add_gmst(df):
    from astropy import time
    g = df.groupby('datetime')
    (timestamps, indices) = zip(*g.groups.iteritems())
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

# take calibration output data frame, and make UV dictionary lookup table
def uvdict(filename):
    import hops
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

