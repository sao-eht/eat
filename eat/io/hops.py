# I/O routines for HOPS ASCII tables
# 2016-10-11 Lindy Blackburn

from pkg_resources import parse_version
import pandas as pd
if parse_version(pd.__version__) < parse_version('0.15.1dev'):
    print "pandas version too old and buggy, please update"
import datetime
import numpy as np
import os
import sys

def condense_formats(fmtlist):
    return map(lambda fmt: fmt if fmt.count('%') <= 1 else "%s", fmtlist)

# from write_fsumm.c

fformat_v5 = condense_formats("%1d %s 2 %2d %3d %3d %3d %4d %s %02d%03d-%02d%02d%02d %4d\
 %03d-%02d%02d%02d %3d %-8s %s %c%c %c%02d %2s %4d %6.2f %#5.4g %5.1f %#5.4g %2s %6.3f %8.5f\
 %6.4f %8.3f %4.1f %4.1f %5.1f %5.1f %7.4g %7.4g %06d %02d%02d %8.2f %5.1f %11.8f\
 %13.6f %5.3f %3d %3d\n".replace(' 2 ', ' %d ').strip().split())

fformat_v6 = condense_formats("%1d %s 2 %2d %3d %3d %3d %4d %8s %04d%03d-%02d%02d%02d\
 %4d %03d-%02d%02d%02d %3d %32s %2s %c%c\
 %c%02d %2s %5d\
 %#13.8g %#13.8g %11.6f %#11.6g %2s\
 %+12.9f %+12.9f %11.9f\
 %+11.6f %5.2f %5.2f %6.2f %6.2f %7.4g %7.4g %06d\
 %02d%02d %9.3f %10.6f %11.8f\
 %13.6f %+9.6f %8d %8d %+10.6f %+10.6f %+13.10f\n".replace(' 2 ', ' %d ').strip().split())

tformat_v6 = condense_formats("%1d %4d 3 %8s %4d %03d-%02d%02d%02d %3d %32s\
 %c%c %4d %3s %20s %11s %14s\
 %3d %3d  %c  %c %06d\
 %10.3f %8.3f %+8.3f %2s %7.4f\
 %8.5f %6.4f %9.5f %14s %11s\
  %02d%02d %10.3f %7d\n".replace(' 3 ', ' %d ').strip().split())

ffields_v5 = [a.strip() for a in """
version,
root_id,
two,
extent_no,
duration,
length,
offset,
expt_no,
scan_id,
procdate,
year,
timetag,
scan_offset,
source,
baseline,
quality,
freq_code,
polarization,
lags,
amp,
snr,
resid_phas,
phase_snr,
datatype,
sbdelay,
mbdelay,
ambiguity,
delay_rate,
ref_elev,
rem_elev,
ref_az,
rem_az,
u,
v,
esdesp,
epoch,
ref_freq,
total_phas,
total_rate,
total_mbdelay,
total_sbresid,
srch_cotime,
noloss_cotime
""".split(',')]

ffields_v6 = [a.strip() for a in """
version,
root_id,
two,
extent_no,
duration,
length,
offset,
expt_no,
scan_id,
procdate,
year,
timetag,
scan_offset,
source,
baseline,
quality,
freq_code,
polarization,
lags,
amp,
snr,
resid_phas,
phase_snr,
datatype,
sbdelay,
mbdelay,
ambiguity,
delay_rate,
ref_elev,
rem_elev,
ref_az,
rem_az,
u,
v,
esdesp,
epoch,
ref_freq,
total_phas,
total_rate,
total_mbdelay,
total_sbresid,
srch_cotime,
noloss_cotime,
ra_hrs,
dec_deg,
resid_delay
""".split(',')]

tfields_v6 = [a.strip() for a in """
version,
expt_no,
three,
scan_id,
year,
timetag,
scan_offset,
source,
freq_code,
lags,
triangle,
roots,
extents,
lengths,
duration,
offset,
scanqual,
dataqual,
esdesp,
bis_amp,
bis_snr,
bis_phas,
datatype,
csbdelay,
cmbdelay,
ambiguity,
cdelay_rate,
elevations,
azimuths,
epoch,
ref_freq,
cotime
""".split(',')]

fsumm_v5_pandasargs = dict(
    delim_whitespace=True,
    comment='*',
    header=None,
    # fringe quality code
    dtype={15:str},
    parse_dates={'datetime':[10,11]},
    # index_col='datetime',
    keep_date_col=True,
    # note: pandas 0.15.1 cannot use generator for date_parser (unlike 0.18), so changed to a list comprehension
    date_parser=lambda years,times: [datetime.datetime.strptime(x+y, '%y%j-%H%M%S') for (x,y) in zip(years,times)],
    names=ffields_v5,
)

fsumm_v6_pandasargs = dict(
    delim_whitespace=True,
    comment='*',
    header=None,
    dtype={15:str},
    parse_dates={'datetime':[10,11]},
    # index_col='datetime',
    keep_date_col=True,
    # note: pandas 0.15.1 cannot use generator for date_parser (unlike 0.18), so changed to a list comprehension
    date_parser=lambda years,times: [datetime.datetime.strptime(x+y, '%Y%j-%H%M%S') for (x,y) in zip(years,times)],
    names=ffields_v6,
)

tsumm_pandasargs = dict(
    delim_whitespace=True,
    comment='*',
    header=None,
    dtype={16:str, 17:str},
    parse_dates={'datetime':[4,5]},
    # index_col='datetime',
    keep_date_col=True,
    # note: pandas 0.15.1 cannot use generator for date_parser (unlike 0.18), so changed to a list comprehension
    date_parser=lambda years,times: [datetime.datetime.strptime(x+y, '%Y%j-%H%M%S') for (x,y) in zip(years,times)],
)

def read_alist_v5(filename):
    table = pd.read_csv(filename, **fsumm_v5_pandasargs)
    return table

def read_alist_v6(filename):
    table = pd.read_csv(filename, **fsumm_v6_pandasargs)
    return table

def read_tlist_v6(filename):
    table = pd.read_csv(filename, names=tfields_v6, **tsumm_pandasargs)
    return table

fformatters_v5 = {col:lambda x, fmt=fmt: fmt % x for col,fmt in zip(ffields_v5, fformat_v5)}
fformatters_v5['year'] = lambda x: '%s' % x # necessary because pandas parse_dates will keep date col as str (!?)
fformatters_v6 = {col:lambda x, fmt=fmt: fmt % x for col,fmt in zip(ffields_v6, fformat_v6)}
fformatters_v6['year'] = lambda x: '%s' % x # necessary because pandas parse_dates will keep date col as str (!?)
tformatters_v6 = {col:lambda x, fmt=fmt: fmt % x for col,fmt in zip(tfields_v6, tformat_v6)}
tformatters_v6['year'] = lambda x: '%s' % x # necessary because pandas parse_dates will keep date col as str (!?)

def write_alist_v5(df, out=sys.stdout):
    if type(out) is str:
        out = open(out, 'w')
    df.to_string(buf=out, columns=ffields_v5, formatters=fformatters_v5, header=False, index=False)

def write_alist_v6(df, out=sys.stdout):
    if type(out) is str:
        out = open(out, 'w')
    df.to_string(buf=out, columns=ffields_v6, formatters=fformatters_v6, header=False, index=False)

def write_tlist_v6(df, out=sys.stdout):
    if type(out) is str:
        out = open(out, 'w')
    df.to_string(buf=out, columns=tfields_v6, formatters=tformatters_v6, header=False, index=False)

def get_alist_version(filename):
    code = (a[0] for a in open(filename) if a[0].isdigit())
    return int(code.next())

# read_alist automatically determine version
# ALIST notes:
# mk4 correlated data will have az,el = 0
# difx correlated data will have az,el != 0, but may still have u,v = 0
def read_alist(filename):
    ver = get_alist_version(filename)
    if ver == 5:
        table = read_alist_v5(filename)
    elif ver == 6:
        table = read_alist_v6(filename)
    else:
        import sys
        sys.exit('alist is not version 5 or 6')
    return table

# master calibration file from vincent
MASTERCAL_FIELDS = (
    ('dd_hhmm', str), # timetag without seconds
    ('source', str),  # radio source name
    ('len', int),     # length of observation in seconds
    ('el_ap', int),   # Ap APEX -- elevation --
    ('el_az', int),   # Az SMTL
    ('el_sm', int),   # Sm JCMTR
    ('el_cm', int),   # Cm CARMA1L
    ('el_pl', int),   # PL PICOL
    ('el_pb', int),   # PB PDBL
    ('pa_ap', int),   # Ap -- parallactic angle --
    ('pa_az', int),   # Az
    ('pa_sm', int),   # Sm
    ('pa_cm', int),   # Cm
    ('pa_pl', int),   # PL
    ('pa_pb', int),   # PB
    ('smt_tsysl', float),    # -- [S/T] SMT --
    ('smt_tsysr', float),
    ('smt_tau', float),
    ('smalow_sefd', float),  # -- [P/Q] SMA low --
    ('smalow_pheff', float),
    ('smahigh_sefd', float), # -- [P/Q] SMA high --
    ('smahigh_pheff', float),
    ('d_sefd', float),       # -- [D] CARMA Ref LCP --
    ('e_sefd', float),       # -- [E] CARMA Ref RCP --
    ('flow_sefd', float),    # -- [F] CARMA Phased LCP low --
    ('fhigh_sefd', float),   # -- [F] CARMA Phased LCP high --
    ('glow_sefd', float),    # -- [G] CARMA Phased RCP low --
    ('ghigh_sefd', float),   # -- [G] CARMA Phased RCP high --
    ('carma_tau', float),    # -- CARMA --
    ('carma_path', float),
    ('carma_phef', float),
    ('jcmt_tsys', float),    # -- [-/J] JCMT --
    ('jcmt_tau', float),
    ('jcmt_gap', float),
)

mastercal_pandasargs = dict(
    delim_whitespace=True,
    comment='#',
    header=None,
    parse_dates={'datetime':[0]},
    # index_col='datetime',
    keep_date_col=True,
    date_parser=lambda dates: [datetime.datetime.strptime('13'+y, '%y%j-%H%M') for y in dates],
    names=[a[0] for a in MASTERCAL_FIELDS]
)

def read_mastercal(filename):
    table = pd.read_csv(filename, **mastercal_pandasargs)
    return table

# calibrated data table, after calibration amplitdues from alist
CALTABLE_FIELDS = (
    ('day', str), # day
    ('hhmm', str),
    ('source', str),
    ('baseline', str),
    ('u', float),
    ('v', float),
    ('uvmag', float),
    ('sefd_1', float),
    ('sefd_2', float),
    ('calamp', float),
    ('amp', float),
    ('snr', float),
    ('el_1', float),
    ('el_2', float),
    ('expt', int),
    ('band', int),
)

caltable_pandasargs = dict(
    delim_whitespace=True,
    comment='#',
    header=None,
    parse_dates={'datetime':[0,1]},
    keep_date_col=True,
    date_parser=lambda dates,times: [x+y for (x,y) in zip(dates,times)],
    names=[a[0] for a in CALTABLE_FIELDS],
    error_bad_lines=False,
    warn_bad_lines=False,
)

# calibrated data will have u,v filled in
def read_caltable(filename):
    table = pd.read_csv(filename, **caltable_pandasargs)
    # keep missing data (u,v coords still good)
    # table.dropna(how="any", inplace=True)
    return table

# network solution calibrated data 2013 by Michael
# http://eht-wiki.haystack.mit.edu/Event_Horizon_Telescope_Home/EHT_Data/2013_March/Network_Calibration_Solution/Non-Sgr_A*_Network_Solution
# 1: Day
# 2: HourMinute
# 3: Source
# 4: Baseline (note: "p" and "q" are high-band data for the SMA. I use lowercase because the high-band gains at the SMA are not equal to the low-band gains because they have different phasing efficiencies)
# 5-7: u, v, |{u,v}|
# 8: SEFD_1
# 9: SEFD_2
# 10: a-priori visibility amplitude (from cal_2013)
# 11: visibility amplitude in correlator units
# 12: SNR of visibility amplitude, after incoherent averaging (the thermal noise)
# 13: Elevation_1
# 14: Elevation_2
# 15: Experiment Code
# 16: 0=Low-Band, 1=High-Band
# 17: Gain_1
# 18: Formal Uncertainty in Gain_1 from the network solution
# 19: Gain_2
# 20: Formal Uncertainty in Gain_2 from the network solution
# 21: chi^2 of the visibility after using the network solution (i.e., departure from self-consistent solution, in units of \sigma, squared)
# 22. Calibrated Visibility Amplitude (Jy)
# 23. Estimated systematic uncertainty, as a fraction of the Calibrated Visibility Amplitude (#2). 
NETWORKSOL_FIELDS = (
    ('day', str), # day
    ('hhmm', str),
    ('source', str),
    ('baseline', str),
    ('u', float),
    ('v', float),
    ('uvmag', float),
    ('sefd_1', float),
    ('sefd_2', float),
    ('calamp_apriori', float),
    ('amp', float),
    ('snr', float),
    ('el_1', float),
    ('el_2', float),
    ('expt', int),
    ('band', int),
    ('gain_1', float),
    ('gainerr_1', float),
    ('gain_2', float),
    ('gainerr_2', float),
    ('chisq', float),
    ('calamp_network', float),
#    ('syserr_fraction', float),
)

networksol_pandasargs = dict(
    delim_whitespace=True,
    comment='#',
    header=None,
    parse_dates={'datetime':[0,1]},
    keep_date_col=True,
    # date_parser=lambda dates,times: [x+y for (x,y) in zip(dates,times)],
    date_parser=lambda day,hhmm: [datetime.datetime.strptime(
        '13' + '%03d' % int(x) + '%04d' % int(y), '%y%j%H%M') for (x,y) in zip(day,hhmm)],
    names=[a[0] for a in NETWORKSOL_FIELDS],
    error_bad_lines=True,
    warn_bad_lines=True,
)

def read_networksol(filename):
    table = pd.read_csv(filename, **networksol_pandasargs)
    table.dropna(how="any", inplace=True)
    return table

# modified networksol by Michael with model visibs and flag
NETWORKSOL2_FIELDS = (
    ('day', str), # day
    ('hhmm', str),
    ('source', str),
    ('baseline', str),
    ('u', float),
    ('v', float),
    ('uvmag', float),
    ('sefd_1', float),
    ('sefd_2', float),
    ('calamp_apriori', float),
    ('amp', float),
    ('snr', float),
    ('el_1', float),
    ('el_2', float),
    ('expt', int),
    ('band', int),
	('model', float),
    ('gain_1', float),
    ('gainerr_1', float),
    ('gain_2', float),
    ('gainerr_2', float),
    ('chisq', float),
    ('calamp_network', float),
	('flag', int),
#    ('syserr_fraction', float),
)

networksol2_pandasargs = dict(
    delim_whitespace=True,
    comment='#',
    header=None,
    parse_dates={'datetime':[0,1]},
    keep_date_col=True,
    date_parser=lambda dates,times: [x+y for (x,y) in zip(dates,times)],
    names=[a[0] for a in NETWORKSOL2_FIELDS],
    error_bad_lines=True,
    warn_bad_lines=True,
)

def read_networksol2(filename):
    table = pd.read_csv(filename, **networksol2_pandasargs)
    table.dropna(how="any", inplace=True)
    return table


# aedit output produced by Vincent with channel-specific amplitudes
# 30 channel mode
#   1. path to fringe file (including experiment number, doy-hhmmss, baseline)
#   2. source
#   3. number of channels
#   4. amplitude (times 10^{-4}) [technically, correlation coefficient]
#   5. signal-to-noise ratio
#   6-35. amplitudes in each of the channels
BANDPASS_FIELDS = [
    ('path', str),
    ('source', str),
    ('nchan', str),
    ('amp', float),
    ('snr', float),
]

bandpass_pandasargs = dict(
    delim_whitespace=True,
    comment='*',
    header=None,
)

def read_bandpass(filename):
    table = pd.read_csv(filename, **bandpass_pandasargs)
    table.columns = [a[0] for a in BANDPASS_FIELDS] + \
        ['amp_%d' % (i+1) for i in range(len(table.columns) - 5)]
    table['experiment'], table['scan'], table['filename'] = \
        zip(*table['path'].apply(lambda x: x.split('/')))
    table['baseline'] = table['filename'].apply(lambda x: x[:2])
    return table

