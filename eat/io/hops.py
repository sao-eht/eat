from pkg_resources import parse_version
import pandas as pd
if parse_version(pd.__version__) < parse_version('0.15.1dev'):
    print "pandas version too old and buggy, please update"
import datetime
import numpy as np
import os

# Taken from vlbidata.alist
# Taken from Haystack's aformat.doc
#       = ( (FIELD_NAME, TYPE_FUNC),
#           ...
#           )
ALIST_FIELDS = (
    ('format', int),  # >=5 implies Mk4
    ('root', str),  # 6-char lower case
    ('type', int),  # General format for fringe data
    ('fileset', int),  # Part of filename
    ('duration', int),  # Nominal duration (sec)
    ('length', int),  # Seconds of data represented.
    ('offset', int),  # Offset (sec) of mean time of data
    ('experiment', int),  # Part of filename
    ('scan', str),  # From obsvex, blanks trimmed
    ('procdate', str),  # FRNGE/fourfit processing time
    ('year', int),  # Four-digit year.
    ('timetag', str),  # Nominal start time of data record
    ('scan_offset', int),  # Scan time to time tag (sec)
    ('source', str),  # Blank-padded ascii string
    ('baseline', str),  # 2-char baseline id
    ('errcode', str),  # 2-char qcode and errcode
    ('band', str),  # e.g. X08 for X-band 8 freqs
    ('pol', str),  # RR, LL, RL, LR
    ('lags', int),  # Number of lags in correlation
    ('amp', float),  # In units of 1/10000
    ('snr', float),  # 4 significant digits
    ('phase_deg', float),  # Can be of various type
    ('phase_snr', float),  # 4 significant digits
    ('data_type', str  ),  # First char is data origin
    ('sbd', float),  # Microseconds
    ('mbd', float),  # Microseconds
    ('mbd_amb', float),  # Microseconds
    ('rate', float),  # Picoseconds/second DRATE
    ('ref_el', float),  # At reference epoch, degrees
    ('rem_el', float),  # At reference epoch, degrees
    ('ref_az', float),  # At reference epoch, degrees
    ('rem_az', float),  # At reference epoch, degrees
    ('u', float),  # precision 4 sig. digits
    ('v', float),  # precision 4 sig. digits
    ('ESDESP', str),  # E=ref.tape error rate exponent:
    ('epoch', str),  # mmss
    ('freq', float),  # Precision 10 KHz [REF_FREQ]
    ('ecphase', float),  # Regardless of field 21 [TPHAS]
    ('drate', float),  # At ref epoch, microsec/sec [TOTDRATE]
    ('total_mbd', float),  # At ref epoch, microsec [TOTMBDELAY]
    ('total_sbdmbd', float),  # At ref epoch, microsec [TOTSBDMMBD]
    ('scotime', int),  # Seconds [COHTIMES]
    ('ncotime', int),  # Seconds
    )

alist_pandasargs = dict(
    delim_whitespace=True,
    comment='*',
    header=None,
    dtype={15:str},
    parse_dates={'datetime':[10,11]},
    # index_col='datetime',
    keep_date_col=True,
    # note: pandas 0.15.1 cannot use generator for date_parser (unlike 0.18), so changed to a list comprehension
    date_parser=lambda years,times: [datetime.datetime.strptime(x+y, '%y%j-%H%M%S') for (x,y) in zip(years,times)],
    names=[a[0] for a in ALIST_FIELDS]
)

# ALIST notes:
# mk4 correlated data will have az,el = 0
# difx correlated data will have az,el != 0, but may still have u,v = 0
def read_alist(filename):
    table = pd.read_csv(filename, **alist_pandasargs)
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
    table.dropna(how="any", inplace=True)
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
    date_parser=lambda dates,times: [x+y for (x,y) in zip(dates,times)],
    names=[a[0] for a in NETWORKSOL_FIELDS],
    error_bad_lines=True,
    warn_bad_lines=True,
)

def read_networksol(filename):
    table = pd.read_csv(filename, **networksol_pandasargs)
    table.dropna(how="any", inplace=True)
    return table

