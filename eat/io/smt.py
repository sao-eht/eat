"""EHT tables
"""
from __future__ import division
from __future__ import print_function

from future import standard_library
standard_library.install_aliases()
from builtins import zip
from builtins import map
from pkg_resources import parse_version
import pandas as pd
try:
    assert(parse_version(pd.__version__) >= parse_version('0.15.1dev'))
except:
    print("pandas version too old")
import datetime
import numpy as np
import os

# reference MJD at 01/01/2000 00:00:00.000
mjd2000 = 51544.

# SMT total power table
TOTALPOWER_FIELDS = (
    ('mjd', float),
    ('state', str), # "Idle"
    ('n', int), # some number that counts up
    ('v1', float), # voltages, eventaully we want to figure out what LSB/USB/L/R these are
    ('v2', float),
    ('v3', float),
    ('v4', float),
    ('v5', float),
    ('v6', float),
    ('v7', float),
    ('v8', float),
)

totalpower_pandasargs = dict(
    delim_whitespace=True,
    comment='#',
    header=None,
    parse_dates={'datetime':[0]},
    keep_date_col=True,
    date_parser=lambda dates: [datetime.datetime(2000, 1, 1) + datetime.timedelta(days=float(x)-mjd2000) for x in dates],
    names=[a[0] for a in TOTALPOWER_FIELDS],
    usecols="mjd state n v3 v7".split(),
    dtype=dict(TOTALPOWER_FIELDS)
)

def read_totalpower(filename):
    table = pd.read_csv(filename, **totalpower_pandasargs)
    return table

TAU_FIELDS = (
    ('mjd', float),
    ('tau', float),
    ('label_pm', str), # +/- symbol
    ('tau_err', float),
    ('label_zero', float), # always zero?
)

tau_pandasargs = dict(
    delim_whitespace=True,
    comment='#',
    header=None,
    parse_dates={'datetime':[0]},
    keep_date_col=True,
    date_parser=lambda dates: [datetime.datetime(2000, 1, 1) + datetime.timedelta(days=float(x)-mjd2000) for x in dates],
    names=[a[0] for a in TAU_FIELDS],
    usecols=[a[0] for a in TAU_FIELDS if "label" not in a[0]],
    dtype=dict(TAU_FIELDS)
)

def read_tau(filename):
    table = pd.read_csv(filename, **tau_pandasargs)
    return table

# SMT tsys table
TSYS_FIELDS = (
    ('label_scan', str),
    ('scan', int),
    ('yyyymmdd', str),
    ('hhmm', str),
    ('label_channel', str),
    ('channel', int),
    ('source', str),
    ('control_key', str),
    ('label_tcal', str),
    ('tcal', float),
    ('label_tsys', str),
    ('tsys', float),
    ('label_pwv', str),
    ('pwv', float),
)

tsys_pandasargs = dict(
    delim_whitespace=True,
    comment='#',
    header=None,
    parse_dates={'datetime':[1,2]}, # after ignoring the label columns
    keep_date_col=True,
    date_parser=lambda dates,times: [datetime.datetime.strptime(x+y, '%Y-%m-%d%H:%M') for (x,y) in zip(dates,times)],
    names=[a[0] for a in TSYS_FIELDS],
    usecols=[a[0] for a in TSYS_FIELDS if "label" not in a[0]],
    dtype=dict(TSYS_FIELDS)
)

def read_tsys(filename):
    table = pd.read_csv(filename, **tsysdata_kwargs)
    table = table[(table.tsys > 50) & (table.tsys < 2000)]
    return table

# get scan times from unipops sdd text file
def sddscantimes(filename):
    from collections import defaultdict
    out = defaultdict(list)
    a = open(filename, 'r')
    for line in a:
        if (line[0:2] != 'sc') and (line[0:2] != 'ut'):
            continue
        tok = line.strip().split()
        if tok[0] == "scan":
            tok[1] = tok[1][:-4]
            (scan, chan) = list(map(int, tok[1].split('.')))
        elif tok[0] == "utdate":
            utdate = datetime.datetime.strptime(tok[1], "%Y.%m%d00")
        elif tok[0] == "ut":
            hours = float(tok[1])
            out['scan'].append(scan)
            out['channel'].append(chan)
            out['datetime'].append(utdate + datetime.timedelta(hours=hours))
    return pd.DataFrame(out)

# from unipops sdd text file, extract certain parameters from class and put them in a big table
# column def will be {Class ID}_{field}
# the index will have Class ID = 0
def sddfile(filename,
            cols='c1_scan c1_object c1_obsmode c3_utdate c3_ut c5_tamb c12_tcal c12_stsys c12_rtsys c12_tauh2o'.split()):
    import io
    out = []
    f = open(filename)
    classno = 'B' # classno of bootstrap
    fields = dict()
    for line in f:
        tok = line.strip().split()
        # starting new row
        if line.startswith('directory'):
            if classno is not 'B':
                out.append(",".join((fields[col] for col in cols))) # print out previous scan
            classno = 'I'   # classno of directory fields
            fields = dict() # new set of fields
        if line.startswith('Class'):
            (ignore, classno) = line.split() # set classno
        elif len(tok) == 2:
            fields['c%s_%s' % (classno, tok[0])] = tok[1]
    # last entry
    out.append(",".join((fields[col] for col in cols))) # print out previous scan
    table = pd.read_csv(io.StringIO("\n".join(out)), names=cols, header=None, dtype={'c3_utdate':str})
    if 'c1_scan' in cols:
        table['scan'] = [int(0.5+scan) for scan in table.c1_scan]
        table['channel'] = [int(0.5+100*(scan - int(scan))) for scan in table.c1_scan]
    if 'c3_utdate' in cols and 'c3_ut' in cols:
        table['datetime'] = [datetime.datetime.strptime(utdate, "%Y.%m%d00") + datetime.timedelta(hours=hours) for (utdate, hours) in table[['c3_utdate', 'c3_ut']].itertuples(index=False)]
        table['mjd'] = [(datetime.datetime.strptime(utdate, "%Y.%m%d00") - datetime.datetime(2000,1,1)).days + mjd2000 + hours/24. for (utdate, hours) in table[['c3_utdate', 'c3_ut']].itertuples(index=False)]
    return table

def tpreduce(tp):
# possible states in tp file
#              v1   2   3   4   5   6   7   8
# voltages are ch1 ch1 ch2 ch2 ch3 ch3 ch4 ch4
# and we care about the first number: e.g. v3 v7 for ch2&4
# tpreduce will isolate each "state" and run a given function (min, max, average, etc) on the voltages depending on the state
# we probably want this scan-specific, but lining up the scan info with tp is a little messy
#    {'BSP',
#     'CAL',
#     'COLD',
#     'FIVE',
#     'FOC',
#     'Idle',
#     'NSFC',
#     'PCAL',
#     'PS',
#     'QK5',
#     'SEQ',
#     'STIP',
#     'VLBI'}
    # tp['statechange'] = np.cumsum(np.hstack([1, np.diff(tp.state.map(hash)) != 0]))
    # tp['statechange'] = np.cumsum(~(tp.state == tp.state.shift(1)))
    from collections import defaultdict, OrderedDict
    g = tp.groupby(np.cumsum(~(tp.state == tp.state.shift(1))))
    def vagg(tpgroup):
        out = OrderedDict()
        out['datetime_start'] = tpgroup.datetime.iloc[0]
        out['datetime_stop'] = tpgroup.datetime.iloc[-1]
        out['datetime'] = out['datetime_start'] + (out['datetime_stop'] - out['datetime_start'])/2.
        out['state'] = tpgroup.state.iloc[0]
        out['n'] = tpgroup.n.iloc[0]
        fmap = defaultdict(lambda: np.mean)
        fmap.update(CAL=np.max, COLD=np.min)
        for key in "v1 v3 v5 v7".split():
            if key in tp.columns:
                out[key] = fmap[out['state']](tpgroup[key])
        return pd.Series(out)
    h = g.apply(vagg)
    return h

# get segments based on column based on repeated values and timestamps
def group(df, col):
    from collections import defaultdict, OrderedDict
    g = df.groupby(np.cumsum(~(df[col] == df[col].shift(1))))
    out = OrderedDict()
    def vagg(subgroup):
        out = OrderedDict()
        out['datetime_start'] = subgroup.datetime.iloc[0]
        out['datetime_stop'] = subgroup.datetime.iloc[-1]
        out['datetime'] = out['datetime_start'] + (out['datetime_stop'] - out['datetime_start'])/2.
        out[col] = subgroup[col].iloc[0]
        # Series does not work here because cannot prevent pandas from converting str columns to Timestamp
        # gets a multiindex -- probably can reduce somehow but oh well
        # but.. iterrows() will NOT preserve the dtype across rows wtf pandas
        return pd.DataFrame(out, index=subgroup.index[:1])
    h = g.apply(vagg)
    return h

# get a date range from current x limits in matplotlib
def xlim2range(ax=None):
    if ax is None:
        import matplotlib.dates
        import matplotlib.pyplot as plt
        ax = plt.gca()
    x0, x1 = ax.get_xlim()
    d0 = matplotlib.dates.num2date(x0)
    d1 = matplotlib.dates.num2date(x1)
    return d0, d1

def subset(df):
    d0, d1 = xlim2range()
    return df[(df.datetime >= d0) & (df.datetime <= d1)]

# trec=100 seems to be default, 284.8 is Kazu's number for average chopper temp
def tp2tsys(tp, trec=100., thot=284.8, tcal3=284., tcal7=284.):
    a = subset(tp) # get subset that overlaps with plot
    (v3s, v3h) = (np.min(a.v3), np.max(a.v3))
    (v7s, v7h) = (np.min(a.v7), np.max(a.v7))
    tsys3 = tcal3 * (v3s / (v3h - v3s))
    tsys7 = tcal7 * (v7s / (v7h - v7s))
    return tsys3, tsys7
