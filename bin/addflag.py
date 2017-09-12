#!/usr/bin/env python
# Add flags to Vincent's master calibration files based on fourfit control file
# 2017.08.29 L. Blackburn

from eat.hops import util as hu

import argparse
import pandas as pd
from StringIO import StringIO
import sys

parser = argparse.ArgumentParser()
parser.add_argument('masterlog', help='master log csv file')
parser.add_argument('controlfile', help='fourfit control file')
args = parser.parse_args()

# the single :30 scan in 2017
scan2timetag = {'101-0102':'101-010230'}

ml = pd.read_csv(args.masterlog, dtype=str)
cf = hu.ControlFile(args.controlfile)

# add in the timetag so we can compare with control file scan times
ml['timetag'] = ml.scan.apply(lambda x: scan2timetag.get(x, x+'00'))

# take scanlog and control file and make start/stop flags
def makeflags(cf, df, station):
    start = []
    stop = []
    for row in df.iterrows():
        a = cf.filter(scan=row[1].timetag, baseline=None, station=station, dropmissing=True).actions()
        if(a.get('skip', False)):
            start.append(0)
            stop.append(0)
        else:
            start.append(a.get('start', ''))
            stop.append(a.get('stop', ''))
    return (start, stop)

for (s, name) in hu.sdict.items():
    (start, stop) = makeflags(cf, ml, s)
    ml[name+'_vf_start'] = start
    ml[name+'_vf_stop'] = stop
    ml[name+'_vf_comment'] = ''

# keep first columns and sort station fields
ml = ml[list(ml.columns[:3]) + sorted(ml.columns[3:])].fillna('')

out = StringIO()
# output new csv file
ml.to_csv(out, index=False)

sys.stdout.write(out.getvalue())


