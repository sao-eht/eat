"""EHT tables for SMA data products
"""
from __future__ import print_function

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

# reference MJD at 01/01/2000 00:00:00.000
mjd2000 = 51544.

# SMT total power table
SWARM_FIELDS = (
    ('scan', int),
    ('source', str),
    ('ut_hours', float),
    ('baseline', str),
    ('sideband', str),
    ('tsys', float),
    ('el', float),
    ('s49_amp', float),
    ('s49_phase', float),
    ('s50_amp', float),
    ('s50_phase', float),
)

swarm_pandasargs = dict(
    delim_whitespace=True,
    comment='#',
    header=None,
    names=[a[0] for a in SWARM_FIELDS],
    dtype=dict(SWARM_FIELDS)
)

def read_swarm(filename):
    table = pd.read_csv(filename, **swarm_pandasargs)
    return table
