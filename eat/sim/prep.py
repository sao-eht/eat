# 12/9/2015 LLB
# 1/12/2016 LLB - bug fix in thermal and phase noise

import sys
import os
import pandas as pd
import numpy as np
import StringIO
from argparse import Namespace

if len(sys.argv) < 2 or not os.path.exists(sys.argv[1]):
    print "use: python prep.py visibs_true.txt > visibs_measured.txt"
    sys.exit()

# note SPT345 SEFD is completely made up
sitesstring = \
"""
n  site sefd230 sefd345
1   LMT     560    1370
2   SMA    4900    8100
3   SMT   11900   23100
4  ALMA     110     140
5   SPT    7300   15000
"""
sites = pd.read_csv(StringIO.StringIO(sitesstring), delim_whitespace=True, index_col='n')

# Michael's synthetic true visibilities table
VISIBS_FIELDS = (
    ('hour', float),
    ('site1', int),
    ('site2', int),
    ('elev1', float),
    ('elev2', float),
    ('u', float),
    ('v', float),
    ('re', float),
    ('im', float),
)

visibs_kwargs = dict(
    delim_whitespace=True,
    comment='#',
    header=None,
    names=[a[0] for a in VISIBS_FIELDS],
    dtype=dict(VISIBS_FIELDS)
)

def read_visibs(filename):
    table = pd.read_csv(filename, **visibs_kwargs)
    return table

# read in all visibs tables
visibs = read_visibs(sys.argv[1])
visibs = visibs[visibs.hour > 0] # get rid of total flux line

# set the seed according to a collection of arguments and return random gaussian var
def hashrandn(*args):
    np.random.seed(hash(",".join(map(repr,args))) % 4294967295)
    return np.random.randn()

# set the seed according to a collection of arguments and return random number in 0,1
def hashrand(*args):
    np.random.seed(hash(",".join(map(repr,args))) % 4294967295)
    return np.random.rand()

# randomize thermal + systematic uncertainties for each scan and create "noisy" calibrated complex visibs
def populate(row):
    # string representation of site
    site1 = sites.loc[int(row.site1)]
    site2 = sites.loc[int(row.site2)]
    # opacity = 0.1 + 10% time-dependent
    tau1 = 0.1 + 0.01 * hashrandn(site1, 'tau', row.hour)
    tau2 = 0.1 + 0.01 * hashrandn(site2, 'tau', row.hour)
    # amplitude gain (forward gain on power) = 1.0 + 10% fixed + 10% time-dependent
    gain1 = 1.0 + 0.1 * hashrandn(site1, 'gain') + 0.1 * hashrandn(site1, 'gain', row.hour)
    gain2 = 1.0 + 0.1 * hashrandn(site2, 'gain') + 0.1 * hashrandn(site2, 'gain', row.hour)
    # average SEFD
    sefd1 = site1.sefd230 
    sefd2 = site2.sefd230
    # include opacity attenuation for the true SEFD (to derive true SNR)
    sefd1true = sefd1 * np.exp(tau1/np.sin(np.deg2rad(row.elev1))) / gain1
    sefd2true = sefd2 * np.exp(tau2/np.sin(np.deg2rad(row.elev2))) / gain2
    # assumed (noisy) SEFD based on 0.1 estimated opacity (used for calibration)
    sefd1cal = sefd1 * np.exp(0.1/np.sin(np.deg2rad(row.elev1)))
    sefd2cal = sefd2 * np.exp(0.1/np.sin(np.deg2rad(row.elev2)))
    # geometric mean for baseline
    sefdtrue = np.sqrt(sefd1true * sefd2true) # SEFD the instrument actually has
    sefdcal = np.sqrt(sefd1cal * sefd2cal)    # SEFD we think the instrument has
    noisetrue = sefdtrue / np.sqrt(2 * 30 * 0.82 * 2e9) # true RMS noise in Jy 30s INTTIME
    snr = np.sqrt(row.re**2 + row.im**2) / noisetrue # true SNR of perfect signal
    noisecal = noisetrue * (sefdcal / sefdtrue) # RMS noise in miscalibrated visibility
    # site-dependent time-dependent random phase
    phase1 = 2 * np.pi * hashrand(site1, 'phase', row.hour)
    phase2 = 2 * np.pi * hashrand(site2, 'phase', row.hour)
    # true (normalized) visibility w/thermal noise and including random phase
    vtrue = ((row.re + 1j * row.im) + (np.random.randn(2).view('complex')[0] * noisetrue)) \
        * np.exp(1j * (phase2 - phase1))
    # reported visibility including any "miscalibration" due to unknown factors
    vcal = vtrue * sefdcal / sefdtrue
    return pd.Series(dict(sefdtrue=sefdtrue, sefdcal=sefdcal, recal=vcal.real, imcal=vcal.imag, noisecal=noisecal))

# miscalibrate visibs
q = visibs.apply(populate, axis=1)

visibs['re'] = q['recal']
visibs['im'] = q['imcal']
visibs['sigma'] = q['noisecal']

sys.stdout.write("# ")
out = visibs.to_csv(index=False, sep=" ")
sys.stdout.write(out)
