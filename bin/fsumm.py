#!/usr/bin/env python

# fringe summary for a bunch of fringe files

from eat.hops import util as hu
import pandas as pd
import sys

name = []
baseline = []
polarization = []
amp = []
phase = []
snr = []
delay = []
rate = []
freq = []
fedge = []
code = []

for f in sys.argv[1:]:
    p = hu.params(f, quiet=True)
    name.append(p.name)
    baseline.append(p.baseline)
    polarization.append(p.pol)
    amp.append(p.amplitude)
    phase.append(p.resid_phas)
    snr.append(p.snr)
    delay.append(p.delay)
    rate.append(p.rate)
    freq.append(p.ref_freq)
    fedge.append(p.fedge[0])
    code.append(''.join(p.code))

df = pd.DataFrame(dict(
    fname = name,
    bl = baseline,
    pol = polarization,
    amp = amp,
    phase = phase,
    snr = snr,
    delay = delay,
    rate = rate,
    ref_freq = freq,
    f0 = fedge,
    code = code))

print(df.to_string())
