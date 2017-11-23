#!/usr/bin/env python

import sys
import pandas as pd
import numpy  as np

def unwrap(phi):
    for i in range(len(phi)-1):
        dphi = phi[i+1] - phi[i]
        if dphi > 180:
            phi[i+1:] -= 360
        elif dphi < -180:
            phi[i+1:] += 360
    return phi    

# Read stdin to a Pandas DataFrame
df = pd.read_table(sys.stdin, header=None, sep='\s+')

for i, r in df.iterrows():
    uw = unwrap(r.values[1:])
    y1 = uw[-4]
    y2 = uw[-3]
    y3 = uw[-2]
    
    a  =       y1
    b  = (-3.0*y1 + 4.0*y2 - y3) / 2.0
    c  = (     y1 - 2.0*y2 + y3) / 2.0
    uw[-1] = a + 3.0*b + 9.0*c
    
    df.iloc[i,1:] = uw

df.to_csv(sys.stdout, index=False, header=False, sep=" ", float_format="%.9g")
