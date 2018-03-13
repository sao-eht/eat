from __future__ import print_function
import sys, os, datetime, itertools
import scipy.special as ss
import scipy.optimize as so
import scipy.stats as st 
import pandas as pd
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from eat.io import hops, util
from eat.hops import util as hu
from eat.aips import aips2alist as a2a
from eat.inspect import closures as closures
import statsmodels.stats.stattools as sss
import statsmodels.robust.scale as srs
from sklearn.cluster import KMeans
from astropy import units as u
from astropy.coordinates import EarthLocation, AltAz, ICRS, Angle
from astropy.time import Time, TimeDelta

def MC_CP_dist(amp,sig,bsp_avg=1, N=int(1e5)):
    '''
    calculates cphase kde and circular std
    '''
    V1x = amp[0] + sig[0]*npr.randn(N)
    V1y = sig[0]*npr.randn(N)
    V2x = amp[1] + sig[1]*npr.randn(N)
    V2y = sig[1]*npr.randn(N)
    V3x = amp[2] + sig[2]*npr.randn(N)
    V3y = sig[2]*npr.randn(N)
    V1 = V1x + 1j*V1y
    V2 = V2x + 1j*V2y
    V3 = V3x + 1j*V3y
    bsp = V1*V2*V3
    bsp = bsp.reshape((int(N/bsp_avg),bsp_avg))
    bsp = np.mean(bsp,1)

    cphase = np.angle(bsp)*180/np.pi
    kde = st.gaussian_kde(cphase)
    MCsig = circular_std(cphase)
    return kde, MCsig


def fake_CP_data(amp,sig,bsp_avg=1, N=int(1e5)):

    V1x = amp[0] + sig[0]*npr.randn(N)
    V1y = sig[0]*npr.randn(N)
    V2x = amp[1] + sig[1]*npr.randn(N)
    V2y = sig[1]*npr.randn(N)
    V3x = amp[2] + sig[2]*npr.randn(N)
    V3y = sig[2]*npr.randn(N)
    V1 = V1x + 1j*V1y
    V2 = V2x + 1j*V2y
    V3 = V3x + 1j*V3y
    bsp = V1*V2*V3
    bsp = bsp.reshape((int(N/bsp_avg),bsp_avg))
    bsp = np.mean(bsp,1)
    cphase = np.angle(bsp)*180/np.pi
    return cphase

def fake_LA_data(amp,sig,bsp_avg=1, N=int(1e5)):

    V1x = amp + sig*npr.randn(N)
    V1y = sig*npr.randn(N)
    V1 = V1x + 1j*V1y
    A1 = np.abs(V1)
    LA1 = np.log(A1)
    #bsp = bsp.reshape((int(N/bsp_avg),bsp_avg))
    #bsp = np.mean(bsp,1)
    #cphase = np.angle(bsp)*180/np.pi
    return LA1

#CODES FOR COMPARISON OF STRATEGIES for CPHASEs

#STRATEGY 1
def chi2(real_value,values,errors):
    if len([errors])==1:
        errors = errors*np.ones(len(values))

    chi2 = np.sum(((values-real_value)/errors)**2)/len(values)
    return chi2
