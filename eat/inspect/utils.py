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
from eat.inspect import closures as cl
import statsmodels.stats.stattools as sss
import statsmodels.robust.scale as srs
from sklearn.cluster import KMeans
from astropy import units as u
from astropy.coordinates import EarthLocation, AltAz, ICRS, Angle
from astropy.time import Time, TimeDelta

#BUNCH OF FUNCTIONS FOR GENERAL DESCRIPTION OF SCAN

def circular_mean(theta):
    '''
    theta in deg
    '''
    theta = np.asarray(theta, dtype=np.float32)*np.pi/180.
    theta = theta[theta==theta]
    if len(theta)==0:
        return None
    else:
        C = np.mean(np.cos(theta))
        S = np.mean(np.sin(theta))
        mt = np.arctan2(S,C)*180./np.pi
        return np.mod(mt,360)

def unb_amp_no_trend(t_A): 
    #t_A = list(zip(fooLoc.mjd,fooLoc.amp))
    time = np.asarray([x[0] for x in t_A])
    time = time - np.mean(time)
    amp = np.asarray([x[1] for x in t_A])
    #removing linear trend, but not the mean value
    amp = amp-np.polyfit(time,amp,1)[0]*time
    amp2 = np.asarray(amp, dtype=np.float32)**2
    m = np.mean(amp2)
    s = np.std(amp2)
    delta = m**2 - s**2
    if delta >=0:
        a0 = delta**(0.25)
        #s0 = np.sqrt((m -np.sqrt(delta))/2.)
    else:
        a0 = np.nan
        #s0 = np.nan
    return a0

def unb_std_no_trend(t_A): 
    #t_A = list(zip(fooLoc.mjd,fooLoc.amp))
    time = np.asarray([x[0] for x in t_A])
    time = time - np.mean(time)
    amp = np.asarray([x[1] for x in t_A])
    #removing linear trend, but not the mean value
    amp = amp-np.polyfit(time,amp,1)[0]*time
    amp2 = np.asarray(amp, dtype=np.float32)**2
    m = np.mean(amp2)
    s = np.std(amp2)
    delta = m**2 - s**2
    if delta >=0:
        #a0 = delta**(0.25)
        s0 = np.sqrt((m -np.sqrt(delta))/2.)
    else:
        #a0 = np.nan
        s0 = np.nan
    return s0

def cut_outliers_circ(vector,no_sigmas):
    #cuts outliers that are in distance from mean value larger than no_sigmas
    vector = vector[vector==vector]
    sigma = circular_std(vector)
    m = circular_mean(vector)
    dist = np.minimum(np.abs(vector - m), np.abs(360. - np.abs(vector - m)))
    vector = vector[dist < no_sigmas*sigma]
    return vector

def circ_cut_and_mean(vector,no_sigmas):
    vector = vector[vector==vector]
    vector = cut_outliers_circ(vector,no_sigmas)
    return circular_mean(vector)

def circ_cut_and_std(vector,no_sigmas):
    vector = vector[vector==vector]
    vector = cut_outliers_circ(vector,no_sigmas)
    return circular_std(vector)

def circular_std(theta):
    theta = theta[theta==theta]
    theta = np.asarray(theta, dtype=np.float32)*np.pi/180.
    C = np.mean(np.cos(theta))
    S = np.mean(np.sin(theta))
    st = np.sqrt(-2.*np.log(np.sqrt(C**2+S**2)))*180./np.pi
    return st

def circular_std_of_mean(theta):
    return circular_std(theta)/np.sqrt(len(theta))

def mean(theta):
    theta = np.asarray(theta, dtype=np.float32)
    return np.mean(theta)

def median(theta):
    theta = np.asarray(theta, dtype=np.float32)
    return np.median(theta)

def minv(theta):
    theta = np.asarray(theta, dtype=np.float32)
    return np.min(theta)

def maxv(theta):
    theta = np.asarray(theta, dtype=np.float32)
    return np.max(theta)

def std(theta):
    theta = np.asarray(theta, dtype=np.float32)
    return np.std(theta)

def wrapped_std(theta):
    theta = np.mod(np.asarray(theta, dtype=np.float32),360)
    m = np.mod(circular_mean(theta),360)
    #now theta and m are in (0-360)
    difs = np.minimum(np.abs(theta-m),360-np.abs(theta-m))
    wrstd = np.sqrt(np.sum(difs**2)/len(theta))
    return wrstd

def unbiased_amp(amp):
    amp2 = np.asarray(amp, dtype=np.float32)**2
    m = np.mean(amp2)
    s = np.std(amp2)
    delta = m**2 - s**2
    if delta >=0:
        a0 = delta**(0.25)
    else:
        a0 = 0.*(m**2)**(0.25)
    return a0

def unbiased_amp_boot(amp):
    amp2 = np.asarray(amp, dtype=np.float32)**2
    #m = np.mean(amp2)
    #s = np.std(amp2)
    m = bootstrap(amp2,N,np.mean)[0]
    s = bootstrap(amp2,N,np.std)[0]
    delta = m**2 - s**2
    if delta >=0:
        a0 = delta**(0.25)
    else:
        a0 = 0.*(m**2)**(0.25)
    return a0

def unbiased_std(amp):
    amp2 = np.asarray(amp, dtype=np.float32)**2
    m = np.mean(amp2)
    s = np.std(amp2)
    delta = m**2 - s**2
    if delta >= 0:
        s0 = np.sqrt((m -np.sqrt(delta))/2.)
    else:
        s0 = 0.*np.sqrt(m/2.)
    return s0

def unbiased_snr(amp):
    return unbiased_amp(amp)/unbiased_std(amp)

def skew(theta):
    theta = np.asarray(theta, dtype=np.float32)
    return st.skew(theta)

def kurt(theta):
    theta = np.asarray(theta, dtype=np.float32)
    return st.kurtosis(theta)

def mad(theta):
    theta = np.asarray(theta, dtype=np.float32)
    madev = float(srs.mad(theta))
    return madev

def circular_mad(theta):
    theta = np.asarray(theta, dtype=np.float32)*np.pi/180.
    C = np.median(np.cos(theta))
    S = np.median(np.sin(theta))
    st = np.sqrt(-2.*np.log(np.sqrt(C**2+S**2)))*180./np.pi
    return st

def medcouple(theta):
    #theta = np.asarray(theta, dtype=np.float32)
    theta = np.asarray(theta)
    try:
        mc = float(sss.medcouple(theta))
    except ValueError:
        return 0
    return mc

def circular_median(theta):
    theta = np.asarray(theta, dtype=np.float32)*np.pi/180.
    theta = theta[theta==theta]  
    if len(theta)==0:
        return None
    else:
        C = np.median(np.cos(theta))
        S = np.median(np.sin(theta))
        mt = np.arctan2(S,C)*180./np.pi
        return mt

def do_quart(theta):
    theta = np.asarray(theta, dtype=np.float32)
    q1 = np.percentile(theta,25)
    return q1

def up_quart(theta):
    theta = np.asarray(theta, dtype=np.float32)
    q3 = np.percentile(theta,75)
    return q3

def iqr(theta):
    theta = np.asarray(theta, dtype=np.float32)
    return st.iqr(theta)

def range_adjust_box(vec,scaler=1.5):
    vec = np.asarray(vec, dtype=np.float32)
    #print(len(vec))
    quart3 = np.percentile(vec,75)
    quart1 = np.percentile(vec,25)
    iqr = quart3-quart1
    mc = medcouple(vec)
    if mc > 0:
        whisk_plus = scaler*iqr*np.exp(3*mc)
        whisk_min = scaler*iqr*np.exp(-4*mc)
    else:
        whisk_plus = scaler*iqr*np.exp(4*mc)
        whisk_min = scaler*iqr*np.exp(-3*mc)       
    range_plus = quart3 + whisk_plus
    range_min = quart1 - whisk_min   
    return [range_min, range_plus]
def adj_box_outlier(vec,scaler=2.):
    vec = np.asarray(vec, dtype=np.float32)
    vec_no_nan=vec[vec==vec]
    if len(vec_no_nan)>0:
        range_box = range_adjust_box(vec_no_nan,scaler)
    else: range_box=[0,0]
    is_in = (vec<range_box[1])&(vec>range_box[0])
    return 1-is_in

def adj_box_outlier_borders(vec,scaler=2.):
    vec = np.asarray(vec, dtype=np.float32)
    vec_no_nan=vec[vec==vec]
    if len(vec_no_nan)>0:
        range_box = range_adjust_box(vec_no_nan,scaler)
    else: range_box=[0,0]
    return range_box

def adj_box_outlier_minus(vec):
    range_box = range_adjust_box(vec)
    is_out = (vec<range_box[0])
    return is_out
def adj_box_outlier_plus(vec):
    range_box = range_adjust_box(vec)
    is_out = (vec>range_box[1])
    return is_out

def number_out(vec):
    return len(adj_box_outlier(vec))

def correlate_tuple(x):
    time = np.asarray([y[0] for y in x])
    sth = np.asarray([y[1] for y in x])
    r = st.pearsonr(time,sth)[0]
    return r

def detect_dropouts_kmeans(x):
    foo_data = list(zip(np.zeros(len(x)),x))
    dataKM = [list(y) for y in foo_data]
    dataKM = np.asarray(dataKM)
    test1 = KMeans(n_clusters=1, random_state=0).fit(dataKM)
    test2 = KMeans(n_clusters=2, random_state=0).fit(dataKM)
    return test1.inertia_/test2.inertia_

def get_dropout_indicator(x,inertia_treshold=3.9):
    foo_data = list(zip(np.zeros(len(x)),x))
    dataKM = [list(y) for y in foo_data]
    dataKM = np.asarray(dataKM)
    test1 = KMeans(n_clusters=1, random_state=0).fit(dataKM)
    test2 = KMeans(n_clusters=2, random_state=0).fit(dataKM)
    inertia_ratio = test1.inertia_/test2.inertia_
    indic = np.zeros(len(x))
    #print(test2.cluster_centers_)
    #if inertia_ratio > inertia_treshold:
    #    indic = test2.labels_
    #    if test2.cluster_centers_[1][1] > test2.cluster_centers_[0][1]:
    #        indic = 1-indic
    return indic

def bootstrap(data, num_samples, statistic, alpha=0.05):
    """Returns bootstrap estimate of 100.0*(1-alpha) CI for statistic."""
    data = np.asarray(data)
    data = np.asarray(data)
    n = len(data)
    idx = npr.randint(0, n, (num_samples, n))
    samples = data[idx]
    stat = np.sort(statistic(samples, 1))
    return np.median(stat),stat[int((alpha/2.0)*num_samples)], stat[int((1-alpha/2.0)*num_samples)]
    #return stat[int(num_samples/2.)]

def bootstrap2(data, num_samples, statistic, alpha=0.05):
    """Returns bootstrap estimate of 100.0*(1-alpha) CI for statistic."""

    stat = np.zeros(num_samples)
    data = np.asarray(data)
    n = len(data)
    idx = npr.randint(0, n, (num_samples, n))
    samples = data[idx]
    for cou in range(num_samples):
        stat[cou] = statistic(samples[cou,:])
    stat = np.sort(stat)
    return np.median(stat),stat[int((alpha/2.0)*num_samples)], stat[int((1-alpha/2.0)*num_samples)]
    #return stat[int(num_samples/2.)]

def scans_statistics(alist,statL='all'):
    if 'band' not in alist.columns:
        alist = add_band(alist,None)
    if 'amp_no_ap' not in alist.columns:
        alist['amp_no_ap']=alist['amp']
    if 'scan_id' not in alist.columns:
        alist['scan_id']=alist['scan_no_tot']
    if 'resid_phas' not in alist.columns:
        alist['resid_phas']=alist['phase']
    foo = alist[['scan_id','polarization','expt_no','band','datetime','resid_phas','amp','baseline','source','amp_no_ap','mjd']]
    time_0 = list(foo['datetime'])[0]
    # means
    foo['mean_amp'] = foo['amp']
    foo['mean_phase'] = foo['resid_phas']
    foo['median_amp'] = foo['amp']
    foo['median_phase'] = foo['resid_phas']
    foo['unbiased_amp'] = foo['amp']
    foo['unb_dtr_amp'] = list(zip(foo.mjd,foo.amp))
    foo['rice_amp'] = foo['amp']
    foo['inv_amp'] = foo['amp']
    
    # variation
    foo['std_amp'] = foo['amp']
    foo['std_phase'] = foo['resid_phas']
    foo['unbiased_std'] = foo['amp']
    foo['EB_sigma'] = foo['amp']
    foo['unbiased_std_no_ap'] = foo['amp_no_ap']
    foo['unbiased_snr'] = foo['amp']
    foo['unb_dtr_std'] = list(zip(foo.mjd,foo.amp))
    foo['rice_std'] = foo['amp']
    foo['inv_std'] = foo['amp']
    foo['mad_amp'] = foo['amp']
    foo['mad_phase'] = foo['resid_phas']
    foo['iqr_amp'] = foo['amp']
    foo['iqr_phase'] = foo['resid_phas']
    foo['q1_amp'] = foo['amp']
    foo['q3_amp'] = foo['amp']
    foo['q1_phase'] = foo['resid_phas']
    foo['q3_phase'] = foo['resid_phas']
    foo['max_amp'] = foo['amp']
    foo['min_amp'] = foo['amp']
    foo['max_phas'] = foo['resid_phas']
    foo['min_phas'] = foo['resid_phas']

    # skewness
    
    foo['skew_amp'] = foo['amp']
    foo['skew_phas'] = foo['resid_phas']
    foo['medc_amp'] = foo['amp']
    foo['medc_phas'] = foo['resid_phas']
    
    #time correlation
    time_sec = list(map(lambda x: float((x - time_0).seconds), foo['datetime']))
    foo['corr_phase'] = list(zip(time_sec,foo['resid_phas']))
    foo['corr_amp'] = list(zip(time_sec,foo['amp']))

    # other
    
    foo['length'] = foo['amp']
    
    foo['number_out'] = foo['amp']

    foo['kurt_amp'] = foo['amp']
    foo['kurt_phase'] = foo['resid_phas']
    
    # dropout detection
    foo['dropout'] = foo['amp']
    

    scans_stats = foo.groupby(('scan_id','polarization','expt_no','band','baseline','source')).agg(
    { 'datetime': min,
    'mean_amp': mean,
    'mean_phase': circular_mean,
    'median_amp': median,
    'median_phase': circular_median,
    'unbiased_amp': unbiased_amp,
    #'unb_dtr_amp': unb_amp_no_trend,
    'rice_amp': find_rice_amp_low_snr,
    'inv_amp': Koay_inv_amp,
    'std_amp': std,
    'std_phase': circular_std,
    'unbiased_std': unbiased_std,
    'EB_sigma': lambda x: bootstrap2(x,int(1e3),std)[0],
    #'unb_dtr_std': unb_std_no_trend,
    'rice_std': find_rice_sig_low_snr,
    'inv_std': Koay_inv_std,
    'unbiased_std_no_ap': unbiased_std,
    'unbiased_snr': unbiased_snr,
    'mad_amp': mad,
    'mad_phase': circular_mad,
    'iqr_amp': iqr,
    'iqr_phase': iqr,
    'q1_amp': do_quart,
    'q3_amp': up_quart,
    'q1_phase': do_quart,
    'q3_phase': up_quart,
    'max_amp': maxv,
    'min_amp': minv,
    'max_phas': maxv,
    'min_phas': minv,
    'skew_amp': skew,
    'skew_phas': skew,
    'medc_amp': medcouple,
    'medc_phas': medcouple,
    'corr_phase': correlate_tuple,
    'corr_amp': correlate_tuple,
    'length': len,
    'number_out': number_out,
    'kurt_amp': kurt,
    'kurt_phase': kurt
    #'dropout': detect_dropouts_kmeans
    })

    return scans_stats.reset_index()

def add_drop_out_indic(alist):
    alist.sort_values('datetime', inplace=True)
    #foo = alist[['scan_id','polarization','expt_no','band','baseline','source','amp']]
  
    #print(alist)
    foo = alist[['scan_id','polarization','expt_no','band','baseline','source','amp']]
    #foo['drop_ind'] = list(zip(np.arange(len(foo['amp'])),foo['amp']))
    foofoo = foo.groupby(('scan_id','polarization','expt_no','band','baseline','source')).transform(lambda x: get_dropout_indicator(x))
    #foo = foo.transform()
    alist['drop_ind'] = foofoo
    return alist

def add_outlier_indic(alist,scaler=2.0):
    if 'outl_ind' in alist.columns:
        alist.drop('outl_ind',axis=1,inplace=True)
    #alist.sort_values('datetime', inplace=True)
    foo = alist[['scan_id','polarization','expt_no','band','baseline','source','amp']]
    foofoo = foo.groupby(('scan_id','polarization','expt_no','band','baseline','source')).transform(lambda x: get_outlier_indicator(x,scaler=scaler))
    alist['outl_ind'] = foofoo
    return alist


def coh_avg_vis_empiric(alist, tcoh='scan'):

    if 'sigma' not in alist.columns:
        alist.loc[:,'sigma'] = alist.loc[:,'amp']/alist.loc[:,'snr']
    if 'band' not in alist.columns:
        alist.loc[:,'band'] = [None]*np.shape(alist)[0]
    if 'phase' not in alist.columns:
        alist.loc[:,'phase'] = alist.loc[:,'resid_phas']
    if 'u' not in alist.columns:
        alist.loc[:,'u'] = [np.nan]*np.shape(alist)[0]
    if 'v' not in alist.columns:
        alist.loc[:,'v'] = [np.nan]*np.shape(alist)[0]
    if 'snr' not in alist.columns:
        alist.loc[:,'snr'] = [np.nan]*np.shape(alist)[0]
    
    alist_loc = alist[['expt_no','source','band','scan_id','polarization','baseline','datetime','u','v','amp','phase']]
    #these quantities are estimated from statistics of amp so we substitute amp in here
    alist_loc['sigma'] = alist_loc['amp']
    alist_loc['snr'] = alist_loc['amp']

    if tcoh=='scan':
        alist_loc = alist_loc.groupby(('expt_no','source','band','scan_id','polarization','baseline')).agg({'datetime': 'min','u': 'mean', 'v': 'mean', 'amp': unbiased_amp, 'phase': circular_mean, 'sigma': lambda x: unbiased_std(x)/np.sqrt(len(x)), 'snr': lambda x : unbiased_snr(x)*np.sqrt(len(x)) })
    else:
        alist_loc['round_time'] = list(map(lambda x: np.round((x- datetime.datetime(2017,4,4)).total_seconds()/tcoh),alist_loc['datetime']))
        alist_loc = alist_loc.groupby(('expt_no','source','band','scan_id','polarization','baseline','round_time')).agg({'datetime': 'min','u': 'mean', 'v': 'mean', 'amp': unbiased_amp,
           'phase': circular_mean, 'sigma': lambda x: unbiased_std(x)/np.sqrt(len(x)), 'snr': lambda x : unbiased_snr(x)*np.sqrt(len(x)) })

    return alist_loc.reset_index()


def coh_avg_vis_thermal(alist, tcoh='scan'):
    if 'sigma' not in alist.columns:
        alist.loc[:,'sigma'] = alist.loc[:,'amp']/alist.loc[:,'snr']
    if 'band' not in alist.columns:
        alist.loc[:,'band'] = [None]*np.shape(alist)[0]
    if 'phase' not in alist.columns:
        alist.loc[:,'phase'] = alist.loc[:,'resid_phas']
    if 'u' not in alist.columns:
        alist.loc[:,'u'] = [np.nan]*np.shape(alist)[0]
    if 'v' not in alist.columns:
        alist.loc[:,'v'] = [np.nan]*np.shape(alist)[0]
    if 'snr' not in alist.columns:
        alist.loc[:,'snr'] = [np.nan]*np.shape(alist)[0]
    
    #alist.loc[:,'median'] = alist.loc[:,'amp']
    #list.loc[:,'mad'] = alist.loc[:,'amp']
    alist_loc = alist[['expt_no','source','band','scan_id','polarization','baseline','datetime','u','v','snr','sigma']]
    alist_loc['vis'] = alist['amp']*np.exp(1j*alist['phase']*np.pi/180)
    
    if tcoh=='scan':
        alist_loc = alist_loc.groupby(('expt_no','source','band','scan_id','polarization','baseline')).agg({'datetime': 'min','u': 'mean', 'v': 'mean', 'vis': np.mean, 
            'sigma': lambda x: np.sqrt(np.sum(x**2))/len(x), 'snr': lambda x : np.sqrt(np.sum(x**2)) })
    else:
        alist_loc['round_time'] = list(map(lambda x: np.round((x- datetime.datetime(2017,4,4)).total_seconds()/tcoh),alist_loc['datetime']))
        alist_loc = alist_loc.groupby(('expt_no','source','band','scan_id','polarization','baseline','round_time')).agg({'datetime': 'min','u': 'mean', 'v': 'mean', 'vis': np.mean,
            'sigma': lambda x: np.sqrt(np.sum(x**2))/len(x), 'snr': lambda x : np.sqrt(np.sum(x**2))  })
        #alist_loc = alist_loc.groupby(('expt_no','source','band','scan_id','polarization','baseline','round_time')).agg({ 'vis': np.mean })


    alist_loc['phase'] = np.angle(np.asarray(alist_loc['vis']))*180/np.pi
    alist_loc['amp'] = np.abs(np.asarray(alist_loc['vis']))
    
    return alist_loc.reset_index()

def add_baselength(alist):
    alist['baselength'] = np.sqrt(np.asarray(alist.u)**2+np.asarray(alist.v)**2)
    return alist

def drop_outliers(alist):
    alist[np.isfinite(alist['amp'])&(alist['outl_ind']==0)]
    return alist

def add_mjd(alist):
    alist['mjd'] = Time(list(alist.datetime)).mjd
    return alist

def add_mjd_smart(df):
    """add *mjd* column to data frame with *datetime* field using astropy for conversion"""
    from astropy import time
    g = df.groupby('datetime')
    (timestamps, indices) = list(zip(*iter(g.groups.items())))
    times_mjd = Time(timestamps).mjd # vectorized
    df['mjdsmart'] = 0. # initialize new column
    for (mjd, idx) in zip(times_mjd, indices):
        df.ix[idx, 'mjdsmart'] = mjd

def add_fmjd(alist):
    alist['fmjd'] = list(map(lambda x: x%1 ,Time(list(alist.datetime)).mjd))
    return alist

def add_band(alist,band):
    alist['band'] = [band]*np.shape(alist)[0]
    return alist

def add_sigma(alist):
    alist['sigma'] = alist['amp']/alist['snr']
    return alist

def add_col(alist,what_add,what_val):
    alist[what_add] = [what_val]*np.shape(alist)[0]
    return alist


def fit_circ(x,y):
    
    x_m = np.mean(x)
    y_m = np.mean(y)
    u = x - x_m
    v = y - y_m
    Suv  = sum(u*v)
    Suu  = sum(u**2)
    Svv  = sum(v**2)
    Suuv = sum(u**2 * v)
    Suvv = sum(u * v**2)
    Suuu = sum(u**3)
    Svvv = sum(v**3) 
    A = np.array([ [ Suu, Suv ], [Suv, Svv]])
    B = np.array([ Suuu + Suvv, Svvv + Suuv ])/2.0
    uc, vc = np.linalg.solve(A, B)
    xc_1 = x_m + uc
    yc_1 = y_m + vc
    Ri_1 = np.sqrt((x-xc_1)**2 + (y-yc_1)**2)
    R_1  = np.mean(Ri_1)
    residu_1 = sum((Ri_1-R_1)**2)
    return R_1, xc_1, yc_1



def MC_CP_dist(amp,sig,bsp_avg=1, N=int(1e5)):

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

def find_rice_sig_low_snr(A):
    A = np.asarray(A)
    m = np.mean(A)
    s = np.mean(A**2)
    L12 = lambda x: np.exp(x/2)*((1-x)*ss.iv(0,-x/2)-x*ss.iv(1,-x/2))
    eq = lambda x: x*L12(1-s/x**2/2) - m*np.sqrt(2/np.pi)
    Esig = so.fsolve(eq,0.5)[0]
    #EA0 = np.sqrt(s - 2*Esig**2)
    return Esig

def find_rice_amp_low_snr(A):
    A = np.asarray(A)
    m = np.mean(A)
    s = np.mean(A**2)
    L12 = lambda x: np.exp(x/2)*((1-x)*ss.iv(0,-x/2)-x*ss.iv(1,-x/2))
    eq = lambda x: x*L12(1-s/x**2/2) - m*np.sqrt(2/np.pi)
    sig0 = np.std(A)
    Esig = so.fsolve(eq,sig0)[0]
    EA0 = np.sqrt(s - 2*Esig**2)
    return EA0


def Koay_inv_amp(A):
    A = np.asarray(A)
    m = np.mean(A)
    s = np.std(A)
    r = m/s
    zeta = lambda x: 2 + x**2 - (np.pi/8)*np.exp(-x**2/2)*((2+x**2)*ss.iv(0,x**2/4) + (x**2)*ss.iv(1,x**2/4))**2
    g = lambda x: (zeta(x)*(1+r**2)-2) #no sqrt

    rel = lambda x: g(x) - x**2
    x0 = r
    sol = so.fsolve(rel,r)[0]

    unb_std = s/np.sqrt(zeta(sol))
    unb_amp = np.sqrt(m**2 +(zeta(sol) - 2)*unb_std**2)
    return unb_amp

def Koay_inv_std(A):
    A = np.asarray(A)
    m = np.mean(A)
    s = np.std(A)
    r = m/s
    zeta = lambda x: 2 + x**2 - (np.pi/8)*np.exp(-x**2/2)*((2+x**2)*ss.iv(0,x**2/4) + (x**2)*ss.iv(1,x**2/4))**2
    g = lambda x: (zeta(x)*(1+r**2)-2) #no sqrt

    rel = lambda x: g(x) - x**2
    x0 = r
    sol = so.fsolve(rel,r)[0]

    unb_std = s/np.sqrt(zeta(sol))
    unb_amp = np.sqrt(m**2 +(zeta(sol) - 2)*unb_std**2)
    return unb_std

def Koay_inv_snr(A):
    A = np.asarray(A)
    m = np.mean(A)
    s = np.std(A)
    r = m/s
    zeta = lambda x: 2 + x**2 - (np.pi/8)*np.exp(-x**2/2)*((2+x**2)*ss.iv(0,x**2/4) + (x**2)*ss.iv(1,x**2/4))**2
    g = lambda x: (zeta(x)*(1+r**2)-2) #no sqrt

    rel = lambda x: g(x) - x**2
    x0 = r
    sol = so.fsolve(rel,r)[0]

    return sol
    

def match_frames(frame1, frame2, what_is_same, dt = 0):

    if dt > 0:
        frame1['round_time'] = list(map(lambda x: np.round((x- datetime.datetime(2017,4,4)).total_seconds()/dt),frame1['datetime']))
        frame2['round_time'] = list(map(lambda x: np.round((x- datetime.datetime(2017,4,4)).total_seconds()/dt),frame2['datetime']))
        what_is_same += ['round_time']
    
    frame1['all_ind'] = list(zip(*[frame1[x] for x in what_is_same]))   
    frame2['all_ind'] = list(zip(*[frame2[x] for x in what_is_same]))
    frames_common = set(frame1['all_ind'])&set(frame2['all_ind'])

    frame1 = frame1[list(map(lambda x: x in frames_common, frame1.all_ind))]
    frame2 = frame2[list(map(lambda x: x in frames_common, frame2.all_ind))]

    frame1 = frame1.sort_values('all_ind').reset_index()
    frame2 = frame2.sort_values('all_ind').reset_index()
    #frame1.drop('all_ind', axis=1)
    #frame2.drop('all_ind', axis=1)
    
    return frame1, frame2

def incoh_avg_vis(frame,tavg='scan',columns_out0=[],phase_type='resid_phas'):
    if 'band' not in frame.columns:
        frame['band'] = np.nan
     
    #minimum set of columns that identifies a scan
    grouping0 = ('scan_id','expt_no','band','polarization','baseline')
    groupingSc = list(set(['scan_id','expt_no','band','polarization','baseline','u','v','source','sbdelay',
                 'mbdelay','delay_rate','total_rate','total_mbdelay','total_sbresid','ref_elev','rem_elev'])&set(frame.columns))
    
    frame.drop_duplicates(subset=list(grouping0)+['datetime'], keep='first', inplace=True)
    frame['vis']=frame['amp']*np.exp(1j*frame[phase_type]*np.pi/180.)
    frame['number']=0.

    if 'phase' not in frame.columns:
        frame['phase'] = frame[phase_type]
    
    aggregating = {#'datetime': lambda x: min(x)+ 0.5*(max(x)-min(x)),
    'datetime':min,
    'amp': np.mean,
    'phase': lambda x: circular_mean(x*np.pi/180),
    'number': len}
        
    if tavg=='scan': #average for entire scan
        frame_avg = frame.groupby(groupingSc).agg(aggregating)
        
    else: # average for 
        frame['round_time'] = list(map(lambda x: np.round((x- datetime.datetime(2017,4,4)).total_seconds()/float(tavg)),frame.datetime))
        grouping = groupingSc+['round_time']
        frame_avg = frame.groupby(grouping).agg(aggregating)

    frame_avg['amp'] = frame_avg['vis'].apply(np.abs)
    frame_avg['phase']=frame_avg['vis'].apply(lambda x: np.angle(x)*180./np.pi)
    frame_avg['sigma']=frame_avg['amp']/frame_avg['snr']

    frame_avg = frame_avg.reset_index()
    columns_out = columns_out0+list(grouping0)+list(set(['source','u','v','datetime','amp','phase','snr','sigma','number'])&set(frame_avg.columns))
    frame_avg_out = frame_avg[columns_out].copy()
    
    util.add_gmst(frame_avg_out)
    frame_avg_out = add_mjd(frame_avg_out)
    frame_avg_out = add_fmjd(frame_avg_out)
    return frame_avg_out


def coh_avg_vis(frame,tavg='scan',columns_out0=[],phase_type='resid_phas'):
    if 'band' not in frame.columns:
        frame['band'] = np.nan
     
    #minimum set of columns that identifies a scan
    grouping0 = ('scan_id','expt_no','band','polarization','baseline')
    groupingSc = list(set(['scan_id','expt_no','band','polarization','baseline','u','v','source','sbdelay',
                 'mbdelay','delay_rate','total_rate','total_mbdelay','total_sbresid','ref_elev','rem_elev'])&set(frame.columns))
    
    frame.drop_duplicates(subset=list(grouping0)+['datetime'], keep='first', inplace=True)
    frame['vis']=frame['amp']*np.exp(1j*frame[phase_type]*np.pi/180.)
    frame['number']=0.
    
    aggregating = {#'datetime': lambda x: min(x)+ 0.5*(max(x)-min(x)),
    'datetime': min,
    'vis': np.mean,
    'snr': lambda x: np.sqrt(np.sum(x**2)),
    'number': len}

    if 'EB_sigma' in frame.columns:
        aggregating['EB_sigma'] = lambda x: np.mean(x)/np.sqrt(len(x))
        columns_out0 += ['EB_sigma']

    if 'fracpol' in frame.columns:
        aggregating['fracpol'] = lambda x: np.mean(x)
        columns_out0 += ['fracpol']
        
    if tavg=='scan': #average for entire scan
        frame_avg = frame.groupby(groupingSc).agg(aggregating)
        
    else: # average for 
        frame['round_time'] = list(map(lambda x: np.round((x- datetime.datetime(2017,4,4)).total_seconds()/float(tavg)),frame.datetime))
        grouping = groupingSc+['round_time']
        frame_avg = frame.groupby(grouping).agg(aggregating)

    frame_avg['amp'] = frame_avg['vis'].apply(np.abs)
    frame_avg['phase']=frame_avg['vis'].apply(lambda x: np.angle(x)*180./np.pi)
    frame_avg['sigma']=frame_avg['amp']/frame_avg['snr']

    frame_avg = frame_avg.reset_index()
    columns_out = columns_out0+list(grouping0)+list(set(['source','u','v','datetime','amp','phase','snr','sigma','number'])&set(frame_avg.columns))
    frame_avg_out = frame_avg[columns_out].copy()
    
    util.add_gmst(frame_avg_out)
    frame_avg_out = add_mjd(frame_avg_out)
    frame_avg_out = add_fmjd(frame_avg_out)
    return frame_avg_out

def coh_avg_bsp(frame,tavg='scan',columns_out0=[]):
    if 'band' not in frame.columns:
        frame['band'] = np.nan
     
    #minimum set of columns that identifies a scan
    grouping0 = ('scan_id','expt_no','band','polarization','triangle')
    groupingSc = list(set(['scan_id','expt_no','band','polarization','triangle','u','v','source','sbdelay',
                 'mbdelay','delay_rate','total_rate','total_mbdelay','total_sbresid','ref_elev','rem_elev'])&set(frame.columns))
    
    frame.drop_duplicates(subset=list(grouping0)+['datetime'], keep='first', inplace=True)
    frame['bsp']=frame['amp']*np.exp(1j*frame['cphase']*np.pi/180.)
    frame['number']=0.
    if 'cphase_fix_amp' not in frame.columns:
        frame['cphase_fix_amp']=frame['cphase']
    aggregating = {#'datetime': lambda x: min(x)+ 0.5*(max(x)-min(x)),
    'datetime': min,
    'bsp': np.mean,
    'snr': lambda x: np.sqrt(np.sum(x**2)),
    'number': len,
    'cphase_fix_amp': circular_mean}

    def agg_mean_3tup(x):  
        x0 = np.asarray([float(y[0]) for y in x if y[0]==y[0] ])
        x1 = np.asarray([float(y[1]) for y in x if y[1]==y[1] ])
        x2 = np.asarray([float(y[2]) for y in x if y[2]==y[2] ])
        return(np.mean(x0[x0==x0]),np.mean(x1[x1==x1]),np.mean(x2[x2==x2]))

    if 'EB_sigma' in frame.columns:
        aggregating['EB_sigma'] = lambda x: np.mean(x)/np.sqrt(len(x))
        columns_out0 += ['EB_sigma']

    if 'fracpols' in frame.columns:
        aggregating['fracpols'] = agg_mean_3tup
        columns_out0 += ['fracpols']
    
    if 'amps' in frame.columns:
        aggregating['amps'] = agg_mean_3tup
        columns_out0 += ['amps']
        
    if tavg=='scan': #average for entire scan
        frame_avg = frame.groupby(groupingSc).agg(aggregating)
        
    else: # average for 
        frame['round_time'] = list(map(lambda x: np.round((x- datetime.datetime(2017,4,4)).total_seconds()/float(tavg)),frame.datetime))
        grouping = groupingSc+['round_time']
        frame_avg = frame.groupby(grouping).agg(aggregating)

    frame_avg['amp'] = frame_avg['bsp'].apply(np.abs)
    frame_avg['cphase']=frame_avg['bsp'].apply(lambda x: np.angle(x)*180./np.pi) #deg
    frame_avg['sigma']=frame_avg['amp']/frame_avg['snr']
    frame_avg['sigmaCP'] = 1./frame_avg['snr']*180./np.pi #deg
    columns_out0 += ['cphase_fix_amp']
    frame_avg = frame_avg.reset_index()
    columns_out = list(grouping0)+list(set(columns_out0)|set(['source','datetime','amp','cphase','snr','number','sigma','sigmaCP'])&set(frame_avg.columns))
    frame_avg_out = frame_avg[columns_out].copy()
    
    util.add_gmst(frame_avg_out)
    frame_avg_out = add_mjd(frame_avg_out)
    frame_avg_out = add_fmjd(frame_avg_out)
    return frame_avg_out


def prepare_ER3_vis(path='/Users/mwielgus/Dropbox (Smithsonian External)/EHT/Data/ReleaseE3/hops/ER3v1/', filen='alist.v6.2s',bands=['lo','hi'],reverse_pol=False):
    
    bandL=[]
    if 'lo' in bands:
        print('loading lo band data...')
        lo_path = path+'lo/'+filen
        hops_lo = hops.read_alist(lo_path)
        hops_lo = cl.add_band(hops_lo,'lo')
        bandL+=[hops_lo]

    if 'hi' in bands:
        print('loading hi band data...')
        hi_path = path+'hi/'+filen
        hops_hi = hops.read_alist(hi_path)
        hops_hi = cl.add_band(hops_hi,'hi')
        bandL+=[hops_hi]

    hops_frame = pd.concat(bandL,ignore_index=True)

    print('remove duplicates...')
    hops_frame.drop_duplicates(['scan_id','expt_no', 'baseline','polarization','band','datetime'], keep='first', inplace=True)
    print('removing autocorrelations...')
    #remove autocorrelations
    hops_frame.drop(list(hops_frame[hops_frame.baseline.str[0]==hops_frame.baseline.str[1]].index.values),inplace=True)
    print('removing SR...')
    #remove SR baseline
    hops_frame.drop(list(hops_frame[hops_frame.baseline=='SR'].index.values),inplace=True)
    #INVERT POLARIZATION THIS FIX BUG IN ER3
    
    #if reverse_pol==True:
    #    print('reverse polarization labels...')
    #    hops_frame['polarization'] = hops_frame['polarization'].apply(lambda x: x[::-1])
    
    fix_alist(hops_frame)

    print('adding mjd...')
    #add mjd
    hops_frame = add_mjd(hops_frame)
    print('adding fmjd...')
    hops_frame = add_fmjd(hops_frame)
    #add baselength
    print('adding baselength...')
    hops_frame = add_baselength(hops_frame)
    #add gmst
    print('adding gmst...')
    util.add_gmst(hops_frame)
    return hops_frame


def add_outlier_ind(frame,what='amp',scaler=2.,remove=True):
    if 'band' not in frame.columns:
        frame['band'] = np.nan
    frame['outlier'] = frame[what]
    #minimum set of columns that identifies a scan
    grouping0 = ('scan_id','expt_no','band','polarization','baseline')
    groupingSc = list(set(['scan_id','expt_no','band','polarization','baseline','u','v','source','sbdelay',
                 'mbdelay','delay_rate','total_rate','total_mbdelay','total_sbresid','ref_elev','rem_elev'])&set(frame.columns))
    frame['outlier'] = frame[groupingSc+['outlier']].groupby(groupingSc).transform(lambda x: adj_box_outlier(x,scaler))
    if remove==True:
        frame.drop(list(frame[frame.outlier==1].index.values),inplace=True)
        frame.drop('outlier',axis=1,inplace=True)
    return frame


def add_unique_sc(df):
    if 'baseline' in df.columns:
        element = 'baseline'
    elif 'triangle' in df.columns:
        element='triangle'
    elif 'quadrangle' in df.columns:
        element='quadrangle'
    df['unique_sc'] = list(zip(df.expt_no,df.scan_id,df.band,df.polarization,df[element]))
    return df


def add_empirical_error(df,stats):
    '''
    amplitude error from fitting to Rice distribution
    '''
    if 'unique_sc' not in df:
        df = add_unique_sc(df).sort_values('unique_sc').reset_index()
    
    if 'unique_sc' not in stats:
        stats = add_unique_sc(stats).sort_values('unique_sc').reset_index()

    dict_sig = dict(zip(stats['unique_sc'],stats['unbiased_std']))
    df['emp_sigma']=list(map(lambda x: dict_sig[x],df['unique_sc']))
    return df

#def add_MC_error_estimator_CP(df,stats):

def add_EB_scan_variability(df):
    #EB = empirical bootstrap
    if 'baseline' in df.columns:
        element = 'baseline'
        quantity = 'amp'
        fun_var = lambda x: np.std(x)
    elif 'triangle' in df.columns:
        element='triangle'
        quantity='cphase'
        fun_var = lambda x: wrapped_std(x)
    elif 'quadrangle' in df.columns:
        element='quadrangle'
        quantity='logamp'
        fun_var = lambda x: np.std(x)

    df['EB_sigma'] = df[quantity]
    df['unique_sc'] = list(zip(df.expt_no,df.scan_id,df.band,df.polarization,df[element]))
    grouping = ['expt_no','scan_id','band','polarization','unique_sc']+[element]

    df2 = df.groupby(grouping).agg({'EB_sigma': lambda x: bootstrap2(x,int(1e2),fun_var)[0] }).reset_index()
    #df2['unique_sc'] = list(zip(df2.expt_no,df2.scan_id,df2.band,df2.polarization,df2[element]))

    dict_sig = dict(zip(df2['unique_sc'],df2['EB_sigma']))
    df['EB_sigma']=list(map(lambda x: dict_sig[x],df['unique_sc']))
    return df

def add_fracpol(df):
    grouping = ['expt_no','scan_id','band','baseline','datetime']
    
    pivpol = pd.pivot_table(df,values='amp',index=grouping,columns='polarization').reset_index()
    pivpol['polars'] = list(zip(pivpol.LL,pivpol.RR,pivpol.LR,pivpol.RL))

    def calc_fracpol(x):
        if (x[0]!=x[0])& (x[1]==x[1]):
            x=(x[1],x[1],x[2],x[3])
        if (x[1]!=x[1])& (x[0]==x[0]):
            x=(x[0],x[0],x[2],x[3])
        if (x[2]!=x[2])& (x[3]==x[3]):
            x=(x[0],x[1],x[3],x[3])
        if (x[3]!=x[3])& (x[2]==x[2]):
            x=(x[0],x[1],x[2],x[2])
        try:
            fracpol = np.sqrt(float(x[2])*float(x[3])/float(x[1])/float(x[0]))
        except:
            fracpol=np.nan
        return fracpol
    pivpol['fracpol'] = list(map(calc_fracpol,pivpol['polars']))
    pivpol['uni_fracpol'] = list(zip(pivpol.expt_no,pivpol.scan_id,pivpol.band,pivpol.baseline,pivpol.datetime))
    df['uni_fracpol'] = list(zip(df.expt_no,df.scan_id,df.band,df.baseline,df.datetime))

    dict_fracpol=dict(zip(pivpol.uni_fracpol,pivpol.fracpol))
    df['fracpol'] = list(map(lambda x: dict_fracpol[x], df.uni_fracpol))
   
    return df

def fix_alist(df):
    #Lindy's routine to deal with all swaps in the data
    # sqrt2 fix er2lo:('zplptp', 'zrmvon') er2hi:('zplscn', 'zrmvoi')
    idx = (df.baseline.str.count('A') == 1) & (df.root_id > 'zpaaaa') & (df.root_id < 'zrzzzz')
    df.loc[idx,'snr'] /= np.sqrt(2.0)
    df.loc[idx,'amp'] /= np.sqrt(2.0)
    # swap polarization fix er3lo:('zxuerf', 'zyjmiy') er3hi:('zymrse', 'zztobd')
    idx1 = df.baseline.str.contains('A') & (df.polarization == 'LR') & (df.root_id > 'zxaaaa') & (df.root_id < 'zzzzzz')
    idx2 = df.baseline.str.contains('A') & (df.polarization == 'RL') & (df.root_id > 'zxaaaa') & (df.root_id < 'zzzzzz')
    df.loc[idx1,'polarization'] = 'RL'
    df.loc[idx2,'polarization'] = 'LR'
    # SMA polarization swap EHT high band D05
    idx1 = (df.baseline.str[0] == 'S') & (df.root_id > 'zxaaaa') & (df.root_id < 'zztzzz') & (df.expt_no == 3597)
    idx2 = (df.baseline.str[1] == 'S') & (df.root_id > 'zxaaaa') & (df.root_id < 'zztzzz') & (df.expt_no == 3597)
    df.loc[idx1,'polarization'] = df.loc[idx1,'polarization'].map({'LL':'RL', 'LR':'RR', 'RL':'LL', 'RR':'LR'})
    df.loc[idx2,'polarization'] = df.loc[idx2,'polarization'].map({'LL':'LR', 'LR':'LL', 'RL':'RR', 'RR':'RL'})