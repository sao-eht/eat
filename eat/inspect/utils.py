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
#from eat.aips import aips2alist as a2a
#from eat.inspect import closures as cl
#from eat.inspect import closures as cl
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

def circular_std_vector(theta):

    N = np.shape(theta)[1]
    st = np.zeros(N)
    for cou in range(N):
        theta_foo = theta[:,cou]
        theta_foo = theta_foo[theta_foo==theta_foo]
        theta_foo = np.asarray(theta_foo, dtype=np.float32)*np.pi/180.
        C = np.mean(np.cos(theta_foo))
        S = np.mean(np.sin(theta_foo))
        st[cou] = np.sqrt(-2.*np.log(np.sqrt(C**2+S**2)))*180./np.pi
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
    amp=amp[amp==amp]
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
    amp=amp[amp==amp]
    amp = np.asarray(amp, dtype=np.float32)
    amp2 = amp**2
    m = np.mean(amp2)
    s = np.std(amp2)
    delta = m**2 - s**2
    if delta >= 0:
        s0 = np.sqrt((m -np.sqrt(delta))/2.)
    else:
        s0 = np.std(amp)/np.sqrt(2-np.pi/2)
    return s0

def unbiased_snr(amp):
    return unbiased_amp(amp)/unbiased_std(amp)

def skew(theta):
    theta = np.asarray(theta, dtype=np.float32)
    return st.skew(theta)

def kurt(theta):
    theta = np.asarray(theta, dtype=np.float32)
    return st.kurtosis(theta)

    theta = np.asarray(theta, dtype=np.float32)
    madev = float(srs.mad(theta))
    return madev

def mad_deg(theta):
    '''
    mad with shift against 2pi wrapping
    '''
    theta = np.asarray(theta, dtype=np.float32)
    m = circular_median(theta)
    foo = np.exp(1j*theta*np.pi/180)*np.exp(-1j*m*np.pi/180)
    foo = np.angle(foo)*180/np.pi
    madev = float(srs.mad(foo))
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


def circular_weighted_median(theta,weights=[]):
    import weightedstats as ws
    theta = np.asarray(theta, dtype=np.float32)*np.pi/180.
    theta = theta[theta==theta]
    if len(theta)==0:
        return None
    else:
        C = ws.weighted_median(np.cos(theta),weights)
        S = ws.weighted_median(np.sin(theta),weights)
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
    return sum(adj_box_outlier(vec))

def correlate_tuple(x):
    time = np.asarray([y[0] for y in x])
    sth = np.asarray([y[1] for y in x])
    r = st.pearsonr(time,sth)[0]
    return r

def detect_dropouts_kmeans(x):
    foo_data = list(zip(np.zeros(len(x)),x))
    if len(foo_data)>4:
        dataKM = [list(y) for y in foo_data]
        dataKM = np.asarray(dataKM)
        test1 = KMeans(n_clusters=1, random_state=0).fit(dataKM)
        test2 = KMeans(n_clusters=2, random_state=0).fit(dataKM)
        out_stat = test1.inertia_/test2.inertia_
    else:
        out_stat = 0
    return out_stat

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
    if alpha=='1s':
        alpha=0.3173
    elif alpha=='2s':
        alpha=0.0455
    elif alpha=='3s':
        alpha=0.0027
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
    'kurt_phase': kurt,
    'dropout': detect_dropouts_kmeans
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
            'sigma': lambda x: np.sqrt(np.sum(x**2))/len(x) })
    else:
        alist_loc['round_time'] = list(map(lambda x: np.round((x- datetime.datetime(2017,4,4)).total_seconds()/tcoh),alist_loc['datetime']))
        alist_loc = alist_loc.groupby(('expt_no','source','band','scan_id','polarization','baseline','round_time')).agg({'datetime': 'min','u': 'mean', 'v': 'mean', 'vis': np.mean,
            'sigma': lambda x: np.sqrt(np.sum(x**2))/len(x) })
        #alist_loc = alist_loc.groupby(('expt_no','source','band','scan_id','polarization','baseline','round_time')).agg({ 'vis': np.mean })

    #alist_loc['snr'] = alist
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
    g = df.groupby('datetime')
    (timestamps, indices) = list(zip(*iter(g.groups.items())))
    times_mjd = Time(timestamps).mjd # vectorized
    df['mjdsmart'] = 0. # initialize new column
    for (mjd, idx) in zip(times_mjd, indices):
        df.loc[idx, 'mjdsmart'] = mjd

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
    
def sigma_fn(x):
    return np.sqrt(np.sum(x**2)) / len(x)

def match_frames(frame1, frame2, what_is_same, dt = 0, uniquely=True):

    if dt > 0:
        frame1['round_time'] = list(map(lambda x: np.round((x- datetime.datetime(2017,4,4)).total_seconds()/dt),frame1['datetime']))
        frame2['round_time'] = list(map(lambda x: np.round((x- datetime.datetime(2017,4,4)).total_seconds()/dt),frame2['datetime']))
        what_is_same += ['round_time']
    
    frame1['all_ind'] = list(zip(*[frame1[x] for x in what_is_same]))   
    frame2['all_ind'] = list(zip(*[frame2[x] for x in what_is_same]))
    frames_common = set(frame1['all_ind'])&set(frame2['all_ind'])

    frame1 = frame1[list(map(lambda x: x in frames_common, frame1.all_ind))]
    frame2 = frame2[list(map(lambda x: x in frames_common, frame2.all_ind))]

    if uniquely:
        frame1 = frame1.drop_duplicates(subset=['all_ind'], keep='first')
        frame2 = frame2.drop_duplicates(subset=['all_ind'], keep='first')

    frame1 = frame1.sort_values('all_ind').reset_index(drop=True)
    frame2 = frame2.sort_values('all_ind').reset_index(drop=True)
    frame1.drop('all_ind', axis=1,inplace=True)
    frame2.drop('all_ind', axis=1,inplace=True)
    
    return frame1, frame2


def match_multiple_frames(frames, what_is_same, dt = 0,uniquely=True):

    if dt > 0:
        for frame in frames:
            frame['round_time'] = list(map(lambda x: np.round((x- datetime.datetime(2017,4,4)).total_seconds()/dt),frame['datetime']))
        what_is_same += ['round_time']
    
    frames_common = {}
    for frame in frames:
        frame['all_ind'] = list(zip(*[frame[x] for x in what_is_same]))   
        if frames_common != {}:
            frames_common = frames_common&set(frame['all_ind'])
        else:
            frames_common = set(frame['all_ind'])

    frames_out = []
    for frame in frames:
        frame = frame[list(map(lambda x: x in frames_common, frame.all_ind))]
        if uniquely:
            frame.drop_duplicates(subset=['all_ind'], keep='first', inplace=True)

        frame = frame.sort_values('all_ind').reset_index(drop=True)
        frame.drop('all_ind', axis=1,inplace=True)
        frames_out.append(frame.copy())
    return frames_out

def incoh_avg_vis(frame,tavg='scan',columns_out0=[],phase_type='resid_phas',debias=True, robust=False):
    frame = frame.loc[:,~frame.columns.duplicated()]
    if 'band' not in frame.columns:
        frame['band'] = np.nan
    #print('incoh_avg anana',frame.columns)
    if 'sigma' not in frame.columns:
        frame['sigma'] = np.asarray(frame['amp'])/np.asarray(frame['snr'])

    if 'scan_id' not in frame.columns:
        frame['scan_id'] = frame['scan_no_tot']
     
    #minimum set of columns that identifies a scan
    grouping0 = ('scan_id','expt_no','band','polarization','baseline')
    groupingSc = list(set(['scan_id','expt_no','band','polarization','baseline','source','sbdelay',
                 'mbdelay','delay_rate','total_rate','total_mbdelay','total_sbresid','ref_elev','rem_elev'])&set(frame.columns))
    if tavg!='scan':
        if tavg>600:
            grouping0 = ('expt_no','band','polarization','baseline')
            groupingSc = list(set(['expt_no','band','polarization','baseline','source','sbdelay',
                 'mbdelay','delay_rate','total_rate','total_mbdelay','total_sbresid','ref_elev','rem_elev'])&set(frame.columns))


    frame.drop_duplicates(subset=list(grouping0)+['datetime'], keep='first', inplace=True)
    if 'vis' not in frame.columns:
        frame['vis']=frame['amp']*np.exp(1j*frame[phase_type]*np.pi/180.)
    frame['number']=0.

    if 'phase' not in frame.columns:
        frame['phase'] = frame[phase_type]

    #columns_out0 += ['amp_db']
    #frame['amp_db']= list( (np.asarray(frame['amp'])**2)*(1.-2./np.asarray(frame['snr'])**2) )

    aggregating = {#'datetime': lambda x: min(x)+ 0.5*(max(x)-min(x)),
    'datetime': 'min',
    #'amp_db': lambda x: np.sqrt(np.mean(x)),
    'amp': lambda vec: incoh_avg_amp_vector(vec,debias=debias,robust=robust),
    'amp_ndb': lambda vec: incoh_avg_amp_vector(vec,debias=False,robust=robust),
    'phase': circular_mean,
    'number': 'size',
    #'snr': lambda x: np.sqrt(np.sum(x**2))
    'sigma': sigma_fn,
    }

    if 'u' in frame.columns:
        aggregating['u'] = 'mean'
        columns_out0 += ['u']
    if 'v' in frame.columns:
        aggregating['v'] = 'mean'
        columns_out0 += ['v']

    #ACTUAL AVERAGING
    frame['amp_ndb'] = list(zip(frame['amp'],frame['sigma'])) 
    frame['amp'] = list(zip(frame['amp'],frame['sigma']))   
    if tavg=='scan': #average for entire scan
        frame_avg = frame.groupby(groupingSc, observed=True).agg(aggregating)
        
    else: # average for 
        date0 = datetime.datetime(2017,4,4)
        frame['round_time'] = list(map(lambda x: np.round((x- date0).total_seconds()/float(tavg)),frame.datetime))
        grouping = groupingSc+['round_time']
        frame_avg = frame.groupby(grouping, observed=True).agg(aggregating)
        frame_avg = frame_avg.reset_index()
        frame_avg['datetime'] =  list(map(lambda x: date0 + datetime.timedelta(seconds= int(tavg*x)), frame_avg['round_time']))
    
    frame_avg['snr'] = frame_avg['amp']/frame_avg['sigma']

    #frame_avg['amp'] = frame_avg['vis'].apply(np.abs)
    #frame_avg['phase']=frame_avg['vis'].apply(lambda x: np.angle(x)*180./np.pi)
    #if 'snr' in frame_avg.columns:
    #print(list(frame_avg['snr']))
    #print(frame_avg['snr'])
    #frame_avg['sigma']=frame_avg['amp']/frame_avg['snr']

    frame_avg = frame_avg.reset_index()
    columns_out = columns_out0+list(grouping0)+list(set(['source','datetime','amp','amp_ndb','phase','snr','sigma','number'])&set(frame_avg.columns))
    frame_avg_out = frame_avg[columns_out].copy()
    
    #util.add_gmst(frame_avg_out)
    #frame_avg_out = add_mjd(frame_avg_out)
    #frame_avg_out = add_fmjd(frame_avg_out)
    frame_avg_out = frame_avg_out.loc[:,~frame_avg_out.columns.duplicated()]
    return frame_avg_out


def incoh_avg_amp_vector(vec,debias=True,robust=False):
    #print(debias)
    #print(len(vec))
    #print('printing amp',vec)
    amp = np.asarray([x[0] for x in vec])
    sig = np.asarray([x[1] for x in vec])
    #print(len(amp),len(sig))
    amp0=amp; sig0=sig
    amp=amp0[(amp0==amp0)&(sig0==sig0)]
    sig=sig0[(amp0==amp0)&(sig0==sig0)]
    try:
        if debias==True:
            if robust==False:
                A02 = np.sum(amp**2 - ( 2. -1./len(amp) )*sig**2)/len(amp)
                A02 = np.maximum(A02,0.)
                amp_out = np.sqrt(A02)
                #eq 9.86 Thompson 
            else:
                A02 = np.median(amp**2) - ( 2. -1./len(amp) )*np.median(sig**2) 
                A02 = np.maximum(A02,0.)
                amp_out = np.sqrt(A02)
        else:
            if robust==False:
                amp_out = np.sqrt(np.sum(amp**2/len(amp)))
            else:
                amp_out  = np.median(amp)
    except ZeroDivisionError: amp_out = float('nan')

    return amp_out


def coh_avg_vis(frame,tavg='scan',columns_out0=[],phase_type='resid_phas'):
    if 'band' not in frame.columns:
        frame['band'] = 0
     
    if 'scan_id' not in frame.columns:
        frame['scan_id'] = frame['scan_no_tot']
    
    if 'sigma' not in frame.columns:
        frame['sigma'] = frame['amp']/frame['snr']

    meanF = lambda x: np.nanmean(np.asarray(x))
    def meanerrF(x):
        x = np.asarray(x)
        x = x[x==x]
        try: ret = np.sqrt(np.sum(x**2)/len(x)**2)
        except: ret = np.nan +1j*np.nan
        return ret

    #if 'snr' not in frame.columns:
    #    frame['snr'] = frame['amp']/frame['sigma']

    #minimum set of columns that identifies a scan
    grouping0 = ('scan_id','expt_no','band','polarization','baseline')
    groupingSc = list(set(['scan_id','expt_no','band','polarization','baseline','source','sbdelay',
                 'mbdelay','delay_rate','total_rate','total_mbdelay','total_sbresid','ref_elev','rem_elev'])&set(frame.columns))
    #print(groupingSc)
    frame.drop_duplicates(subset=list(grouping0)+['datetime'], keep='first', inplace=True)
    if 'vis' not in frame.columns:
        frame['vis']=frame['amp']*np.exp(1j*frame[phase_type]*np.pi/180.)
    frame['number']=0.

    aggregating = {#'datetime': lambda x: min(x)+ 0.5*(max(x)-min(x)),
    'datetime': 'min',
    #'vis': np.mean,
    'vis': meanF,
    #'snr': lambda x: np.sqrt(np.sum(x**2)),
    'sigma': meanerrF,
    #'sigma': lambda x: np.sqrt(np.sum(x**2))/len(x),
    'number': 'size'}

    if 'u' in frame.columns:
        aggregating['u'] = 'mean'
        columns_out0 += ['u']
    if 'v' in frame.columns:
        aggregating['v'] = 'mean'
        columns_out0 += ['v']

    if 'rrvis' in frame.columns:
        #polrep='circ' stuff
        aggregating['rrvis'] = meanF
        columns_out0 += ['rrvis']
        aggregating['llvis'] = meanF
        columns_out0 += ['llvis']
        aggregating['lrvis'] = meanF
        columns_out0 += ['lrvis']
        aggregating['rlvis'] = meanF
        columns_out0 += ['rlvis']
        aggregating['rrsigma'] = meanerrF
        columns_out0 += ['rrsigma']
        aggregating['llsigma'] = meanerrF
        columns_out0 += ['llsigma']
        aggregating['lrsigma'] = meanerrF
        columns_out0 += ['lrsigma']
        aggregating['rlsigma'] = meanerrF
        columns_out0 += ['rlsigma']

    if 'EB_sigma' in frame.columns:
        aggregating['EB_sigma'] = meanerrF
        columns_out0 += ['EB_sigma']

    if 'fracpol' in frame.columns:
        aggregating['fracpol'] = lambda x: np.mean(x)
        columns_out0 += ['fracpol']

    if 'std_by_mean' in frame.columns:
        #if it is there, we assume that it's filled with amplitudes
        aggregating['std_by_mean'] = lambda x: np.std(x)/np.mean(x)
        columns_out0 += ['std_by_mean']

    if 'amp_moments' in frame.columns:
        #if it is there, we assume that it's filled with amplitudes
        aggregating['amp_moments'] = unbiased_amp
        columns_out0 += ['amp_moments']

    if 'sig_moments' in frame.columns:
        #if it is there, we assume that it's filled with amplitudes
        aggregating['sig_moments'] =  unbiased_std
        columns_out0 += ['sig_moments']

    if tavg=='scan': #average for entire scan
        frame_avg = frame.groupby(groupingSc, observed=True).agg(aggregating)

    else: # average for tcoh seconds
        frame['round_time'] = list(map(lambda x: np.round((x- datetime.datetime(2017,4,4)).total_seconds()/float(tavg)),frame.datetime))
        grouping = groupingSc+['round_time']
        frame_avg = frame.groupby(grouping, observed=True).agg(aggregating)
        #frame.drop('datetime',axis=1,inplace=True)
        frame_avg = frame_avg.reset_index()
        #print(frame_avg.columns)
        frame_avg['datetime'] =  list(map(lambda x: datetime.datetime(2017,4,4) + datetime.timedelta(seconds= int(tavg*x)), frame_avg['round_time']))
        
        #frame['datetime']= 0
    frame_avg['amp'] = frame_avg['vis'].apply(np.abs)
    frame_avg['phase']=frame_avg['vis'].apply(lambda x: np.angle(x)*180./np.pi)
    #frame['snr'] = frame['amp']/frame['sigma']
    #frame_avg['sigma']=frame_avg['amp']/frame_avg['snr']

    frame_avg = frame_avg.reset_index()
    #columns_out0 = list(set(columns_out0))
    columns_out = list(set(columns_out0)&set(frame_avg.columns))+list(grouping0)+list(set(['source','datetime','amp','phase','snr','sigma','number'])&set(frame_avg.columns))
    #print('columns_out0:', columns_out0)
    #print('grouping0:', list(grouping0))
    frame_avg_out = frame_avg[columns_out].copy()
    #'''
    try:
        util.add_gmst(frame_avg_out)
    except ValueError:
        pass

    frame_avg_out = add_mjd(frame_avg_out)
    frame_avg_out = add_fmjd(frame_avg_out)
    #'''
    if 'snr' not in frame_avg_out:
        frame_avg_out['snr'] = frame_avg_out['amp']/frame_avg_out['sigma']

    frame_avg_out = frame_avg_out.loc[:,~frame_avg_out.columns.duplicated()]
    return frame_avg_out


def incoh_avg_band(frame,singleband='keep',add_same=[],columns_out0=[],phase_type='phase'):
    
    if 'scan_id' not in frame.columns:
        frame['scan_id'] = frame['scan_no_tot']
    if 'sigma' not in frame.columns:
        frame['sigma'] = frame['amp']/frame['snr']
    if 'number' not in frame.columns:
        frame['number'] = 1
        
    same =['scan_id','baseline','polarization']+add_same
    grouping = list(set(same+['expt_no','source','sbdelay',
                 'mbdelay','delay_rate','total_rate','total_mbdelay','total_sbresid','ref_elev','rem_elev'])&set(frame.columns))
            
    if singleband=='drop':
        frame = frame.groupby(grouping).filter(lambda x: len(x)==2).copy()
        #print(frame.baseline.unique())
    else:
        frame = frame.groupby(grouping).filter(lambda x: len(x)<=2).copy()
    frame=frame.reset_index()
    
    aggregating = {#'datetime': lambda x: min(x)+ 0.5*(max(x)-min(x)),
    'amp': 'mean',
    'sigma': lambda x: np.sqrt(np.sum(x**2))/len(x),
    phase_type: circular_mean,
    'number': 'sum'}
    if 'datetime' not in same: aggregating['datetime'] = 'min'
    if 'u' in frame.columns:
        aggregating['u'] = 'mean'
        columns_out0 += ['u']
    if 'v' in frame.columns:
        aggregating['v'] = 'mean'
        columns_out0 += ['v']
    if 'rrvis' in frame.columns:
        #polrep='circ' stuff
        aggregating['rrvis'] = 'mean'
        columns_out0 += ['rrvis']
        aggregating['llvis'] = 'mean'
        columns_out0 += ['llvis']
        aggregating['lrvis'] = 'mean'
        columns_out0 += ['lrvis']
        aggregating['rlvis'] = 'mean'
        columns_out0 += ['rlvis']
        aggregating['rrsigma'] = 'mean'
        columns_out0 += ['rrsigma']
        aggregating['llsigma'] = 'mean'
        columns_out0 += ['llsigma']
        aggregating['lrsigma'] = 'mean'
        columns_out0 += ['lrsigma']
        aggregating['rlsigma'] = 'mean'
        columns_out0 += ['rlsigma']
    if 'EB_sigma' in frame.columns:
        aggregating['EB_sigma'] = lambda x: np.sqrt(np.sum(x**2))/len(x)
        columns_out0 += ['EB_sigma']

    if 'fracpol' in frame.columns:
        aggregating['fracpol'] = lambda x: np.mean(x)
        columns_out0 += ['fracpol']

    if 'std_by_mean' in frame.columns:
        #if it is there, we assume that it's filled with amplitudes
        aggregating['std_by_mean'] = 'mean'
        columns_out0 += ['std_by_mean']
       
    if 'amp_moments' in frame.columns:
        #if it is there, we assume that it's filled with amplitudes
        aggregating['amp_moments'] = 'mean'
        columns_out0 += ['amp_moments']
     
    if 'sig_moments' in frame.columns:
        #if it is there, we assume that it's filled with amplitudes
        aggregating['sig_moments'] = 'mean'
        columns_out0 += ['sig_moments']        
    frame_avg = frame.groupby(grouping).agg(aggregating).reset_index()
    frame_avg['snr'] = frame_avg['amp']/frame_avg['sigma']

    columns_out = list(set(columns_out0)&set(frame_avg.columns))+list(grouping)+list(set(['source','datetime','amp',phase_type,'snr','sigma','number'])&set(frame_avg.columns))
    columns_out = list(set(columns_out))
    frame_avg_out = frame_avg[columns_out].copy()
    if 'gmst' not in frame_avg_out.columns:
        try: util.add_gmst(frame_avg_out)
        except: pass
    frame_avg_out = add_mjd(frame_avg_out)
    frame_avg_out = add_fmjd(frame_avg_out)
    frame_avg_out['band']='LH'

    frame_avg_out = frame_avg_out.loc[:,~frame_avg_out.columns.duplicated()]
    return frame_avg_out


def avg_Stokes(frame,singlepol=[],add_same=[],columns_out0=[],phase_type='phase'):
    
    frame = frame[frame.polarization.str[0]==frame.polarization.str[1]].copy()
    if 'scan_id' not in frame.columns:
        frame['scan_id'] = frame['scan_no_tot']
    if 'sigma' not in frame.columns:
        frame['sigma'] = frame['amp']/frame['snr']
    if 'number' not in frame.columns:
        frame['number'] = 1
    if 'band' not in frame.columns:
        frame['band'] = 'dummy'
    if 'vis' not in frame.columns:
        frame['vis']=frame['amp']*np.exp(1j*frame[phase_type]*np.pi/180.)
        
    same =['scan_id','baseline','band']+add_same
    grouping = list(set(same+['expt_no','source','sbdelay',
                 'mbdelay','delay_rate','total_rate','total_mbdelay','total_sbresid','ref_elev','rem_elev'])&set(frame.columns))  
    
    if singlepol==[]:
        frame = frame.groupby(grouping).filter(lambda x: len(x)==2).copy()
    else:
        is_in_singlepol_list = lambda x: any([y in x for y in singlepol])
        frame['singlepol']=list(map(is_in_singlepol_list,frame.baseline))
        frame = frame.groupby(grouping+['singlepol']).filter(lambda x: (len(x)==2)|(all(x.singlepol==True))).copy()
    frame=frame.reset_index()
    
    aggregating = {#'datetime': lambda x: min(x)+ 0.5*(max(x)-min(x)),
    'vis': 'mean',
    'sigma': lambda x: np.sqrt(np.sum(x**2))/len(x),
    'number': 'sum'}
    if 'datetime' not in same: aggregating['datetime'] = 'min'
    if 'u' in frame.columns:
        aggregating['u'] = 'mean'
        columns_out0 += ['u']
    if 'v' in frame.columns:
        aggregating['v'] = 'mean'
        columns_out0 += ['v']
    if 'EB_sigma' in frame.columns: 
        aggregating['EB_sigma'] = lambda x: np.sqrt(np.sum(x**2))/len(x)
        columns_out0 += ['EB_sigma']

    if 'fracpol' in frame.columns:
        aggregating['fracpol'] = lambda x: np.mean(x)
        columns_out0 += ['fracpol']

    if 'std_by_mean' in frame.columns:
        #if it is there, we assume that it's filled with amplitudes
        aggregating['std_by_mean'] = 'mean'
        columns_out0 += ['std_by_mean']
       
    if 'amp_moments' in frame.columns:
        #if it is there, we assume that it's filled with amplitudes
        aggregating['amp_moments'] = 'mean'
        columns_out0 += ['amp_moments']
    
    if 'sig_moments' in frame.columns:
        #if it is there, we assume that it's filled with amplitudes
        aggregating['sig_moments'] = 'mean'
        columns_out0 += ['sig_moments'] 
    frame_avg = frame.groupby(grouping).agg(aggregating).reset_index()
    frame_avg['amp'] = frame_avg['vis'].apply(np.abs)
    frame_avg['phase']=frame_avg['vis'].apply(lambda x: np.angle(x)*180./np.pi)
    frame_avg['snr'] = frame_avg['amp']/frame_avg['sigma']
    columns_out = list(set(columns_out0)&set(frame_avg.columns))+list(grouping)+list(set(['source','datetime','amp',phase_type,'snr','sigma','number'])&set(frame_avg.columns))
    columns_out = list(set(columns_out))
    frame_avg_out = frame_avg[columns_out].copy()
    if 'gmst' not in frame_avg_out.columns:
        try: util.add_gmst(frame_avg_out)
        except: pass
    frame_avg_out = add_mjd(frame_avg_out)
    frame_avg_out = add_fmjd(frame_avg_out)

    frame_avg_out = frame_avg_out.loc[:,~frame_avg_out.columns.duplicated()]
    return frame_avg_out

def coh_avg_bsp(frame,tavg='scan',columns_out0=[]):
    if 'band' not in frame.columns:
        frame['band'] = np.nan
    if 'sigma' not in frame.columns:
        frame['sigma'] = frame['amp']/frame['snr']
    if 'snr' not in frame.columns:
        frame['snr'] = frame['amp']/frame['sigma']
     
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
    'datetime': 'min',
    'bsp': 'mean',
    #'snr': lambda x: np.sqrt(np.sum(x**2)),
    'sigma': lambda x: np.sqrt(np.sum(x**2))/len(x),
    'number': 'size',
    'cphase_fix_amp': circular_mean}
    
    
    def agg_mean_3tup(x):  
        x0 = np.asarray([float(y[0]) for y in x if y[0]==y[0] ])
        x1 = np.asarray([float(y[1]) for y in x if y[1]==y[1] ])
        x2 = np.asarray([float(y[2]) for y in x if y[2]==y[2] ])
        return(np.mean(x0[x0==x0]),np.mean(x1[x1==x1]),np.mean(x2[x2==x2]))

    def agg_snr_3tup(x):  
        x0 = np.asarray([float(y[0]) for y in x if y[0]==y[0] ])
        x1 = np.asarray([float(y[1]) for y in x if y[1]==y[1] ])
        x2 = np.asarray([float(y[2]) for y in x if y[2]==y[2] ])
        return(np.sqrt(np.sum(x0[x0==x0]**2)),np.sqrt(np.sum(x1[x1==x1]**2)),np.sqrt(np.sum(x2[x2==x2]**2)))


    if 'EB_sigma' in columns_out0:
        #aggregating['EB_sigma'] = lambda x: bootstrap2(x,int(1e3),lambda y: wrapped_std(y))[0]
        aggregating['EB_sigma'] = lambda x: bootstrap_circ_mean(x,int(1e3))[1]
        aggregating['EB_cphase'] = lambda x: bootstrap_circ_mean(x,int(1e3))[0]
        frame['EB_sigma'] =  frame['cphase']
        frame['EB_cphase'] =  frame['cphase']
        columns_out0 += ['EB_cphase','EB_sigma']

    if 'median_cp' not in frame.columns:
        frame['median_cp']=frame['cphase']
    aggregating['median_cp'] = circular_median
    columns_out0 += ['median_cp']

    if 'mad_cp' not in frame.columns:
        frame['mad_cp']=frame['cphase']
    aggregating['mad_cp'] = mad_deg
    columns_out0 += ['mad_cp']

    if 'fracpols' in frame.columns:
        aggregating['fracpols'] = agg_mean_3tup
        columns_out0 += ['fracpols']
    
    if 'amps' in frame.columns:
        aggregating['amps'] = agg_mean_3tup
        columns_out0 += ['amps']

    if 'snrs' in frame.columns:
        aggregating['snrs'] = agg_snr_3tup
        columns_out0 += ['snrs']

    #AVERAGING-------------------------------    
    if tavg=='scan': #average for entire scan
        frame_avg = frame.groupby(groupingSc, observed=True).agg(aggregating)
        
    else: # average for 
        frame['round_time'] = list(map(lambda x: np.round((x- datetime.datetime(2017,4,4)).total_seconds()/float(tavg)),frame.datetime))
        grouping = groupingSc+['round_time']
        frame_avg = frame.groupby(grouping, observed=True).agg(aggregating)

    frame_avg['amp'] = frame_avg['bsp'].apply(np.abs)
    frame_avg['cphase']=frame_avg['bsp'].apply(lambda x: np.angle(x)*180./np.pi) #deg
    #frame_avg['sigma']=frame_avg['amp']/frame_avg['snr']
    frame_avg['snr']=frame_avg['amp']/frame_avg['sigma']
    frame_avg['sigmaCP'] = 1./frame_avg['snr']*180./np.pi #deg
    columns_out0 += ['cphase_fix_amp']
    frame_avg = frame_avg.reset_index()
    columns_out = list(grouping0)+list(set(columns_out0)|set(['source','datetime','amp','cphase','snr','number','sigma','sigmaCP'])&set(frame_avg.columns))
    frame_avg_out = frame_avg[columns_out].copy()
    
    util.add_gmst(frame_avg_out)
    frame_avg_out = add_mjd(frame_avg_out)
    frame_avg_out = add_fmjd(frame_avg_out)
    frame_avg_out['rel_err'] = np.asarray(frame_avg_out['cphase'])/np.asarray(frame_avg_out['sigmaCP'])
    return frame_avg_out


def prepare_ER3_vis(path='/Users/mwielgus/Dropbox (Smithsonian External)/EHT/Data/ReleaseE3/hops/ER3v1/', filen='alist.v6.2s',bands=['lo','hi'],reverse_pol=False,apply_fixes=False):
    from eat.inspect import closures as cl
    bandL=[]
    if 'lo' in bands:
        print('loading lo band data...')
        lo_path = path+'lo/'+filen
        hops_lo = hops.read_alist(lo_path)
        if apply_fixes==True:
            util.fix(hops_lo)
        hops_lo = cl.add_band(hops_lo,'lo')
        bandL+=[hops_lo]

    if 'hi' in bands:
        print('loading hi band data...')
        hi_path = path+'hi/'+filen
        hops_hi = hops.read_alist(hi_path)
        if apply_fixes==True:
            util.fix(hops_hi)
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



def prepare_hops_raw(path='/Users/mwielgus/Dropbox (Smithsonian External)/EHT/Data/ReleaseE3/hops/ER3v1/', filen='alist.v6.2s',bands=['lo','hi'],reverse_pol=False):
    from eat.inspect import closures as cl
    bandL=[]
    if 'lo' in bands:
        print('loading lo band data...')
        lo_path = path+'lo/'+filen
        hops_lo = hops.read_alist(lo_path)
        #util.fix(hops_lo)
        hops_lo = cl.add_band(hops_lo,'lo')
        bandL+=[hops_lo]

    if 'hi' in bands:
        print('loading hi band data...')
        hi_path = path+'hi/'+filen
        hops_hi = hops.read_alist(hi_path)
        #util.fix(hops_hi)
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
    df.drop('unique_sc',axis=1,inplace=True)
    return df

def add_fracpol(df,scan_avg=False):
    if scan_avg==False:
        grouping = ['expt_no','scan_id','band','baseline','datetime']
    else:
        grouping = ['expt_no','scan_id','band','baseline']
    
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
    if scan_avg==False:
        pivpol['uni_fracpol'] = list(zip(pivpol.expt_no,pivpol.scan_id,pivpol.band,pivpol.baseline,pivpol.datetime))
        df['uni_fracpol'] = list(zip(df.expt_no,df.scan_id,df.band,df.baseline,df.datetime))
    else:
        pivpol['uni_fracpol'] = list(zip(pivpol.expt_no,pivpol.scan_id,pivpol.band,pivpol.baseline))
        df['uni_fracpol'] = list(zip(df.expt_no,df.scan_id,df.band,df.baseline))

    dict_fracpol=dict(zip(pivpol.uni_fracpol,pivpol.fracpol))
    df['fracpol'] = list(map(lambda x: dict_fracpol[x], df.uni_fracpol))
    df.drop('uni_fracpol',axis=1,inplace=True)
   
    return df

def add_fracpol_to_scan_cphase(bsp,vis):
    '''
    take scan averaged cphases and add fracpol from scan averaged vis
    '''
    if 'fracpol' not in vis.columns:
        vis = add_fracpol(vis,scan_avg=True)
    
    frac_b = []
   
    for index, row in bsp.iterrows():
        #print(row['scan_id'])
        tr = row['triangle']
        sc = row['scan_id']
        ex = row['expt_no']
        ba = row['band']
        base1 = [tr[0],tr[1]]
        base2 = [tr[1],tr[2]]
        base3 = [tr[2],tr[0]]

        foo_b1 = vis[(vis.scan_id==sc)&(vis.expt_no==ex)&(vis.band==ba)&vis.baseline.str.contains(base1[0])&vis.baseline.str.contains(base1[1])].fracpol
        foo_b2 = vis[(vis.scan_id==sc)&(vis.expt_no==ex)&(vis.band==ba)&vis.baseline.str.contains(base2[0])&vis.baseline.str.contains(base2[1])].fracpol
        foo_b3 = vis[(vis.scan_id==sc)&(vis.expt_no==ex)&(vis.band==ba)&vis.baseline.str.contains(base3[0])&vis.baseline.str.contains(base3[1])].fracpol
    
        foo=()
        if len(foo_b1)>0:
            foo += (np.asarray(foo_b1)[0] ,)
        else: foo+= (np.nan,)
        if len(foo_b2)>0:
            foo += (np.asarray(foo_b2)[0] ,)
        else: foo+= (np.nan,)
        if len(foo_b3)>0:
            foo += (np.asarray(foo_b3)[0] ,)
        else: foo+= (np.nan,)
        
        #print(tr+', '+sc+', '+str(ex)+', '+ba+': '+str(foo))
        
        #print(frac_b)
        #print(row.triangle,foo)
        frac_b += [foo]
        #print(frac_b)

    bsp['fracpol_sc'] = frac_b

    return bsp



def add_snrs_to_scan_cphase(bsp,vis):
    '''
    take scan averaged cphases and add fracpol from scan averaged vis
    '''
    if 'fracpol' not in vis.columns:
        vis = add_fracpol(vis,scan_avg=True)
    
    frac_b = []
   
    for index, row in bsp.iterrows():
        tr = row['triangle']
        sc = row['scan_id']
        ex = row['expt_no']
        ba = row['band']
        po = row['polarization']
        base1 = [tr[0],tr[1]]
        base2 = [tr[1],tr[2]]
        base3 = [tr[2],tr[0]]

        foo_b1 = vis[(vis.scan_id==sc)&(vis.expt_no==ex)&(vis.band==ba)&(vis.polarization==po)&vis.baseline.str.contains(base1[0])&vis.baseline.str.contains(base1[1])].snr
        foo_b2 = vis[(vis.scan_id==sc)&(vis.expt_no==ex)&(vis.band==ba)&(vis.polarization==po)&vis.baseline.str.contains(base2[0])&vis.baseline.str.contains(base2[1])].snr
        foo_b3 = vis[(vis.scan_id==sc)&(vis.expt_no==ex)&(vis.band==ba)&(vis.polarization==po)&vis.baseline.str.contains(base3[0])&vis.baseline.str.contains(base3[1])].snr
    
        foo=()
        if len(foo_b1)>0:
            foo += (np.asarray(foo_b1)[0] ,)
        else: foo+= (np.nan,)
        if len(foo_b2)>0:
            foo += (np.asarray(foo_b2)[0] ,)
        else: foo+= (np.nan,)
        if len(foo_b3)>0:
            foo += (np.asarray(foo_b3)[0] ,)
        else: foo+= (np.nan,)
        
        #print(frac_b)
        #print(row.triangle,foo)
        frac_b += [foo]
        #print(frac_b)

    bsp['snr_st1'] = [x[0] for x in frac_b]
    bsp['snr_st2'] = [x[1] for x in frac_b]
    bsp['snr_st3'] = [x[2] for x in frac_b]

    return bsp


def snr_threshold_scan_vis(data,vis,snr_cut=5.):
    '''
    removes data that doesn't correspond to detection in scan-averaged visibilities (vis)
    '''
    #only leave low snr parallel hands
    vis.drop(list(vis[vis.snr>snr_cut].index.values),inplace=True)
    vis.drop(list(vis[vis.polarization.str[0]!=vis.polarization.str[1]].index.values),inplace=True)
    #print('vis ',np.shape(vis))
    #print('data ',np.shape(data))
    scans_to_remove = list(zip(vis.scan_id,vis.band,vis.baseline))
    scans_to_remove0 = list(zip(vis.scan_id,vis.band))
    scans_to_removeB = list(vis.baseline)
    scans_to_removeC = list(zip(scans_to_remove0,scans_to_removeB))
    
    if 'baseline' in data.columns:
        tester = list(zip(data.scan_id,data.band,data.baseline))
        data['tester'] = list(map(lambda x: x in scans_to_remove, tester ))
    elif 'triangle' in data.columns:
        tester = list(zip(zip(data.scan_id,data.band),data.triangle))
        data['tester'] = list(map(lambda x: (x[0] in scans_to_remove0)&(int(x[1][0] in scans_to_removeB)+int(x[1][1] in scans_to_removeB)+int(x[1][2] in scans_to_removeB)==2), tester ))
    #print(scans_to_remove)
    data['tester'] = list(map(lambda x: x in scans_to_remove, tester ))
    data.drop(list(data[data.tester==True].index.values),inplace=True)
    data.drop('tester', axis=1,inplace=True)

    return data

def sigmaCA_fn(x):
    lambda x: np.sqrt(np.sum(x**2))/len(x)

def avg_camp(frame,tavg='scan',debias='no'):

    if 'band' not in frame.columns:
        frame['band'] = np.nan

    if 'scan_id' not in frame.columns:
        frame['scan_id'] = frame['scan_no_tot']
     

    #print(frame.columns)
    #minimum set of columns that identifies a scan
    grouping0 = ('scan_id','expt_no','band','polarization','quadrangle')
    groupingSc = list(set(['scan_id','expt_no','band','polarization','quadrangle','source'])&set(frame.columns))
    
    frame.drop_duplicates(subset=list(grouping0)+['datetime'], keep='first', inplace=True)
    frame['number']=0.

    #columns_out0 += ['amp_db']
    #frame['amp_db']= list( (np.asarray(frame['amp'])**2)*(1.-2./np.asarray(frame['snr'])**2) )

    aggregating = {#'datetime': lambda x: min(x)+ 0.5*(max(x)-min(x)),
    'datetime': 'min',
    #'amp_db': lambda x: np.sqrt(np.mean(x)),
    'camp': 'mean',
    'number': 'size',
    #'snr': lambda x: np.sqrt(np.sum(x**2))
    'sigmaCA': sigmaCA_fn
    }
    #if debias==True:


    if 'u' in frame.columns:
        aggregating['u'] = 'mean'
        columns_out0 += ['u']
    if 'v' in frame.columns:
        aggregating['v'] = 'mean'
        columns_out0 += ['v']

    #ACTUAL AVERAGING
  
    if tavg=='scan': #average for entire scan
        frame_avg = frame.groupby(groupingSc, observed=True).agg(aggregating)
        
    else: # average for 
        date0 = datetime.datetime(2017,4,4)
        frame['round_time'] = list(map(lambda x: np.round((x- date0).total_seconds()/float(tavg)),frame.datetime))
        grouping = groupingSc+['round_time']
        frame_avg = frame.groupby(grouping, observed=True).agg(aggregating)
        frame_avg = frame_avg.reset_index()
        frame_avg['datetime'] =  list(map(lambda x: date0 + datetime.timedelta(seconds= int(tavg*x)), frame_avg['round_time']))
    
    frame_avg['snr'] = frame_avg['camp']/frame_avg['sigmaCA']

    frame_avg = frame_avg.reset_index()

    #frame_avg['amp'] = frame_avg['vis'].apply(np.abs)
    #frame_avg['phase']=frame_avg['vis'].apply(lambda x: np.angle(x)*180./np.pi)
    #if 'snr' in frame_avg.columns:
    #print(list(frame_avg['snr']))
    #print(frame_avg['snr'])
    #frame_avg['sigma']=frame_avg['amp']/frame_avg['snr']

    #print(frame_avg.columns)
    columns_out = list(grouping0)+list(set(['source','datetime','camp','phase','snr','sigmaCA','number'])&set(frame_avg.columns))
    frame_avg_out = frame_avg[columns_out].copy()
    
    util.add_gmst(frame_avg_out)
    frame_avg_out = add_mjd(frame_avg_out)
    frame_avg_out = add_fmjd(frame_avg_out)
    return frame_avg_out



def generate_closure_time_series(df, ctypes=['CP','LCA','CFP'],sourL='def',polarL=['RR','LL'],exptL='def',out_path='def',min_elem=200):
    from eat.inspect import closures as cl
    import os
    if out_path=='def':
        out_path='Closures_timeseries/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if sourL=='def': sourL = list(df.source.unique())
    if exptL=='def': exptL = [3597,3598,3599,3600,3601]

    if 'band' not in df.columns:
        df['band'] = ''
    bandL = list(df.band.unique())

    if 'CP' in ctypes:
        if 'resid_phas' in df.columns:
            print('Calculating bispectra...')
            cp_df = cl.all_bispectra(df, phase_type='resid_phas')
        elif 'phase' in df.columns:
            print('Calculating bispectra...')
            cp_df = cl.all_bispectra(df, phase_type='phase')
        if 'mjd' not in cp_df.columns:
            cp_df = add_mjd(cp_df)
        triL = list(cp_df.triangle.unique())
        cp_path = out_path+'CP/'
        if not os.path.exists(cp_path):
            os.makedirs(cp_path)
        print('Saving closure phases...')
        for sour in sourL:
            for expt in exptL:
                for band in bandL:
                    for polar in polarL:
                        cp_foo = cp_df
                        for tri in triL:
                            cp_foo2 = cp_foo[(cp_foo.source==sour)&(cp_foo.polarization==polar)&(cp_foo.expt_no==expt)&(cp_foo.band==band)&(cp_foo.triangle==tri)]
                            if np.shape(cp_foo2)[0]>min_elem:
                                namef = 'cp_'+tri+'_'+sour+'_'+str(expt)+'_'+polar+'_'+band+'.txt'
                                print(namef)
                                cp_foo3 = cp_foo2[['mjd','cphase','sigmaCP']]
                                print(cp_path+namef)
                                cp_foo3.to_csv(cp_path+namef,sep=' ',index=False, header=False)

    if 'LCA' in ctypes:
        print('Calculating log closure amplitudes...')
        lca_df = cl.all_quadruples_log(df)
        print('shape LCA:', np.shape(lca_df))
        if 'mjd' not in lca_df.columns:
            lca_df = add_mjd(lca_df)
        quadL = list(lca_df.quadrangle.unique())
        lca_path = out_path+'LCA/'
        if not os.path.exists(lca_path):
            os.makedirs(lca_path)
        print('Saving log closure amplitudes...')
        for sour in sourL:
            for expt in exptL:
                for band in bandL:
                    for polar in polarL:  
                        lca_foo = lca_df                         
                        for quad in quadL:
                            lca_foo2 = lca_foo[(lca_foo.source==sour)&(lca_foo.polarization==polar)&(lca_foo.expt_no==expt)&(lca_foo.band==band)&(lca_foo.quadrangle==quad)]
                            if np.shape(lca_foo2)[0]>min_elem:
                                #namef = 'lca_'+str(quad[0])+'_'+str(quad[1])+'_'+str(quad[2])+'_'+str(quad[3])+'_'+sour+'_'+str(expt)+'_'+polar+'_'+band+'.txt'
                                namef = 'lca_'+quad+'_'+sour+'_'+str(expt)+'_'+polar+'_'+band+'.txt'  
                                print(namef)
                                lca_foo3 = lca_foo2[['mjd','logamp','sigma']]
                                lca_foo3.to_csv(lca_path+namef,sep=' ',index=False, header=False)
                            
    
    if 'CFP' in ctypes:
        print('Calculating closure fractional polarizations...')
        #cfp_df = cl.get_closepols(df)
        cfp_df = cl.get_logclosepols(df)
        if 'mjd' not in cfp_df.columns:
            cfp_df = add_mjd(cfp_df)
        baseL = list(cfp_df.baseline.unique())
        cfp_path = out_path+'CFP/'
        if not os.path.exists(cfp_path):
            os.makedirs(cfp_path)
        print('Saving closure fractional polarizations...')
        print(np.shape(cfp_df))
        for sour in sourL:
            for expt in exptL:
                for band in bandL:
                    cfp_foo = cfp_df       
                    for base in baseL:
                        cfp_foo2 = cfp_foo[(cfp_foo.source==sour)&(cfp_foo.expt_no==expt)&(cfp_foo.band==band)&(cfp_foo.baseline==base)]
                        if np.shape(cfp_foo2)[0]>min_elem:
                            namef = 'cfp_'+base+'_'+sour+'_'+str(expt)+'_'+band+'.txt'
                            print(namef)
                            cfp_foo3 = cfp_foo2[['mjd','fracpol','sigma']]
                            cfp_foo3.to_csv(cfp_path+namef,sep=' ',index=False, header=False)
                        


def generate_vis_amp_time_series(df,sourL='def',polarL=['RR','LL'],exptL='def',out_path='def',min_elem=200):
    from eat.inspect import closures as cl
    import os
    if out_path=='def':
        out_path='Closures_timeseries/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if sourL=='def': sourL = list(df.source.unique())
    if exptL=='def': exptL = list(df.expt_no.unique())

    if 'band' not in df.columns:
        df['band'] = ''
    bandL = list(df.band.unique())

    #GENERATE AMP TEXTFILES    
    if 'mjd' not in df.columns:
        df = add_mjd(df)
    baseL = list(df.baseline.unique())
    amp_path = out_path+'AMP/'
    if not os.path.exists(amp_path):
        os.makedirs(amp_path)
    #print('Saving closure phases...')
    for sour in sourL:
        for expt in exptL:
            for band in bandL:
                for polar in polarL:
                    foo = df
                    for base in baseL:
                        foo2 = foo[(foo.source==sour)&(foo.polarization==polar)&(foo.expt_no==expt)&(foo.band==band)&(foo.baseline==base)]
                        if np.shape(foo2)[0]>min_elem:
                            #namef = 'amp_'+base+'_'+sour+'_'+str(expt)+'_'+polar+'_'+band+'.txt'
                            namef = 'amp_pha_'+base+'_'+sour+'_'+polar+'_'+band+'.txt'
                            print(namef)
                            print(np.shape(foo2)[0],[np.min(foo2.mjd),np.max(foo2.mjd)])
                            foo3 = foo2[['mjd','amp','phase','sigma']]
                            #print(amp_path+namef)
                            foo3.to_csv(amp_path+namef,sep=' ',index=False, header=False)



def bootstrap_circ_mean(data, num_samples=int(1e4), alpha='1s'):
    """Returns bootstrap estimate of 100.0*(1-alpha) CI for statistic."""
    statistic = circular_mean
    if alpha=='1s':
        alpha=0.3173
    elif alpha=='2s':
        alpha=0.0455
    elif alpha=='3s':
        alpha=0.0027
    stat = np.zeros(num_samples)
    data = np.asarray(data)
    n = len(data)
    idx = npr.randint(0, n, (num_samples, n))
    samples = data[idx]
    for cou in range(num_samples):
        stat[cou] = statistic(samples[cou,:])
    
    stat = np.asarray(stat)
    m_stat = circular_mean(stat) #deg

    stat = np.exp(1j*(np.pi/180.)*np.asarray(stat))*np.exp(-1j*(np.pi/180.)*m_stat)
    ang = sorted(np.angle(stat)*180./np.pi)
    boot_mean = np.median(ang) + m_stat
    boot_sig = 0.5*(ang[int((1-alpha/2.0)*num_samples)] - ang[int((alpha/2.0)*num_samples)])
    return boot_mean, boot_sig



def filter_nondetections(vis_short,vis_scan):
    '''
    Filter out the visibilities corresponding to non-detections based on
    data in scan-avg
    '''

def rename_baselines(frame,stations_2lett_1lett):
    t1 = list(map(lambda x: x.split('-')[0],frame.baseline))
    t2 = list(map(lambda x: x.split('-')[1],frame.baseline))
    frame['baseline'] = list(map(lambda x: stations_2lett_1lett[x[0].decode('unicode_escape')]+stations_2lett_1lett[x[1].decode('unicode_escape')],zip(t1,t2)))
    return frame



def old_format(df):
    """
    turns polrep circ type with rrvis, llvis columns into old format used for figures etc
    """
    cols = df.columns
    pols = ['rrvis','llvis','lrvis','rlvis']
    base = [x for x in cols if x not in pols]
    

    df_rr = df[base+['rrvis']].copy()
    #df_rr = df_rr[df_rr.rrvis==df_rr.rrvis].copy()
    df_rr['vis']=df_rr['rrvis']
    df_rr['sigma']=df_rr['rrsigma']
    df_rr['polarization']='RR'
    df_rr.drop('rrvis',axis=1,inplace=True)
    df_rr.drop('rrsigma',axis=1,inplace=True)

    df_ll = df[base+['llvis']].copy()
    #df_ll = df_ll[df_ll.llvis==df_ll.llvis].copy()
    df_ll['vis']=df_ll['llvis']
    df_ll['sigma']=df_ll['llsigma']
    df_ll['polarization']='LL'
    df_ll.drop('llvis',axis=1,inplace=True)
    df_ll.drop('llsigma',axis=1,inplace=True)

    df_rl = df[base+['rlvis']].copy()
    #df_rl = df_rl[df_rl.rlvis==df_rl.rlvis].copy()
    df_rl['vis']=df_rl['rlvis']
    df_rl['sigma']=df_rl['rlsigma']
    df_rl['polarization']='RL'
    df_rl.drop('rlvis',axis=1,inplace=True)
    df_rl.drop('rlsigma',axis=1,inplace=True)

    df_lr = df[base+['lrvis']].copy()
    #df_lr = df_lr[df_lr.lrvis==df_lr.lrvis].copy()
    df_lr['vis']=df_lr['lrvis']
    df_lr['sigma']=df_lr['lrsigma']
    df_lr['polarization']='LR'
    df_lr.drop('lrvis',axis=1,inplace=True)
    df_lr.drop('lrsigma',axis=1,inplace=True)

    common = list(set(df_rr.columns)&set(df_ll.columns)&set(df_rl.columns)&set(df_lr.columns))
    df_old = pd.concat([df_rr[common].copy(),df_ll[common].copy(),df_rl[common].copy(),df_lr[common].copy()],ignore_index=True)
    df_old['amp'] = np.abs(df_old['vis'])
    df_old['phase'] = np.angle(df_old['vis'])*180./np.pi
    df_old['snr'] = df_old['amp']/df_old['sigma']

    return df_old


def prepare_polgains(vis,band=None,snr_cut=None):
    '''
    Adds columns related to polcal gains inspection
    '''
    vis = vis[vis.polarization.str[0]==vis.polarization.str[1]]
    if snr_cut is None: pass
    else: vis=vis[vis.snr>snr_cut]
    if band is None: pass
    else: vis = vis[vis.band==band]
        
    visRR = vis[vis.polarization=='RR']
    visLL = vis[vis.polarization=='LL']
    visRR2,visLL2 = match_frames(visRR.copy(),visLL.copy(),['scan_id','band','baseline'])
    visRR2['ampR'] = visRR2['amp']
    visRR2['ampL'] = visLL2['amp']
    visRR2['phaseR'] = visRR2['phase']
    visRR2['phaseL'] = visLL2['phase']
    visRR2['sigmaR'] = visRR2['sigma']
    visRR2['sigmaL'] = visLL2['sigma']
    visRR2['snrL'] = visLL2['snr']
    visRR2['snrR'] = visRR2['snr']

    visRR2['RLphase'] = np.mod(visRR2['phaseR'] - visRR2['phaseL'] +180,360)-180
    visRR2['RLphaseErr'] = np.sqrt(1./np.asarray(visRR2['snr'])**2 + 1./np.asarray(visLL2['snr'])**2)*180./np.pi
    visRR2['AmpRatio'] = np.asarray(visRR2.ampR)/np.asarray(visRR2.ampL)
    visRR2['AmpRatioErr'] = visRR2['AmpRatio']*np.sqrt(np.asarray(1./visRR2['snrL'])**2 + np.asarray(1./visRR2['snrR'])**2)

    visRR2['baseline'] = list(map(str,visRR2['baseline']))
    visRR2=visRR2.dropna(subset=['ampR','ampL','phaseR','phaseL','sigmaR','sigmaL'])
    polgains = visRR2.copy()
    return polgains


def prepare_bandgains(vis,pol=None,snr_cut=None):
    '''
    Adds columns related to polcal gains inspection
    '''
    #vis = vis[vis.polarization.str[0]==vis.polarization.str[1]]
    if snr_cut is None: pass
    else: vis=vis[vis.snr>snr_cut]
    if pol is None: pass
    else: vis = vis[vis.polarization==pol]
        
    visLO = vis[vis.band=='lo']
    visHI = vis[vis.band=='hi']
    visLO2,visHI2 = match_frames(visLO.copy(),visHI.copy(),['scan_id','polarization','baseline'])
    visLO2['ampLO'] = visLO2['amp']
    visLO2['ampHI'] = visHI2['amp']
    visLO2['phaseLO'] = visLO2['phase']
    visLO2['phaseHI'] = visHI2['phase']
    visLO2['sigmaLO'] = visLO2['sigma']
    visLO2['sigmaHI'] = visHI2['sigma']
    visLO2['snrLO'] = visLO2['snr']
    visLO2['snrHI'] = visHI2['snr']

    visLO2['LOHIphase'] = np.mod(visLO2['phaseLO'] - visLO2['phaseHI'] +180,360)-180
    visLO2['LOHIphaseErr'] = np.sqrt(1./np.asarray(visLO2['snrLO'])**2 + 1./np.asarray(visLO2['snrHI'])**2)*180./np.pi
    visLO2['AmpRatio'] = np.asarray(visLO2.ampLO)/np.asarray(visLO2.ampHI)
    visLO2['AmpRatioErr'] = visLO2['AmpRatio']*np.sqrt(np.asarray(1./visLO2['snrLO'])**2 + np.asarray(1./visLO2['snrHI'])**2)

    visLO2['baseline'] = list(map(str,visLO2['baseline']))
    visLO2=visLO2.dropna(subset=['ampLO','ampHI','phaseLO','phaseHI','sigmaLO','sigmaHI','snrLO','snrHI'])
    bandgains = visLO2.copy()
    return bandgains