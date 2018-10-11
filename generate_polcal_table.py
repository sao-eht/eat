#!/usr/bin/env python2
#
# Maciek Wielgus 02/Oct/2018

from __future__ import division
import numpy as np
import pandas as pd
import sys,os
from eat.inspect import closures as cl
from eat.io import hops, util
from eat.hops import util as hu
from eat.polcal import polcal
from eat.inspect import utils as ut
import ehtim as eh
import datetime
from astropy.time import Time
from eat.polcal import polcal
#import weightedstats as ws

Z2AZ = {'Z':'AZ', 'P':'PV', 'S':'SM', 'R':'SR','J':'JC', 'A':'AA','X':'AP', 'L':'LM','Y':'SP'}

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Mean, weighted mean, median, and weighted median.
WeightedStats includes four functions (mean, weighted_mean, median,
weighted_median) which accept lists as arguments, and two functions
(numpy_weighted_mean, numpy weighted_median) which accept either lists
or numpy arrays.
Example:
    import weightedstats as ws
    my_data = [1, 2, 3, 4, 5]
    my_weights = [10, 1, 1, 1, 9]
    # Ordinary (unweighted) mean and median
    ws.mean(my_data)    # equivalent to ws.weighted_mean(my_data)
    ws.median(my_data)  # equivalent to ws.weighted_median(my_data)

    # Weighted mean and median
    ws.weighted_mean(my_data, weights=my_weights)
    ws.weighted_median(my_data, weights=my_weights)
    # Special weighted mean and median functions for use with numpy arrays
    ws.numpy_weighted_mean(my_data, weights=my_weights)
    ws.numpy_weighted_median(my_data, weights=my_weights)
"""

def mean(data):
    """Calculate the mean of a list."""
    return sum(data) / float(len(data))

def weighted_mean(data, weights=None):
    """Calculate the weighted mean of a list."""
    if weights is None:
        return mean(data)
    total_weight = float(sum(weights))
    weights = [weight / total_weight for weight in weights]
    w_mean = 0
    for i, weight in enumerate(weights):
        w_mean += weight * data[i]
    return w_mean

def numpy_weighted_mean(data, weights=None):
    """Calculate the weighted mean of an array/list using numpy."""
    import numpy as np
    weights = np.array(weights).flatten() / float(sum(weights))
    return np.dot(np.array(data), weights)

def median(data):
    """Calculate the median of a list."""
    data.sort()
    num_values = len(data)
    half = num_values // 2
    if num_values % 2:
        return data[half]
    return 0.5 * (data[half-1] + data[half])

def weighted_median(data, weights=None):
    """Calculate the weighted median of a list."""
    data=list(data)
    if weights is None:
        ret = median(data)
        if ret is None: return 1.
        else: return ret
    else: weights=list(weights)
    midpoint = 0.5 * sum(weights)
    if any([j > midpoint for j in weights]):
        ret = data[weights.index(max(weights))]
        if ret is None: return 1.
        else: return ret
    if any([j > 0 for j in weights]):
        sorted_data, sorted_weights = zip(*sorted(zip(data, weights)))
        cumulative_weight = 0
        below_midpoint_index = 0
        while cumulative_weight <= midpoint:
            below_midpoint_index += 1
            cumulative_weight += sorted_weights[below_midpoint_index-1]
        cumulative_weight -= sorted_weights[below_midpoint_index-1]
        if cumulative_weight - midpoint < sys.float_info.epsilon:
            bounds = sorted_data[below_midpoint_index-2:below_midpoint_index]
            ret = sum(bounds) / float(len(bounds))
            if ret is None: return 1.
            else: return ret
        ret = sorted_data[below_midpoint_index-1]
        if ret is None: return 1.
        else: return ret

def numpy_weighted_median(data, weights=None):
    """Calculate the weighted median of an array/list using numpy."""
    import numpy as np
    if weights is None:
        return np.median(np.array(data).flatten())
    data, weights = np.array(data).flatten(), np.array(weights).flatten()
    if any(weights > 0):
        sorted_data, sorted_weights = map(np.array, zip(*sorted(zip(data, weights))))
        midpoint = 0.5 * sum(sorted_weights)
        if any(weights > midpoint):
            return (data[weights == np.max(weights)])[0]
        cumulative_weight = np.cumsum(sorted_weights)
        below_midpoint_index = np.where(cumulative_weight <= midpoint)[0][-1]
        if cumulative_weight[below_midpoint_index] - midpoint < sys.float_info.epsilon:
            return np.mean(sorted_data[below_midpoint_index:below_midpoint_index+2])
        return sorted_data[below_midpoint_index+1]

def poly_from_str(strcoeffs):
    '''from string with coefficients to polynomial
    '''
    coeffs = list(map(float, strcoeffs.split(',')))
    return np.polynomial.polynomial.Polynomial(coeffs)

def apply_correction(corrected,ratios,station):
    '''applies polarimetric correction from 'ratios' df to 'corrected' df, but only to chosen station
    '''
    for cou,row in ratios.iterrows():
        if row.station==station:
            corrected_foo1=corrected[(corrected.mjd>=row.mjd_start)&(corrected.mjd<=row.mjd_stop)&(corrected.baseline.str[0]==row.station)].copy()
            corrected_foo2=corrected[(corrected.mjd>=row.mjd_start)&(corrected.mjd<=row.mjd_stop)&(corrected.baseline.str[1]==row.station)].copy()
            corrected_rest=corrected[~((corrected.mjd>=row.mjd_start)&(corrected.mjd<=row.mjd_stop)&(corrected.baseline.str.contains(row.station)))].copy()
            polyf = poly_from_str(str(row.ratio_phas))
            delta_phas1 = polyf(corrected_foo1['mjd'] - row.mjd_start)
            delta_phas2 = polyf(corrected_foo2['mjd'] - row.mjd_start)
            corrected_foo1['phaseL'] = corrected_foo1['phaseL'] +delta_phas1
            corrected_foo2['phaseL'] = corrected_foo2['phaseL'] -delta_phas2
            polyamp = poly_from_str(str(row.ratio_amp))
            delta_amp1 = polyamp(corrected_foo1['mjd'] - row.mjd_start)
            delta_amp2 = polyamp(corrected_foo2['mjd'] - row.mjd_start)
            corrected_foo1['ampL'] = corrected_foo1['ampL']*delta_amp1
            corrected_foo2['ampL'] = corrected_foo2['ampL']*delta_amp2
            corrected = pd.concat([corrected_foo1,corrected_foo2,corrected_rest],ignore_index=True)
    corrected['RLphase'] = np.mod( corrected['phaseR'] - corrected['phaseL'] +180,360)-180
    corrected['RLphaseErr'] = np.sqrt(1./np.asarray(corrected.snrL)**2 + 1./np.asarray(corrected.snrR)**2)*180./np.pi
    corrected['AmpRatio'] = np.asarray(corrected.ampR)/np.asarray(corrected.ampL)
    corrected['AmpRatioErr'] = corrected['AmpRatio']*np.sqrt(np.asarray(1./corrected['snrL'])**2 + np.asarray(1./corrected['snrR'])**2)
    return corrected


def get_polcal(path_data,path_out,degSMA=3,degAPEX=1,snr_cut=1.):

    if path_data.endswith('.pic'):
        vis = pd.read_pickle(path_data)
        vis.drop(list(vis[vis.baseline.str.contains('R')].index.values),inplace=True)

    elif (path_data.endswith('.hdf')) or (path_data.endswith('.h5')):
        vis=pd.read_hdf(path_data)
        vis.drop(list(vis[vis.baseline.str.contains('R')].index.values),inplace=True)

    else: raise Exception('Use .pic or .h5 or .hdf files!')

    #PREPARE DATASET FOR POLCAL GAINS CALCULATION
    vis=vis[vis.snr>snr_cut].copy()
    vis = vis[vis.polarization.str[0]==vis.polarization.str[1]]
    #vis = vis[vis.band==band]
    visRR = vis[vis.polarization=='RR']
    visLL = vis[vis.polarization=='LL']
    visRR2,visLL2 = ut.match_frames(visRR.copy(),visLL.copy(),['scan_id','band','baseline'])
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
    corrected = visRR2.copy()
    ##-------------------------------------------------------
    #INITIALIZE POLCAL RATIOS TABLE
    stationL = list(set(list(map(lambda x: x[0],vis.baseline))+list(map(lambda x: x[1],vis.baseline))))
    exptL = list(vis.expt_no.unique())
    bandL = list(vis.band.unique())
    ratios = pd.DataFrame(columns = ['station','mjd_start','mjd_stop','ratio_amp', 'ratio_phas'])
    #time margin for calibration [h]
    toff = 2./24.
    ##-------------------------------------------------------
    #ALMA CALIBRATION
    #ALMA IS ASSUMED TO HAVE 1+0j gains and used as reference
    sourLA = list(vis[vis.baseline.str.contains('A')].source.unique())
    ratios = pd.concat([ratios,pd.DataFrame([{'station':'A',
                                 'mjd_start': vis.mjd.min() - toff,
                                 'mjd_stop': vis.mjd.max() + toff,
                                 'ratio_amp': "%.3f" % 1.,
                                 'ratio_phas': "%.3f" % -0.}])],ignore_index=True)
    corrected = apply_correction(corrected,ratios,'A')
    ##-------------------------------------------------------
    #LMT CALIBRATION
    #LMT is calibrated with a single value for all nights from ALMA-LMT baseline
    sourLL = list(vis[vis.baseline.str.contains('L')].source.unique())
    base='AL'
    foo = visRR2[visRR2['baseline']==base]
    wph =weighted_median(foo.RLphase, weights=1./np.asarray(foo.RLphaseErr))
    wam =weighted_median(foo.AmpRatio, weights=1./np.asarray(foo.AmpRatioErr))
    ratios = pd.concat([ratios,pd.DataFrame([{'station':'L',
                                    'mjd_start': vis.mjd.min() - toff,
                                    'mjd_stop': vis.mjd.max() + toff,
                                    'ratio_amp': "%.3f" % wam,
                                    'ratio_phas': "%.3f" % -wph}])],ignore_index=True)
    corrected = apply_correction(corrected,ratios,'L')
    ##-------------------------------------------------------
    #PV CALIBRATION
    #PV is calibrated with a single value for all nights from ALMA-PV baseline
    sourLP = list(vis[vis.baseline.str.contains('P')].source.unique())
    base='AP'
    foo = visRR2[visRR2['baseline']==base]
    wph =weighted_median(foo.RLphase, weights=1./np.asarray(foo.RLphaseErr))
    wam =weighted_median(foo.AmpRatio, weights=1./np.asarray(foo.AmpRatioErr))
    ratios = pd.concat([ratios,pd.DataFrame([{'station':'P',
                                    'mjd_start': vis.mjd.min() - toff,
                                    'mjd_stop': vis.mjd.max() + toff,
                                    'ratio_amp': "%.3f" % wam,
                                    'ratio_phas': "%.3f" % -wph}])],ignore_index=True)
    corrected = apply_correction(corrected,ratios,'P')
    ##-------------------------------------------------------
    #SPT CALIBRATION
    #SPT is calibrated with single value from ALMA baseline, but on night 3597 LMT is used instead
    sourLY = list(vis[vis.baseline.str.contains('Y')].source.unique())
    base='AY'
    foo = visRR2[(visRR2['baseline']==base)&(visRR2.expt_no!=3597)]
    wph =weighted_median(foo.RLphase, weights=1./np.asarray(foo.RLphaseErr))
    wam =weighted_median(foo.AmpRatio, weights=1./np.asarray(foo.AmpRatioErr))
    #print(wam)
    ratios = pd.concat([ratios,pd.DataFrame([{'station':'Y',
                                    'mjd_start': foo.mjd.min() - toff,
                                    'mjd_stop': foo.mjd.max() + toff,
                                    'ratio_amp': "%.3f" % wam,
                                    'ratio_phas': "%.3f" % -wph}])],ignore_index=True)
    base='LY'
    foo = visRR2[(visRR2['baseline']==base)&(visRR2.expt_no==3597)]
    wph =weighted_median(foo.RLphase, weights=1./np.asarray(foo.RLphaseErr))
    wam =weighted_median(foo.AmpRatio, weights=1./np.asarray(foo.AmpRatioErr))
    doo = float(ratios[(ratios.station=='L')].ratio_phas)
    goo = -wph+float(doo)
    ratios = pd.concat([ratios,pd.DataFrame([{'station':'Y',
                                    'mjd_start': foo.mjd.min() - toff,
                                    'mjd_stop': foo.mjd.max() + toff,
                                    'ratio_amp': "%.3f" % wam,
                                    'ratio_phas': "%.3f" % goo}])],ignore_index=True)
    corrected = apply_correction(corrected,ratios,'Y')
    ##-------------------------------------------------------
    #SMT is calibrated with single value per night, from ALMA-SMT baseline
    #05/Oct/2018 SMT 3601 end of SGRA track added linear slope fit
    #
    sourLZ = list(vis[vis.baseline.str.contains('Z')].source.unique())
    exptL = list(vis.expt_no.unique())
    base='AZ'
    foo = visRR2[visRR2['baseline']==base]
    wam =weighted_median(foo.AmpRatio, weights=1./np.asarray(foo.AmpRatioErr))
    for expt in exptL:
        foo2 = foo[(foo.expt_no==expt)&(foo.mjd<57854.368)]
        foo_for_mjd = visRR2[(visRR2['expt_no']==expt)&(visRR2.mjd<57854.368)]
        wph =weighted_median(foo2.RLphase, weights=1./np.asarray(foo2.RLphaseErr))
        mjd_start = foo_for_mjd.mjd.min() - toff
        mjd_stop = np.minimum(foo_for_mjd.mjd.max() + toff,57854.368)
        ratios = pd.concat([ratios,pd.DataFrame([{'station':'Z',
                            'mjd_start': mjd_start,
                            'mjd_stop': mjd_stop,
                            'ratio_amp': "%.3f" % wam,
                            'ratio_phas': "%.3f" % -wph}])],ignore_index=True)

    foo2 = foo[(foo.mjd>57854.368)]
    foo_for_mjd = visRR2[(visRR2.mjd>57854.368)]
    wph =weighted_median(foo2.RLphase, weights=1./np.asarray(foo2.RLphaseErr))
    mjd_start = 57854.368
    mjd_stop = foo_for_mjd.mjd.max() + toff
    fit_coef = np.polyfit(np.asarray(foo2.mjd) - mjd_start, np.unwrap(np.asarray(foo2.RLphase)*np.pi/180)*180/np.pi, deg=1, full=False, w=1./np.asarray(foo2['RLphaseErr']))
    ratios = pd.concat([ratios,pd.DataFrame([{'station':'Z',
                        'mjd_start': mjd_start,
                        'mjd_stop': mjd_stop,
                        'ratio_amp': "%.3f" % wam,
                        'ratio_phas': "{}, {}".format( "%.3f" % -fit_coef[1], "%.3f" % -fit_coef[0])}])],ignore_index=True)
    corrected = apply_correction(corrected,ratios,'Z')
    ##-------------------------------------------------------
    #APEX is calibrated with linear functions on predefined time intervals
    #it's often just 1 interval per night, but e.g. 3601 is 4 segments

    '''
    #OLD APEX CALIBRATION, I KEEP IT HERE FOR NOW - MW 05/Oct/2018
    sourLX = list(vis[vis.baseline.str.contains('X')].source.unique())
    exptL = list(vis.expt_no.unique())

    base='AX'
    otherB='L'
    fooAX = visRR2[visRR2['baseline']==base]
    fooXL = visRR2[visRR2['baseline']=='X'+otherB].copy()
    fooLX=fooXL.copy()
    fooLX['RLphase'] = -fooXL['RLphase']
    foo=pd.concat([fooAX,fooLX],ignore_index=True)

    #fooAX = visRR2[visRR2['baseline']==base]
    #fooXL = visRR2[visRR2['baseline']=='XL']
    #fooXL['RLphase'] = -fooXL['RLphase']
    #foo=pd.concat([fooAX,fooXL],ignore_index=True)
    foo = foo[foo.amp==foo.amp]
    foo = foo[foo.phase==foo.phase]

    wam =ws.weighted_median(foo.AmpRatio, weights=1./np.asarray(foo.AmpRatioErr))
    for expt in exptL:
        foo2 = foo[foo.expt_no==expt]
        if expt==3601:
            #print(foo2.source.unique())
            sources3601a = ['OJ287', '1055+018']
            sources3601b = ['M87', '3C279']
            sources3601c = ['J1924-2914', 'SGRA']
            foo2a = foo2[list(map(lambda x: x in sources3601a, foo2.source))]
            foo2b = foo2[list(map(lambda x: x in sources3601b, foo2.source))]
            foo2c = foo2[list(map(lambda x: x in sources3601c, foo2.source))&(foo2.mjd<57854.58368287)]
            foo2d = foo2[(foo2.mjd>57854.58368287)]
            mjd_start_a = foo2a.mjd.min() - 0.005
            mjd_stop_a = foo2a.mjd.max() + 0.005
            mjd_start_b = foo2b.mjd.min() - 0.005
            mjd_stop_b = foo2b.mjd.max() + 0.005
            mjd_start_c = foo2c.mjd.min() - 0.005
            mjd_stop_c = foo2c.mjd.max() + 0.004
            mjd_start_d = foo2d.mjd.min() - 0.004
            mjd_stop_d = foo2d.mjd.max() + 0.005

            fit_coef_a = np.polyfit(np.asarray(foo2a.mjd) - mjd_start_a, np.unwrap(np.asarray(foo2a.RLphase)*np.pi/180)*180/np.pi, deg=1, full=False, w=1./np.asarray(foo2a['RLphaseErr']))
            fit_coef_b = np.polyfit(np.asarray(foo2b.mjd) - mjd_start_b, np.unwrap(np.asarray(foo2b.RLphase)*np.pi/180)*180/np.pi, deg=1, full=False, w=1./np.asarray(foo2b['RLphaseErr']))
            fit_coef_c = np.polyfit(np.asarray(foo2c.mjd) - mjd_start_c, np.unwrap(np.asarray(foo2c.RLphase)*np.pi/180)*180/np.pi, deg=1, full=False, w=1./np.asarray(foo2c['RLphaseErr']))
            fit_coef_d = np.polyfit(np.asarray(foo2d.mjd) - mjd_start_d, np.unwrap(np.asarray(foo2d.RLphase)*np.pi/180)*180/np.pi, deg=1, full=False, w=1./np.asarray(foo2d['RLphaseErr']))

            wph_d = ws.weighted_median(foo2d.RLphase, weights=1./np.asarray(foo2d.RLphaseErr))
            #foo=-fit_coef_d[1]#+float(ratios[ratios.station=='L'].ratio_phas)
            foo = -wph_d + float(ratios[(ratios.station==otherB)&(ratios.mjd_stop>57854.58368287)].ratio_phas)

            #hacky, it calibrates to LMT
            #foo=-fit_coef_d[1]+float(ratios[ratios.station=='L'].ratio_phas)

            ratios = pd.concat([ratios,pd.DataFrame([{'station':'X',
                                'mjd_start': mjd_start_a,
                                'mjd_stop': mjd_stop_a,
                                'ratio_amp': "%.3f" % wam,
                                'ratio_phas': "{}, {}".format( "%.3f" % -fit_coef_a[1], "%.3f" % -fit_coef_a[0])}])],ignore_index=True)
            ratios = pd.concat([ratios,pd.DataFrame([{'station':'X',
                                'mjd_start': mjd_start_b,
                                'mjd_stop': mjd_stop_b,
                                'ratio_amp': "%.3f" % wam,
                                'ratio_phas': "{}, {}".format( "%.3f" % -fit_coef_b[1], "%.3f" % -fit_coef_b[0])}])],ignore_index=True)
            ratios = pd.concat([ratios,pd.DataFrame([{'station':'X',
                                'mjd_start': mjd_start_c,
                                'mjd_stop': mjd_stop_c,
                                'ratio_amp': "%.3f" % wam,
                                'ratio_phas': "{}, {}".format( "%.3f" % -fit_coef_c[1], "%.3f" % -fit_coef_c[0])}])],ignore_index=True)
            ratios = pd.concat([ratios,pd.DataFrame([{'station':'X',
                                'mjd_start': mjd_start_d,
                                'mjd_stop': mjd_stop_d,
                                'ratio_amp': "%.3f" % wam,
                                #'ratio_phas': "{}, {}".format( "%.3f" % -fit_coef_d[1], "%.3f" % -fit_coef_d[0]  )
                                'ratio_phas': "%.3f" % foo
                                }])],ignore_index=True)
        else:
            foo_for_mjd = visRR2[(visRR2['expt_no']==expt)]
            mjd_start = foo_for_mjd.mjd.min() - toff
            mjd_stop = foo_for_mjd.mjd.max() + toff
            fit_coef = np.polyfit(np.asarray(foo2.mjd) - mjd_start, np.unwrap(np.asarray(foo2.RLphase)*np.pi/180)*180/np.pi, deg=1, full=False, w=1./np.asarray(foo2['RLphaseErr']))
            ratios = pd.concat([ratios,pd.DataFrame([{'station':'X',
                        'mjd_start': mjd_start,
                        'mjd_stop': mjd_stop,
                        'ratio_amp': "%.3f" % wam,
                        'ratio_phas': "{}, {}".format( "%.3f" % -fit_coef[1], "%.3f" % -fit_coef[0])}])],ignore_index=True)
    '''

    mjd_startAP = [57847.92,57848.06,57848.25,57849.00,57850.15,57852.95,57853.90,57854.02,57854.37]
    mjd_stopAP =  [57848.06,57848.25,57848.68,57849.64,57850.85,57853.65,57854.02,57854.37,57854.66]

    deg=degAPEX
    strratio = ('{}, '*(deg+1))[:-2]

    foo=corrected
    fooAX = foo[foo['baseline']=='AX'].copy()
    if 'XL' in list(foo.baseline.unique()):
        fooXL = foo[foo['baseline']=='XL'].copy()
        fooLX=fooXL.copy()
        fooLX['RLphase'] = -fooXL['RLphase']
        fooLX['baseline'] = 'LX'
        foo=pd.concat([fooAX,fooLX],ignore_index=True)
    elif 'LX' in list(foo.baseline.unique()):
        fooXL = foo[foo['baseline']=='XL'].copy()
        foo=pd.concat([fooAX,fooLX],ignore_index=True)
    else:
        foo=fooAX
    foo=foo.sort_values('mjd').copy()
    wam =weighted_median(foo.AmpRatio, weights=1./np.asarray(foo.AmpRatioErr))
    for cou, mjd_sta in enumerate(mjd_startAP):
        try:
            mjd_sto=mjd_stopAP[cou]
            #print([mjd_sta,mjd_sto])
            foo2=foo[(foo.mjd>mjd_sta)&(foo.mjd<=mjd_sto)]
            fit_coef = np.polyfit(np.asarray(foo2.mjd) - mjd_sta, np.unwrap(np.asarray(foo2.RLphase)*np.pi/180)*180/np.pi, deg=deg, full=False, w=1./np.asarray(foo2['RLphaseErr']))
            listcoef = ["%.3f" % -fit_coef[cou] for cou in range(deg,-1,-1)]
            ratios = pd.concat([ratios,pd.DataFrame([{'station':'X',
                                    'mjd_start': mjd_sta,
                                    'mjd_stop': mjd_sto,
                                    'ratio_amp': "%.3f" % wam,
                                    'ratio_phas': strratio.format(*listcoef) }])],ignore_index=True)
        except: continue
    corrected = apply_correction(corrected,ratios,'X')

    ##-------------------------------------------------------
    #For SMA we manually specify mjd ranges for the 3rd order polynomial fitting

    mjd_startV = [57848.02,57848.42,57849.10,57849.40,57850.40,57853.00,57853.07,57853.18,57853.42,57854.10,57854.40]
    mjd_stopV =  [57848.42,57848.80,57849.40,57849.70,57850.90,57853.07,57853.18,57853.42,57853.70,57854.40,57854.70]

    deg=degSMA
    strratio = ('{}, '*(deg+1))[:-2]
    #################
    foo=corrected[corrected.baseline.str[1]=='S']
    #only use ALMA, LMT, SMT
    foo=foo[(foo.baseline=='AS')|(foo.baseline=='LS')|(foo.baseline=='ZS')]
    foo=foo.sort_values('mjd').copy()
    wam =weighted_median(foo.AmpRatio, weights=1./np.asarray(foo.AmpRatioErr))
    for cou, mjd_sta in enumerate(mjd_startV):
        try:
            mjd_sto=mjd_stopV[cou]
            foo2=foo[(foo.mjd>mjd_sta)&(foo.mjd<=mjd_sto)]
            fit_coef = np.polyfit(np.asarray(foo2.mjd) - mjd_sta, np.unwrap(np.asarray(foo2.RLphase)*np.pi/180)*180/np.pi, deg=deg, full=False, w=1./np.asarray(foo2['RLphaseErr']))
            listcoef = ["%.3f" % -fit_coef[cou] for cou in range(deg,-1,-1)]
            ratios = pd.concat([ratios,pd.DataFrame([{'station':'S',
                                    'mjd_start': mjd_sta,
                                    'mjd_stop': mjd_sto,
                                    'ratio_amp': "%.3f" % wam,
                                    'ratio_phas': strratio.format(*listcoef) }])],ignore_index=True)
        except: continue
    corrected = apply_correction(corrected,ratios,'S')

    ##-------------------------------------------------------
    #JCMT is singlepol, SMAR is not really used, so these get 1+0j correction

    ratios = pd.concat([ratios,pd.DataFrame([{'station':'J',
                                 'mjd_start': vis.mjd.min() - toff,
                                 'mjd_stop': vis.mjd.max() + toff,
                                 'ratio_amp': "%.3f" % 1.,
                                 'ratio_phas': "%.3f" % -0.}])])
    ratios = pd.concat([ratios,pd.DataFrame([{'station':'R',
                                    'mjd_start': vis.mjd.min() - toff,
                                    'mjd_stop': vis.mjd.max() + toff,
                                    'ratio_amp': "%.3f" % 1.,
                                    'ratio_phas': "%.3f" % -0.}])])

    ratios2 = ratios.copy()
    ratios2['station']=list(map(lambda x: Z2AZ[x],ratios2['station']))
    ratios2[['station','mjd_start','mjd_stop','ratio_amp','ratio_phas']].to_csv(path_out,index=False)
    return ratios2

##################################################################################################################################
##########################  Main FUNCTION ########################################################################################
##################################################################################################################################
def main(path_data,path_out,degSMA=3,degAPEX=1,snr_cut=1.):
    print("********************************************************")
    print("******************GENERATE POLCAL***********************")
    print("********************************************************")

    get_polcal(path_data,path_out,degSMA=degSMA)
    return 0

if __name__=='__main__':

    if ("-h" in sys.argv) or ("--h" in sys.argv):
        print("generating polcal csv file")
        sys.exit()

    if "--datadir" in sys.argv:
        for a in range(0, len(sys.argv)):
            if(sys.argv[a] == '--datadir'):
                path_data = sys.argv[a+1]
    else:
        raise Exception("must provide data directory!")

    if "--outpath" in sys.argv:
        for a in range(0, len(sys.argv)):
            if(sys.argv[a] == '--outpath'):
                path_out = sys.argv[a+1]
    else:   path_out='polcal.csv'

    if "--degSMA" in sys.argv:
        for a in range(0, len(sys.argv)):
            if(sys.argv[a] == '--degSMA'):
                degSMA = int(sys.argv[a+1])
    else: degSMA = 3

    if "--degAPEX" in sys.argv:
        for a in range(0, len(sys.argv)):
            if(sys.argv[a] == '--degAPEX'):
                degAPEX = int(sys.argv[a+1])
    else: degAPEX = 1

    if "--snr_cut" in sys.argv:
        for a in range(0, len(sys.argv)):
            if(sys.argv[a] == '--snr_cut'):
                snr_cut = float(sys.argv[a+1])
    else: snr_cut = 1.

    main(path_data,path_out,degSMA=degSMA,degAPEX=degAPEX,snr_cut=snr_cut)
