import numpy as np
import pandas as pd
#import sys
#sys.path.append('/Users/mwielgus/Works/MyEAT/eat/')
from eat.inspect import closures as cl
from eat.inspect import utils as ut
import scipy.interpolate as si
import scipy.optimize as so
import matplotlib.pyplot as plt

lat_dict = {'A': -23.02922,
            'X': -23.00578,
            'Z': 32.70161,
            'L': 18.98577,
            'P': 37.06614,
            'J': 19.82284,
            'S': 19.82423,
            'R': 19.82423,
            'Y':  -90.00000,
            'G': 38.43306}

lon_dict = {'A': 67.75474,
            'X': 67.75914,
            'Z': 109.89124,
            'L': 97.31478,
            'P': 3.39260,
            'J': 155.47703,
            'S': 155.47755,
            'R': 155.47755,
            'Y':  -45.00000,
            'G':79.83972}#Greenbank

ant_locat ={
    'A': [2225061.16360, -5440057.36994, -2481681.15054],
    'X': [2225039.52970, -5441197.62920, -2479303.35970],
    'P': [5088967.74544, -301681.18586, 3825012.20561],
    'Y': [0.01000, 0.01000, -6359609.70000],
    'L': [-768715.63200, -5988507.07200, 2063354.85200],
    'Z': [-1828796.20000, -5054406.80000, 3427865.20000],
    'J': [-5464584.67600, -2493001.17000, 2150653.98200],
    'S': [-5464555.49300, -2492927.98900, 2150797.17600],
    'R': [-5464555.49300, -2492927.98900, 2150797.17600],
    'F': [-1324009.327,  -5332181.955,  3231962.395],
    'T': [-1640953.938,  -5014816.024,  3575411.792],
    'C': [-1449752.584,  -4975298.576,  3709123.846],
    'K': [-1995678.840,  -5037317.697,  3357328.025],
    'M': [-5464075.185,  -2495248.106,  2148297.365],
    'B': [-2112065.206,  -3705356.505,  4726813.676],
    'N': [-130872.499,  -4762317.093,  4226851.001],
    'O': [-2409150.402,  -4478573.118,  3838617.339],
    'G': [882589.983,  -4924873.024,  3943728.983]
}

ras_dict = {'1055+018': 10.974891,
            '1749+096': 17.859116,
            '1921-293': 19.414182999999998,
            '3C273': 12.485194,
            '3C279': 12.936435000000001,
            '3C454.3': 22.899373999999998,
            '3C84': 3.3300449999999997,
            'BLLAC': 22.045359000000001,
            'CENA': 13.424337,
            'CTA102': 22.543447,
            'CYGX-3': 20.540490999999999,
            'J0006-0623': 0.10385899999999999,
            'J0132-1654': 1.5454129999999999,
            'J1733-1304': 17.550751000000002,
            'NRAO530': 17.550751000000002,
            'J1924-2914': 19.414182999999998,
            'NGC1052': 2.684666,
            'OJ287': 8.9135759999999991,
            'SGRA': 17.761122472222,
            'M87': 12.5137287166667} #in hours

dec_dict = {'1055+018': 1.5663400000000001,
            '1749+096': 9.6502029999999994,
            '1921-293': -29.241701000000003,
            '3C273': 2.0523880000000001,
            '3C279': -5.7893120000000007,
            '3C454.3': 16.148211,
            '3C84': 41.511696000000001,
            'BLLAC': 42.277771000000001,
            'CENA': -43.019112,
            'CTA102': 11.730805999999999,
            'CYGX-3': 40.957751999999999,
            'J0006-0623': -6.3931490000000002,
            'J0132-1654': -16.913479000000002,
            'J1733-1304': -13.08043,
            'NRAO530':-13.08043,
            'J1924-2914': -29.241701000000003,
            'NGC1052': -8.2557639999999992,
            'OJ287': 20.108511,
            'SGRA': -29.0078105555556,
            'M87': 12.39112330555556,
            } #in deg



def compute_elev(ra_source, dec_source, xyz_antenna, time):
    #this one is by Michael Janssen
   """
   given right ascension and declination of a sky source [ICRS: ra->(deg,arcmin,arcsec) and dec->(hour,min,sec)]
   and given the position of the telescope from the vex file [Geocentric coordinates (m)]
   and the time of the observation (e.g. '2012-7-13 23:00:00') [UTC:yr-m-d],
   returns the elevation of the telescope.
   Note that every parameter can be an array (e.g. the time)
   """
   from astropy import units as u
   from astropy.coordinates import EarthLocation, AltAz, ICRS, Angle
   #angle conversions:
   ra_src_deg       = Angle(ra_source, unit=u.hour)
   ra_src_deg       = ra_src_deg.degree * u.deg
   dec_src_deg      = Angle(dec_source, unit=u.deg)

   source_position  = ICRS(ra=ra_src_deg, dec=dec_src_deg)
   antenna_position = EarthLocation(x=xyz_antenna[0]*u.m, y=xyz_antenna[1]*u.m, z=xyz_antenna[2]*u.m)
   altaz_system     = AltAz(location=antenna_position, obstime=time)
   trans_to_altaz   = source_position.transform_to(altaz_system)
   elevation        = trans_to_altaz.alt
   return elevation.degree


def paralactic_angle(alist):
    from astropy.time import Time
    get_GST_rad = lambda x: Time(x).sidereal_time('mean','greenwich').hour*2.*np.pi/24.
    GST = np.asarray(list(map(get_GST_rad,alist.datetime))) #in radians
    station1 = [x[0] for x in alist.baseline]
    station2 = [x[1] for x in alist.baseline]
    #print(set(station1))
    #print(set(station2))
    lon1 = np.asarray(list(map(lambda x: lon_dict[x],station1)))*np.pi*2./360. #converted from deg to radians
    lat1 = np.asarray(list(map(lambda x: lat_dict[x],station1)))*np.pi*2./360. #converted from deg to radians
    lon2 = np.asarray(list(map(lambda x: lon_dict[x],station2)))*np.pi*2./360. #converted from deg to radians
    lat2 = np.asarray(list(map(lambda x: lat_dict[x],station2)))*np.pi*2./360. #converted from deg to radians
    #ras = np.asarray(alist.ra_hrs)*np.pi*2./24. #converted from hours to radians
    #dec = np.asarray(alist.dec_deg)*np.pi*2./360. #converted from deg to radians    
    ras = np.asarray(list(map(lambda x: ras_dict[x], alist.source)))*np.pi*2./24. #converted from hours to radians
    dec = np.asarray(list(map(lambda x: dec_dict[x], alist.source)))*np.pi*2./360. #converted from deg to radians
    HA1 = GST - lon1 - ras #in rad
    HA2 = GST - lon2 - ras #in rad
    par1I = np.sin(HA1)
    par1R = np.cos(dec)*np.tan(lat1) - np.sin(dec)*np.cos(HA1)
    par1 = np.angle(par1R + 1j*par1I )
    par2I = np.sin(HA2)
    par2R = np.cos(dec)*np.tan(lat2) - np.sin(dec)*np.cos(HA2)
    par2 = np.angle(par2R + 1j*par2I )
    alist.loc[:,'par1'] = par1
    alist.loc[:,'par2'] = par2

    return alist

def field_rotation(fra_data):
    par = fra_data[0]
    elev = fra_data[1]
    station = fra_data[2]
    if station in {'A','J','Y'}:
        fra = par
    elif station in {'X','Z'}:
        fra = par+elev
    elif station in {'L','P'}:
        fra = par-elev
    elif station=='S':
        fra = 45. - elev + par
    else:
        fra = np.nan
    return fra

    
def add_computed_elev(alist):
    station1 = [x[0] for x in alist.baseline]
    station2 = [x[1] for x in alist.baseline]
    elev_data1 = zip(alist.source,station1,alist.datetime)
    elev_data2 = zip(alist.source,station2,alist.datetime)
    prep_elev = lambda x: compute_elev(ras_dict[x[0]],dec_dict[x[0]],ant_locat[x[1]],str(x[2]))
    ref_elev = list(map(prep_elev,elev_data1))
    rem_elev = list(map(prep_elev,elev_data2))
    alist_out = alist
    alist_out.loc[:,'ref_elev'] = ref_elev
    alist_out.loc[:,'rem_elev'] = rem_elev
    return alist_out
   

def add_field_rotation(alist):
    
    station1 = [x[0] for x in alist.baseline]
    station2 = [x[1] for x in alist.baseline]
    fra_data1 = zip(alist.par1*180./np.pi,alist.ref_elev,station1)
    fra_data2 = zip(alist.par2*180./np.pi,alist.rem_elev,station2)
    
    fra1 = list(map(field_rotation,fra_data1))
    fra2 = list(map(field_rotation,fra_data2))

    alist.loc[:,'fra1'] = fra1
    alist.loc[:,'fra2'] = fra2
    return alist

def total_field_rotation(alist):
    alist.loc[:,'tot_fra'] = -2.*(alist.fra1 - alist.fra2)
    return alist

def add_total_field_rotation(alist, recompute_elev = False):
    alist_out = alist
    alist = paralactic_angle(alist)
    if recompute_elev:
        alist = add_computed_elev(alist)
    alist = add_field_rotation(alist)
    alist = total_field_rotation(alist)
    alist_out.loc[:,'tot_fra'] = alist['tot_fra']
    return alist_out


def generate_ratios(alist,recompute_elev=True):
    alist = add_total_field_rotation(alist, recompute_elev = recompute_elev)
    #don't mess up the order of components, L,R here, R,L later
    fooL, fooR = cl.match_2_bsp_frames(alist,alist,match_what='polarization',dt = 0.5,what_is_same='baseline')
    #print(fooR.columns)
    fooR = fooR.sort_values(['datetime','baseline'])
    fooL = fooL.sort_values(['datetime','baseline'])
    fooR['RR2LL_amp'] = np.asarray(fooR.amp)/np.asarray(fooL.amp)
    fooL['RR2LL_amp'] = np.asarray(fooR.amp)/np.asarray(fooL.amp)
    fooR['RR2LL_phas'] = np.mod(np.asarray(fooR.resid_phas) - np.asarray(fooL.resid_phas) - (fooR.tot_fra),360)
    fooL['RR2LL_phas'] = np.mod(np.asarray(fooR.resid_phas) - np.asarray(fooL.resid_phas) - (fooR.tot_fra),360)
    fooR['RR2LL_sigma'] = np.asarray(fooR.amp)*np.asarray(fooL.sigma)+np.asarray(fooL.amp)*np.asarray(fooR.sigma)
    fooL['RR2LL_sigma'] = np.asarray(fooR.amp)*np.asarray(fooL.sigma)+np.asarray(fooL.amp)*np.asarray(fooR.sigma)
    return fooR, fooL   

def solve_amp_ratio(alist,no_sigmas=4,weightsA=True, weightsP=True):
    fooR, fooL = cl.match_2_bsp_frames(alist,alist,match_what='polarization',dt = 0.5,what_is_same='baseline')
    fooR = fooR.sort_values(['datetime','baseline'])
    fooL = fooL.sort_values(['datetime','baseline'])
    #print(np.shape(fooL))
    #print(np.shape(fooR))
    #solving for amplitudes of RR to LL ratios
    #prepare mean values of ratios amplitudes
    amp_ratio = []
    amp_weights = []
    list_baselines = sorted(list(set(fooR.baseline)))
    for base in list_baselines:
        fooRb = fooR[fooR.baseline==base]
        fooLb = fooL[fooL.baseline==base]
        vec = np.asarray(fooRb.amp)/np.asarray(fooLb.amp)
        vec = cut_outliers(vec,no_sigmas)
        amp_ratio.append(np.mean(vec))
        amp_weights.append(np.mean(fooRb.snr))
    #print(amp_weights)
    phase_weights = amp_weights
    if weightsA==False:
        amp_weights = np.ones(len(list_baselines))
    else:
        amp_weights = np.asarray(amp_weights)
    list_stations = sorted(list(set(''.join(list_baselines) )))
    amp_matrix = np.zeros((len(list_baselines),len(list_stations)))
    for couS in range(len(list_stations)):
        st = list_stations[couS]
        for couB in range(len(list_baselines)):
            bs = list_baselines[couB]
            if st in bs:
                amp_matrix[couB,couS] = 1.*amp_weights[couB]
    #print(amp_matrix)
    try:
        #check out scipy linalg least squares
        amp_ratio_st =  (np.linalg.lstsq(amp_matrix,np.transpose(amp_weights*np.log(np.asarray(amp_ratio))))[0])
        approx_res = np.exp(np.multiply(amp_matrix,amp_ratio_st))
        amp_ratio_st = np.exp(amp_ratio_st)
    except ValueError:
        amp_ratio_st = np.nan

    #solving for phases
    phas_diff = []
    if weightsP==False:
        phase_weights = np.ones(len(list_baselines))
    else:
        phase_weights = np.asarray(phase_weights)
    for base in list_baselines:
        fooRb = fooR[fooR.baseline==base]
        fooLb = fooL[fooL.baseline==base]
        vec = np.asarray(fooRb.resid_phas) - np.asarray(fooLb.resid_phas) - (-fooRb.tot_fra)
        #print(cl.circular_mean(vec))
        old_mean = cl.circular_mean(vec)
        #print(old_mean)
        vec = cut_outliers_circ(vec,no_sigmas)
        new_mean = cl.circular_mean(vec)
        if new_mean!=new_mean:
            new_mean = old_mean
        #print(new_mean)
        #print(vec)
    
        phas_diff.append(new_mean)
    
    #print('Now phas_diff at the begin:')
    #print(phas_diff)

    phas_matrix = np.zeros((len(list_baselines),len(list_stations)))
    for couS in range(len(list_stations)):
        st = list_stations[couS]
        for couB in range(len(list_baselines)):
            bs = list_baselines[couB]
            if bs[0]==st:
                phas_matrix[couB,couS] = 1.
            elif bs[1]==st:
                phas_matrix[couB,couS] = -1.

    func = lambda x: np.sum(phase_weights**2*np.abs(np.exp(1j*(2.*np.pi/360.)*(phas_matrix.dot(x) - phas_diff))-1.)**2)
    from scipy.optimize import minimize
    x0 =  180.*np.ones(len(list_stations))
    #x0[0] = 150.
    #meth = 'Nelder-Mead'
    #meth = 'Powell'
    #meth = 'CG'
    #meth = 'dogleg'
    #meth='trust-krylov'
    #meth='COBYLA'
    meth='L-BFGS-B'
    bounds = [(0.,360.)]*len(list_stations)
    #foo = minimize(func, x0,method=meth,bounds=bounds)
    try:
        foo = minimize(func, x0,method=meth)
        phas_ratio_st = foo.x
        ret_phas = np.mod(phas_ratio_st,360)
    except TypeError:
        ret_phas = np.nan
    #print(list_baselines)
    #print('Now phas_diff at the end:')
    #print(phas_diff)
    return amp_ratio_st, ret_phas, list_stations


def solve_only_amp_ratio(alist,weights=True):
    if 'RR2LL_amp' not in alist.columns:
        alistR,alistL = generate_ratios(alist)
    else: alistR = alist
    baselist=sorted(list(set(alistR.baseline)))
    amp_ratio = np.zeros(len(baselist))
    amp_weights = np.zeros(len(baselist))

    cou=0
    for base in baselist:
        foo = alistR[(alistR.baseline==base)].RR2LL_amp
        mean_loc = np.mean(foo)
        amp_ratio[cou] =mean_loc
        std_loc = np.std(foo)
        amp_weight_loc = mean_loc*np.sqrt(len(foo))/std_loc
        if np.isfinite(amp_weight_loc)==False:
            amp_weight_loc=0
        amp_weights[cou] = amp_weight_loc
        cou=cou+1

    list_baselines=baselist
    list_stations = sorted(list(set(''.join(list_baselines) )))
    amp_matrix = np.zeros((len(list_baselines),len(list_stations)))
    for couS in range(len(list_stations)):
        st = list_stations[couS]
        for couB in range(len(list_baselines)):
            bs = list_baselines[couB]
            if st in bs:
                if weights==True:
                    amp_matrix[couB,couS] = 1.*amp_weights[couB]
                else: amp_matrix[couB,couS] = 1.
    #print(amp_matrix)
    try:
        #check out scipy linalg least squares
        amp_ratio_st =  (np.linalg.lstsq(amp_matrix,np.transpose(amp_weights*np.log(np.asarray(amp_ratio))))[0])
        approx_res = np.exp(np.multiply(amp_matrix,amp_ratio_st))
        amp_ratio_st = np.exp(amp_ratio_st)
    except ValueError:
        print(amp_ratio)
        print(amp_weights)
        amp_ratio_st = np.nan

    #print(list_stations)
    #print(amp_ratio_st)
    return dict(zip(list_stations,amp_ratio_st))

def solve_only_phas_ratio(alist,degSMA=5):
    if 'RR2LL_phas' not in alist.columns:
        fil,alistL = generate_ratios(alist)
    else: fil = alist
    baselist=sorted(list(set(fil.baseline)))

    #find constant phases with respect to ALMA for all EXCEPT for SMA
    dicPh = {'A':0}
    for base in baselist:
        foo = np.mod(fil[(fil.baseline==base)].RR2LL_phas,360)
        mjd = fil[(fil.baseline==base)].mjd
        if (base[0]=='A')&(base[1]!='S'):
            dicPh[base[1]] = np.mod(-ut.circular_mean(foo),360)
    
    #now solve for SMA as 5-th order poly in time

    #gather all the available data for all baselines,
    mjdS=[]
    fooS=[]
    for base in baselist:
        if base[0]=='S':
            foo = list(np.mod(fil[(fil.baseline==base)].RR2LL_phas,360))
            fooS = fooS+[np.mod(x-dicPh[base[1]],360) for x in foo]
            mjd = list(fil[(fil.baseline==base)].mjd)
            mjdS = mjdS+mjd  
        elif base[1]=='S':
            foo = list(np.mod(fil[(fil.baseline==base)].RR2LL_phas,360))
            fooS = fooS+[np.mod(dicPh[base[0]]-x,360) for x in foo]
            #print(base[0])
            #print([np.mod(dicPh[base[0]]-x,360) for x in foo])
            mjd = list(fil[(fil.baseline==base)].mjd)
            mjdS = mjdS+mjd
        else: continue

    #sort, average data for the same time (neglect snr for now)
    mjdS = np.asarray(mjdS)
    fooS = np.asarray(fooS)
    #print(fooS)
    #plt.plot(mjdS,fooS,'*')
    #plt.show()
    fooS = np.unwrap(fooS*np.pi/180)*180/np.pi

    mjdso = np.asarray([x[0] for x in sorted(zip(mjdS,fooS))])
    fooso = np.asarray([x[1] for x in sorted(zip(mjdS,fooS))])
    aa = pd.DataFrame(columns=['mjd','phase'])
    aa['mjd'] = mjdso
    aa['phase'] = fooso
    meanfooso = np.asarray(list(aa.groupby('mjd').agg('mean').phase))
    mjdso =np.asarray(sorted(set(mjdso)))
    kwi = np.poly1d(np.polyfit(mjdso-57848,meanfooso,degSMA))
    SMAfit = lambda x: kwi(x-57848)
    dicPh['S'] = SMAfit
    #plt.plot(mjdso-57848,meanfooso,'*')
    #plt.show()
    #print(meanfooso)
    return dicPh


def solve_only_phas_ratio_fun(alist,degSMA=5):
    if 'RR2LL_phas' not in alist.columns:
        fil,alistL = generate_ratios(alist)
    else: fil = alist
    baselist=sorted(list(set(fil.baseline)))

    #find constant phases with respect to ALMA for all EXCEPT for SMA
    
    dicPh = {'A': lambda x: np.poly1d([0])(x)}
    valuesPh = {}
    for base in baselist:
        foo = np.mod((fil[(fil.baseline==base)].RR2LL_phas),360)
        mjd = np.asarray(fil[(fil.baseline==base)].mjd)
        if (base[0]=='A')&(base[1]!='S')&(base[1]!='X'):
            #dicPh[base[1]] = np.mod(-ut.circular_mean(foo),360)
            #print([base[1],ut.circular_mean(foo),np.mod(-ut.circular_mean(foo),360)])
            #zer_ord = np.poly1d(np.polyfit(mjd-57848,np.mod(-ut.circular_mean(foo),360)+ np.zeros(len(mjd)),0))
            #fit0 = lambda x: zer_ord(x-57848)
            #dicPh[base[1]] = fit0
            #fun0[base[1]]=np.poly1d(np.polyfit(mjd-57848,np.mod(-ut.circular_mean(foo),360)+ np.zeros(len(mjd)),0))
            #print(base[1])
            ph_stat = np.mod(-ut.circular_mean(foo),360)
            valuesPh[base[1]] = ph_stat
        if (base[0]=='A')&(base[1]!='S')&(base[1]=='X'):
            #dicPh[base[1]] = np.mod(-ut.circular_mean(foo),360)
            fir_ord = np.poly1d(np.polyfit(mjd-57848,np.zeros(len(mjd))-foo,1))
            fit1 = lambda x: fir_ord(x-57848)
            dicPh['X'] = fit1
    dicPh['Z'] = lambda x: np.poly1d([valuesPh['Z']])(x)
    dicPh['L'] = lambda x: np.poly1d([valuesPh['L']])(x)
    dicPh['P'] = lambda x: np.poly1d([valuesPh['P']])(x)
    dicPh['Y'] = lambda x: np.poly1d([valuesPh['Y']])(x)
    #print('L ', dicPh['L'](0))
    #print(valuesPh)
    #now solve for SMA as 5-th order poly in time
    #print(dicPh)
    #gather all the available data for all baselines,
    mjdS=[]
    fooS=[]
    for base in baselist:
        if base[0]=='S':
            fooLetter = base[1]
            other_ph_fun = dicPh[fooLetter]
            foo = list(np.mod(fil[(fil.baseline==base)].RR2LL_phas,360))
            #fooS = fooS+[np.mod(x-dicPh[base[1]],360) for x in foo]
            #fooS = fooS+[np.mod(x-dicPh[base[1]](x),360) for x in foo]
            fooS = fooS+[np.mod(x-other_ph_fun(x),360) for x in foo]
            mjd = list(fil[(fil.baseline==base)].mjd)
            mjdS = mjdS+mjd  
        elif base[1]=='S':
            fooLetter = base[0]
            other_ph_fun = dicPh[fooLetter]
            #print([fooLetter,other_ph_fun(0)])
            foo = list(np.mod(fil[(fil.baseline==base)].RR2LL_phas,360))
            #fooS = fooS+[np.mod(dicPh[base[0]]-x,360) for x in foo]
            mjd = list(fil[(fil.baseline==base)].mjd)
            fooS = fooS+[np.mod(other_ph_fun(mjd[cou])-foo[cou],360) for cou in range(len(foo))]
            
            #print(fooLetter)
            #print([np.mod(other_ph_fun(x)-x,360) for x in foo])
            
            mjdS = mjdS+mjd
            if fooLetter=='X':
                mjdX = mjd; fooX = foo
        else: continue

    #sort, average data for the same time (neglect snr for now)
    mjdS = np.asarray(mjdS)
    fooS = np.mod(np.asarray(fooS),360)
    #plt.plot(mjdS,fooS,'*')
    #plt.plot(mjdX,fooX,'r*')
    #plt.show()
    fooS = np.unwrap(fooS*np.pi/180)*180/np.pi
    mjdso = np.asarray([x[0] for x in sorted(zip(mjdS,fooS))])
    fooso = np.asarray([x[1] for x in sorted(zip(mjdS,fooS))])
    aa = pd.DataFrame(columns=['mjd','phase'])
    aa['mjd'] = mjdso
    aa['phase'] = fooso
    meanfooso = np.asarray(list(aa.groupby('mjd').agg(ut.circular_mean).phase))
    
    meanfoosoC = np.cos(meanfooso*np.pi/180)
    meanfoosoS = np.sin(meanfooso*np.pi/180)
    mjdso =np.asarray(sorted(set(mjdso)))
    try:
        if degSMA=='cubic':
            #kwi = si.interp1d(mjdso-57848, meanfooso, kind='cubic')
            kwiC = si.interp1d(mjdso-57848, meanfoosoC, kind='cubic')
            kwiS = si.interp1d(mjdso-57848, meanfoosoS, kind='cubic')
        elif degSMA=='linear':
            #kwi = si.interp1d(mjdso-57848, meanfooso, kind='linear')
            kwiC = si.interp1d(mjdso-57848, meanfoosoC, kind='linear')
            kwiS = si.interp1d(mjdso-57848, meanfoosoS, kind='linear')
        else:
            #kwi = np.poly1d(np.polyfit(mjdso-57848,meanfooso,degSMA))
            kwiC = np.poly1d(np.polyfit(mjdso-57848,meanfoosoC,degSMA))
            kwiS = np.poly1d(np.polyfit(mjdso-57848,meanfoosoS,degSMA))
        #SMAfit = lambda x: kwi(x-57848)
        SMAfit = lambda x: np.angle(kwiC(x-57848) + 1j*kwiS(x-57848))*180/np.pi
        dicPh['S'] = SMAfit
    except TypeError: #happens when no SMA in the array
        dicPh['S'] = lambda x: np.poly1d([0])(x)
    
    #plt.plot(mjdso-57848,meanfooso,'*')
    #print(meanfooso)
    #plt.show()

    return dicPh

def solve_ratios_scans(alist, recompute_elev=False,remove_phas_dof=False, zero_station=''):

    src_sc_list = list(set(zip(alist.source,alist.scan_id)))
    sc_list = [x for x in src_sc_list]
    scan_id_list = [x[1] for x in sc_list]
    list_stations_tot = list(set(''.join(list(set(alist.baseline)))))
    foo_columns = sorted(list(map(lambda x: x+'_amp',list_stations_tot))+list(map(lambda x: x+'_phas',list_stations_tot)))
    columns = ['datetime','source','scan_id']+foo_columns
    ratios = pd.DataFrame(columns=columns)
    for cou in range(len(sc_list)):
        local_scan_id = sc_list[cou][1] 
        local_source = sc_list[cou][0] 
        print(local_scan_id,' ',local_source)
        fooHloc = alist[alist.scan_id==local_scan_id]
        fooHloc = add_total_field_rotation(fooHloc, recompute_elev=recompute_elev)
        try:
            amp, pha, list_stations_loc = solve_amp_ratio(fooHloc,weightsA=True,weightsP=True)
            print(list_stations_loc)
            print(amp)
            print(pha)
            print('\n')
            Ratios_loc = pd.DataFrame(columns=columns)
            Ratios_loc['scan_id'] = [local_scan_id]
            Ratios_loc['source'] = [local_source]
            Ratios_loc['datetime'] = [min(fooHloc.datetime)]
            for cou_st in range(len(list_stations_loc)):
                stat = list_stations_loc[cou_st]
                try:
                    Ratios_loc[stat+'_amp'] = amp[cou_st]
                except TypeError:
                    Ratios_loc[stat+'_amp'] = np.nan
                try:
                    Ratios_loc[stat+'_phas'] = pha[cou_st]
                except TypeError:
                    Ratios_loc[stat+'_phas'] = np.nan
            ratios = pd.concat([ratios,Ratios_loc],ignore_index=True)
        except: #KeyError when scan is empty
            continue

    if remove_phase_dof==True:
        ratios = remove_phase_dof(ratios,zero_station)

    return ratios



def remove_phase_dof(ratios,zero_phase_station=''):
    '''
    Phase calculation with solve_amp_ratio keeps a degree of freedom i.e.
    if ALL phases are shifted by any phi_0, results remain the same.
    This function just assumes that phase at one chosen station is zero, subtracting
    estimated phase at this station from all stations present in this scan
    '''

    ratios_fixed = ratios.copy()
    #list of columns with phase on station
    phas_list = ratios.columns[list(map(lambda x:'phas' in x, ratios.columns))]
    
    #if zero phase station unspecified, choose the one with best coverage
    #that is not ALMA
    if zero_phase_station=='':
        phas_list_noA = phas_list.drop('A_phas')
        zero_phase_station = ratios_fixed[phas_list_noA].isnull().sum().argmin()[0]


    #when zero station is active, subtract its phase from all ratios
    phase_on_zero_station = np.asarray(ratios_fixed[zero_phase_station+'_phas'].copy())
    
    #table of all phases when zps in the array
    zps_present = ratios_fixed.loc[phase_on_zero_station==phase_on_zero_station,phas_list]
    
    #table of all phases with zps not present
    zps_not_present = ratios_fixed.loc[phase_on_zero_station!=phase_on_zero_station,phas_list]

    #put zero instead of nans so that when subtracted it doesn't change anything
    phase_on_zero_station[phase_on_zero_station!=phase_on_zero_station]=0
    #print(phase_on_zero_station[phase_on_zero_station!=phase_on_zero_station])
    #print(phase_on_zero_station)
    
    #subtract zps phase, this will make zps phase zero
    for station in phas_list:
        ratios_fixed.loc[:,station] = np.mod(np.asarray(ratios_fixed.loc[:,station]) - phase_on_zero_station,360)
    

    #when zero phase station isn't present, solve for value to subtract
    
    #phase_on_zero_station = np.asarray(ratios_fixed[zero_phase_station+'_phas'].copy())
    #print(ratios_fixed.loc[phase_on_zero_station!=phase_on_zero_station,phas_list])
    #print(ratios_fixed.loc[phase_on_zero_station==phase_on_zero_station,phas_list].mean())
    #print(zps_present)
    phas_list_noS = phas_list.drop('S_phas') #no S because phase on S isn't constant

    #mean phase at each station across all scans from when zps is present 
    zps_present_mean_phases = zps_present.apply(cl.circular_mean)

    #print('zps_present_mean= ', zps_present_mean_phases)

    #for scan without zps, mean value from when there is zps subtracted
    zps_differences = zps_not_present-zps_present_mean_phases
    #print(zps_differences)

    #error_m360 = lambda vect,x: np.sum(np.minimum(vect - x, 360 - (vect-x))**2)
    #from scipy.optimize import minimize

    #iterating over all scans
    for indexR, row in ratios_fixed.iterrows():

        #if this is the scan without zps
        if indexR in zps_differences.index:
            try:
                #print(indexR)
                #pick this row of phases, but without SMA
                row = zps_differences[phas_list_noS][zps_differences.index==indexR]
                rowLoc = np.asarray(row,dtype=np.float32)
                #print('dupa3=',rowLoc)
                #take mean phase of different stations present during this scan, no S
                rowLoc = rowLoc[rowLoc==rowLoc]
                delta_phase_scan = cl.circular_mean(rowLoc)
                #print(ratios_fixed.iloc[index])
                #print(index)
                #print('dupa1=',np.asarray(ratios_fixed.loc[(ratios_fixed.index==indexR),phas_list]))
                #print('dupa2=',delta_phase_scan)

                #subtract that mean across stations from all phases  available for this particular scan
                foo = np.asarray(ratios_fixed.loc[(ratios_fixed.index==indexR),phas_list]) - delta_phase_scan 
                #print(foo)
                ratios_fixed.loc[(ratios_fixed.index==indexR),phas_list] = foo
                #print(ratios_fixed.loc[(ratios_fixed.index==indexR),phas_list])
                #print(ratios_fixed[(ratios_fixed.index==indexR)][phas_list])
                #print(foo)
                #ratios_fixed[phas_list].replace([(ratios_fixed.index==indexR)], foo, inplace=True)
                #print(ratios_fixed[(ratios_fixed.index==indexR)][phas_list])
            except TypeError:
                continue
    return ratios_fixed

#def minimize_with_one_subtraction(vect):



def cut_outliers(vector,no_sigmas):
    #cuts outliers that are in distance from mean value larger than no_sigmas
    sigma = np.std(vector)
    m = np.mean(vector)
    vector = vector[np.abs(vector - m) < no_sigmas*sigma]
    return vector


def cut_outliers_circ(vector,no_sigmas):
    #cuts outliers that are in distance from mean value larger than no_sigmas
    sigma = cl.circular_std(vector)
    m = cl.circular_mean(vector)
    dist = np.minimum(np.abs(vector - m), np.abs(360. - np.abs(vector - m)))
    vector = vector[dist < no_sigmas*sigma]
    return vector

def get_mean_phases(foo,zero_stat='A',cut_outl=3.,to_return='all',expt=None,band=None):
    phas_list=['A_phas','L_phas','P_phas','S_phas','X_phas','Z_phas']
    station_list = [x[0] for x in phas_list]
    zero_ph = zero_stat+'_phas'
    fooR = foo.sort_values('datetime').copy()
    fooR = remove_phase_dof(fooR,zero_stat)

    #take only scans with zero phase baseline present
    fooR=fooR[fooR[zero_ph]==fooR[zero_ph]]

    circm = lambda x: ut.circ_cut_and_mean(x,cut_outl)
    circs = lambda x: ut.circ_cut_and_std(x,cut_outl)
    phases = fooR[phas_list].apply(circm)
    dev_ph = fooR[phas_list].apply(circs)
    len_cut = lambda x: len(ut.cut_outliers_circ(x[x==x],cut_outl) )
    how_many = fooR[phas_list].apply(len_cut)
    fooP = pd.DataFrame({'station': station_list,'mean_ph': list(phases),'std_ph': list(dev_ph),'num_scans':list(how_many)})
    fooP = fooP[['station','mean_ph','std_ph','num_scans']]
    fooP.loc[fooP.station==zero_stat,'mean_ph']=0
    fooP.loc[fooP.station==zero_stat,'std_ph']=0
    fooP['band'] = [band]*np.shape(fooP)[0]
    fooP['expt_no'] = [expt]*np.shape(fooP)[0]
    fooP['datetime_min'] = [fooR.datetime.min()]*np.shape(fooP)[0]
    fooP['datetime_max'] = [fooR.datetime.max()]*np.shape(fooP)[0]
    #print(fooP)
    if to_return=='all':
        return fooP
    else: return fooP[list(map(lambda x: x in to_return,fooP.station))]

def apply_mean_phases(RatiosRaw,mean_phases,which_stations=['Z','X','P','L']):
    for sta in which_stations:
        mean_phase = float(mean_phases[mean_phases.station==sta].mean_ph)
        phase_col = sta+'_phas'
        RatiosRaw[phase_col] = mean_phase
    return RatiosRaw

def fix_phases_without_zps(RatiosRaw,zps,which_st_use):
    RatiosRaw_with_zps = RatiosRaw[RatiosRaw[zps+'_phas']==RatiosRaw[zps+'_phas']]
    RatiosRaw_without_zps = RatiosRaw[RatiosRaw[zps+'_phas']!=RatiosRaw[zps+'_phas']]
    #foo = remove_phase_dof(RatiosRaw_without_zps,which_st_use)
    
    circm = lambda x: ut.circ_cut_and_mean(x,3)
    
    foo = np.asarray(RatiosRaw_without_zps[which_st_use+'_phas'])
    #print(RatiosRaw_without_zps)
    foo=foo[foo==foo]
    fooMean = circm(foo)

    #foo = RatiosRaw_without_zps[which_st_use+'_phas'].mean()
    #print(foo)
    ph_foo = get_mean_phases(RatiosRaw,zps,10.,which_st_use)
    phase_2 = np.mod(float(ph_foo[ph_foo.station==which_st_use].mean_ph),360)
    #print(phase_2)
    #print([phase_2])
    
    delta_phas = fooMean-phase_2
    #print(foo-phase_2)
    phas_list=['A_phas','L_phas','P_phas','S_phas','X_phas','Z_phas']
    RatiosRaw_without_zps[phas_list] = RatiosRaw_without_zps[phas_list].apply(lambda x: np.mod(x - delta_phas,360))
    RatiosRaw2 = pd.concat([RatiosRaw_with_zps,RatiosRaw_without_zps],ignore_index=True)
    return RatiosRaw2

def solve_for_SMA_phase_scan(RatiosRaw,zero_station=['A'],zero_st_vs_A=[0],smooth=0):
    #zero_st_vs_A - phase of the instantaneous zero station with respect to ALMA
    RatiosRaw = ut.add_mjd(RatiosRaw).sort_values('datetime')
    zero_stat = zero_station[0]
    
    RatiosSMA= RatiosRaw[list(map(lambda x: x==x,RatiosRaw.S_phas))].sort_values('datetime').copy()
    Ratios_for_interp = []

    RatiosRaw0 = remove_phase_dof(RatiosRaw,zero_stat)

    #there is SMA and ALMA in the array
    zero_ph = zero_station[0]+'_phas'
    cond1 = np.asarray(list(map(lambda x: x==x,RatiosRaw0.S_phas)))
    cond2 = np.asarray(list(map(lambda x: x==x,RatiosRaw0[zero_ph])))
    RatiosSMA_with_zero = RatiosRaw0[cond1*cond2]
    if len(zero_station)>1:
        list_with_first_mjd = list(RatiosSMA_with_zero.mjd)
        list_with_first_phase = list(RatiosSMA_with_zero.S_phas)
        #there is SMA, no ALMA, and second zero station in the array
        zero_stat = zero_station[1]
        zero_ph_2 = zero_station[1]+'_phas'
        RatiosRaw2 = remove_phase_dof(RatiosRaw,zero_stat)
        cond1b = np.asarray(list(map(lambda x: x==x,RatiosRaw2.S_phas)))
        cond2b = np.asarray(list(map(lambda x: x!=x,RatiosRaw2[zero_ph])))
        cond3 = np.asarray(list(map(lambda x: x==x,RatiosRaw2[zero_ph_2])))
        RatiosSMA_with_second_zero = RatiosRaw2[cond1*cond2b*cond3]

        list_with_second_mjd = list(RatiosSMA_with_second_zero.mjd)
        list_with_second_phase = list(RatiosSMA_with_second_zero.S_phas)
        #print(list_with_second_mjd)
        #print(list_with_second_phase)
        #to make phase related to alma, not to second zero station
        list_with_second_phase = [np.mod(x + zero_st_vs_A[1],360) for x in list_with_second_phase]
        #print(list_with_second_phase)
        list_interp_mjd = list_with_first_mjd+list_with_second_mjd
        list_interp_phase = list_with_first_phase+list_with_second_phase
        #print(list_interp_phase)
        #print(list_interp_mjd)
        #print(list_interp_mjd)
        #rint(list_interp_phase)

    else:
        list_interp_mjd = list(RatiosSMA_with_zero.mjd)
        list_interp_phase = list(RatiosSMA_with_zero.S_phas)

    #zip(list(RatiosSMA_with_zero.mjd),list(RatiosSMA_with_zero.S_phas))

    #PhaseSpline = si.UnivariateSpline(RatiosSMA_with_zero.mjd,RatiosSMA_with_zero.S_phas,s=smooth)
    PhaseSpline = si.UnivariateSpline(list_interp_mjd,list_interp_phase,s=smooth)
    PhaseSpline = si.interp1d(list_interp_mjd,list_interp_phase,fill_value='extrapolate')
    
    RatiosSMA['S_phas'] = PhaseSpline(RatiosSMA.mjd)
    RatiosSMA['S_phas_new'] = PhaseSpline(np.asarray(RatiosSMA.mjd))
    #print(PhaseSpline(57850.67))
    #get scan averaged phases from alist
    #and for all SMA baselines use other ratios
    return RatiosSMA


def add_gains_to_df(alist,recompute_elev=False,degSMA=5):
    '''
    Solves for gains and adds info to the dataframe.
    Should be apply to a single expt and single band data.
    '''
    if 'RR2LL_phas' not in alist.columns:
        fil,alistL = generate_ratios(alist,recompute_elev)
        one_pol_dat = True
    else: fil = alist

    dicPh = solve_only_phas_ratio_fun(fil,degSMA)
    dicAm = solve_only_amp_ratio(fil)
    dicPh['J'] = lambda x: np.nan
    dicAm['J'] = np.nan
    dicPh['R'] = lambda x: np.nan
    dicAm['R'] = np.nan

    mjd_bas = list(zip(alist.mjd,alist.baseline))
    '''
    mjd_bas = list(zip(fil.mjd,fil.baseline))
    #print(x[1][0],x[0])
    fil['gain_rat_amp_1'] = list(map(lambda x: dicAm[x[1][0]] ,mjd_bas))
    fil['gain_rat_amp_2'] = list(map(lambda x: dicAm[x[1][1]] ,mjd_bas))
    fil['gain_rat_phas_1'] = list(map(lambda x: dicPh[x[1][0]](x[0]) ,mjd_bas))
    fil['gain_rat_phas_2'] = list(map(lambda x: dicPh[x[1][1]](x[0]) ,mjd_bas))

    if one_pol_dat==True:
        alistL['gain_rat_amp_1'] = fil['gain_rat_amp_1']
        alistL['gain_rat_amp_2'] = fil['gain_rat_amp_2']
        alistL['gain_rat_phas_1'] = fil['gain_rat_phas_1']
        alistL['gain_rat_phas_2'] = fil['gain_rat_phas_2']
        fil_out = pd.concat([fil,alistL],ignore_index=True)
    '''
    alist['gain_rat_amp_1'] = list(map(lambda x: dicAm[x[1][0]] ,mjd_bas))
    alist['gain_rat_amp_2'] = list(map(lambda x: dicAm[x[1][1]] ,mjd_bas))
    alist['gain_rat_phas_1'] = list(map(lambda x: dicPh[x[1][0]](x[0]) ,mjd_bas))
    alist['gain_rat_phas_2'] = list(map(lambda x: dicPh[x[1][1]](x[0]) ,mjd_bas))


    #return fil_out
    return alist

def plot_consistency_gain_ratios(alist,degSMA=5):
    if 'RR2LL_phas' not in alist.columns:
        fil,alistL = generate_ratios(alist)
    else: fil = alist

    dicPh = solve_only_phas_ratio(fil,degSMA)
    dicAm = solve_only_amp_ratio(fil)


    for cou in range(len(baselist)):
        
        base = baselist[cou]
        if 'S' not in base:
            fig, ax = plt.subplots(1,2,sharex='row',figsize=(10,5))
            fooA = fil[(fil.baseline==base)].RR2LL_amp
            fooP = np.mod(fil[(fil.baseline==base)].RR2LL_phas,360)
            fooS = fil[(fil.baseline==base)].RR2LL_sigma
            mjd = np.asarray(fil[(fil.baseline==base)].mjd)
            ph_loc = np.mod(dicPh[base[0]]-dicPh[base[1]],360)
            amp_loc = dicAm[base[0]]*dicAm[base[1]]
            #ax[0].plot(mjd,fooP,'*')
            ax[0].errorbar(mjd,fooP,fooS/fooA*180/np.pi,fmt='bo',markersize=5,capsize=5)
            ax[0].plot(mjd,np.zeros(len(mjd))+ph_loc,'r--')
            ax[0].grid()
            ax[0].set_title(base+' phas')
            ax[0].set_ylabel('RR-LL phase difference [deg]')
            ax[0].set_xlabel('mjd time')
            #ax[0].set_ylim((0,360))

            #ax[1].plot(mjd,fooA,'*')
            ax[1].errorbar(mjd,fooA,fooS,fmt='bo',markersize=5,capsize=5)
            ax[1].plot(mjd,np.zeros(len(mjd))+amp_loc,'r--')
            ax[1].grid()
            ax[1].set_title(base+' amplitude')
            ax[1].set_ylabel('RR-LL amplitude ratio')
            ax[1].set_xlabel('mjd time')
            #ax[1].set_ylim((0.4,1.8))
            ax[0].grid()
            ax[1].grid()
            plt.show()

        else:
            fig, ax = plt.subplots(1,2,sharex='row',figsize=(10,5))
            fooA = np.mod(fil[(fil.baseline==base)].RR2LL_amp,360)
            fooP = np.mod(fil[(fil.baseline==base)].RR2LL_phas,360)
            fooS = fil[(fil.baseline==base)].RR2LL_sigma
            mjd = np.asarray(fil[(fil.baseline==base)].mjd)
            amp_loc = dicAm[base[0]]*dicAm[base[1]]
            #ax[0].plot(mjd,fooP,'*')
            ax[0].errorbar(mjd,fooP,np.asarray(fooS)/np.asarray(fooA)*180/np.pi,fmt='bo',markersize=5,capsize=5)
            ph_vec = dicPh['S'](mjd)
            if base[1]=='S':
                ax[0].plot(mjd,np.mod(dicPh[base[0]]-ph_vec,360),'r--')
            else:
                ax[0].plot(mjd,np.mod(ph_vec-dicPh[base[1]],360),'r--')
            ax[0].grid()
            ax[0].set_title(base+' phas')
            ax[0].set_ylabel('RR-LL phase difference [deg]')
            ax[0].set_xlabel('mjd time')
            #ax[0].set_ylim((0,360))

            #ax[1].plot(mjd,fooA,'*')
            ax[1].errorbar(mjd,fooA,fooS,fmt='bo',markersize=5,capsize=5)
            ax[1].plot(mjd,np.zeros(len(mjd))+amp_loc,'r--')
            ax[1].grid()
            ax[1].set_title(base+' amplitude')
            ax[1].set_ylabel('RR-LL amplitude ratio')
            ax[1].set_xlabel('mjd time')
            #ax[1].set_ylim((0.4,1.8))
            ax[0].grid()
            ax[1].grid()
            plt.show()

def plot_consistency_gain_ratios_fun(alist,degSMA=5):
    if 'RR2LL_phas' not in alist.columns:
        fil,alistL = generate_ratios(alist)
    else: fil = alist
    baselist=sorted(list(set(fil.baseline)))

    dicPh = solve_only_phas_ratio_fun(fil,degSMA)
    dicAm = solve_only_amp_ratio(fil)

    
    for cou in range(len(baselist)):
        
        base = baselist[cou]
        fig, ax = plt.subplots(1,2,sharex='row',figsize=(10,5))
        fooA = np.mod(fil[(fil.baseline==base)].RR2LL_amp,360)
        fooA_OJ = np.mod(fil[(fil.baseline==base)&(fil.source=='OJ287')].RR2LL_amp,360)
        fooA_3C = np.mod(fil[(fil.baseline==base)&(fil.source=='3C279')].RR2LL_amp,360)
        fooP = np.mod(fil[(fil.baseline==base)].RR2LL_phas,360)
        fooP_OJ = np.mod(fil[(fil.baseline==base)&(fil.source=='OJ287')].RR2LL_phas,360)
        fooP_3C = np.mod(fil[(fil.baseline==base)&(fil.source=='3C279')].RR2LL_phas,360)
        fooS = fil[(fil.baseline==base)].RR2LL_sigma
        fooS_OJ = fil[(fil.baseline==base)&(fil.source=='OJ287')].RR2LL_sigma
        fooS_3C = fil[(fil.baseline==base)&(fil.source=='3C279')].RR2LL_sigma
        mjd = np.asarray(fil[(fil.baseline==base)].mjd)
        mjd_OJ = np.asarray(fil[(fil.baseline==base)&(fil.source=='OJ287')].mjd)
        mjd_3C = np.asarray(fil[(fil.baseline==base)&(fil.source=='3C279')].mjd)
        

        

        amp_loc = dicAm[base[0]]*dicAm[base[1]]
        #ax[0].plot(mjd,fooP,'*')
        ax[0].errorbar(mjd,fooP,0*np.asarray(fooS)/np.asarray(fooA)*180/np.pi,fmt='bo',markersize=5,capsize=5,label='others')
        ax[0].errorbar(mjd_3C,fooP_3C,0*np.asarray(fooS_3C)/np.asarray(fooA_3C)*180/np.pi,fmt='ko',markersize=5,capsize=5,label='3C279')
        ax[0].errorbar(mjd_OJ,fooP_OJ,0*np.asarray(fooS_OJ)/np.asarray(fooA_OJ)*180/np.pi,fmt='go',markersize=5,capsize=5,label='OJ287')
        
        #ph_vec_0 = dicPh[base[0]](mjd)
        #ph_vec_1 = dicPh[base[1]](mjd)
        mjd_foo = np.linspace(np.min(mjd),np.max(mjd),256)
        ph_vec_0 = dicPh[base[0]](mjd_foo)
        ph_vec_1 = dicPh[base[1]](mjd_foo)
        #ax[0].plot(mjd,np.mod(ph_vec_0-ph_vec_1,360),'r--')
        ax[0].plot(mjd_foo,np.mod(ph_vec_0-ph_vec_1,360),'r--')
        ax[0].grid()
        ax[0].set_title(base+' phas')
        ax[0].set_ylabel('RR-LL phase difference [deg]')
        ax[0].set_xlabel('mjd time')
        #plt.legend()
        #ax[0].set_ylim((0,360))

        #ax[1].plot(mjd,fooA,'*')
        ax[1].errorbar(mjd,fooA,0*fooS,fmt='bo',markersize=5,capsize=5)
        ax[1].errorbar(mjd_3C,fooA_3C,0*fooS_3C,fmt='ko',markersize=5,capsize=5)
        ax[1].errorbar(mjd_OJ,fooA_OJ,0*fooS_OJ,fmt='go',markersize=5,capsize=5)
        ax[1].plot(mjd,np.zeros(len(mjd))+amp_loc,'r--')
        ax[1].grid()
        ax[1].set_title(base+' amplitude')
        ax[1].set_ylabel('RR-LL amplitude ratio')
        ax[1].set_xlabel('mjd time')
        #ax[1].set_ylim((0.4,1.8))
        ax[0].grid()
        ax[1].grid()
        plt.show()



'''
reload(polcal)
hopsDd = hopsD[hopsD.expt_no==3597]
hopsDd = hopsDd[map(lambda x: x[0]!=x[1], hopsDd.baseline)]
hopsDd = hopsDd[map(lambda x: x[0]==x[1], hopsDd.polarization)]
src_sc_list = list(set(zip(hopsDd.source,hopsDd.scan_id)))
sc_list = [x for x in src_sc_list if x[0]=='OJ287']
scan_id_list = [x[1] for x in sc_list]
list_stations_tot = list(set(''.join(list(set(hopsDd.baseline)))))
foo_columns = sorted(map(lambda x: x+'_amp',list_stations_tot)+map(lambda x: x+'_phas',list_stations_tot))
columns = ['datetime','scan_id']+foo_columns
Ratios = pd.DataFrame(columns=columns)
#Ratios['scan_id']= scan_id_list
for cou in range(len(sc_list)):
    local_scan_id = sc_list[cou][1] 
    print(local_scan_id)
    fooHloc = hopsDd[hopsDd.scan_id==local_scan_id]
    fooHloc = polcal.add_total_field_rotation(fooHloc)
    amp, pha, list_stations_loc = polcal.solve_amp_ratio(fooHloc)
    print(list_stations_loc)
    print(amp)
    print(pha)
    Ratios_loc = pd.DataFrame(columns=columns)
    Ratios_loc['scan_id'] = [local_scan_id]
    Ratios_loc['datetime'] = [min(fooHloc.datetime)]
    for cou_st in range(len(list_stations_loc)):
        stat = list_stations_loc[cou_st]
        Ratios_loc[stat+'_amp'] = amp[cou_st]
        Ratios_loc[stat+'_phas'] = pha[cou_st]
    Ratios = pd.concat([Ratios,Ratios_loc],ignore_index=True)

'''


def only_full_pol(alist,dt = 0):
    '''
    only keeps data where full polarizxation is available with given timestamp
    '''
    if dt > 0:
        alist = cl.add_round_time(alist, dt)
        alist_out = alist.groupby(('baseline','band','round_time')).filter(lambda x: len(x)==4)
    else:
        alist_out = alist.groupby(('baseline','band','datetime')).filter(lambda x: len(x)==4)
        #alist_gr = alist.groupby(('baseline','band','datetime'))
        #grouped_df = alist_gr
        #for key, item in alist_gr:
        #    print(alist_gr.get_group(key))

    return alist_out

def only_with_2pol(alist,dt = 0):
    '''
    only keeps data where full polarizxation is available with given timestamp
    '''
    if dt > 0:
        alist = cl.add_round_time(alist, dt)
        alist_out = alist.groupby(('baseline','band','round_time')).filter(lambda x: len(x)>=2)
    else:
        alist_out = alist.groupby(('baseline','band','datetime')).filter(lambda x: len(x)>=2)
        #alist_gr = alist.groupby(('baseline','band','datetime'))
        #grouped_df = alist_gr
        #for key, item in alist_gr:
        #    print(alist_gr.get_group(key))

    return alist_out



def solve_D_AX(alist,frac_pol):
    '''
    must be matched in time so for each time there are 4 polarizations
    '''

    RLamp = np.asarray(alist[alist.polarization=='RL'].sort_values('mjd').amp)
    LRamp = np.asarray(alist[alist.polarization=='LR'].sort_values('mjd').amp)
    RRamp = np.asarray(alist[alist.polarization=='RR'].sort_values('mjd').amp)
    LLamp = np.asarray(alist[alist.polarization=='LL'].sort_values('mjd').amp)

    RLphas = np.asarray(alist[alist.polarization=='RL'].sort_values('mjd').resid_phas)
    LRphas = np.asarray(alist[alist.polarization=='LR'].sort_values('mjd').resid_phas)
    RRphas = np.asarray(alist[alist.polarization=='RR'].sort_values('mjd').resid_phas)
    LLphas = np.asarray(alist[alist.polarization=='LL'].sort_values('mjd').resid_phas)
    one_mjd = np.asarray(alist[alist.polarization=='LL'].sort_values('mjd').mjd)

    # complex gain ratios
    gr1amp = np.asarray(alist[alist.polarization=='RR'].sort_values('mjd').gain_rat_amp_1)
    gr2amp = np.asarray(alist[alist.polarization=='RR'].sort_values('mjd').gain_rat_amp_2)
    gr1phas = np.asarray(alist[alist.polarization=='RR'].sort_values('mjd').gain_rat_phas_1)
    gr2phas = np.asarray(alist[alist.polarization=='RR'].sort_values('mjd').gain_rat_phas_2)

    gr1 = gr1amp*np.exp(1j*gr1phas*np.pi/180)
    gr2 = gr2amp*np.exp(1j*gr2phas*np.pi/180)

    #print(gr2)
    
    # phasors field rot
    tot_fra = np.asarray(alist[alist.polarization=='RR'].sort_values('mjd').tot_fra)
    fra1 = np.asarray(alist[alist.polarization=='RR'].sort_values('mjd').fra1)
    fra2 = np.asarray(alist[alist.polarization=='RR'].sort_values('mjd').fra2)
    phasor_tot_fra = np.exp(1j*tot_fra*np.pi/180) #tot_fra = -2*(phi1 - phi2)
    #neg_phasor_tot_fra = np.exp(-1j*tot_fra*np.pi/180) #neg_tot_fra = 2*(phi1 - phi2)
    phasor_fra1 = np.exp(1j*2*fra1*np.pi/180)
    phasor_fra2 = np.exp(1j*2*fra2*np.pi/180)


    #LHS of eq. 19 in calibration note
    RLbyRR = RLamp*np.exp(1j*RLphas*np.pi/180)/( RRamp*np.exp(1j*RRphas*np.pi/180) )
    LRbyRR = LRamp*np.exp(1j*LRphas*np.pi/180)/( RRamp*np.exp(1j*RRphas*np.pi/180) )
    RLbyLL = RLamp*np.exp(1j*RLphas*np.pi/180)/( LLamp*np.exp(1j*LLphas*np.pi/180) )
    LRbyLL = LRamp*np.exp(1j*LRphas*np.pi/180)/( LLamp*np.exp(1j*LLphas*np.pi/180) )

    #LHS of eq. 19 times gain terms minus frac_pol term
    RLbyRRg0 = RLbyRR/np.conjugate(1./gr2)
    LRbyRRg0 = LRbyRR/(1./gr1)
    RLbyLLg0 = RLbyLL/gr1
    LRbyLLg0 = LRbyLL/np.conjugate(gr2)

    RLbyRRg = RLbyRR/np.conjugate(1./gr2) - frac_pol*np.conjugate(phasor_fra2)
    LRbyRRg = LRbyRR/(1./gr1) - np.conjugate(frac_pol)*phasor_fra1
    RLbyLLg = RLbyLL/gr1 - frac_pol*np.conjugate(phasor_fra1)
    LRbyLLg = LRbyLL/np.conjugate(gr2) - np.conjugate(frac_pol)*phasor_fra2

    #plt.plot(np.real(RLbyRRg),np.imag(RLbyRRg),'*b')
    #plt.plot(np.imag(RLbyRRg),'*r')
    #plt.show()  
    cou = 0
    #vecY = np.array([LRbyRRg[cou],LRbyLLg[cou],RLbyRRg[cou],RLbyLLg[cou]])
    #sys_matrix = np.eye(4)
    #sys_matrix[0,1] = np.conjugate(phasor_tot_fra)

    #order of found D-terms:
    #D_1L, D_2R*, D_2L*, D_1R
    Nscans=len(RLbyRR)
    vecY = np.asarray(list(LRbyRRg)+list(LRbyLLg)+list(RLbyRRg)+list(RLbyLLg))
    sys_matrix = np.zeros((4*Nscans,4)) + 0*1j
    sys_matrix[0:Nscans,0] = 1
    sys_matrix[Nscans:2*Nscans,1]=1
    sys_matrix[2*Nscans:3*Nscans,2]=1
    sys_matrix[3*Nscans:4*Nscans,3]=1

    sys_matrix[0:Nscans,1]=np.conjugate(phasor_tot_fra)
    sys_matrix[Nscans:2*Nscans,0]=phasor_tot_fra
    sys_matrix[2*Nscans:3*Nscans,3]=np.conjugate(phasor_tot_fra)
    sys_matrix[3*Nscans:4*Nscans,2]=phasor_tot_fra
    
    Dterms =  np.linalg.lstsq(sys_matrix,np.transpose(vecY))[0]
    
    
    estimatedY = sys_matrix.dot(Dterms)

    print(np.shape(sys_matrix.dot(Dterms)))

    full_mjd = np.asarray(list(one_mjd)*4)
    
    #plt.plot(full_mjd,np.real(vecY),'r*')
    plt.plot(one_mjd,np.real(vecY[0:Nscans]),'b*')
    plt.plot(one_mjd,np.real(vecY[Nscans:2*Nscans]),'r*')
    plt.plot(one_mjd,np.real(estimatedY[0:Nscans]),'g*')
    plt.show()

    plt.plot(one_mjd,np.imag(vecY[Nscans:2*Nscans]),'r*')
    plt.plot(one_mjd,np.imag(vecY[0:Nscans]),'b*')
    plt.plot(one_mjd,np.imag(estimatedY[0:Nscans]),'g*')
    plt.show()


    plt.plot(full_mjd,np.real(vecY),'r*')
    plt.show()

    plt.plot(np.real(RLbyRRg),np.imag(RLbyRRg),'k*')
    plt.show()

    return sys_matrix, vecY, phasor_tot_fra,Dterms


'''
def solve_D_AX_frac_pol(alist,frac_pol):

    #must be matched in time so for each time there are 4 polarizations
    RLamp = np.asarray(alist[alist.polarization=='RL'].sort_values('mjd').amp)
    LRamp = np.asarray(alist[alist.polarization=='LR'].sort_values('mjd').amp)
    RRamp = np.asarray(alist[alist.polarization=='RR'].sort_values('mjd').amp)
    LLamp = np.asarray(alist[alist.polarization=='LL'].sort_values('mjd').amp)

    RLphas = np.asarray(alist[alist.polarization=='RL'].sort_values('mjd').resid_phas)
    LRphas = np.asarray(alist[alist.polarization=='LR'].sort_values('mjd').resid_phas)
    RRphas = np.asarray(alist[alist.polarization=='RR'].sort_values('mjd').resid_phas)
    LLphas = np.asarray(alist[alist.polarization=='LL'].sort_values('mjd').resid_phas)
    one_mjd = np.asarray(alist[alist.polarization=='LL'].sort_values('mjd').mjd)

    # complex gain ratios
    gr1amp = np.asarray(alist[alist.polarization=='RR'].sort_values('mjd').gain_rat_amp_1)
    gr2amp = np.asarray(alist[alist.polarization=='RR'].sort_values('mjd').gain_rat_amp_2)
    gr1phas = np.asarray(alist[alist.polarization=='RR'].sort_values('mjd').gain_rat_phas_1)
    gr2phas = np.asarray(alist[alist.polarization=='RR'].sort_values('mjd').gain_rat_phas_2)

    gr1 = gr1amp*np.exp(1j*gr1phas*np.pi/180)
    gr2 = gr2amp*np.exp(1j*gr2phas*np.pi/180)

    #print(gr2)
    
    # phasors field rot
    tot_fra = np.asarray(alist[alist.polarization=='RR'].sort_values('mjd').tot_fra)
    fra1 = np.asarray(alist[alist.polarization=='RR'].sort_values('mjd').fra1)
    fra2 = np.asarray(alist[alist.polarization=='RR'].sort_values('mjd').fra2)
    phasor_tot_fra = np.exp(1j*tot_fra*np.pi/180) #tot_fra = -2*(phi1 - phi2)
    #neg_phasor_tot_fra = np.exp(-1j*tot_fra*np.pi/180) #neg_tot_fra = 2*(phi1 - phi2)
    phasor_fra1 = np.exp(1j*2*fra1*np.pi/180)
    phasor_fra2 = np.exp(1j*2*fra2*np.pi/180)


    #LHS of eq. 19 in calibration note
    RLbyRR = RLamp*np.exp(1j*RLphas*np.pi/180)/( RRamp*np.exp(1j*RRphas*np.pi/180) )
    LRbyRR = LRamp*np.exp(1j*LRphas*np.pi/180)/( RRamp*np.exp(1j*RRphas*np.pi/180) )
    RLbyLL = RLamp*np.exp(1j*RLphas*np.pi/180)/( LLamp*np.exp(1j*LLphas*np.pi/180) )
    LRbyLL = LRamp*np.exp(1j*LRphas*np.pi/180)/( LLamp*np.exp(1j*LLphas*np.pi/180) )

    #LHS of eq. 19 times gain terms minus frac_pol term
    RLbyRRg0 = RLbyRR/np.conjugate(1./gr2)
    LRbyRRg0 = LRbyRR/(1./gr1)
    RLbyLLg0 = RLbyLL/gr1
    LRbyLLg0 = LRbyLL/np.conjugate(gr2)

    RLbyRRg = RLbyRR/np.conjugate(1./gr2) - frac_pol*np.conjugate(phasor_fra2)
    LRbyRRg = LRbyRR/(1./gr1) - np.conjugate(frac_pol)*phasor_fra1
    RLbyLLg = RLbyLL/gr1 - frac_pol*np.conjugate(phasor_fra1)
    LRbyLLg = LRbyLL/np.conjugate(gr2) - np.conjugate(frac_pol)*phasor_fra2

    #plt.plot(np.real(RLbyRRg),np.imag(RLbyRRg),'*b')
    #plt.plot(np.imag(RLbyRRg),'*r')
    #plt.show()  
    cou = 0
    #vecY = np.array([LRbyRRg[cou],LRbyLLg[cou],RLbyRRg[cou],RLbyLLg[cou]])
    #sys_matrix = np.eye(4)
    #sys_matrix[0,1] = np.conjugate(phasor_tot_fra)

    #order of found D-terms:
    #D_1L, D_2R*, D_2L*, D_1R
    Nscans=len(RLbyRR)
    
    vecY = np.asarray(list(LRbyRRg)+list(LRbyLLg)+list(RLbyRRg)+list(RLbyLLg))
    vecY_real = np.real(vecY)
    vecY_imag = np.imag(vecY)
    sys_matrix_real = np.zeros((4*Nscans,5))
    sys_matrix_imag = np.zeros((4*Nscans,5))
    sys_matrix_real[0:Nscans,0] = 1
    sys_matrix_imag[0:Nscans,0] = 1
    sys_matrix_real[Nscans:2*Nscans,1]=1
    sys_matrix_imag[Nscans:2*Nscans,1]=1
    sys_matrix_real[2*Nscans:3*Nscans,2]=1
    sys_matrix_imag[2*Nscans:3*Nscans,2]=1
    sys_matrix_real[3*Nscans:4*Nscans,3]=1
    sys_matrix_imag[3*Nscans:4*Nscans,3]=1

    sys_matrix[0:Nscans,1]=np.conjugate(phasor_tot_fra)
    sys_matrix[Nscans:2*Nscans,0]=phasor_tot_fra
    sys_matrix[2*Nscans:3*Nscans,3]=np.conjugate(phasor_tot_fra)
    sys_matrix[3*Nscans:4*Nscans,2]=phasor_tot_fra

    sys_matrix[0:Nscans,5]= np.conjugate(frac_pol)*phasor_fra1
    sys_matrix[Nscans:2*Nscans,5]=
    sys_matrix[2*Nscans:3*Nscans,5]=
    sys_matrix[3*Nscans:4*Nscans,5]=
    
    Dterms =  np.linalg.lstsq(sys_matrix,np.transpose(vecY))[0]
    
    
    estimatedY = sys_matrix.dot(Dterms)

    print(np.shape(sys_matrix.dot(Dterms)))

    full_mjd = np.asarray(list(one_mjd)*4)
    
    #plt.plot(full_mjd,np.real(vecY),'r*')
    plt.plot(one_mjd,np.real(vecY[0:Nscans]),'b*')
    plt.plot(one_mjd,np.real(vecY[Nscans:2*Nscans]),'r*')
    plt.plot(one_mjd,np.real(estimatedY[0:Nscans]),'g*')
    plt.show()

    plt.plot(one_mjd,np.imag(vecY[Nscans:2*Nscans]),'r*')
    plt.plot(one_mjd,np.imag(vecY[0:Nscans]),'b*')
    plt.plot(one_mjd,np.imag(estimatedY[0:Nscans]),'g*')
    plt.show()


    plt.plot(full_mjd,np.real(vecY),'r*')
    plt.show()

    plt.plot(np.real(RLbyRRg),np.imag(RLbyRRg),'k*')
    plt.show()

    return sys_matrix, vecY, phasor_tot_fra,Dterms
'''

'''
def solve_D_real_frac_pol(alist,frac_pol):

    #must be matched in time so for each time there are 4 polarizations
    RLamp = np.asarray(alist[alist.polarization=='RL'].sort_values('mjd').amp)
    LRamp = np.asarray(alist[alist.polarization=='LR'].sort_values('mjd').amp)
    RRamp = np.asarray(alist[alist.polarization=='RR'].sort_values('mjd').amp)
    LLamp = np.asarray(alist[alist.polarization=='LL'].sort_values('mjd').amp)

    RLphas = np.asarray(alist[alist.polarization=='RL'].sort_values('mjd').resid_phas)
    LRphas = np.asarray(alist[alist.polarization=='LR'].sort_values('mjd').resid_phas)
    RRphas = np.asarray(alist[alist.polarization=='RR'].sort_values('mjd').resid_phas)
    LLphas = np.asarray(alist[alist.polarization=='LL'].sort_values('mjd').resid_phas)
    one_mjd = np.asarray(alist[alist.polarization=='LL'].sort_values('mjd').mjd)

    # complex gain ratios
    gr1amp = np.asarray(alist[alist.polarization=='RR'].sort_values('mjd').gain_rat_amp_1)
    gr2amp = np.asarray(alist[alist.polarization=='RR'].sort_values('mjd').gain_rat_amp_2)
    gr1phas = np.asarray(alist[alist.polarization=='RR'].sort_values('mjd').gain_rat_phas_1)
    gr2phas = np.asarray(alist[alist.polarization=='RR'].sort_values('mjd').gain_rat_phas_2)

    gr1 = gr1amp*np.exp(1j*gr1phas*np.pi/180)
    gr2 = gr2amp*np.exp(1j*gr2phas*np.pi/180)

    #print(gr2)
    
    # phasors field rot
    tot_fra = np.asarray(alist[alist.polarization=='RR'].sort_values('mjd').tot_fra)
    fra1 = np.asarray(alist[alist.polarization=='RR'].sort_values('mjd').fra1)
    fra2 = np.asarray(alist[alist.polarization=='RR'].sort_values('mjd').fra2)
    phasor_tot_fra = np.exp(1j*tot_fra*np.pi/180) #tot_fra = -2*(phi1 - phi2)
    #neg_phasor_tot_fra = np.exp(-1j*tot_fra*np.pi/180) #neg_tot_fra = 2*(phi1 - phi2)
    phasor_fra1 = np.exp(1j*2*fra1*np.pi/180)
    phasor_fra2 = np.exp(1j*2*fra2*np.pi/180)


    #LHS of eq. 19 in calibration note
    RLbyRR = RLamp*np.exp(1j*RLphas*np.pi/180)/( RRamp*np.exp(1j*RRphas*np.pi/180) )
    LRbyRR = LRamp*np.exp(1j*LRphas*np.pi/180)/( RRamp*np.exp(1j*RRphas*np.pi/180) )
    RLbyLL = RLamp*np.exp(1j*RLphas*np.pi/180)/( LLamp*np.exp(1j*LLphas*np.pi/180) )
    LRbyLL = LRamp*np.exp(1j*LRphas*np.pi/180)/( LLamp*np.exp(1j*LLphas*np.pi/180) )

    #LHS of eq. 19 times gain terms minus frac_pol term
    RLbyRRg0 = RLbyRR/np.conjugate(1./gr2)
    LRbyRRg0 = LRbyRR/(1./gr1)
    RLbyLLg0 = RLbyLL/gr1
    LRbyLLg0 = LRbyLL/np.conjugate(gr2)

    RLbyRRg = RLbyRR/np.conjugate(1./gr2) - frac_pol*np.conjugate(phasor_fra2)
    LRbyRRg = LRbyRR/(1./gr1) - np.conjugate(frac_pol)*phasor_fra1
    RLbyLLg = RLbyLL/gr1 - frac_pol*np.conjugate(phasor_fra1)
    LRbyLLg = LRbyLL/np.conjugate(gr2) - np.conjugate(frac_pol)*phasor_fra2

    #plt.plot(np.real(RLbyRRg),np.imag(RLbyRRg),'*b')
    #plt.plot(np.imag(RLbyRRg),'*r')
    #plt.show()  
    cou = 0
    #vecY = np.array([LRbyRRg[cou],LRbyLLg[cou],RLbyRRg[cou],RLbyLLg[cou]])
    #sys_matrix = np.eye(4)
    #sys_matrix[0,1] = np.conjugate(phasor_tot_fra)

    #order of found D-terms:
    #D_1L, D_2R*, D_2L*, D_1R
    Nscans=len(RLbyRR)
    
    vecY = np.asarray(list(LRbyRRg)+list(LRbyLLg)+list(RLbyRRg)+list(RLbyLLg))
    vecY_real = np.real(vecY)
    vecY_imag = np.imag(vecY)
    sys_matrix_real = np.zeros((4*Nscans,5))
    sys_matrix_imag = np.zeros((4*Nscans,5))
    sys_matrix_real[0:Nscans,0] = 1
    sys_matrix_imag[0:Nscans,0] = 1
    sys_matrix_real[Nscans:2*Nscans,1]=1
    sys_matrix_imag[Nscans:2*Nscans,1]=1
    sys_matrix_real[2*Nscans:3*Nscans,2]=1
    sys_matrix_imag[2*Nscans:3*Nscans,2]=1
    sys_matrix_real[3*Nscans:4*Nscans,3]=1
    sys_matrix_imag[3*Nscans:4*Nscans,3]=1

    sys_matrix[0:Nscans,1]=np.conjugate(phasor_tot_fra)
    sys_matrix[Nscans:2*Nscans,0]=phasor_tot_fra
    sys_matrix[2*Nscans:3*Nscans,3]=np.conjugate(phasor_tot_fra)
    sys_matrix[3*Nscans:4*Nscans,2]=phasor_tot_fra

    sys_matrix[0:Nscans,5]= np.conjugate(frac_pol)*phasor_fra1
    sys_matrix[Nscans:2*Nscans,5]=
    sys_matrix[2*Nscans:3*Nscans,5]=
    sys_matrix[3*Nscans:4*Nscans,5]=
    
    Dterms =  np.linalg.lstsq(sys_matrix,np.transpose(vecY))[0]
    
    
    estimatedY = sys_matrix.dot(Dterms)

    print(np.shape(sys_matrix.dot(Dterms)))

    full_mjd = np.asarray(list(one_mjd)*4)
    
    #plt.plot(full_mjd,np.real(vecY),'r*')
    plt.plot(one_mjd,np.real(vecY[0:Nscans]),'b*')
    plt.plot(one_mjd,np.real(vecY[Nscans:2*Nscans]),'r*')
    plt.plot(one_mjd,np.real(estimatedY[0:Nscans]),'g*')
    plt.show()

    plt.plot(one_mjd,np.imag(vecY[Nscans:2*Nscans]),'r*')
    plt.plot(one_mjd,np.imag(vecY[0:Nscans]),'b*')
    plt.plot(one_mjd,np.imag(estimatedY[0:Nscans]),'g*')
    plt.show()


    plt.plot(full_mjd,np.real(vecY),'r*')
    plt.show()

    plt.plot(np.real(RLbyRRg),np.imag(RLbyRRg),'k*')
    plt.show()

    return sys_matrix, vecY, phasor_tot_fra,Dterms
'''


def solve_RLbyRR(alist,frac_pol):

    #must be matched in time so for each time there are 4 polarizations
    RLamp = np.asarray(alist[alist.polarization=='RL'].sort_values('mjd').amp)
    RRamp = np.asarray(alist[alist.polarization=='RR'].sort_values('mjd').amp)
    
    RLphas = np.asarray(alist[alist.polarization=='RL'].sort_values('mjd').resid_phas)
    RRphas = np.asarray(alist[alist.polarization=='RR'].sort_values('mjd').resid_phas)
    one_mjd = np.asarray(alist[alist.polarization=='LL'].sort_values('mjd').mjd)

    # complex gain ratios
    gr2amp = np.asarray(alist[alist.polarization=='RR'].sort_values('mjd').gain_rat_amp_2)
    gr2phas = np.asarray(alist[alist.polarization=='RR'].sort_values('mjd').gain_rat_phas_2)
    gr2 = gr2amp*np.exp(1j*gr2phas*np.pi/180)

    #print(gr2)
    
    # phasors field rot
    tot_fra = np.asarray(alist[alist.polarization=='RR'].sort_values('mjd').tot_fra)
    fra2 = np.asarray(alist[alist.polarization=='RR'].sort_values('mjd').fra2)
    phasor_tot_fra = np.exp(1j*tot_fra*np.pi/180) #tot_fra = -2*(phi1 - phi2)
    #neg_phasor_tot_fra = np.exp(-1j*tot_fra*np.pi/180) #neg_tot_fra = 2*(phi1 - phi2)
    phasor_fra2 = np.exp(1j*2*fra2*np.pi/180)

    #LHS of eq. 19 in calibration note
    RLbyRR = RLamp*np.exp(1j*RLphas*np.pi/180)/( RRamp*np.exp(1j*RRphas*np.pi/180) )
    
    #LHS of eq. 19 times gain terms minus frac_pol term
    RLbyRRg0 = RLbyRR/np.conjugate(1./gr2)
    RLbyRRg = RLbyRRg0 - frac_pol*np.conjugate(phasor_fra2)

    Nscans=len(RLbyRR)
    
    vecY = np.asarray(list(RLbyRRg))
    sys_matrix = np.zeros((Nscans,2)) +0*1j
    sys_matrix[0:Nscans,0]=np.conjugate(phasor_tot_fra)
    sys_matrix[0:Nscans,1]= 1
    #print(np.angle(vecY))
    Dterms =  np.linalg.lstsq(sys_matrix,(vecY))[0]
    #print(Dterms[0])
    estimatedY = sys_matrix.dot(Dterms)
    print(estimatedY)
    plt.figure(figsize=(10,10))
    plt.plot(np.real(RLbyRRg),np.imag(RLbyRRg),'*b')
    plt.plot(np.real(estimatedY),np.imag(estimatedY),'r*')
    foo = Dterms[1]*np.conjugate(phasor_tot_fra) + Dterms[0]
    plt.plot(np.real(foo),np.imag(foo),'g*')
    #print('foo=', foo)
    plt.axis('equal')
    #plt.axis([-0.4,0.4,-0.4,0.4])
    plt.axhline(0,color='k',linestyle='--')
    plt.axvline(0,color='k',linestyle='--')
    plt.grid(which='both')
    plt.show()

    rhs = np.conjugate(1./gr2)*(frac_pol*np.conjugate(phasor_fra2) + Dterms[0]*np.conjugate(phasor_tot_fra) +Dterms[1])
    plt.figure(figsize=(10,10))
    t = np.linspace(0,2*np.pi,128)
    
    plt.plot(np.real(RLbyRR),np.imag(RLbyRR),'*b')
    plt.plot(np.real(rhs),np.imag(rhs),'r*')

    plt.axis('equal')
    #plt.axis([-0.4,0.4,-0.4,0.4])
    plt.axhline(0,color='k',linestyle='--')
    plt.axvline(0,color='k',linestyle='--')
    plt.grid(which='both')
    plt.show()
    #print('r =', r2)
    #print('r_m =', np.sqrt(x2**2+y2**2))
    #print('x_m =', x2)
    #print('y_m =', y2)

    return Dterms



def solve_RLbyRR_real(alist,frac_pol):

    #must be matched in time so for each time there are 4 polarizations
    RLamp = np.asarray(alist[alist.polarization=='RL'].sort_values('mjd').amp)
    RRamp = np.asarray(alist[alist.polarization=='RR'].sort_values('mjd').amp)
    
    RLphas = np.asarray(alist[alist.polarization=='RL'].sort_values('mjd').resid_phas)
    RRphas = np.asarray(alist[alist.polarization=='RR'].sort_values('mjd').resid_phas)
    one_mjd = np.asarray(alist[alist.polarization=='LL'].sort_values('mjd').mjd)

    # complex gain ratios
    gr2amp = np.asarray(alist[alist.polarization=='RR'].sort_values('mjd').gain_rat_amp_2)
    gr2phas = np.asarray(alist[alist.polarization=='RR'].sort_values('mjd').gain_rat_phas_2)
    gr2 = gr2amp*np.exp(1j*gr2phas*np.pi/180)

    #print(gr2)
    
    # phasors field rot
    tot_fra = np.asarray(alist[alist.polarization=='RR'].sort_values('mjd').tot_fra)
    fra2 = np.asarray(alist[alist.polarization=='RR'].sort_values('mjd').fra2)
    phasor_tot_fra = np.exp(1j*tot_fra*np.pi/180) #tot_fra = -2*(phi1 - phi2)
    #neg_phasor_tot_fra = np.exp(-1j*tot_fra*np.pi/180) #neg_tot_fra = 2*(phi1 - phi2)
    phasor_fra2 = np.exp(1j*2*fra2*np.pi/180)

    #LHS of eq. 19 in calibration note
    RLbyRR = RLamp*np.exp(1j*RLphas*np.pi/180)/( RRamp*np.exp(1j*RRphas*np.pi/180) )
    
    #LHS of eq. 19 times gain terms minus frac_pol term
    RLbyRRg0 = RLbyRR/np.conjugate(1./gr2)
    RLbyRRg = RLbyRRg0 - frac_pol*np.conjugate(phasor_fra2)

    Nscans=len(RLbyRR)
    
    vecY = np.asarray(list(RLbyRRg))
    sys_matrix = np.zeros((Nscans,2)) +0*1j
    sys_matrix[0:Nscans,0]=np.conjugate(phasor_tot_fra)
    sys_matrix[0:Nscans,1]= 1
    #print(np.angle(vecY))
    Dterms =  np.linalg.lstsq(sys_matrix,(vecY))[0]
    #print(Dterms[0])
    estimatedY = sys_matrix.dot(Dterms)
    print(estimatedY)
    plt.figure(figsize=(10,10))
    plt.plot(np.real(RLbyRRg),np.imag(RLbyRRg),'*b')
    plt.plot(np.real(estimatedY),np.imag(estimatedY),'r*')
    foo = Dterms[1]*np.conjugate(phasor_tot_fra) + Dterms[0]
    plt.plot(np.real(foo),np.imag(foo),'g*')
    #print('foo=', foo)
    plt.axis('equal')
    #plt.axis([-0.4,0.4,-0.4,0.4])
    plt.axhline(0,color='k',linestyle='--')
    plt.axvline(0,color='k',linestyle='--')
    plt.grid(which='both')
    plt.show()

    rhs = np.conjugate(1./gr2)*(frac_pol*np.conjugate(phasor_fra2) + Dterms[0]*np.conjugate(phasor_tot_fra) +Dterms[1])
    plt.figure(figsize=(10,10))
    t = np.linspace(0,2*np.pi,128)
    
    plt.plot(np.real(RLbyRR),np.imag(RLbyRR),'*b')
    plt.plot(np.real(rhs),np.imag(rhs),'r*')

    plt.axis('equal')
    #plt.axis([-0.4,0.4,-0.4,0.4])
    plt.axhline(0,color='k',linestyle='--')
    plt.axvline(0,color='k',linestyle='--')
    plt.grid(which='both')
    plt.show()
    #print('r =', r2)
    #print('r_m =', np.sqrt(x2**2+y2**2))
    #print('x_m =', x2)
    #print('y_m =', y2)

    return Dterms


def prep_df_for_dterms(alist):

    RLamp = np.asarray(alist[alist.polarization=='RL'].sort_values('mjd').amp)
    LRamp = np.asarray(alist[alist.polarization=='LR'].sort_values('mjd').amp)
    RRamp = np.asarray(alist[alist.polarization=='RR'].sort_values('mjd').amp)
    LLamp = np.asarray(alist[alist.polarization=='LL'].sort_values('mjd').amp)

    RLphas = np.asarray(alist[alist.polarization=='RL'].sort_values('mjd').resid_phas)
    LRphas = np.asarray(alist[alist.polarization=='LR'].sort_values('mjd').resid_phas)
    RRphas = np.asarray(alist[alist.polarization=='RR'].sort_values('mjd').resid_phas)
    LLphas = np.asarray(alist[alist.polarization=='LL'].sort_values('mjd').resid_phas)
    one_mjd = np.asarray(alist[alist.polarization=='LL'].sort_values('mjd').mjd)
    '''
    # complex gain ratios
    gr1amp = np.asarray(alist[alist.polarization=='RR'].sort_values('mjd').gain_rat_amp_1)
    gr2amp = np.asarray(alist[alist.polarization=='RR'].sort_values('mjd').gain_rat_amp_2)
    gr1phas = np.asarray(alist[alist.polarization=='RR'].sort_values('mjd').gain_rat_phas_1)
    gr2phas = np.asarray(alist[alist.polarization=='RR'].sort_values('mjd').gain_rat_phas_2)

    #frac_pol = frac_pol#*np.exp(1j*np.pi/2)
    gr1 = gr1amp*np.exp(1j*gr1phas*np.pi/180)#*np.exp(-1j*np.pi/2)#*np.exp(-1j*np.pi/2)
    gr2 = gr2amp*np.exp(1j*gr2phas*np.pi/180)#*np.exp(-1j*np.pi/2)#*np.exp(-1j*np.pi/2)
    #gr2fix = (-1.039+0.039*1j)*np.ones(len(gr2))#*np.exp(-1j*np.pi/2)
    #print(gr2)
    
    # phasors field rot
    tot_fra = np.asarray(alist[alist.polarization=='RR'].sort_values('mjd').tot_fra)
    fra1 = np.asarray(alist[alist.polarization=='RR'].sort_values('mjd').fra1)
    fra2 = np.asarray(alist[alist.polarization=='RR'].sort_values('mjd').fra2)
    phasor_tot_fra = np.exp(-1j*tot_fra*np.pi/180) #tot_fra = -2*(phi1 - phi2)
    #neg_phasor_tot_fra = np.exp(-1j*tot_fra*np.pi/180) #neg_tot_fra = 2*(phi1 - phi2)
    phasor_fra1 = np.exp(1j*2*fra1*np.pi/180)
    phasor_fra2 = np.exp(1j*2*fra2*np.pi/180)
    '''
    
    alist['gr1'] = alist['gain_rat_amp_1']*np.exp(1j*alist['gain_rat_phas_1']*np.pi/180)
    alist['gr2'] = alist['gain_rat_amp_2']*np.exp(1j*alist['gain_rat_phas_2']*np.pi/180)
    alist['phasor_fra1'] =np.exp(1j*2*alist['fra1']*np.pi/180)
    alist['phasor_fra2'] =np.exp(1j*2*alist['fra2']*np.pi/180)
    alist['phasor_tot_fra'] =np.exp(1j*2*alist['tot_fra']*np.pi/180)


    #alist_out = alist.groupby('scan_id').agg({'datetime': 'min'}).reset_index()
    alist['scan_id'] = list(alist['scan_id'])
    alist_out = alist.groupby('scan_id').agg({'datetime': 'min','gr1':np.mean, 'gr2': np.mean, 'phasor_fra1': np.mean, 'phasor_fra2': np.mean,'phasor_tot_fra': np.mean}).reset_index()
    gr1 = np.asarray(alist_out.gr1)
    gr2 = np.asarray(alist_out.gr2)
    #LHS of eq. 19 in calibration note
    try:
        RLbyRR = RLamp*np.exp(1j*RLphas*np.pi/180)/( RRamp*np.exp(1j*RRphas*np.pi/180) )
        RLbyRRg0 = RLbyRR*np.conjugate(gr2)
        alist_out['RLbyRR'] = RLbyRR
        alist_out['RLbyRRg0'] = RLbyRRg0
    except ValueError:
        RLbyRR = np.nan
    try:
        LRbyRR = LRamp*np.exp(1j*LRphas*np.pi/180)/( RRamp*np.exp(1j*RRphas*np.pi/180) )
        LRbyRRg0 = np.conjugate(LRbyRR)*np.conjugate(gr1)
        alist_out['LRbyRR'] = LRbyRR
        alist_out['LRbyRRg0'] = LRbyRRg0
    except ValueError:
        LRbyRR = np.nan
    try:  
        RLbyLL = RLamp*np.exp(1j*RLphas*np.pi/180)/( LLamp*np.exp(1j*LLphas*np.pi/180) )
        RLbyLLg0 = RLbyLL/gr1
        alist_out['RLbyLL'] = RLbyLL
        alist_out['RLbyLLg0'] = RLbyLLg0
    except ValueError:
        RLbyLL = np.nan
    try:
        LRbyLL = LRamp*np.exp(1j*LRphas*np.pi/180)/( LLamp*np.exp(1j*LLphas*np.pi/180) )
        LRbyLLg0 = np.conjugate(LRbyLL)/gr2
        alist_out['LRbyLL'] = LRbyLL
        alist_out['LRbyLLg0'] = LRbyLLg0
    except ValueError:
        LRbyLL= np.nan

    '''
    #LHS of eq. 19 times gain terms minus frac_pol term
    RLbyRRg0 = RLbyRR/np.conjugate(1./gr2)
    LRbyRRg0 = LRbyRR/(1./gr1)
    RLbyLLg0 = RLbyLL/gr1
    LRbyLLg0 = LRbyLL/np.conjugate(gr2)

    # New version of the system
    #the order for solvinf is LR/LL, LR/RR, RL/LL, RL/RR
    
    alist_out['gr1'] = gr1
    alist_out['gr2'] = gr2

    alist_out['phasor_tot_fra'] = phasor_tot_fra
    alist_out['phasor_fra1'] = phasor_fra1
    alist_out['phasor_fra2'] = phasor_fra2
    '''
    return alist_out

def solve_LRbyLL_phase(dataLoc):
    
    data_solve =prep_df_for_dterms(dataLoc)
    V1 = np.asarray(data_solve.LRbyLL)
    G2 = np.asarray(data_solve.gr2)
    f1 = np.asarray(data_solve.phasor_fra1)
    f2 = np.asarray(data_solve.phasor_fra2)

    Yvec = np.asarray(np.conjugate(V1)/G2)
    Nscans = len(Yvec)
    mat_sys = np.zeros((Nscans,3)) + 0.*1j

    #mat_sys[:,0] = np.asarray(data_solve.phasor_fra2)
    #mat_sys[:,1] = np.conjugate(np.asarray(data_solve.phasor_tot_fra))
    #mat_sys[:,2] = np.ones((Nscans))

    mat_sys[:Nscans,0] = np.conjugate(f2)
    mat_sys[:Nscans,1] = f1*np.conjugate(f2)
    mat_sys[:Nscans,2] = np.ones((Nscans))
    #3 variables - (m* p*),(D_1L p*), (D_2R* p*)
 
    Dterms =  np.linalg.lstsq(mat_sys,Yvec)[0]
    ApproxVal = mat_sys.dot(Dterms)

    plt.plot(np.real(Yvec),np.imag(Yvec),'rd')
    plt.plot(np.real(ApproxVal),np.imag(ApproxVal),'ko')
    plt.axvline(0,color='k',linestyle='--')
    plt.axhline(0,color='k',linestyle='--')
    plt.axis('equal')
    plt.title('RL/LL')
    plt.ylabel('Im')
    plt.xlabel('Re')
    plt.grid(which='both')
    plt.show()
    return Dterms

def solve_D1L_D2R_phase(data_solve):
    
    f0 = list(data_solve.LRbyLLg0)
    f1 = list(data_solve.LRbyRRg0)
    Yvec = np.asarray(f0+f1)
    Nscans = len(f0)
    mat_sys = np.zeros((2*Nscans,3))

    mat_sys[:Nscans,0] = np.asarray(data_solve.phasor_fra2)
    mat_sys[:Nscans,1] = np.conjugate(np.asarray(data_solve.phasor_tot_fra))
    mat_sys[:Nscans,2] = np.ones((Nscans))
    mat_sys[Nscans:2*Nscans,0] = np.asarray(data_solve.phasor_fra1)
    mat_sys[Nscans:2*Nscans,1] = np.ones((Nscans))
    mat_sys[Nscans:2*Nscans,2] = np.asarray(data_solve.phasor_tot_fra)
    #3 variables - (m* p*),(D_1L p*), (D_2R* p*)
    Dterms =  np.linalg.lstsq(mat_sys,Yvec)[0]
    return Dterms
 
def solve_D1R_D2L_phase(data_solve):
    
    f0 = list(data_solve.RLbyLLg0)
    f1 = list(data_solve.RLbyRRg0)
    Yvec = np.asarray(f0+f1)
    Nscans = len(f0)
    mat_sys = np.zeros((2*Nscans,3))

    mat_sys[:Nscans,0] = np.conjugate(np.asarray(data_solve.phasor_fra1))
    mat_sys[:Nscans,1] = np.ones((Nscans))
    mat_sys[:Nscans,2] = np.conjugate(np.asarray(data_solve.phasor_tot_fra))
    mat_sys[Nscans:2*Nscans,0] = np.conjugate(np.asarray(data_solve.phasor_fra2))
    mat_sys[Nscans:2*Nscans,1] = np.asarray(data_solve.phasor_tot_fra)
    mat_sys[Nscans:2*Nscans,2] = np.ones((Nscans))
    
    Dterms =  np.linalg.lstsq(mat_sys,Yvec)[0]

    ApproxVal = mat_sys.dot(Dterms)

    return Dterms, ApproxVal

def solve_Dterms(dataLoc,ph0=0,use_m=False,m=0, return_raw = True, use_gains='both'):
    
    p = np.exp(1j*ph0)
    data_solve =prep_df_for_dterms(dataLoc)
    
    if use_gains=='both':

        V1 = np.asarray(data_solve.LRbyLL)
        V2 = np.asarray(data_solve.LRbyRR)
        V3 = np.asarray(data_solve.RLbyLL)
        V4 = np.asarray(data_solve.RLbyRR)
        G2 = np.asarray(data_solve.gr2)
        G1 = np.asarray(data_solve.gr1)
        f1 = np.asarray(data_solve.phasor_fra1)
        f2 = np.asarray(data_solve.phasor_fra2)
        Nscans = len(V1)

        y0 = list(np.conjugate(V1))
        y1 = list(np.conjugate(V2))
        y2 = list(V3)
        y3 = list(V4)
        Yvec = np.asarray(y0+y1+y2+y3)
        mat_sys = np.zeros((4*Nscans,5)) +0*1j

        #equations with LRbyLL
        mat_sys[:Nscans,0] = np.conjugate(f2)*G2
        mat_sys[:Nscans,1] = f1*np.conjugate(f2)*G2
        mat_sys[:Nscans,2] = np.ones((Nscans))*G2
        mat_sys[:Nscans,3] = np.zeros((Nscans))
        mat_sys[:Nscans,4] = np.zeros((Nscans))

        #equations with LRbyRR
        mat_sys[Nscans:2*Nscans,0] = np.conjugate(f1)/np.conjugate(G1)
        mat_sys[Nscans:2*Nscans,1] = np.ones((Nscans))/np.conjugate(G1)
        mat_sys[Nscans:2*Nscans,2] = np.conjugate(f1)*f2/np.conjugate(G1)
        mat_sys[Nscans:2*Nscans,3] = np.zeros((Nscans))
        mat_sys[Nscans:2*Nscans,4] = np.zeros((Nscans))

        #equations with RLbyLL
        mat_sys[2*Nscans:3*Nscans,0] = np.conjugate(f1)*G1
        mat_sys[2*Nscans:3*Nscans,1] = np.zeros((Nscans))
        mat_sys[2*Nscans:3*Nscans,2] = np.zeros((Nscans))
        mat_sys[2*Nscans:3*Nscans,3] = np.ones((Nscans))*G1
        mat_sys[2*Nscans:3*Nscans,4] = np.conjugate(f1)*f2*G1
        
        #equations with RLbyRR
        mat_sys[3*Nscans:4*Nscans,0] = np.conjugate(f2)/np.conjugate(G2)
        mat_sys[3*Nscans:4*Nscans,1] = np.zeros((Nscans))
        mat_sys[3*Nscans:4*Nscans,2] = np.zeros((Nscans))
        mat_sys[3*Nscans:4*Nscans,3] = f1*np.conjugate(f2)/np.conjugate(G2)
        mat_sys[3*Nscans:4*Nscans,4] = np.ones((Nscans))/np.conjugate(G2)

    elif use_gains==1:

        V2 = np.asarray(data_solve.LRbyRR)
        V3 = np.asarray(data_solve.RLbyLL)
        G1 = np.asarray(data_solve.gr1)
        f1 = np.asarray(data_solve.phasor_fra1)
        f2 = np.asarray(data_solve.phasor_fra2)
        Nscans = len(V2)
        y1 = list(np.conjugate(V2))
        y2 = list(V3)
        Yvec = np.asarray(y1+y2)
        mat_sys = np.zeros((2*Nscans,5)) +0*1j

        #equations with LRbyRR
        mat_sys[:Nscans,0] = np.conjugate(f1)/np.conjugate(G1)
        mat_sys[:Nscans,1] = np.ones((Nscans))/np.conjugate(G1)
        mat_sys[:Nscans,2] = np.conjugate(f1)*f2/np.conjugate(G1)
        mat_sys[:Nscans,3] = np.zeros((Nscans))
        mat_sys[:Nscans,4] = np.zeros((Nscans))

        #equations with RLbyLL
        mat_sys[Nscans:2*Nscans,0] = np.conjugate(f1)*G1
        mat_sys[Nscans:2*Nscans,1] = np.zeros((Nscans))
        mat_sys[Nscans:2*Nscans,2] = np.zeros((Nscans))
        mat_sys[Nscans:2*Nscans,3] = np.ones((Nscans))*G1
        mat_sys[Nscans:2*Nscans,4] = np.conjugate(f1)*f2*G1

    elif use_gains==2:

        try:
            V1 = np.asarray(data_solve.LRbyLL)
        except AttributeError: pass #print('No LR by LL data')
        try:
            V4 = np.asarray(data_solve.RLbyRR)
        except AttributeError: pass #print('No RL by RR data')
        #print(data_solve.columns)
        G2 = np.asarray(data_solve.gr2)
        f1 = np.asarray(data_solve.phasor_fra1)
        f2 = np.asarray(data_solve.phasor_fra2)
        Nscans = np.shape(data_solve)[0]
        y=[]
        if 'V1' in locals():#V1 is LR/LL
            y0 = list(np.conjugate(V1))
            y=y+y0
            print('y with V1 ',np.shape(y))
        if 'V4' in locals():#V1 is RL/RR
            y3 = list(V4)
            y=y+y3
            print('y with V4 ',np.shape(y))
        #Yvec = np.asarray(y0+y3)
        Yvec=y
        mat_sys = np.zeros((len(y),3)) +0*1j

        #equations with LRbyLL
        if 'V1' in locals():
            mat_sys[:Nscans,0] = np.conjugate(f2)*G2
            mat_sys[:Nscans,1] = f1*np.conjugate(f2)*G2
            mat_sys[:Nscans,2] = np.ones((Nscans))*G2
        
        #equations with RLbyRR
        if 'V4' in locals():#V1 is RL/RR
            mat_sys[:Nscans,0] = np.conjugate(f2)/np.conjugate(G2)
            mat_sys[:Nscans,1] = f1*np.conjugate(f2)/np.conjugate(G2)
            mat_sys[:Nscans,2] = np.ones((Nscans))/np.conjugate(G2)

    #solution with linear least squares
    print('sizes ',[np.shape(mat_sys),np.shape(Yvec)])
    if np.shape(mat_sys)[0]>3:
        Dterms =  np.linalg.lstsq(mat_sys,Yvec)[0]
    else: Dterms = [np.nan,np.nan,np.nan]
    ####################################
    #solution with least squares+loss
    '''
    from scipy.optimize import least_squares
    def fun_min(x): return mat_sys.dot(x) - Yvec
    def f_wrap(x): 
        x_cpl = [x[0]+1j*x[1], x[2]+1j*x[3],x[4]+1j*x[5],x[6]+1j*x[7], x[8]+1j*x[9]]
        x_cpl = np.asarray(x_cpl)
        y = fun_min(x_cpl)
        yr = np.real(y)
        yi = np.imag(y)
        y_out = np.asarray(list(yr)+list(yi))
        return y_out

    x0 = [0.1, 0.*1, 0., 0.*1, 0., 0.*1, 0., 0.*1, 0.,0.*1]
    res = least_squares(f_wrap,x0, loss='huber')
    x = res.x
    Dterms = np.asarray([x[0]+1j*x[1], x[2]+1j*x[3],x[4]+1j*x[5],x[6]+1j*x[7], x[8]+1j*x[9]])
    '''
    ####################################

    ApproxVal = mat_sys.dot(Dterms)
    Dterms_no_leakage = np.zeros(np.shape(Dterms))
    Dterms_no_leakage[0] = Dterms[0]
    ApproxVal_no_leakage = mat_sys.dot(Dterms_no_leakage)
    '''
    to_minimize = lambda x: np.sum(np.abs((mat_sys.dot(x) - Yvec))**2)
    Dterms2 = so.minimize(to_minimize,Dterms)
    print(Dterms2)
    for cou in range(4):
        plt.plot(np.real(Yvec[cou*Nscans:(cou+1)*Nscans]),np.imag(Yvec[cou*Nscans:(cou+1)*Nscans]),'rd')
        plt.plot(np.real(ApproxVal[cou*Nscans:(cou+1)*Nscans]),np.imag(ApproxVal[cou*Nscans:(cou+1)*Nscans]),'ko')
        plt.axvline(0,color='k',linestyle='--')
        plt.axhline(0,color='k',linestyle='--')
        plt.axis('equal')
        plt.title('RL/LL')
        plt.ylabel('Im')
        plt.xlabel('Re')
        plt.grid(which='both')
        plt.show()
    '''

    if return_raw==False:
        if use_m==False:
            D_out = [Dterms[0]/p, np.conjugate(Dterms[1]/p), Dterms[2]/p, Dterms[3]/p, np.conjugate(Dterms[4]/p)]
        else:
            p = Dterms[0]/m
            D_out=[p, np.conjugate(Dterms[1]/p), Dterms[2]/p, Dterms[3]/p, np.conjugate(Dterms[4]/p)]
        return D_out, ApproxVal, ApproxVal_no_leakage
    else:
        #print('Raw Dterms')
        #this returns [(m p), (D1L* p), (D2R p), (D1R p), (D2L* p)]
        return Dterms, ApproxVal, ApproxVal_no_leakage


def solve_single_ratio(dataLoc, which_ratio,ph0=0,use_m=False,m=0, return_raw = True):
    
    p = np.exp(1j*ph0)
    data_solve =prep_df_for_dterms(dataLoc)

    G2 = np.asarray(data_solve.gr2)
    G1 = np.asarray(data_solve.gr1)
    f1 = np.asarray(data_solve.phasor_fra1)
    f2 = np.asarray(data_solve.phasor_fra2)
    V = np.asarray(data_solve[which_ratio])
    Nscans = len(V)
    mat_sys = np.zeros((Nscans,3)) +0*1j
    if which_ratio=='LRbyLL':
        #equations with LRbyLL
        Yvec = np.asarray(np.conjugate(V))
        mat_sys[:Nscans,0] = np.conjugate(f2)*G2
        mat_sys[:Nscans,1] = f1*np.conjugate(f2)*G2
        mat_sys[:Nscans,2] = np.ones((Nscans))*G2
    if which_ratio=='LRbyRR':
        Yvec = np.asarray(np.conjugate(V))
        #equations with LRbyRR
        mat_sys[:Nscans,0] = np.conjugate(f1)/np.conjugate(G1)
        mat_sys[:Nscans,1] = np.ones((Nscans))/np.conjugate(G1)
        mat_sys[:Nscans,2] = np.conjugate(f1)*f2/np.conjugate(G1)
    if which_ratio=='RLbyLL':
        Yvec = np.asarray(V)
        #equations with RLbyLL
        mat_sys[:Nscans,0] = np.conjugate(f1)*G1
        mat_sys[:Nscans,1] = np.ones((Nscans))*G1
        mat_sys[:Nscans,2] = np.conjugate(f1)*f2*G1
    if which_ratio=='RLbyRR':
        Yvec = np.asarray(V)
        #equations with RLbyRR
        mat_sys[:Nscans,0] = np.conjugate(f2)/np.conjugate(G2)
        mat_sys[:Nscans,1] = f1*np.conjugate(f2)/np.conjugate(G2)
        mat_sys[:Nscans,2] = np.ones((Nscans))/np.conjugate(G2)

    Dterms =  np.linalg.lstsq(mat_sys,Yvec)[0]
    ApproxVal = mat_sys.dot(Dterms)
    
    if return_raw==False:
        if use_m==False:
            #[m, D1L, D2R, D1R, D2L]
            D_out = [Dterms[0]/p, np.conjugate(Dterms[1]/p), Dterms[2]/p, Dterms[3]/p, np.conjugate(Dterms[4]/p)]
        else:
            p = Dterms[0]/m
            print('returning with fracpol', [Dterms[0],m])
            D_out[0]=[p, np.conjugate(Dterms[1]/p), Dterms[2]/p, Dterms[3]/p, np.conjugate(Dterms[4]/p)]
        return D_out, ApproxVal
    else:
        #print('Raw Dterms')
        #this returns [(m p), (D1L* p), (D2R p), (D1R p), (D2L* p)]
        return Dterms, ApproxVal

def inspect_dterms_quality(dataLoc,use_m=False,m=0,use_gains='both',return_raw = True):
    data_solve =prep_df_for_dterms(dataLoc)
    DT,App, App0 = solve_Dterms(dataLoc,return_raw=True,use_gains=use_gains)
    
    '''
    DT=np.zeros(5)
    DTs = solve_LRbyLL_phase(dataLoc)
    DT[0:3]=DTs
    '''
    V1 = np.asarray(data_solve.LRbyLL)
    V2 = np.asarray(data_solve.LRbyRR)
    V3 = np.asarray(data_solve.RLbyLL)
    V4 = np.asarray(data_solve.RLbyRR)
    
    lhs1 = np.conjugate(V1)
    lhs2 = np.conjugate(V2)
    lhs3 = V3
    lhs4 = V4
    Nscans = len(V1)

    if use_gains=='both':
        rhs1 = App[:Nscans]
        rhs2 = App[Nscans:2*Nscans]
        rhs3 = App[2*Nscans:3*Nscans]
        rhs4 = App[3*Nscans:4*Nscans]
        rhs1_0 = App0[:Nscans]
        rhs2_0 = App0[Nscans:2*Nscans]
        rhs3_0 = App0[2*Nscans:3*Nscans]
        rhs4_0 = App0[3*Nscans:4*Nscans]
    elif use_gains==1:
        rhs2 = App[:Nscans]
        rhs3 = App[Nscans:2*Nscans]
        rhs1 = np.zeros(Nscans)
        rhs4 = np.zeros(Nscans)
        rhs2_0 = App0[:Nscans]
        rhs3_0 = App0[Nscans:2*Nscans]
        rhs1_0 = np.zeros(Nscans)
        rhs4_0 = np.zeros(Nscans)
    elif use_gains==2:
        rhs1 = App[:Nscans]
        rhs4 = App[Nscans:2*Nscans]
        rhs2 = np.zeros(Nscans)
        rhs3 = np.zeros(Nscans)
        rhs1_0 = App0[:Nscans]
        rhs4_0 = App0[Nscans:2*Nscans]
        rhs2_0 = np.zeros(Nscans)
        rhs3_0 = np.zeros(Nscans)

    fig, ax = plt.subplots(2,2,figsize=(10,10))

    ax[0,0].axvline(0,color='k',linestyle='--')
    ax[0,0].axhline(0,color='k',linestyle='--')
    ax[0,0].plot(np.real(lhs1),np.imag(lhs1),'kd',label='data',markersize=12)
    ax[0,0].plot(np.real(rhs1),np.imag(rhs1),'ro',label='fit')
    ax[0,0].plot(np.real(rhs1_0),np.imag(rhs1_0),'o',label='fit 0 leakage',markerfacecolor='None',markeredgecolor='b',markeredgewidth=1)
    ax[0,0].grid()
    ax[0,0].legend(frameon=True,framealpha = 1.,edgecolor='k')
    ax[0,0].axis('equal')
    ax[0,0].set_title('(LR/LL)*')
    ax[0,0].set_ylabel('Im')
    ax[0,0].set_xlabel('Re')

    ax[0,1].axvline(0,color='k',linestyle='--')
    ax[0,1].axhline(0,color='k',linestyle='--')
    ax[0,1].plot(np.real(lhs2),np.imag(lhs2),'kd',label='data',markersize=12)
    ax[0,1].plot(np.real(rhs2),np.imag(rhs2),'ro',label='fit')
    ax[0,1].plot(np.real(rhs2_0),np.imag(rhs2_0),'o',label='fit 0 leakage',markerfacecolor='None',markeredgecolor='b',markeredgewidth=1)
    ax[0,1].grid()
    ax[0,1].axis('equal')
    ax[0,1].set_title('(LR/RR)*')
    ax[0,1].set_ylabel('Im')
    ax[0,1].legend(frameon=True,framealpha = 1.,edgecolor='k')
    ax[0,1].set_xlabel('Re')

    ax[1,0].axvline(0,color='k',linestyle='--')
    ax[1,0].axhline(0,color='k',linestyle='--')
    ax[1,0].plot(np.real(lhs3),np.imag(lhs3),'kd',label='data',markersize=12)
    ax[1,0].plot(np.real(rhs3),np.imag(rhs3),'ro',label='fit')
    ax[1,0].plot(np.real(rhs3_0),np.imag(rhs3_0),'o',label='fit 0 leakage',markerfacecolor='None',markeredgecolor='b',markeredgewidth=1)
    ax[1,0].grid()
    ax[1,0].axis('equal')
    ax[1,0].set_title('RL/LL')
    ax[1,0].set_ylabel('Im')
    ax[1,0].legend(frameon=True,framealpha = 1.,edgecolor='k')
    ax[1,0].set_xlabel('Re')


    ax[1,1].axvline(0,color='k',linestyle='--')
    ax[1,1].axhline(0,color='k',linestyle='--')
    ax[1,1].plot(np.real(lhs4),np.imag(lhs4),'kd',label='data',markersize=12)
    ax[1,1].plot(np.real(rhs4),np.imag(rhs4),'ro',label='fit')
    ax[1,1].plot(np.real(rhs4_0),np.imag(rhs4_0),'o',label='fit 0 leakage',markerfacecolor='None',markeredgecolor='b',markeredgewidth=1)
    ax[1,1].grid()
    ax[1,1].axis('equal')
    ax[1,1].set_title('RL/RR')
    ax[1,1].set_ylabel('Im')
    ax[1,1].set_xlabel('Re')
    ax[1,1].legend(frameon=True,framealpha = 1.,edgecolor='k')
    #leg = legend.get_frame()
    #leg.set_facecolor('white')
    ax[0,0].grid()
    ax[0,1].grid()
    ax[1,0].grid()
    ax[1,1].grid()
    plt.show()
