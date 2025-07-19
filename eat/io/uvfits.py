'''
Imports uvfits using ehtim library
and remodels them into pandas dataframes
Maciek Wielgus 06/07/2018
maciek.wielgus@gmail.com
Iniyan Natarajan 17/07/2025: Add save_uvfits method to save Uvfits_data object to UVFITS format.
'''
from __future__ import print_function
from __future__ import division
import pandas as pd
import os,sys,glob
import numpy as np
from astropy.time import Time, TimeDelta
from astropy.io import fits
import datetime as datetime
from eat.io import vex
from eat.io.datastructures import Uvfits_data, Obs_info, Antenna_info, Datastruct
import logging

# Configure logging
loglevel = getattr(logging, 'INFO', None)
logging.basicConfig(level=loglevel,
                    format='%(asctime)s %(levelname)s:: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

MJD_0 = 2400000.5
RDATE_DEGPERDY = 360.98564497330 # TODO: def. from AIPS; get the actual value?
ROUND_SCAN_INT = 20 # decimal precision for the scan start & stop times (fractional day)
DTARR = [('site', 'a32'), ('x','f8'), ('y','f8'), ('z','f8')]
DTPOL = [('time','f8'),('freq','f8'),('tint','f8'),
            ('t1','a32'),('t2','a32'),
            ('u','f8'),('v','f8'),
            ('rr','c16'),('ll','c16'),('rl','c16'),('lr','c16'),
            ('rrweight','f8'),('llweight','f8'),('rlweight','f8'),('lrweight','f8')]

def get_info(observation='EHT2017',path_vex='vex/'):
    '''
    gets info about stations, scans, expt for a given observation,
    by default it's EHT2017 campaign
    '''
    if observation=='EHT2017':

        stations_2lett_1lett = {'AZ': 'Z', 'PV': 'P', 'SM':'S', 'SR':'R','JC':'J', 'AA':'A','AP':'X', 'LM':'L','SP':'Y'}
        jd_expt = jd2expt2017
        if path_vex!='':
            scans = make_scan_list_EHT2017(path_vex)  
        else: scans = {}

    if observation=='EHT2018':

        stations_2lett_1lett = {'MG': 'Z', 'GL': 'G', 'PV': 'P', 'SW':'S', 'SR':'R','MM':'J', 'AA':'A','AX':'X', 'LM':'L','SZ':'Y'}
        jd_expt = jd2expt2018
        if path_vex!='':
            scans = make_scan_list_EHT2018(path_vex)
        else: scans = {}

    if observation=='EHT2021':
        stations_2lett_1lett = {'MG':'Z', 'GL':'G', 'PV':'P', 'SW':'S', 'SR':'R', 'MM':'J', 'AA':'A', 'AX':'X', 'LM':'L','SZ':'Y','NN':'N','KT':'K'}
        jd_expt = jd2expt2021
        if path_vex!='':
            scans = make_scan_list_EHT2021(path_vex)
        else: scans = {}
        
    if observation=='EHT2022':
        stations_2lett_1lett = {'NN': 'N', 'AX': 'X', 'PV': 'P', 'GL': 'G', 'AA': 'A', 'LM': 'L', 'KT': 'K', 'MG': 'Z', 'MM': 'J', 'SW': 'S', 'SZ': 'Y'}
        jd_expt = jd2expt2022
        if path_vex!='':
            scans = make_scan_list_EHT2022(path_vex)
        else: scans = {}
        

    if observation=='EHT2017_Dan_Mel':

        stations_2lett_1lett = {'ALMA': 'A', 'APEX': 'X', 'JCMT': 'J', 'LMT':'L', 'SMR':'R','SMAP':'S', 'SMA':'S', 'SMT':'Z', 'SMTO':'Z', 'PV':'P', 'PICOVEL':'P','SPT':'Y'}
        jd_expt = jd2expt2017
        if path_vex!='':
            scans = make_scan_list_EHT2017(path_vex)  
        else: scans = {}

    if observation=='GMVA':

        stations_2lett_1lett = {'ALMA': 'A', 'AA':'A', 'GBT': 'G', 'GB': 'G', 'FD': 'F', 'LMT':'L', 'PT':'T','LA':'C', 'KP':'K', 'MK':'M', 'BR':'B', 'PV':'P', 'NL':'N','OV':'O','YS':'Y','EB':'E'}
        jd_expt = jd2expt2017
        if path_vex!='':
            scans = make_scan_list_EHT2017(path_vex)  
        else: scans = {}
    return stations_2lett_1lett, jd_expt, scans

def jd2expt2017(jd):
    '''
    Function translating from jd to expt for April 2017 EHT
    '''
    if (jd > 2457853.470)&(jd < 2457854.132 ):
        return 3600
    elif (jd > 2457849.531)&(jd < 2457850.177):
        return 3598
    elif (jd > 2457850.667)&(jd < 2457851.363):
        return 3599
    elif (jd > 2457848.438)&(jd < 2457849.214):
        return 3597
    elif (jd > 2457854.427)&(jd < 2457855.141):
        return 3601
    else:
        return None

def jd2expt2018(jd):
    '''
    Function translating from jd to expt for April 2018 EHT
    '''
    if (jd > 2458229.416)&(jd < 2458230.125):
        return 3644
    elif (jd > 2458230.416)&(jd < 2458231.125):
        return 3645
    elif (jd > 2458232.625)&(jd < 2458233.167):
        return 3646
    elif (jd > 2458233.416)&(jd < 2458234.125):
        return 3647
    elif (jd > 2458235.291)&(jd < 2458235.917):
        return 3648
    elif (jd > 2458236.375)&(jd < 2458236.917):
        return 3649
    else:
        return None


def jd2expt2021(jd):
    '''
    Function translating from jd to expt for April 2021 EHT
    '''
    if (jd > 2459313.458)&(jd < 2459314.166):
        return 3762
    elif (jd > 2459317.291)&(jd < 2459318.042):
        return 3763
    elif (jd > 2459318.458)&(jd < 2459319.208):
        return 3764
    elif (jd > 2459319.458)&(jd < 2459320.167):
        return 3765
    elif (jd > 2459320.458)&(jd < 2459321.167):
        return 3766
    elif (jd > 2459321.458)&(jd < 2459322.167):
        return 3767
    elif (jd > 2459322.291)&(jd < 2459323.0):
        return 3768
    elif (jd > 2459323.541)&(jd < 2459323.792):
        return 3769
    else:
        return None

def jd2expt2022(jd):
    '''
    Function translating from jd to expt for April 2022 EHT
    '''
    if (jd > 2459656.436)&(jd < 2459657.028):
        return 3803
    elif (jd > 2459657.635)&(jd < 2459658.301):
        return 3804
    elif (jd > 2459658.733)&(jd < 2459659.398):
        return 3805
    elif (jd > 2459660.395)&(jd < 2459661.164):
        return 3806
    elif (jd > 2459661.609)&(jd < 2459661.609):
        return 3807
    elif (jd > 2459664.399)&(jd < 2459664.985):
        return 3808
    elif (jd > 2459665.390)&(jd < 2459666.012):
        return 3809
    else:
        return None

def make_scan_list_EHT2017(fpath):
    '''
    generates data frame with information about scans for EHT2017
    '''
    nam2lett = {'ALMA':'A','APEX':'X','LMT':'L','PICOVEL':'P','SMTO':'Z','SPT':'T','JCMT':'J','SMAP':'S'}
    track2expt ={'D':3597,'B':3598, 'C':3599,'A':3600,'E':3601}
    list_files = [x.split('/')[-1] for x in glob.glob(fpath+'/*.vex')]
    #list_files = os.listdir(fpath)
    scans = pd.DataFrame({'source' : []})

    for fi in list_files:#loop over vex files in folder
        track_loc = fi[3].upper()
        vpath = os.path.join(fpath, fi)
        aa = vex.Vex(vpath)
        dec = []
        for cou in range(len(aa.source)):
            dec_h = float(aa.source[cou]['dec'].split('d')[0])
            dec_m = float((aa.source[cou]['dec'].split('d')[1])[0:2])
            dec_s = float((aa.source[cou]['dec'].split('d')[1])[3:-1])
            dec.append(tuple((dec_h,dec_m,dec_s)))    
        ra = []
        for cou in range(len(aa.source)):
            ra_d = float(aa.source[cou]['ra'].split('h')[0])
            ra_m = float(aa.source[cou]['ra'].split('h')[1].split('m')[0])
            ra_s = float(aa.source[cou]['ra'].split('h')[1].split('m')[1][:-1])
            ra.append(tuple((ra_d,ra_m,ra_s)))      
        sour_name = [aa.source[x]['source'] for x in range(len(aa.source))]
        dict_ra = dict(zip(sour_name,ra))
        dict_dec = dict(zip(sour_name,dec))
        t_min = [aa.sched[x]['start_hr'] for x in range(len(aa.sched))]
        sour = [aa.sched[x]['source'] for x in range(len(aa.sched))]
        datet = []
        elev = []
        antenas = []
        duration=[]
        for x in range(len(aa.sched)):#loop over scans in given file
            t = Time(aa.sched[x]['mjd_floor'], format='mjd', scale='utc')
            tiso = Time(t, format='iso', scale='utc')
            tiso = tiso + TimeDelta(t_min[x]*3600., format='sec')
            datet.append(tiso)
            ant_foo = set([nam2lett[aa.sched[x]['scan'][y]['site']] for y in range(len(aa.sched[x]['scan']))])
            antenas.append(ant_foo)
            duration_foo =max([aa.sched[x]['scan'][y]['scan_sec'] for y in range(len(aa.sched[x]['scan']))])
            duration.append(duration_foo)
        #time_min = [pd.tslib.Timestamp(datet[x].datetime) for x in range(len(datet))]
        time_min = [pd.Timestamp(datet[x].datetime) for x in range(len(datet))]
        time_max = [time_min[x] + datetime.timedelta(seconds=duration[x]) for x in range(len(aa.sched))]
        foo = pd.DataFrame(aa.sched)
        foo = foo[['source','mjd_floor','start_hr']]
        foo['time_min']=time_min
        foo['time_max']=time_max
        foo['scan_no'] = foo.index
        foo['scan_no'] = list(map(int,foo['scan_no']))
        foo['track'] = [track_loc]*foo.shape[0]
        foo['expt'] = [int(track2expt[track_loc])]*foo.shape[0]
        foo['antenas'] = antenas
        foo['duration'] = duration
        scans = pd.concat([scans,foo], ignore_index=True,sort=True)
    scans = scans.reindex(['mjd_floor','expt','track','scan_no','source','time_min','time_max','duration','antenas'],axis=1)
    scans = scans.sort_values('time_max')
    scans = scans.reset_index(drop=True)
    scans['scan_no_tot'] = scans.index
    return scans

def make_scan_list_EHT2018(fpath):
    '''
    generates data frame with information about scans for EHT2018
    '''
    nam2lett = {'ALMA':'A','APEX':'X','THULE':'G','LMT':'L','PICOVEL':'P','SMTO':'Z','SPT':'Y','JCMT':'J','SMAP':'S'}
    track2expt ={'C21':3644,'E22':3645, 'A24':3646, 'C25':3647,'G27':3648, 'D28':3649}
    list_files = [x.split('/')[-1] for x in glob.glob(fpath+'/*.vex')]
    #list_files = os.listdir(fpath)
    scans = pd.DataFrame({'source' : []})

    for fi in list_files:#loop over vex files in folder
        track_loc = fi[3:6].upper()
        vpath = os.path.join(fpath, fi)
        aa = vex.Vex(vpath)
        dec = []
        for cou in range(len(aa.source)):
            dec_h = float(aa.source[cou]['dec'].split('d')[0])
            dec_m = float((aa.source[cou]['dec'].split('d')[1])[0:2])
            dec_s = float((aa.source[cou]['dec'].split('d')[1])[3:-1])
            dec.append(tuple((dec_h,dec_m,dec_s)))    
        ra = []
        for cou in range(len(aa.source)):
            ra_d = float(aa.source[cou]['ra'].split('h')[0])
            ra_m = float(aa.source[cou]['ra'].split('h')[1].split('m')[0])
            ra_s = float(aa.source[cou]['ra'].split('h')[1].split('m')[1][:-1])
            ra.append(tuple((ra_d,ra_m,ra_s)))      
        sour_name = [aa.source[x]['source'] for x in range(len(aa.source))]
        dict_ra = dict(zip(sour_name,ra))
        dict_dec = dict(zip(sour_name,dec))
        t_min = [aa.sched[x]['start_hr'] for x in range(len(aa.sched))]
        sour = [aa.sched[x]['source'] for x in range(len(aa.sched))]
        datet = []
        elev = []
        antenas = []
        duration=[]
        for x in range(len(aa.sched)):#loop over scans in given file
            t = Time(aa.sched[x]['mjd_floor'], format='mjd', scale='utc')
            tiso = Time(t, format='iso', scale='utc')
            tiso = tiso + TimeDelta(t_min[x]*3600., format='sec')
            datet.append(tiso)
            ant_foo = set([nam2lett[aa.sched[x]['scan'][y]['site']] for y in range(len(aa.sched[x]['scan']))])
            antenas.append(ant_foo)
            duration_foo =max([aa.sched[x]['scan'][y]['scan_sec'] for y in range(len(aa.sched[x]['scan']))])
            duration.append(duration_foo)
        #time_min = [pd.tslib.Timestamp(datet[x].datetime) for x in range(len(datet))]
        time_min = [pd.Timestamp(datet[x].datetime) for x in range(len(datet))]
        time_max = [time_min[x] + datetime.timedelta(seconds=duration[x]) for x in range(len(aa.sched))]
        foo = pd.DataFrame(aa.sched)
        foo = foo[['source','mjd_floor','start_hr']]
        foo['time_min']=time_min
        foo['time_max']=time_max
        foo['scan_no'] = foo.index
        foo['scan_no'] = list(map(int,foo['scan_no']))
        foo['track'] = [track_loc]*foo.shape[0]
        foo['expt'] = [int(track2expt[track_loc])]*foo.shape[0]
        foo['antenas'] = antenas
        foo['duration'] = duration
        scans = pd.concat([scans,foo], ignore_index=True,sort=True)
    scans = scans.reindex(['mjd_floor','expt','track','scan_no','source','time_min','time_max','duration','antenas'],axis=1)
    scans = scans.sort_values('time_max')
    scans = scans.reset_index(drop=True)
    scans['scan_no_tot'] = scans.index
    return scans

def make_scan_list_EHT2021(fpath):
    '''
    generates data frame with information about scans for EHT2021
    '''
    nam2lett = {'ALMA':'A','APEX':'X','THULE':'G','LMT':'L','PICOVEL':'P','SMTO':'Z','SPT':'Y','JCMT':'J','SMAP':'S','NOEMA':'N','KITTPEAK':'K'}
    track2expt ={'B09':3762,'E13':3763, 'A14':3764, 'D15':3765,'A16':3766, 'A17':3767, 'E18':3768, 'F19':3769}
    list_files = [x.split('/')[-1] for x in glob.glob(fpath+'/*.vex')]
    #list_files = os.listdir(fpath)
    scans = pd.DataFrame({'source' : []})


    for fi in list_files:#loop over vex files in folder
        track_loc = fi[3:6].upper()
        vpath = os.path.join(fpath, fi)
        aa = vex.Vex(vpath)
        dec = []
        for cou in range(len(aa.source)):
            dec_h = float(aa.source[cou]['dec'].split('d')[0])
            dec_m = float((aa.source[cou]['dec'].split('d')[1])[0:2])
            dec_s = float((aa.source[cou]['dec'].split('d')[1])[3:-1])
            dec.append(tuple((dec_h,dec_m,dec_s)))    
        ra = []
        for cou in range(len(aa.source)):
            ra_d = float(aa.source[cou]['ra'].split('h')[0])
            ra_m = float(aa.source[cou]['ra'].split('h')[1].split('m')[0])
            ra_s = float(aa.source[cou]['ra'].split('h')[1].split('m')[1][:-1])
            ra.append(tuple((ra_d,ra_m,ra_s)))      
        sour_name = [aa.source[x]['source'] for x in range(len(aa.source))]
        dict_ra = dict(zip(sour_name,ra))
        dict_dec = dict(zip(sour_name,dec))
        t_min = [aa.sched[x]['start_hr'] for x in range(len(aa.sched))]
        sour = [aa.sched[x]['source'] for x in range(len(aa.sched))]
        datet = []
        elev = []
        antenas = []
        duration=[]
        for x in range(len(aa.sched)):#loop over scans in given file
            t = Time(aa.sched[x]['mjd_floor'], format='mjd', scale='utc')
            tiso = Time(t, format='iso', scale='utc')
            tiso = tiso + TimeDelta(t_min[x]*3600., format='sec')
            datet.append(tiso)
            ant_foo = set([nam2lett[aa.sched[x]['scan'][y]['site']] for y in range(len(aa.sched[x]['scan']))])
            antenas.append(ant_foo)
            duration_foo =max([aa.sched[x]['scan'][y]['scan_sec'] for y in range(len(aa.sched[x]['scan']))])
            duration.append(duration_foo)
        #time_min = [pd.tslib.Timestamp(datet[x].datetime) for x in range(len(datet))]
        time_min = [pd.Timestamp(datet[x].datetime) for x in range(len(datet))]
        time_max = [time_min[x] + datetime.timedelta(seconds=duration[x]) for x in range(len(aa.sched))]
        foo = pd.DataFrame(aa.sched)
        foo = foo[['source','mjd_floor','start_hr']]
        foo['time_min']=time_min
        foo['time_max']=time_max
        foo['scan_no'] = foo.index
        foo['scan_no'] = list(map(int,foo['scan_no']))
        foo['track'] = [track_loc]*foo.shape[0]
        foo['expt'] = [int(track2expt[track_loc])]*foo.shape[0]
        foo['antenas'] = antenas
        foo['duration'] = duration
        scans = pd.concat([scans,foo], ignore_index=True,sort=True)
    scans = scans.reindex(['mjd_floor','expt','track','scan_no','source','time_min','time_max','duration','antenas'],axis=1)
    scans = scans.sort_values('time_max')
    scans = scans.reset_index(drop=True)
    scans['scan_no_tot'] = scans.index
    return scans

def make_scan_list_EHT2022(fpath):
    '''
    generates data frame with information about scans for EHT2022
    '''
    nam2lett = {'ALMA':'A','APEX':'X','THULE':'G','LMT':'L','PICOVEL':'P','SMTO':'Z','SPT':'Y','JCMT':'J','SMAP':'S','NOEMA':'N','KITTPEAK':'K'}
    track2expt ={'G18':3803,'B19':3804, 'C20':3805, 'E22':3806,'D23':3807, 'A26':3808, 'F27':3809}
    list_files = [x.split('/')[-1] for x in glob.glob(fpath+'/*.vex')]
    #list_files = os.listdir(fpath)
    scans = pd.DataFrame({'source' : []})
    

    for fi in list_files:#loop over vex files in folder
        track_loc = fi[3:6].upper()
        vpath = os.path.join(fpath, fi)
        aa = vex.Vex(vpath)
        dec = []
        for cou in range(len(aa.source)):
            dec_h = float(aa.source[cou]['dec'].split('d')[0])
            dec_m = float((aa.source[cou]['dec'].split('d')[1])[0:2])
            dec_s = float((aa.source[cou]['dec'].split('d')[1])[3:-1])
            dec.append(tuple((dec_h,dec_m,dec_s)))    
        ra = []
        for cou in range(len(aa.source)):
            ra_d = float(aa.source[cou]['ra'].split('h')[0])
            ra_m = float(aa.source[cou]['ra'].split('h')[1].split('m')[0])
            ra_s = float(aa.source[cou]['ra'].split('h')[1].split('m')[1][:-1])
            ra.append(tuple((ra_d,ra_m,ra_s)))      
        sour_name = [aa.source[x]['source'] for x in range(len(aa.source))]
        dict_ra = dict(zip(sour_name,ra))
        dict_dec = dict(zip(sour_name,dec))
        t_min = [aa.sched[x]['start_hr'] for x in range(len(aa.sched))]
        sour = [aa.sched[x]['source'] for x in range(len(aa.sched))]
        datet = []
        elev = []
        antenas = []
        duration=[]
        for x in range(len(aa.sched)):#loop over scans in given file
            t = Time(aa.sched[x]['mjd_floor'], format='mjd', scale='utc')
            tiso = Time(t, format='iso', scale='utc')
            tiso = tiso + TimeDelta(t_min[x]*3600., format='sec')
            datet.append(tiso)
            ant_foo = set([nam2lett[aa.sched[x]['scan'][y]['site']] for y in range(len(aa.sched[x]['scan']))])
            antenas.append(ant_foo)
            duration_foo =max([aa.sched[x]['scan'][y]['scan_sec'] for y in range(len(aa.sched[x]['scan']))])
            duration.append(duration_foo)
        #time_min = [pd.tslib.Timestamp(datet[x].datetime) for x in range(len(datet))]
        time_min = [pd.Timestamp(datet[x].datetime) for x in range(len(datet))]
        time_max = [time_min[x] + datetime.timedelta(seconds=duration[x]) for x in range(len(aa.sched))]
        foo = pd.DataFrame(aa.sched)
        foo = foo[['source','mjd_floor','start_hr']]
        foo['time_min']=time_min
        foo['time_max']=time_max
        foo['scan_no'] = foo.index
        foo['scan_no'] = list(map(int,foo['scan_no']))
        foo['track'] = [track_loc]*foo.shape[0]
        foo['expt'] = [int(track2expt[track_loc])]*foo.shape[0]
        foo['antenas'] = antenas
        foo['duration'] = duration
        scans = pd.concat([scans,foo], ignore_index=True,sort=True)
    scans = scans.reindex(['mjd_floor','expt','track','scan_no','source','time_min','time_max','duration','antenas'],axis=1)
    scans = scans.sort_values('time_max')
    scans = scans.reset_index(drop=True)
    scans['scan_no_tot'] = scans.index
    return scans


def match_scans_bysource(scans,data):
    '''
    matches data with scans
    '''
    sources = data.source.unique()
    for sour in sources:
        scans.drop(scans[scans.source != sour].index, inplace = True)
    #data =data0.copy()
    bins_labels = [None]*(2*scans.shape[0]-1)
    bins_labels[1::2] = map(lambda x: -x-1,list(scans['scan_no_tot'])[:-1])
    bins_labels[::2] = list(scans['scan_no_tot'])
    dtmin = datetime.timedelta(seconds = 2.) 
    dtmax = datetime.timedelta(seconds = 2.)
    binsT = [None]*(2*scans.shape[0])
    binsT[::2] = list(map(lambda x: x - dtmin,list(scans.time_min)))
    binsT[1::2] = list(map(lambda x: x + dtmax,list(scans.time_max)))
    ordered_labels = pd.cut(data.datetime, binsT,labels = bins_labels)
    data['scan_no_tot'] = ordered_labels
    data = data[list(map(lambda x: x >= 0, data['scan_no_tot']))]
    data['scan_id'] = data['scan_no_tot']
    data.drop('scan_no_tot',axis=1,inplace=True)
    return data



def round_time(t,round_s=1):
    """rounding time to given accuracy
    Args:
        t: time
        round_s: delta time to round to in seconds
    Returns:
        round_t: rounded time
    """
    t0 = datetime.datetime(t.year,1,1)
    foo = t - t0
    foo_s = foo.days*24*3600 + foo.seconds + foo.microseconds*(1e-6)
    foo_s = np.round(foo_s/round_s)*round_s
    days = np.floor(foo_s/24/3600)
    seconds = np.floor(foo_s - 24*3600*days)
    microseconds = int(1e6*(foo_s - days*3600*24 - seconds))
    round_t = t0+datetime.timedelta(days,seconds,microseconds)
    return round_t

def obsdata_2_df(obs):
    """converts visibilities from obs.data to DataFrame format
    Args:
        obs: ObsData object
        round_s: accuracy of datetime object in seconds
    Returns:
        df: observation visibility data in DataFrame format
    """
    sour=obs.source
    df = pd.DataFrame(data=obs.data)
    df['fmjd'] = df['time']/24.
    df['mjd'] = obs.mjd + df['fmjd']
    telescopes = list(zip(df['t1'],df['t2']))
    #reorder stations alphabetically
    df['is_alphabetic'] = df['t1']<df['t2']
    df_alphabetic = df[df.is_alphabetic==True].copy()
    df_non_alphabetic = df[df.is_alphabetic==False].copy()
    fooT1 = list(df_non_alphabetic['t1'])
    fooT2= list(df_non_alphabetic['t2'])
    df_non_alphabetic['t1'] = fooT2
    df_non_alphabetic['t2'] = fooT1
    if 'rrvis' in df.columns:
        df_non_alphabetic['rrvis'] = np.conjugate(df_non_alphabetic['rrvis'])
        df_non_alphabetic['llvis'] = np.conjugate(df_non_alphabetic['llvis'])
        df_non_alphabetic['rlvis'] = np.conjugate(df_non_alphabetic['lrvis'])#conj+transpose!
        df_non_alphabetic['lrvis'] = np.conjugate(df_non_alphabetic['rlvis'])#conj+transpose!
    if 'vis' in df.columns:
        df_non_alphabetic['vis'] = np.conjugate(df_non_alphabetic['vis'])
    if 'phase' in df.columns:
        df_non_alphabetic['phase'] = - df_non_alphabetic['phase']
    df = pd.concat([df_alphabetic,df_non_alphabetic],ignore_index=True)
    #---------------------------------
    try:
        telescopes = [(x[0].decode('unicode_escape'),x[1].decode('unicode_escape') ) for x in telescopes]
    except AttributeError:
        telescopes = [(x[0],x[1] ) for x in telescopes]
    df['baseline'] = [x[0]+'-'+x[1] for x in telescopes]
    df['jd'] = Time(df['mjd'], format='mjd').jd
    df['source'] = sour
    df['baselength'] = np.sqrt(np.asarray(df.u)**2+np.asarray(df.v)**2)
    if 'vis' not in df.columns:
        try: df['vis'] = df['rrvis']
        except:  pass
    if 'amp' not in df.columns:
        try: df['amp'] = np.abs(df['vis'])
        except:  pass
    if 'phase' not in df.columns:
        try: df['phase'] = np.angle(df['rrvis'])*180/np.pi
        except:  pass
    if 'sigma' not in df.columns:
        try: df['sigma'] = df['rrsigma']
        except: pass
    return df
    
def get_df_from_uvfit(pathf,observation='EHT2017',path_vex='',force_singlepol='no',band='unknown',
    round_s=0.1,only_parallel=False,rescale_noise=False,polrep=None,path_ehtim='',fix_sigma=0,scale_sigma=1.):
    """generate DataFrame from uvfits file
    Args:
        pathf: path to uvfits file to import
        observation: name of observation campaign, to generate 'expt' info and 1-letter station codes
        by default EHT2017
        path_vex: path to folder with vex files used to generate 'scan_id' data
        by default no vex file, so 'scan_id' not provided in the dataframe
        force_singlepol: '' if all polarizations in the uvfits file, otherwise specify the unique 
        polarization in the uvfits file
        band: manually provide info about the band for the 'band' column
        round_s: precision of converting fractional mjd to datetime object, in seconds
    """
    if path_ehtim!='':
        sys.path.insert(0,path_ehtim)
        logging.info(f'Inserting path {path_ehtim}')
    #if 'ehtim' in list(sys.modules):
    #    print('Ehtim already loaded!')
    #    import ehtim as eh
    #    #try: 
    #    reload(eh)
    #    print('Ehtim reloaded...')
    #    #except: pass
    import ehtim as eh
    path_eh = os.path.dirname(eh.__file__)
    logging.info(f'Using eht-imaging library from {path_eh}')

    if force_singlepol=='LL':
        force_singlepol='L'
    if force_singlepol=='RR':
        force_singlepol='R'

    if force_singlepol=='no':
        logging.info(f'Reading data without forcing singlepol and using polrep = {polrep}')
        filen = os.path.basename(pathf)
        if polrep in ['circ','stokes']:
            obsXX = eh.io.load.load_obs_uvfits(pathf,polrep=polrep)
            logging.info(f'Polrep is {obsXX.polrep}')
        else: 
            logging.warning("Polrep unspecified. Default ehtim behavior (polrep='stokes') will be used.") 
            obsXX = eh.io.load.load_obs_uvfits(pathf)
        dfXX = obsdata_2_df(obsXX)
        
        if 'RR' in filen:
            dfXX['polarization'] = 'RR'    
        elif 'LL' in filen:
            dfXX['polarization'] = 'LL' 
        else:
            dfXX['polarization'] = 'WTF' 
        df = dfXX.copy()
        df['band'] = band

        #Scale sigma
        df['sigma']=scale_sigma*df['sigma']
        if 'rrsigma' in df.columns:
            df['rrsigma'] = scale_sigma*df['rrsigma']
            df['llsigma'] = scale_sigma*df['llsigma']
            df['rlsigma'] = scale_sigma*df['rlsigma']
            df['lrsigma'] = scale_sigma*df['lrsigma']
        
        if fix_sigma>0:
            logging.info(f'Fixing constant sigma: {fix_sigma}')
            df['sigma']=fix_sigma
            if 'rrsigma' in df.columns:
                df['rrsigma'] = fix_sigma
                df['llsigma'] = fix_sigma
                df['rlsigma'] = fix_sigma
                df['lrsigma'] = fix_sigma

        if rescale_noise==True:
            obsXX10 = obsXX.avg_coherent(10.)
            rsc = obsXX10.estimate_noise_rescale_factor(max_diff_sec=1000.)
            logging.info(f'Applying noise rescaling to data, factor is: {rsc}')
            df['sigma']=rsc*df['sigma']
            if 'rrsigma' in df.columns:
                df['rrsigma'] = rsc*df['rrsigma']
                df['llsigma'] = rsc*df['llsigma']
                df['rlsigma'] = rsc*df['rlsigma']
                df['lrsigma'] = rsc*df['lrsigma']

    elif force_singlepol=='':
        if polrep in ['circ','stokes']:
            obsRR = eh.io.load.load_obs_uvfits(pathf,  force_singlepol='R',polrep=polrep)
            obsLL = eh.io.load.load_obs_uvfits(pathf,  force_singlepol='L',polrep=polrep)
        else:
            obsRR = eh.io.load.load_obs_uvfits(pathf,  force_singlepol='R')
            obsLL = eh.io.load.load_obs_uvfits(pathf,  force_singlepol='L')
        dfRR = obsdata_2_df(obsRR)
        dfLL = obsdata_2_df(obsLL)
        dfRR['polarization'] = 'RR'
        dfLL['polarization'] = 'LL'
        df = pd.concat([dfRR,dfLL],ignore_index=True)

        # Load off-diagonal visibilities if only_parallel is False and append to df
        if only_parallel==False:
            if polrep in ['circ','stokes']:
                obsRL = eh.io.load.load_obs_uvfits(pathf,  force_singlepol='RL',polrep=polrep)
                obsLR = eh.io.load.load_obs_uvfits(pathf,  force_singlepol='LR',polrep=polrep)
            else:
                obsRL = eh.io.load.load_obs_uvfits(pathf,  force_singlepol='RL')
                obsLR = eh.io.load.load_obs_uvfits(pathf,  force_singlepol='LR')
            dfRL = obsdata_2_df(obsRL)
            dfLR = obsdata_2_df(obsLR)
            dfRL['polarization'] = 'RL'
            dfLR['polarization'] = 'LR'
            df = pd.concat([df,dfLR,dfRL],ignore_index=True)

        df['band'] = band  
        if rescale_noise==True:
            rscRR = obsRR.estimate_noise_rescale_factor()
            rscLL = obsLL.estimate_noise_rescale_factor()
            rsc=0.5*(rscRR+rscLL)
            df['sigma']=rsc*df['sigma']

    else: 
        obs = eh.io.load.load_obs_uvfits(pathf,  force_singlepol=force_singlepol)
        df = obsdata_2_df(obs)
        if len(force_singlepol)==1:
            df['polarization']=force_singlepol+force_singlepol
        elif len(force_singlepol)==2:
            df['polarization']=force_singlepol
        df['band'] = band
        if rescale_noise==True:
            obs10 = obs.avg_coherent(10.)
            rsc= obs10.estimate_noise_rescale_factor()
            df['sigma']=rsc*df['sigma']
     
    stations_2lett_1lett, jd_2_expt, scans = get_info(observation=observation, path_vex=path_vex)
    
    df['datetime'] = Time(df['mjd'], format='mjd').datetime
    df['datetime'] =list(map(lambda x: round_time(x,round_s=round_s),df['datetime']))
    df['expt_no'] = list(map(jd_2_expt,df['jd']))
    try:
        df['baseline'] = list(map(lambda x: stations_2lett_1lett[x[0].decode('unicode_escape')]+stations_2lett_1lett[x[1].decode('unicode_escape')],zip(df['t1'],df['t2'])))
    except AttributeError:
        df['baseline'] = list(map(lambda x: stations_2lett_1lett[x[0]]+stations_2lett_1lett[x[1]],zip(df['t1'],df['t2'])))
    df = df[df.baseline.str[0]!=df.baseline.str[1]]
    
    #is_alphabetic = list(map(lambda x: float(x== ''.join(sorted([x[0],x[1]]))),df['baseline']))
    #df['baseline'] = list(map(lambda x: ''.join(sorted([x[0],x[1]])),df['baseline']))
    if 'vis' in df.columns:
        df['amp'] = list(map(np.abs,df['vis']))
        df['snr'] = df['amp']/df['sigma']
        df['phase'] = np.angle(df['vis'])*180./np.pi
    #conjugate phase if baseline letters order has been reversed to make it alphabetic
    #df['phase'] = list(map(lambda x: (2.*x[1]-1.)*(180./np.pi)*np.angle(x[0]),zip(df['vis'],is_alphabetic)))
    if path_vex!='': 
        df = match_scans_bysource(scans,df)   
    return df

def save_uvfits(datastruct, fname):
    """
    Save information in UVFITS format to a UVFITS file.
    Parameters
    ----------
    datastruct : Datastruct
        A datastruct object with type 'UVFITS' containing the data and metadata to be saved.
    fname : str
        Filename (including path) to save the UVFITS file to.
    Raises
    ------
    Exception
        If `datastruct.dtype` is not 'UVFITS'.
        If the shapes of u, v, bls, jds, tints, and outdat are not consistent.
    Notes
    -----
    - The function writes the UVFITS file using astropy.io.fits, including the primary data, antenna table (AIPS AN),
      frequency table (AIPS FQ), and scan table (AIPS NX).
    - Some header fields and table columns are set to default or placeholder values and may require further customization
      for specific instruments or datasets.
    - The function assumes that the input datastruct contains all necessary fields and arrays in the correct format.
    Returns
    -------
    int
        Returns 0 upon successful completion.
    """

    # unpack data
    if datastruct.dtype != 'UVFITS':
        raise Exception("datastruct.dtype != 'UVFITS' in save_uvfits()!")

    src = datastruct.obs_info.src
    ra = datastruct.obs_info.ra
    dec = datastruct.obs_info.dec
    ref_freq = datastruct.obs_info.ref_freq
    ch_bw = datastruct.obs_info.ch_bw
    ch_spacing = datastruct.obs_info.ch_spacing
    ch1_freq = datastruct.obs_info.ch_1
    nchan = datastruct.obs_info.nchan
    scan_arr = datastruct.obs_info.scans
    bw = nchan*ch_bw

    antnames = datastruct.antenna_info.antnames
    antnums = datastruct.antenna_info.antnums
    xyz = datastruct.antenna_info.xyz
    nsta = len(antnames)

    u = datastruct.data.u
    v = datastruct.data.v
    bls = datastruct.data.bls
    jds = datastruct.data.jds
    tints = datastruct.data.tints
    outdat = datastruct.data.datatable

    if (len(u) != len(v) != len(bls) != len(jds) != len(tints) != len(outdat)):
        raise Exception("rg parameter shapes and data shape not consistent!")

    ndat = len(u)
    mjd = int(np.min(jds) - MJD_0)
    jd_start = (MJD_0 + mjd)
    fractimes = (jds - jd_start)
    jds_only = np.ones(ndat) * jd_start

    #print "timedur uvfits " , (np.max(jds) - np.min(jds)) * 3600 * 24, (np.max(fractimes) - np.min(fractimes)) * 3600 * 24
    nsubchan = 1
    nstokes = 4

    # Create new HDU
    hdulist = fits.HDUList()
    hdulist.append(fits.GroupsHDU())

    ##################### DATA TABLE ##################################################################################################
    # Data header
    header = hdulist['PRIMARY'].header

    #mandatory
    header['OBJECT'] = src
    header['TELESCOP'] = 'VLBA' # TODO Can we change this field?
    header['INSTRUME'] = 'VLBA'
    header['OBSERVER'] = 'EHT'
    header['BSCALE'] = 1.0
    header['BZERO'] = 0.0
    header['BUNIT'] = 'JY'
    header['EQUINOX'] = 2000
    header['ALTRPIX'] = 1.e0
    header['ALTRVAL'] = 0.e0

    #optional
    header['OBSRA'] = ra * 180./12.
    header['OBSDEC'] = dec
    header['MJD'] = float(mjd)
    # new astropy broke this subfmt for jd for some reason
    # header['DATE-OBS'] = Time(mjd + MJD_0, format='jd', scale='utc', out_subfmt='date').iso
    header['DATE-OBS'] = Time(mjd + MJD_0, format='jd', scale='utc').iso[:10]
    #header['DATE-MAP'] = ??
    #header['VELREF'] = 3

    # DATA AXES #
    header['NAXIS'] = 7
    header['NAXIS1'] = 0

    # real, imag, weight
    header['CTYPE2'] = 'COMPLEX'
    header['NAXIS2'] = 3
    header['CRVAL2'] = 1.e0
    header['CDELT2'] = 1.e0
    header['CRPIX2'] = 1.e0
    header['CROTA2'] = 0.e0
    # RR, LL, RL, LR
    header['CTYPE3'] = 'STOKES'
    header['NAXIS3'] = nstokes
    header['CRVAL3'] = -1.e0 #corresponds to RR LL RL LR
    header['CDELT3'] = -1.e0
    header['CRPIX3'] = 1.e0
    header['CROTA3'] = 0.e0
    # frequencies
    header['CTYPE4'] = 'FREQ'
    header['NAXIS4'] = nsubchan
    header['CRPIX4'] = 1.e0
    # header['CRVAL4'] = ch1_freq # is this the right ref freq? in Hz
    header['CRVAL4'] = ref_freq # is this the right ref freq? in Hz
    header['CDELT4'] = ch_bw
    header['CROTA4'] = 0.e0
    # frequencies
    header['CTYPE5'] = 'IF'
    header['NAXIS5'] = nchan
    header['CRPIX5'] = 1.e0
    header['CRVAL5'] = 1.e0
    header['CDELT5'] = 1.e0
    header['CROTA5'] = 0.e0
    # RA
    header['CTYPE6'] = 'RA'
    header['NAXIS6'] = 1.e0
    header['CRPIX6'] = 1.e0
    header['CRVAL6'] = header['OBSRA']
    header['CDELT6'] = 1.e0
    header['CROTA6'] = 0.e0
    # DEC
    header['CTYPE7'] = 'DEC'
    header['NAXIS7'] = 1.e0
    header['CRPIX7'] = 1.e0
    header['CRVAL7'] = header['OBSDEC']
    header['CDELT7'] = 1.e0
    header['CROTA7'] = 0.e0

    ##RANDOM PARAMS##
    header['PTYPE1'] = 'UU---SIN'
    header['PSCAL1'] = 1/ref_freq
    header['PZERO1'] = 0.e0
    header['PTYPE2'] = 'VV---SIN'
    header['PSCAL2'] = 1.e0/ref_freq
    header['PZERO2'] = 0.e0
    header['PTYPE3'] = 'WW---SIN'
    header['PSCAL3'] = 1.e0/ref_freq
    header['PZERO3'] = 0.e0
    header['PTYPE4'] = 'BASELINE'
    header['PSCAL4'] = 1.e0
    header['PZERO4'] = 0.e0
    header['PTYPE5'] = 'DATE'
    header['PSCAL5'] = 1.e0
    header['PZERO5'] = 0.e0
    header['PTYPE6'] = 'DATE'
    header['PSCAL6'] = 1.e0
    header['PZERO6'] = 0.0
    header['PTYPE7'] = 'INTTIM'
    header['PSCAL7'] = 1.e0
    header['PZERO7'] = 0.e0
    header['history'] = "AIPS SORT ORDER='TB'"

    # Save data
    pars = ['UU---SIN', 'VV---SIN', 'WW---SIN', 'BASELINE', 'DATE', 'DATE', 'INTTIM']
    x = fits.GroupData(outdat, parnames=pars, pardata=[u, v, np.zeros(ndat), np.array(bls).reshape(-1), jds_only, fractimes, tints], bitpix=-32)

    hdulist['PRIMARY'].data = x
    hdulist['PRIMARY'].header = header

    ####################### AIPS AN TABLE ###############################################################################################
    #Antenna Table entries
    col1 = fits.Column(name='ANNAME', format='8A', array=antnames)
    col2 = fits.Column(name='STABXYZ', format='3D', unit='METERS', array=xyz)
    col3= fits.Column(name='ORBPARM', format='D', array=np.zeros(0))
    col4 = fits.Column(name='NOSTA', format='1J', array=antnums)

    #TODO get the actual information for these parameters for each station
    col5 = fits.Column(name='MNTSTA', format='1J', array=np.zeros(nsta)) #zero = alt-az
    col6 = fits.Column(name='STAXOF', format='1E', unit='METERS', array=np.zeros(nsta)) #zero = no axis  offset
    col7 = fits.Column(name='POLTYA', format='1A', array=np.array(['R' for i in range(nsta)], dtype='|S1')) #RCP
    col8 = fits.Column(name='POLAA', format='1E', unit='DEGREES', array=np.zeros(nsta)) #feed orientation A
    col9 = fits.Column(name='POLCALA', format='2E', array=np.zeros((nsta,2))) #zero = no pol cal info TODO should have extra dim for nif
    col10 = fits.Column(name='POLTYB', format='1A', array=np.array(['L' for i in range(nsta)], dtype='|S1')) #LCP
    col11 = fits.Column(name='POLAB', format='1E', unit='DEGREES', array=90*np.ones(nsta)) #feed orientation A
    col12 = fits.Column(name='POLCALB', format='2E', array=np.zeros((nsta,2))) #zero = no pol cal info

    # create table
    tbhdu = fits.BinTableHDU.from_columns(fits.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12]), name='AIPS AN')
    hdulist.append(tbhdu)

    # header information
    head = hdulist['AIPS AN'].header
    head['EXTVER'] = 1
    head['ARRAYX'] = 0.e0
    head['ARRAYY'] = 0.e0
    head['ARRAYZ'] = 0.e0

    # TODO change the reference date
    #rdate_out = RDATE
    #rdate_gstiao_out = RDATE_GSTIA0
    #rdate_offset_out = RDATE_OFFSET

    # new astropy broke this subfmt, it should be a day boundary hopefully
    # rdate_tt_new = Time(mjd + MJD_0, format='jd', scale='utc', out_subfmt='date')
    rdate_tt_new = Time(mjd + MJD_0, format='jd', scale='utc')
    rdate_out = rdate_tt_new.iso[:10]
    rdate_jd_out = rdate_tt_new.jd
    rdate_gstiao_out = rdate_tt_new.sidereal_time('apparent','greenwich').degree
    rdate_offset_out = (rdate_tt_new.ut1.datetime.second - rdate_tt_new.utc.datetime.second)
    rdate_offset_out += 1.e-6*(rdate_tt_new.ut1.datetime.microsecond - rdate_tt_new.utc.datetime.microsecond)

    head['RDATE'] = rdate_out
    head['GSTIA0'] = rdate_gstiao_out
    head['DEGPDY'] = RDATE_DEGPERDY
    head['UT1UTC'] = rdate_offset_out   #difference between UT1 and UTC ?
    head['DATUTC'] = 0.e0
    head['TIMESYS'] = 'UTC'

    head['FREQ']= ref_freq
    head['POLARX'] = 0.e0
    head['POLARY'] = 0.e0

    head['ARRNAM'] = 'VLBA'  # TODO must be recognized by aips/casa
    head['XYZHAND'] = 'RIGHT'
    head['FRAME'] = '????'
    head['NUMORB'] = 0
    head['NO_IF'] = nchan
    head['NOPCAL'] = 0  #TODO add pol cal information
    head['POLTYPE'] = 'VLBI'
    head['FREQID'] = 1

    hdulist['AIPS AN'].header = head

    ##################### AIPS FQ TABLE #####################################################################################################
    # Convert types & columns
    freqid = np.array([1])
    bandfreq = np.array([ch1_freq + ch_spacing*i - ref_freq for i in range(nchan)]).reshape([1,nchan])
    chwidth = np.array([ch_bw for i in range(nchan)]).reshape([1,nchan])
    totbw = np.array([ch_bw for i in range(nchan)]).reshape([1,nchan])
    sideband = np.array([1 for i in range(nchan)]).reshape([1,nchan])

    freqid = fits.Column(name="FRQSEL", format="1J", array=freqid)
    bandfreq = fits.Column(name="IF FREQ", format="%dD"%(nchan), array=bandfreq, unit='HZ')
    chwidth = fits.Column(name="CH WIDTH",format="%dE"%(nchan), array=chwidth, unit='HZ')
    totbw = fits.Column(name="TOTAL BANDWIDTH",format="%dE"%(nchan),array=totbw, unit='HZ')
    sideband = fits.Column(name="SIDEBAND",format="%dJ"%(nchan),array=sideband)
    cols = fits.ColDefs([freqid, bandfreq, chwidth, totbw, sideband])

    # create table
    tbhdu = fits.BinTableHDU.from_columns(cols)

    # header information
    tbhdu.header.append(("NO_IF", nchan, "Number IFs"))
    tbhdu.header.append(("EXTNAME","AIPS FQ"))
    tbhdu.header.append(("EXTVER",1))
    hdulist.append(tbhdu)

    ##################### AIPS NX TABLE #####################################################################################################

    scan_times = []
    scan_time_ints = []
    start_vis = []
    stop_vis = []

    #TODO make sure jds AND scan_info MUST be time sorted!!
    jj = 0
    #print scan_info

    comp_fac = 3600*24*100 # compare to 100th of a second

    for scan in  scan_arr:
        scan_start = round(scan[0], ROUND_SCAN_INT)
        scan_stop = round(scan[1], ROUND_SCAN_INT)
        scan_dur = (scan_stop - scan_start)

        if jj>=len(jds):
            #print start_vis, stop_vis
            break

        # print "%.12f %.12f %.12f" %( jds[jj], scan_start, scan_stop)
        jd = round(jds[jj], ROUND_SCAN_INT)*comp_fac # ANDREW TODO precision??

        if (np.floor(jd) >= np.floor(scan_start*comp_fac)) and (np.ceil(jd) <= np.ceil(comp_fac*scan_stop)):
            start_vis.append(jj)
            # TODO AIPS MEMO 117 says scan_times should be midpoint!, but AIPS data looks likes it's at the start?
            #scan_times.append(scan_start  - rdate_jd_out)
            scan_times.append(scan_start + 0.5*scan_dur - rdate_jd_out)
            scan_time_ints.append(scan_dur)
            while (jj < len(jds) and np.floor(round(jds[jj],ROUND_SCAN_INT)*comp_fac) <= np.ceil(comp_fac*scan_stop)):
                jj += 1
            stop_vis.append(jj-1)
        else:
            continue

    if jj < len(jds):
        if len(scan_arr) == 0:
            print("len(scan_arr) == 0")
        else:
            print(scan_arr[-1])
            print(round(scan_arr[-1][0],ROUND_SCAN_INT),round(scan_arr[-1][1],ROUND_SCAN_INT))
        print(jj, len(jds), round(jds[jj], ROUND_SCAN_INT))
        print("WARNING!!!: in save_uvfits NX table, didn't get to all entries when computing scan start/stop!")
        #raise Exception("in save_uvfits NX table, didn't get to all entries when computing scan start/stop!")

    time_nx = fits.Column(name="TIME", format="1D", unit='DAYS', array=np.array(scan_times))
    timeint_nx = fits.Column(name="TIME INTERVAL", format="1E", unit='DAYS', array=np.array(scan_time_ints))
    sourceid_nx = fits.Column(name="SOURCE ID",format="1J", unit='', array=np.ones(len(scan_times)))
    subarr_nx = fits.Column(name="SUBARRAY",format="1J", unit='', array=np.ones(len(scan_times)))
    freqid_nx = fits.Column(name="FREQ ID",format="1J", unit='', array=np.ones(len(scan_times)))
    startvis_nx = fits.Column(name="START VIS",format="1J", unit='', array=np.array(start_vis)+1)
    endvis_nx = fits.Column(name="END VIS",format="1J", unit='', array=np.array(stop_vis)+1)
    cols = fits.ColDefs([time_nx, timeint_nx, sourceid_nx, subarr_nx, freqid_nx, startvis_nx, endvis_nx])

    tbhdu = fits.BinTableHDU.from_columns(cols)

    # header information
    tbhdu.header.append(("EXTNAME","AIPS NX"))
    tbhdu.header.append(("EXTVER",1))

    hdulist.append(tbhdu)

    # Write final HDUList to file
    #hdulist.writeto(fname, clobber=True)#this is deprecated and changed to overwrite
    hdulist.writeto(fname, overwrite=True)

    return 0

def load_hops_uvfits(filename):
    """
    Reads a UVFITS file and returns its data in a Datastruct object.
    Parameters
    ----------
    filename : str
        Path to the UVFITS file to be read.
    Returns
    -------
    Datastruct
        An object containing the parsed UVFITS data, including observation info,
        antenna info, and visibility data.
    Raises
    ------
    Exception
        If the observing frequency/bandwidth or number of channels cannot be found.
    Notes
    -----
    - The function expects specific HDU tables and header keywords to be present in the UVFITS file.
    - Scan times are calculated as midpoints based on scan start and duration.
    - Visibility data is converted from light seconds to lambda units using the reference frequency.
    """
    # Read the uvfits file
    logging.debug(f"Reading uvfits file: {filename}")
    hdulist = fits.open(filename)
    header = hdulist[0].header
    data = hdulist[0].data

    # Load the telescope array data
    tnames = hdulist['AIPS AN'].data['ANNAME']
    tnums = hdulist['AIPS AN'].data['NOSTA'] - 1
    xyz = hdulist['AIPS AN'].data['STABXYZ']
    antennainfo = Antenna_info(tnames, tnums, xyz)

    # Load the various observing header parameters
    if 'OBSRA' not in header.keys(): header['OBSRA'] = header['CRVAL6']
    ra = header['OBSRA'] * 12./180.
    if 'OBSDEC' not in header.keys(): header['OBSDEC'] = header['CRVAL7']
    dec = header['OBSDEC']
    src = header['OBJECT']

    rf = hdulist['AIPS AN'].header['FREQ']

    if header['CTYPE4'] == 'FREQ':
        ch1_freq = header['CRVAL4'] + hdulist['AIPS FQ'].data['IF FREQ'][0][0]
        ch_bw = header['CDELT4']
    else: raise Exception('Cannot find observing frequency/bandwidth!')

    if header['CTYPE5'] == 'IF':
        nchan = header['NAXIS5']
    else: raise Exception('Cannot find number of channels!')

    num_ifs = len(hdulist['AIPS FQ'].data['IF FREQ'][0])
    if (num_ifs>1):
        ch_spacing = hdulist['AIPS FQ'].data['IF FREQ'][0][1] - hdulist['AIPS FQ'].data['IF FREQ'][0][0]
    else: raise Exception('Cannot find uvfits channel spacing in AIPS FREQ table!')

    # load the scan information
    
    try: 
        refdate_str = hdulist['AIPS AN'].header['RDATE'] # in iso
        refdate = Time(refdate_str, format='isot', scale='utc').jd
    except ValueError: 
        logging.warning('ValueError in reading AIPS AN RDATE! Using PRIMARY DATE-OBS value')
        refdate_str = hdulist['PRIMARY'].header['DATE-OBS'] # in iso
        refdate = Time(refdate_str, format='isot', scale='utc').jd

    try: scan_starts = hdulist['AIPS NX'].data['TIME'] #in days since reference date
    except KeyError: scan_starts=[]
    try: scan_durs = hdulist['AIPS NX'].data['TIME INTERVAL']
    except KeyError: scan_durs=[]
    scan_arr = []
    
    for kk in range(len(scan_starts)):
        scan_start = scan_starts[kk]
        scan_dur = scan_durs[kk]
        # TODO AIPS MEMO 117 says scan_times should be midpoint!, but AIPS data looks likes it's at the start?
        #scan_arr.append([scan_start + refdate,
        #                  scan_start + scan_dur + refdate])
        scan_arr.append([scan_start - 0.5*scan_dur + refdate,
                         scan_start + 0.5*scan_dur + refdate])

    scan_arr = np.array(scan_arr)
    obsinfo = Obs_info(src, ra, dec, rf, ch_bw, ch_spacing, ch1_freq, nchan, scan_arr)

    # Load the random group parameters and the visibility data
    # Convert uv in lightsec to lambda by multiplying by rf
    try:
        u = data['UU---SIN'] * rf
        v = data['VV---SIN'] * rf
    except KeyError:
        u = data['UU'] * rf
        v = data['VV'] * rf
    baselines = data['BASELINE']
    jds = data['DATE'].astype('d') + data['_DATE'].astype('d')


    try: tints = data['INTTIM']
    except KeyError: tints = np.array([1]*np.shape(data)[0], dtype='float32')
    obsdata = data['DATA']

    alldata = Uvfits_data(u,v,baselines,jds, tints, obsdata)

    return Datastruct(obsinfo, antennainfo, alldata)

def convert_uvfits_to_datastruct(filename):
    """
    Converts a UVFITS file to a Datastruct object compatible with EHTIM.
    This function loads a UVFITS file, extracts relevant header and data parameters,
    and organizes them into a Datastruct object. It processes telescope array information,
    baseline data, observation times, and visibility data for all polarization products.
    The output Datastruct contains all necessary metadata and data arrays for further analysis.
    Parameters
    ----------
    filename : str
        Path to the UVFITS file to be converted.
    Returns
    -------
    datastruct_out : Datastruct
        Datastruct object containing the parsed UVFITS data, including observation info,
        telescope array information, and a datatable of visibilities and weights.
    Notes
    -----
    - The function assumes the existence of supporting types and functions such as
      `load_hops_uvfits`, `DTARR`, `DTPOL`, and `Datastruct`.
    - Flags from the UVFITS file are not currently extracted.
    - The function is tailored for EHTIM data structures and may require adaptation
      for other use cases.
    """
    
    # Load the uvfits file to a UVFITS format datasctruct
    datastruct = load_hops_uvfits(filename)

    # get the various necessary header parameters
    ch1_freq = datastruct.obs_info.ch_1
    ch_spacing = datastruct.obs_info.ch_spacing
    nchan = datastruct.obs_info.nchan

    # put the array data in a telescope array format
    tnames = datastruct.antenna_info.antnames
    tnums = datastruct.antenna_info.antnums
    xyz = datastruct.antenna_info.xyz

    tarr = [np.array((tnames[i], xyz[i][0], xyz[i][1], xyz[i][2]),
            dtype=DTARR) for i in range(len(tnames))]
    tarr = np.array(tarr)

    # put the random group data and vis data into a data array
    # Convert uv in lightsec to lambda by multiplying by rf
    u = datastruct.data.u
    v = datastruct.data.v
    baseline = datastruct.data.bls
    jds = datastruct.data.jds
    tints = datastruct.data.tints
    obsdata = datastruct.data.datatable

    # Sites - add names
    t1 = baseline.astype(int)//256 # python3
    t2 = baseline.astype(int) - t1*256
    t1 = t1 - 1
    t2 = t2 - 1
    t1 = np.array([tarr[i]['site'] for i in t1])
    t2 = np.array([tarr[i]['site'] for i in t2])

    # Obs Times
    #mjd = int(np.min(jds) - MJD_0)
    #times = (jds - MJD_0 - mjd) * 24.0

    # Get vis data
    rr = obsdata[:,0,0,:,0,0,0] + 1j*obsdata[:,0,0,:,0,0,1]
    ll = obsdata[:,0,0,:,0,1,0] + 1j*obsdata[:,0,0,:,0,1,1]
    rl = obsdata[:,0,0,:,0,2,0] + 1j*obsdata[:,0,0,:,0,2,1]
    lr = obsdata[:,0,0,:,0,3,0] + 1j*obsdata[:,0,0,:,0,3,1]

    rrweight = obsdata[:,0,0,:,0,0,2]
    llweight = obsdata[:,0,0,:,0,1,2]
    rlweight = obsdata[:,0,0,:,0,2,2]
    lrweight = obsdata[:,0,0,:,0,3,2]

    # Make a datatable
    # TODO check that the jd not cut off by precision
    datatable = np.empty((len(jds)*nchan), dtype=DTPOL)
    idx = 0
    for i in range(len(jds)):
        for j in range(nchan):
            freq = j*ch_spacing + ch1_freq
            datatable[idx] = np.array((jds[i], freq, tints[i], t1[i], t2[i], u[i], v[i], rr[i,j], ll[i,j], rl[i,j], lr[i,j], \
                              rrweight[i,j], llweight[i,j], rlweight[i,j], lrweight[i,j]), dtype=DTPOL)

            idx += 1

    datastruct_out = Datastruct(datastruct.obs_info, tarr, datatable, dtype="EHTIM")

    #TODO get flags from uvfits?
    return datastruct_out