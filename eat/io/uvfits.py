'''
Imports uvfits using ehtim library
and remodels them into pandas dataframes
Maciek Wielgus 06/07/2018
maciek.wielgus@gmail.com
'''
from __future__ import print_function
from __future__ import division
import pandas as pd
import os,sys,glob
import numpy as np
from astropy.time import Time, TimeDelta
import datetime as datetime
from eat.io import vex
import logging

# Configure logging
loglevel = getattr(logging, 'INFO', None)
logging.basicConfig(level=loglevel,
                    format='%(asctime)s %(levelname)s:: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

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
        stations_2lett_1lett = {'MG':'Z', 'GL':'G', 'PV':'P', 'SW':'S', 'MM':'J', 'AA':'A', 'AX':'X', 'LM':'L','SZ':'Y','NN':'N','KT':'K'}
        jd_expt = jd2expt2021
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
        vpath = fpath+fi
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
        vpath = fpath+fi
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
        vpath = fpath+fi
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
    logging.info("get_info worked, got stations_2lett_1lett and jd_2_expt")
    
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
