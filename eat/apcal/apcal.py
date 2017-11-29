'''
#This module allows to read in ANTAB files
#match them with VEX schedules
#and generate output SEFD files in a folder structure
#single SEFD file columns are:
#0) fractional MJD, 
#1) square root of SEFD R-polarization,
#2) zeros
#3) square root of SEFD L-polarization, 
#4) zeros
#Maciek Wielgus, Nov/29/2017, maciek.wielgus@gmail.com

#INSTRUCTION FOR USING APCAL MODULE
import apcal as ap
#provide path to ANTAB calibration files
antab_path = 'ANTABS/'
#provide path to VEX scans information
vex_path = 'VexFiles/'
#for which sources, antenas, nights we want to generate calibration
sourL = ['OJ287','3C279']
antL = ['S','J','P','Z','X','L','R','A']
exptL = [3597,3598,3599,3600,3601]
#run the SEFD files generator
ap.get_sefds(antab_path,vex_path,sourL,antL,exptL)

'''

import pandas as pd
import numpy as np
import os, datetime
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import EarthLocation, AltAz, ICRS, Angle
from astropy.time import Time, TimeDelta
import vex as vex

AZ2Z = {'AZ': 'Z', 'PV': 'P', 'SM':'S', 'SR':'R','JC':'J', 'AA':'A','AP':'X', 'LM':'L'}
SMT2Z = {'APEX': 'X', 'JCMT': 'J', 'LMT':'L', 'SMR':'R', 'SMA':'S', 'SMT':'Z', 'PV':'P'}
Z2AZ = {'Z':'AZ', 'P':'PV', 'S':'SM', 'R':'SR','J':'JC', 'A':'AA','X':'AP', 'L':'LM'}
track2expt ={'D':3597,'B':3598, 'C':3599,'A':3600,'E':3601}
expt2track ={3597:'D',3598:'B', 3599:'C',3600:'A',3601:'E'}
nam2lett = {'ALMA':'A','APEX':'X','LMT':'L','PICOVEL':'P','SMTO':'Z','SPT':'T','JCMT':'J','SMAP':'S'}
ant_locat ={
    'A': [2225061.16360, -5440057.36994, -2481681.15054],
    'X': [2225039.52970, -5441197.62920, -2479303.35970],
    'P': [5088967.74544, -301681.18586, 3825012.20561],
    'T': [0.01000, 0.01000, -6359609.70000],
    'L': [-768715.63200, -5988507.07200, 2063354.85200],
    'Z': [-1828796.20000, -5054406.80000, 3427865.20000],
    'J': [-5464584.67600, -2493001.17000, 2150653.98200],
    'S': [-5464555.49300, -2492927.98900, 2150797.17600],
    'R': [-5464555.49300, -2492927.98900, 2150797.17600]
}

sourL = ['OJ287','3C279']
antL0 = ['S','J','P','Z','X','L','R','A']
exptL0 = [3597,3598,3599,3600,3601]

def compute_elev(ra_source, dec_source, xyz_antenna, time):
    #this one is by Michael Janssen
   """
   given right ascension and declination of a sky source [ICRS: ra->(deg,arcmin,arcsec) and dec->(hour,min,sec)]
   and given the position of the telescope from the vex file [Geocentric coordinates (m)]
   and the time of the observation (e.g. '2012-7-13 23:00:00') [UTC:yr-m-d],
   returns the elevation of the telescope.
   Note that every parameter can be an array (e.g. the time)
   """
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


def list_of_scans(alist, deltat_sec = 0):
    #prepares dataframe of scans to propagate information about time of scans and elevation into calibration
    columns=['expt_no', 'scan_id', 'time_min','time_max','source','elevZ','elevP','antenas']
    scans = pd.DataFrame(columns=columns)
    a0 = alist
    for ext in list(set(a0.expt_no)):
        for scid in list(set(a0.scan_id)):
            cond= (a0['expt_no']==ext)&(a0['scan_id']==scid)
            if len(a0.loc[cond])>0:
                t1,t2 = time_scans(a0,ext,scid, deltat_sec = 0)
                source = list(a0.loc[cond,'source'])[0]
                condZ1=map(lambda x: x[0]=='Z', a0['baseline'])
                condZ2=map(lambda x: x[1]=='Z', a0['baseline'])
                if len(a0.loc[cond&condZ1])>0:
                    elevZ = list(a0.loc[cond&condZ1,'ref_elev'])[0]
                elif len(a0.loc[cond&condZ2])>0:
                    elevZ = list(a0.loc[cond&condZ2,'rem_elev'])[0]
                else:
                    elevZ = np.nan
                condP1=map(lambda x: x[0]=='P', a0['baseline'])
                condP2=map(lambda x: x[1]=='P', a0['baseline'])
                if len(a0.loc[cond&condP1])>0:
                    elevP = list(a0.loc[cond&condP1,'ref_elev'])[0]
                elif len(a0.loc[cond&condP2])>0:
                    elevP = list(a0.loc[cond&condP2,'rem_elev'])[0]
                else:
                    elevP = np.nan
                antenas = set(list(''.join(list(set(list(a0.loc[cond].baseline))))))
                s1 = pd.DataFrame.from_records([(ext,scid,t1,t2,source,elevZ,elevP, antenas)],columns=columns)
                scans = pd.concat([scans,s1],ignore_index=True)
    return scans


def dict_DPFU_GFIT(filepath):
    #loading DPFU and GFIT in for from a single file (one track, many antenas)
    myfile = open(filepath, 'r')
    cou=0
    foo = myfile.readline()
    DPFU = []
    Gcoef = []
    ant = []
    track = []
    while ((len(foo)>5)):    
        listFoo = foo.split(' ')
        
        if listFoo[5][-1]!=',':
            DPFU.append(listFoo[5])
            Gcoef.append(listFoo[8:-1])
        else:
            DPFU.append(listFoo[5][:-1])
            Gcoef.append(listFoo[9:-1])

        ant.append(AZ2Z[listFoo[1]])
        track.append(filepath[-4])
        foo = myfile.readline()  
    
    for cou in range(len(DPFU)):
        DPFU[cou]=float(DPFU[cou])
        for cou2 in range(len(Gcoef[cou])):
            if (Gcoef[cou][cou2][-1] == ','):
                Gcoef[cou][cou2]=float(Gcoef[cou][cou2][:-1])
            else:
                Gcoef[cou][cou2]=float(Gcoef[cou][cou2])
    
    dict_dpfu = dict(zip(zip(ant,track),DPFU))
    dict_gfit = dict(zip(zip(ant,track),Gcoef))

    return dict_dpfu, dict_gfit, track[0]

def merge_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def prepare_calibration_data(folder_path):
    dict_dpfu = {}; dict_gfit = {}; Tsys={};
    list_files = os.listdir(folder_path)
    list_files = [f for f in list_files if f[0] =='e']
    cols = ['datetime','Tsys_st_R_lo','Tsys_st_L_lo','Tsys_st_R_hi','Tsys_st_L_hi']
    antena='NoAntena'
    FooDF = pd.DataFrame(columns=cols)

    for f in list_files:
        fpath = folder_path+f
        dict_dpfu_loc, dict_gfit_loc, track_loc = dict_DPFU_GFIT(fpath)
        dict_dpfu = merge_dicts(dict_dpfu,dict_dpfu_loc)
        dict_gfit = merge_dicts(dict_gfit,dict_gfit_loc)
    
        with open(fpath) as f:
            content = f.readlines()
        for line in content:
            #first line of data for given antenna
            if 'TSYS ' in line:
                antena = AZ2Z[line[5:7]]
                print(fpath+', '+antena)
                if line.split(' ')[2]=='timeoff=':
                    timeoff = int(float(line.split(' ')[3]))
                else:
                    timeoff = 0
                timeoff = datetime.timedelta(seconds = timeoff)
                FooDF = pd.DataFrame(columns=cols)
            
            #data rows are the ones strarting with sth that can be converted to flow
            if isfloat(line.split(' ')[0]):
                foo = line.split(' ')
                foo = [x for x in foo if len(x)>0]
                if antena=='A':
                    datetime_loc = time2datetime1(foo[0],'00:00:00')
                    datetime_loc = datetime_loc + ALMAtime2STANDARDtime(foo[1]) + timeoff
                else:
                    datetime_loc = time2datetime1(foo[0],foo[1]) + timeoff
                
                Tsys_R_lo_loc = float(foo[2])
                if antena in {'S', 'A'}:
                    Tsys_L_lo_loc = float(foo[3])
                    Tsys_R_hi_loc = float(foo[4])
                    Tsys_L_hi_loc = float(foo[5])   
                else:
                    try:
                        Tsys_L_lo_loc = float(foo[3])
                    except IndexError:
                        Tsys_L_lo_loc = Tsys_R_lo_loc           
                    Tsys_R_hi_loc = Tsys_R_lo_loc
                    Tsys_L_hi_loc = Tsys_L_lo_loc       
                data_loc = [datetime_loc,Tsys_R_lo_loc,Tsys_L_lo_loc,Tsys_R_hi_loc,Tsys_L_hi_loc]
                line_df = pd.DataFrame([data_loc], columns = cols)
                
                FooDF = FooDF.append(line_df, ignore_index=True)
            
            #lines at the end of data for given antena
            if line[0]=='/':
                if (antena,track_loc) in Tsys:
                    Tsys[(antena,track_loc)]=pd.concat([Tsys[(antena,track_loc)],FooDF ])
                else:
                    Tsys[(antena,track_loc)] = FooDF
    for key in Tsys:
        foo = Time(list(Tsys[key].datetime)).mjd
        Tsys[key]['mjd'] = foo

    return dict_dpfu, dict_gfit, Tsys

def time_scans(a0,expt_no,scan_id, deltat_sec = 0):
    #given a scan finds data with smallest and largest time
    cond= (a0['expt_no']==expt_no)&(a0['scan_id']==scan_id)
    if len(a0.loc[cond])>0:
        tmin = min(a0.loc[(a0['expt_no']==expt_no)&(a0['scan_id']==scan_id),'datetime'])
        tmax = max(a0.loc[(a0['expt_no']==expt_no)&(a0['scan_id']==scan_id),'datetime'])
        tmin = tmin - datetime.timedelta(seconds = deltat_sec)
        tmax = tmax + datetime.timedelta(seconds = deltat_sec)
        return tmin, tmax
    else:
        return np.nan, np.nan

def time2datetime1(day,hour):
    #calculateslist of datetime stamps
    foo = hour.split(':')
    day = int(day)
    h = int(foo[0])%24; m = int(foo[1]); s = int(foo[2])
    datet = (datetime.datetime(2017, 1,1,h,m,s) + datetime.timedelta(days=day-1))    
    return datet

def make_single_Tsys_table(Tsys):
    list_Tsys = []
    for key in Tsys:
        Tsys[key]['antena'] = [key[0]]*Tsys[key].shape[0]
        Tsys[key]['track'] = [key[1]]*Tsys[key].shape[0]
        list_Tsys.append(Tsys[key])

    Tsys_full =  pd.concat(list_Tsys,ignore_index=True)
    Tsys_full = Tsys_full.drop_duplicates()
    Tsys_full['expt_no'] =  map(lambda x: track2expt[x],Tsys_full['track'])
    return Tsys_full


def ALMAtime2STANDARDtime(atime):
    h = int(atime.split(':')[0])
    m = int(atime.split(':')[1].split('.')[0])
    s = round(60*(float(atime.split(':')[1].split('.')[1])/100))
    dt = datetime.timedelta(hours = h, minutes=m,seconds = s)
    return dt

def prepare_data_for_sefd(alist, path_antab ='ANTABS/'):

    #information about scans
    scans = list_of_scans(alist)

    #get dpfu, gain fits, raw system temperatures star
    dict_dpfu, dict_gfit, Tsys = prepare_calibration_data(path_antab)

    #add mjd information
    for key in Tsys:
        foo = Time(list(Tsys[key].datetime)).mjd
        Tsys[key]['mjd'] = foo
    Tsys_full = make_single_Tsys_table(Tsys)
    Tsys_full = Tsys_full.drop_duplicates()

    scans = scans.sort_values('time_max')
    scans = scans.reset_index(drop=True)
    scans['gainP'] = [1.]*scans.shape[0]
    scans['gainZ'] = [1.]*scans.shape[0]

    #calculating elevation gains for Pv and Az
    gainPf = lambda x: dict_gfit[('P','A')][0] + dict_gfit[('P','A')][1]*x + dict_gfit[('P','A')][2]*x**2
    gainZf = lambda x: dict_gfit[('Z','A')][0] + dict_gfit[('Z','A')][1]*x + dict_gfit[('Z','A')][2]*x**2
    for index, row in scans.iterrows():
        if 'P' in row.antenas:
            foo = gainPf(scans['elevP'][index])
            scans.gainP[index] = float(foo)
        if 'Z' in row.antenas:
            foo = gainZf(scans['elevZ'][index])
            scans.gainZ[index] = float(foo) 

    bins_labels = range((scans.shape[0]))
    DictSource = dict(zip(bins_labels, list(scans.source)))
    DictGainP = dict(zip(bins_labels, list(scans.gainP)))
    DictGainZ = dict(zip(bins_labels, list(scans.gainZ)))
    bins_labels = range((scans.shape[0]))
    binsT = [scans.time_min[0]]+list(scans.time_max)
    ordered_labels = pd.cut(Tsys_full.datetime, binsT,labels = bins_labels)
    Tsys_full['BinScan'] = ordered_labels
    Tsys_full = Tsys_full[Tsys_full.BinScan == Tsys_full.BinScan]
    Tsys_full['source'] = map(lambda x: DictSource[x], list(Tsys_full['BinScan']))
    Tsys_full['gainP'] = map(lambda x: DictGainP[x], list(Tsys_full['BinScan']))
    Tsys_full['gainZ'] = map(lambda x: DictGainZ[x], list(Tsys_full['BinScan']))
    return Tsys_full, dict_dpfu, dict_gfit

def get_exact_elevation(Tsys, dict_gfit, fpath =  'VexFiles/'):

    dict_ra, dict_dec = make_ra_dec_list(fpath)
    Tsys['gainZex'] = [1.]*Tsys.shape[0]
    Tsys['gainPex'] = [1.]*Tsys.shape[0]
    gainPf = lambda x: dict_gfit[('P','A')][0] + dict_gfit[('P','A')][1]*x + dict_gfit[('P','A')][2]*x**2
    gainZf = lambda x: dict_gfit[('Z','A')][0] + dict_gfit[('Z','A')][1]*x + dict_gfit[('Z','A')][2]*x**2
    
    for index, row in Tsys.iterrows():
        ant = Tsys.antena[index]
        sour = Tsys.source[index]
        strtime = str(Tsys.datetime[index])
        if ant=='P':
            #try:
            elevPloc = compute_elev(dict_ra[sour],dict_dec[sour],ant_locat['P'],strtime)
            foo = gainPf(elevPloc)
            print('p'+str(index)+'= ', foo)
            #except KeyError:
            #foo = 1.
            Tsys.gainPex[index] = float(foo)
        if ant=='Z':
            #try:
            elevZloc = compute_elev(dict_ra[sour],dict_dec[sour],ant_locat['Z'],strtime)
            foo = gainZf(elevZloc)
            print('z'+str(index)+'=', foo)
            #except KeyError:
            #foo = 1.
            Tsys.gainZex[index] = float(foo)
    return Tsys

def generate_and_save_sefd_data(Tsys_full, dict_dpfu, dict_gfit, sourL=sourL, antL=antL0, exptL=exptL0,pathSave = 'SEFDs'):
    if not os.path.exists(pathSave):
        os.makedirs(pathSave)
    dirHI = pathSave+'/SEFD_HI'
    dirLO = pathSave+'/SEFD_LO'

    if not os.path.exists(dirHI):
        os.makedirs(dirHI)
    if not os.path.exists(dirLO):
        os.makedirs(dirLO)

    for expt in exptL:
        dir_expt_HI = dirHI+'/'+str(expt)
        dir_expt_LO = dirLO+'/'+str(expt)
        if not os.path.exists(dir_expt_HI):
            os.makedirs(dir_expt_HI)
        if not os.path.exists(dir_expt_LO):
            os.makedirs(dir_expt_LO)
        
        for ant in antL:
            for sour in sourL:
                
                condS = (Tsys_full['source']==sour)
                condA = (Tsys_full['antena']==ant)
                condE = (Tsys_full['track']==expt2track[expt])
                Tsys_local = Tsys_full.loc[condS&condA&condE]
                Tsys_local.loc[:,'sefd_lo_L'] = np.sqrt(Tsys_local['Tsys_st_L_lo']/dict_dpfu[(ant,'A')])
                Tsys_local.loc[:,'sefd_lo_R'] = np.sqrt(Tsys_local['Tsys_st_R_lo']/dict_dpfu[(ant,'A')])
                Tsys_local.loc[:,'sefd_hi_L'] = np.sqrt(Tsys_local['Tsys_st_L_hi']/dict_dpfu[(ant,'A')])
                Tsys_local.loc[:,'sefd_hi_R'] = np.sqrt(Tsys_local['Tsys_st_R_hi']/dict_dpfu[(ant,'A')])        
                if ant=='P':
                    Tsys_local.loc[:,'sefd_lo_L'] = Tsys_local['sefd_lo_L']/np.sqrt(Tsys_local['gainP'])
                    Tsys_local.loc[:,'sefd_lo_R'] = Tsys_local['sefd_lo_R']/np.sqrt(Tsys_local['gainP'])
                    Tsys_local.loc[:,'sefd_hi_L'] = Tsys_local['sefd_hi_L']/np.sqrt(Tsys_local['gainP'])
                    Tsys_local.loc[:,'sefd_hi_R'] = Tsys_local['sefd_hi_R']/np.sqrt(Tsys_local['gainP'])                  
                elif ant=='Z':
                    Tsys_local.loc[:,'sefd_lo_L'] = Tsys_local['sefd_lo_L']/np.sqrt(Tsys_local['gainZ'])
                    Tsys_local.loc[:,'sefd_lo_R'] = Tsys_local['sefd_lo_R']/np.sqrt(Tsys_local['gainZ'])
                    Tsys_local.loc[:,'sefd_hi_L'] = Tsys_local['sefd_hi_L']/np.sqrt(Tsys_local['gainZ'])
                    Tsys_local.loc[:,'sefd_hi_R'] = Tsys_local['sefd_hi_R']/np.sqrt(Tsys_local['gainZ'])      
                try:
                    Tsys_local.loc[:,'foo_Imag_1'] = 0.*Tsys_local['sefd_hi_R']
                    Tsys_local.loc[:,'foo_Imag_2'] = 0.*Tsys_local['sefd_hi_R']
                    SEFDS_lo = Tsys_local.loc[:,['mjd','sefd_lo_R','foo_Imag_1','sefd_lo_L','foo_Imag_2']]
                    SEFDS_hi = Tsys_local.loc[:,['mjd','sefd_hi_R','foo_Imag_1','sefd_hi_L','foo_Imag_2']]
                    SEFDS_lo = SEFDS_lo.sort_values('mjd')
                    SEFDS_hi = SEFDS_hi.sort_values('mjd')
                    NameF_lo = dir_expt_LO+'/'+sour+'_'+Z2AZ[ant]+'.txt'
                    NameF_hi = dir_expt_HI+'/'+sour+'_'+Z2AZ[ant]+'.txt'
                    if SEFDS_lo.shape[0]>2:
                        SEFDS_lo.to_csv(NameF_lo,sep=' ',index=False, header=False)
                    if SEFDS_hi.shape[0]>2:
                        SEFDS_hi.to_csv(NameF_hi, sep=' ', index=False, header = False)
                    print(sour+'_'+Z2AZ[ant]+' ok')
                except ValueError:
                    print(sour+'_'+Z2AZ[ant]+'crap, not ok')


def make_ra_dec_list(fpath):

    list_files = os.listdir(fpath)
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

    return dict_ra, dict_dec

def make_scan_list(fpath,dict_gfit):

    list_files = os.listdir(fpath)
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
            elevPloc = compute_elev(dict_ra[sour[x]],dict_dec[sour[x]],ant_locat['P'],datet[x]+TimeDelta(100., format='sec'))
            elevZloc = compute_elev(dict_ra[sour[x]],dict_dec[sour[x]],ant_locat['Z'],datet[x]+TimeDelta(100., format='sec'))
            elevloc ={'P':elevPloc, 'Z':elevZloc}
            elev.append(elevloc)
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
        foo['elev']=elev
        foo['scan_no'] = foo.index
        foo['scan_no'] = map(int,foo['scan_no'])
        foo['track'] = [track_loc]*foo.shape[0]
        foo['expt'] = [int(track2expt[track_loc])]*foo.shape[0]
        foo['antenas'] = antenas
        foo['duration'] = duration

        scans = pd.concat([scans,foo], ignore_index=True)

    scans = scans.reindex_axis(['mjd_floor','expt','track','scan_no','source','time_min','time_max','duration','elev','antenas'],axis=1)
    scans = scans.sort_values('time_max')
    scans = scans.reset_index(drop=True)
    scans['scan_no_tot'] = scans.index

    scanelevP = [scans.elev[x]['P'] for x in range(scans.shape[0])]
    scanelevZ = [scans.elev[x]['Z'] for x in range(scans.shape[0])]
    scans['gainP'] = [1.]*scans.shape[0]
    scans['gainZ'] = [1.]*scans.shape[0]
    gainPf = lambda x: dict_gfit[('P','A')][0] + dict_gfit[('P','A')][1]*x + dict_gfit[('P','A')][2]*x**2
    gainZf = lambda x: dict_gfit[('Z','A')][0] + dict_gfit[('Z','A')][1]*x + dict_gfit[('Z','A')][2]*x**2
    for index, row in scans.iterrows():
        if 'P' in row.antenas:
            foo = gainPf(scans.elev[index]['P'])
            scans.gainP[index] = float(foo)
        if 'Z' in row.antenas:
            foo = gainZf(scans.elev[index]['Z'])
            scans.gainZ[index] = float(foo)

    return scans

def match_scans_Tsys(scans,Tsys):
    
    bins_labels = [None]*(2*scans.shape[0]-1)
    bins_labels[1::2] = map(lambda x: -x-1,list(scans['scan_no_tot'])[:-1])
    bins_labels[::2] = list(scans['scan_no_tot'])
    dtmin = datetime.timedelta(seconds = 0.) 
    dtmax = datetime.timedelta(seconds = 0.) 
    binsT = [None]*(2*scans.shape[0])
    binsT[::2] = list(map(lambda x: x - dtmin,list(scans.time_min)))
    binsT[1::2] = list(map(lambda x: x + dtmax,list(scans.time_max))) 
    ordered_labels = pd.cut(Tsys.datetime, binsT,labels = bins_labels)
    
    DictSource = dict(zip(list(scans.scan_no_tot), list(scans.source)))
    DictGainP = dict(zip(list(scans.scan_no_tot), list(scans.gainP)))
    DictGainZ = dict(zip(list(scans.scan_no_tot), list(scans.gainZ)))
    Tsys['scan_no_tot'] = ordered_labels
    Tsys = Tsys[map(lambda x: x >= 0, Tsys['scan_no_tot'])]
    Tsys.loc[:,'source'] = map(lambda x: DictSource[x], Tsys['scan_no_tot'])
    Tsys.loc[:,'gainP'] = map(lambda x: DictGainP[x], Tsys['scan_no_tot'])
    Tsys.loc[:,'gainZ'] = map(lambda x: DictGainZ[x], Tsys['scan_no_tot'])
    Tsys = Tsys.sort_values('datetime').reset_index(drop=True)
    
    return Tsys


def global_match_scans_Tsys(scans,Tsys_full):
    Tsys_match = pd.DataFrame({'source' : []})

    for ant in antL0:
        for expt in exptL0:

            condA = (Tsys_full['antena']==ant)
            condE = (Tsys_full['expt_no']==expt)
            Tsys_loc = Tsys_full.loc[condA&condE].sort_values('datetime').reset_index(drop=True)
            scans_loc = scans[(scans.expt == expt)].sort_values('time_min').reset_index(drop=True)
            Tsys_foo = match_scans_Tsys(scans_loc,Tsys_loc)
            Tsys_match = pd.concat([Tsys_match,Tsys_foo], ignore_index=True)

    Tsys_match = Tsys_match.sort_values('datetime').reset_index(drop=True)

    return Tsys_match


def get_sefds(antab_path ='ANTABS/', vex_path = 'VexFiles/', sourL=sourL,antL=antL0, exptL = exptL0):

    print('Getting the calibration data...')
    #TABLE of CALIBRATION DATA from ANTAB files
    dict_dpfu, dict_gfit, Tsys = prepare_calibration_data(antab_path)
    Tsys_full = make_single_Tsys_table(Tsys)

    print('Getting the scans data...')
    #TABLE of SCANS from VEX files, using elevation gain info
    scans = make_scan_list(vex_path,dict_gfit)

    print('Matching calibration to scans...')
    #MATCH CALIBRATION with SCANS to determine the source and 
    Tsys_match = global_match_scans_Tsys(scans,Tsys_full)

    print('Saving sefd files...')
    #produce a priori calibration data
    generate_and_save_sefd_data(Tsys_match, dict_dpfu, dict_gfit, sourL, antL, exptL)


