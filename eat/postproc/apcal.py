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
import re
import pandas as pd
import numpy as np
import os, datetime
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import EarthLocation, AltAz, ICRS, Angle
from astropy.time import Time, TimeDelta
from eat.io import vex
import scipy.interpolate as si
from numpy.polynomial import Polynomial

'''AZ2Zold = {'AZ': 'Z', 'PV': 'P', 'SM':'S', 'SR':'R','JC':'J', 'AA':'A','AP':'X', 'LM':'L','SP':'Y'}
AZ2Z = {'MG': 'Z', 'PV': 'P', 'SW':'S','MM':'J', 'AA':'A','AX':'X', 'LM':'L','SZ':'Y','GL':'G','NN':'N','KT':'K'}
SMT2Zold = {'ALMA': 'A', 'APEX': 'X', 'JCMT': 'J', 'LMT':'L', 'SMR':'R', 'SMA':'S', 'SMT':'Z', 'PV':'P','SPT':'Y'}
SMT2Z = {'ALMA': 'A', 'APEX': 'X', 'JCMT': 'J', 'LMT':'L', 'SMR':'R', 'SMA':'S', 'SMT':'Z', 'PV':'P','SPT':'Y', 'GLT':'G', 'NOEMA':'N','KT':'K'}
Z2SMT = {v: k for k, v in SMT2Z.items()}
Z2AZ = {'Z':'MG', 'P':'PV', 'S':'SW', 'R':'SR','J':'MM', 'A':'AA','X':'AX', 'L':'LM','Y':'SZ','G':'GL','N':'NN','K':'KT'}
#track2expt ={'D':3597,'B':3598, 'C':3599,'A':3600,'E':3601}
#expt2track ={3597:'D',3598:'B', 3599:'C',3600:'A',3601:'E'}
track2expt ={'e18c21':3644,
             'e18e22':3645,
             'e18a24':3646,
             'e18c25':3647,
             'e18g27':3648,
             'e18d28':3649,
             'e21b09':3762,
             'e21e13':3763,
             'e21a14':3764,
             'e21d15':3765,
             'e21a16':3766,
             'e21a17':3767,
             'e21e18':3768,
             'e21f19':3769}
expt2track ={v: k for k, v in track2expt.items()}
nam2lett = {'ALMA':'A','APEX':'X','LMT':'L','PICOVEL':'P','SMTO':'Z','SPT':'Y','JCMT':'J','SMAP':'S','THULE':'G','NOEMA':'N','KITTPEAK':'K'}
ant_locat ={
    'A': [2225061.16360, -5440057.36994, -2481681.15054],
    'X': [2225039.52970, -5441197.62920, -2479303.35970],
    'P': [5088967.74544, -301681.18586, 3825012.20561],
    'T': [0.01000, 0.01000, -6359609.70000],
    'L': [-768715.63200, -5988507.07200, 2063354.85200],
    'Z': [-1828796.20000, -5054406.80000, 3427865.20000],
    'J': [-5464584.67600, -2493001.17000, 2150653.98200],
    'S': [-5464555.49300, -2492927.98900, 2150797.17600],
    'R': [-5464555.49300, -2492927.98900, 2150797.17600],
    'Y': [0.01, 0.01, -6359609.7],
    'G': [541547.00000, -1387978.60000, 6180982.00000],
    'N': [4524000.43000, -468042.14000, 4460309.76000],
    'K': [-1995957.62800, -5037394.44600, 3357047.27400]
}
dic_band={'1':'b1','2':'b2','3':'b3','4':'b4'}
sourL = ['OJ287','3C279']
sourFull = ['OJ287', '1055+018', 'M87', '3C279', 'SGRA', 'J1733-1304',
       'J1924-2914', 'CYGX-3', '3C273', 'J0132-1654', 'NGC1052',
       'J0006-0623', 'BLLAC', 'CTA102', '3C454.3', '3C84', 'CENA',
       '1749+096', '1921-293']
antL0 = ['S','J','P','Z','X','L','R','A','Y','G','N','K']
exptL0 = [3597,3598,3599,3600,3601]'''

AZ2Z = {}
SMT2Z = {}
track2expt = {}
ant_locat = {}
sourL = []
antL0 = []
exptL0 = []
bandL0 = ['b1', 'b2', 'b3', 'b4']

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
        if 'sideband' in filepath:
            track.append(filepath[-14])
        else:
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

def dict_DPFU_GFIT_sb(filepath):
    '''
    new extraction of dpfu and gfit from sideband files
    we just apply this to a single file
    '''
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
        if 'sideband' in filepath:
            track.append(filepath[-14])
        else:
            track.append(filepath[-4])
        foo = myfile.readline()  
    
    for cou in range(len(DPFU)):
        DPFU[cou]=float(DPFU[cou])
        for cou2 in range(len(Gcoef[cou])):
            if (Gcoef[cou][cou2][-1] == ','):
                Gcoef[cou][cou2]=float(Gcoef[cou][cou2][:-1])
            else:
                Gcoef[cou][cou2]=float(Gcoef[cou][cou2])
    
    #dict_dpfu = dict(zip(zip(ant,track),DPFU))
    #dict_gfit = dict(zip(zip(ant,track),Gcoef))
    dict_dpfu = dict(zip(ant,DPFU))
    dict_gfit = dict(zip(ant,Gcoef))

    return dict_dpfu, dict_gfit, track[0]


def dict_DPFU_GFIT_sb4(filepath):
    '''
    version with dictionaries in a form (station, track, band, polarization)
    '''
    #loading DPFU and GFIT in for from a single file (one track, many antenas)
    myfile = open(filepath, 'r')
    cou=0
    foo = myfile.readline()
    DPFU = []
    Gcoef = []
    ant = []
    trackLoc = filepath[-14]
    bandLoc = dic_band[filepath[-4]]
    polar = []
    while ((len(foo)>5)):    
        listFoo = foo.split(' ')
        
        if listFoo[5][-1]!=',':
            DPFU.append(listFoo[5])
            Gcoef.append(listFoo[8:-1])
            polar.append('R')
            DPFU.append(listFoo[5])
            Gcoef.append(listFoo[8:-1])
            polar.append('L')
        else:
            DPFU.append(listFoo[5][:-1])
            Gcoef.append(listFoo[9:-1])
            polar.append('R')
            DPFU.append(listFoo[6])
            Gcoef.append(listFoo[9:-1])
            polar.append('L')

        # 2x for two polarizations
        ant.append(AZ2Z[listFoo[1]])
        ant.append(AZ2Z[listFoo[1]])
        foo = myfile.readline()  
    
    for cou in range(len(DPFU)):
        DPFU[cou]=float(DPFU[cou])
        for cou2 in range(len(Gcoef[cou])):
            if (Gcoef[cou][cou2][-1] == ','):
                Gcoef[cou][cou2]=float(Gcoef[cou][cou2][:-1])
            else:
                Gcoef[cou][cou2]=float(Gcoef[cou][cou2])
    
    
    track = [trackLoc]*len(ant)
    band = [bandLoc]*len(ant)
    #print(len(ant),len(track), len(band), len(DPFU),len(Gcoef))


    dict_dpfu = dict(zip(zip(ant,track,band,polar),DPFU))
    dict_gfit = dict(zip(zip(ant,track,band,polar),Gcoef))

    return dict_dpfu, dict_gfit


def dict_DPFU_GFIT_2018(filepath, AZ2Z):
    '''
    version for ANTAB formats from 2018 onwards
    '''
    #loading DPFU and GFIT in for from a single file (one track, many antenas)
    myfile = open(filepath, 'r')
    cou=0
    foo = myfile.readline()
    DPFU = []
    Gcoef = []
    ant = []
    trackLoc = filepath[-17:-11]
    bandLoc = filepath[-10:-8]
    
    polar = []
    while ((len(foo)>5)):    
        listFoo = foo.split(' ')
        
        if listFoo[5][-1]!=',':
            DPFU.append(listFoo[5])
            Gcoef.append(listFoo[8:-1])
            polar.append('R')
            DPFU.append(listFoo[5])
            Gcoef.append(listFoo[8:-1])
            polar.append('L')
        else:
            DPFU.append(listFoo[5][:-1])
            Gcoef.append(listFoo[9:-1])
            polar.append('R')
            DPFU.append(listFoo[6])
            Gcoef.append(listFoo[9:-1])
            polar.append('L')

        # 2x for two polarizations
        ant.append(AZ2Z[listFoo[1]])
        ant.append(AZ2Z[listFoo[1]])
        foo = myfile.readline()  
    
    for cou in range(len(DPFU)):
        DPFU[cou]=float(DPFU[cou])
        for cou2 in range(len(Gcoef[cou])):
            if (Gcoef[cou][cou2][-1] == ','):
                Gcoef[cou][cou2]=float(Gcoef[cou][cou2][:-1])
            else:
                Gcoef[cou][cou2]=float(Gcoef[cou][cou2])
    
    
    track = [trackLoc]*len(ant)
    band = [bandLoc]*len(ant)
    #print(len(ant),len(track), len(band), len(DPFU),len(Gcoef))
    

    dict_dpfu = dict(zip(zip(ant,track,band,polar),DPFU))
    dict_gfit = dict(zip(zip(ant,track,band,polar),Gcoef))
    return dict_dpfu, dict_gfit


def extract_dpfu_gfit_from_antab(filename, az2z):
    """
    Parses a file to extract DPFU and GFIT values for antennas from ANTAB files for a single band and returns them as dictionaries.

    Parameters
    ----------
    filename : str
        The ANTAB filename with full path.
    az2z : dict
        A dictionary mapping 2-letter station codes to their respective 1-letter codes.

    Returns
    -------
    tuple
        A tuple containing two dictionaries:
        - dpfudict (dict): A dictionary with tuples of (antenna, track, band, polarization) as keys and DPFU values as values.
        - gfitdict (dict): A dictionary with tuples of (antenna, track, band, polarization) as keys and GFIT coefficients as values.

    Notes
    -----
    The ANTAB file is expected to contain DPFU and POLY (gain coefficients) for multiple antennas and polarizations.
    The values for each station are contained in lines starting with 'GAIN' followed by the strings 'DPFU' and 'POLY' with corresponding values.
    The function reads the file line by line, processes the data, and converts it into the required format.
    """
    track, band = os.path.basename(filename).split('_')[:2]
    pol = ['R', 'L']    # Polarizations

    # Regular expressions to match DPFU and POLY values
    dpfu_pattern = re.compile(r'DPFU\s*=\s*([\d\.\-eE]+(?:,\s*[\d\.\-eE]+)*)')
    poly_pattern = re.compile(r'POLY\s*=\s*([\d\.\-eE]+(?:,\s*[\d\.\-eE]+)*)')

    dict_dpfu = {}
    dict_gfit = {}

    # Open the file and process only those lines that start with "GAIN"
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith("GAIN"):
                ant = az2z[line.split(' ')[1]]
                dpfu_match = dpfu_pattern.search(line)
                poly_match = poly_pattern.search(line)
                if dpfu_match and poly_match:
                    dpfu = [float(value.strip()) for value in dpfu_match.group(1).split(',')]
                    poly = [float(value.strip()) for value in poly_match.group(1).split(',')]

                    # Add the extracted values to the dictionary
                    dict_dpfu[(ant, track, band, pol[0])] = dpfu[0]
                    dict_dpfu[(ant, track, band, pol[1])] = dpfu[-1]
                    dict_gfit[(ant, track, band, pol[0])] = poly
                    dict_gfit[(ant, track, band, pol[1])] = poly

    return dict_dpfu, dict_gfit

def merge_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z

def extract_dpfu_gfit_from_all_antab(folder_path, AZ2Z=AZ2Z, bandL=bandL0):
    """
    Reads ANTAB format files in a specified folder and returns dictionaries containing DPFU and GFIT (gain coefficient) values.

    Parameters
    ----------
    folder_path : str
        The path to the folder containing ANTAB format files.
    AZ2Z : dict
        A dictionary mapping 2-letter station codes to 1-letter station codes.
    bandL : list
        A list of bands for which to generate the outputs.

    Returns
    -------
    dict
        A dictionary containing DPFU values from all ANTAB files.
    dict
        A dictionary containing GFIT values from all ANTAB files.
    """
    dict_dpfu = {}; dict_gfit = {}
    list_files = [f for f in os.listdir(folder_path) if f[0] == 'e' and any(f'_{band}_' in f for band in bandL)]

    for f in list_files:
        fpath = os.path.join(folder_path, f)
        dict_dpfu_loc, dict_gfit_loc = extract_dpfu_gfit_from_antab(fpath, AZ2Z)
        dict_dpfu = {**dict_dpfu, **dict_dpfu_loc}
        dict_gfit = {**dict_gfit, **dict_gfit_loc}
    
    return dict_dpfu, dict_gfit


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

#def get_dicts(folder_path,):

def prepare_calibration_data(folder_path):
    dict_dpfu = {}; dict_gfit = {}; Tsys={};
    list_files = os.listdir(folder_path)
    list_files = [f for f in list_files if f[0] =='e']
    cols = ['datetime','Tsys_st_R_lo','Tsys_st_L_lo','Tsys_st_R_hi','Tsys_st_L_hi']
    antena='NoAntena'
    FooDF = pd.DataFrame(columns=cols)
    RCP_first=True
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

            if 'INDEX'in line:
                RCP_first = line.find('R1')<line.find('L1')
                print('Fixing rder RCP - LCP: ', RCP_first)
            
                
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

                if RCP_first==True:
                    data_loc = [datetime_loc,Tsys_R_lo_loc,Tsys_L_lo_loc,Tsys_R_hi_loc,Tsys_L_hi_loc]
                else:
                    data_loc = [datetime_loc,Tsys_L_lo_loc,Tsys_R_lo_loc,Tsys_L_hi_loc,Tsys_R_hi_loc]
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


#def read_ALMA_Tsys():


def prepare_Tsys_data(folder_path, AZ2Z=AZ2Z, track2expt=track2expt, bandL=bandL0):
    from astropy.time import Time, TimeDelta
    Tsys={}
    #list_files = os.listdir(folder_path)
    #list_files = [f for f in list_files if f[0] =='e']
    list_files = [f for f in os.listdir(folder_path) if f[0] == 'e' and any(f'_{band}_' in f for band in bandL)]
    print(f"list_files: {list_files}")
    cols = ['datetime','Tsys_st_R','Tsys_st_L','band']
    antena='NoAntena'
    FooDF = pd.DataFrame(columns=cols)
    #print(list_files)
    for f in list_files:
        #print('guu', f)
        fpath = os.path.join(folder_path, f)
        year = '20' + fpath[-16:-14]
        #print(f)
        with open(fpath) as f:
            content = f.readlines()
        for line in content:
            #first line of data for given antenna
            if 'TSYS ' in line:
                #antena = AZ2Z[line[5:7]]
                antena = AZ2Z[line.split(' ')[1]]
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
              
                #get the timestamp
                if antena=='A':
                    datetime_loc = time2datetimeyear(year,foo[0],'00:00:00')
                    
                    datetime_loc = datetime_loc + ALMAtime2STANDARDtime(foo[1]) + timeoff
                    #print(datetime_loc)
                else:
                    datetime_loc = time2datetimeyear(year,foo[0],foo[1]) + timeoff
               
                #get the band
                band_loc = fpath[-10:-8]

                #now get the Tsys data
                
                #JCMT and SMAR have single pol, so we just fill both with same value
                if antena=='A':
                    TsysAA = np.asarray(list(map(float,foo[2:-1])))
                    TsysAA = TsysAA[(TsysAA!=0)&(TsysAA==TsysAA)]
                    if len(TsysAA) > 0:
                        Tsys_R_loc = (1./np.mean(1./np.sqrt(TsysAA)))**2
                        #Tsys_R_loc = np.mean(TsysAA)   
                    else:
                        Tsys_R_loc = np.nan
                    Tsys_L_loc = Tsys_R_loc
                    
                else:
                    Tsys_R_loc = float(foo[2])
                    try:
                        Tsys_L_loc = float(foo[3])
                    except IndexError:
                        Tsys_L_loc = Tsys_R_loc
                    
                data_loc = [datetime_loc,Tsys_R_loc,Tsys_L_loc,band_loc]
                
                if np.sum(np.isnan(data_loc[1:3]))==0:
                    line_df = pd.DataFrame([data_loc], columns = cols)
                    #FooDF = FooDF.append(line_df, ignore_index=True)
                    FooDF = pd.concat([FooDF, line_df], ignore_index=True)
                else:
                    continue
            
            #lines at the end of data for given antena
            #print(f.name)
        
            track_loc = fpath[-17:-11]

            if line[0]=='/':
                #print(f)
                #print((antena,track_loc))
                if (antena,track_loc) in Tsys:
                    Tsys[(antena,track_loc)]=pd.concat([Tsys[(antena,track_loc)],FooDF ])
                else:
                    Tsys[(antena,track_loc)] = FooDF
    #add mjd
    for key in Tsys:
        df = Tsys[key]
        if df.empty:
            print(f"Empty dataframe for {key}. Skipping...")
            continue
        foo = Time(list(Tsys[key].datetime)).mjd
        Tsys[key]['mjd'] = foo
    Tsfull = make_single_Tsys_table(Tsys, track2expt)
    return Tsfull


def prepare_Tsys_data_separate(folder_path):
    Tsys={}
    list_files = os.listdir(folder_path)
    list_files = [f for f in list_files if f[0] =='e']
    cols = ['datetime','Tsys_st_R','Tsys_st_L','band']
    antena='NoAntena'
    FooDF = pd.DataFrame(columns=cols)

    for f in list_files:
        fpath = os.path.join(folder_path, f) # INI: what is this?!
    
        with open(fpath) as f:
            content = f.readlines()
        for line in content:
            #first line of data for given antenna
            if 'TSYS ' in line:
                #antena = AZ2Z[line[5:7]]
                antena = AZ2Z[line.split(' ')[1]]
                print(fpath+', '+antena)
                if line.split(' ')[2]=='timeoff=':
                    timeoff = int(float(line.split(' ')[3]))
                else:
                    timeoff = 0
                timeoff = datetime.timedelta(seconds = timeoff)
                FooDF = pd.DataFrame(columns=cols)
            
            if 'INDEX'in line:
                RCP_first = line.find('R1')<line.find('L1')
                print('Order RCP - LCP: ', RCP_first)
            
            #data rows are the ones strarting with sth that can be converted to flow
            
            if isfloat(line.split(' ')[0]):
                foo = line.split(' ')
                foo = [x for x in foo if len(x)>0]
                #get the timestamp
                if antena=='A':
                    
                    datetime_loc = time2datetime1(foo[0],'00:00:00')
                    datetime_loc = datetime_loc + ALMAtime2STANDARDtime(foo[1]) + timeoff
                    #print(datetime_loc)
                else:
                    datetime_loc = time2datetime1(foo[0],foo[1]) + timeoff
                
                #get the band
                band_loc = dic_band[f.name[-4]]

                #now get the Tsys data
                
                #JCMT and SMAR have single pol, so we just fill both with same value
                if antena=='A':
                    TsysAA = np.asarray(list(map(float,foo[2:-1])))
                    if len(TsysAA)==0:
                        TsysAA = np.asarray(list(map(float,foo[2:])))
                    TsysAA = TsysAA[(TsysAA!=0)&(TsysAA==TsysAA)]
                    #print(TsysAA)
                    if len(TsysAA) > 0:
                        Tsys_R_loc = (1./np.mean(1./np.sqrt(TsysAA)))**2
                        #Tsys_R_loc = np.mean(TsysAA)   
                    else:
                        Tsys_R_loc = np.nan
                    Tsys_L_loc = Tsys_R_loc
                    
                else:
                    Tsys_R_loc = float(foo[2])
                    try:
                        Tsys_L_loc = float(foo[3])
                    except IndexError:
                        Tsys_L_loc = Tsys_R_loc
                
                if RCP_first==True:
                    data_loc = [datetime_loc,Tsys_R_loc,Tsys_L_loc,band_loc]
                else:
                    data_loc = [datetime_loc,Tsys_L_loc,Tsys_R_loc,band_loc] 
                ############################
                #FORCE L R order for APEX
                #if antena=='X':
                #    print('MANUALLY REVERSING APEX!!!')
                #    data_loc = [datetime_loc,Tsys_L_loc,Tsys_R_loc,band_loc] 
                ############################
                #data_loc = [datetime_loc,Tsys_R_loc,Tsys_L_loc,band_loc]
                if np.sum(np.isnan(data_loc[1:3]))==0:
                    line_df = pd.DataFrame([data_loc], columns = cols)
                    FooDF = FooDF.append(line_df, ignore_index=True)
                else:
                    continue
            
            #lines at the end of data for given antena
            #print(f.name)
        
            track_loc = f.name.split('_')[-2]
            if line[0]=='/':
                if (antena,track_loc) in Tsys:
                    Tsys[(antena,track_loc)]=pd.concat([Tsys[(antena,track_loc)],FooDF ])
                else:
                    Tsys[(antena,track_loc)] = FooDF
    return Tsys

def merge_all_Tsys(Tsys):

    for key in Tsys:
        print(key)
        print('Tsys shape: ',np.shape(Tsys[key]))
        foo = Time(list(Tsys[key].datetime)).mjd
        Tsys[key]['mjd'] = foo
    Tsfull = make_single_Tsys_table(Tsys)
    return Tsfull


def prepare_Tsys_data_ALMA(folder_path):
    Tsys={}
    list_files = os.listdir(folder_path)
    list_files = [f for f in list_files if f[0] =='e']

    Tsyscols = ['Tsys_ch'+str(x) for x in range(1,33)]
    cols = ['datetime']+Tsyscols+['band']
    antena='NoAntena'
    FooDF = pd.DataFrame(columns=cols)

    for f in list_files:
        fpath = os.path.join(folder_path, f) # INI: what is this?!
    
        with open(fpath) as f:
            content = f.readlines()
        for line in content:
            #first line of data for given antenna
            if 'TSYS ' in line:
                #antena = AZ2Z[line[5:7]]
                antena = AZ2Z[line.split(' ')[1]]
                if antena=='A':
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
                #get the timestamp
                if antena=='A':
                    datetime_loc = time2datetime1(foo[0],'00:00:00')
                    datetime_loc = datetime_loc + ALMAtime2STANDARDtime(foo[1]) + timeoff
                    #print(datetime_loc)
                else:
                    continue
                    #datetime_loc = time2datetime1(foo[0],foo[1]) + timeoff
                
                #get the band
                band_loc = dic_band[f.name[-4]]

                #now get the Tsys data
                
                #JCMT and SMAR have single pol, so we just fill both with same value
                if antena=='A':
                    if isfloat(foo[-1]):
                        TsysAA = list(map(float,foo[2:]))
                    else:
                        TsysAA = list(map(float,foo[2:-1]))
                    #if len(TsysAA)<32:
                    #    print(datetime_loc,len(TsysAA),isfloat(foo[-1]),len(foo))
                    if all(np.asarray(TsysAA)>0):
                        data_loc = [datetime_loc]+TsysAA+[band_loc]
                        line_df = pd.DataFrame([data_loc], columns = cols)
                        FooDF = FooDF.append(line_df, ignore_index=True)
                    else: continue
                else: continue
            
            #lines at the end of data for given antena
            #print(f.name)
        
            track_loc = f.name.split('_')[3]
            if line[0]=='/':
                if antena=='A':
                    if (antena,track_loc) in Tsys:
                        Tsys[(antena,track_loc)]=pd.concat([Tsys[(antena,track_loc)],FooDF ])
                    else:
                        Tsys[(antena,track_loc)] = FooDF
                else:
                    continue
    #add mjd
    for key in Tsys:
        foo = Time(list(Tsys[key].datetime)).mjd
        Tsys[key]['mjd'] = foo
    Tsfull = make_single_Tsys_table(Tsys)
    return Tsfull


def prepare_Tsys_data_ALMA_ER6(folder_path,only_ALMA=False,avg_Tsys=False):
    Tsys={}
    list_files = os.listdir(folder_path)
    list_files = [f for f in list_files if f[0] =='e']

    Tsyscols = ['Tsys_ch'+str(x) for x in range(1,33)]
    cols = ['datetime']+Tsyscols+['band']
    if avg_Tsys: cols = ['datetime','Tsys_st_R','Tsys_st_L','band']
    antena='NoAntena'
    FooDF = pd.DataFrame(columns=cols)

    for f in list_files:
        fpath = os.path.join(folder_path, f) # INI: what is this?!
        year = '20' + fpath[-16:-14]
    
        with open(fpath) as f:
            content = f.readlines()
       
        for line in content:
            #first line of data for given antenna
            if 'TSYS ' in line:
                #antena = AZ2Z[line[5:7]]
                antena = AZ2Z[line.split(' ')[1]]
                if antena=='A':
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
                
                #get the timestamp
                if antena=='A':
                    datetime_loc = time2datetimeyear(year,foo[0],'00:00:00')
                    datetime_loc = datetime_loc + ALMAtime2STANDARDtime(foo[1]) + timeoff
                    #print(datetime_loc)
                else:
                    continue
                    #datetime_loc = time2datetime1(foo[0],foo[1]) + timeoff
                
                #get the band
                #band_loc = dic_band[f.name[-4]]
                band_loc = fpath[-10:-8]
                
                
                #now get the Tsys data
                
                #JCMT and SMAR have single pol, so we just fill both with same value
                if antena=='A':
                    if isfloat(foo[-1]):
                        TsysAA = list(map(float,foo[2:]))
                    else:
                        TsysAA = list(map(float,foo[2:-1]))
                    #if len(TsysAA)<32:
                    #    print(datetime_loc,len(TsysAA),isfloat(foo[-1]),len(foo))
                    if all(np.asarray(TsysAA)>0):
                        
                        if avg_Tsys: # if we want to average down ALMA Tsys across channels
                            Tsys_R_loc = (1./np.mean(1./np.sqrt(TsysAA)))**2
                            data_loc = [datetime_loc, Tsys_R_loc, Tsys_R_loc, band_loc]
                        else:
                            data_loc = [datetime_loc]+TsysAA+[band_loc]
                        line_df = pd.DataFrame([data_loc], columns = cols)
                        FooDF = FooDF.append(line_df, ignore_index=True)
                
                    else: continue
                else: continue
            
            #lines at the end of data for given antena
            #print(f.name)
        
            #track_loc = f.name.split('_')[3]
            track_loc = fpath[-17:-11]

            if line[0]=='/':
                if antena=='A':
                    if (antena,track_loc) in Tsys:
                        Tsys[(antena,track_loc)]=pd.concat([Tsys[(antena,track_loc)],FooDF ])
                    else:
                        Tsys[(antena,track_loc)] = FooDF
                else:
                    continue
    #add mjd
    for key in Tsys:
        foo = Time(list(Tsys[key].datetime)).mjd
        Tsys[key]['mjd'] = foo
    
    
    Tsfull = make_single_Tsys_table(Tsys)
        
    return Tsfull




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

def time2datetimeyear(year, day, hour):
    """
    Convert strings representing year, day, and hour to a datetime object.

    Parameters
    ----------
    year : str
        The year as a string.
    day : str
        The day of the year as a string.
    hour : str
        The time in 'HH:MM:SS' format as a string.

    Returns
    -------
    datetime.datetime
        A datetime object representing the specified date and time.
    """
    
    day = int(day)

    hms = hour.split(':')
    h = int(hms[0])%24
    m = int(hms[1])
    s = int(hms[2])
    
    datet = (datetime.datetime(int(year), 1, 1, h, m, s) + datetime.timedelta(days=day-1))    
    
    return datet

def make_single_Tsys_table(Tsys, track2expt=track2expt):
    list_Tsys = []
    
    for key in Tsys:
        Tsys[key]['antena'] = [key[0]]*Tsys[key].shape[0]
        Tsys[key]['track'] = [key[1]]*Tsys[key].shape[0]
        list_Tsys.append(Tsys[key])

    Tsys_full =  pd.concat(list_Tsys,ignore_index=True)
    Tsys_full = Tsys_full.drop_duplicates()
    Tsys_full['expt_no'] =  list(map(lambda x: track2expt[x],Tsys_full['track']))
    return Tsys_full

def ALMAtime2STANDARDtime(atime):
    h = int(atime.split(':')[0])
    m = int(atime.split(':')[1].split('.')[0])
    #s = round(60*(float(atime.split(':')[1].split('.')[1])/100))
    sec = (atime.split(':')[1].split('.')[1])
    frac_min = float(sec)/10**(len(sec))
    sec_with_frac = 60*frac_min
    s = np.floor(sec_with_frac)
    us = int((sec_with_frac - s)*1e6)
    dt = datetime.timedelta(hours = h, minutes=m,seconds = s, microseconds=us)
    return dt

def group_tsys_blocks(filename):
    """
    Groups lines from a file into blocks based on 'TSYS' headers and '/' delimiters.

    Parameters
    ----------
    filename : str
        The file to be read with full path.

    Returns
    -------
    blocks : list of list of str
        A list where each element is a block of lines (as a list of strings) 
        starting with 'TSYS' and ending with the second occurrence of '/'.
    """
    
    with open(filename, 'r') as file:
        lines = file.readlines()

    blocks = []
    current_block = []
    in_block = False
    first_slash_found = False

    for line in lines:
        if line.startswith('TSYS'):
            if in_block:
                blocks.append(current_block)
            current_block = [line.strip()]
            in_block = True
            first_slash_found = False
        elif in_block:
            current_block.append(line.strip())
            if '/' in line:
                if first_slash_found:
                    blocks.append(current_block)
                    current_block = []
                    in_block = False
                else:
                    first_slash_found = True

    if in_block and current_block:
        blocks.append(current_block)

    return blocks

def extract_Tsys_from_antab(antabpath, AZ2Z=AZ2Z, track2expt=track2expt, bandL=bandL0, avg_channels=True):
    """
    Extracts Tsys values from ANTAB files and returns them as a DataFrame.
    Parameters
    ----------
    antabpath : str
        Path to the directory containing ANTAB files.
    AZ2Z : dict, optional
        Dictionary mapping station codes to their respective identifiers.
    track2expt : dict, optional
        Dictionary mapping track identifiers to experiment numbers.
    bandL : list, optional
        List of bands to filter the ANTAB files.
    avg_channels : bool, optional
        Whether to average Tsys values per channel.
    Returns
    -------
    pd.DataFrame
        DataFrame containing the extracted Tsys values with columns:
        ['datetime', 'mjd', 'Tsys_star_pol1', 'Tsys_star_pol2', 'band', 'station', 'track', 'expt_no'].
    Notes
    -----
    - The function assumes that the ANTAB files are named in a specific format and contain Tsys blocks.
    - The function handles different formats of Tsys values, including averaging per channel values if necessary.
    """

    list_files = [f for f in os.listdir(antabpath) if f[0] == 'e' and any(f'_{band}_' in f for band in bandL)]

    cols = ['datetime', 'mjd', 'Tsys_star_pol1','Tsys_star_pol2','band', 'station', 'track', 'expt_no']
    Tsys = pd.DataFrame(columns=cols)

    for f in list_files:
        fname = os.path.join(antabpath, f)
        track, band = os.path.basename(fname).split('_')[:2] # get track and band from the filename
        expt_no = track2expt[track] # get expt number from track2expt dict
        year = f"20{track[1:3]}"

        # get Tsys blocks from the file
        blocks = group_tsys_blocks(fname)

        for block in blocks:
            rowdict = {}
            rowdict['track'] = track
            rowdict['band'] = band
            rowdict['expt_no'] = expt_no

            first_slash_encountered = False
            for line in block:
                if line.startswith('TSYS'):
                    parts = line.split()
                    rowdict['station'] = AZ2Z[parts[1]] # get station code
                    print(rowdict['station'])

                    match = re.search(r'timeoff=\s*([\d.]+)', line)
                    if match:
                        timeoff = float(match.group(1))
                    else:
                        timeoff = 0.0
                    timeoff = datetime.timedelta(seconds = timeoff) # a datetime duration

                    # check if the first slash is at the end of the TSYS line
                    if line.endswith('/'):
                        first_slash_encountered = True

                # move on to the next block if the line starts with '/'
                if line.startswith('/'):
                    if first_slash_encountered:
                        break
                    else:
                        first_slash_encountered = True                    

                if isfloat(line.split()[0]):
                    parts = line.split()
                    parts = [x for x in parts if len(x) > 0]

                    # get datetime_loc
                    if rowdict['station'] == 'A':
                        datetime_loc = time2datetimeyear(year, parts[0], '00:00:00')
                        datetime_loc = datetime_loc + ALMAtime2STANDARDtime(parts[1]) + timeoff
                    else:
                        datetime_loc = time2datetimeyear(year, parts[0], parts[1]) + timeoff

                    rowdict['datetime'] = datetime_loc
                    rowdict['mjd'] = Time(datetime_loc).mjd # get mjd

                    # get Tsys values
                    if len(parts) == 3:
                        Tsys_star_pol1 = Tsys_star_pol2 = float(parts[2])
                    elif len(parts) == 4:
                        Tsys_star_pol1 = float(parts[2])
                        Tsys_star_pol2 = float(parts[3])
                    else:
                        # this station has Tsys values per channel (but not per pol); average them
                        #print(f"Station {rowdict['station']} has Tsys values per channel (but not per pol). Averaging them...")
                        Tsysarr = np.asarray(list(map(float,parts[2:])))
                        Tsysarr = Tsysarr[(Tsysarr != 0) & ~np.isnan(Tsysarr)]
                        if Tsysarr.size > 0:
                            Tsys_star_pol1 = Tsys_star_pol2 = (1./np.mean(1./np.sqrt(Tsysarr)))**2
                        else:
                            Tsys_star_pol1 = Tsys_star_pol2 = np.nan

                    rowdict['Tsys_star_pol1'] = Tsys_star_pol1
                    rowdict['Tsys_star_pol2'] = Tsys_star_pol2

                    rowdf = pd.DataFrame([rowdict], columns=cols)
                    Tsys = pd.concat([Tsys, rowdf], ignore_index=True)

    return Tsys

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
        dir_expt_HI = dirHI+'/'+str(int(expt))
        dir_expt_LO = dirLO+'/'+str(int(expt))
        if not os.path.exists(dir_expt_HI):
            os.makedirs(dir_expt_HI)
        if not os.path.exists(dir_expt_LO):
            os.makedirs(dir_expt_LO)
        
        for ant in antL:
            for sour in sourL:
                
                condS = (Tsys_full['source']==sour)
                condA = (Tsys_full['antena']==ant)
                condE = (Tsys_full['track']==expt2track[expt])
                condPositive = (Tsys_full['Tsys_st_L_lo']>0)&(Tsys_full['Tsys_st_R_lo']>0)&(Tsys_full['Tsys_st_R_hi']>0)&(Tsys_full['Tsys_st_L_hi']>0)
                Tsys_local = Tsys_full.loc[condS&condA&condE&condPositive]
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
                    if SEFDS_lo.shape[0]>0:
                        SEFDS_lo.to_csv(NameF_lo,sep=' ',index=False, header=False)
                    if SEFDS_hi.shape[0]>0:
                        SEFDS_hi.to_csv(NameF_hi, sep=' ', index=False, header = False)
                    print(sour+'_'+Z2AZ[ant]+' ok')
                except ValueError:
                    print(sour+'_'+Z2AZ[ant]+'crap, not ok')


def generate_and_save_sefd_data_new(Tsys_full, dict_dpfu, sourL=sourL, antL=antL0, exptL=exptL0, bandL=bandL0, pathSave = 'SEFDs'):
    if not os.path.exists(pathSave):
        os.makedirs(pathSave)

    for band in bands:
        dirBand = pathSave+'/SEFD_'+band
        if not os.path.exists(dirBand):
            os.makedirs(dirBand)
        for expt in exptL:
            dir_expt = dirBand+'/'+str(int(expt))
            if not os.path.exists(dir_expt):
                os.makedirs(dir_expt)
            
            for ant in antL:
                print('no ad hoc fix')
                for sour in sourL:

                    condB = (Tsys_full['band']==band)
                    condS = (Tsys_full['source']==sour)
                    condA = (Tsys_full['antena']==ant)
                    condE = (Tsys_full['track']==expt2track[expt])
                    condPositive = (Tsys_full['Tsys_st_L']>0)&(Tsys_full['Tsys_st_R']>0)
                    Tsys_local = Tsys_full.loc[condB&condS&condA&condE&condPositive]
                    
                    try:
                        Tsys_local.loc[:,'sefd_L'] = np.sqrt(Tsys_local['Tsys_st_L']/dict_dpfu[(ant,expt2track[expt],band,'L')])
                        Tsys_local.loc[:,'sefd_R'] = np.sqrt(Tsys_local['Tsys_st_R']/dict_dpfu[(ant,expt2track[expt],band,'R')])
                        if ant=='P':
                            Tsys_local.loc[:,'sefd_L'] = Tsys_local['sefd_L']/np.sqrt(Tsys_local['gainP'])
                            Tsys_local.loc[:,'sefd_R'] = Tsys_local['sefd_R']/np.sqrt(Tsys_local['gainP'])
                        elif ant=='Z':
                            Tsys_local.loc[:,'sefd_lo_L'] = Tsys_local['sefd_L']/np.sqrt(Tsys_local['gainZ'])
                            Tsys_local.loc[:,'sefd_lo_R'] = Tsys_local['sefd_R']/np.sqrt(Tsys_local['gainZ'])
                        elif ant=='X':
                            Tsys_local.loc[:,'sefd_lo_L'] = Tsys_local['sefd_L']/np.sqrt(Tsys_local['gainX'])
                            Tsys_local.loc[:,'sefd_lo_R'] = Tsys_local['sefd_R']/np.sqrt(Tsys_local['gainX'])
                     
                       
                        
                        try:
                            Tsys_local.loc[:,'foo_Imag_1'] = 0.*Tsys_local['sefd_R']
                            Tsys_local.loc[:,'foo_Imag_2'] = 0.*Tsys_local['sefd_R']
                            SEFDS = Tsys_local.loc[:,['mjd','sefd_R','foo_Imag_1','sefd_L','foo_Imag_2']]
                            SEFDS = SEFDS.sort_values('mjd')
                            ######
                            #RENAME '1921-293' to 'J1924-2914'
                            if sour=='1921-293': sour='J1924-2914'
                            #####################################
                        
                            NameF = dir_expt+'/'+sour+'_'+Z2AZ[ant]+'.txt'
                            ###APPLY AD HOC FIXES

                            SEFDS = ad_dummy_values(SEFDS)
                            #SEFDS = ad_hoc_fixes(SEFDS,ant,sour)
                            #CUT out too low SEFDs
                            SEFDS = SEFDS[(SEFDS['sefd_R']>1.)|(SEFDS['sefd_L']>1.)]
                            #####################
                            if SEFDS.shape[0]>0:
                                SEFDS.to_csv(NameF,sep=' ',index=False, header=False)
                            print(sour+'_'+Z2AZ[ant]+'_'+str(int(expt))+'_'+band+' ok')
                        except ValueError:
                            print(sour+'_'+Z2AZ[ant]+'_'+str(int(expt))+'_'+band+' crap, not ok')
                    
                    except KeyError:
                        print(sour+'_'+Z2AZ[ant]+'_'+str(int(expt))+'_'+band+' not available')
                            
def ad_hoc_fixes(df,ant,sour):
    fix_T = [[57854.5,57854.6],[57850.0,57856.2]]
    if (ant=='A')&(sour in ['SGRA','J1924-2914','J1733-1304']):
        for fixt in fix_T:
            cond=(df['mjd']>fixt[0])&(df['mjd']<fixt[1])&(df['sefd_R']<7.)
            df.loc[cond,'sefd_R'] *= np.sqrt(10.)
            df.loc[cond,'sefd_L'] *= np.sqrt(10.)
    return df

def ad_dummy_values(df,dmjd=0.001):
    print('Adding dummy boundary points to SEFDs')
    first = df.head(1).copy()
    last = df.tail(1).copy()
    first['mjd'] = list(map(lambda x: x-dmjd,first['mjd']))
    last['mjd'] = list(map(lambda x: x+dmjd,last['mjd']))
    df = pd.concat([first,df,last],ignore_index=True)
    #df.iloc[-1]
    return df
    

def generate_and_save_sefd_data_ALMA(Tsys_full, dict_dpfu, sourL=sourL, antL=antL0, exptL=exptL0,pathSave = 'SEFDs'):
    if not os.path.exists(pathSave):
        os.makedirs(pathSave)
    dirHI = pathSave+'/SEFD_HI'
    dirLO = pathSave+'/SEFD_LO'

    if not os.path.exists(dirHI):
        os.makedirs(dirHI)
    if not os.path.exists(dirLO):
        os.makedirs(dirLO)

    sefds_ch = ['sefd_ch'+str(x) for x in range(1,33)]

    for band in ['lo','hi']:
        dirBand = pathSave+'/SEFD_'+band.upper()
        if not os.path.exists(dirBand):
            os.makedirs(dirBand)
        for expt in exptL:
            dir_expt = dirBand+'/'+str(int(expt))
            if not os.path.exists(dir_expt):
                os.makedirs(dir_expt)
            for sour in sourL:
                
                condB = (Tsys_full['band']==band)
                condS = (Tsys_full['source']==sour)
                condE = (Tsys_full['track']==expt2track[expt])
                Tsys_local = Tsys_full.loc[condB&condS&condE]
                for cou in range(1,33):
                    sefd_ch = 'sefd_ch'+str(cou)
                    Tsys_ch = 'Tsys_ch'+str(cou)
                    Tsys_local.loc[:,sefd_ch] = np.sqrt(Tsys_local[Tsys_ch]/dict_dpfu[('A',expt2track[expt],band,'L')])
                try:
                    #Tsys_local.loc[:,'foo_Imag_1'] = 0.*Tsys_local['sefd_R']
                    #Tsys_local.loc[:,'foo_Imag_2'] = 0.*Tsys_local['sefd_R']
                    SEFDS = Tsys_local.loc[:,['mjd']+sefds_ch]
                    SEFDS = SEFDS.sort_values('mjd')
                    NameF = dir_expt+'/'+sour+'_'+'AA_full.txt'
                    if SEFDS.shape[0]>0:
                        SEFDS.to_csv(NameF,sep=' ',index=False, header=False)
                    print(sour+'_'+'AA_full'+'_'+str(int(expt))+'_'+band+' ok')
                except ValueError:
                    print(sour+'_'+'AA_full'+'_'+str(int(expt))+'_'+band+' crap, not ok')



def make_ra_dec_list(fpath):

    list_files = os.listdir(fpath)
    scans = pd.DataFrame({'source' : []})

    for fi in list_files:#loop over vex files in folder
        track_loc = fi[3].upper()
        vpath = os.path.join(fpath, fi) # INI: what is this?!
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

def make_scan_list(fpath,dict_gfit,version='2021',ant_locat=ant_locat,only_ALMA=False):
    '''
    version 'new' means dictionarries with keys (station, track, band, polarization)
    '''
    list_files = os.listdir(fpath)
    scans = pd.DataFrame({'source' : []})

    for fi in list_files:#loop over vex files in folder
        track_loc = fi[:-4]
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
            elevPloc = compute_elev(dict_ra[sour[x]],dict_dec[sour[x]],ant_locat['P'],datet[x]+TimeDelta(100., format='sec'))
            elevZloc = compute_elev(dict_ra[sour[x]],dict_dec[sour[x]],ant_locat['Z'],datet[x]+TimeDelta(100., format='sec'))
            elevXloc = compute_elev(dict_ra[sour[x]],dict_dec[sour[x]],ant_locat['X'],datet[x]+TimeDelta(100., format='sec'))
            elevloc ={'P':elevPloc, 'Z':elevZloc, 'X':elevXloc}
            elev.append(elevloc)
            ant_foo = set([nam2lett[aa.sched[x]['scan'][y]['site']] for y in range(len(aa.sched[x]['scan']))])
            if 'S' in ant_foo: ant_foo = ant_foo|{'R'}
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
        foo['scan_no'] = list(map(int,foo['scan_no']))
        foo['track'] = [track_loc]*foo.shape[0]
        foo['expt'] = [int(track2expt[track_loc])]*foo.shape[0]
        foo['antenas'] = antenas
        foo['duration'] = duration

        scans = pd.concat([scans,foo], ignore_index=True)

    scans = scans.reindex(['mjd_floor','expt','track','scan_no','source','time_min','time_max','duration','elev','antenas'],axis=1)
    scans = scans.sort_values('time_max')
    scans = scans.reset_index(drop=True)
    scans['scan_no_tot'] = scans.index
    
    if only_ALMA==False:
        scanelevP = [scans.elev[x]['P'] for x in range(scans.shape[0])]
        scanelevZ = [scans.elev[x]['Z'] for x in range(scans.shape[0])]
        scanelevX = [scans.elev[x]['X'] for x in range(scans.shape[0])]
        scans['gainP'] = [1.]*scans.shape[0]
        scans['gainZ'] = [1.]*scans.shape[0]
        scans['gainX'] = [1.]*scans.shape[0]
    
        if version=='2018':
            gainPf = lambda x: dict_gfit[('P','e18c21','b3','R')][0] + dict_gfit[('P','e18c21','b3','R')][1]*x + dict_gfit[('P','e18c21','b3','R')][2]*x**2
            gainZf = lambda x: dict_gfit[('Z','e18c21','b3','R')][0] + dict_gfit[('Z','e18c21','b3','R')][1]*x + dict_gfit[('Z','e18c21','b3','R')][2]*x**2
            gainXf = lambda x: dict_gfit[('X','e18c21','b3','R')][0] + dict_gfit[('X','e18c21','b3','R')][1]*x + dict_gfit[('X','e18c21','b3','R')][2]*x**2
        elif version=='2021':
            gainPf = lambda x: dict_gfit[('P','e21e18','b3','R')][0] + dict_gfit[('P','e21e18','b3','R')][1]*x + dict_gfit[('P','e21e18','b3','R')][2]*x**2
            gainZf = lambda x: dict_gfit[('Z','e21e18','b3','R')][0] + dict_gfit[('Z','e21e18','b3','R')][1]*x + dict_gfit[('Z','e21e18','b3','R')][2]*x**2
            gainXf = lambda x: dict_gfit[('X','e21e18','b3','R')][0] + dict_gfit[('X','e21e18','b3','R')][1]*x + dict_gfit[('X','e21e18','b3','R')][2]*x**2
            gainKf = lambda x: dict_gfit[('K','e21e18','b3','R')][0] + dict_gfit[('K','e21e18','b3','R')][1]*x + dict_gfit[('X','e21e18','b3','R')][2]*x**2
        else:
            gainPf = lambda x: dict_gfit[('P','e18c21')][0] + dict_gfit[('P','e18c21')][1]*x + dict_gfit[('P','e18c21')][2]*x**2
            gainZf = lambda x: dict_gfit[('Z','e18c21')][0] + dict_gfit[('Z','e18c21')][1]*x + dict_gfit[('Z','e18c21')][2]*x**2
            gainXf = lambda x: dict_gfit[('X','e18c21')][0] + dict_gfit[('X','e18c21')][1]*x + dict_gfit[('X','e18c21')][2]*x**2
        
        for index, row in scans.iterrows():
            if 'P' in row.antenas:
                foo = gainPf(scans.elev[index]['P'])
                scans.gainP[index] = float(foo)
            if 'Z' in row.antenas:
                foo = gainZf(scans.elev[index]['Z'])
                scans.gainZ[index] = float(foo)
            if 'X' in row.antenas:
                foo = gainXf(scans.elev[index]['X'])
                scans.gainX[index] = float(foo)
        
    #add SPT scan information
    #if SPT_total_scan==True:
    #    scans = add_SPT_total_scan(scans)

    return scans


def add_SPT_total_scan(scans):
    '''
    this adds column with global scan number for SPT
    because Tsys timestamps for SPT are too different from scans timestamps
    to be processed normally
    '''
    scansY = scans[list(map(lambda x: 'Y' in x, scans.antenas))].sort_values('time_min').reset_index(drop=True)
    scansY['scan_no_tot_Y'] = scansY.index
    scans_noY = scans[list(map(lambda x: 'Y' not in x, scans.antenas))]
    scans = pd.concat([scansY,scans_noY],ignore_index=True)
    scans=scans.sort_values('time_min').reset_index(drop=True)
    return scans

# Function to extract degrees, minutes, and seconds from dec
def extract_dms(dec):
    """
    Extracts degrees, minutes, and seconds from a declination string.
    Parameters
    ----------
    dec : str
        A string representing the declination in the format "±DdMmSs.s".
    Returns
    -------
    tuple or None
        A tuple containing degrees, minutes, and seconds as floats if the input matches the expected format.
        Returns None if the input does not match the expected format.
    Examples
    --------
    >>> extract_dms("+12d34'56.7\"")
    (12.0, 34.0, 56.7)
    >>> extract_dms("-12d34'56.7\"")
    (-12.0, 34.0, 56.7)
    >>> extract_dms("invalid")
    None
    """

    match = re.match(r"([+-]?\d+)d(\d+)'(\d+\.\d+)\"", dec)
    if match:
        degrees = float(match.group(1))
        minutes = float(match.group(2))
        seconds = float(match.group(3))
        return (degrees, minutes, seconds)
    return None

# Function to extract hours, minutes, and seconds from ra
def extract_hms(ra):
    """
    Extract hours, minutes, and seconds from a right ascension string.
    Parameters
    ----------
    ra : str
        A string representing the right ascension in the format "XhYmZs",
        where X, Y, and Z are numbers.
    Returns
    -------
    tuple of float or None
        A tuple containing hours, minutes, and seconds as floats if the input
        string matches the expected format. Returns None if the input string
        does not match the expected format.
    Examples
    --------
    >>> extract_hms("12h34m56.78s")
    (12.0, 34.0, 56.78)
    >>> extract_hms("invalid_string")
    None
    """

    match = re.match(r"(\d+)h(\d+)m(\d+\.\d+)s", ra)
    if match:
        hours = float(match.group(1))
        minutes = float(match.group(2))
        seconds = float(match.group(3))
        return (hours, minutes, seconds)
    return None

def extract_scans_from_all_vex(fpath, dict_gfit, version='2021', SMT2Z=SMT2Z, track2expt=track2expt, ant_locat=ant_locat, only_ALMA=False):
    """
    Generate a list of scans from all the VEX files in a given directory.

    Parameters
    ----------
    fpath : str
        Path to the directory containing VEX files.
    dict_gfit : dict
        Dictionary containing gain fit parameters.
    version : str, optional
        Version of the gain fit parameters to use ('2018', '2021', or other), by default '2021'.
    ant_locat : dict
        Dictionary containing antenna locations.
    only_ALMA : bool, optional
        If False, do some additional processing for other stations. Default is False.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing scan information including source, time, elevation, antennas, and gain.

    Notes
    -----
    The function processes VEX files to extract scan information, computes elevation for specified antennas,
    and applies gain corrections based on the provided version and gain fit parameters.
    """

    # get list of all VEX files in fpath
    list_files = [os.path.join(fpath, fname) for fname in os.listdir(fpath)]

    # initialize DataFrame to store scan information for all tracks
    tracks = pd.DataFrame({'source' : []})

    # extract only those sites that have polynomial coefficients for gains in the ANTAB files
    # convert to set to avoid duplicate sites that show up for different polarizations
    polygain_stations = list(set([key[0] for key, value in dict_gfit.items() if len(value) > 1]))

    # loop over all VEX files in fpath; one VEX file per observing track
    for fi in list_files:
        track_loc = os.path.splitext(os.path.basename(fi))[0] # get track name from vex file name

        aa = vex.Vex(fi) # read VEX file

        # Create dict_ra with 'source' as keys and the 3-tuple (hours, minutes, seconds) as values
        dict_ra = {d['source']: extract_hms(d['ra']) for d in aa.source}

        # Create dict_dec with 'source' as keys and the 3-tuple (degrees, minutes, seconds) as values
        dict_dec = {d['source']: extract_dms(d['dec']) for d in aa.source}

        # populate dataframe with scan information
        tstart_hr = [aa.sched[x]['start_hr'] for x in range(len(aa.sched))]
        source = [aa.sched[x]['source'] for x in range(len(aa.sched))]
        datet = []
        elev = []
        stations = []
        duration=[]

        # loop over each item in aa.sched to extract scan information
        for scanind in range(len(aa.sched)):
            # extract MJD floor from VEX file and convert to ISO format
            mjd_floor = Time(aa.sched[scanind]['mjd_floor'], format='mjd', scale='utc')
            mjd_floor_iso = Time(mjd_floor, format='iso', scale='utc')
            mjd_floor_iso = mjd_floor_iso + TimeDelta(tstart_hr[scanind]*3600., format='sec')
            datet.append(mjd_floor_iso)

            # Include only those stations in the elevation dict that have polynomial (degree > 1) coefficients for gains.
            # Exclude stations that are absent in SMT2Z dict (derived from ovex files) which contains only those stations
            # that actually observed. The info derived from VEX files may contain stations that were scheduled but ended up not observing.
            stations_in_scan = [value['site'] for value in aa.sched[scanind]['scan'].values()]
            stations_in_scan = [SMT2Z[station] for station in stations_in_scan if station in SMT2Z.keys()]

            # compute elevation for each station in the scan and append to elevation list
            elevloc = {}
            for station in stations_in_scan:
                if station in polygain_stations:
                    elevloc[station] = compute_elev(dict_ra[source[scanind]], dict_dec[source[scanind]], ant_locat[station], datet[scanind] + TimeDelta(100., format='sec'))
            elev.append(elevloc)

            # append list of stations in the scan to stations list
            if version == '2017' and 'S' in stations_in_scan:
                stations_in_scan = stations_in_scan | {'R'}
            stations.append(stations_in_scan)

            # append scan durations to duration list
            scan_sec = max([aa.sched[scanind]['scan'][y]['scan_sec'] for y in range(len(aa.sched[scanind]['scan']))])
            duration.append(scan_sec)

        time_min = [pd.Timestamp(datet[x].datetime) for x in range(len(datet))]
        time_max = [time_min[x] + datetime.timedelta(seconds=duration[x]) for x in range(len(aa.sched))]

        pervexdf = pd.DataFrame(aa.sched)
        pervexdf = pervexdf[['source','mjd_floor','start_hr']]
        pervexdf['time_min'] = time_min
        pervexdf['time_max'] = time_max
        pervexdf['elev'] = elev
        pervexdf['scan_no'] = pervexdf.index
        pervexdf['scan_no'] = list(map(int, pervexdf['scan_no']))
        pervexdf['track'] = [track_loc]*pervexdf.shape[0]
        pervexdf['expt'] = [int(track2expt[track_loc])]*pervexdf.shape[0]
        pervexdf['stations'] = stations
        pervexdf['duration'] = duration

        # concatenate pervexdf to scans DataFrame
        tracks = pd.concat([tracks, pervexdf], ignore_index=True)

    tracks = tracks.reindex(['mjd_floor','expt','track','scan_no','source','time_min','time_max','duration','elev','stations'], axis=1)
    tracks = tracks.sort_values('time_max')
    tracks = tracks.reset_index(drop=True)
    tracks['scan_no_tot'] = tracks.index

    # Compute gain curves and add to dataframe
    # Get the current band from dict_gfit and set pol to 'R' for accessing gfit coeffs.
    # This works because all keys pertain to the same band and both polarizations.
    gfitband = list(dict_gfit.keys())[0][2]
    gfitpol = 'R'
    for station in polygain_stations:
        tracks[f'gain{station}'] = [1.]*tracks.shape[0]
        for index, row in tracks.iterrows():
            if station in row.stations:
                #gainf = lambda x: dict_gfit[(station, 'e21e18', 'b3', 'R')][0] + dict_gfit[(station, 'e21e18', 'b3', 'R')][1]*x + dict_gfit[(station, 'e21e18', 'b3', 'R')][2]*x**2
                coeffs = dict_gfit[(station, row.track, gfitband, gfitpol)]
                gainf = Polynomial(coeffs)
                foo = gainf(tracks.elev[index][station])
                tracks[f'gain{station}'][index] = float(foo)

    return tracks

def match_scans_Tsys(scans, Tsys, only_ALMA=False):
    #negative labels for ANTAB data taken between scans

    #create scan labels to match Tsys with scans
    bins_labels = [None]*(2*scans.shape[0]-1)
    bins_labels[1::2] = list(map(lambda x: -x,list(scans['scan_no_tot'])[1:]))
    bins_labels[::2] = list(scans['scan_no_tot'])
    #print('lab ', np.shape(bins_labels))
    #add bin for time before the first scan

    foo = list(scans['scan_no_tot'])[0]
    if foo==0: foo=10000
    #print(foo)
    bins_labels = [-foo]+bins_labels
    #print(bins_labels)
    #print('lab ', np.shape(bins_labels))
    #print(bins_labels)
    dtmin = datetime.timedelta(seconds = 0.) 
    dtmax = datetime.timedelta(seconds = 0.) 
    binsT = [None]*(2*scans.shape[0])
    binsT[::2] = list(map(lambda x: x - dtmin,list(scans.time_min)))
    binsT[1::2] = list(map(lambda x: x + dtmax,list(scans.time_max)))
    #print('bins ', np.shape(binsT))
    #add bin for time before the first scan
    min_time = min(scans.time_min)-datetime.timedelta(seconds = 1600.)
    binsT = [min_time]+binsT
    #print('bins ', np.shape(binsT))
    #add scan indexed label to Tsys 
    ordered_labels = pd.cut(Tsys.datetime, binsT,labels = bins_labels)
    if list(Tsys.antena.unique())==['Y']:
        Tsys.loc[:,'scan_no_tot'] = np.abs(np.asarray(list(ordered_labels)))
    else:
        Tsys.loc[:,'scan_no_tot'] = list(ordered_labels)
    
    #if there is a column with SPT scans count, then treat SPT separately
    #print( list(Tsys.antena.unique()) )
    '''
    if list(Tsys.antena.unique())==['Y']:
        scansY = scans[list(map(lambda x: 'Y' in x, scans.antenas))].sort_values('time_min').reset_index(drop=True)
        bins_labels_Y = [None]*(2*scansY.shape[0]-1)
        bins_labels_Y[1::2] = list(map(lambda x: -x-1,list(scansY['scan_no_tot_Y'])[:-1]))
        bins_labels_Y[::2] = list(scansY['scan_no_tot_Y'])
        #append for data after last scan
        #bins_labels_Y.append(10000)
        #bins_labels_Y = [-10000]+bins_labels_Y

        dtmin = datetime.timedelta(seconds = 5.) 
        dtmax = datetime.timedelta(seconds = 5.) 
        binsTY = [None]*(2*scansY.shape[0])
        binsTY[::2] = list(map(lambda x: x - dtmin,list(scansY.time_min)))
        binsTY[1::2] = list(map(lambda x: x + dtmax,list(scansY.time_max)))
        #append for data after last scan
        #binsTY.append(max(scansY.time_max)+datetime.timedelta(seconds = 300.))
        #min_time = min(scansY.time_min)-datetime.timedelta(seconds = 1600.)
        #binsTY = [min_time]+binsTY

        ordered_labels_Y = pd.cut(Tsys.datetime, binsTY,labels = bins_labels_Y)
        #absolute value so that Tsys value before scan is attributed to the SPT
        #scan taken after
        #print(any(np.isnan(np.asarray(ordered_labels_Y))))

        #print(Tsys[Tsys['scan_no_tot_Y']!=Tsys['scan_no_tot_Y'] ])
        
        Tsys.loc[:,'scan_no_tot_Y'] = np.abs(np.asarray(list(ordered_labels_Y)))
        Tsys = Tsys[Tsys['scan_no_tot_Y']==Tsys['scan_no_tot_Y']]
        #Tsys = Tsys[Tsys['scan_no_tot_Y']<5000]
        DictSourceY = dict(zip(list(scans.scan_no_tot_Y), list(scans.source)))

    '''

    DictSource = dict(zip(list(scans.scan_no_tot), list(scans.source)))
    if only_ALMA==False:
        DictGainP = dict(zip(list(scans.scan_no_tot), list(scans.gainP)))
        DictGainZ = dict(zip(list(scans.scan_no_tot), list(scans.gainZ)))
        DictGainX = dict(zip(list(scans.scan_no_tot), list(scans.gainX)))
    DictTmin = dict(zip(list(scans.scan_no_tot), list(scans.time_min)))


    #select only the data taken during the scan but for SPT
    #use data taken between scans by attributing then to the scan after
    #Tsys.loc[(Tsys['antena']=='Y')&(Tsys['scan_no_tot']<0), 'scan_no_tot'] = Tsys.loc[(Tsys['antena']=='Y')&(Tsys['scan_no_tot']<0), 'scan_no_tot'].apply(np.abs)
    Tsys = Tsys[list(map(lambda x: x >= 0, (Tsys['scan_no_tot'])   ))]
    '''
    if list(Tsys.antena.unique())==['Y']:
        Tsys.loc[:,'source'] = list(map(lambda x: DictSourceY[x], Tsys['scan_no_tot_Y']))
    else:
    '''   
    Tsys.loc[:,'source'] = list(map(lambda x: DictSource[x], Tsys['scan_no_tot']))
    if only_ALMA==False:
        Tsys.loc[:,'gainP'] = list(map(lambda x: DictGainP[x], Tsys['scan_no_tot']))
        Tsys.loc[:,'gainZ'] = list(map(lambda x: DictGainZ[x], Tsys['scan_no_tot']))
        Tsys.loc[:,'gainX'] = list(map(lambda x: DictGainX[x], Tsys['scan_no_tot']))
    Tsys.loc[:,'t_scan'] = list(map(lambda x: DictTmin[x], Tsys['scan_no_tot']))
    
    Tsys = Tsys.sort_values('datetime').reset_index(drop=True)
    
    return Tsys


def global_match_scans_Tsys_both_bands(scans,Tsys_full,only_ALMA=False):
    Tsys_match_lo = global_match_scans_Tsys(scans,Tsys_full[Tsys_full.band=='lo'],only_ALMA=False)   
    Tsys_match_hi = global_match_scans_Tsys(scans,Tsys_full[Tsys_full.band=='hi'],only_ALMA=False)

    Tsys_match = pd.concat([Tsys_match_lo,Tsys_match_hi],ignore_index=True)
    return Tsys_match

def global_match_scans_Tsys(scans, Tsys_full, antL=antL0, only_ALMA=False):

    Tsys_match = pd.DataFrame({'source' : []})

    #print(Tsys_full.scan_no_tot)
    for ant in antL:
        #for expt in exptL0:
        for expt in list(Tsys_full.expt_no.unique()):
            #print(ant,expt)
            
            condA = (Tsys_full['station']==ant)
            condE = (Tsys_full['expt_no']==expt)
            Tsys_loc = Tsys_full.loc[condA&condE].sort_values('datetime').reset_index(drop=True)
            scans_loc = scans[(scans.expt == expt)&list(map(lambda x: ant in x,scans.stations))].sort_values('time_min').reset_index(drop=True)
            #print(np.shape(Tsys_loc),np.shape(scans_loc))
            if(np.shape(Tsys_loc)[0]>0):
                Tsys_foo = match_scans_Tsys(scans_loc,Tsys_loc,only_ALMA=only_ALMA)
                Tsys_match = pd.concat([Tsys_match,Tsys_foo], ignore_index=True)
            else: continue

    Tsys_match = Tsys_match.sort_values('datetime').reset_index(drop=True)

    return Tsys_match


def get_sefds(antab_path ='ANTABS/', vex_path = 'VexFiles/', sourL=sourL,antL=antL0, exptL = exptL0):

    print('Getting the calibration data...')
    #TABLE of CALIBRATION DATA from ANTAB files
    dict_dpfu, dict_gfit, Tsys = prepare_calibration_data(antab_path)
    Tsys_full = make_single_Tsys_table(Tsys)

    print('Getting the scans data...')
    #TABLE of SCANS from VEX files, using elevation gain info
    scans = make_scan_list(vex_path,dict_gfit,version='old')

    print('Matching calibration to scans...')
    #MATCH CALIBRATION with SCANS to determine the source and 
    Tsys_match = global_match_scans_Tsys(scans,Tsys_full)

    print('Saving sefd files...')
    #produce a priori calibration data
    generate_and_save_sefd_data(Tsys_match, dict_dpfu, dict_gfit, sourL, antL, exptL)

def get_sefds_new(antab_path ='ANTABS/', vex_path = 'VexFiles/', version = '2021', sourL=sourL, antL=antL0, AZ2Z=AZ2Z, SMT2Z=SMT2Z, track2expt=track2expt, ant_locat=ant_locat, exptL = exptL0, bandL=bandL0, pathSave = 'SEFDs'):
    '''
    new version for when files are separate for bands
    '''
    print('Getting the calibration data...')
    #TABLE of CALIBRATION DATA from ANTAB files
    dp, gf = extract_dpfu_gfit_from_all_antab(antab_path, AZ2Z, bandL)
    Ts = extract_Tsys_from_antab(antab_path, AZ2Z, track2expt, bandL)

    print('Getting the scans data...')
    #TABLE of SCANS from VEX files, using elevation gain info
    scans = extract_scans_from_all_vex(vex_path, gf, version=version, SMT2Z=SMT2Z, track2expt=track2expt, ant_locat=ant_locat)

    print('Matching calibration to scans...')
    #MATCH CALIBRATION with SCANS to determine the source and 
    Tsys_match = global_match_scans_Tsys(scans, Ts, antL=antL)

    print('Saving sefd files...')
    #produce a priori calibration data
    generate_and_save_sefd_data_new(Tsys_match, dp, sourL, antL, exptL, bandL)

def get_sefds_ALMA(antab_path ='ANTABS/', vex_path = 'VexFiles/',dpfu_path=None, sourL=sourL,antL=antL0, exptL = exptL0, bandL=bandL0, pathSave = 'SEFDs_ALMA',version='ER5',only_ALMA=False,avg_Tsys=False):
    '''
    new version for when files are separate for bands
    '''
    print('Getting the calibration data...')
    #TABLE of CALIBRATION DATA from ANTAB files

    if dpfu_path:
        dp, gf = extract_dpfu_gfit_from_all_antab(dpfu_path)
    else:
        dp, gf = extract_dpfu_gfit_from_all_antab(antab_path)

    if version=='ER6':
        TsA = prepare_Tsys_data_ALMA_ER6(antab_path,only_ALMA=only_ALMA,avg_Tsys=avg_Tsys)
    else:
        TsA = prepare_Tsys_data_ALMA(antab_path)

    print('Getting the scans data...')
    #TABLE of SCANS from VEX files, using elevation gain info
    scans = make_scan_list(vex_path,gf,only_ALMA=only_ALMA)

    print('Matching calibration to scans...')
    #MATCH CALIBRATION with SCANS to determine the source and 
    TsysA_match = global_match_scans_Tsys(scans,TsA,only_ALMA=only_ALMA)

    print('Saving sefd files...')
    #produce a priori calibration data
    if avg_Tsys:
        generate_and_save_sefd_data_new(TsysA_match, dp, sourL, antL, exptL, bandL)
    else:
        generate_and_save_sefd_data_ALMA(TsysA_match, dp, sourL, antL, exptL, bandL, pathSave)


def modify_Tsys_match(Tsys_match,dict_dpfu):
    Tsys_match['dpfu'] = list(map(lambda x: dict_dpfu[x],list(zip(Tsys_match['antena'],Tsys_match['track']))))
    Tsys_P = Tsys_match[Tsys_match.antena=='P']
    Tsys_Z = Tsys_match[Tsys_match.antena=='Z']
    Tsys_rest = Tsys_match[list(map(lambda x: x not in ['P', 'Z'],Tsys_match.antena))]
    Tsys_P.loc[:,'gain'] = Tsys_P['gainP']
    Tsys_Z.loc[:,'gain'] = Tsys_Z['gainZ']
    Tsys_rest.loc[:,'gain'] = [1.]*np.shape(Tsys_rest)[0]
    Tsys = pd.concat([Tsys_P,Tsys_Z,Tsys_rest],ignore_index=True)
    Tsys['sefd_L_lo'] = Tsys['Tsys_st_L_lo']/Tsys['gain']/Tsys['dpfu']
    Tsys['sefd_R_lo'] = Tsys['Tsys_st_R_lo']/Tsys['gain']/Tsys['dpfu']
    Tsys['sefd_L_hi'] = Tsys['Tsys_st_L_hi']/Tsys['gain']/Tsys['dpfu']
    Tsys['sefd_R_hi'] = Tsys['Tsys_st_R_hi']/Tsys['gain']/Tsys['dpfu']
    SEFD = Tsys[['datetime','antena','expt_no','source','mjd','scan_no_tot','sefd_L_lo','sefd_R_lo','sefd_L_hi','sefd_R_hi']].copy()
    SEFD.sort_values(['datetime','antena'], inplace=True)
    SEFD.reset_index(inplace=True)
    return SEFD

def modify_Tsys_match_new(Tsys_match,dict_dpfu):
    Tsys_match['dpfu_R'] = list(map(lambda x: dict_dpfu[x],list(zip(Tsys_match['antena'],Tsys_match['track'],Tsys_match['band'],['R']*len(Tsys_match['antena'])))))
    Tsys_match['dpfu_L'] = list(map(lambda x: dict_dpfu[x],list(zip(Tsys_match['antena'],Tsys_match['track'],Tsys_match['band'],['L']*len(Tsys_match['antena'])))))
    Tsys_P = Tsys_match[Tsys_match.antena=='P']
    Tsys_Z = Tsys_match[Tsys_match.antena=='Z']
    Tsys_X = Tsys_match[Tsys_match.antena=='X']
    Tsys_rest = Tsys_match[list(map(lambda x: x not in ['P', 'Z','X'],Tsys_match.antena))]
    Tsys_P.loc[:,'gain'] = Tsys_P['gainP']
    Tsys_Z.loc[:,'gain'] = Tsys_Z['gainZ']
    Tsys_X.loc[:,'gain'] = Tsys_X['gainX']
    Tsys_rest.loc[:,'gain'] = [1.]*np.shape(Tsys_rest)[0]
    Tsys = pd.concat([Tsys_P,Tsys_Z,Tsys_X,Tsys_rest],ignore_index=True)
    Tsys['sefd_L'] = Tsys['Tsys_st_L']/Tsys['gain']/Tsys['dpfu_L']
    Tsys['sefd_R'] = Tsys['Tsys_st_R']/Tsys['gain']/Tsys['dpfu_R']
    SEFD = Tsys[['datetime','antena','expt_no','source','mjd','scan_no_tot','sefd_L','sefd_R']].copy()
    SEFD.sort_values(['datetime','antena'], inplace=True)
    SEFD.reset_index(inplace=True)
    return SEFD


def apply_sefds(alist,SEFDfits):

    alist['mjd'] = Time(list(alist.datetime)).mjd
    list_expt = list(alist.expt_no.unique())
    list_source = list(alist.source.unique())
    list_baseline = list(alist.baseline.unique())
    list_polarization = list(alist.polarization.unique())
    list_band = list(alist.band.unique())

    columns=list(alist.columns)
    alist_sefd = pd.DataFrame(columns=columns)
    for expt in list_expt:
        for sour in list_source:
            for base in list_baseline:
                for band in list_band:
                    for polar in list_polarization:

                        alist_loc = alist[(alist.expt_no==expt)&(alist.source==sour)&(alist.baseline==base)&(alist.band==band)&(alist.polarization==polar)]
                        time_mjd = np.asarray(alist_loc.mjd)
                        try:
                            alist_loc['sefd_cal'] = np.sqrt(SEFDfits[(expt,sour,base[0],polar[0],band)](time_mjd)*SEFDfits[(expt,sour,base[1],polar[1],band)](time_mjd))
                            alist_sefd = pd.concat([alist_sefd,alist_loc],ignore_index=True)
                        except KeyError:
                            continue
            print(expt, sour)                
    return alist_sefd


def apply_sefds_trans(alist,SEFDfits):

    alist['mjd'] = Time(list(alist.datetime)).mjd
    alist['sefd'] = list(zip(alist['expt_no'],alist['source'],alist['baseline'],alist['polarization'],alist['band'],alist['mjd']))
    
    def get_sefd(x):
        try:
            answ = np.sqrt(SEFDfits[(x[0],x[1],x[2][0],x[3][0],x[4])](x[5])*SEFDfits[(x[0],x[1],x[2][1],x[3][1],x[4])](x[5]))   
        except KeyError:
            answ = None
        return answ

    foo = alist['sefd'].transform(get_sefd)       
    alist['sefd']= foo
    alist['amp_no_ap'] = alist['amp']
    alist['amp_ap'] = alist['amp_no_ap']*alist['sefd']/10000
    alist.drop('amp',axis=1,inplace=True)
    alist.rename(index=str, columns={"amp_ap": "amp"},inplace=True)

    if 'sigma' in alist.columns:
        alist['sigma_no_ap'] = alist['sigma']
        alist['sigma_ap'] = alist['sigma']*alist['sefd']/10000
    else:
        alist['sigma'] = alist['amp_no_ap']/alist['snr']
        alist['sigma_no_ap'] = alist['amp_no_ap']/alist['snr']
        alist['sigma_ap'] = (alist['amp_no_ap']/alist['snr'])*alist['sefd']/10000
    
    alist.drop('sigma',axis=1,inplace=True)
    alist.rename(index=str, columns={"sigma_ap": "sigma"},inplace=True)
    
    return alist  

def dict_sefd_fits(SEFD,kind='nearest'):
    dict_sefd_fits = {}
    list_expt = list(SEFD.expt_no.unique())
    list_source = list(SEFD.source.unique())
    list_antena = list(SEFD.antena.unique())
    for expt in list_expt:
        for sour in list_source:
            for ant in list_antena:
                try:
                    SEFD_loc = SEFD[(SEFD.expt_no==expt)&(SEFD.source==sour)&(SEFD.antena==ant)]
            
                    sefd_L_lo_fit = si.interp1d(SEFD_loc[list(map(lambda x: x==x,SEFD_loc.sefd_L_lo))].mjd,SEFD_loc[list(map(lambda x: x==x,SEFD_loc.sefd_L_lo))].sefd_L_lo,kind=kind,fill_value='extrapolate')
                    sefd_L_hi_fit = si.interp1d(SEFD_loc[list(map(lambda x: x==x,SEFD_loc.sefd_L_hi))].mjd,SEFD_loc[list(map(lambda x: x==x,SEFD_loc.sefd_L_hi))].sefd_L_hi,kind=kind,fill_value='extrapolate')
                    sefd_R_lo_fit = si.interp1d(SEFD_loc[list(map(lambda x: x==x,SEFD_loc.sefd_R_lo))].mjd,SEFD_loc[list(map(lambda x: x==x,SEFD_loc.sefd_R_lo))].sefd_R_lo,kind=kind,fill_value='extrapolate')
                    sefd_R_hi_fit = si.interp1d(SEFD_loc[list(map(lambda x: x==x,SEFD_loc.sefd_R_hi))].mjd,SEFD_loc[list(map(lambda x: x==x,SEFD_loc.sefd_R_hi))].sefd_R_hi,kind=kind,fill_value='extrapolate')

                    dict_sefd_fits[(expt,sour,ant,'L','lo')] = sefd_L_lo_fit
                    dict_sefd_fits[(expt,sour,ant,'L','hi')] = sefd_L_hi_fit
                    dict_sefd_fits[(expt,sour,ant,'R','lo')] = sefd_R_lo_fit
                    dict_sefd_fits[(expt,sour,ant,'R','hi')] = sefd_R_hi_fit
                except (ValueError,AttributeError):
                    #attribute error happens when zero length vector is passed
                    continue
    return dict_sefd_fits

              
def add_mjd(alist):
    alist['mjd'] = Time(list(alist.datetime)).mjd
    return alist

