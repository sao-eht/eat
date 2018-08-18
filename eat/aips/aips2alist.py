'''
Codes to import uvfits files to pandas
similar to HOPS alists importet by eat

Maciek Wielgus, maciek.wielgus@gmail.com
'''

import pandas as pd
import time, datetime, os, glob
import numpy as np
import numpy.random as npr
import sys
from astropy.time import Time, TimeDelta

#2017 release E1
#dictBase = { 'A': {1:'A',2:'X',3:'Z', 4:'J',5:'L',6:'P',7:'S',8:'R'},
#             'D': {1:'A',2:'X',3:'Z', 4:'J',5:'L',6:'P',7:'S',8:'R'},
#             'C': {1:'A',2:'X',3:'Z',4:'J',5:'S',6:'R',7:'L',8:'P'} }

#2017 release E1B
dictBase_ER1 = { 'A': {1: 'A', 2: 'X', 3: 'Z', 4: 'L', 5: 'P', 6: 'J', 7: 'S', 8: 'R'},
             'B': {1: 'A', 2: 'X', 3: 'Z', 4: 'J', 5: 'L', 6: 'P', 7: 'S', 8: 'R'},
             'C': {1: 'A', 2: 'X', 3: 'Z', 4: 'J', 5: 'L', 6: 'P', 7: 'S', 8: 'R'},
             'D': {1: 'A', 2: 'X', 3: 'Z', 4: 'J', 5: 'L', 6: 'P', 7: 'S', 8: 'R'},
             'E': {1: 'A', 2: 'X', 3: 'Z', 4: 'P', 5: 'L', 6: 'J', 7: 'S', 8: 'R'} }
dictBase_ER3 = { 'A': {1: 'A', 2: 'X', 3: 'Z', 4: 'L', 5: 'P', 6: 'J', 7: 'S', 8: 'R'},
             'B': {1: 'A', 2: 'X', 3: 'Z', 4: 'J', 5: 'L', 6: 'P', 7: 'S', 8: 'R'},
             'C': {1: 'A', 2: 'X', 3: 'Z', 4: 'J', 5: 'L', 6: 'P', 7: 'S', 8: 'R'},
             'D': {1: 'A', 2: 'X', 3: 'Z', 4: 'J', 5: 'L', 6: 'P', 7: 'S', 8: 'R', 9:'Y'},
             'E': {1: 'A', 2: 'X', 3: 'P', 4: 'Z', 5: 'J', 6: 'L', 7: 'S', 8: 'Y', 9:'R'} }

dictBase=dictBase_ER3
           
AZ2Z = {'AZ': 'Z', 'PV': 'P', 'SM':'S', 'SR':'R','JC':'J', 'AA':'A','AP':'X', 'LM':'L'}
SMT2Z = {'APEX': 'X', 'JCMT': 'J', 'LMT':'L', 'SMR':'R', 'SMA':'S', 'SMT':'Z', 'PV':'P'}
Z2AZ = {'Z':'AZ', 'P':'PV', 'S':'SM', 'R':'SR','J':'JC', 'A':'AA','X':'AP', 'L':'LM'}
track2expt ={'D':3597,'B':3598, 'C':3599,'A':3600,'E':3601}
expt2track ={3597:'D',3598:'B', 3599:'C',3600:'A',3601:'E'}
nam2lett = {'ALMA':'A','APEX':'X','LMT':'L','PICOVEL':'P','SMTO':'Z','SPT':'T','JCMT':'J','SMAP':'S'}

#from fractional jd to track for April 2017 observations 
def jd2track2017(jd):
    if (jd > 2457853.470)&(jd < 2457854.132 ):
        return 'A'
    elif (jd > 2457849.531)&(jd < 2457850.177):
        return 'B'
    elif (jd > 2457850.667)&(jd < 2457851.363):
        return 'C'
    elif (jd > 2457848.438)&(jd < 2457849.214):
        return 'D'
    elif (jd > 2457854.427)&(jd < 2457855.141):
        return 'E'
    else:
        return None

def jd2expt2017(jd):
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
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


def uvfits2csvs(folder_path,folder_destin=''):
    '''
    takes folder folder_path of uvfits files and rewrites them as csv files
    in folder_destination
    uses Kazu's sparse imaging library codes
    '''
    from eat.aips import vex, uvdata
    listFiles = os.listdir(folder_path)
    
    if folder_destin=='':
        folder_destin = folder_path
    if not os.path.exists(folder_destin):
        os.makedirs(folder_destin)
    for fileN in listFiles:
        try:
            pathFile = folder_path+fileN
            print(fileN)
            uvfits = uvdata.UVFITS(pathFile)
            uvfitsRR = uvfits.select_stokes('RR')
            uvfitsLL = uvfits.select_stokes('LL')
            uvfitsLR = uvfits.select_stokes('LR')
            uvfitsRL = uvfits.select_stokes('RL')
            vistableRR = uvfitsRR.make_vistable()
            vistableLL = uvfitsLL.make_vistable()
            vistableLR = uvfitsLR.make_vistable()
            vistableRL = uvfitsRL.make_vistable()
            fileN_RR_csv = fileN[:-6]+'RR.csv'
            fileN_LL_csv = fileN[:-6]+'LL.csv'
            fileN_LR_csv = fileN[:-6]+'LR.csv'
            fileN_RL_csv = fileN[:-6]+'RL.csv'
            print('preparing to saving files')
            if fileN.split('.')[0][-2:] in {'1l','lo','LO'}:
                if not os.path.exists(folder_destin+'LO/'):
                    os.makedirs(folder_destin+'LO/')
                vistableRR.to_csv(folder_destin+'LO/'+fileN_RR_csv)
                vistableLL.to_csv(folder_destin+'LO/'+fileN_LL_csv)
                vistableLR.to_csv(folder_destin+'LO/'+fileN_LR_csv)
                vistableRL.to_csv(folder_destin+'LO/'+fileN_RL_csv)
            elif fileN.split('.')[0][-2:] in {'2h','hi','HI'}:
                if not os.path.exists(folder_destin+'HI/'):
                    os.makedirs(folder_destin+'HI/')
                vistableRR.to_csv(folder_destin+'HI/'+fileN_RR_csv)
                vistableLL.to_csv(folder_destin+'HI/'+fileN_LL_csv)
                vistableLR.to_csv(folder_destin+'HI/'+fileN_LR_csv)
                vistableRL.to_csv(folder_destin+'HI/'+fileN_RL_csv)
            else:
                if not os.path.exists(folder_destin+'bandZ/'):
                    os.makedirs(folder_destin+'bandZ/')
                vistableRR.to_csv(folder_destin+'bandZ/'+fileN_RR_csv)
                vistableLL.to_csv(folder_destin+'bandZ/'+fileN_LL_csv)
                vistableLR.to_csv(folder_destin+'bandZ/'+fileN_LR_csv)
                vistableRL.to_csv(folder_destin+'bandZ/'+fileN_RL_csv)

        except IOError:
            print(IOError)
            continue

def uvfits2csvs_NEW(folder_path,folder_destin='',polarL=['RR','LL','LR','RL']):
    '''
    takes folder folder_path of uvfits files and rewrites them as csv files
    in folder_destination
    uses Kazu's sparse imaging library codes
    '''
    from eat.aips import vex, uvdata
    listFiles = os.listdir(folder_path)
    
    if folder_destin=='':
        folder_destin = folder_path
    if not os.path.exists(folder_destin):
        os.makedirs(folder_destin)
    for fileN in listFiles:
        try:
            pathFile = folder_path+fileN
            print(fileN)
            uvfits = uvdata.UVFITS(pathFile)
            uvfitsD ={}
            vistableD={}
            fileND={}
            for polar in polarL:
                uvfitsD[polar] = uvfits.select_stokes(polar)
                vistableD[polar] = uvfitsD[polar].make_vistable()
                fileND[polar] = fileN[:-6]+'.'+polar+'.csv'
                print('preparing to saving files')
                if fileND[polar].split('.')[0][-2:] in {'1l','lo','LO'}:
                    if not os.path.exists(folder_destin+'LO/'):
                        os.makedirs(folder_destin+'LO/')
                    vistableD[polar].to_csv(folder_destin+'LO/'+fileND[polar])
                elif fileND[polar].split('.')[0][-2:] in {'2h','hi','HI'}:
                    if not os.path.exists(folder_destin+'HI/'):
                        os.makedirs(folder_destin+'HI/')
                    vistableD[polar].to_csv(folder_destin+'HI/'+fileND[polar])
                else:
                    if not os.path.exists(folder_destin+'bandZ/'):
                        os.makedirs(folder_destin+'bandZ/')
                    vistableD[polar].to_csv(folder_destin+'bandZ/'+fileND[polar])
        except IOError:
            print(IOError)
            continue

def uvfits2csvs_file(file_path,folder_destin='',polarL=['RR','LL','LR','RL']):
    '''
    takes file_path of uvfits files and rewrites them as csv files
    in folder_destination
    uses Kazu's sparse imaging library codes
    '''
    from eat.aips import vex, uvdata
    fileN = file_path.split('/')[-1]
    try:
        uvfits = uvdata.UVFITS(file_path)
        uvfitsD ={}
        vistableD={}
        fileND={}
        for polar in polarL:
            uvfitsD[polar] = uvfits.select_stokes(polar)
            vistableD[polar] = uvfitsD[polar].make_vistable()
            rawname=''.join(fileN.split('.')[:-1])
            if len(fileN.split('.')) ==1:
                rawname = fileN
            fileND[polar] = rawname+'.'+polar+'.csv'
            vistableD[polar].to_csv(folder_destin+fileND[polar])
    except IOError:
        print(IOError)


def make_scan_list(fpath):
    '''
    generates data frame with information about scans based on vex files
    uses vex parser written by Hotaka
    '''
    from eat.aips import vex, uvdata
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
        scans = pd.concat([scans,foo], ignore_index=True)
    scans = scans.reindex_axis(['mjd_floor','expt','track','scan_no','source','time_min','time_max','duration','antenas'],axis=1)
    scans = scans.sort_values('time_max')
    scans = scans.reset_index(drop=True)
    scans['scan_no_tot'] = scans.index
    return scans


def add_datetime(AIPS):
    ddV= []
    for cou in range(AIPS.shape[0]):
        h = int(AIPS['hour'][cou])
        m = int(AIPS['min'][cou])
        s = int(AIPS['sec'][cou])
        us = int(AIPS['us'][cou])
        doy = int(AIPS['doy'][cou])
        ddV.append(datetime.datetime(2017, 1,1,h,m,s,us) + datetime.timedelta(days=doy-1))
    AIPS['datetime'] = ddV
    return AIPS

def add_datetime_OPT(AIPS):   
    h_m_s_doy = list(zip(AIPS['hour'].astype('int32'),AIPS['min'].astype('int32'),AIPS['sec'].astype('int32'),AIPS['us'].astype('int32'), AIPS['doy'].astype('int32') ))
    AIPS['datetime'] = list(map(lambda x: datetime.datetime(2017,1,1,x[0],x[1],x[2],x[3]) + datetime.timedelta(days=int(x[4])-1) , h_m_s_doy ))
    return AIPS

def add_baseline(AIPS):
    foo=[]
    for cou in range(AIPS.shape[0]):
        first = dictBase[jd2track2017(AIPS['jd'][cou])][AIPS['st1'][cou]]
        second= dictBase[jd2track2017(AIPS['jd'][cou])][AIPS['st2'][cou]]
        foo.append(first+second)
        #print(float(cou)/float(APIS.shape[0]))
    AIPS['baseline']=foo
    return AIPS

def add_baseline_OPT(AIPS):
    jd_st1_st2 = list(zip(AIPS['jd'],AIPS['st1'],AIPS['st2']))
    AIPS['baseline'] = list(map(lambda x: dictBase[jd2track2017(x[0])][x[1]]+dictBase[jd2track2017(x[0])][x[2]] , jd_st1_st2 ))
    return AIPS

def add_source_polar(AIPS,filename):
    sour = filename.split('.')[-3]
    pol = filename.split('.')[-2]
    AIPS['source'] = [sour]*AIPS.shape[0]
    AIPS['polarization'] = [pol]*AIPS.shape[0]
    return AIPS

def add_track_expt(AIPS):
    foo1=[]
    foo2=[]
    for cou in range(AIPS.shape[0]):
        first = jd2track2017(AIPS['jd'][cou])
        second= jd2expt2017(AIPS['jd'][cou])
        foo1.append(first)
        foo2.append(second)
    AIPS['track']=foo1
    AIPS['expt_no']=foo2
    return AIPS

def add_track_expt_OPT(AIPS):
    AIPS['track']=list(map(lambda x: jd2track2017(x), AIPS['jd'] ))
    AIPS['expt_no']=list(map(lambda x: jd2expt2017(x), AIPS['jd'] ))
    return AIPS

def initialize_falist(path_file):
    filename = path_file.split('/')[-1]
    AIPS = pd.read_csv(path_file)
    AIPS = add_baseline_OPT(AIPS)
    AIPS = add_datetime_OPT(AIPS)
    AIPS = add_track_expt_OPT(AIPS)
    AIPS['vis'] = AIPS['amp']*np.exp(1j*AIPS['phase']*np.pi/180)
    AIPS['std'] = AIPS['amp']
    AIPS = AIPS.groupby(('expt_no','track','datetime','baseline')).agg({'vis': np.mean, 'std': np.std, 'sigma': lambda x: np.std(x)/len(x),'u': np.mean, 'v': np.mean})
    AIPS['amp'] = np.abs(AIPS['vis'])
    AIPS['phase'] = np.angle(AIPS['vis'])*180/np.pi
    AIPS = add_source_polar(AIPS,filename)
    AIPS = AIPS.reset_index()
    AIPS = AIPS[['datetime','baseline','source','amp','phase','sigma','std','polarization','track','expt_no','u','v']]
    AIPS = AIPS.sort_values('datetime').reset_index(drop=True)
    return AIPS

def initialize_falist_channels(path_file):

    return path_file

def match_scans(scans,AIPS):
    bins_labels = [None]*(2*scans.shape[0]-1)
    bins_labels[1::2] = map(lambda x: -x-1,list(scans['scan_no_tot'])[:-1])
    bins_labels[::2] = list(scans['scan_no_tot'])
    dtmin = datetime.timedelta(seconds = 2.) 
    dtmax = datetime.timedelta(seconds = 2.) 
    binsT = [None]*(2*scans.shape[0])
    binsT[::2] = list(map(lambda x: x - dtmin,list(scans.time_min)))
    binsT[1::2] = list(map(lambda x: x + dtmax,list(scans.time_max))) 
    ordered_labels = pd.cut(AIPS.datetime, binsT,labels = bins_labels)
    AIPS['scan_no_tot'] = ordered_labels
    AIPS = AIPS[list(map(lambda x: x >= 0, AIPS['scan_no_tot']))]
    return AIPS

def folder_into_falist(folder_path, Vex_path = 'VexFiles/', saveDF = ''):
    '''
    takes all csv files from folder_path
    and generates data frame with frequency-averaged data
    optionally saves to file
    '''
    AIPS_OUT = pd.DataFrame({})
    list_files = os.listdir(folder_path)
    list_files = [x for x in list_files if x[0]!='.']
    #os.chdir(folder_path)
    cou = 1
    print('creating list of scans...')
    scans = make_scan_list(Vex_path)
    for fi in list_files:
        path_file = folder_path+fi
        print(fi+', file '+str(cou)+'/'+str(len(list_files)))
        print('initializing falist...')
        #try:
        #print('tutaj')
        AIPS = initialize_falist(path_file)
        #print('tutaaj2')
        print('matching scans... inputs found: '+str(AIPS.shape[0]))
        AIPS = match_scans(scans,AIPS)
        print('done! inputs processed: '+str(AIPS.shape[0]))
        AIPS_OUT = pd.concat([AIPS_OUT,AIPS], ignore_index=True)
        #except:
        #    continue
        cou += 1
        if saveDF != '':
            AIPS_OUT.to_pickle(saveDF+'.pic')
    return AIPS_OUT




def coh_average(AIPS, tcoh = 5.):
    AIPS['round_time'] = map(lambda x: np.round((x- datetime.datetime(2017,4,4)).total_seconds()/tcoh),AIPS['datetime'])
    AIPS['vis'] = AIPS['vis'] = AIPS['amp']*np.exp(1j*AIPS['phase']*np.pi/180)
    AIPS = AIPS[['datetime','baseline','source','polarization','vis','std','sigma','track','expt_no','scan_no_tot','round_time']]
    AIPS = AIPS.groupby(('baseline','source','polarization','track','expt_no','scan_no_tot','round_time')).agg({'datetime': 'min', 'vis': np.mean, 'sigma': lambda x: np.sqrt(np.sum(x**2))/len(x), 'std': lambda x: np.sqrt(np.sum(x**2)/len(x))  })
    #std after agg here is a sample std on 1s segment, 1 frequency channel
    AIPS = AIPS.reset_index()
    AIPS['amp'] = np.abs(AIPS['vis'])
    AIPS['phase'] = np.angle(AIPS['vis'])*180/np.pi
    AIPS = AIPS[['datetime','baseline','source','polarization','amp', 'phase','std', 'sigma','track','expt_no','scan_no_tot']]
    return AIPS



def add_scanid(APIS,begtime,endtime,label):
    APIS['scan_id'] = [0]*APIS.shape[0]
    TotalScans = len(endtime)
    for x in range(TotalScans):
        foo = APIS.loc[(APIS['datetime']>=begtime[x])&(APIS['datetime']<=endtime[x]),['scan_id']]
        APIS.loc[(APIS['datetime']>=begtime[x])&(APIS['datetime']<=endtime[x]),['scan_id']] = label[x]    
    return APIS

def add_time_bucket_ind(APIS,deltaT):
    Tstart = min(APIS['datetime'])
    Tend = max(APIS['datetime'])
    TotalDelta = (max(APIS['datetime'])-min(APIS['datetime'])).seconds
    NumBuckets = int(np.floor(float(TotalDelta)/float(deltaT)))+1
    begTB = [Tstart + datetime.timedelta(seconds=deltaT)*x for x in range(NumBuckets)]
    endTB = [Tstart + datetime.timedelta(seconds=deltaT)*x for x in range(1,NumBuckets+1)]
    APIS['time_bucket'] = [-20]*APIS.shape[0]
    TotalScans = len(endTB)
    for x in range(TotalScans):
        foo = APIS.loc[(APIS['datetime']>=begTB[x])&(APIS['datetime']<endTB[x]),['time_bucket']]
        APIS.loc[(APIS['datetime']>=begTB[x])&(APIS['datetime']<endTB[x]),['time_bucket']] = [x]*foo.shape[0]    
    return APIS

def read_until_end_of_scan(APIS,start=0):
    cou = start-1
    listScans = []
    while (cou < APIS.shape[0]-3):
    #while (cou<14000):
        cou += 1
        foo = pd.DataFrame(APIS.iloc[cou:cou+1])
    
        while (APIS['datetime'][cou+1] - APIS['datetime'][cou] < datetime.timedelta(seconds=20)):
            foo = foo.append(APIS.iloc[cou+1])
            cou += 1
            if (cou == APIS.shape[0]-1):
                break
            #print(cou)
        listScans.append(foo)     
        print(cou,len(listScans))
    return listScans

def make_into_2d_table(APIS):
    NumCh = max(APIS['ifid'])+1
    NumT = (max(APIS['datetime']) - min(APIS['datetime'])).seconds + 1
    TableAmp = np.zeros((NumCh,NumT))
    TablePha = np.zeros((NumCh,NumT))
    Tlist = range(NumT)
    foo = list(set(listAX[0]['datetime']))
    foo.sort()
    DictDatetime2Tind = dict(zip(foo, Tlist))
    for cou in range(APIS.shape[0]):
        Ch = int(APIS['ifid'][cou])
        Tind = DictDatetime2Tind[APIS['datetime'][cou]]
        TableAmp[Ch,Tind] = APIS['amp'][cou]
        TablePha[Ch,Tind] = APIS['phase'][cou]
    return TableAmp, TablePha

def remove_outliers(vec,sigma=6):
    m = np.mean(vec)
    s = np.std(vec)
    vecInd = np.abs(vec - m) < sigma*s
    vecNew = vec[vecInd]
    return vecNew, vecInd
    
    
def bootstrap(data, num_samples, statistic, alpha=0.05):
    """Returns bootstrap estimate of 100.0*(1-alpha) CI for statistic."""
    data = np.asarray(data)
    n = len(data)
    idx = npr.randint(0, n, (num_samples, n))
    samples = data[idx]
    stat = np.sort(statistic(samples, 1))
    return (np.median(stat),stat[int((alpha/2.0)*num_samples)],
            stat[int((1-alpha/2.0)*num_samples)])
    #return stat[int(num_samples/2.)]

def bootstrap_std(data, num_samples=10000):
    data = np.asarray(data)
    statistic = np.std
    n = len(data)
    idx = npr.randint(0, n, (num_samples, n))
    samples = data[idx]
    stat = np.sort(statistic(samples, 1))
    return stat[int(num_samples/2.)]

def APIS_coherent_average(APISframe, t_coh):
    
    APIS_groups_baseline_scanid = APISframe.groupby(('polarization','source','expt_no','baseline','scan_id'))
    APIS_new_list = []
    for name, group in APIS_groups_baseline_scanid:
        print(name)
        #t_coh = 5 #time of coherent averaging in seconds
        foo = add_time_bucket_ind(group,t_coh)
        #print(foo.columns)
        #print(foo)
        foo = foo[['datetime','time_bucket','vis','gmst']]
        foo['boot_std'] = foo['vis']
        foo = foo.groupby('time_bucket').agg({'vis': np.average, 'datetime': 'min','boot_std': bootstrap_std,'gmst': 'min'})
       
        #foo = foo.groupby('time_bucket').agg({'vis': np.average,'boot_std': bootstrap_std})
       
        foo['polarization'] = [list(group.polarization)[0]]*foo.shape[0]
        foo['source'] = [list(group.source)[0]]*foo.shape[0]
        foo['expt_no'] = [list(group.expt_no)[0]]*foo.shape[0]
        foo['baseline'] = [list(group.baseline)[0]]*foo.shape[0]
        foo['scan_id'] = [list(group.scan_id)[0]]*foo.shape[0]
        #print([list(group.scan_id)[0], min(group['gmst']), max(group['gmst']) ])
        print([foo['scan_id'][0], min(foo['gmst']), max(foo['gmst']) ])
        APIS_new_list.append(foo)
        
    APIS_new = pd.concat(APIS_new_list)
    APIS_new['amp'] = np.abs(APIS_new['vis'])
    APIS_new['total_phas'] = np.angle(APIS_new['vis'])*180./np.pi
    APIS_new['sigma'] = APIS_new['boot_std']/np.sqrt(t_coh*32.)
    APIS_new['snr'] = APIS_new['amp']/APIS_new['sigma']
    
    return APIS_new


def APIS_coherent_average2(APISframe, t_coh):
    #print('dupa dupa')
     
    APISframe = APISframe.groupby(('polarization','baseline','datetime','source','expt_no','scan_id')).agg({'vis': np.average, 'gmst': 'min'})
    APISframe = APISframe.reset_index()

    APIS_groups_baseline_scanid = APISframe.groupby(('polarization','source','expt_no','baseline','scan_id'))
    APIS_new_list = []
    for name, group in APIS_groups_baseline_scanid:
        print(name)
        #t_coh = 5 #time of coherent averaging in seconds
        foo = add_time_bucket_ind(group,t_coh)
        #print(foo.columns)
        #print(foo)
        foo = foo[['datetime','time_bucket','vis','gmst']]
        foo['boot_std'] = foo['vis']
        foo = foo.groupby('time_bucket').agg({'vis': np.average, 'datetime': 'min','boot_std': 'std','gmst': 'min'})
       
        #foo = foo.groupby('time_bucket').agg({'vis': np.average,'boot_std': bootstrap_std})
       
        foo['polarization'] = [list(group.polarization)[0]]*foo.shape[0]
        foo['source'] = [list(group.source)[0]]*foo.shape[0]
        foo['expt_no'] = [list(group.expt_no)[0]]*foo.shape[0]
        foo['baseline'] = [list(group.baseline)[0]]*foo.shape[0]
        foo['scan_id'] = [list(group.scan_id)[0]]*foo.shape[0]
        #print([list(group.scan_id)[0], min(group['gmst']), max(group['gmst']) ])
        print([foo['scan_id'][0], min(foo['gmst']), max(foo['gmst']) ])
        APIS_new_list.append(foo)
        
    APIS_new = pd.concat(APIS_new_list)
    APIS_new['amp'] = np.abs(APIS_new['vis'])
    APIS_new['total_phas'] = np.angle(APIS_new['vis'])*180./np.pi
    APIS_new['sigma'] = APIS_new['boot_std']/np.sqrt(t_coh*32.)
    APIS_new['snr'] = APIS_new['amp']/APIS_new['sigma']
    
    return APIS_new


def APIS_av_freq(APISframe):
    #print('dupa dupa')
     
    APIS_groups_baseline_scanid = APISframe.groupby(('polarization','baseline','datetime'))
    APIS_new_list = []
    for name, group in APIS_groups_baseline_scanid:
        print(name)
        #t_coh = 5 #time of coherent averaging in seconds
        foo = add_time_bucket_ind(group,t_coh)
        #print(foo.columns)
        #print(foo)
        foo = foo[['datetime','time_bucket','vis','gmst']]
        foo['boot_std'] = foo['vis']
        foo = foo.groupby('time_bucket').agg({'vis': np.average, 'datetime': 'min','boot_std': 'std','gmst': 'min'})
       
        #foo = foo.groupby('time_bucket').agg({'vis': np.average,'boot_std': bootstrap_std})
       
        foo['polarization'] = [list(group.polarization)[0]]*foo.shape[0]
        foo['source'] = [list(group.source)[0]]*foo.shape[0]
        foo['expt_no'] = [list(group.expt_no)[0]]*foo.shape[0]
        foo['baseline'] = [list(group.baseline)[0]]*foo.shape[0]
        foo['scan_id'] = [list(group.scan_id)[0]]*foo.shape[0]
        #print([list(group.scan_id)[0], min(group['gmst']), max(group['gmst']) ])
        print([foo['scan_id'][0], min(foo['gmst']), max(foo['gmst']) ])
        APIS_new_list.append(foo)
        
    APIS_new = pd.concat(APIS_new_list)
    APIS_new['amp'] = np.abs(APIS_new['vis'])
    APIS_new['total_phas'] = np.angle(APIS_new['vis'])*180./np.pi
    APIS_new['sigma'] = APIS_new['boot_std']/np.sqrt(t_coh*32.)
    APIS_new['snr'] = APIS_new['amp']/APIS_new['sigma']
    
    return APIS_new

def prepare_APIS_data(path,pol,begt, endt, label=[], sourc='OJ287'):
    if not label:
        label = range(len(endt))
    APIS = pd.read_csv(path)    
    APIS = add_datetime(APIS);
    APIS = add_scanid(APIS,begt,endt,label);
    APIS['vis'] =APIS['amp']*np.exp(1j*(np.pi/180.)*APIS['phase'])
    APIS['source'] = [sourc]*APIS.shape[0]
    APIS['expt_no'] = [3597]*APIS.shape[0]
    APIS['polarization'] = [pol]*APIS.shape[0]
    APIS['total_phas'] = APIS['phase']
    util.add_gmst(APIS)
    APIS = add_baseline(APIS)
    
    return APIS


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


#get list of scans and their times from hops alist
def list_of_scans(alist_path, deltat_sec = 0):
    #prepare dataframe of scans   
    columns=['expt_no', 'scan_id', 'time_min','time_max','source','elevZ','elevP','antenas']
    scans = pd.DataFrame(columns=columns)
    a0 = hops.read_alist(alist_path)
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

#to read fake alist from csvs
def read_falist_csv(path,expt=3597):
    falist = pd.read_csv(path)
    falist['datetime'] = map(pd._libs.tslib.Timestamp,falist.datetime)
    falist['vis'] = map(complex,falist.vis)
    falist['expt_no'] = [expt]*falist.shape[0]
    return falist


def add_baselineold(APIS):
    APIS['baseline'] = [0]*APIS.shape[0]
    for cou in range(APIS.shape[0]):   
        APIS['baseline'][cou] = dictBase[APIS['st1'][cou]]+dictBase[APIS['st2'][cou]]
        #print(float(cou)/float(APIS.shape[0]))
    return APIS

