from __future__ import print_function
import sys, os, datetime, itertools
import scipy.special as ss
import scipy.optimize as so
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from eat.io import hops, util
from eat.hops import util as hu
from eat.aips import aips2alist as a2a
from eat.inspect import utils as ut

hrs = [0,24.,48.]

def make_baselines_alphabetic(alist,what_phase='resid_phas'):

    baseL = list(set(alist.baseline))
    alist_out = pd.DataFrame({})
    for base in baseL:
        foo = alist.loc[alist['baseline']==base,:]
        if base[1]<base[0]:
            new_base = base[1]+base[0]
            foo.loc[:,what_phase] = -foo[what_phase]
            foo.loc[:,'baseline'] = [new_base]*np.shape(foo)[0]
        alist_out = pd.concat([alist_out, foo])
    alist_out = alist_out.reset_index()
    return alist_out



def list_all_triangles(alist):
    all_baselines = set(alist.baseline)
    if all(['-' in x for x in all_baselines]):
        all_stations = set( [y for sub in [x.split('-') for x in all_baselines] for y in sub])
    else:
        all_stations = set(''.join( list(all_baselines)))
    foo = list(itertools.combinations(all_stations, 3))
    foo = [list(x) for x in foo if ('R' not in set(x))|('S' not in set(x))]
    foo = [''.join(sorted(x)) for x in foo] 
    return foo

def list_all_quadrangles(alist):
    all_stations = set(''.join( list(set(alist.baseline))))
    foo = list(itertools.combinations(all_stations, 4))
    foo = [set(x) for x in foo if ('R' not in set(x))|('S' not in set(x))]
    foo = [''.join(sorted(x)) for x in foo] 
    return foo

def triangles2baselines(tri,alist):
    all_baselines = set(alist.baseline)
    foo_base = []
    signat = []
    for cou in range(len(tri)):
        b0 = tri[cou][0:2]
        b1 = tri[cou][1:3]
        b2 = tri[cou][2]+tri[cou][0]
        #print([b0,b1,b2])
        if b0 in all_baselines:
            base0 = b0
            sign0 = 1
        elif b0[1]+b0[0] in all_baselines:
            base0 = b0[1]+b0[0]
            sign0 = -1
        else:
            base0 = -1
            sign0 = 0
            
        if b1 in all_baselines:
            base1 = b1
            sign1 = 1
        elif b1[1]+b1[0] in all_baselines:
            base1 = b1[1]+b1[0]
            sign1 = -1
        else:
            base1 = -1
            sign1 = 0
            
        if b2 in all_baselines:
            base2 = b2
            sign2 = 1
        elif b2[1]+b2[0] in all_baselines:
            base2 = b2[1]+b2[0]
            sign2 = -1
        else:
            base2 = -1
            sign2 = 0
        baselines = [base0,base1,base2]
        
        baselinesSTR = map(lambda x: type(x)==str,baselines)
        if all(baselinesSTR):
            foo_base.append(baselines)
            signat.append([sign0,sign1,sign2])
    return foo_base, signat


def quadrangles2baselines(quad,alist):
    all_baselines = set(alist.baseline)
    foo_base = []
    for cou in range(len(quad)):
        b0 = quad[cou][0:2]
        b1 = quad[cou][2:4]
        b2 = quad[cou][0]+quad[cou][3]
        b3 = quad[cou][1]+quad[cou][2]
        bo2 = quad[cou][0]+quad[cou][2]
        bo3 = quad[cou][1]+quad[cou][3]

        if b0 in all_baselines: base0 = b0
        elif b0[1]+b0[0] in all_baselines: base0 = b0[1]+b0[0]
        else: base0 = -1

        if b1 in all_baselines: base1 = b1
        elif b1[1]+b1[0] in all_baselines: base1 = b1[1]+b1[0]
        else: base1 = -1

        #square quadrangle
        if b2 in all_baselines: base2 = b2
        elif b2[1]+b2[0] in all_baselines: base2 = b2[1]+b2[0]
        else: base2 = -1

        if b3 in all_baselines: base3 = b3
        elif b3[1]+b3[0] in all_baselines: base3 = b3[1]+b3[0]
        else: base3 = -1

        #bow quadrangle
        if bo2 in all_baselines: base2bo = bo2
        elif bo2[1]+bo2[0] in all_baselines: base2bo = bo2[1]+bo2[0]
        else: base2bo = -1

        if bo3 in all_baselines: base3bo = bo3
        elif bo3[1]+bo3[0] in all_baselines: base3bo = bo3[1]+bo3[0]
        else: base3bo = -1
            
        baselines = [base0,base1,base2,base3]
        baselinesbo = [base0,base1,base2bo,base3bo]
        baselinesx = [base2bo,base3bo,base2,base3]
        
        baselinesSTR = list(map(lambda x: type(x)==str,baselines))
        baselinesboSTR = list(map(lambda x: type(x)==str,baselinesbo))
        baselinesxSTR = list(map(lambda x: type(x)==str,baselinesx))
        if all(baselinesSTR):
            foo_base.append(baselines)
        if all(baselinesboSTR):
            foo_base.append(baselinesbo)
        if all(baselinesxSTR):
            foo_base.append(baselinesx)
    return foo_base

def quadrangle2str(quad):
    quad = list(quad)
    A = quad[0][0]
    B = quad[0][1]
    if A in quad[2]:
        C = quad[2].replace(A,'')
        D = quad[3].replace(B,'')
    elif A in quad[3]:
        C = quad[3].replace(A,'')
        D = quad[2].replace(B,'')
    return quad[0]+C+D

def str2quadrangle(quad_str):
    return [quad_str[:2],quad_str[2:4], quad_str[0]+quad_str[2],quad_str[1]+quad_str[3]]
  


def baselines2triangles(basel):
    tri = [''.join(sorted(list(set(''.join(x))))) for x in basel]
    return tri

def all_bispectra(alist,phase_type='resid_phas',debias_snr=False,match_by_scan=False,verbose=False):
    polars=[]
    if phase_type in alist.columns:
        try:
            bsp_LL = all_bispectra_polar(alist,'LL',phase_type,debias_snr=debias_snr,match_by_scan=match_by_scan,verbose=verbose)
            polars=polars+['LL']
        except: pass
        try:
            bsp_RR = all_bispectra_polar(alist,'RR',phase_type,debias_snr=debias_snr,match_by_scan=match_by_scan,verbose=verbose)
            polars=polars+['RR']
        except: pass
        if verbose: print('polarz', polars)
        if len(polars)==2:
            bsp = pd.concat([bsp_LL,bsp_RR],ignore_index=True)
        elif polars[0]=='LL':
            bsp=bsp_LL
        else: bsp = bsp_RR
        return bsp
    else:
        print('Wrong name for the phase column!')

def all_bispectra_polar(alist,polar,phase_type='resid_phas',snr_cut=0.,debias_snr=False,match_by_scan=False,verbose=False):
    '''
    match_by_scan: option to only use scan_id rather than timestamp to find triplets of phases to form closure phases
    should only be used for scan-averaged data
    '''
    if match_by_scan:
        alist=ut.coh_avg_vis(alist,tavg='scan',phase_type=phase_type)
    if 'snr' not in alist.columns:
        alist.loc[:,'snr'] = alist.loc[:,'amp']/alist.loc[:,'sigma']
    if debias_snr==True:
        foo = np.maximum(np.asarray(alist['snr'])**2 - 1,0)
        alist['snr'] = np.sqrt(foo)
    alist.drop(list(alist[alist.snr<snr_cut].index.values),inplace=True)
    #print(alist)
    alist = alist[alist['polarization']==polar]
    alist = alist.loc[:,~alist.columns.duplicated()]
    
    if 'scan_id' not in alist.columns:
        alist.loc[:,'scan_id'] = alist.loc[:,'scan_no_tot']
    if 'band' not in alist.columns:
        alist.loc[:,'band'] = np.nan
    
    alist['amps']=alist['amp']
    alist['snrs']=alist['snr']
    
    if 'fracpol' in alist.columns:
        #print(alist.columns)
        alist['fracpols']=alist['fracpol']
    else: alist['fracpols'] = 0
    triL = list_all_triangles(alist)
    #print(triL)
    tri_baseL, sgnL = triangles2baselines(triL,alist)
    #this is done twice to remove some non-present triangles
    #print(triL, tri_baseL)
    triL = baselines2triangles(tri_baseL)
    tri_baseL, sgnL = triangles2baselines(triL,alist)
    #print(tri_baseL)
    #print(sgnL)
    bsp_out = pd.DataFrame({})
    #print('wtf?',triL, tri_baseL)
    #triL=sorted(triL)
    for cou in range(len(triL)):
        
        Tri = tri_baseL[cou]
        if verbose: print(Tri)
        signat = sgnL[cou]
        #print(Tri)
        condB1 = (alist['baseline']==Tri[0])
        condB2 = (alist['baseline']==Tri[1])
        condB3 = (alist['baseline']==Tri[2])
        condB = condB1|condB2|condB3
        alist_Tri = alist.loc[condB,['expt_no','scan_id','source','datetime','baseline',phase_type,'amp','snr','gmst','band','amps','snrs','fracpols']]
        #print(alist_Tri)
        #print(alist_Tri)
        #print(np.shape(alist_Tri))
        #throw away times without full triangle
        if match_by_scan:
            tlist = alist_Tri.groupby(['band','scan_id']).filter(lambda x: len(x) == 3)
        else:
            tlist = alist_Tri.groupby(['band','datetime']).filter(lambda x: len(x) == 3)
        tlist.loc[:,'sigma'] = (tlist.loc[:,'amp']/(tlist.loc[:,'snr']))
        #print(tlist.loc[:,phaseType])
        #print(tlist)
        
        for cou2 in range(3):
            tlist.loc[(tlist.loc[:,'baseline']==Tri[cou2]),phase_type] *= signat[cou2]*np.pi/180.
        tlist.loc[:,'sigma'] = 1./tlist.loc[:,'snr']**2 #put 1/snr**2 in the sigma column to aggregate
        #print(tlist.columns)
        if match_by_scan:
            bsp = tlist.groupby(['expt_no','source','band','scan_id']).agg({phase_type: lambda x: np.sum(x),'amp': lambda x: np.prod(x), 'sigma': lambda x: np.sqrt(np.sum(x)),
        'amps': lambda x: tuple(x),'snrs': lambda x: tuple(x),'fracpols': lambda x: tuple(x),'datetime':min})  
        else:
            bsp = tlist.groupby(['expt_no','source','band','scan_id','datetime'], observed=True).agg({phase_type: lambda x: np.sum(x),'amp': lambda x: np.prod(x), 'sigma': lambda x: np.sqrt(np.sum(x)),
        'amps': lambda x: tuple(x),'snrs': lambda x: tuple(x),'fracpols': lambda x: tuple(x)})
        #sigma above is the CLOSURE PHASE ERROR
        #print(bsp.columns)
        
        bsp.loc[:,'bsp'] = bsp.loc[:,'amp']*np.exp(1j*bsp.loc[:,phase_type])
        bsp.loc[:,'snr'] = 1./bsp.loc[:,'sigma']
        bsp.loc[:,'sigma'] = bsp.loc[:,'amp']*bsp.loc[:,'sigma'] #sigma of bispectrum
        bsp.loc[:,'triangle'] = [triL[cou]]*np.shape(bsp)[0]
        bsp.loc[:,'polarization'] = [polar]*np.shape(bsp)[0]
        #bsp.loc[:,'signature'] = [signat]*np.shape(bsp)[0]
        bsp.loc[:,'cphase'] = np.angle(bsp.loc[:,'bsp'])*180./np.pi
        bsp.loc[:,'amp'] = np.abs(bsp.loc[:,'bsp'])
        #bsp.loc[:,'snr'] = bsp.loc[:,'amp']/bsp.loc[:,'sigma']
        bsp.loc[:,'sigmaCP'] = 1./bsp.loc[:,'snr']*180./np.pi #deg
        bsp_out = pd.concat([bsp_out, bsp])
        if verbose: print(triL[cou]+': '+str(np.shape(bsp)[0])+' closure phases')
    #print(bsp_out.columns)
    bsp_out = bsp_out.reset_index()
    #print(bsp_out.columns)
    #try:
    #    bsp_out = bsp_out[['datetime','source','triangle','polarization','cphase','sigmaCP','amp','sigma','snr','scan_id','expt_no','band','amps','snrs','fracpols']] 
    #except KeyError:
    #   pass
    
    bsp_out['rel_err'] = np.asarray(bsp_out['cphase'])/np.asarray(bsp_out['sigmaCP'])
    return bsp_out


def all_bispectra_polar_scan_MC(alist,polar,phaseType='resid_phas'):
    alist = alist[alist['polarization']==polar]
    alist = alist.loc[:,~alist.columns.duplicated()]
    if 'scan_id' not in alist.columns:
        alist.loc[:,'scan_id'] = alist.loc[:,'scan_no_tot']
    if 'band' not in alist.columns:
        alist.loc[:,'band'] = [None]*np.shape(alist)[0]

    triL = list_all_triangles(alist)
    tri_baseL, sgnL = triangles2baselines(triL,alist)
    #this is done twice to remove some non-present triangles
    triL = baselines2triangles(tri_baseL)
    tri_baseL, sgnL = triangles2baselines(triL,alist)
    bsp_out = pd.DataFrame({})
    for cou in range(len(triL)):
        Tri = tri_baseL[cou]
        signat = sgnL[cou]
        condB1 = (alist['baseline']==Tri[0])
        condB2 = (alist['baseline']==Tri[1])
        condB3 = (alist['baseline']==Tri[2])
        condB = condB1|condB2|condB3
        alist_Tri = alist.loc[condB,['expt_no','scan_id','source','datetime','baseline',phaseType,'unbiased_amp','snr','gmst','band']]
        
        #throw away times without full triangle
        tlist = alist_Tri.groupby(['band','datetime']).filter(lambda x: len(x) > 2)
        tlist.loc[:,'sigma'] = (tlist.loc[:,'amp']/(tlist.loc[:,'snr']))

        for cou2 in range(3):
            tlist.loc[(tlist.loc[:,'baseline']==Tri[cou2]),phaseType] *= signat[cou2]*np.pi/180.
        tlist.loc[:,'sigma'] = 1./tlist.loc[:,'snr']**2 #put 1/snr**2 in the sigma column to aggregate
        
        bsp = tlist.groupby(['expt_no','source','band','scan_id','datetime']).agg({phaseType: lambda x: np.sum(x),'amp': lambda x: np.prod(x), 'sigma': lambda x: np.sqrt(np.sum(x))})
        #sigma above is the CLOSURE PHASE ERROR
        
        bsp.loc[:,'bisp'] = bsp.loc[:,'amp']*np.exp(1j*bsp.loc[:,phaseType])
        bsp.loc[:,'snr'] = 1./bsp.loc[:,'sigma']
        bsp.loc[:,'sigma'] = bsp.loc[:,'amp']*bsp.loc[:,'sigma'] #sigma of bispectrum
        bsp.loc[:,'triangle'] = [triL[cou]]*np.shape(bsp)[0]
        bsp.loc[:,'polarization'] = [polar]*np.shape(bsp)[0]
        #bsp.loc[:,'signature'] = [signat]*np.shape(bsp)[0]
        bsp.loc[:,'cphase'] = np.angle(bsp.loc[:,'bisp'])*180./np.pi
        bsp.loc[:,'amp'] = np.abs(bsp.loc[:,'bisp'])
        bsp.loc[:,'snr'] = bsp.loc[:,'amp']/bsp.loc[:,'sigma']
        bsp.loc[:,'sigmaCP'] = 1./bsp.loc[:,'snr']*180./np.pi #deg
        bsp_out = pd.concat([bsp_out, bsp])
    bsp_out = bsp_out.reset_index()
    #print(bsp_out.columns)
    bsp_out = bsp_out[['datetime','source','triangle','polarization','cphase','sigmaCP','amp','sigma','snr','scan_id','expt_no','band']] 
    
    return bsp_out

def only_trivial_triangles(bsp,whichB = 'all'):
    if whichB =='AX':
        condTri = list(map(lambda x: (('A' in x)&('X' in x)), bsp['triangle']))
    elif whichB =='JS':
        condTri = list(map(lambda x: (('J' in x)&('S' in x)), bsp['triangle']))
    elif whichB =='JR':
        condTri = list(map(lambda x: (('J' in x)&('R' in x)), bsp['triangle']))
    else:
        condTri = list(map(lambda x: (('A' in x)&('X' in x))|(('J' in x)&('S' in x))|(('J' in x)&('R' in x)), bsp['triangle']))
    bsp = bsp[condTri]
    return bsp

def only_non_trivial_triangles(bsp):
    condTri = list(map(lambda x: (('A' not in x)|('X' not in x))&(('J' not in x)|('S' not in x))&(('J' not in x)|('R' not in x)), bsp['triangle']))
    bsp = bsp[condTri]
    return bsp

def all_quadruples_log(alist):
    quad=pd.DataFrame({})
    try:
        quad_LL =all_quadruples_polar_log(alist,'LL')
        #print('shape LL', np.shape(quad_LL))
        quad = pd.concat([quad,quad_LL],ignore_index=True)
    except: pass
    #print('shape add LL', np.shape(quad))
    try:
        quad_RR =all_quadruples_polar_log(alist,'RR')
        quad = pd.concat([quad,quad_RR],ignore_index=True)
    except: pass
    return quad

def debias_A_sig(A,sigma):
    '''
    given vector of Rice-distributed amplitudes A and 
    associated thermal errors sigma, get A0
    '''
    A = np.asarray(A)
    sigma=np.asarray(sigma)
    import scipy.optimize as so
    rho = A/sigma
    fun = lambda x: np.mean(np.log(rho)) - (np.log(x) - 0.5* ss.expi(-x**2/2.))
    x1 = 0.057967 #min real snr
    x2 = 30. #max real snr
    #print(fun(x1),fun(x2))
    rho0 = so.brentq(fun,x1,x2)
    A0 = rho0*np.mean(sigma)
    return A0

#def all_quads(alist,tavg='scan',debias=True):


def all_quadruples_new(alist,ctype='camp',debias='no',debias_snr=False,match_by_scan=False,verbose=False):
    '''
    This one used!
    ctype = 'camp' or 'logcamp'
    debias = 'no'/whatever , 'amp', 'camp'
    '''
    if match_by_scan:
        if 'phase' in alist.columns:
            alist=ut.coh_avg_vis(alist,tavg='scan',phase_type='phase')
        elif 'resid_phas' in alist.columns:
             alist=ut.coh_avg_vis(alist,tavg='scan',phase_type='resid_phas')
        else: pass

    alist = alist[(alist['polarization']=='LL')|(alist['polarization']=='RR')|(alist['polarization']=='I')]
    if debias_snr==True:
        foo = np.maximum(np.asarray(alist['snr'])**2 - 1,0)
        alist['snr'] = np.sqrt(foo)
    if 'band' not in alist.columns:
        alist.loc[:,'band'] = 'unknown'
    if 'scan_id' not in alist.columns:
        alist.loc[:,'scan_id'] = alist.loc[:,'scan_no_tot']
    if 'sigma' not in alist.columns:
        alist['sigma'] = alist['amp']/alist['snr']
    quaL = list_all_quadrangles(alist)
    quad_baseL = sorted(quadrangles2baselines(quaL,alist))
    quad_out = pd.DataFrame({})
    alist['amps'] = alist['amp']
    if debias=='camp':
        alist['snr'] = get_snr(alist['snr'])
    alist['snrs'] = alist['snr']
    alist['snr1'] = alist['snr']
    alist['snr2'] = alist['snr']
    alist['snr3'] = alist['snr']
    alist['snr4'] = alist['snr']

    if debias=='amp':
        alist['amp'] = np.sqrt(np.maximum(0, alist['amp']**2 - alist['sigma']**2))

    #select quadrangle
    for cou in range(len(quad_baseL)):
        Quad = quad_baseL[cou]
        condB0 = (alist['baseline']==Quad[0])
        condB1 = (alist['baseline']==Quad[1])
        condB2 = (alist['baseline']==Quad[2])
        condB3 = (alist['baseline']==Quad[3])
        condB = condB0|condB1|condB2|condB3
        alist_Quad = alist.loc[condB,['expt_no','scan_id','source','datetime','baseline','polarization','amp','snr','gmst','band','amps','snrs','snr1','snr2','snr3','snr4','sigma']]
        if verbose: print(Quad, np.shape(alist_Quad)[0])
        #throw away times without full quadrangle
        if match_by_scan:
            tlist = alist_Quad.groupby(['polarization','band','scan_id']).filter(lambda x: len(x) == 4)
        else:
            tlist = alist_Quad.groupby(['polarization','band','datetime']).filter(lambda x: len(x) == 4)
        tlist['snr1'] = tlist['snr1']*(tlist['baseline']==Quad[0])
        tlist['snr2'] = tlist['snr2']*(tlist['baseline']==Quad[1])
        tlist['snr3'] = tlist['snr3']*(tlist['baseline']==Quad[2])
        tlist['snr4'] = tlist['snr4']*(tlist['baseline']==Quad[3])
    
        # prepare amplitudes (last two go into denominator)
        tlist['camp'] = 0
        power=[1,1,-1,-1]
        for cou2 in range(4):
            tlist.loc[(tlist.loc[:,'baseline']==Quad[cou2]),'camp'] = (tlist.loc[(tlist.loc[:,'baseline']==Quad[cou2]),'amp'])**power[cou2]

        if match_by_scan:
            grouping =  ['expt_no','band','polarization','source','scan_id']
            aggregate = {'amps': lambda x: tuple(x),'datetime': 'min', 'snrs': lambda x: tuple(x), 'sigmaCA': lambda x: np.sqrt(np.sum(x**2)),
        'snr1': 'sum', 'snr2': 'sum','snr3': 'sum', 'snr4': 'sum'}
        else:
            grouping =  ['expt_no','band','polarization','source','scan_id','datetime']
            aggregate = {'amps': lambda x: tuple(x), 'snrs': lambda x: tuple(x), 'sigmaCA': lambda x: np.sqrt(np.sum(x**2)),
        'snr1': 'sum', 'snr2': 'sum','snr3': 'sum', 'snr4': 'sum'}

        if ctype=='camp':
            tlist['sigmaCA'] = tlist['sigma']/tlist['amp'] # 1/snr
            aggregate['camp'] = np.prod
            
        elif ctype=='logcamp':
            tlist['sigmaCA'] = tlist['sigma']/tlist['amp'] # 1/snr
            aggregate['camp'] = lambda x: np.sum(np.log(x))
    
        #actual formation of camp
        #print(tlist.columns)
        quadlist = tlist.groupby(grouping, observed=True).agg(aggregate)

        #if camp we need to multiply sigmaCA by CA
        if ctype=='camp':
            quadlist['sigmaCA'] = quadlist['sigmaCA']*quadlist['camp']

        # debiasing in logcamp
        if debias=='camp':
           
            if ctype=='camp':
                quadlist['camp'] = quadlist['camp']*np.exp( - log_debias(quadlist['snr1']) - log_debias(quadlist['snr2']) + log_debias(quadlist['snr3']) + log_debias(quadlist['snr4']) )
            elif ctype=='logcamp':
                quadlist['camp'] = quadlist['camp'] - log_debias(quadlist['snr1']) - log_debias(quadlist['snr2']) + log_debias(quadlist['snr3']) + log_debias(quadlist['snr4'])
        
        #print(quadlist.columns)
        quadlist = quadlist.reset_index()
        quadlist['quadrangle'] = quadrangle2str(Quad)
        quad_out = pd.concat([quad_out, quadlist],ignore_index=True)

    quad_out = quad_out.reset_index(drop=True)
    return quad_out

def all_quadruples_polar_log(alist,polar):
    alist = alist[alist['polarization']==polar]
    if 'band' not in alist.columns:
        alist.loc[:,'band'] = 'unknown'
    if 'scan_id' not in alist.columns:
        alist.loc[:,'scan_id'] = alist.loc[:,'scan_no_tot']
    quaL = list_all_quadrangles(alist)
    quad_baseL = quadrangles2baselines(quaL,alist)
    quad_out = pd.DataFrame({})
    alist['amps'] = alist['amp']
    alist['snrs'] = alist['snr']
    #alist.loc[:,'log_amp'] = np.log(alist.loc[:,'amp'])
    for cou in range(len(quad_baseL)):
    #loop over quadrangles
    #for cou in range(2):
        Quad = quad_baseL[cou]
        condB0 = (alist['baseline']==Quad[0])
        condB1 = (alist['baseline']==Quad[1])
        condB2 = (alist['baseline']==Quad[2])
        condB3 = (alist['baseline']==Quad[3])
        condB = condB0|condB1|condB2|condB3
        alist_Quad = alist.loc[condB,['expt_no','scan_id','source','datetime','baseline','amp','snr','gmst','band','amps','snrs']]
        print(Quad)
        #throw away times without full quadrangle
        tlist = alist_Quad.groupby(['band','datetime']).filter(lambda x: len(x) == 4)

        ###MONTE CARLO VARIATION
        #foo = list(zip(tlist.loc[:,'snr'],tlist.loc[:,'amp']))
        #foo = [list(elem) for elem in foo]
        #tlist['sigma'] = foo
        ####

        ###STANDARD FORMULA FOR VARIATION
        tlist.loc[:,'sigma'] = 1./tlist.loc[:,'snr']**2 #put 1/snr**2 in the sigma column to aggregate
    
        #print(foo)
        #tlist.loc[:,'sigma'] = (tlist.loc[:,'amp']/(tlist.loc[:,'snr']))
        power=[1,1,-1,-1]
        for cou2 in range(4):
            tlist.loc[(tlist.loc[:,'baseline']==Quad[cou2]),'amp'] = (tlist.loc[(tlist.loc[:,'baseline']==Quad[cou2]),'amp'])**power[cou2]
        tlist.loc[:,'logamp'] = tlist.loc[:,'amp']
        #print((foo))
        #print(np.shape(tlist))
        
        quadlist_group = tlist.groupby(['expt_no','band','source','scan_id','datetime'])
        
        ###STANDARD FORMULA FOR VARIATION
        quadlist = quadlist_group.agg({'logamp': lambda x: np.sum(np.log(x)),'sigma': lambda x: np.sqrt(np.sum(x)),
        'amps': lambda x: tuple(x), 'snrs': lambda x: tuple(x) })
        quadlist = quadlist.reset_index()
        #print(quadlist)
        ###MONTE CARLO VARIATION
        #quadlist = quadlist_group.agg({'logamp': lambda x: np.sum(np.log(x)),'sigma': lambda x: log_camp_sigma(1,x) })
        ####

        #sigma above is the CLOSURE PHASE ERROR
        #print(bsp.columns)
        ##bsp.loc[:,'bisp'] = bsp.loc[:,'amp']*np.exp(1j*bsp.loc[:,phaseType])
        ##bsp.loc[:,'snr'] = 1./bsp.loc[:,'sigma']
        #print(tuple(quad_baseL[cou]))
        quadlist['quadrangle'] = [tuple(quad_baseL[cou])]*np.shape(quadlist)[0]

        quadlist['polarization'] = polar
        #bsp.loc[:,'signature'] = [signat]*np.shape(bsp)[0]
        ##bsp.loc[:,'cphase'] = np.angle(bsp.loc[:,'bisp'])*180./np.pi
        #quadlist.loc[:,'logamp'] = np.log(quadlist.loc[:,'amp'])
        ##bsp.loc[:,'snr'] = bsp.loc[:,'amp']/bsp.loc[:,'sigma']
        ##bsp.loc[:,'sigmaCP'] = 1./bsp.loc[:,'snr']*180./np.pi #deg
        quad_out = pd.concat([quad_out, quadlist])
    quad_out = quad_out.reset_index()
    #print(quad_out)
    print(quad_out.columns)
    quad_out = quad_out[['datetime','band','source','quadrangle','polarization','logamp','sigma','scan_id','expt_no']] 
    #quad_out['quadrangle'] = list(map(tuple,quad_out.quadrangle))
    quad_out['quadrangle'] = list(map(quadrangle2str, quad_out.quadrangle))
    return quad_out

def only_trivial_quadrangles_str(quad, whichB='all'):
    
    if whichB =='AX':
        condQuad= list(map(lambda x: ((x[0]=='A')&(x[3]=='X'))|((x[0]=='X')&(x[3]=='A'))|((x[1]=='A')&(x[2]=='X'))|((x[1]=='X')&(x[2]=='A')) ,quad.quadrangle))
    elif whichB =='JS':
        condQuad=list(map(lambda x: ((x[0]=='J')&(x[3]=='S'))|((x[0]=='S')&(x[3]=='J'))|((x[1]=='J')&(x[2]=='S'))|((x[1]=='S')&(x[2]=='J')) ,quad.quadrangle))
    elif whichB =='JR':
        condQuad= list(map(lambda x: ((x[0]=='J')&(x[3]=='R'))|((x[0]=='R')&(x[3]=='J'))|((x[1]=='J')&(x[2]=='R'))|((x[1]=='R')&(x[2]=='J')) ,quad.quadrangle))
    else:
        condAX= list(map(lambda x: ((x[0]=='A')&(x[3]=='X'))|((x[0]=='X')&(x[3]=='A'))|((x[1]=='A')&(x[2]=='X'))|((x[1]=='X')&(x[2]=='A')) ,quad.quadrangle))
        condJS= list(map(lambda x: ((x[0]=='J')&(x[3]=='S'))|((x[0]=='S')&(x[3]=='J'))|((x[1]=='J')&(x[2]=='S'))|((x[1]=='S')&(x[2]=='J')) ,quad.quadrangle))
        condJR= list(map(lambda x: ((x[0]=='J')&(x[3]=='R'))|((x[0]=='R')&(x[3]=='J'))|((x[1]=='J')&(x[2]=='R'))|((x[1]=='R')&(x[2]=='J')) ,quad.quadrangle))
        condQuad = np.asarray(condAX)+np.asarray(condJS)+np.asarray(condJR)
    quad = quad[condQuad]
    return quad

def only_nontrivial_quadrangles(quad):
    
    condAX= list(map(lambda x: ('AX' in x)|('A' not in ''.join(x))|('X' not in ''.join(x)), quad.quadrangle))
    condJS= list(map(lambda x: ('JS' in x)|('J' not in ''.join(x))|('S' not in ''.join(x)), quad.quadrangle))
    condJR= list(map(lambda x: ('JR' in x)|('J' not in ''.join(x))|('R' not in ''.join(x)), quad.quadrangle))
    condQuad = np.asarray(condAX)*np.asarray(condJS)*np.asarray(condJR)
    quad = quad[condQuad]
    return quad

#CALCULATE VARIABILITY IN AMP, LOG AMP
def MCamp(n,snr,A=1.,product='log_amp_sq',Ntot=int(1e5)):   
    #n - how many incoherently averaged amplitudes
    #A - amplitude
    #snr - signal to noise of amplitudes before incoherent averaging
    #We generate Ntot realizations of 2n independent Gaussian variables
    #variance in the final variable is (A/snr)**2 
    # so before averaging 2n variables it was (A/snr)**2 
    #each of them 
    GaussMatrix = np.random.normal(A/np.sqrt(2.),A/snr,[Ntot,2*n])
    GaussMatrix = GaussMatrix**2
    if product=='amp_sq':
        Chi2Vec = np.sum(GaussMatrix,1)/n
    if product=='log_amp_sq':
        Chi2Vec = np.log(np.sum(GaussMatrix,1)/n)
    if product=='amp':
        GaussMatrix2 = np.zeros((Ntot,n))
        for cou in range(n):
            GaussMatrix2[:,cou] = np.sqrt(GaussMatrix[:,2*cou]+GaussMatrix[:,2*cou+1])
        Chi2Vec = np.sum(GaussMatrix2,1)/n
    if product=='log_amp':
        GaussMatrix2 = np.zeros((Ntot,n))
        for cou in range(n):
            GaussMatrix2[:,cou] = np.sqrt(GaussMatrix[:,2*cou]+GaussMatrix[:,2*cou+1])
        Chi2Vec = np.log(np.sum(GaussMatrix2,1)/n)
    return np.mean(Chi2Vec),np.std((Chi2Vec))

def log_camp_sigma(n,x,product= 'log_amp'):
    
    x = np.asarray(x)
    #print(x)
    #x == [(A1,snr1), ..., (A4,snr4)]
    sig = np.zeros(4)
    for cou in range(4):
        m,sig[cou] = MCamp(n,x[cou][0],x[cou][1],product=product)
    sig = np.sqrt(np.sum(np.asarray(sig)**2))
    return sig
    


def coh_average_bsp(AIPS, tcoh = 5.):
    AIPS.loc[:,'vis'] = AIPS.loc[:,'vis'] = AIPS.loc[:,'amp']*np.exp(1j*AIPS.loc[:,'cphase']*np.pi/180)
    AIPS.loc[:,'circ_sigma'] = AIPS.loc[:,'cphase']
    if 'band' not in AIPS.columns:
        AIPS.loc[:,'band'] = ['unknown']*np.shape(AIPS)[0]
    if tcoh == 'scan':
        AIPS = AIPS[['datetime','band','triangle','source','polarization','vis','sigmaCP','snr','scan_id','expt_no','circ_sigma']]
        AIPS = AIPS.groupby(['triangle','band','source','polarization','expt_no','scan_id']).agg({'datetime': 'min', 'vis': np.mean, 'sigmaCP': lambda x: np.sqrt(np.sum(x**2))/len(x),'snr': lambda x: np.sqrt(np.sum(x**2)),'circ_sigma': circular_std_of_mean_dif})
    else:
        AIPS.loc[:,'round_time'] = list(map(lambda x: np.round((x- datetime.datetime(2017,4,4)).total_seconds()/tcoh),AIPS.loc[:,'datetime']))
        AIPS = AIPS[['datetime','band','triangle','source','polarization','vis','sigma','sigmaCP','scan_id','expt_no','round_time','snr','circ_sigma']]
        AIPS = AIPS.groupby(['triangle','band','source','polarization','expt_no','scan_id','round_time']).agg({'datetime': 'min', 'vis': np.mean, 'sigmaCP': lambda x: np.sqrt(np.sum(x**2))/len(x),'snr': lambda x: np.sqrt(np.sum(x**2)),'circ_sigma': circular_std_of_mean_dif })
    AIPS = AIPS.reset_index()
    AIPS['amp'] = np.abs(AIPS['vis'])
    AIPS['cphase'] = np.angle(AIPS['vis'])*180/np.pi
    AIPS = AIPS[['datetime','band','triangle','source','polarization','amp', 'cphase', 'sigmaCP','snr','expt_no','scan_id','circ_sigma']]
    return AIPS
    

def circular_mean(theta):
    theta = np.asarray(theta, dtype=np.float32)*np.pi/180.
    theta = theta[theta==theta]
    
    if len(theta)==0:
        return None
    else:
        C = np.mean(np.cos(theta))
        S = np.mean(np.sin(theta))
        mt = np.arctan2(S,C)*180./np.pi
        return mt

def circular_std(theta):
    theta = np.asarray(theta, dtype=np.float32)*np.pi/180.
    C = np.mean(np.cos(theta))
    S = np.mean(np.sin(theta))
    st = np.sqrt(-2.*np.log(np.sqrt(C**2+S**2)))*180./np.pi
    return st

def circular_std_of_mean(theta):
    theta = np.asarray(theta, dtype=np.float32)*np.pi/180.
    C = np.mean(np.cos(theta))
    S = np.mean(np.sin(theta))
    st = np.sqrt(-2.*np.log(np.sqrt(C**2+S**2)))*180./np.pi/np.sqrt(len(theta))
    return st

def diff_side(x):
    x = np.asarray(x, dtype=np.float32)
    xp = x[1:]
    xm = x[:-1]
    xdif = xp-xm
    dx = np.angle(np.exp(1j*xdif*np.pi/180.))*180./np.pi
    return dx



def circular_std_of_mean_dif(theta):
    theta = np.asarray(theta, dtype=np.float32)*np.pi/180.
    #dif_theta = np.diff(theta)
    dif_theta = diff_side(theta)
    C = np.mean(np.cos(dif_theta))
    S = np.mean(np.sin(dif_theta))
    st = np.sqrt(-2.*np.log(np.sqrt(C**2+S**2)))*180./np.pi/np.sqrt(len(theta))/np.sqrt(2)
    return st

def std_dif(amp):
    amp = np.diff(np.asarray(amp, dtype=np.float32))
    return np.std(amp)/np.sqrt(2)

def unbiased_sigma(amp):
    amp2 = np.asarray(amp, dtype=np.float32)**2
    m = np.mean(amp2)
    s = np.std(amp2)
    delta = m**2 - s**2
    if delta >= 0:
        s0 = np.sqrt((m -np.sqrt(delta))/2.)
    else:
        s0 = 0.*np.sqrt(m/2.)
    return s0

def unbiased_sigma_dif(amp):
    amp2 = np.diff(np.asarray(amp, dtype=np.float32)**2)
    m = np.mean(amp2)
    s = np.std(amp2)
    k = np.mean(amp2**4)
    delta = 579*s**4 - k
    if delta >= 0:
        s0 = (24*s**2 - np.sqrt(delta))**(0.25)
    else:
        s0 = 0.
    return s0

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

def unbiased_amp2(amp):
    amp = np.asarray(amp, dtype=np.float32)
    m = np.mean(amp); q = np.mean(amp**2)
    eq_for_sig = lambda x: x*np.sqrt(np.pi/2.)*ss.hyp1f1(-0.5, 1., 1. - q/2./x**2) - m
    try:
        Esig = so.brentq(eq_for_sig, 1.e-10, 1.5*np.std(amp))
    except ValueError:
        Esig = np.std(amp)
    delta = q - 2.*Esig**2
    if delta >=0:
        EA0 = np.sqrt(delta)
    else:
        EA0 = 0.
    return EA0

def unbiased_sigma2(amp, dtype=np.float32):
    amp = np.asarray(amp)
    m = np.mean(amp); q = np.mean(amp**2)
    eq_for_sig = lambda x: x*np.sqrt(np.pi/2.)*ss.hyp1f1(-0.5, 1., 1. - q/2./x**2) - m
    try:
        Esig = so.brentq(eq_for_sig, 1.e-10, 1.5*np.std(amp))
    except ValueError:
        Esig = np.std(amp)
    return Esig

def phase_diff(v1,v2):
    v2 = np.asarray(np.mod(v2,360), dtype=np.float32)
    v1 = np.asarray(np.mod(v1,360), dtype=np.float32)
    v2b = v2 + 360
    v2c = v2 - 360
    e1 = np.abs(v1 - v2)
    e2 = np.abs(v1 - v2b)
    v2[e2 < e1] = v2b[e2 < e1]
    e1 = np.abs(v1 - v2)
    e3 = np.abs(v1 - v2c)
    v2[e3 < e1] = v2c[e3 < e1]
    return v2

'''
def coh_avg(frame, tcoh='scan',phaseType='resid_phas',ampType='amp',group=VIS):
'''
#coherently averages complex quantity
'''
if 'band' not in frame.columns:
    frame['band'] = np.nan
if group==VIS:
    grouping = ('scan_id','expt_no','band','polarization')

if tcoh=='scan':

else:
    grouping = grouping+('datetime')
'''

def coh_average_vis(AIPS, tcoh = 5.,phaseType='resid_phas'):
    #print(AIPS.columns)
    if 'scan_no_tot' not in AIPS.columns:
        AIPS.loc[:,'scan_no_tot'] = AIPS.loc[:,'scan_id']
    if 'sigma' not in AIPS.columns:
        AIPS.loc[:,'sigma'] = AIPS.loc[:,'amp']/AIPS.loc[:,'snr']
    if 'std' not in AIPS.columns:
        AIPS.loc[:,'std'] = AIPS.loc[:,'sigma']
    if 'band' not in AIPS.columns:
        AIPS.loc[:,'band'] = ['unknown']*np.shape(AIPS)[0]

    AIPS['track'] = list(map(lambda x: a2a.expt2track[x],AIPS['expt_no']))
    AIPS['round_time'] = list(map(lambda x: np.round((x- datetime.datetime(2017,4,4)).total_seconds()/tcoh),AIPS['datetime']))
    AIPS['vis'] = AIPS['vis'] = AIPS['amp']*np.exp(1j*AIPS[phaseType]*np.pi/180)
    if 'snr' in AIPS.columns:
        AIPS = AIPS[['datetime','band','baseline','source','polarization','vis','snr','std','sigma','track','expt_no','scan_no_tot','round_time']]
        #AIPS = AIPS.groupby(('baseline','source','polarization','track','expt_no','scan_no_tot','round_time')).agg({'datetime': 'min', 'vis': np.mean, 'sigma': lambda x: np.sqrt(np.sum(x**2))/len(x), 'std': lambda x: np.sqrt(np.sum(x**2))/len(x), 'snr': lambda x: np.average(x)*np.sqrt(len(x))})
        AIPS = AIPS.groupby(('baseline','band','source','polarization','track','expt_no','scan_no_tot','round_time')).agg({'datetime': 'min', 'vis': np.mean, 'sigma': lambda x: np.sqrt(np.sum(x**2))/len(x), 'std': lambda x: np.sqrt(np.sum(x**2))/len(x), 'snr': lambda x: np.sqrt(np.sum(x**2))})
    else:
        AIPS = AIPS[['datetime','band','baseline','source','polarization','vis','std','sigma','track','expt_no','scan_no_tot','round_time']]
        AIPS = AIPS.groupby(('baseline','band','source','polarization','track','expt_no','scan_no_tot','round_time')).agg({'datetime': 'min', 'vis': np.mean, 'sigma': lambda x: np.sqrt(np.sum(x**2))/len(x), 'std': lambda x: np.sqrt(np.sum(x**2))/len(x)})
    AIPS = AIPS.reset_index()
    AIPS['amp'] = np.abs(AIPS['vis'])
    AIPS[phaseType] = np.angle(AIPS['vis'])*180/np.pi

    if 'snr' in AIPS.columns:
        AIPS = AIPS[['datetime','band','baseline','source','polarization','amp', phaseType,'snr','std', 'sigma','track','expt_no','scan_no_tot']]
    else:
        AIPS = AIPS[['datetime','band','baseline','source','polarization','amp', phaseType,'std', 'sigma','track','expt_no','scan_no_tot']]
    return AIPS

def coh_average_vis_uv(AIPS, tcoh = 5.,phaseType='resid_phas'):
    #print(AIPS.columns)
    if 'scan_no_tot' not in AIPS.columns:
        AIPS.loc[:,'scan_no_tot'] = AIPS.loc[:,'scan_id']
    if 'sigma' not in AIPS.columns:
        AIPS.loc[:,'sigma'] = AIPS.loc[:,'amp']/AIPS.loc[:,'snr']
    if 'std' not in AIPS.columns:
        AIPS.loc[:,'std'] = AIPS.loc[:,'sigma']

    AIPS['track'] = list(map(lambda x: a2a.expt2track[x],AIPS['expt_no']))
    AIPS['round_time'] = list(map(lambda x: np.round((x- datetime.datetime(2017,4,4)).total_seconds()/tcoh),AIPS['datetime']))
    AIPS['vis'] = AIPS['vis'] = AIPS['amp']*np.exp(1j*AIPS[phaseType]*np.pi/180)
    if 'snr' in AIPS.columns:
        AIPS = AIPS[['datetime','baseline','source','polarization','vis','snr','std','sigma','track','expt_no','scan_no_tot','round_time','u','v']]
        #AIPS = AIPS.groupby(('baseline','source','polarization','track','expt_no','scan_no_tot','round_time')).agg({'datetime': 'min', 'vis': np.mean, 'sigma': lambda x: np.sqrt(np.sum(x**2))/len(x), 'std': lambda x: np.sqrt(np.sum(x**2))/len(x), 'snr': lambda x: np.average(x)*np.sqrt(len(x))})
        AIPS = AIPS.groupby(('baseline','source','polarization','track','expt_no','scan_no_tot','round_time')).agg({'datetime': 'min', 'vis': np.mean, 
        'sigma': lambda x: np.sqrt(np.sum(x**2))/len(x), 'std': lambda x: np.sqrt(np.sum(x**2))/len(x), 'snr': lambda x: np.sqrt(np.sum(x**2)), 'u': np.mean, 'v': np.mean})
    else:
        AIPS = AIPS[['datetime','baseline','source','polarization','vis','std','sigma','track','expt_no','scan_no_tot','round_time','u','v']]
        AIPS = AIPS.groupby(('baseline','source','polarization','track','expt_no','scan_no_tot','round_time')).agg({'datetime': 'min', 'vis': np.mean, 
        'sigma': lambda x: np.sqrt(np.sum(x**2))/len(x), 'std': lambda x: np.sqrt(np.sum(x**2))/len(x),'u': np.mean, 'v': np.mean})
    AIPS = AIPS.reset_index()
    AIPS['amp'] = np.abs(AIPS['vis'])
    AIPS[phaseType] = np.angle(AIPS['vis'])*180/np.pi

    if 'snr' in AIPS.columns:
        AIPS = AIPS[['datetime','baseline','source','polarization','amp', phaseType,'snr','std', 'sigma','track','expt_no','scan_no_tot','u','v']]
    else:
        AIPS = AIPS[['datetime','baseline','source','polarization','amp', phaseType,'std', 'sigma','track','expt_no','scan_no_tot','u','v']]
    return AIPS

def incoh_average_amp(AIPS, tinc = 'scan',scale_amp=1.):
    AIPS['sigmaB'] = AIPS['amp']
    AIPS['sigma'] = AIPS['amp'] 
    AIPS['ampB'] = AIPS['amp']

    if 'scan_id' not in AIPS.columns:   
        AIPS['scan_id'] = AIPS['scan_no_tot']
    if tinc == 'scan':
        if 'snr' in AIPS.columns:
            AIPS = AIPS[['datetime','baseline','source','polarization','amp','ampB','snr','scan_id','expt_no','sigma','sigmaB','band']]
            AIPS = AIPS.groupby(('baseline','source','polarization','expt_no','scan_id','band')).agg({'datetime': 'min', 'ampB': np.mean, 'amp': unbiased_amp, 
            'sigmaB': lambda x: np.std(x)/np.sqrt(len(x)), 'sigma': lambda x: unbiased_sigma(x)/np.sqrt(len(x)), 'snr': lambda x : np.sqrt(np.sum(x**2)) })
        else:
            AIPS = AIPS[['datetime','baseline','source','polarization','amp','ampB','scan_id','expt_no','sigma','sigmaB']]
            AIPS = AIPS.groupby(('baseline','source','polarization','expt_no','scan_id')).agg({'datetime': 'min', 'ampB': np.mean, 'amp': unbiased_amp, 'sigmaB': lambda x: np.std(x)/np.sqrt(len(x)), 'sigma': lambda x: unbiased_sigma(x)/np.sqrt(len(x)) })
    else:
        AIPS.loc[:,'round_time'] = list(map(lambda x: np.round((x- datetime.datetime(2017,4,4)).total_seconds()/tinc),AIPS.loc[:,'datetime']))
        if 'snr' in AIPS.columns:
            AIPS = AIPS[['datetime','baseline','source','polarization','amp','ampB','snr','sigma','sigmaB','scan_id','expt_no','round_time']]
            AIPS = AIPS.groupby(('baseline','source','polarization','expt_no','scan_id','round_time')).agg({'datetime': 'min', 'ampB': np.mean, 'amp': unbiased_amp, 
            'sigmaB': lambda x: np.std(x)/np.sqrt(len(x)), 'sigma': lambda x: unbiased_sigma(x)/np.sqrt(len(x)), 'snr': lambda x : np.sqrt(np.sum(x**2)) })
        else:
            AIPS = AIPS[['datetime','baseline','source','polarization','amp','ampB','sigma','sigmaB','scan_id','expt_no','round_time']]
            AIPS = AIPS.groupby(('baseline','source','polarization','expt_no','scan_id','round_time')).agg({'datetime': 'min', 'ampB': np.mean, 'amp': unbiased_amp, 'sigmaB': lambda x: np.std(x)/np.sqrt(len(x)), 'sigma': lambda x: unbiased_sigma(x)/np.sqrt(len(x)) })
    
    AIPS.loc[:,'amp'] = scale_amp*AIPS.loc[:,'amp']
    AIPS.loc[:,'ampB'] = scale_amp*AIPS.loc[:,'ampB']
    AIPS.loc[:,'sigmaB'] = scale_amp*AIPS.loc[:,'sigmaB']
    AIPS.loc[:,'sigma'] = scale_amp*AIPS.loc[:,'sigma']
    
    return AIPS.reset_index()

def average_quad(quad, tinc = 'scan'):
    
    if 'scan_id' not in quad.columns:   
        quad['scan_id'] = quad['scan_no_tot']
    if 'band' not in quad.columns:
        quad.loc[:,'band'] = ['unknown']*np.shape(quad)[0]
        
    if tinc == 'scan':

        quad= quad[['datetime','quadrangle','source','polarization','logamp','scan_id','expt_no','sigma','band']]
        AIPS = quad.groupby(('quadrangle','source','polarization','expt_no','scan_id','band')).agg({'datetime': 'min', 'logamp': np.mean, 
        'sigma': lambda x: np.sqrt(np.sum(x**2))/len(x) })
    else:
        quad.loc[:,'round_time'] = list(map(lambda x: np.round((x- datetime.datetime(2017,4,4)).total_seconds()/tinc),quad.loc[:,'datetime']))
        quad = quad[['datetime','quadrangle','source','polarization','logamp','sigma','scan_id','expt_no','round_time','band']]
        quad = quad.groupby(('quadrangle','source','polarization','expt_no','scan_id','band','round_time')).agg({'datetime': 'min', 'logamp': np.mean, 'sigma': lambda x: np.sqrt(np.sum(x**2))/len(x) })
    
    return quad.reset_index()


    #AIPS['round_time'] = map(lambda x: np.round((x- datetime.datetime(2017,4,4)).total_seconds()/tcoh),AIPS['datetime'])
    #AIPS['vis'] = AIPS['vis'] = AIPS['amp']*np.exp(1j*AIPS[phaseType]*np.pi/180)
    #AIPS = AIPS[['datetime','baseline','source','polarization','vis','std','sigma','track','expt_no','scan_no_tot','round_time']]
    #AIPS = AIPS.groupby(('baseline','source','polarization','track','expt_no','scan_no_tot','round_time')).agg({'datetime': 'min', 'vis': np.mean, 'sigma': lambda x: np.sqrt(np.sum(x**2))/len(x), 'std': lambda x: np.sqrt(np.sum(x**2))/len(x)})
    #AIPS = AIPS.reset_index()
    #AIPS['amp'] = np.abs(AIPS['vis'])
    #AIPS[phaseType] = np.angle(AIPS['vis'])*180/np.pi
    #AIPS = AIPS[['datetime','baseline','source','polarization','amp', phaseType,'std', 'sigma','track','expt_no','scan_no_tot']]
    #return AIPS

def incoh_average_amp_uv(AIPS, tinc = 'scan',scale_amp=1.):
    AIPS['sigmaB'] = AIPS['amp']
    AIPS['sigma'] = AIPS['amp'] 
    AIPS['ampB'] = AIPS['amp']

    if 'scan_id' not in AIPS.columns:   
        AIPS['scan_id'] = AIPS['scan_no_tot']
    if tinc == 'scan':
        if 'snr' in AIPS.columns:
            AIPS = AIPS[['datetime','baseline','source','polarization','amp','ampB','snr','scan_id','expt_no','sigma','sigmaB','u','v']]
            AIPS = AIPS.groupby(('baseline','source','polarization','expt_no','scan_id')).agg({'datetime': 'min', 'ampB': np.mean, 'amp': unbiased_amp, 
            'sigmaB': lambda x: np.std(x)/np.sqrt(len(x)), 'sigma': lambda x: unbiased_sigma(x)/np.sqrt(len(x)), 'snr': lambda x : np.sqrt(np.sum(x**2)),'u': np.mean, 'v': np.mean })
        else:
            AIPS = AIPS[['datetime','baseline','source','polarization','amp','ampB','scan_id','expt_no','sigma','sigmaB','u','v']]
            AIPS = AIPS.groupby(('baseline','source','polarization','expt_no','scan_id')).agg({'datetime': 'min', 'ampB': np.mean, 'amp': unbiased_amp, 
            'sigmaB': lambda x: np.std(x)/np.sqrt(len(x)), 'sigma': lambda x: unbiased_sigma(x)/np.sqrt(len(x)),'u': np.mean, 'v': np.mean })
    else:
        AIPS.loc[:,'round_time'] = map(lambda x: np.round((x- datetime.datetime(2017,4,4)).total_seconds()/tinc),AIPS.loc[:,'datetime'])
        if 'snr' in AIPS.columns:
            AIPS = AIPS[['datetime','baseline','source','polarization','amp','ampB','snr','sigma','sigmaB','scan_id','expt_no','round_time','u','v']]
            AIPS = AIPS.groupby(('baseline','source','polarization','expt_no','scan_id','round_time')).agg({'datetime': 'min', 'ampB': np.mean, 'amp': unbiased_amp, 
            'sigmaB': lambda x: np.std(x)/np.sqrt(len(x)), 'sigma': lambda x: unbiased_sigma(x)/np.sqrt(len(x)), 'snr': lambda x : np.sqrt(np.sum(x**2)),'u': np.mean, 'v': np.mean })
        else:
            AIPS = AIPS[['datetime','baseline','source','polarization','amp','ampB','sigma','sigmaB','scan_id','expt_no','round_time','u','v']]
            AIPS = AIPS.groupby(('baseline','source','polarization','expt_no','scan_id','round_time')).agg({'datetime': 'min', 'ampB': np.mean, 'amp': unbiased_amp, 
            'sigmaB': lambda x: np.std(x)/np.sqrt(len(x)), 'sigma': lambda x: unbiased_sigma(x)/np.sqrt(len(x)),'u': np.mean, 'v': np.mean })
    
    
    AIPS.loc[:,'amp'] = scale_amp*AIPS.loc[:,'amp']
    AIPS.loc[:,'ampB'] = scale_amp*AIPS.loc[:,'ampB']
    AIPS.loc[:,'sigmaB'] = scale_amp*AIPS.loc[:,'sigmaB']
    AIPS.loc[:,'sigma'] = scale_amp*AIPS.loc[:,'sigma']
    
    return AIPS.reset_index()

def add_round_time(frame, dt = 20.):
    frame.loc[:,'round_time'] = list(map(lambda x: np.round((x- datetime.datetime(2017,4,4)).total_seconds()/dt),frame['datetime']))
    return frame

def add_polar_frac(frame, dt = 20.):
    frame = frame[list(map(lambda x: x[0]!=x[1], frame.baseline))]
    frame = add_round_time(frame, dt)
    frame.loc[:,'polar_frac'] = [0.]*np.shape(frame)[0]
    frameG = frame.groupby(('baseline','round_time')).filter(lambda x: len(x) >1)
    frame = frame.groupby(('baseline','round_time')).filter(lambda x: len(x) <5)
    frame = frame.groupby(('baseline','round_time')).filter(lambda x: ('RL' in list(x.polarization)  )|('LR' in list(x.polarization)) )
    #print(frame)
    
    #frame_LL_RR = frame[(frame.polarization=='LL')|(frame.polarization=='RR')]
    
    polar_fracL = []
    #for index, row in frame_LL_RR.iterrows():
    for index, row in frame.iterrows():
        dt_foo = row.round_time
        base_foo = row.baseline
        amp_RL = list(frame[(frame.round_time==dt_foo)&(frame.baseline==base_foo)&(frame.polarization=='RL')].amp)
        amp_LR = list(frame[(frame.round_time==dt_foo)&(frame.baseline==base_foo)&(frame.polarization=='LR')].amp)
        if len(amp_RL)==0:
            amp_RL = 0.0
        else: amp_RL = amp_RL[0]
        if len(amp_LR)==0:
            amp_LR = 0.0
        else: amp_LR = amp_LR[0]
        amp_cross = np.maximum(amp_RL,amp_LR)
        polar_fracL.append(amp_cross/row.amp)

    #frame_LL_RR['polar_frac']= polar_fracL
    frame['polar_frac']= polar_fracL
    return frame

def f_help(group):
    fooRL =list(group[group['polarization']=='RL'].amp)
    fooLR =list(group[group['polarization']=='LR'].amp)
    fooRR =list(group[group['polarization']=='RR'].amp)
    fooLL =list(group[group['polarization']=='LL'].amp)
    if len(fooRL)==0:
        fooRL = 0.0
    else: fooRL = fooRL[0]
    if len(fooLR)==0:
        fooLR = 0.0
    else: fooLR = fooLR[0]
    if len(fooRR)==0:
        fooRR = 0.0
    else: fooRR = fooRR[0]
    if len(fooLL)==0:
        fooLL = 0.0
    else: fooLL = fooLL[0]
    return (fooRR,fooLL,fooRL,fooLR)


def match_2_dataframes(frame1, frame2, what_is_same=None):
#what_is_same, e.g., triangle, then for given datetime matches only same triangles    
    if what_is_same==None:
        S1 = set(frame1.datetime)
        S2 = set(frame2.datetime)
        Sprod = S1&S2
        cond1 = list(map(lambda x: x in Sprod, zip(frame1.datetime,frame1[what_is_same])))
        cond2 = list(map(lambda x: x in Sprod, zip(frame1.datetime,frame1[what_is_same])))
    else: 
        S1 = set(zip(frame1.datetime,frame1[what_is_same]))
        S2 = set(zip(frame2.datetime,frame2[what_is_same]))
        Sprod = S1&S2
        cond1 = list(map(lambda x: x in Sprod, zip(frame1.datetime,frame1[what_is_same])))
        cond2 = list(map(lambda x: x in Sprod, zip(frame2.datetime,frame2[what_is_same])))
    frame1 = frame1[cond1]
    frame2 = frame2[cond2]
    return frame1, frame2

def match_2_dataframes_approxT(frame1, frame2, what_is_same=None, dt = 5.):
#what_is_same, e.g., triangle, then for given datetime matches only same triangles
    frame1['round_time'] = list(map(lambda x: np.round((x- datetime.datetime(2017,4,4)).total_seconds()/dt),frame1['datetime']))
    frame2['round_time'] = list(map(lambda x: np.round((x- datetime.datetime(2017,4,4)).total_seconds()/dt),frame2['datetime']))
    if what_is_same==None:
        S1 = set(frame1.round_time)
        S2 = set(frame2.round_time)
        Sprod = S1&S2
        cond1 = list(map(lambda x: x in Sprod, frame1.round_time))
        cond2 = list(map(lambda x: x in Sprod, frame2.round_time))
    else: 
        S1 = set(zip(frame1.round_time,frame1[what_is_same]))
        S2 = set(zip(frame2.round_time,frame2[what_is_same]))
        Sprod = S1&S2
        cond1 = list(map(lambda x: x in Sprod, zip(frame1.round_time,frame1[what_is_same])))
        cond2 = list(map(lambda x: x in Sprod, zip(frame2.round_time,frame2[what_is_same])))
    frame1 = frame1[cond1]
    frame2 = frame2[cond2]
    return frame1, frame2

def match_2_bsp_frames(frame1,frame2,match_what='pipeline',dt = 15.,what_is_same='triangle'):
    
    frame1_lo_ll = frame1[(frame1.band=='lo')&(frame1.polarization=='LL')].reset_index(drop='True')
    frame1_hi_ll = frame1[(frame1.band=='hi')&(frame1.polarization=='LL')].reset_index(drop='True')
    frame1_lo_rr = frame1[(frame1.band=='lo')&(frame1.polarization=='RR')].reset_index(drop='True')
    frame1_hi_rr = frame1[(frame1.band=='hi')&(frame1.polarization=='RR')].reset_index(drop='True')

    frame2_lo_ll = frame2[(frame2.band=='lo')&(frame2.polarization=='LL')].reset_index(drop='True')
    frame2_hi_ll = frame2[(frame2.band=='hi')&(frame2.polarization=='LL')].reset_index(drop='True')
    frame2_lo_rr = frame2[(frame2.band=='lo')&(frame2.polarization=='RR')].reset_index(drop='True')
    frame2_hi_rr = frame2[(frame2.band=='hi')&(frame2.polarization=='RR')].reset_index(drop='True')

    if match_what=='pipeline':
        #match everything from first frame to second, keeping polarization and band
        frame1_lo_ll, frame2_lo_ll = match_2_dataframes_approxT(frame1_lo_ll, frame2_lo_ll, what_is_same, dt)
        frame1_hi_ll, frame2_hi_ll = match_2_dataframes_approxT(frame1_hi_ll, frame2_hi_ll, what_is_same, dt)
        frame1_lo_rr, frame2_lo_rr = match_2_dataframes_approxT(frame1_lo_rr, frame2_lo_rr, what_is_same, dt)
        frame1_hi_rr, frame2_hi_rr = match_2_dataframes_approxT(frame1_hi_rr, frame2_hi_rr, what_is_same, dt)

        frame1 = pd.concat([frame1_lo_ll,frame1_hi_ll,frame1_lo_rr,frame1_hi_rr], ignore_index=True)
        frame2 = pd.concat([frame2_lo_ll,frame2_hi_ll,frame2_lo_rr,frame2_hi_rr], ignore_index=True)

    elif match_what=='polarization':
        #print('dududdududu')
        #match ll polarization from the first frame to rr polarization in the second frame, keepieng band
        frame1_lo_ll, frame2_lo_rr = match_2_dataframes_approxT(frame1_lo_ll, frame2_lo_rr, what_is_same, dt)
        frame1_hi_ll, frame2_hi_rr = match_2_dataframes_approxT(frame1_hi_ll, frame2_hi_rr, what_is_same, dt)
        
        frame1 = pd.concat([frame1_lo_ll,frame1_hi_ll], ignore_index=True)
        frame2 = pd.concat([frame2_lo_rr,frame2_hi_rr], ignore_index=True)

    elif match_what=='band':
        #match lo band in first frame to hi band in 2nd frame, keeping the polarizations equal
        frame1_lo_ll, frame2_hi_ll = match_2_dataframes_approxT(frame1_lo_ll, frame2_hi_ll, what_is_same, dt)
        frame1_lo_rr, frame2_hi_rr = match_2_dataframes_approxT(frame1_lo_rr, frame2_hi_rr, what_is_same, dt)
        
        frame1 = pd.concat([frame1_lo_ll,frame1_lo_rr], ignore_index=True)
        frame2 = pd.concat([frame2_hi_ll,frame2_hi_rr], ignore_index=True)

    return frame1, frame2

#def match_subscan(frame1,frame2, what_same):
#    what_same = ['datetime', 'source', 'baseline','scan_id','expt_no']
#
#    cond1zip(frame1.datetime,frame1.baseline)
#    what_different = 


def add_band(bisp,band):
    bisp['band'] = [band]*np.shape(bisp)[0]
    return bisp

def add_error(bisp,to_what='cphase'):
    
    if to_what=='cphase':
        bisp['TotErr'] = np.mod(np.asarray(bisp['cphase']),360.)
        bisp['TotErr'] = np.minimum( np.asarray(bisp['TotErr']), np.abs(np.asarray(bisp['TotErr']) -360.))
        bisp['RelErr'] = np.asarray(bisp['TotErr'])/np.asarray(bisp['sigmaCP'])
    elif to_what=='amp':
        bisp['TotErr'] = bisp['amp']
        bisp['RelErr'] = np.asarray(bisp['TotErr'])/np.asarray(bisp['sigma'])
    return bisp

def use_measured_circ_std_as_sigmaCP(bsp):
    bsp.loc[:,'sigmaCP'] = bsp.loc[:,'circ_sigma']
    return bsp

def get_bsp_from_alist(alist_path,tcoh = 5.,tav = 'scan',phaseType='resid_phas',typeA='alist',band='',data_int_time=1.):

    if typeA=='alist':
        alist = hops.read_alist(alist_path)
    elif typeA=='pickle':
        alist = pd.read_pickle(alist_path)

    if data_int_time < tcoh:
        alist = coh_average_vis(alist, tcoh, phaseType)

    bsp_ll = all_bispectra_polar(alist,'LL',phaseType)
    bsp_rr = all_bispectra_polar(alist,'RR',phaseType) 
    bsp = pd.concat([bsp_ll,bsp_rr],ignore_index=True) 
    bsp_av = coh_average_bsp(bsp,tav)

    bsp_av =use_measured_circ_std_as_sigmaCP(bsp_av)
    bsp_av = add_error(bsp_av)

    if band != '':
        bsp_av = add_band(bsp_av,band)

    return bsp_av

def DataBaseline(alist,basename,polar):
    condB = (alist['baseline']==basename)
    condP = (alist['polarization']==polar)
    if 't_coh' in alist.keys():
        alistB = alist.loc[condB&condP,['expt_no','scan_id','source','datetime','baseline','total_phas','amp','snr','gmst','t_coh','t_coh_bias']]
    else:
        alistB = alist.loc[condB&condP,['expt_no','scan_id','source','datetime','baseline','total_phas','amp','snr','gmst']]
    alistB.loc[:,'sigma'] = (alistB.loc[:,'amp']/(alistB.loc[:,'snr']))
    alistB.loc[:,'snrstd'] = (alistB.loc[:,'snr'])
    alistB.loc[:,'ampstd'] = (alistB.loc[:,'amp'])
    alistB.loc[:,'phasestd'] = (alistB.loc[:,'total_phas'])
    if 't_coh' in alist.keys():
        alistB = alistB.groupby(('source','expt_no','scan_id')).agg({'total_phas': lambda x: np.average(x),
        'amp': lambda x: np.average(x), 'sigma': lambda x: np.sqrt(np.sum(x**2))/len(x), 'snr': 'mean','snrstd': 'std',
                        'gmst': 'min', 'ampstd': 'std', 'phasestd': 'std', 't_coh': 'mean','t_coh_bias': 'mean'})
    else:
        alistB = alistB.groupby(('source','expt_no','scan_id')).agg({'total_phas': lambda x: np.average(x),
        'amp': lambda x: np.average(x), 'sigma': lambda x: np.sqrt(np.sum(x**2))/len(x), 'snr': 'mean','snrstd': 'std',
             'gmst': 'min', 'ampstd': 'std', 'phasestd': 'std'})
        
    return alistB

def DataTriangle(alist,Tri,signat,pol):
    condP = (alist['polarization']==pol)
    condB1 = (alist['baseline']==Tri[0])
    condB2 = (alist['baseline']==Tri[1])
    condB3 = (alist['baseline']==Tri[2])
    condB = condB1|condB2|condB3

    alistRR_Tri = alist.loc[condP&condB,['expt_no','scan_id','source','datetime','baseline','total_phas','amp','snr','gmst']]
    #alistRR_Tri = alist.loc[condP&condB,['expt_no','scan_id','source','datetime','baseline','res_phas','amp','snr','gmst']]
    
    #throw away times without full triangle
    tlist = alistRR_Tri.groupby('datetime').filter(lambda x: len(x) > 2)
    tlist.loc[:,'sigma'] = (tlist.loc[:,'amp']/(tlist.loc[:,'snr']))
    for cou in range(3):
        tlist.loc[(tlist.loc[:,'baseline']==Tri[cou]),'total_phas'] *= signat[cou]*np.pi/180.
    #tlist

    tlist.loc[:,'sigma'] = 1./tlist.loc[:,'snr']**2 #put 1/snr**2 in the sigma column to aggregate
    tlist.loc[:,'footime'] = tlist.loc[:,'datetime'] #dummy time to aggregate CP

    #first aggregation to get bispectra
    bsp = tlist.groupby(('expt_no','source','scan_id','footime')).agg({'total_phas': lambda x: np.sum(x),
        'amp': lambda x: np.prod(x), 'sigma': lambda x: np.sqrt(np.sum(x)),'gmst': 'min','datetime': 'min'})
    #sigma above is the CLOSURE PHASE ERROR
    
    bsp.loc[:,'bisp'] = bsp.loc[:,'amp']*np.exp(1j*bsp.loc[:,'total_phas'])
    bsp.loc[:,'snr'] = 1./bsp.loc[:,'sigma']
    bsp.loc[:,'sigma'] = bsp.loc[:,'amp']*bsp.loc[:,'sigma'] #sigma of bispectrum

    #second aggregation to average the bispectra
    bsp = bsp.groupby(('expt_no','source','scan_id')).agg({'bisp': np.average,'gmst' :'min','datetime' :'min','sigma': lambda x: np.sqrt(sum(x**2))/len(x)})
    bsp.loc[:,'cphase'] = np.angle(bsp.loc[:,'bisp'])*180./np.pi #deg
    bsp.loc[:,'amp'] = np.abs(bsp.loc[:,'bisp'])
    bsp.loc[:,'snr'] = bsp.loc[:,'amp']/bsp.loc[:,'sigma']
    bsp.loc[:,'sigmaCP'] = 1./bsp.loc[:,'snr']*180./np.pi #deg
    del bsp['bisp']
    
    return bsp

def DataTriangleNotAv(alist,Tri,signat,pol):
    condP = (alist['polarization']==pol)
    condB1 = (alist['baseline']==Tri[0])
    condB2 = (alist['baseline']==Tri[1])
    condB3 = (alist['baseline']==Tri[2])
    condB = condB1|condB2|condB3

    alistRR_Tri = alist.loc[condP&condB,['expt_no','scan_id','source','datetime','baseline','total_phas','amp','snr','gmst']]
    #alistRR_Tri = alist.loc[condP&condB,['expt_no','scan_id','source','datetime','baseline','res_phas','amp','snr','gmst']]
    
    #throw away times without full triangle
    tlist = alistRR_Tri.groupby('datetime').filter(lambda x: len(x) > 2)
    tlist.loc[:,'sigma'] = (tlist.loc[:,'amp']/(tlist.loc[:,'snr']))
    for cou in range(3):
        tlist.loc[(tlist.loc[:,'baseline']==Tri[cou]),'total_phas'] *= signat[cou]*np.pi/180.
    #tlist

    tlist.loc[:,'sigma'] = 1./tlist.loc[:,'snr']**2 #put 1/snr**2 in the sigma column to aggregate
    tlist.loc[:,'footime'] = tlist.loc[:,'datetime'] #dummy time to aggregate CP

    #first aggregation to get bispectra
    bsp = tlist.groupby(('expt_no','source','scan_id','footime')).agg({'total_phas': lambda x: np.sum(x),
        'amp': lambda x: np.prod(x), 'sigma': lambda x: np.sqrt(np.sum(x)),'gmst': 'min','datetime': 'min'})
    #sigma above is the CLOSURE PHASE ERROR
    bsp0 = bsp
    bsp.loc[:,'bisp'] = bsp.loc[:,'amp']*np.exp(1j*bsp.loc[:,'total_phas'])
    bsp.loc[:,'snr'] = 1./bsp.loc[:,'sigma']
    bsp.loc[:,'sigma'] = bsp.loc[:,'amp']*bsp.loc[:,'sigma'] #sigma of bispectrum

    #second aggregation to average the bispectra
    bsp = bsp.groupby(('expt_no','source','scan_id')).agg({'bisp': np.average,'gmst' :'min','datetime' :'min','sigma': lambda x: np.sqrt(sum(x**2))/len(x)})
    bsp.loc[:,'cphase'] = np.angle(bsp.loc[:,'bisp'])*180./np.pi #deg
    bsp.loc[:,'amp'] = np.abs(bsp.loc[:,'bisp'])
    bsp.loc[:,'snr'] = bsp.loc[:,'amp']/bsp.loc[:,'sigma']
    bsp.loc[:,'sigmaCP'] = 1./bsp.loc[:,'snr']*180./np.pi #deg
    del bsp['bisp']
    
    return bsp0


def DataTriangleRP(alist,Tri,signat,pol):
    condP = (alist['polarization']==pol)
    condB1 = (alist['baseline']==Tri[0])
    condB2 = (alist['baseline']==Tri[1])
    condB3 = (alist['baseline']==Tri[2])
    condB = condB1|condB2|condB3

    alistRR_Tri = alist.loc[condP&condB,['expt_no','scan_id','source','datetime','baseline','resid_phas','amp','snr','gmst']]
    #alistRR_Tri = alist.loc[condP&condB,['expt_no','scan_id','source','datetime','baseline','res_phas','amp','snr','gmst']]
    
    #throw away times without full triangle
    tlist = alistRR_Tri.groupby('datetime').filter(lambda x: len(x) > 2)
    tlist.loc[:,'sigma'] = (tlist.loc[:,'amp']/(tlist.loc[:,'snr']))
    for cou in range(3):
        tlist.loc[(tlist.loc[:,'baseline']==Tri[cou]),'resid_phas'] *= signat[cou]*np.pi/180.
    #tlist

    tlist.loc[:,'sigma'] = 1./tlist.loc[:,'snr']**2 #put 1/snr**2 in the sigma column to aggregate
    tlist.loc[:,'footime'] = tlist.loc[:,'datetime'] #dummy time to aggregate CP

    #first aggregation to get bispectra
    bsp = tlist.groupby(('expt_no','source','scan_id','footime')).agg({'resid_phas': lambda x: np.sum(x),
        'amp': lambda x: np.prod(x), 'sigma': lambda x: np.sqrt(np.sum(x)),'gmst': 'min','datetime': 'min'})
    #sigma above is the CLOSURE PHASE ERROR

    bsp.loc[:,'bisp'] = bsp.loc[:,'amp']*np.exp(1j*bsp.loc[:,'resid_phas'])
    bsp.loc[:,'snr'] = 1./bsp.loc[:,'sigma']
    bsp.loc[:,'sigma'] = bsp.loc[:,'amp']*bsp.loc[:,'sigma'] #sigma of bispectrum

    #second aggregation to average the bispectra
    bsp = bsp.groupby(('expt_no','source','scan_id')).agg({'bisp': np.average,'gmst' :'min','datetime' :'min','sigma': lambda x: np.sqrt(sum(x**2))/len(x)})
    bsp.loc[:,'cphase'] = np.angle(bsp.loc[:,'bisp'])*180./np.pi #deg
    bsp.loc[:,'amp'] = np.abs(bsp.loc[:,'bisp'])
    bsp.loc[:,'snr'] = bsp.loc[:,'amp']/bsp.loc[:,'sigma']
    bsp.loc[:,'sigmaCP'] = 1./bsp.loc[:,'snr']*180./np.pi #deg
    del bsp['bisp']
    
    return bsp


def DataQuadrangle(alist,Quad,pol,method=0, debias=0, signat=[1,1,1,1]):

    condP = (alist['polarization']==pol)
    condB1 = (alist['baseline']==Quad[0])
    condB2 = (alist['baseline']==Quad[1])
    condB3 = (alist['baseline']==Quad[2])
    condB4 = (alist['baseline']==Quad[3])
    condB = condB1|condB2|condB3|condB4

    alist_Quad = alist.loc[condP&condB,['expt_no','scan_id','source','datetime','baseline','total_phas','amp','snr','gmst']]
    tlist = alist_Quad.groupby('datetime').filter(lambda x: len(x) > 3)
    tlist.loc[:,'sigma'] = tlist.loc[:,'amp']/(tlist.loc[:,'snr'])
    tlist.loc[:,'datetime_foo'] = tlist.loc[:,'datetime']
    #if debiasing amplitudes
    if debias != 0:
        tlist.loc[:,'amp'] = tlist.loc[:,'amp']*np.sqrt(1.- debias*2./tlist.loc[:,'snr']**2 )
        
    #####################################################################
    #form quadAmplitudes on 5s and average quadAmplitudes over whole scan
    #####################################################################
    if method == 0:
        for cou in range(2,4):
        #inverse amplitude for the visibilities in the denominator
            tlist.loc[(tlist.loc[:,'baseline']==Quad[cou]),'amp'] = 1./tlist.loc[(tlist.loc[:,'baseline']==Quad[cou]),'amp']

        #aggregating to get quadProducts on 5s segments 
        tlist.loc[:,'sigma'] = 1./tlist.loc[:,'snr']**2#put snr in place of sigma to sum for 4product

        quadAmp = tlist.groupby(('expt_no','source','scan_id','datetime_foo')).agg({'amp': lambda x: np.prod(x), 
                                'gmst': 'min', 'datetime': 'min', 'sigma' : lambda x: np.sqrt(np.sum(x)) })
        quadAmp.loc[:,'sigma'] = quadAmp.loc[:,'sigma']*quadAmp.loc[:,'amp']

        quadAmp = quadAmp.groupby(('expt_no','source','scan_id')).agg({'amp': np.average,'gmst' :'min','datetime' :'min','sigma': lambda x: np.sqrt(np.sum(x**2))/len(x)})
        quadAmp.loc[:,'snr'] = quadAmp.loc[:,'amp']/quadAmp.loc[:,'sigma']

    
    #####################################################################
    #calculate visibilities over whole scan, collapse into quadAmplitude
    #####################################################################
    elif method == 1:
        #aggregation to get visibilities over scan
        quadAmp = tlist.groupby(('expt_no','source','scan_id','baseline')).agg({'amp': lambda x: np.average(x), 
                                'gmst': 'min','sigma': lambda x: np.sqrt(np.sum(x**2))/len(x)})
        quadAmp.reset_index(level=3, inplace=True)

        for cou in range(2,4):
            #inverse complex number for the visibilities in the denominator
            #tlist.loc[(tlist['baseline']==Quad[cou]),'total_phas'] *= -1.
            quadAmp.loc[(quadAmp.loc[:,'baseline']==Quad[cou]),'amp'] = 1./quadAmp.loc[(quadAmp.loc[:,'baseline']==Quad[cou]),'amp']

        tlist.loc[:,'sigma'] = tlist.loc[:,'sigma']/tlist.loc[:,'amp']#put sigma by amplitude for 4product
        quadAmp = quadAmp.groupby(('expt_no','source','scan_id')).agg({'amp': np.prod,'gmst' :'min', 'sigma': lambda x: np.sqrt(np.sum(x**2))/len(x)})
        quadAmp.loc[:,'sigma'] = quadAmp.loc[:,'sigma']*quadAmp.loc[:,'amp']
        quadAmp.loc[:,'snr'] = quadAmp.loc[:,'amp']/quadAmp.loc[:,'sigma']

    
    #####################################################################
    #coherent averaging with phase information
    #####################################################################
    #conjugation because we have data for YX not XY
    elif method == 2:
        for cou in range(4):
            tlist.loc[(tlist.loc[:,'baseline']==Quad[cou]),'total_phas'] *= signat[cou]*np.pi/180.   

        for cou in range(2,4):
            tlist.loc[(tlist['baseline']==Quad[cou]),'total_phas'] *= -1.
            tlist.loc[(tlist.loc[:,'baseline']==Quad[cou]),'amp'] = 1./tlist.loc[(tlist.loc[:,'baseline']==Quad[cou]),'amp']

        tlist.loc[:,'sigma'] = 1./tlist.loc[:,'snr']**2#put snr in place of sigma to sum for 4product
        quadAmp = tlist.groupby(('expt_no','source','scan_id','datetime')).agg({'amp': lambda x: np.prod(x), 
                                    'gmst': 'min', 'total_phas': np.sum, 'sigma' : lambda x: np.sqrt(np.sum(x))})
        quadAmp.loc[:,'sigma'] = quadAmp.loc[:,'sigma']*quadAmp.loc[:,'amp']


        quadAmp.loc[:,'quadProd'] = quadAmp.loc[:,'amp']*np.exp(1j*quadAmp.loc[:,'total_phas'])

        quadAmp = quadAmp.groupby(('expt_no','source','scan_id')).agg({'quadProd': np.average,'gmst' :'min','sigma': lambda x: np.sqrt(np.sum(x**2))/len(x)})

        quadAmp.loc[:,'amp'] = np.abs(quadAmp.loc[:,'quadProd'])
        quadAmp.loc[:,'snr'] = quadAmp.loc[:,'amp']/quadAmp.loc[:,'sigma']


    #####################################################################
    #use log amplitudes !!WORK IN PROGRESS!!
    #####################################################################
    elif method==3:
        quadAmp = tlist
        quadAmp.loc[:,'logamp'] = np.log(quadAmp.loc[:,'amp'])
        #approximated formula for std of log(X)
        quadAmp.loc[:,'sigma'] = quadAmp.loc[:,'sigma']/(quadAmp.loc[:,'amp']) 

        for cou in range(2,4):
        #negative sign for denominator amplitudes
            tlist.loc[(tlist.loc[:,'baseline']==Quad[cou]),'logamp'] *= -1.

        #aggregating to get quadProducts on 5s segments 
        quadAmp = quadAmp.groupby(('expt_no','source','scan_id','datetime')).agg({'logamp': lambda x: np.sum(x), 
                                'gmst': 'min', 'sigma' : lambda x: np.sqrt(np.sum(x**2)) })

        quadAmp = quadAmp.groupby(('expt_no','source','scan_id')).agg({'logamp': np.average,'gmst' :'min','sigma': lambda x: np.sqrt(np.sum(x**2))/len(x)})
        quadAmp.loc[:,'snr'] = quadAmp.loc[:,'logamp']/quadAmp.loc[:,'sigma']

    return quadAmp

def Baseline(alist,basename,source):
    condB = (alist['baseline']==basename)
    alistB = alist.loc[condB,['expt_no','scan_id','source','datetime','baseline','total_phas','polarization','amp','snr','gmst']]
    alistB.loc[:,'sigma'] = (alistB.loc[:,'amp']/(alistB.loc[:,'snr']))
    
    snr_listB = tlist.groupby(('expt_no','source','scan_id','datetime','polarization')).agg({'total_phas': lambda x: np.average(x),
        'amp': lambda x: np.average(x), 'sigma': lambda x: np.sqrt(np.sum(x**2))/len(x),'gmst': 'min'})
   
    #aggregate for source on baseline



def PreparePlot(bsp,source,days,hrs):
    t = []; cp =[]; yerr = []
    for x in range(len(days)):
        t.append(hrs[x] + np.asarray(bsp.loc[(bsp.index.get_level_values('source') == source)&(bsp.index.get_level_values('expt_no') == days[x]),['gmst']]))
        cp.append(np.asarray(bsp.loc[(bsp.index.get_level_values('source') == source)&(bsp.index.get_level_values('expt_no') == days[x]),['cphase']]))
        yerr.append(np.asarray(bsp.loc[(bsp.index.get_level_values('source') == source)&(bsp.index.get_level_values('expt_no') == days[x]),['sigmaCP']]))
    tf = np.concatenate(t).ravel()
    cpf = np.concatenate(cp).ravel()
    yerrf = np.concatenate(yerr).ravel()
    return tf, cpf, yerrf



def plotTrianglePolar(alist,Triangle,Signature,pol,MaxAngle,printSummary=1):

    bsp = DataTriangle(alist,Triangle,Signature,pol)
    days = [3597,3600,3601]; source = '3C279'
    #hrs = [0,24.,48.] #to add to time
    #'OJ287' '3C279' 'J1924-2914' 'CENA' '1055+018'
    t3C,cp3C,yerr3C = PreparePlot(bsp,source,days,hrs)
    source = 'OJ287'
    tOJ,cpOJ,yerrOJ = PreparePlot(bsp,source,days,hrs)
    source = 'J1924-2914'
    tJ1,cpJ1,yerrJ1 = PreparePlot(bsp,source,days,hrs)
    source = '1055+018'
    t10,cp10,yerr10 = PreparePlot(bsp,source,days,hrs)
    source = '1749+096'
    t17,cp17,yerr17 = PreparePlot(bsp,source,days,hrs)
    source = '1921-293'
    t19,cp19,yerr19 = PreparePlot(bsp,source,days,hrs)
    source = 'CYGX-3'
    tCY,cpCY,yerrCY = PreparePlot(bsp,source,days,hrs)
    source = 'J1733-1304'
    tJ17,cpJ17,yerrJ17 = PreparePlot(bsp,source,days,hrs)
    source = 'CENA'
    tCE,cpCE,yerrCE = PreparePlot(bsp,source,days,hrs)

    #plot
    #-----------------------------------------------
    plt.figure(figsize=(15,6))
    if len(t3C)> 0:
        plt.errorbar(t3C,cp3C,xerr=0, yerr = 1.*yerr3C, fmt='bo', label = '3C279')
    if len(tOJ)> 0:
        plt.errorbar(tOJ,cpOJ,xerr=0, yerr = 1.*yerrOJ, fmt='ro', label = 'OJ287')
    if len(tJ1)> 0:
        plt.errorbar(tJ1,cpJ1,xerr=0, yerr = 1.*yerrJ1, fmt='go', label = 'J1924-2914')
    if len(t10)> 0:
        plt.errorbar(t10,cp10,xerr=0, yerr = 1.*yerr10, fmt='mo', label = '1055+018')
    if len(t17)> 0:
        plt.errorbar(t17,cp17,xerr=0, yerr = 1.*yerr17, fmt='ko', label = '1921-293')
    if len(t19)> 0:
        plt.errorbar(t19,cp19,xerr=0, yerr = 1.*yerr19, fmt='yo', label = 'CYGX-3')
    if len(tJ17)> 0:
        plt.errorbar(tJ17,cpJ17,xerr=0, yerr = 1.*yerrJ17, fmt='sb', label = 'J1733-1304')
    if len(tCE)> 0:
        plt.errorbar(tCE,cpCE,xerr=0, yerr = 1.*yerrCE, fmt='co', label = 'CENA')

    plt.xlabel('time',fontsize=15)
    plt.ylabel('closure phase [deg]',fontsize=15)
    plt.axhline(y=0,linewidth=2, color='k')
    plt.axvline(x=24.,linewidth=1,color='k',linestyle='--')
    plt.axvline(x=48.,linewidth=1,color='k',linestyle='--')
    plt.axis([np.amin(np.concatenate((t3C,tOJ,tJ1,t10,t17,t19,tJ17,tCE)))-0.5,np.amax(np.concatenate((t3C,tOJ,tJ1,t10,t17,t19,tJ17,tCE)))+0.5,-MaxAngle,MaxAngle])
    plt.title(str(Triangle)+', '+pol, fontsize=15)
    plt.legend()
    plt.show()
    if printSummary != 0:
        PrintSummaryTri(bsp)



def plotQuadranglePolar(alist,Quadrangle,pol,MaxErr=2.,method=0,Signature = [1,1,1,1]):

    DataLabel='amp'
    ErrorLabel = 'sigma'
    alist = DataQuadrangle(alist,Quadrangle,pol,method,Signature)
    days = [3597,3600,3601]; source = '3C279'
    #hrs = [0,24.,48.] #to add to time
    t3C,cp3C,yerr3C = GetPlotData(alist,source,DataLabel,ErrorLabel,days,hrs)
    source = 'OJ287'
    tOJ,cpOJ,yerrOJ = GetPlotData(alist,source,DataLabel,ErrorLabel,days,hrs)
    source = 'J1924-2914'
    tJ1,cpJ1,yerrJ1 = GetPlotData(alist,source,DataLabel,ErrorLabel,days,hrs)
    source = '1055+018'
    t10,cp10,yerr10 = GetPlotData(alist,source,DataLabel,ErrorLabel,days,hrs)
    source = '1749+096'
    t17,cp17,yerr17 = GetPlotData(alist,source,DataLabel,ErrorLabel,days,hrs)
    source = '1921-293'
    t19,cp19,yerr19 = GetPlotData(alist,source,DataLabel,ErrorLabel,days,hrs)
    source = 'CYGX-3'
    tCY,cpCY,yerrCY = GetPlotData(alist,source,DataLabel,ErrorLabel,days,hrs)
    source = 'J1733-1304'
    tJ17,cpJ17,yerrJ17 = GetPlotData(alist,source,DataLabel,ErrorLabel,days,hrs)
    source = 'CENA'
    tCE,cpCE,yerrCE = GetPlotData(alist,source,DataLabel,ErrorLabel,days,hrs)

    #plot
    #-----------------------------------------------
    plt.figure(figsize=(15,6))
    if len(t3C)> 0:
        plt.errorbar(t3C,cp3C,xerr=0, yerr = 1.*yerr3C, fmt='bo', label = '3C279')
    if len(tOJ)> 0:
        plt.errorbar(tOJ,cpOJ,xerr=0, yerr = 1.*yerrOJ, fmt='ro', label = 'OJ287')
    if len(tJ1)> 0:
        plt.errorbar(tJ1,cpJ1,xerr=0, yerr = 1.*yerrJ1, fmt='go', label = 'J1924-2914')
    if len(t10)> 0:
        plt.errorbar(t10,cp10,xerr=0, yerr = 1.*yerr10, fmt='mo', label = '1055+018')
    if len(t17)> 0:
        plt.errorbar(t17,cp17,xerr=0, yerr = 1.*yerr17, fmt='ko', label = '1921-293')
    if len(t19)> 0:
        plt.errorbar(t19,cp19,xerr=0, yerr = 1.*yerr19, fmt='yo', label = 'CYGX-3')
    if len(tJ17)> 0:
        plt.errorbar(tJ17,cpJ17,xerr=0, yerr = 1.*yerrJ17, fmt='sb', label = 'J1733-1304')
    if len(tCE)> 0:
        plt.errorbar(tCE,cpCE,xerr=0, yerr = 1.*yerrCE, fmt='co', label = 'CENA')

    plt.xlabel('time',fontsize=15)
    #plt.xticks(x, my_xticks)
    plt.ylabel('closure amplitudes',fontsize=15)
    plt.axhline(y=1.,linewidth=2, color='k')
    plt.axvline(x=24.,linewidth=1,color='k',linestyle='--')
    plt.axvline(x=48.,linewidth=1,color='k',linestyle='--')
    plt.axis([np.amin(np.concatenate((t3C,tOJ,tJ1,t10,t17,t19,tJ17,tCE)))-0.5,np.amax(np.concatenate((t3C,tOJ,tJ1,t10,t17,t19,tJ17,tCE)))+0.5,1./(1.+MaxErr),1.+MaxErr])
    plt.title(str(Quadrangle)+', '+pol, fontsize=15)
    plt.legend()
    plt.show()
    PrintSummaryQuad(alist)
    

def plotBaseline(alist,basename,pol,DataLabel='snr',ErrorLabel='snrstd',logscaley=False):

    #DataLabel='amp'
    #ErrorLabel = 'sigma'
    alist = DataBaseline(alist,basename,pol)
    days = [3597,3600,3601]; source = '3C279'
    #hrs = [0,24.,48.] #to add to time
    t3C,cp3C,yerr3C = GetPlotData(alist,source,DataLabel,ErrorLabel,days,hrs)
    source = 'OJ287'
    tOJ,cpOJ,yerrOJ = GetPlotData(alist,source,DataLabel,ErrorLabel,days,hrs)
    source = 'J1924-2914'
    tJ1,cpJ1,yerrJ1 = GetPlotData(alist,source,DataLabel,ErrorLabel,days,hrs)
    source = '1055+018'
    t10,cp10,yerr10 = GetPlotData(alist,source,DataLabel,ErrorLabel,days,hrs)
    source = '1749+096'
    t17,cp17,yerr17 = GetPlotData(alist,source,DataLabel,ErrorLabel,days,hrs)
    source = '1921-293'
    t19,cp19,yerr19 = GetPlotData(alist,source,DataLabel,ErrorLabel,days,hrs)
    source = 'CYGX-3'
    tCY,cpCY,yerrCY = GetPlotData(alist,source,DataLabel,ErrorLabel,days,hrs)
    source = 'J1733-1304'
    tJ17,cpJ17,yerrJ17 = GetPlotData(alist,source,DataLabel,ErrorLabel,days,hrs)
    source = 'CENA'
    tCE,cpCE,yerrCE = GetPlotData(alist,source,DataLabel,ErrorLabel,days,hrs)

    #plot
    #-----------------------------------------------
    plt.figure(figsize=(15,6))
    if len(t3C)> 0:
        plt.errorbar(t3C,cp3C,xerr=0, yerr = 1.*yerr3C, fmt='bo', label = '3C279')
    if len(tOJ)> 0:
        plt.errorbar(tOJ,cpOJ,xerr=0, yerr = 1.*yerrOJ, fmt='ro', label = 'OJ287')
    if len(tJ1)> 0:
        plt.errorbar(tJ1,cpJ1,xerr=0, yerr = 1.*yerrJ1, fmt='go', label = 'J1924-2914')
    if len(t10)> 0:
        plt.errorbar(t10,cp10,xerr=0, yerr = 1.*yerr10, fmt='mo', label = '1055+018')
    if len(t17)> 0:
        plt.errorbar(t17,cp17,xerr=0, yerr = 1.*yerr17, fmt='ko', label = '1921-293')
    if len(t19)> 0:
        plt.errorbar(t19,cp19,xerr=0, yerr = 1.*yerr19, fmt='yo', label = 'CYGX-3')
    if len(tJ17)> 0:
        plt.errorbar(tJ17,cpJ17,xerr=0, yerr = 1.*yerrJ17, fmt='sb', label = 'J1733-1304')
    if len(tCE)> 0:
        plt.errorbar(tCE,cpCE,xerr=0, yerr = 1.*yerrCE, fmt='co', label = 'CENA')

    plt.xlabel('time',fontsize=15)
    #plt.xticks(x, my_xticks)
    if logscaley==True:
        plt.yscale('log')
    plt.ylabel(DataLabel+' in 5s',fontsize=15)
    if (DataLabel=='t_coh_bias')|(DataLabel=='t_coh'):
        plt.ylabel(DataLabel,fontsize=15)
    #plt.axhline(y=1.,linewidth=2, color='k')
    plt.axvline(x=24.,linewidth=1,color='k',linestyle='--')
    plt.axvline(x=48.,linewidth=1,color='k',linestyle='--')
    plt.xlim((np.amin(np.concatenate((t3C,tOJ,tJ1,t10,t17,t19,tJ17,tCE)))-0.5,np.amax(np.concatenate((t3C,tOJ,tJ1,t10,t17,t19,tJ17,tCE)))+0.5,))
    plt.title(str(basename)+', '+pol, fontsize=15)
    plt.legend()
    plt.show()

    



def PreparePlot(bsp,source,days,hrs):
    t = []; cp =[]; yerr = []
    for x in range(len(days)):
        t.append(hrs[x] + np.asarray(bsp.loc[(bsp.index.get_level_values('source') == source)&(bsp.index.get_level_values('expt_no') == days[x]),['gmst']]))
        cp.append(np.asarray(bsp.loc[(bsp.index.get_level_values('source') == source)&(bsp.index.get_level_values('expt_no') == days[x]),['cphase']]))
        yerr.append(np.asarray(bsp.loc[(bsp.index.get_level_values('source') == source)&(bsp.index.get_level_values('expt_no') == days[x]),['sigmaCP']]))
    tf = np.concatenate(t).ravel()
    cpf = np.concatenate(cp).ravel()
    yerrf = np.concatenate(yerr).ravel()
    return tf, cpf, yerrf

def GetPlotData(alist,source,DataLabel,ErrorLabel,days,hrs):
    t = []; dat =[]; yerr = []
    for x in range(len(days)):
        tfoo = np.asarray(alist.loc[(alist.index.get_level_values('source') == source)&(alist.index.get_level_values('expt_no') == days[x]),['gmst']])
        tfoo =  tfoo + hrs[x]
        t.append(tfoo)
        dat.append(np.asarray(alist.loc[(alist.index.get_level_values('source') == source)&(alist.index.get_level_values('expt_no') == days[x]),[DataLabel]]))
        yerr.append(np.asarray(alist.loc[(alist.index.get_level_values('source') == source)&(alist.index.get_level_values('expt_no') == days[x]),[ErrorLabel]]))
    tf = np.concatenate(t).ravel()
    datf = np.concatenate(dat).ravel()
    yerrf = np.concatenate(yerr).ravel()
    return tf, datf, yerrf


def PrintSummaryQuad(alist):
    #alist with just this quadrangle
    qaf = alist
    sigmaLim = 0.25
    n1 = len(qaf['amp'])
    n2 = len(qaf.loc[qaf['sigma']<sigmaLim,'sigma'])
    n3 = len(qaf.loc[(qaf['sigma']<sigmaLim)&(np.abs(qaf['amp'] - 1. )<3.*qaf['sigma']),'sigma'])
    n4 = len(qaf.loc[(np.abs(qaf['amp'] - 1. )<0.1)&(np.abs(qaf['amp'] - 1. )>=3.*qaf['sigma']),'sigma'])
    n5 = len(qaf.loc[(qaf['sigma']>=sigmaLim)&(np.abs(qaf['amp'] - 1. )<3.*qaf['sigma']),'sigma'])
    print('Total scans: ', n1)
    print('Scans with sigma <', sigmaLim,': ', n2)
    print('Scans with sigma <', sigmaLim,', consistent with 4AMP==1 within 3 sigma: ', n3)
    print('Scans inconsistent with 4AMP==1 within 3 sigma, but error smaller than 0.1: ', n4)
    print('Scans consistent with 4AMP==1 within 3 sigma, but sigma > ', sigmaLim,': ', n5)


def PrintSummaryTri(alist):
    #alist with just this quadrangle
    qaf = alist
    sigmaLim = 2.5 #deg
    n1 = len(qaf['cphase'])
    n2 = len(qaf.loc[qaf['sigmaCP']<sigmaLim,'sigmaCP'])
    n3 = len(qaf.loc[(qaf['sigmaCP']<sigmaLim)&(np.abs(qaf['cphase'] - 0. )<3.*qaf['sigmaCP']),'sigmaCP'])
    n4 = len(qaf.loc[(np.abs(qaf['cphase'] - 0.)< 1.5)&(np.abs(qaf['cphase'] - 0. )>=3.*qaf['sigmaCP']),'sigmaCP'])
    n5 = len(qaf.loc[(qaf['sigmaCP']>=sigmaLim)&(np.abs(qaf['cphase'] - 0. )<3.*qaf['sigmaCP']),'sigmaCP'])
    print('Total scans: ', n1)
    print('Scans with sigma <', sigmaLim,'deg: ', n2)
    print('Scans with sigma <', sigmaLim,'deg, consistent with CP==0 within 3 sigma: ', n3)
    print('Scans inconsistent with CP==0 within 3 sigma, but error smaller than 1.5 deg: ', n4)
    print('Scans consistent with CP==0 within 3 sigma, but sigma > ', sigmaLim,'deg: ', n5)


def DataTriangle3(alist,Tri,signat,pol):
    condP = (alist['polarization']==pol)
    condB1 = (alist['baseline']==Tri[0])
    condB2 = (alist['baseline']==Tri[1])
    condB3 = (alist['baseline']==Tri[2])
    condB = condB1|condB2|condB3

    alistRR_Tri = alist.loc[condP&condB,['expt_no','scan_id','source','datetime','baseline','total_phas','amp','snr','gmst']]
    #throw away times without full triangle
    tlist = alistRR_Tri.groupby('datetime').filter(lambda x: len(x) > 2)
    tlist.loc[:,'sigma'] = (tlist.loc[:,'amp']/(tlist.loc[:,'snr']))
    for cou in range(3):
        tlist.loc[(tlist.loc[:,'baseline']==Tri[cou]),'total_phas'] *= signat[cou]*np.pi/180.
    #tlist
    #print('scan_idttt', sorted(list(set(tlist.loc[tlist['scan_id']==1,'gmst']))))
    #print(set(alistRR_Tri['gmst']))
    tlist.loc[:,'sigma'] = 1./tlist.loc[:,'snr']**2 #put 1/snr**2 in the sigma column to aggregate
    #first aggregation to get bispectra
    bsp = tlist.groupby(('expt_no','source','scan_id','datetime')).agg({'total_phas': lambda x: np.sum(x),
        'amp': lambda x: np.prod(x), 'sigma': lambda x: np.sqrt(np.sum(x)),'gmst': 'min'})
    #sigma above is the CLOSURE PHASE ERROR
    #print('scan_idttt', list(set(tlist.loc[bsp['scan_id']==1,'gmst'])).sorted)
    bsp.loc[:,'bisp'] = bsp.loc[:,'amp']*np.exp(1j*bsp.loc[:,'total_phas'])
    bsp.loc[:,'snr'] = 1./bsp.loc[:,'sigma']
    bsp.loc[:,'sigma'] = bsp.loc[:,'amp']*bsp.loc[:,'sigma'] #sigma of bispectrum

    #second aggregation to average the bispectra
    #print('ddd', set(bsp['gmst']))
    #print('scan_id', set(bsp['scan_id']))
    #bsp = bsp.groupby(('expt_no','source','scan_id')).agg({'bisp': np.average,'gmst' :'min','sigma': lambda x: np.sqrt(sum(x**2))/len(x)})
    #print('eee', set(bsp['gmst']))
    bsp.loc[:,'cphase'] = np.angle(bsp.loc[:,'bisp'])*180./np.pi #deg
    bsp.loc[:,'amp'] = np.abs(bsp.loc[:,'bisp'])
    bsp.loc[:,'snr'] = bsp.loc[:,'amp']/bsp.loc[:,'sigma']
    bsp.loc[:,'sigmaCP'] = 1./bsp.loc[:,'snr']*180./np.pi #deg
    #del bsp['bisp']
    bsp = bsp.reset_index()
    return bsp


def DataQuadrangle3(alist,Quad,pol, debias=1):

    condP = (alist['polarization']==pol)
    condB1 = (alist['baseline']==Quad[0])
    condB2 = (alist['baseline']==Quad[1])
    condB3 = (alist['baseline']==Quad[2])
    condB4 = (alist['baseline']==Quad[3])
    condB = condB1|condB2|condB3|condB4

    alist_Quad = alist.loc[condP&condB,['expt_no','scan_id','source','datetime','baseline','total_phas','amp','snr','gmst']]
    tlist = alist_Quad.groupby('datetime').filter(lambda x: len(x) > 3)
    tlist.loc[:,'sigma'] = tlist.loc[:,'amp']/(tlist.loc[:,'snr'])
    tlist.loc[:,'datetime_foo'] = tlist.loc[:,'datetime']
    #if debiasing amplitudes
    if debias != 0:
        tlist.loc[:,'amp'] = tlist.loc[:,'amp']*np.sqrt(1.- debias*2./tlist.loc[:,'snr']**2 )
        
    #####################################################################
    #form quadAmplitudes on 5s and average quadAmplitudes over whole scan
    #####################################################################
    
    for cou in range(2,4):
    #inverse amplitude for the visibilities in the denominator
        tlist.loc[(tlist.loc[:,'baseline']==Quad[cou]),'amp'] = 1./tlist.loc[(tlist.loc[:,'baseline']==Quad[cou]),'amp']

    #aggregating to get quadProducts on 5s segments 
    tlist.loc[:,'sigma'] = 1./tlist.loc[:,'snr']**2#put snr in place of sigma to sum for 4product

    quadAmp = tlist.groupby(('expt_no','source','scan_id','datetime_foo')).agg({'amp': lambda x: np.prod(x), 
                            'gmst': 'min', 'datetime': 'min', 'sigma' : lambda x: np.sqrt(np.sum(x)) })
    quadAmp.loc[:,'sigma'] = quadAmp.loc[:,'sigma']*quadAmp.loc[:,'amp']
    return quadAmp

def get_closure_phases(path_alist, tcoh, tav ='scan',phaseType='phase'):

    falist = pd.read_pickle(path_ailist)
    falist_tcoh = a2a.coh_average(falist, tcoh)
    bisp_LL = all_bispectra_polar(falist_tcoh,'LL',phaseType)
    bisp_RR = all_bispectra_polar(falist_tcoh,'RR',phaseType)
    bisp_LL = coh_average_bsp(bisp_LL,tav)
    bisp_RR = coh_average_bsp(bisp_RR,tav)
    bisp_LL= use_measured_circ_std_as_sigmaCP(bisp_LL)
    bisp_RR = use_measured_circ_std_as_sigmaCP(bisp_RR)
    bisp = pd.concat([bisp_LL, bisp_RR],ignore_index=True)

def check_match(fooH,fooA):
    #CHECK QUALITY OF MATCHING AIPS HOPS, only last one should be false
    print([all(fooH['datetime']==fooA['datetime']),
    all(fooH['baseline']==fooA['baseline']),
    all(fooH['polarization']==fooA['polarization']),
    all(fooH['band']==fooA['band'])])


def all_polar_line(alist):
    '''
    forms baseline-based closure quantity
    (RL*)(LR*)/(RR*)(LL*)
    '''
    if 'scan_id' not in alist.columns:
        alist.loc[:,'scan_id'] = alist.loc[:,'scan_no_tot']
    if 'band' not in alist.columns:
        alist.loc[:,'band'] = [None]*np.shape(alist)[0]

    #select only scans/baselines with 4 polarized components
    polrats =data.groupby(('datetime','scan_id','expt_no','baseline','band')).filter(lambda x: len(x)==4)


    bands = list(polrats.band.unique())
    expts = list(polrats.expt_no.unique())
    baselines = list(polrats.baseline.unique())

    for band in bands:
        for expt in expts:
            for baseline in baselines:
                fooLR = polrats[polrats.polarization=='LR']
                fooRL = polrats[polrats.polarization=='RL']
                fooLL = polrats[polrats.polarization=='LL']
                fooRR = polrats[polrats.polarization=='RR']



    

def all_cpol(alist,what_phase='resid_phas'):
    alist.groupby(('band','datetime','baseline')).filter(lambda x: len(x) == 4)

    cols = ['datetime','band','baseline']
    vis_RR = alist[alist.polarization=='RR'].sort_values(['band','datetime','baseline'])['amp',what_phase]
    vis_LL = alist[alist.polarization=='LL'].sort_values(['band','datetime','baseline'])['amp',what_phase]
    vis_LR = alist[alist.polarization=='LR'].sort_values(['band','datetime','baseline'])['amp',what_phase]
    vis_RL = alist[alist.polarization=='RL'].sort_values(['band','datetime','baseline'])['amp',what_phase]

    data = alist[alist.polarization=='RR'].sort_values(['band','datetime','baseline'])
    

def save_cp(bsp,folder,sourL='all'):

    if sourL!='all':
        bsp = bsp[list(map(lambda x:  x in sourL,bsp.source))]
    bsp = ut.add_mjd(bsp)

    exptL = list(bsp.expt_no.unique())
    for expt in exptL:
        bspE = bsp[bsp.expt_no==expt]
        bandL = list(bspE.band.unique())
        for band in bandL:
            bspEBa = bspE[bspE.band==band]
            polarL = list(bspEBa.polarization.unique())
            for polar in polarL:
                bspEBaP = bspEBa[bspEBa.polarization==polar]
                sourceL = list(bspEBaP.source.unique())
                for source in sourceL:
                    bspEBaPS = bspEBaP[bspEBaP.source==source]
                    triangleL = list(bspEBaPS.triangle.unique())
                    for triangle in triangleL:
                        bspEBaPST = bspEBaPS[bspEBaPS.triangle==triangle]
                        nameLoc = folder+source+'_'+triangle+'_'+str(expt)+'_'+polar+'_'+band+'.csv'
                        foo = bspEBaPST[['mjd','cphase','sigmaCP']]
                        foo.to_csv(nameLoc,header=False,index=False)



def save_lca(quad,folder,sourL='all'):

    if sourL!='all':
        quad = quad[list(map(lambda x:  x in sourL,quad.source))]
    quad = ut.add_mjd(quad)

    exptL = list(quad.expt_no.unique())
    for expt in exptL:
        quadE = quad[quad.expt_no==expt]
        bandL = list(quadE.band.unique())
        for band in bandL:
            quadEBa = quadE[quadE.band==band]
            polarL = list(quadEBa.polarization.unique())
            for polar in polarL:
                quadEBaP = quadEBa[quadEBa.polarization==polar]
                sourceL = list(quadEBaP.source.unique())
                for source in sourceL:
                    quadEBaPS = quadEBaP[quadEBaP.source==source]
                    quadrangleL = list(quadEBaPS.quadrangle.unique())
                    for quadr in quadrangleL:
                        quadEBaPST = quadEBaPS[quadEBaPS.quadrangle==quadr]
                        strquad = str(quadr[0])+'_'+str(quadr[1])+'_'+str(quadr[2])+'_'+str(quadr[3])
                        nameLoc = folder+source+'_'+strquad+'_'+str(expt)+'_'+polar+'_'+band+'.csv'
                        foo = quadEBaPST[['mjd','logamp','sigma']]
                        foo.to_csv(nameLoc,header=False,index=False)
                        

def get_closepols(data):
    
    if 'sigma' not in data.columns:
        data['sigma']=data['amp']/data['snr']
    if 'mjd' not in data.columns:
        data = ut.add_mjd(data)

    data=data.groupby(['datetime','scan_id','band','baseline','expt_no']).filter(lambda x: len(x) == 4)
    fooRL = data[(data.polarization=='RL')].sort_values(['datetime','band','baseline']).reset_index()
    fooLR = data[(data.polarization=='LR')].sort_values(['datetime','band','baseline']).reset_index()
    fooRR = data[(data.polarization=='RR')].sort_values(['datetime','band','baseline']).reset_index()
    fooLL = data[(data.polarization=='LL')].sort_values(['datetime','band','baseline']).reset_index()
    
    fooRR['fracpol'] = np.sqrt(np.asarray(fooLR.amp)*np.asarray(fooRL.amp)/np.asarray(fooRR.amp)/np.asarray(fooLL.amp))
    fooRR['sigma'] = 0.5*np.asarray(fooRL.amp)*np.asarray(fooLR.sigma)/np.asarray(fooRR.amp)/np.asarray(fooLL.amp)
    + 0.5*np.asarray(fooRL.sigma)*np.asarray(fooLR.amp)/np.asarray(fooRR.amp)/np.asarray(fooLL.amp)
    + 0.5*np.asarray(fooRL.amp)*np.asarray(fooLR.amp)/np.asarray(fooRR.amp)/(np.asarray(fooLL.amp)**2)*np.asarray(fooLL.sigma)
    + 0.5*np.asarray(fooRL.amp)*np.asarray(fooLR.amp)/(np.asarray(fooRR.amp)**2)/np.asarray(fooLL.amp)*np.asarray(fooRR.sigma)
    
    return fooRR[['mjd','datetime','fracpol','sigma','baseline','scan_id','band','expt_no','source']].copy()

def get_logclosepols(data):
    
    if 'sigma' not in data.columns:
        data['sigma']=data['amp']/data['snr']
    if 'mjd' not in data.columns:
        data = ut.add_mjd(data)

    #print('data shape ',np.shape(data))
    #print('data shape ',data.columns)
    data=data.groupby(['datetime','scan_id','band','baseline','expt_no']).filter(lambda x: len(x) == 4)
    fooRL = data[(data.polarization=='RL')].sort_values(['datetime','band','baseline']).reset_index()
    fooLR = data[(data.polarization=='LR')].sort_values(['datetime','band','baseline']).reset_index()
    fooRR = data[(data.polarization=='RR')].sort_values(['datetime','band','baseline']).reset_index()
    fooLL = data[(data.polarization=='LL')].sort_values(['datetime','band','baseline']).reset_index()
    #print('data shape ',np.shape(data))

    fooRR['fracpol'] = np.log(np.asarray(fooLR.amp))+np.log(np.asarray(fooRL.amp)) - np.log(np.asarray(fooRR.amp)) - np.log(np.asarray(fooLL.amp))
    fooRR['sigma'] = np. sqrt(1./fooLR.snr**2+1./fooRL.snr**2+1./fooRR.snr**2+1./fooLL.snr**2)
    #+ 0.5*np.asarray(fooRL.sigma)*np.asarray(fooLR.amp)/np.asarray(fooRR.amp)/np.asarray(fooLL.amp)
    #+ 0.5*np.asarray(fooRL.amp)*np.asarray(fooLR.amp)/np.asarray(fooRR.amp)/(np.asarray(fooLL.amp)**2)*np.asarray(fooLL.sigma)
    #+ 0.5*np.asarray(fooRL.amp)*np.asarray(fooLR.amp)/(np.asarray(fooRR.amp)**2)/np.asarray(fooLL.amp)*np.asarray(fooRR.sigma)
    
    return fooRR[['mjd','datetime','fracpol','sigma','baseline','scan_id','band','expt_no','source']].copy()
     

def get_snr_help(Esnr):
    """estimates snr given a single biased snr measurement
    """
    if Esnr**2 >= 1.0: 
        #return np.sqrt(Esnr**2 - 1.0)
        fun = lambda x: x
        return np.sqrt(Esnr**2 - 1.0)

    else:
        return 0.0

def get_snr(Esnr):
    """"applies get_snr_help on vector
    """
    if type(Esnr) == float or type(Esnr)==np.float64 or type(Esnr)==int:
        return get_snr_help(Esnr)
    else:
        return np.asarray(list(map(get_snr_help,Esnr)))

import scipy.special as ss
    
def log_debias(snr0):
    if type(snr0) == float or type(snr0)==np.float64 or type(snr0)==int:
        return -ss.expi(-snr0**2/2.)/2.
    else:
        snr0 = np.asarray(snr0)
        snr0[snr0<=0]=10000
        ld = -ss.expi(-snr0**2/2.)/2.
        ld[ld!=ld] = 0
        return ld


def debias_A_in_log(A,sigma):
    '''
    given vector of Rice-distributed amplitudes A and 
    associated thermal errors sigma, get A0
    '''
    if type(A) == float or type(A)==np.float64 or type(A)==int:
        A= [A]
    if type(sigma) == float or type(sigma)==np.float64 or type(sigma)==int:
        sigma=[sigma]
    A = np.asarray(A)
    sigma=np.asarray(sigma)
    import scipy.optimize as so
    rho = A/sigma
    fun = lambda x: np.mean(np.log(rho)) - (np.log(x) - 0.5* ss.expi(-x**2/2.))
    x1 = 0.057967 #min real snr
    x2 = 30. #max real snr
    #print(fun(x1),fun(x2))
    rho0 = so.brentq(fun,x1,x2)
    A0 = rho0*np.mean(sigma)
    return A0

def debias_A_in_lin(A,sigma):
    if type(A) == float or type(A)==np.float64 or type(A)==int:
        A= [A]
    if type(sigma) == float or type(sigma)==np.float64 or type(sigma)==int:
        sigma=[sigma]
    amp = np.asarray(A)
    sigma=np.asarray(sigma)
    A02 = np.sum(amp**2 - ( 2. -1./len(amp) )*sigma**2)/len(amp)
    A02 = np.maximum(A02,0.)
    amp_out = np.sqrt(A02)
    return amp_out



def all_quadruples_ultimate(alist0,tavg='scan',ctype='camp',debias=True):
    '''
    ctype = 'camp' or 'logcamp'
    debias = True/False
    '''

    #0. PREPARE DATAFRAME
    alist0 = alist0[(alist0['polarization']=='LL')|(alist0['polarization']=='RR')]
    
    if 'band' not in alist0.columns:
        alist0.loc[:,'band'] = 'unknown'
    if 'scan_id' not in alist0.columns:
        alist0.loc[:,'scan_id'] = alist0.loc[:,'scan_no_tot']
    if 'sigma' not in alist0.columns:
        alist0['sigma'] = alist0['amp']/alist0['snr']
    if 'phase' not in alist0.columns:
        alist0['phase'] = alist0['resid_phas']

    #1. AVERAGE WITH DEBIASING
    #print(debias)
    alist = ut.incoh_avg_vis(alist0.copy(),tavg=tavg,columns_out0=[],phase_type='phase',debias=debias, robust=False)

    #2. CONSTRUCT CLOSURES
    quaL = list_all_quadrangles(alist)
    quad_baseL = sorted(quadrangles2baselines(quaL,alist))
    quad_out = pd.DataFrame({})

    #select quadrangle
    for cou in range(len(quad_baseL)):
        Quad = quad_baseL[cou]
        condB0 = (alist['baseline']==Quad[0])
        condB1 = (alist['baseline']==Quad[1])
        condB2 = (alist['baseline']==Quad[2])
        condB3 = (alist['baseline']==Quad[3])
        condB = condB0|condB1|condB2|condB3
        alist_Quad = alist.loc[condB,['expt_no','scan_id','source','datetime','baseline','polarization','amp','snr','gmst','band','sigma']]
        print(Quad, np.shape(alist_Quad)[0])
        #throw away times without full quadrangle
        tlist = alist_Quad.groupby(('polarization','band','datetime')).filter(lambda x: len(x) == 4)
    
        # prepare amplitudes (last two go into denominator)
        tlist['camp'] = 0
        power=[1,1,-1,-1]
        for cou2 in range(4):
            tlist.loc[(tlist.loc[:,'baseline']==Quad[cou2]),'camp'] = (tlist.loc[(tlist.loc[:,'baseline']==Quad[cou2]),'amp'])**power[cou2]

        grouping =  ['expt_no','band','polarization','source','scan_id','datetime']
        aggregate = {'sigmaCA': lambda x: np.sqrt(np.sum(x**2)),'snr': np.min}

        if ctype=='camp':
            tlist['sigmaCA'] = tlist['sigma']/tlist['amp'] # 1/snr
            aggregate['camp'] = np.prod
            
        elif ctype=='logcamp':
            tlist['sigmaCA'] = tlist['sigma']/tlist['amp'] # 1/snr
            aggregate['camp'] = lambda x: np.sum(np.log(x))
    
        #actual formation of camp
        #print(tlist.columns)
        quadlist = tlist.groupby(grouping).agg(aggregate)

        #if camp we need to multiply sigmaCA by CA
        if ctype=='camp':
            quadlist['sigmaCA'] = quadlist['sigmaCA']*quadlist['camp']
        '''
        # debiasing in logcamp
        if debias=='camp':
           
            if ctype=='camp':
                quadlist['camp'] = quadlist['camp']*np.exp( - log_debias(quadlist['snr1']) - log_debias(quadlist['snr2']) + log_debias(quadlist['snr3']) + log_debias(quadlist['snr4']) )
            elif ctype=='logcamp':
                quadlist['camp'] = quadlist['camp'] - log_debias(quadlist['snr1']) - log_debias(quadlist['snr2']) + log_debias(quadlist['snr3']) + log_debias(quadlist['snr4'])
        '''
        #print(quadlist.columns)
        quadlist = quadlist.reset_index()
        quadlist['quadrangle'] = quadrangle2str(Quad)
        quad_out = pd.concat([quad_out, quadlist],ignore_index=True)

    quad_out = quad_out.reset_index(drop=True)
    return quad_out
