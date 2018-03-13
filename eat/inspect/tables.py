import numpy as np
import pandas as pd
import itertools
from eat.inspect.closures import *

def polar_CP_error_hi_lo(bisp_rr_hi, bisp_ll_hi, bisp_rr_lo, bisp_ll_lo):
    #consistency in polarizations, separately for hi and lo
    bisp_rr_hi, bisp_ll_hi = match_2_dataframes(bisp_rr_hi, bisp_ll_hi, 'triangle')
    bisp_rr_lo, bisp_ll_lo = match_2_dataframes(bisp_rr_lo, bisp_ll_lo, 'triangle')
    
    sigmaDif = np.sqrt(np.asarray(bisp_rr_hi['sigmaCP'])**2+np.asarray(bisp_ll_hi['sigmaCP'])**2)
    bisp_rr_hi['TotErr'] = np.abs(((np.asarray(bisp_rr_hi['cphase'])-np.asarray(bisp_ll_hi['cphase']))))
    bisp_rr_hi['TotErr'] = np.minimum(np.asarray(bisp_rr_hi['TotErr']),np.abs(np.asarray(bisp_rr_hi['TotErr']) -360))
    bisp_rr_hi['RelErr'] = np.asarray(bisp_rr_hi['TotErr'])/sigmaDif
    sigmaDif = np.sqrt(np.asarray(bisp_rr_lo['sigmaCP'])**2+np.asarray(bisp_ll_lo['sigmaCP'])**2)
    bisp_rr_lo['TotErr'] = np.abs(((np.asarray(bisp_rr_lo['cphase'])-np.asarray(bisp_ll_lo['cphase']))))
    bisp_rr_lo['TotErr'] = np.minimum(np.asarray(bisp_rr_lo['TotErr']),np.abs(np.asarray(bisp_rr_lo['TotErr']) -360))
    bisp_rr_lo['RelErr'] = np.asarray(bisp_rr_lo['TotErr'])/sigmaDif

    TriStat = pd.DataFrame({})
    AllTri = sorted(list(set(bisp_rr_hi.triangle)))
    TriStat['triangle'] = AllTri
    scans_RR_LL_lo = [np.shape(bisp_rr_lo[bisp_rr_lo.triangle == Tri])[0] for Tri in AllTri]
    scans_RR_LL_hi = [np.shape(bisp_rr_hi[bisp_rr_hi.triangle == Tri])[0] for Tri in AllTri]
    scans_RR_LL_lo_3sig = [np.shape(bisp_rr_lo[(bisp_rr_lo.triangle == Tri)&(bisp_rr_lo.RelErr < 3.)])[0] for Tri in AllTri]
    scans_RR_LL_hi_3sig = [np.shape(bisp_rr_hi[(bisp_rr_hi.triangle == Tri)&(bisp_rr_hi.RelErr < 3.)])[0] for Tri in AllTri]
    TriStat['sc_lo'] = scans_RR_LL_lo
    TriStat['sc__hi'] = scans_RR_LL_hi
    TriStat['sc_total'] = np.asarray(scans_RR_LL_hi)+np.asarray(scans_RR_LL_lo)
    TriStat['sc_3sig_lo'] = scans_RR_LL_lo_3sig
    TriStat['sc_3sig_hi'] = scans_RR_LL_hi_3sig
    TriStat['sc_3sig'] = np.asarray(scans_RR_LL_hi_3sig)+np.asarray(scans_RR_LL_lo_3sig)
    TriStat['sc_3sig_proc'] = np.asarray(map(float,TriStat['sc_3sig']))/np.asarray(map(float,TriStat['sc_total']))
    #TriStat['MedianAbs_lo'] = [np.median(np.asarray(bisp_rr_lo.loc[bisp_rr_lo['triangle'] == Tri,['TotErr']])) for Tri in AllTri]
    #TriStat['MedianAbs_hi'] = [np.median(np.asarray(bisp_rr_hi.loc[bisp_rr_hi['triangle'] == Tri,['TotErr'] ])) for Tri in AllTri]
    TriStat['MedianAbs'] = [np.median(np.asarray(list(bisp_rr_lo.loc[bisp_rr_lo['triangle'] == Tri,'TotErr'])+list(bisp_rr_hi.loc[bisp_rr_hi['triangle'] == Tri,'TotErr' ]))) for Tri in AllTri]
    TriStat['MedianSigma'] = [np.median(np.asarray(list(bisp_rr.loc[bisp_rr['triangle'] == Tri,'sigmaCP']))) for Tri in AllTri]
    TriStat = TriStat.sort_values('sc_3sig_proc')

    return TriStat

def band_CP_error_rr_ll(bisp_rr_hi, bisp_ll_hi, bisp_rr_lo, bisp_ll_lo):
    #matches 1st with 3rd and 2nd with 4th
    #consistency in bands, separately for rr and ll
    bisp_rr_hi, bisp_rr_lo = match_2_dataframes(bisp_rr_hi, bisp_rr_lo, 'triangle')
    bisp_ll_hi, bisp_ll_lo = match_2_dataframes(bisp_ll_hi, bisp_ll_lo, 'triangle')
    
    #rr
    sigmaDif = np.sqrt(np.asarray(bisp_rr_hi['sigmaCP'])**2+np.asarray(bisp_rr_lo['sigmaCP'])**2)
    bisp_rr_hi['TotErr'] = np.abs(((np.asarray(bisp_rr_hi['cphase'])-np.asarray(bisp_rr_lo['cphase']))))
    bisp_rr_hi['TotErr'] = np.minimum(np.asarray(bisp_rr_hi['TotErr']),np.abs(np.asarray(bisp_rr_hi['TotErr']) -360))
    bisp_rr_hi['RelErr'] = np.asarray(bisp_rr_hi['TotErr'])/sigmaDif
    bisp_rr_hi['sigmaDif'] = sigmaDif
    #ll
    sigmaDif = np.sqrt(np.asarray(bisp_ll_hi['sigmaCP'])**2+np.asarray(bisp_ll_lo['sigmaCP'])**2)
    bisp_ll_hi['TotErr'] = np.abs(((np.asarray(bisp_ll_hi['cphase'])-np.asarray(bisp_ll_lo['cphase']))))
    bisp_ll_hi['TotErr'] = np.minimum(np.asarray(bisp_ll_hi['TotErr']),np.abs(np.asarray(bisp_ll_hi['TotErr']) -360))
    bisp_ll_hi['RelErr'] = np.asarray(bisp_ll_hi['TotErr'])/sigmaDif
    bisp_ll_hi['sigmaDif'] = sigmaDif

    TriStat = pd.DataFrame({})
    AllTri = sorted(list(set(bisp_rr_hi.triangle)))
    TriStat['triangle'] = AllTri
    scans_RR_hi_lo = [np.shape(bisp_rr_hi[bisp_rr_hi.triangle == Tri])[0] for Tri in AllTri]
    scans_LL_hi_lo = [np.shape(bisp_ll_hi[bisp_ll_hi.triangle == Tri])[0] for Tri in AllTri]
    scans_RR_hi_lo_3sig = [np.shape(bisp_rr_hi[(bisp_rr_hi.triangle == Tri)&(bisp_rr_hi.RelErr < 3.)])[0] for Tri in AllTri]
    scans_LL_hi_lo_3sig = [np.shape(bisp_ll_hi[(bisp_ll_hi.triangle == Tri)&(bisp_ll_hi.RelErr < 3.)])[0] for Tri in AllTri]
    TriStat['sc_RR'] = scans_RR_hi_lo
    TriStat['sc_LL'] = scans_LL_hi_lo
    TriStat['sc_total'] = np.asarray(scans_RR_hi_lo)+np.asarray(scans_LL_hi_lo)
    TriStat['sc_3sig_RR'] = scans_RR_hi_lo_3sig
    TriStat['sc_3sig_LL'] = scans_LL_hi_lo_3sig
    TriStat['sc_3sig'] = np.asarray(scans_RR_hi_lo_3sig)+np.asarray(scans_LL_hi_lo_3sig)
    TriStat['sc_3sig_proc'] = np.asarray(map(float,TriStat['sc_3sig']))/np.asarray(map(float,TriStat['sc_total']))
    #TriStat['MedianAbs_lo'] = [np.median(np.asarray(bisp_rr_lo.loc[bisp_rr_lo['triangle'] == Tri,['TotErr']])) for Tri in AllTri]
    #TriStat['MedianAbs_hi'] = [np.median(np.asarray(bisp_rr_hi.loc[bisp_rr_hi['triangle'] == Tri,['TotErr'] ])) for Tri in AllTri]
    TriStat['MedianAbs'] = [np.median(np.asarray(list(bisp_rr_hi.loc[bisp_rr_hi['triangle'] == Tri,'TotErr'])+list(bisp_ll_hi.loc[bisp_ll_hi['triangle'] == Tri,'TotErr' ]))) for Tri in AllTri]
    TriStat['MedianSigma'] = [np.median(np.asarray(list(bisp_rr_hi.loc[bisp_rr_hi['triangle'] == Tri,'sigmaDif'])+list(bisp_ll_hi.loc[bisp_ll_hi['triangle'] == Tri,'sigmaDif' ]))) for Tri in AllTri]
    TriStat = TriStat.sort_values('sc_3sig_proc')

    return TriStat

def pipe_CP_error_rr_ll(bisp_rr_hi, bisp_ll_hi, bisp_rr_lo, bisp_ll_lo):
    #matches 1st with 3rd and 2nd with 4th
    #consistency in bands, separately for rr and ll
    bisp_rr_hi, bisp_rr_lo = match_2_dataframes(bisp_rr_hi, bisp_rr_lo, 'triangle')
    bisp_ll_hi, bisp_ll_lo = match_2_dataframes(bisp_ll_hi, bisp_ll_lo, 'triangle')
    
    #rr
    sigmaDif = np.sqrt(np.asarray(bisp_rr_hi['sigmaCP'])**2+np.asarray(bisp_rr_lo['sigmaCP'])**2)
    bisp_rr_hi['TotErr'] = np.abs(((np.asarray(bisp_rr_hi['cphase'])-np.asarray(bisp_rr_lo['cphase']))))
    bisp_rr_hi['TotErr'] = np.minimum(np.asarray(bisp_rr_hi['TotErr']),np.abs(np.asarray(bisp_rr_hi['TotErr']) -360))
    bisp_rr_hi['RelErr'] = np.asarray(bisp_rr_hi['TotErr'])/sigmaDif
    bisp_rr_hi['sigmaDif'] = sigmaDif
    #ll
    sigmaDif = np.sqrt(np.asarray(bisp_ll_hi['sigmaCP'])**2+np.asarray(bisp_ll_lo['sigmaCP'])**2)
    bisp_ll_hi['TotErr'] = np.abs(((np.asarray(bisp_ll_hi['cphase'])-np.asarray(bisp_ll_lo['cphase']))))
    bisp_ll_hi['TotErr'] = np.minimum(np.asarray(bisp_ll_hi['TotErr']),np.abs(np.asarray(bisp_ll_hi['TotErr']) -360))
    bisp_ll_hi['RelErr'] = np.asarray(bisp_ll_hi['TotErr'])/sigmaDif
    bisp_ll_hi['sigmaDif'] = sigmaDif

    TriStat = pd.DataFrame({})
    AllTri = sorted(list(set(bisp_rr_hi.triangle)))
    TriStat['triangle'] = AllTri
    scans_RR_hi_lo = [np.shape(bisp_rr_hi[bisp_rr_hi.triangle == Tri])[0] for Tri in AllTri]
    scans_LL_hi_lo = [np.shape(bisp_ll_hi[bisp_ll_hi.triangle == Tri])[0] for Tri in AllTri]
    scans_RR_hi_lo_3sig = [np.shape(bisp_rr_hi[(bisp_rr_hi.triangle == Tri)&(bisp_rr_hi.RelErr < 3.)])[0] for Tri in AllTri]
    scans_LL_hi_lo_3sig = [np.shape(bisp_ll_hi[(bisp_ll_hi.triangle == Tri)&(bisp_ll_hi.RelErr < 3.)])[0] for Tri in AllTri]
    TriStat['sc_RR'] = scans_RR_hi_lo
    TriStat['sc_LL'] = scans_LL_hi_lo
    TriStat['sc_total'] = np.asarray(scans_RR_hi_lo)+np.asarray(scans_LL_hi_lo)
    TriStat['sc_3sig_RR'] = scans_RR_hi_lo_3sig
    TriStat['sc_3sig_LL'] = scans_LL_hi_lo_3sig
    TriStat['sc_3sig'] = np.asarray(scans_RR_hi_lo_3sig)+np.asarray(scans_LL_hi_lo_3sig)
    TriStat['sc_3sig_proc'] = np.asarray(map(float,TriStat['sc_3sig']))/np.asarray(map(float,TriStat['sc_total']))
    #TriStat['MedianAbs_lo'] = [np.median(np.asarray(bisp_rr_lo.loc[bisp_rr_lo['triangle'] == Tri,['TotErr']])) for Tri in AllTri]
    #TriStat['MedianAbs_hi'] = [np.median(np.asarray(bisp_rr_hi.loc[bisp_rr_hi['triangle'] == Tri,['TotErr'] ])) for Tri in AllTri]
    TriStat['MedianAbs'] = [np.median(np.asarray(list(bisp_rr_hi.loc[bisp_rr_hi['triangle'] == Tri,'TotErr'])+list(bisp_ll_hi.loc[bisp_ll_hi['triangle'] == Tri,'TotErr' ]))) for Tri in AllTri]
    TriStat['MedianSigma'] = [np.median(np.asarray(list(bisp_rr_hi.loc[bisp_rr_hi['triangle'] == Tri,'sigmaDif'])+list(bisp_ll_hi.loc[bisp_ll_hi['triangle'] == Tri,'sigmaDif' ]))) for Tri in AllTri]
    TriStat = TriStat.sort_values('sc_3sig_proc')

    return TriStat




def band_CP_error_rr_ll_source(bisp_rr_hi, bisp_ll_hi, bisp_rr_lo, bisp_ll_lo):
    #consistency in bands, separately for rr and ll
    bisp_rr_hi, bisp_rr_lo = match_2_dataframes(bisp_rr_hi, bisp_rr_lo, 'triangle')
    bisp_ll_hi, bisp_ll_lo = match_2_dataframes(bisp_ll_hi, bisp_ll_lo, 'triangle')
    
    #rr
    sigmaDif = np.sqrt(np.asarray(bisp_rr_hi['sigmaCP'])**2+np.asarray(bisp_rr_lo['sigmaCP'])**2)
    bisp_rr_hi['TotErr'] = np.abs(((np.asarray(bisp_rr_hi['cphase'])-np.asarray(bisp_rr_lo['cphase']))))
    bisp_rr_hi['TotErr'] = np.minimum(np.asarray(bisp_rr_hi['TotErr']),np.abs(np.asarray(bisp_rr_hi['TotErr']) -360))
    bisp_rr_hi['RelErr'] = np.asarray(bisp_rr_hi['TotErr'])/sigmaDif
    bisp_rr_hi['sigmaDif'] = sigmaDif
    #ll
    sigmaDif = np.sqrt(np.asarray(bisp_ll_hi['sigmaCP'])**2+np.asarray(bisp_ll_lo['sigmaCP'])**2)
    bisp_ll_hi['TotErr'] = np.abs(((np.asarray(bisp_ll_hi['cphase'])-np.asarray(bisp_ll_lo['cphase']))))
    bisp_ll_hi['TotErr'] = np.minimum(np.asarray(bisp_ll_hi['TotErr']),np.abs(np.asarray(bisp_ll_hi['TotErr']) -360))
    bisp_ll_hi['RelErr'] = np.asarray(bisp_ll_hi['TotErr'])/sigmaDif
    bisp_ll_hi['sigmaDif'] = sigmaDif

    TriStat = pd.DataFrame({})
    AllSo = sorted(list(set(bisp_rr_hi.source)))
    TriStat['source'] = AllSo
    scans_RR_hi_lo = [np.shape(bisp_rr_hi[bisp_rr_hi.source == So])[0] for So in AllSo]
    scans_LL_hi_lo = [np.shape(bisp_ll_hi[bisp_ll_hi.source == So])[0] for So in AllSo]
    scans_RR_hi_lo_3sig = [np.shape(bisp_rr_hi[(bisp_rr_hi.source == So)&(bisp_rr_hi.RelErr < 3.)])[0] for So in AllSo]
    scans_LL_hi_lo_3sig = [np.shape(bisp_ll_hi[(bisp_ll_hi.source == So)&(bisp_ll_hi.RelErr < 3.)])[0] for So in AllSo]
    TriStat['sc_RR'] = scans_RR_hi_lo
    TriStat['sc_LL'] = scans_LL_hi_lo
    TriStat['sc_total'] = np.asarray(scans_RR_hi_lo)+np.asarray(scans_LL_hi_lo)
    TriStat['sc_3sig_RR'] = scans_RR_hi_lo_3sig
    TriStat['sc_3sig_LL'] = scans_LL_hi_lo_3sig
    TriStat['sc_3sig'] = np.asarray(scans_RR_hi_lo_3sig)+np.asarray(scans_LL_hi_lo_3sig)
    TriStat['sc_3sig_proc'] = np.asarray(map(float,TriStat['sc_3sig']))/np.asarray(map(float,TriStat['sc_total']))
    #TriStat['MedianAbs_lo'] = [np.median(np.asarray(bisp_rr_lo.loc[bisp_rr_lo['triangle'] == Tri,['TotErr']])) for Tri in AllTri]
    #TriStat['MedianAbs_hi'] = [np.median(np.asarray(bisp_rr_hi.loc[bisp_rr_hi['triangle'] == Tri,['TotErr'] ])) for Tri in AllTri]
    TriStat['MedianAbs'] = [np.median(np.asarray(list(bisp_rr_hi.loc[bisp_rr_hi['source'] == So,'TotErr'])+list(bisp_ll_hi.loc[bisp_ll_hi['source'] == So,'TotErr' ]))) for So in AllSo]
    TriStat['MedianSigma'] = [np.median(np.asarray(list(bisp_rr_hi.loc[bisp_rr_hi['source'] == So,'sigmaDif'])+list(bisp_ll_hi.loc[bisp_ll_hi['source'] == So,'sigmaDif' ]))) for So in AllSo]
    TriStat = TriStat.sort_values('sc_3sig_proc')

    return TriStat

    

def produce_errors_agreement_band(bisp_rr_hi, bisp_ll_hi, bisp_rr_lo, bisp_ll_lo):
    #consistency in bands, separately for rr and ll
    bisp_rr_hi, bisp_rr_lo = match_2_dataframes(bisp_rr_hi, bisp_rr_lo, 'triangle')
    bisp_ll_hi, bisp_ll_lo = match_2_dataframes(bisp_ll_hi, bisp_ll_lo, 'triangle')
   
    #rr
    sigmaDif = np.sqrt(np.asarray(bisp_rr_hi['sigmaCP'])**2+np.asarray(bisp_rr_lo['sigmaCP'])**2)
    bisp_rr_hi['TotErr'] = np.abs(((np.asarray(bisp_rr_hi['cphase'])-np.asarray(bisp_rr_lo['cphase']))))
    bisp_rr_hi['TotErr'] = np.minimum(np.asarray(bisp_rr_hi['TotErr']),np.abs(np.asarray(bisp_rr_hi['TotErr']) -360))
    bisp_rr_hi['RelErr'] = np.asarray(bisp_rr_hi['TotErr'])/sigmaDif
    
    #ll
    sigmaDif = np.sqrt(np.asarray(bisp_ll_hi['sigmaCP'])**2+np.asarray(bisp_ll_lo['sigmaCP'])**2)
    bisp_ll_hi['TotErr'] = np.abs(((np.asarray(bisp_ll_hi['cphase'])-np.asarray(bisp_ll_lo['cphase']))))
    bisp_ll_hi['TotErr'] = np.minimum(np.asarray(bisp_ll_hi['TotErr']),np.abs(np.asarray(bisp_ll_hi['TotErr']) -360))
    bisp_ll_hi['RelErr'] = np.asarray(bisp_ll_hi['TotErr'])/sigmaDif

    return np.asarray(list(bisp_rr_hi['RelErr'])+list(bisp_ll_hi['RelErr']))

def produce_errors_agreement(bisp1,bisp2):
    
    sigmaDif = np.sqrt(np.asarray(bisp1['sigmaCP'])**2+np.asarray(bisp2['sigmaCP'])**2)
    bisp1['TotErr'] = np.abs(((np.asarray(bisp1['cphase'])-np.asarray(bisp2['cphase']))))
    bisp1['TotErr'] = np.minimum(np.asarray(bisp1['TotErr']),np.abs(np.asarray(bisp1['TotErr']) -360))
    bisp1['RelErr'] = np.asarray(bisp1['TotErr'])/sigmaDif

    return np.asarray(list(bisp1['RelErr'])), 



def band_CP_error_rr_ll_station(bisp_rr_hi, bisp_ll_hi, bisp_rr_lo, bisp_ll_lo):
    
    AllSt = list(set(''.join(list(set(bisp_rr_hi.triangle)|set(bisp_ll_hi.triangle)))))
    #consistency in bands, separately for rr and ll
    bisp_rr_hi, bisp_rr_lo = match_2_dataframes(bisp_rr_hi, bisp_rr_lo, 'triangle')
    bisp_ll_hi, bisp_ll_lo = match_2_dataframes(bisp_ll_hi, bisp_ll_lo, 'triangle')
    
    #rr
    sigmaDif = np.sqrt(np.asarray(bisp_rr_hi['sigmaCP'])**2+np.asarray(bisp_rr_lo['sigmaCP'])**2)
    bisp_rr_hi['TotErr'] = np.abs(((np.asarray(bisp_rr_hi['cphase'])-np.asarray(bisp_rr_lo['cphase']))))
    bisp_rr_hi['TotErr'] = np.minimum(np.asarray(bisp_rr_hi['TotErr']),np.abs(np.asarray(bisp_rr_hi['TotErr']) -360))
    bisp_rr_hi['RelErr'] = np.asarray(bisp_rr_hi['TotErr'])/sigmaDif
    bisp_rr_hi['sigmaDif'] = sigmaDif
    #ll
    sigmaDif = np.sqrt(np.asarray(bisp_ll_hi['sigmaCP'])**2+np.asarray(bisp_ll_lo['sigmaCP'])**2)
    bisp_ll_hi['TotErr'] = np.abs(((np.asarray(bisp_ll_hi['cphase'])-np.asarray(bisp_ll_lo['cphase']))))
    bisp_ll_hi['TotErr'] = np.minimum(np.asarray(bisp_ll_hi['TotErr']),np.abs(np.asarray(bisp_ll_hi['TotErr']) -360))
    bisp_ll_hi['RelErr'] = np.asarray(bisp_ll_hi['TotErr'])/sigmaDif
    bisp_ll_hi['sigmaDif'] = sigmaDif

    TriStat = pd.DataFrame({})
    #AllTri = sorted(list(set(bisp_rr_hi.triangle)))
    TriStat['station'] = AllSt
    scans_RR_hi_lo = [np.shape(bisp_rr_hi[map(lambda x: St in x, bisp_rr_hi.triangle)])[0] for St in AllSt]
    scans_LL_hi_lo = [np.shape(bisp_ll_hi[map(lambda x: St in x, bisp_ll_hi.triangle)])[0] for St in AllSt]
    scans_RR_hi_lo_3sig = [np.shape(bisp_rr_hi[map(lambda x: St in x, bisp_rr_hi.triangle)&(bisp_rr_hi.RelErr < 3.)])[0] for St in AllSt]
    scans_LL_hi_lo_3sig = [np.shape(bisp_ll_hi[map(lambda x: St in x, bisp_ll_hi.triangle)&(bisp_ll_hi.RelErr < 3.)])[0] for St in AllSt]
    TriStat['sc_RR'] = scans_RR_hi_lo
    TriStat['sc_LL'] = scans_LL_hi_lo
    TriStat['sc_total'] = np.asarray(scans_RR_hi_lo)+np.asarray(scans_LL_hi_lo)
    TriStat['sc_3sig_RR'] = scans_RR_hi_lo_3sig
    TriStat['sc_3sig_LL'] = scans_LL_hi_lo_3sig
    TriStat['sc_3sig'] = np.asarray(scans_RR_hi_lo_3sig)+np.asarray(scans_LL_hi_lo_3sig)
    TriStat['sc_3sig_proc'] = np.asarray(map(float,TriStat['sc_3sig']))/np.asarray(map(float,TriStat['sc_total']))
    #TriStat['MedianAbs_lo'] = [np.median(np.asarray(bisp_rr_lo.loc[bisp_rr_lo['triangle'] == Tri,['TotErr']])) for Tri in AllTri]
    #TriStat['MedianAbs_hi'] = [np.median(np.asarray(bisp_rr_hi.loc[bisp_rr_hi['triangle'] == Tri,['TotErr'] ])) for Tri in AllTri]
    TriStat['MedianAbs'] = [np.median(np.asarray(list(bisp_rr_hi.loc[map(lambda x: St in x, bisp_rr_hi.triangle),'TotErr'])+list(bisp_ll_hi.loc[map(lambda x: St in x, bisp_ll_hi.triangle),'TotErr' ]))) for St in AllSt]
    TriStat['MedianSigma'] = [np.median(np.asarray(list(bisp_rr_hi.loc[map(lambda x: St in x, bisp_rr_hi.triangle),'sigmaDif'])+list(bisp_ll_hi.loc[map(lambda x: St in x, bisp_ll_hi.triangle),'sigmaDif' ]))) for St in AllSt]
    TriStat = TriStat.sort_values('sc_3sig_proc')

    return TriStat


def polar_CP_error(bisp_rr, bisp_ll):

    bisp_rr, bisp_ll = match_2_dataframes(bisp_rr, bisp_ll, 'triangle')
    
    sigmaDif = np.sqrt(np.asarray(bisp_rr['sigmaCP'])**2+np.asarray(bisp_ll['sigmaCP'])**2)
    bisp_rr['TotErr'] = np.abs(((np.asarray(bisp_rr['cphase'])-np.asarray(bisp_ll['cphase']))))
    bisp_rr['TotErr'] = np.minimum(np.asarray(bisp_rr['TotErr']),np.abs(np.asarray(bisp_rr['TotErr']) -360))
    bisp_rr['RelErr'] = np.asarray(bisp_rr['TotErr'])/sigmaDif
    
    TriStat = pd.DataFrame({})
    AllTri = sorted(list(set(bisp_rr.triangle)))
    TriStat['triangle'] = AllTri
    scans_RR_LL = [np.shape(bisp_rr[bisp_rr.triangle == Tri])[0] for Tri in AllTri]
    scans_RR_LL_3sig = [np.shape(bisp_rr[(bisp_rr.triangle == Tri)&(bisp_rr.RelErr < 3.)])[0] for Tri in AllTri]
    TriStat['sc_total'] = np.asarray(scans_RR_LL)
    TriStat['sc_3sig'] = scans_RR_LL_3sig
    TriStat['sc_3sig_proc'] = np.asarray(map(float,TriStat['sc_3sig']))/np.asarray(map(float,TriStat['sc_total']))
    #TriStat['MedianAbs_lo'] = [np.median(np.asarray(bisp_rr_lo.loc[bisp_rr_lo['triangle'] == Tri,['TotErr']])) for Tri in AllTri]
    #TriStat['MedianAbs_hi'] = [np.median(np.asarray(bisp_rr_hi.loc[bisp_rr_hi['triangle'] == Tri,['TotErr'] ])) for Tri in AllTri]
    TriStat['MedianAbs'] = [np.median(np.asarray(list(bisp_rr.loc[bisp_rr['triangle'] == Tri,'TotErr']))) for Tri in AllTri]
    TriStat['MedianSigma'] = [np.median(np.asarray(list(bisp_rr.loc[bisp_rr['triangle'] == Tri,'sigmaCP']))) for Tri in AllTri]

    TriStat = TriStat.sort_values('sc_3sig_proc')

    return TriStat

def CP_error(bisp_rr, bisp_ll):

    #bisp_rr, bisp_ll = match_2_dataframes(bisp_rr, bisp_ll, 'triangle')
    
    sigmaDif = np.sqrt(np.asarray(bisp_rr['sigmaCP'])**2+np.asarray(bisp_ll['sigmaCP'])**2)
    bisp_rr['TotErr'] = np.abs(((np.asarray(bisp_rr['cphase'])-np.asarray(bisp_ll['cphase']))))
    bisp_rr['TotErr'] = np.minimum(np.asarray(bisp_rr['TotErr']),np.abs(np.asarray(bisp_rr['TotErr']) -360))
    bisp_rr['RelErr'] = np.asarray(bisp_rr['TotErr'])/sigmaDif
    
    TriStat = pd.DataFrame({})
    AllTri = sorted(list(set(bisp_rr.triangle)))
    TriStat['triangle'] = AllTri
    scans_RR_LL = [np.shape(bisp_rr[bisp_rr.triangle == Tri])[0] for Tri in AllTri]
    scans_RR_LL_3sig = [np.shape(bisp_rr[(bisp_rr.triangle == Tri)&(bisp_rr.RelErr < 3.)])[0] for Tri in AllTri]
    TriStat['sc_total'] = np.asarray(scans_RR_LL)
    TriStat['sc_3sig'] = scans_RR_LL_3sig
    TriStat['sc_3sig_proc'] = np.asarray(map(float,TriStat['sc_3sig']))/np.asarray(map(float,TriStat['sc_total']))
    #TriStat['MedianAbs_lo'] = [np.median(np.asarray(bisp_rr_lo.loc[bisp_rr_lo['triangle'] == Tri,['TotErr']])) for Tri in AllTri]
    #TriStat['MedianAbs_hi'] = [np.median(np.asarray(bisp_rr_hi.loc[bisp_rr_hi['triangle'] == Tri,['TotErr'] ])) for Tri in AllTri]
    TriStat['MedianAbs'] = [np.median(np.asarray(list(bisp_rr.loc[bisp_rr['triangle'] == Tri,'TotErr']))) for Tri in AllTri]
    TriStat['MedianSigma'] = [np.median(np.asarray(list(bisp_rr.loc[bisp_rr['triangle'] == Tri,'sigmaCP']))) for Tri in AllTri]

    TriStat = TriStat.sort_values('sc_3sig_proc')

    return TriStat

def polar_CP_error_source(bisp_rr, bisp_ll):

    AllSo = sorted(list( set(bisp_rr.source) ))
    bisp_rr, bisp_ll = match_2_dataframes(bisp_rr, bisp_ll, 'triangle')
    
    sigmaDif = np.sqrt(np.asarray(bisp_rr['sigmaCP'])**2+np.asarray(bisp_ll['sigmaCP'])**2)
    bisp_rr['TotErr'] = np.abs(((np.asarray(bisp_rr['cphase'])-np.asarray(bisp_ll['cphase']))))
    bisp_rr['TotErr'] = np.minimum(np.asarray(bisp_rr['TotErr']),np.abs(np.asarray(bisp_rr['TotErr']) -360))
    bisp_rr['RelErr'] = np.asarray(bisp_rr['TotErr'])/sigmaDif
    
    TriStat = pd.DataFrame({})
    #AllTri = sorted(list(set(bisp_rr.triangle)))
    TriStat['source'] = AllSo
    scans_RR_LL = [np.shape(bisp_rr[bisp_rr.source== So])[0] for So in AllSo]
    scans_RR_LL_3sig = [np.shape(bisp_rr[(bisp_rr.source == So)&(bisp_rr.RelErr < 3.)])[0] for So in AllSo]
    TriStat['sc_total'] = np.asarray(scans_RR_LL)
    TriStat['sc_3sig'] = scans_RR_LL_3sig
    TriStat['sc_3sig_proc'] = np.asarray(map(float,TriStat['sc_3sig']))/np.asarray(map(float,TriStat['sc_total']))
    TriStat['MedianAbs'] = [np.median(np.asarray(list(bisp_rr.loc[bisp_rr['source'] == So,'TotErr']))) for So in AllSo]
    TriStat = TriStat.sort_values('sc_3sig_proc')

    return TriStat


def polar_CP_error_station(bisp_rr, bisp_ll):

    AllSt = list(set(''.join(list(set(bisp_rr.triangle)))))
    bisp_rr, bisp_ll = match_2_dataframes(bisp_rr, bisp_ll, 'triangle')
    
    sigmaDif = np.sqrt(np.asarray(bisp_rr['sigmaCP'])**2+np.asarray(bisp_ll['sigmaCP'])**2)
    bisp_rr['TotErr'] = np.abs(((np.asarray(bisp_rr['cphase'])-np.asarray(bisp_ll['cphase']))))
    bisp_rr['TotErr'] = np.minimum(np.asarray(bisp_rr['TotErr']),np.abs(np.asarray(bisp_rr['TotErr']) -360))
    bisp_rr['RelErr'] = np.asarray(bisp_rr['TotErr'])/sigmaDif
    
    TriStat = pd.DataFrame({})
    #AllTri = sorted(list(set(bisp_rr.triangle)))
    TriStat['station'] = AllSt
    scans_RR_LL = [np.shape(bisp_rr[map(lambda x: St in x, bisp_rr.triangle)])[0] for St in AllSt]
    scans_RR_LL_3sig = [np.shape(bisp_rr[(map(lambda x: St in x, bisp_rr.triangle))&(bisp_rr.RelErr < 3.)])[0] for St in AllSt]
    TriStat['sc_total'] = np.asarray(scans_RR_LL)
    TriStat['sc_3sig'] = scans_RR_LL_3sig
    TriStat['sc_3sig_proc'] = np.asarray(map(float,TriStat['sc_3sig']))/np.asarray(map(float,TriStat['sc_total']))
    TriStat['MedianAbs'] = [np.median(np.asarray(list(bisp_rr.loc[map(lambda x: St in x, bisp_rr['triangle']),'TotErr']))) for St in AllSt]
    #TriStat['MedianSigma'] = [np.median(np.asarray(list(bisp.loc[bisp['triangle'] == Tri,'sigmaCP']))) for Tri in AllTri]

    TriStat = TriStat.sort_values('sc_3sig_proc')

    return TriStat


def add_band_error(bisp,band):

    bisp['TotErr'] = np.mod(np.asarray(bisp['cphase']),360.)
    bisp['TotErr'] = np.minimum( np.asarray(bisp['TotErr']), np.abs(np.asarray(bisp['TotErr']) -360.))
    bisp['RelErr'] = np.asarray(bisp['TotErr'])/np.asarray(bisp['sigmaCP'])
    bisp['band'] = [band]*np.shape(bisp)[0]
    return bisp


def CP_error_agg(bisp, bispRel, agg_type='triangle',match=False,inflate_errors_factor = 1.):

    bisp['sigmaCP'] = inflate_errors_factor*bisp['sigmaCP']
    bispRel['sigmaCP'] = inflate_errors_factor*bispRel['sigmaCP']
    if match==True:
        bisp, bispRel = match_2_bsp_frames(bisp,bispRel,dt = 60.)
    sigmaTotal = np.sqrt(np.asarray(bisp['sigmaCP'])**2+np.asarray(bispRel['sigmaCP'])**2)
    bisp['sigmaTotal'] = sigmaTotal
    distTotal = np.asarray(np.abs(((np.asarray(bisp['cphase'])-np.asarray(bispRel['cphase'])))))
    bisp['distTotal'] = np.minimum(distTotal,np.abs(distTotal - 360.))
    bisp['distRel'] = distTotal/sigmaTotal

    if agg_type=='triangle':
        all_agg = sorted(list( set(bisp.triangle) ))
        cond_agg = lambda x: (bisp.triangle == x)
    elif agg_type=='source':
        all_agg = sorted(list( set(bisp.source) ))
        cond_agg = lambda x: (bisp.source == x)
    elif agg_type=='expt_no':
        all_agg = sorted(list( set(bisp.expt_no) ))
        cond_agg = lambda x: (bisp.expt_no == x)
    elif agg_type=='station':
        AllTri = sorted(list( set(bisp.triangle) ))
        all_agg = list(set(''.join(list(set(bisp.triangle)))))
        cond_agg = lambda x: map(lambda y: x in y, bisp.triangle)
    elif agg_type=='baseline':
        AllSt = set(''.join(list(set(bisp.triangle))))
        AllBa = list(itertools.combinations(AllSt,2))
        AllBa = [x[0]+x[1] for x in AllBa]
        all_agg = [base for base in AllBa if np.shape(bisp[map(lambda x: (base[0] in x)&(base[1] in x),bisp.triangle)])[0]>0]
        cond_agg = lambda x: map(lambda y: (x[0] in y)&(x[1] in y),bisp.triangle)

    TriStat = pd.DataFrame({})
    TriStat[agg_type] = all_agg
    scans_tot = [np.shape(bisp[cond_agg(x)])[0] for x in all_agg]
    scans_1sig = [np.shape(bisp[( cond_agg(x) )&(bisp.distRel < 1.)])[0] for x in all_agg]
    scans_2sig = [np.shape(bisp[( cond_agg(x) )&(bisp.distRel < 2.)])[0] for x in all_agg]
    scans_3sig = [np.shape(bisp[( cond_agg(x) )&(bisp.distRel < 3.)])[0] for x in all_agg]
    TriStat['sc_total'] = scans_tot
    TriStat['sc_3sig'] = scans_3sig
    TriStat['sc_1sig_proc'] = np.asarray(map(float,scans_1sig))/np.asarray(map(float,TriStat['sc_total']))
    TriStat['sc_2sig_proc'] = np.asarray(map(float,scans_2sig))/np.asarray(map(float,TriStat['sc_total']))
    TriStat['sc_3sig_proc'] = np.asarray(map(float,TriStat['sc_3sig']))/np.asarray(map(float,TriStat['sc_total']))
    TriStat['MedianAbs'] = [np.nanmedian(np.asarray(list(bisp.loc[( cond_agg(x) ),'distTotal']))) for x in all_agg]
    TriStat['MedianSigma'] = [np.nanmedian(np.asarray(list(bisp.loc[( cond_agg(x) ),'sigmaTotal']))) for x in all_agg]
    TriStat = TriStat.sort_values('sc_3sig_proc')
    return TriStat, bisp

def triv_CP_error_agg(bisp,agg_type='triangle',inflate_errors_factor = 1.):

    bisp['sigmaCP'] = inflate_errors_factor*bisp['sigmaCP']
    #get errors from zero
    bisp['TotErr'] = np.mod(np.asarray(bisp['cphase']),360.)
    bisp['TotErr'] = np.minimum( np.asarray(bisp['TotErr']), np.abs(np.asarray(bisp['TotErr']) -360.))
    bisp['RelErr'] = np.asarray(bisp['TotErr'])/np.asarray(bisp['sigmaCP'])

    if agg_type=='triangle':
        all_agg = sorted(list( set(bisp.triangle) ))
        cond_agg = lambda x: (bisp.triangle == x)
    elif agg_type=='source':
        all_agg = sorted(list( set(bisp.source) ))
        cond_agg = lambda x: (bisp.source == x)
    elif agg_type=='expt_no':
        all_agg = sorted(list( set(bisp.expt_no) ))
        cond_agg = lambda x: (bisp.expt_no == x)
    elif agg_type=='station':
        AllTri = sorted(list( set(bisp.triangle) ))
        all_agg = list(set(''.join(list(set(bisp.triangle)))))
        cond_agg = lambda x: map(lambda y: x in y, bisp.triangle)
    elif agg_type=='baseline':
        AllSt = set(''.join(list(set(bisp.triangle))))
        AllBa = list(itertools.combinations(AllSt,2))
        AllBa = [x[0]+x[1] for x in AllBa]
        all_agg = [base for base in AllBa if np.shape(bisp[map(lambda x: (base[0] in x)&(base[1] in x),bisp.triangle)])[0]>0]
        cond_agg = lambda x: map(lambda y: (x[0] in y)&(x[1] in y),bisp.triangle)

    TriStat = pd.DataFrame({})
    TriStat[agg_type] = all_agg
    scans_tot = [np.shape(bisp[cond_agg(x)])[0] for x in all_agg]
    scans_1sig = [np.shape(bisp[( cond_agg(x) )&(bisp.RelErr < 1.)])[0] for x in all_agg]
    scans_2sig = [np.shape(bisp[( cond_agg(x) )&(bisp.RelErr < 2.)])[0] for x in all_agg]
    scans_3sig = [np.shape(bisp[( cond_agg(x) )&(bisp.RelErr < 3.)])[0] for x in all_agg]
    TriStat['sc_total'] = scans_tot
    TriStat['sc_3sig'] = scans_3sig
    TriStat['sc_1sig_proc'] = np.asarray(map(float,scans_1sig))/np.asarray(map(float,TriStat['sc_total']))
    TriStat['sc_2sig_proc'] = np.asarray(map(float,scans_2sig))/np.asarray(map(float,TriStat['sc_total']))
    TriStat['sc_3sig_proc'] = np.asarray(map(float,TriStat['sc_3sig']))/np.asarray(map(float,TriStat['sc_total']))
    TriStat['MedianAbs'] = [np.nanmedian(np.asarray(list(bisp.loc[( cond_agg(x) ),'TotErr']))) for x in all_agg]
    TriStat['MedianSigma'] = [np.nanmedian(np.asarray(list(bisp.loc[( cond_agg(x) ),'sigmaCP']))) for x in all_agg]
    TriStat = TriStat.sort_values('sc_3sig_proc')
    return TriStat

def amp_error_agg(frame, frameRel, agg_type='baseline',match=False, debias=False,inflate_errors_factor = 1. ):

    
    if debias==True:
        which_amp = 'amp'
        which_sigma = 'sigma'
    else:
        which_amp = 'ampB'
        which_sigma = 'sigmaB'

    frame[which_sigma] = inflate_errors_factor*frame[which_sigma] 
    frameRel[which_sigma] = inflate_errors_factor*frameRel[which_sigma] 

    if match==True:
        frame, frameRel = match_2_bsp_frames(frame,frameRel,dt = 60.,what_is_same='baseline')
    sigmaTotal = np.sqrt(np.asarray(frame[which_sigma])**2+np.asarray(frameRel[which_sigma])**2)
    frame['sigmaTotal'] = sigmaTotal
    #distTotal = np.asarray(np.abs(((np.asarray(frame[which_amp])-np.asarray(frameRel[which_amp])))))
    distTotal = np.asarray((((np.asarray(frame[which_amp])-np.asarray(frameRel[which_amp])))))
    frame['distTotal'] = distTotal
    frame['distRel'] = (distTotal)/sigmaTotal
    frame['distRelAmp'] = (distTotal)/frame[which_amp]
    frame['sigmaRelAmp'] = sigmaTotal/frame[which_amp]
    frame['AHObyAAI'] = np.asarray(frame[which_amp])/np.asarray(frameRel[which_amp])

    if agg_type=='baseline':
        all_agg = sorted(list( set(frame.baseline) ))
        cond_agg = lambda x: (frame.baseline == x)
    elif agg_type=='source':
        all_agg = sorted(list( set(frame.source) ))
        cond_agg = lambda x: (frame.source == x)
    elif agg_type=='expt_no':
        all_agg = sorted(list( set(frame.expt_no) ))
        cond_agg = lambda x: (frame.expt_no == x)
    elif agg_type=='station':
        AllBa = sorted(list( set(frame.baseline) ))
        all_agg = list(set(''.join(list(set(frame.baseline)))))
        cond_agg = lambda x: map(lambda y: x in y, frame.baseline)

    BaStat = pd.DataFrame({})
    BaStat[agg_type] = all_agg
    scans_tot = [np.shape(frame[cond_agg(x)])[0] for x in all_agg]
    scans_1sig = [np.shape(frame[( cond_agg(x) )&(frame.distRel < 1.)])[0] for x in all_agg]
    scans_2sig = [np.shape(frame[( cond_agg(x) )&(frame.distRel < 2.)])[0] for x in all_agg]
    scans_3sig = [np.shape(frame[( cond_agg(x) )&(frame.distRel < 3.)])[0] for x in all_agg]
    BaStat['sc_total'] = scans_tot
    BaStat['sc_3sig'] = scans_3sig
    BaStat['sc_1sig_proc'] = np.asarray(map(float,scans_1sig))/np.asarray(map(float,BaStat['sc_total']))
    BaStat['sc_2sig_proc'] = np.asarray(map(float,scans_2sig))/np.asarray(map(float,BaStat['sc_total']))
    BaStat['sc_3sig_proc'] = np.asarray(map(float,BaStat['sc_3sig']))/np.asarray(map(float,BaStat['sc_total']))
    BaStat['MedianAbs'] = [np.nanmedian(np.asarray(list(frame.loc[( cond_agg(x) ),'distTotal']))) for x in all_agg]
    BaStat['MedianRelAbs'] = [np.nanmedian(np.asarray(list(frame.loc[( cond_agg(x) ),'distTotal']))/np.asarray(list(frame.loc[( cond_agg(x) ),'amp'])) ) for x in all_agg]
    BaStat['MedianSigma'] = [np.nanmedian(np.asarray(list(frame.loc[( cond_agg(x) ),'sigmaTotal']))) for x in all_agg]
    BaStat['MedianRelSigma'] = [np.nanmedian(np.asarray(list(frame.loc[( cond_agg(x) ),'sigmaTotal']))/np.asarray(list(frame.loc[( cond_agg(x) ),'amp']))  ) for x in all_agg]
    BaStat = BaStat.sort_values('sc_3sig_proc')
    return BaStat, frame





'''

def triv_CP_error(bisp):

    AllTri = sorted(list( set(bisp.triangle) ))
    
    bisp['TotErr'] = np.mod(np.asarray(bisp['cphase']),360.)
    bisp['TotErr'] = np.minimum( np.asarray(bisp['TotErr']), np.abs(np.asarray(bisp['TotErr']) -360.))
    bisp['RelErr'] = np.asarray(bisp['TotErr'])/np.asarray(bisp['sigmaCP'])

    #print(bisp)
    TriStat = pd.DataFrame({})
    
    TriStat['triangle'] = AllTri
    scans_tot = [np.shape(bisp[bisp.triangle == Tri])[0] for Tri in AllTri]
    scans_3sig = [np.shape(bisp[(bisp.triangle == Tri)&(bisp.RelErr < 3.)])[0] for Tri in AllTri]
    TriStat['sc_total'] = scans_tot
    TriStat['sc_3sig'] = scans_3sig
    TriStat['sc_3sig_proc'] = np.asarray(map(float,TriStat['sc_3sig']))/np.asarray(map(float,TriStat['sc_total']))
    TriStat['MedianAbs'] = [np.median(np.asarray(list(bisp.loc[bisp['triangle'] == Tri,'TotErr']))) for Tri in AllTri]
    TriStat['MedianSigma'] = [np.median(np.asarray(list(bisp.loc[bisp['triangle'] == Tri,'sigmaCP']))) for Tri in AllTri]
    
    TriStat = TriStat.sort_values('sc_3sig_proc')

    return TriStat



def triv_CP_error_station(bisp):

    AllTri = sorted(list( set(bisp.triangle) ))
    AllSt = list(set(''.join(list(set(bisp.triangle)))))

    bisp['TotErr'] = np.mod(np.asarray(bisp['cphase']),360.)
    bisp['TotErr'] = np.minimum( np.asarray(bisp['TotErr']), np.abs(np.asarray(bisp['TotErr']) -360.))
    bisp['RelErr'] = np.asarray(bisp['TotErr'])/np.asarray(bisp['sigmaCP'])

    TriStat = pd.DataFrame({})
    
    TriStat['station'] = AllSt

    scans_tot = [np.shape(bisp[map(lambda x: St in x, bisp.triangle)])[0] for St in AllSt]
    scans_3sig = [np.shape(bisp[map(lambda x: St in x, bisp.triangle)&(bisp.RelErr < 3.)])[0] for St in AllSt]
    TriStat['sc_total'] = scans_tot
    TriStat['sc_3sig'] = scans_3sig
    TriStat['sc_3sig_proc'] = np.asarray(map(float,TriStat['sc_3sig']))/np.asarray(map(float,TriStat['sc_total']))
    TriStat['MedianAbs'] = [np.median(np.asarray(list(bisp.loc[map(lambda x: St in x, bisp.triangle),'TotErr']))) for St in AllSt]
    TriStat['MedianSigma'] = [np.median(np.asarray(list(bisp.loc[map(lambda x: St in x, bisp.triangle),'sigmaCP']))) for St in AllSt]
    
    TriStat = TriStat.sort_values('sc_3sig_proc')

    return TriStat



def triv_CP_error_baseline(bisp):

    
    AllSt = set(''.join(list(set(bisp.triangle))))
    AllBa = list(itertools.combinations(AllSt,2))
    AllBa = [x[0]+x[1] for x in AllBa]
    AllBa = [base for base in AllBa if np.shape(bisp[map(lambda x: (base[0] in x)&(base[1] in x),bisp.triangle)])[0]>0]

    bisp['TotErr'] = np.mod(np.asarray(bisp['cphase']),360.)
    bisp['TotErr'] = np.minimum( np.asarray(bisp['TotErr']), np.abs(np.asarray(bisp['TotErr']) -360.))
    bisp['RelErr'] = np.asarray(bisp['TotErr'])/np.asarray(bisp['sigmaCP'])

    TriStat = pd.DataFrame({})
    
    TriStat['baseline'] = AllBa

    scans_tot = [np.shape(bisp[map(lambda x: (base[0] in x)&(base[1] in x),bisp.triangle)])[0] for base in AllBa]
    scans_3sig = [np.shape(bisp[map(lambda x: (base[0] in x)&(base[1] in x),bisp.triangle)&(bisp.RelErr < 3.)])[0] for base in AllBa]
    TriStat['sc_total'] = scans_tot
    TriStat['sc_3sig'] = scans_3sig
    TriStat['sc_3sig_proc'] = np.asarray(map(float,TriStat['sc_3sig']))/np.asarray(map(float,TriStat['sc_total']))
    TriStat['MedianAbs'] = [np.median(np.asarray(list(bisp.loc[map(lambda x: (base[0] in x)&(base[1] in x),bisp.triangle),'TotErr']))) for base in AllBa]
    TriStat['MedianSigma'] = [np.median(np.asarray(list(bisp.loc[map(lambda x: (base[0] in x)&(base[1] in x),bisp.triangle),'sigmaCP']))) for base in AllBa]
    
    TriStat = TriStat.sort_values('sc_3sig_proc')

    return TriStat



def triv_CP_error_source(bisp):

    AllSo = sorted(list( set(bisp.source) ))
    #AllSt = list(set(''.join(list(set(bisp.triangle)))))

    bisp['TotErr'] = np.mod(np.asarray(bisp['cphase']),360.)
    bisp['TotErr'] = np.minimum( np.asarray(bisp['TotErr']), np.abs(np.asarray(bisp['TotErr']) -360.))
    bisp['RelErr'] = np.asarray(bisp['TotErr'])/np.asarray(bisp['sigmaCP'])

    TriStat = pd.DataFrame({})
    
    TriStat['source'] = AllSo

    scans_tot = [np.shape(bisp[bisp.source==So])[0] for So in AllSo]
    scans_3sig = [np.shape(bisp[(bisp.source==So)&(bisp.RelErr < 3.)])[0] for So in AllSo]
    TriStat['sc_total'] = scans_tot
    TriStat['sc_3sig'] = scans_3sig
    TriStat['sc_3sig_proc'] = np.asarray(map(float,TriStat['sc_3sig']))/np.asarray(map(float,TriStat['sc_total']))
    TriStat['MedianAbs'] = [np.median(np.asarray(list(bisp.loc[bisp['source']==So,'TotErr']))) for So in AllSo]
    TriStat['MedianSigma'] = [np.median(np.asarray(list(bisp.loc[bisp['source']==So,'sigmaCP']))) for So in AllSo]
    
    TriStat = TriStat.sort_values('sc_3sig_proc')

    return TriStat




'''