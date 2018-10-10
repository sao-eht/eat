#!/usr/bin/env python2
#
# Maciek Wielgus 02/Oct/2018
from __future__ import print_function
from __future__ import division
import pandas as pd
import numpy as np
from eat.io import uvfits
from eat.inspect import utils as ut
from eat.inspect import closures as cl
import os,sys,importlib,glob

VEX_DEFAULT='/home/maciek/VEX/'

def import_uvfits_set(path_data_0,path_vex,path_out,out_name,bandname,pipeline_name='hops',tavg='scan',
    only_parallel=False,filend=".uvfits",incoh_avg=False,out_type='hdf',rescale_noise=False,polrep='circ', 
    old_format=True,path_ehtim='',closure='',tavg_closures='scan',precoh_avg_time=0.):
    '''
    Imports whole dataset of uvfits with HOPS folder structure, or even without structure
    '''
    print('path_data_0 = ', path_data_0)
    print('path_vex = ', path_vex)
    print('path_out = ', path_out)
    print('out_name = ', out_name)
    print('pipeline_name= ', pipeline_name)
    print('tavg = ', tavg)
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    df = pd.DataFrame({})

    path0a = glob.glob(path_data_0+'/*/*'+filend)
    path0b = glob.glob(path_data_0+'/*'+filend)
    path0 = sorted(path0a+path0b)

    ###########################################################################
    # VISIBILITIES
    ###########################################################################
    for filen in path0:
        print("********************************************************")
        print('processing ', filen)
        print("********************************************************")
        #try:
        df_foo = uvfits.get_df_from_uvfit(filen,path_vex=path_vex,force_singlepol='no',band=bandname,round_s=0.1,
    only_parallel=only_parallel,rescale_noise=rescale_noise,polrep=polrep,path_ehtim=path_ehtim)
        print('Found datapoints: ',np.shape(df_foo)[0])
        #CONVERT TO OLD DF FORMATTING (SEPARATE DATA RECORD FOR EACH POLARIZATION)
        if old_format:
            df_foo = ut.old_format(df_foo)
        if 'std_by_mean' in df_foo.columns:
            df_foo.drop('std_by_mean',axis=1,inplace=True)
            df_foo['std_by_mean'] = df_foo['amp']
        print('Averaging this file...')
        if incoh_avg==False:
            print('Averaging coherently for ', str(tavg))
            df_scan = ut.coh_avg_vis(df_foo.copy(),tavg=tavg,phase_type='phase')
        else:      
            if precoh_avg_time > 0:
                print('Averaging coherently for ', str(precoh_avg_time))
                df_coh = ut.coh_avg_vis(df_foo.copy(),tavg=precoh_avg_time,phase_type='phase')
                print('Averaging incoherently for ', str(tavg))
                df_scan = ut.incoh_avg_vis(df_coh.copy(),tavg=tavg,phase_type='phase')
            else:
                print('Averaging incoherently for ', str(tavg))
                df_scan = ut.incoh_avg_vis(df_foo.copy(),tavg=tavg,phase_type='phase')
        df = pd.concat([df,df_scan.copy()],ignore_index=True)
        #except: print('Nothing from this file...')

    try: df.drop(list(df[df.baseline.str.contains('R')].index.values),inplace=True)
    except: pass
    try: df['source'] = list(map(str,df['source']))
    except: pass
    try: df.dropna(subset=['snr'],inplace=True)
    except: pass
   
    ###########################################################################
    # CLOSURES
    ###########################################################################
    if (closure=='cphase')|(closure=='lcamp'):

        print("********************************************************")
        print("******************SAVING CLOSURES***********************")
        print("********************************************************")

        print("Saving scan-averaged closure phases...")
        bsp = cl.all_bispectra(df,phase_type='phase')
        bsp.drop('fracpols',axis=1,inplace=True)
        bsp.drop('snrs',axis=1,inplace=True)
        bsp.drop('amps',axis=1,inplace=True)
        bsp_sc = ut.coh_avg_bsp(bsp,tavg=tavg_closures)
        out_name_cp = 'cp_sc_'+out_name
        if out_type=='hdf':
            print('Saving file: '+path_out+out_name_cp+'.h5')
            bsp_sc.to_hdf(path_out+out_name_cp+'.h5', key=out_name_cp, mode='w',format='table')
        elif out_type=='pic':
            print('Saving file: '+path_out+out_name_cp+'.pic')
            bsp_sc.to_pickle(path_out+out_name_cp+'.pic')
        elif out_type=='both':
            print('Saving file: '+path_out+out_name_cp+'.h5')
            bsp_sc.to_hdf(path_out+out_name_cp+'.h5', key=out_name_cp, mode='w',format='table')
            print('Saving file: '+path_out+out_name_cp+'.pic')
            bsp_sc.to_pickle(path_out+out_name_cp+'.pic')

        print("Saving scan-averaged log closure amplitudes...")
        quad=cl.all_quadruples_new(df,ctype='logcamp',debias='camp')
        quad.drop('snrs',axis=1,inplace=True)
        quad.drop('amps',axis=1,inplace=True)
        quad_sc=ut.avg_camp(quad,tavg=tavg_closures)
        out_name_lca= 'lca_sc_'+out_name
        quad_sc['scan_id'] = list(map(np.int64,quad_sc.scan_id))
        if out_type=='hdf':
            print('Saving file: '+path_out+out_name_lca+'.h5')
            quad_sc.to_hdf(path_out+out_name_lca+'.h5', key=out_name_lca, mode='w',format='table')
        elif out_type=='pic':
            print('Saving file: '+path_out+out_name_lca+'.pic')
            quad_sc.to_pickle(path_out+out_name_lca+'.pic')
        elif out_type=='both':
            print('Saving file: '+path_out+out_name_lca+'.h5')
            quad_sc.to_hdf(path_out+out_name_lca+'.h5', key=out_name_lca, mode='w',format='table')
            print('Saving file: '+path_out+out_name_lca+'.pic')
            quad_sc.to_pickle(path_out+out_name_lca+'.pic')

    if out_type=='hdf':
        print('Saving file: '+path_out+out_name+'.h5')
        df.to_hdf(path_out+out_name+'.h5', key=out_name, mode='w',format='table')
    elif out_type=='pic':
        print('Saving file: '+path_out+out_name+'.pic')
        df.to_pickle(path_out+out_name+'.pic')
    elif out_type=='both':
        print('Saving file: '+path_out+out_name+'.h5')
        df.to_hdf(path_out+out_name+'.h5', key=out_name, mode='w',format='table')
        print('Saving file: '+path_out+out_name+'.pic')
        df.to_pickle(path_out+out_name+'.pic')
    else: return df


##################################################################################################################################
##########################  Main FUNCTION ########################################################################################
##################################################################################################################################
def main(path_data_0,path_vex,path_out,out_name,bandname,pipeline_name='hops',tavg='scan',
    only_parallel=True,filend=".uvfits",incoh_avg=False,out_type='hdf',rescale_noise=False,polrep=None, old_format=True,path_ehtim='',closure='',tavg_closures='scan',precoh_avg_time=0.):

    print("********************************************************")
    print("*********************IMPORT DATA************************")
    print("********************************************************")

    import_uvfits_set(path_data_0,path_vex,path_out,out_name,bandname,pipeline_name=pipeline_name,tavg=tavg,
    only_parallel=False,filend=filend,incoh_avg=incoh_avg,out_type=out_type,rescale_noise=rescale_noise,polrep=polrep, old_format=old_format,
    path_ehtim=path_ehtim,closure=closure,tavg_closures=tavg_closures,precoh_avg_time=precoh_avg_time)
    return 0

if __name__=='__main__':

    if ("-h" in sys.argv) or ("--h" in sys.argv):
        print("importing data")
        sys.exit()

    if "--datadir" in sys.argv:
        for a in range(0, len(sys.argv)):
            if(sys.argv[a] == '--datadir'):
                path_data_0 = sys.argv[a+1]
    else:
        raise Exception("must provide data directory!")

    if "--pipeline" in sys.argv:
        for a in range(0, len(sys.argv)):
            if(sys.argv[a] == '--pipeline'):
                pipeline_name = sys.argv[a+1]
    else:   pipeline_name = 'hops'

    if "--outname" in sys.argv:
        for a in range(0, len(sys.argv)):
            if(sys.argv[a] == '--outname'):
                out_name = sys.argv[a+1]
    else:   out_name = pipeline_name

    if "--outdir" in sys.argv:
        for a in range(0, len(sys.argv)):
            if(sys.argv[a] == '--outdir'):
                path_out = sys.argv[a+1]
    else:   path_out = datadir

    if "--ehtdir" in sys.argv:
        for a in range(0, len(sys.argv)):
            if(sys.argv[a] == '--ehtdir'):
                path_ehtim = sys.argv[a+1]

    if "--bandname" in sys.argv:
        for a in range(0, len(sys.argv)):
            if(sys.argv[a] == '--bandname'):
                bandname = sys.argv[a+1]
    else:
        bandname='XXX'

    if "--filend" in sys.argv:
        for a in range(0, len(sys.argv)):
            if(sys.argv[a] == '--filend'):
                filend = sys.argv[a+1]
    else:
        filend='.uvfits'

    if "--out_type" in sys.argv:
        for a in range(0, len(sys.argv)):
            if(sys.argv[a] == '--out_type'):
                out_type = sys.argv[a+1]
    else:
        out_type='hdf'

    if "--path_vex" in sys.argv:
        for a in range(0, len(sys.argv)):
            if(sys.argv[a] == '--path_vex'):
                path_vex = sys.argv[a+1]
    else:
        path_vex=VEX_DEFAULT

    if "--tavg" in sys.argv:
        for a in range(0, len(sys.argv)):
            if(sys.argv[a] == '--tavg'):
                if(sys.argv[a+1]=='scan'):
                    tavg='scan'
                else: tavg=float(sys.argv[a+1])

    tavg_closures='scan'
    if "--tavg_closures" in sys.argv:
        for a in range(0, len(sys.argv)):
            if(sys.argv[a] == '--tavg_closures'):
                if(sys.argv[a+1]=='scan'):
                    tavg_closures='scan'
                else: tavg_closures=float(sys.argv[a+1])

    rescale_noise=False
    if "--rescale_noise" in sys.argv:
        rescale_noise=True

    incoh_avg=False
    if "--incoh_avg" in sys.argv:
        incoh_avg=True

    precoh_avg_time=0
    if "--incoh_avg_time" in sys.argv:
        for a in range(0, len(sys.argv)):
            if(sys.argv[a] == '--precoh_avg_time'):
                precoh_avg_time=float(sys.argv[a+1])

    if "--cphase" in sys.argv:
        closure='cphase'
    else: closure=''

    if "--polrep" in sys.argv:
        for a in range(0, len(sys.argv)):
            if(sys.argv[a] == '--polrep'):
                polrep = sys.argv[a+1]
    else: polrep='circ'

    print("********************************************************")
    print("*********************IMPORT DATA************************")
    print("********************************************************")

    print('path_data_0 = ', path_data_0)
    print('path_vex = ', path_vex)
    print('path_out = ', path_out)
    print('out_name = ', out_name)
    print('pipeline_name= ', pipeline_name)
    print('tavg = ', tavg)
    print('out_type = ', out_type)
    main(path_data_0,path_vex,path_out,out_name,bandname,pipeline_name=pipeline_name,tavg=tavg,
    only_parallel=False,filend=filend,incoh_avg=incoh_avg,out_type=out_type,rescale_noise=rescale_noise,polrep=polrep, 
    old_format=True,path_ehtim=path_ehtim,closure=closure,tavg_closures=tavg_closures,precoh_avg_time=precoh_avg_time)
