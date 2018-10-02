import pandas as pd
import numpy as np
from eat.io import uvfits
from eat.inspect import utils as ut
from eat.inspect import closures as cl
import os,sys,importlib

VEX_DEFAULT='/home/maciek/VEX/'

def import_uvfits_set(path_data_0,data_subfolder,path_vex,path_out,out_name,pipeline_name='hops',tavg='scan',exptL=[3597,3598,3599,3600,3601,''],
    bandL=['lo','hi'],only_parallel=False,filend=".uvfits",incoh_avg=False,out_type='hdf',rescale_noise=False,polrep='circ', 
    old_format=True,path_ehtim='',closure='',tavg_closures='scan'):
    print('path_data_0 = ', path_data_0)
    print('path_vex = ', path_vex)
    print('data_subfolder = ', data_subfolder)
    print('path_out = ', path_out)
    print('out_name = ', out_name)
    print('pipeline_name= ', pipeline_name)
    print('tavg = ', tavg)
    if not os.path.exists(path_out):
        os.makedirs(path_out) 
    df = pd.DataFrame({})
    for band in bandL:  
        for expt in exptL:
            path0 = path_data_0+pipeline_name+'-'+band+'/'+data_subfolder+str(expt)+'/'
            if os.path.exists(path0):
                for filen in os.listdir(path0):
                    if filen.endswith(filend): 
                        print('processing ', filen)
                        #try:
                        df_foo = uvfits.get_df_from_uvfit(path0+filen,path_vex=path_vex,force_singlepol='no',band=band,round_s=0.1,
                        only_parallel=only_parallel,rescale_noise=rescale_noise,polrep=polrep,path_ehtim=path_ehtim)
                        if 'std_by_mean' in df_foo.columns:
                            df_foo.drop('std_by_mean',axis=1,inplace=True)
                        df_foo['std_by_mean'] = df_foo['amp']
                        if incoh_avg==False:
                            print('Averaging coherently for ', str(tavg))
                            df_scan = ut.coh_avg_vis(df_foo.copy(),tavg=tavg,phase_type='phase')
                        else:
                            print('Averaging incoherently for ', str(tavg))
                            df_scan = ut.incoh_avg_vis(df_foo.copy(),tavg=tavg,phase_type='phase')
                        df = pd.concat([df,df_scan.copy()],ignore_index=True)
                        df.drop(list(df[df.baseline.str.contains('R')].index.values),inplace=True)
                        #except: pass
                    else: pass
            else: pass 
    df.drop(list(df[df.baseline.str.contains('R')].index.values),inplace=True)
    df['source'] = list(map(str,df['source']))
    
    if old_format:
        print('Following columns present: ',df.columns)
        df = ut.old_format(df)

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
        out_name_cp = 'cp_'+out_name
        bsp_sc.to_hdf(path_out+out_name_cp+'.h5', key=out_name, mode='w',format='table')
    
        print("Saving scan-averaged log closure amplitudes...")
        quad=cl.all_quadruples_new(df,ctype='logcamp',debias='camp')
        quad.drop('snrs',axis=1,inplace=True)
        quad.drop('amps',axis=1,inplace=True)
        quad_sc=ut.avg_camp(quad,tavg=tavg_closures)
        out_name_lca= 'lca_'+out_name
        quad_sc['scan_id'] = list(map(np.int64,quad_sc.scan_id))
        quad_sc.to_hdf(path_out+out_name_lca+'.h5', key=out_name, mode='w',format='table')

    if len(bandL)==1:
        out_name=out_name+'_'+bandL[0]        
    if out_type=='hdf':
        df.to_hdf(path_out+out_name+'.h5', key=out_name, mode='w',format='table')
    elif out_type=='pic':
        df.to_pickle(path_out+out_name+'.pic')
    elif out_type=='both':
        df.to_hdf(path_out+out_name+'.h5', key=out_name, mode='w',format='table')
        df.to_pickle(path_out+out_name+'.pic')
    else: return df


##################################################################################################################################
##########################  Main FUNCTION ########################################################################################
##################################################################################################################################
def main(path_data_0,data_subfolder,path_vex,path_out,out_name,pipeline_name='hops',tavg='scan',exptL=[3597,3598,3599,3600,3601],
    bandL=['lo','hi'],only_parallel=True,filend=".uvfits",incoh_avg=False,out_type='hdf',rescale_noise=False,polrep=None, old_format=True,path_ehtim='',closure='',tavg_closures='scan'):

    print("********************************************************")
    print("*********************IMPORT DATA************************")
    print("********************************************************")

    import_uvfits_set(path_data_0,data_subfolder,path_vex,path_out,out_name,pipeline_name=pipeline_name,tavg=tavg,exptL=[3597,3598,3599,3600,3601],
    bandL=['lo','hi'],only_parallel=False,filend=filend,incoh_avg=incoh_avg,out_type='hdf',rescale_noise=rescale_noise,polrep=polrep, old_format=old_format,
    path_ehtim=path_ehtim,closure=closure,tavg_closures=tavg_closures)
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

    if "--subfolder" in sys.argv:
        for a in range(0, len(sys.argv)):
            if(sys.argv[a] == '--subfolder'):
                data_subfolder = sys.argv[a+1]
    else:
        data_subfolder=''

    if "--filend" in sys.argv:
        for a in range(0, len(sys.argv)):
            if(sys.argv[a] == '--filend'):
                filend = sys.argv[a+1]
    else:
        filend='.uvfits'

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

    closure=''
    if "--cphase" in sys.argv:
        closure='cphase'
    
    #if "--lcamp" in sys.argv:
    #    closure='lcamp'

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
    print('data_subfolder = ', data_subfolder)
    print('path_out = ', path_out)
    print('out_name = ', out_name)
    print('pipeline_name= ', pipeline_name)
    print('tavg = ', tavg)
    main(path_data_0,data_subfolder,path_vex,path_out,out_name,pipeline_name=pipeline_name,tavg=tavg,exptL=[3597,3598,3599,3600,3601],
    bandL=['lo','hi'],only_parallel=False,filend=filend,incoh_avg=incoh_avg,out_type='hdf',rescale_noise=rescale_noise,polrep=polrep, old_format=True,path_ehtim=path_ehtim,closure=closure,tavg_closures=tavg_closures)
