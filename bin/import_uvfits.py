#!/usr/bin/env python
#
# Maciek Wielgus 02/Oct/2018
# Kazu Akiama    Feb/2020

import pandas as pd
import numpy as np
from eat.io import uvfits
from eat.inspect import utils as ut
from eat.inspect import closures as cl
import os
import sys
import glob
import argparse

def import_uvfits_set(datadir, vexdir, outdir, observation, idtag, band, tavg='scan', only_parallel=False, infileext="uvfits", incoh_avg=False, outfiletype='hdf5',
                      rescale_noise=False, polrep='circ', old_format=True, ehtimpath='', closure='both', tavgclosure='scan', tavgprecoh=0., sigma=0,
                      sigmascalefactor=1.):
    '''
    Imports whole dataset of uvfits with HOPS folder structure, or even without structure
    '''

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    df = pd.DataFrame({})

    #path0a = glob.glob(datadir+'/*/*'+infileext)
    #path0b = glob.glob(datadir+'/*'+infileext)
    #path0 = sorted(path0a+path0b)

    path0 = sorted(glob.glob(f'{datadir}/**/*.{infileext}', recursive=True))

    # extract and store visibilities
    for filen in path0:
      print(f'Processing {filen}')
      try:
        df_foo = uvfits.get_df_from_uvfit(filen, observation=observation, path_vex=vexdir, force_singlepol='no', band=band, round_s=0.1,
                                          only_parallel=only_parallel, rescale_noise=rescale_noise, polrep=polrep, path_ehtim=ehtimpath,
                                          fix_sigma=sigma, scale_sigma=sigmascalefactor)
        print('Found datapoints: ',np.shape(df_foo)[0])
            
        # convert to old format (i.e. separate data record for each polarization)
        if old_format:
            df_foo = ut.old_format(df_foo)
        if 'std_by_mean' in df_foo.columns:
            df_foo.drop('std_by_mean',axis=1,inplace=True)
        df_foo['std_by_mean'] = df_foo['amp']
        if 'amp_moments' in df_foo.columns:
            df_foo.drop('amp_moments',axis=1,inplace=True)
        df_foo['amp_moments'] = df_foo['amp']
        if 'sig_moments' in df_foo.columns:
            df_foo.drop('sig_moments',axis=1,inplace=True)
        df_foo['sig_moments'] = df_foo['amp']
        
        # coherently / incoherently average visibilities
        if incoh_avg==False:
            print('Averaging coherently for ', str(tavg))
            df_scan = ut.coh_avg_vis(df_foo.copy(),tavg=tavg,phase_type='phase')
        else:      
            if tavgprecoh > 0:
                print('Averaging coherently for ', str(tavgprecoh))
                df_coh = ut.coh_avg_vis(df_foo.copy(),tavg=tavgprecoh,phase_type='phase')
                print('Averaging incoherently for ', str(tavg))
                df_scan = ut.incoh_avg_vis(df_coh.copy(),tavg=tavg,phase_type='phase')
            else:
                print('Averaging incoherently for ', str(tavg))
                df_scan = ut.incoh_avg_vis(df_foo.copy(),tavg=tavg,phase_type='phase')
        df = pd.concat([df,df_scan.copy()],ignore_index=True)
      except: print('Nothing from this file...')

    try:
        df.drop(list(df[df.baseline.str.contains('R')].index.values),inplace=True)
    except:
        pass
    try:
        df['source'] = list(map(str,df['source']))
    except:
        pass
    try:
        df.dropna(subset=['snr'],inplace=True)
    except:
        pass
    
    # compute closure phases
    if closure in ['cphase', 'both']:
        print("Saving scan-averaged closure phases...")
        bsp = cl.all_bispectra(df,phase_type='phase')
        bsp.drop('fracpols',axis=1,inplace=True)
        bsp.drop('snrs',axis=1,inplace=True)
        bsp.drop('amps',axis=1,inplace=True)
        bsp_sc = ut.coh_avg_bsp(bsp,tavg=tavgclosure)
        idtag_cp = 'cp_sc_'+idtag

        if outfiletype in ['hdf5', 'both']:
            ftmp = os.path.join(outdir, f'{idtag_cp}.h5')
            print(f'Saving file: {ftmp}')
            bsp_sc.to_hdf(ftmp, key=idtag_cp, mode='w',format='table')
        elif outfiletype in ['pickle', 'both']:
            ftmp = os.path.join(outdir, f'{idtag_cp}.pickle')
            print(f'Saving file: {ftmp}')
            bsp_sc.to_pickle(ftmp)
    
    # compute log closure amplitudes
    if closure in ['lcamp', 'both']:
        print("Saving scan-averaged log closure amplitudes...")
        quad=cl.all_quadruples_new(df,ctype='logcamp',debias='camp')
        quad.drop('snrs',axis=1,inplace=True)
        quad.drop('amps',axis=1,inplace=True)
        quad_sc=ut.avg_camp(quad,tavg=tavgclosure)
        idtag_lca= 'lca_sc_'+idtag
        quad_sc['scan_id'] = list(map(np.int64,quad_sc.scan_id))

        if outfiletype in ['hdf5', 'both']:
            ftmp = os.path.join(outdir, f'{idtag_lca}.h5')
            print(f'Saving file: {ftmp}')
            quad_sc.to_hdf(ftmp, key=idtag_lca, mode='w',format='table')
        elif outfiletype in ['pickle', 'both']:
            ftmp = os.path.join(outdir, f'{idtag_lca}.pickle')
            print(f'Saving file: {ftmp}')
            quad_sc.to_pickle(ftmp)

    # save dataframe to h5
    if outfiletype in ("hdf5", "both"):
        hdf5_path = os.path.join(outdir, f"{idtag}.h5")
        print(f"Saving file: {hdf5_path}")
        df.to_hdf(hdf5_path, key=idtag, mode="w", format="table")

    elif outfiletype in ("pickle", "both"):
        pickle_path = os.path.join(outdir, f"{idtag}.pickle")
        print(f"Saving file: {pickle_path}")
        df.to_pickle(pickle_path)

    else:
        return df

def create_parser():
    p = argparse.ArgumentParser()

    p.add_argument("datadir", help="Directory containing input UVFITS files")
    p.add_argument("vexdir", help="Directory containing VEX schedules")
    p.add_argument("outdir", help="Directory to which HDF5 files must be written")
    p.add_argument('--observation', type=str, default='EHT2021', choices=['EHT2017','EHT2018','EHT2021'], help="string denoting which eht campaign's data are being reduced")
    p.add_argument('--idtag', type=str, default='hops', help="custom identifier tag for HDF5 files")
    p.add_argument('--band', type=str, required=True, choices=['b1', 'b2', 'b3', 'b4', 'lo', 'hi'], help="band name (lo/hi for 2017; b1/b2/b3/b4 for 2018+)")
    p.add_argument('--tavg', type=float, default=-1.0, help="Time-averaging resolution (-1 to average over entire scan)")
    p.add_argument('--parhandonly', action='store_true', help='Consider only parallel hand visibilities')
    p.add_argument('--infileext', type=str, default='uvfits', help="Input uvfits file extension (usually uvfits or uvf)")
    p.add_argument('--incoh_avg', action='store_true', help='Perform incoherent averaging')
    p.add_argument('--outfiletype', type=str, default='hdf5', choices=['hdf5', 'pickle', 'both'], help="Output file format(s)")
    p.add_argument('--rescale_noise', action='store_true', help='Rescale noise')
    p.add_argument('--polrep', type=str, default='circ', choices=['circ', 'stokes'], help="Polarimetric representation in which to load data as")
    p.add_argument('--newformat', action='store_true', help='Store data in new format (i.e. do not separate data records by polarization)')
    p.add_argument('--ehtimpath', type=str, help="Path to custom eht-imaging directory")
    p.add_argument('--closure', type=str, default='', choices=['cphase', 'lcamp', 'both', ''], help="Closure quantities to compute and save")
    p.add_argument('--tavgclosure', type=float, default=-1.0, help="Time-averaging resolution for closure quantities (-1 to average over entire scan)")
    p.add_argument('--tavgprecoh', type=float, default=0.0, help="Time-averaging resolution for coherent averaging")
    p.add_argument('--sigma', type=float, default=0.0, help="Fix sigma to this value")
    p.add_argument('--sigmascalefactor', type=float, default=1.0, help="Scale sigma by this value")

    return p

def main(args):
    print('Converting UVFITS files into a single HDF5/pickle file for easy import and inspection in python...')
    print(f'Arguments passed: {args}')

    # convert tavg to values expected by functions downstream
    if args.tavg == -1.:
        tavg = 'scan'
    else:
        tavg = args.tavg
    
    # If newformat flag not specified, resort to saving in old format
    if args.newformat:
        old_format = False
    else:
        old_format = True

    # convert tav_closure to values expected by functions downstream
    if args.tavgclosure == -1.:
        tavgclosure = 'scan'
    else:
        tavgclosure = args.tavgclosure
    
    # read in all uvfits files in the given path and save them in hdf5 or pickle formats
    import_uvfits_set(args.datadir, args.vexdir, args.outdir, args.observation, args.idtag, args.band, tavg=tavg, only_parallel=args.parhandonly,
                      infileext=args.infileext, incoh_avg=args.incoh_avg, outfiletype=args.outfiletype, rescale_noise=args.rescale_noise, polrep=args.polrep,
                      old_format=old_format, ehtimpath=args.ehtimpath, closure=args.closure, tavgclosure=tavgclosure, tavgprecoh=args.tavgprecoh,
                      sigma=args.sigma, sigmascalefactor=args.sigmascalefactor)

    return 0

if __name__=='__main__':
    args = create_parser().parse_args()
    ret = main(args)
    sys.exit(ret)
