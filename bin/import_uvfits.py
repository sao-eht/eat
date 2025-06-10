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
import logging

# Configure logging
loglevel = getattr(logging, 'INFO', None)
logging.basicConfig(level=loglevel,
                    format='%(asctime)s %(levelname)s:: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

def import_uvfits_set(datadir, vexdir, outdir, observation, idtag, band, tavg='scan', only_parallel=False, infileext="uvfits", incoh_avg=False, outfiletype='hdf5',
                      rescale_noise=False, polrep='circ', old_format=True, ehtimpath='', closure='', tavgclosure='scan', tavgprecoh=0., sigma=0,
                      sigmascalefactor=1.):
    """
    Import (convert) UVFITS files to HDF5 and or pickle files.

    Parameters
    ----------
    datadir : str
        Directory containing input UVFITS files.
    vexdir : str
        Directory containing VEX schedules.
    outdir : str
        Directory to which output files will be written.
    observation : str
        String denoting which EHT campaign's data are being reduced 
        (e.g., 'EHT2017', 'EHT2018', 'EHT2021').
    idtag : str
        Custom identifier tag for output files.
    band : str
        Band name (e.g., 'lo', 'hi' for 2017; 'b1', 'b2', 'b3', 'b4' for 2018+).
    tavg : str or float, optional
        Time-averaging resolution ('scan' for entire scan or a float value).
    only_parallel : bool, optional
        If True, consider only parallel hand visibilities.
    infileext : str, optional
        Input UVFITS file extension (default is 'uvfits').
    incoh_avg : bool, optional
        If True, perform incoherent averaging.
    outfiletype : str, optional
        Output file format(s) ('hdf5', 'pickle', or 'both').
    rescale_noise : bool, optional
        If True, rescale noise.
    polrep : str, optional
        Polarimetric representation ('circ' or 'stokes').
    old_format : bool, optional
        If True, store data in old format (separate data records by polarization).
    ehtimpath : str, optional
        Path to custom eht-imaging directory.
    closure : str, optional
        Closure quantities to compute and save ('cphase', 'lcamp', 'both', or '').
    tavgclosure : str or float, optional
        Time-averaging resolution for closure quantities ('scan' or a float value).
    tavgprecoh : float, optional
        Time-averaging resolution for coherent averaging.
    sigma : float, optional
        Fix sigma to this value.
    sigmascalefactor : float, optional
        Scale sigma by this value.

    Returns
    -------
    None or pandas.DataFrame
        If `outfiletype` is not specified, returns a DataFrame containing the processed data.
        Otherwise, saves the processed data to the specified output format(s).

    Notes
    -----
    This function skips UVFITS files that are already (time and frequency) averaged i.e., those whose filenames end in '+avg'.
    """

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    df = pd.DataFrame({})

    #path0a = glob.glob(datadir+'/*/*'+infileext)
    #path0b = glob.glob(datadir+'/*'+infileext)
    #path0 = sorted(path0a+path0b)

    # Match all files with the specified extension in datadir and its subdirectories
    path0 = sorted(f for f in glob.glob(f'{datadir}/**/*.{infileext}', recursive=True) if not os.path.splitext(os.path.basename(f))[0].endswith('+avg'))

    # extract and store visibilities
    for filen in path0:
      logging.info(f'Processing {filen}')
      try:
        df_foo = uvfits.get_df_from_uvfit(filen, observation=observation, path_vex=vexdir, force_singlepol='no', band=band, round_s=0.1,
                                          only_parallel=only_parallel, rescale_noise=rescale_noise, polrep=polrep, path_ehtim=ehtimpath,
                                          fix_sigma=sigma, scale_sigma=sigmascalefactor)
        logging.info(f'Found datapoints: {np.shape(df_foo)[0]}')
            
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
            logging.info(f'Averaging coherently for {str(tavg)}')
            df_scan = ut.coh_avg_vis(df_foo.copy(),tavg=tavg,phase_type='phase')
        else:      
            if tavgprecoh > 0:
                logging.info(f'Pre-averaging coherently for {str(tavgprecoh)} before incoherently averaging for {str(tavg)}')
                df_coh = ut.coh_avg_vis(df_foo.copy(),tavg=tavgprecoh,phase_type='phase')
                df_scan = ut.incoh_avg_vis(df_coh.copy(),tavg=tavg,phase_type='phase')
            else:
                logging.info(f'No coherent pre-averaging. Incoherently averaging for {str(tavg)}')
                df_scan = ut.incoh_avg_vis(df_foo.copy(),tavg=tavg,phase_type='phase')
        df = pd.concat([df,df_scan.copy()],ignore_index=True)
      except Exception as e:
          logging.warning(f'Exception encountered while processing {filen}: {e}')
          logging.warning('Skipping this file...')

    try:
        df.drop(list(df[df.baseline.str.contains('R')].index.values),inplace=True)
        df['source'] = list(map(str,df['source']))
        df.dropna(subset=['snr'],inplace=True)
    except Exception as e:
        logging.warning(f'Exception encountered while cleaning data: {e}')
        logging.warning('Skipping some cleaning steps...')
    
    # compute closure phases
    if closure in ['cphase', 'both']:
        logging.info("Saving closure phases...")

        bsp = cl.all_bispectra(df,phase_type='phase')

        if bsp.empty:
            logging.warning("No valid bispectra found for any source or epoch! Nothing to save.")
        else:
            bsp.drop('fracpols',axis=1,inplace=True)
            bsp.drop('snrs',axis=1,inplace=True)
            bsp.drop('amps',axis=1,inplace=True)
            bsp_sc = ut.coh_avg_bsp(bsp,tavg=tavgclosure)
            idtag_cp = 'cp_'+idtag

            if outfiletype in ['hdf5', 'both']:
                ftmp = os.path.join(outdir, f'{idtag_cp}.h5')
                logging.info(f'Saving file: {ftmp}')
                bsp_sc.to_hdf(ftmp, key=idtag_cp, mode='w',format='table')
            elif outfiletype in ['pickle', 'both']:
                ftmp = os.path.join(outdir, f'{idtag_cp}.pickle')
                logging.info(f'Saving file: {ftmp}')
                bsp_sc.to_pickle(ftmp)
    
    # compute log closure amplitudes
    if closure in ['lcamp', 'both']:
        logging.info("Saving log closure amplitudes...")

        quad=cl.all_quadruples_new(df,ctype='logcamp',debias='camp')

        if quad.empty:
            logging.warning("No valid quadrangles found for any source or epoch! Nothing to save.")
        else:
            quad.drop('snrs',axis=1,inplace=True)
            quad.drop('amps',axis=1,inplace=True)
            quad_sc=ut.avg_camp(quad,tavg=tavgclosure)
            idtag_lca= 'lca_'+idtag
            quad_sc['scan_id'] = list(map(np.int64,quad_sc.scan_id))

            if outfiletype in ['hdf5', 'both']:
                ftmp = os.path.join(outdir, f'{idtag_lca}.h5')
                logging.info(f'Saving file: {ftmp}')
                quad_sc.to_hdf(ftmp, key=idtag_lca, mode='w',format='table')
            elif outfiletype in ['pickle', 'both']:
                ftmp = os.path.join(outdir, f'{idtag_lca}.pickle')
                logging.info(f'Saving file: {ftmp}')
                quad_sc.to_pickle(ftmp)

    # save dataframe to h5
    if outfiletype in ("hdf5", "both"):
        hdf5_path = os.path.join(outdir, f"{idtag}.h5")
        logging.info(f"Saving file: {hdf5_path}")
        df.to_hdf(hdf5_path, key=idtag, mode="w", format="table")

    elif outfiletype in ("pickle", "both"):
        pickle_path = os.path.join(outdir, f"{idtag}.pickle")
        logging.info(f"Saving file: {pickle_path}")
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
    logging.info('Converting UVFITS files into a single HDF5/pickle file for easy import and inspection in python...')
    logging.info(f'Arguments passed: {args}')

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
