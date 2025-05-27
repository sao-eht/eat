#!/usr/bin/env python

import os
import sys
import argparse
import logging
from eat.postproc.utils import extract_metadata_from_ovex

# metadata files to be created from ovex files
AZ2Z_FILE = 'cf7_AZ2Z.txt'
SMT2Z_FILE = 'cf7_SMT2Z.txt'
TRACK2EXPT_FILE = 'cf7_track2expt.txt'

def create_parser():
    p = argparse.ArgumentParser()

    p.add_argument("datadir", help="Input directory corresponding to single epoch containing scan directories")
    p.add_argument('--az2zfile', type=str, default=AZ2Z_FILE, help="File to write az2z dict to")
    p.add_argument('--smt2zfile', type=str, default=SMT2Z_FILE, help="File to write smt2z dict to")
    p.add_argument('--track2exptfile', type=str, default=TRACK2EXPT_FILE, help="File to write track2expt dict to")
    p.add_argument('--loglevel', type=str, default='INFO', help='Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')

    return p

def main(args):
    # Configure logging
    loglevel = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(loglevel, int):
        raise ValueError(f'Invalid log level: {args.loglevel}')
    logging.basicConfig(level=loglevel,
                        format='%(asctime)s %(levelname)s:: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    logging.info('Converting HOPS fringe files to UVFITS files...')
    logging.debug(f'Arguments passed: {args}')

    # format datadir properly
    datadir = os.path.normpath(args.datadir)

    # get list of only the subdirectories under datadir and sort them; these are the individual scan directories for a given epoch
    scandirs = sorted([os.path.join(datadir, d) for d in os.listdir(datadir) if os.path.isdir(os.path.join(datadir, d))])

    # call the function to extract metadata from the scan directories
    extract_metadata_from_ovex(scandirs, args.az2zfile, args.smt2zfile, args.track2exptfile)
    
if __name__=='__main__':
    args = create_parser().parse_args()
    ret = main(args)
    sys.exit(ret)