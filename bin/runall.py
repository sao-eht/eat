#!/usr/bin/env python

import glob
import os, sys

######### SET PATHS FOR THE DATA DIRECTORY AND THE CALIBRATION TABLES #########

#Katie  
band = 'hi'
datadir = '/Users/klbouman/Research/vlbi_imaging/software/hops/er1-hops-' + band
caldir = '/Users/klbouman/Research/vlbi_imaging/software/hops/SEFDs/SEFD_HI/'

#Andrew
#band= 'lo'
#caldir = '/home/achael/Desktop/imaging_workshop/HOPS_Rev1/SEFDs/SEFD_LO/'
#datadir = '/home/achael/Desktop/imaging_workshop/HOPS_Rev1/er1-hops-lo'

######### RUN THE UVFITS CONVERSION AND APRIORI CALIBRATION #########

# where the fringe files are stored
fringefolder = '/5.+close/data/'

# loop through all of the day folders in the fringe file
for dayident in os.listdir(datadir + fringefolder) :

    # get the full path to the fringe files
    fringepath = os.path.join(datadir + fringefolder, dayident) 
    
    # ignore the adhoc folder inside of the fringe file folder
    if (dayident != 'adhoc') and os.path.isdir(fringepath):
        print fringepath
        
        # the folders where the uvfits files will be placed
        uvfitsfolder = datadir + '/6.uvfits/' + dayident
        uvcalfolder = datadir + '/7.uvfitscal/' + dayident
        
        # generate the python calls
        call_genuvfits = 'python hops2uvfits.py --clean --uv --outdir ' + uvfitsfolder + ' ' + fringepath
        #call_caluvfits = 'python caluvfits.py --caldir ' + caldir + dayident + ' --outdir ' + uvcalfolder + ' ' + uvfitsfolder
        call_caluvfits = 'python cal_apriori_pang_uvfits.py --caldir ' + caldir + dayident + ' --outdir ' + uvcalfolder + ' ' + uvfitsfolder

        # if directories don't exist then create them
        try:
            os.stat(uvfitsfolder)
        except:
            os.mkdir(uvfitsfolder)
            
        try:
            os.stat(uvcalfolder)
        except:
            os.mkdir(uvcalfolder)
        
        # generate the uvfits files from hops files
        #os.system(call_genuvfits)
        # apriori calibrate the uvfits files using the SEFD tables
        os.system(call_caluvfits)
        
        
        
