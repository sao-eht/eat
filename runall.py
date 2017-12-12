import glob
import os, sys

######### SET PATHS FOR THE DATA DIRECTORY AND THE CALIBRATION TABLES #########
    
band = 'hi'
datadir = '/Users/klbouman/Research/vlbi_imaging/software/hops/er1-hops-' + band
caldir = '/Users/klbouman/Research/vlbi_imaging/software/hops/SEFDs'


######### RUN THE UVFITS CONVERSION AND APRIORI CALIBRATION #########

# where the fringe files are stored
fringefolder = '/5.+close/data/'

# loop through all of the day folders in the fringe file
for dayident in os.listdir(datadir + fringefolder) :
    fringepath = os.path.join(datadir + fringefolder, dayident) 
    if (dayident != 'adhoc') and os.path.isdir(fringepath):
        print fringepath
        
        uvfitsfolder = datadir + '/6.uvfits/' + dayident
        uvcalfolder = datadir + '/7.uvfitscal/' + dayident
        
        call_genuvfits = 'python hops2uvfits.py --clean --uv --outdir ' + uvfitsfolder + ' ' + fringepath
        call_caluvfits = 'python caluvfits.py --caldir ' + caldir + '--outdir ' + uvcalfolder + ' ' + uvfitsfolder

        try:
            os.stat(uvfitsfolder)
        except:
            os.mkdir(uvfitsfolder)
            
        try:
            os.stat(uvcalfolder)
        except:
            os.mkdir(uvcalfolder)
        
        os.system(call_genuvfits)
        os.system(call_caluvfits)
        
        
        