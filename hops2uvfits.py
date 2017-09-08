#hops2uvfits.py

#take data from all fringe files in a directory and put them in a uvfits file

 
import numpy as np
import mk4 # part of recent HOPS install, need HOPS ENV variables
import datetime
import ctypes
import astropy.io.fits as fits
from argparse import Namespace
import glob
import os, sys
from eat.io import util
from eat.plots import util as putil
from astropy.time import Time
import numpy.matlib

# For Andrew:
#DATADIR_DEFAULT = '/home/achael/EHT/hops/data/3554/' #/098-0924/'

# For Katie
DATADIR_DEFAULT = '../3554/'# /098-0916/'
# source hops.bash in /Users/klbouman/Research/vlbi_imaging/software/hops/build
# run this from /Users/klbouman/Research/vlbi_imaging/software/hops/eat


MHZ2HZ = 1e6
RADPERARCSEC = (np.pi / 180.) / 3600.
BLTYPE = [('time','f8'),('t1','a32'),('t2','a32')]
DTARR = [('site', 'a32'), ('x','f8'), ('y','f8'), ('z','f8')]

DTPOL = [('time','f8'),('freq','f8'),('tint','f8'),
            ('t1','a32'),('t2','a32'),
            ('u','f8'),('v','f8'),
            ('rr','c16'),('ll','c16'),('rl','c16'),('lr','c16'),
            ('rrweight','f8'),('llweight','f8'),('rlweight','f8'),('lrweight','f8')]
EP = 1.e-5


#######################################################################
##########################  HELPER FUNCTIONS ##########################
####################################################################### 
def getfringefile(b, filelist=False):
    if type(b) is str:
        files = glob.glob(b)
        if len(files) == 0: # try harder to find file
            tok = b.split('/')
            last = getattr(getfringefile, 'last', [])
            if len(tok) < len(last):
                files = glob.glob('/'.join(last[:-len(tok)] + tok))
        if len(files) == 0:
            return "cannot find file: %s or %s" % (b, '/'.join(last[:-len(tok)] + tok))
        files = [f for f in files if '..' not in f] # filter out correlator files
        if filelist:
            return sorted(files)
        files.sort(key=os.path.getmtime)
        getfringefile.last = files[-1].split('/')
        b = mk4.mk4fringe(files[-1]) # use last updated file
    return b
 
def pop212(b):
    b = getfringefile(b)
    (nchan, nap) = (b.n212, b.t212[0].contents.nap)
    data212 = np.zeros((nchan, nap, 3), dtype=np.float32)
    for i in range(nchan):
        q = (mk4.newphasor*nap).from_address(ctypes.addressof(b.t212[i].contents.data))
        data212[i] = np.frombuffer(q, dtype=np.float32, count=-1).reshape((nap, 3))
    vis = data212[:,:,0] * np.exp(1j * data212[:,:,1])
    return vis.T
 
def mk4time(time):
    return datetime.datetime.strptime("%d-%03d %02d:%02d:%02d.%06d" %
                                      (time.year, time.day, time.hour, time.minute, int(time.second), int(0.5+1e6*(time.second-int(time.second)))),
                                      "%Y-%j %H:%M:%S.%f")
 
 
# unwrap short to positive int in multiples from 1e6 to 1024e6
def short2int(short):
    return short2int.lookup[short]

short2int.lookup = {ctypes.c_short(i*1000000).value:i*1000000 for i in range(1024)}

#######################################################################
##########################  Load/Save FUNCTIONS #######################
####################################################################### 
def convert_bl_fringefiles(datadir=DATADIR_DEFAULT, rot_rate=False, rot_delay=False):

    baselineNames = []
    for filename in glob.glob(datadir + '*'):
        baselineNames.append(os.path.basename(filename).split(os.extsep)[0])

    baselineNames = set(baselineNames)
    
    ##########################  LOAD DATA ##########################
    for baselineName in baselineNames:
        nap = 0
        nchan = 0
        nwindow = 1
        nstokes= 4
        first_pass_flag = True

        #TODO will currently create an empty uvfits file for every non hops file not caught here
        #TODO make eat throw errors if its reading something that's not a fringe file
        if baselineName.split("_")[-1] == "bl":
            continue
        if baselineName.split("_")[-1] == "merged":
            continue
        try:
            if baselineName.split("_")[-2] == "hops":
                continue
        except IndexError: pass
        
        # remove auto correlations 
        if baselineName[0] == baselineName[1]:
            continue

        #print "Making uvfits for baseline: ", baselineName        
        for filename in glob.glob(datadir + baselineName + '*'):
            if filename.split(os.extsep)[-1] == "uvfits":
                continue
       
            #print "reading hops fringe file: ", filename
            a = mk4.mk4fringe(filename)
            b = getfringefile(a)

            if first_pass_flag: #some info we get only once per baseline

                ##########################  SOURCE INFO ##########################
                # name of the source
                srcname = b.t201[0].source

                # get the ra
                ra_hrs = b.t201[0].coord.ra_hrs
                ra_min = b.t201[0].coord.ra_mins
                ra_sec = b.t201[0].coord.ra_secs

                # get the dec
                dec_degs = b.t201[0].coord.dec_degs
                dec_mins = b.t201[0].coord.dec_mins
                dec_secs = b.t201[0].coord.dec_secs

                ra =  ra_hrs+ra_min/60.+ra_sec/3600.
                dec = np.sign(dec_degs)*(np.abs(dec_degs)+(dec_mins/60.+dec_secs/3600.))

                ########################## OBSERVATION INFO ########################
                ref_freq = b.t205.contents.ref_freq * MHZ2HZ # refrence frequency for all channels
                nchan = b.n212 # number of channels
                nap = b.t212[0].contents.nap # number of data points

                # observation time
                obs_year = b.t205.contents.start.year
                obs_day = b.t205.contents.start.day
                obs_minute = b.t205.contents.start.minute
                obs_second =  b.t205.contents.start.second

                # get the fixed integration time
                totaltime  = (mk4time(b.t205.contents.stop) - mk4time(b.t205.contents.start)).total_seconds()
                inttime_fixed = totaltime/nap

                ##########################  ANTENNA INFO ##########################
                # names of antennas
                ant1 = b.t202.contents.ref_name # antenna 1 name
                ant2 = b.t202.contents.rem_name # anetnna 2 name
                baselineName = b.t202.contents.baseline # first one is ref antenna second one is rem

                # x, y, z coordinate of antenna 1
                ant1_x = b.t202.contents.ref_xpos # meters
                ant1_y = b.t202.contents.ref_ypos # meters
                ant1_z = b.t202.contents.ref_zpos # meters

                # x, y, z coordinate of antenna 2
                ant2_x = b.t202.contents.rem_xpos # meters
                ant2_y = b.t202.contents.rem_ypos # meters
                ant2_z = b.t202.contents.rem_zpos # meters

                # get the elevation for opacity estimate
                ant1_elevation = b.t202.contents.ref_elev # degrees
                ant2_elevation = b.t202.contents.rem_elev # degrees

                antennas = {} # WARNING: DO NOT REINITLIIZE THIS IF WE LOOP OVER FILES
                xyz = []
                antnames = []
                antnums = []

                if ant1 not in antennas.keys():
                    antennas[ant1] = len(antennas.keys()) + 1
                    xyz.append([ant1_x, ant1_y, ant1_z]) #warning: do not take this out of the if statement with setting the key of the dictionary
                    antnames.append(ant1)
                    antnums.append(antennas[ant1])

                if ant2 not in antennas.keys():
                    antennas[ant2] = len(antennas.keys()) + 1
                    xyz.append([ant2_x, ant2_y, ant2_z]) #warning: do not take this out of the if statement with setting the key of the dictionary
                    antnames.append(ant2)
                    antnums.append(antennas[ant2])

                xyz = np.array(xyz)
                antnames = np.array(antnames)
                antnums = np.array(antnums)
                
                ##########################  MEASUREMENT INFO #########################
                #TODO - need u & v coordinates 

                u_static = (1./RADPERARCSEC) * b.t202.contents.u  #Fringes/arcsec E-W 1GHz
                v_static = (1./RADPERARCSEC) * b.t202.contents.v  #Fringes/arcsec N-S 1GHz
                #print ant1,ant2, u_static/1.e9, v_static/1.e9, np.sqrt(u_static**2 + v_static**2)/1.e9

                outdat = np.zeros((nap, 1, 1, nwindow, nchan, nstokes, 3))
                outdat[:,:,:,:,:,:,2] = -1.0

                # this is a static u/v. we will want to change it with the real time later
                u = u_static * np.ones(outdat.shape[0])
                v = v_static * np.ones(outdat.shape[0])
                w = np.zeros(outdat.shape[0])

                ndat = outdat.shape[0]
                numberofchannels = outdat.shape[4]

                # we believe that this ordeirng of bl has to be the same as the ordering/conjugation in the data
                if antennas[ant1] < antennas[ant2]:
                    bls = (256*antennas[ant1] + antennas[ant2]) * np.ones(outdat.shape[0])
                else:
                    bls = (256*antennas[ant2] + antennas[ant1]) * np.ones(outdat.shape[0])

                ################### ONLY LOAD ONCE ###################################
                first_pass_flag = False


            # loop through channels and get the frequency, polarization, id, and bandwidth
            # get the channel information
            clabel = [q.ffit_chan_id for q in b.t205.contents.ffit_chan[:nchan]]
            cidx = [q.channels[0] for q in b.t205.contents.ffit_chan[:nchan]]
            cinfo = [b.t203[0].channels[i] for i in cidx]

            # TODO CHECK THIS IS THE SAME THROUGHOUT THE LOOP EXCEPT CHANNEL POL
            channel_freq_ant1 = np.zeros([len(cinfo)])
            channel_freq_ant2 = np.zeros([len(cinfo)])
            channel_pol_ant1 = []
            channel_pol_ant2 = []
            channel_id_ant1 = []
            channel_id_ant2 = []
            channel_bw_ant1 = np.zeros([len(cinfo)])
            channel_bw_ant2 = np.zeros([len(cinfo)])
            count = 0

            for ch in cinfo:
                # frequency of the channel (in Hz)
                channel_freq_ant1[count] = ch.ref_freq
                channel_freq_ant2[count] = ch.rem_freq

                # bandwidth of the channel (in Hz) - mutliply by 0.5 because it is the sample_rate
                channel_bw_ant1[count] = 0.5*short2int(ch.sample_rate)
                channel_bw_ant2[count] = 0.5*short2int(ch.sample_rate)

                # the polarization 'L' or 'R'
                channel_pol_ant1.append(ch.refpol)
                channel_pol_ant2.append(ch.rempol)

                # the channel id
                channel_id_ant1.append(ch.ref_chan_id)
                channel_id_ant2.append(ch.rem_chan_id)
                count = count + 1

            channel_pol_ant1 = np.array(channel_pol_ant1)
            channel_id_ant1= np.array(channel_id_ant1)
            channel_pol_ant2 = np.array(channel_pol_ant2)
            channel_id_ant2= np.array(channel_id_ant2)
        
            if len(set(channel_bw_ant1)) != 1:
                print channel_bw_ant1
                raise Exception('the channels on this baseline have different bandwidths')
            if channel_freq_ant1[0] != ref_freq:
                pass
                #print 'warning: channel 1 != ref_freq'
            
            channel_bw = channel_bw_ant1[0]
            channel1_freq = channel_freq_ant1[0]
            nsta = len(antennas)
            bw = channel_bw*numberofchannels
            
            # the proportion of data used to generate each measurement.
            weights = np.zeros([nchan,nap])
            for i in range(0,nchan):
                for j in range(0,nap):
                    weights[i,j] = b.t212[i].contents.data[j].weight

            # the integration time for each measurement
            inttime = inttime_fixed*weights
            tints = np.mean(inttime,axis=0)
            mjd = Time(mk4time(b.t205.contents.start)).mjd
            jds = (2400000.5 + np.floor(mjd)) * np.ones(len(outdat))
            obsseconds = np.arange(0, len(outdat)*inttime_fixed, inttime_fixed )
            fractimes = (mjd - np.floor(mjd)) + ( obsseconds / 86400.)
            jds = jds + fractimes
            #print filename
            #print len(outdat)*inttime_fixed, (np.max(fractimes) - np.min(fractimes)) * 3600 * 24, (np.max(jds) - np.min(jds)) * 3600 * 24

            # get the complex visibilities
            visibilities = pop212(a)
            if antennas[ant2] < antennas[ant1]:
                visibilities =  visibilities.conj()

            amplitude = b.t208[0].amplitude * 10000.
            #amplitude = np.abs(np.mean(visibilities))
            #print 'WARNING: the coherent average is currently off by 4 orders of magnitude - check it!'
        
            snr = b.t208[0].snr
            sigma_full = amplitude/snr
            sigma_ind = sigma_full * np.sqrt( np.sum(weights) / weights )
            recover_snr = np.abs(np.mean(visibilities)) / np.sqrt(np.mean(sigma_ind**2)/np.prod(sigma_ind.shape))

            ##########################  to uvfits style visibility table #################
            #TODO what is visweight ??

            delay = b.t208[0].resid_mbd * 1e-6 # delay in sec
            rate = b.t208[0].resid_rate * 1e-6 # rate in sec/sec^2
            centertime = (mk4time(b.t205[0].utc_central) - mk4time(b.t205.contents.start)).total_seconds()
            

            shift = np.zeros((nchan, nap))
            
            if rot_rate:
                rate_mtx = ref_freq*(np.matlib.repmat( ( inttime_fixed * np.arange(nap) ).reshape(1,nap), nchan, 1 ) - centertime)
                shift = shift + (2 * np.pi *  rate * rate_mtx )
            
                
            if rot_delay:
                fedge = np.array([ ch.ref_freq for ch in cinfo])
                flip = np.array([-1 if ch.refsb == 'L' else 1 for ch in cinfo])
                dfvec = fedge + 0.5*flip*np.array([0.5 * short2int(ch.sample_rate) for ch in cinfo]) - ref_freq
                delay_mtx = np.matlib.repmat( dfvec.reshape(dfvec.shape[0], 1), 1, nap )
                shift = shift + (2 * np.pi *  delay * delay_mtx )


            for i in xrange(nchan):
                
                visweight = 1.0 / (sigma_ind[i,:]**2)
                vis_i = visibilities[:,i]
                vis_i = vis_i * np.exp( 1j * shift[i,:] )
                
                #visweight = inttime[i,:] * channel_bw_ant1[i] #TODO: WHAT WERE WE DOING?
                if channel_pol_ant1[i]=='R' and channel_pol_ant2[i]=='R':
                    outdat[:,0,0,0,i,0,0] = np.real(vis_i)
                    outdat[:,0,0,0,i,0,1] = np.imag(vis_i)
                    outdat[:,0,0,0,i,0,2] = visweight
                if channel_pol_ant1[i]=='L' and channel_pol_ant2[i]=='L':
                    outdat[:,0,0,0,i,1,0] = np.real(vis_i)
                    outdat[:,0,0,0,i,1,1] = np.imag(vis_i)
                    outdat[:,0,0,0,i,1,2] = visweight
                if channel_pol_ant1[i]=='R' and channel_pol_ant2[i]=='L':
                    outdat[:,0,0,0,i,2,0] = np.real(vis_i)
                    outdat[:,0,0,0,i,2,1] = np.imag(vis_i)
                    outdat[:,0,0,0,i,2,2] = visweight
                if channel_pol_ant1[i]=='L' and channel_pol_ant2[i]=='R':
                    outdat[:,0,0,0,i,3,0] = np.real(vis_i)
                    outdat[:,0,0,0,i,3,1] = np.imag(vis_i)
                    outdat[:,0,0,0,i,3,2] = visweight

            print "     ", baselineName, ant1, ant2, ":",channel_pol_ant1[-1], channel_pol_ant2[-1]

        ##########################  save baseline uvfits file ############################
        # pack data
        obs_info = (srcname, ra, dec, ref_freq, channel_bw, channel1_freq, nchan)
        antenna_info = (antnames, antnums, xyz)
        rg_params = (u,v,bls,jds, tints)
        fname= datadir + baselineName + '_hops_bl.uvfits'
        #print "Saving baseline uvfits file: ", fname 
        save_uvfits(obs_info, antenna_info, rg_params, outdat, fname)


def load_hops_uvfits(filename):
    """Load uvfits data from a uvfits file.
    """
        
    # Read the uvfits file
    hdulist = fits.open(filename)
    header = hdulist[0].header
    data = hdulist[0].data
    
    # Load the telescope array data
    tnames = hdulist['AIPS AN'].data['ANNAME']
    tnums = hdulist['AIPS AN'].data['NOSTA'] - 1
    xyz = hdulist['AIPS AN'].data['STABXYZ']
    antenna_info = (tnames, tnums, xyz)

    # Load the various header parameters
    ra = header['OBSRA'] * 12./180.
    dec = header['OBSDEC']   
    src = header['OBJECT']
    
    rf = hdulist['AIPS AN'].header['FREQ']    
    if header['CTYPE4'] == 'FREQ':
        ch1_freq = header['CRVAL4']
        ch_bw = header['CDELT4']
        nchan = header['NAXIS4']
    else: raise Exception('Cannot find observing frequency/bandwidth!')
    
    obs_info = (src, ra, dec, rf, ch_bw, ch1_freq, nchan)

    # Load the random group parameters and the visibility data
    u = data['UU---SIN'] * rf     # Convert uv in lightsec to lambda by multiplying by rf
    v = data['VV---SIN'] * rf  
    baselines = data['BASELINE']  
    jds = data['DATE'].astype('d') + data['_DATE'].astype('d')
    tints = data['INTTIM']       
    obsdata = data['DATA']
        
    rg_params = (u,v,baselines,jds, tints)
    
    return (obs_info, antenna_info, rg_params, obsdata)

def load_and_convert_hops_uvfits(filename):
    """Load uvfits data from a uvfits file.
       Convert data to a format similar to ehtim format for easier merging
    """
        
    # Load the uvfits file using load_hops_uvfits
    alldata = load_hops_uvfits(filename)
    obs_info = alldata[0]
    antenna_info = alldata[1]
    rg_params = alldata[2]
    obsdata =  alldata[3]

    # get the various header parameters
    (src, ra, dec, rf, ch_bw, ch1_freq, nchan) = obs_info

    # put the array data in a telescope array
    (tnames,tnums,xyz) = antenna_info
    tarr = [np.array((tnames[i], xyz[i][0], xyz[i][1], xyz[i][2]),
            dtype=DTARR) for i in range(len(tnames))]
    tarr = np.array(tarr)
    
    # put the random group data and vis data into a data array  
    # Convert uv in lightsec to lambda by multiplying by rf
    u = rg_params[0]
    v = rg_params[1]

    # Sites - add names
    baseline = rg_params[2]
    t1 = baseline.astype(int)/256
    t2 = baseline.astype(int) - t1*256
    t1 = t1 - 1
    t2 = t2 - 1
    #scopes_num = np.sort(list(set(np.hstack((t1,t2)))))
    t1 = np.array([tarr[i]['site'] for i in t1])
    t2 = np.array([tarr[i]['site'] for i in t2])

    # Obs Times
    jds = rg_params[3]
    mjd = int(np.min(jds)-2400000.5)
    times = (jds - 2400000.5 - mjd) * 24.0

    # Integration times
    tints = rg_params[4]
                        
    # Get vis data
    rr = obsdata[:,0,0,0,:,0,0] + 1j*obsdata[:,0,0,0,:,0,1]
    ll = obsdata[:,0,0,0,:,1,0] + 1j*obsdata[:,0,0,0,:,1,1]
    rl = obsdata[:,0,0,0,:,2,0] + 1j*obsdata[:,0,0,0,:,2,1]
    lr = obsdata[:,0,0,0,:,3,0] + 1j*obsdata[:,0,0,0,:,3,1]

    rrweight = obsdata[:,0,0,0,:,0,2]
    llweight = obsdata[:,0,0,0,:,1,2]
    rlweight = obsdata[:,0,0,0,:,2,2]
    lrweight = obsdata[:,0,0,0,:,3,2]
    #mask = ((rrweight >= 0) + (llweight >= 0) + (rlweight >= 0) + (lrweight >= 0))     # Mask to screen bad data

    # Make a datatable
    #TODO check jd not cut off by precision
    datatable = []
    for i in xrange(len(jds)):
        for j in xrange(nchan):
            freq = j*ch_bw + ch1_freq
            datatable.append(np.array(
                             (jds[i], freq, tints[i], 
                              t1[i], t2[i],
                              u[i], v[i],
                              rr[i,j], ll[i,j], rl[i,j], lr[i,j],
                              rrweight[i,j], llweight[i,j], rlweight[i,j], lrweight[i,j]), dtype=DTPOL
                             )
                            )
    datatable = np.array(datatable)
    
    params = (src, ra, dec, rf, ch_bw, ch1_freq, nchan)
    
    #!AC TODO get calibration flags from uvfits?
    return (params, tarr, datatable)

def merge_hops_uvfits(fitsFiles):
    """load and merge all uvfits files in a data directory
    """

    nwindow = 1
    nstokes = 4

    param_list = []
    tarr_list = []
    datatable_list = []
    firstfreq_list = []
    lastfreq_list = []

    for fitsFile in fitsFiles:
        fname=  fitsFile
        #print "read baseline uvfits file: ", fname 
        out = load_and_convert_hops_uvfits(fname)
        
        param_list.append(out[0])
        tarr_list.append(out[1])
        datatable_list.append(out[2])
        firstfreq_list.append(np.min(out[2]['freq']))
        lastfreq_list.append(np.max(out[2]['freq']))

    if (len(set([param[0] for param in param_list]) ) >1 ):
        raise Exception('sources not the same!')
    else:
        src = param_list[0][0]

    if (len(set([param[1] for param in param_list]) )>1):
        raise Exception('ra not the same!') 
    else:
        ra = param_list[0][1]

    if (len(set([param[2] for param in param_list]) )>1):
        raise Exception('dec not the same!') 
    else:
        dec = param_list[0][2]

    if (len(set([param[3] for param in param_list]) )>1):
        print [param[3] for param in param_list]
        raise Exception('rf not the same!') 
    else:
        ref_freq = param_list[0][3]
    
    if (len(set([param[4] for param in param_list]) )>1):
        raise Exception('channel bandwidth not the same!') 
    else:
        ch_bw = float(param_list[0][4])
    
    firstfreq_min = np.min(np.array(firstfreq_list))
    lastfreq_max = np.max(np.array(lastfreq_list))
    for i in range(len(firstfreq_list)):
        diff = (firstfreq_list[i] - firstfreq_min)/ch_bw
        if (np.abs(diff - np.round(diff)) > EP):
            raise Exception('all channels not aligned!')
    
    nchan = (lastfreq_max - firstfreq_min)/ch_bw + 1
    if (np.abs(nchan - np.round(nchan)) > EP):
        raise Exception('channel number not an integer!')
    nchan = int(np.round(nchan))
    ch1_freq = firstfreq_min

    if not (ch1_freq in [param[5] for param in param_list]):
        raise Exception('ch1_freq determined from merging ehtim style datatable not in any read uvfits header!')

    #print nchan, [param[6] for param in param_list]
    #if not (nchan in [param[6] for param in param_list]):
    #    raise Exception('nchan determined from merging ehtim style datatable not in any read uvfits header!')


    obs_info = (src, ra, dec, ref_freq, ch_bw, ch1_freq, nchan)

    print "# baseline files: ", len(fitsFiles)
    print "source: ", src
    print "ra: ", ra
    print "dec: ", dec
    print "ch1 freq: ", ch1_freq/1.e9, "GHz"
    print "ref freq: ", ref_freq/1.e9, "GHz"
    print "# channels: ", nchan

    #print "Merging data ... "

    # merge telescope arrays
    tarr_merge = np.hstack(tarr_list)
    _, idx = np.unique(tarr_merge, return_index=True)  
    tarr_merge = tarr_merge[idx]

    if len(tarr_merge) != len(set(tarr_merge['site'])):
        raise Exception('same name telescopes have different coords!')
    tkeys = {tarr_merge[i]['site']: i for i in range(len(tarr_merge))}
    
    tnames = tarr_merge['site']
    tnums = np.arange(1, len(tarr_merge) + 1)
    xyz = np.array([[tarr_merge[i]['x'],tarr_merge[i]['y'],tarr_merge[i]['z']] for i in np.arange(len(tarr_merge))])
    antenna_info = (tnames, tnums, xyz)
            
    # merge data table 
    datatable_merge = np.hstack(datatable_list)
    datatable_merge.sort(order=['time','t1'])
    #print 'SORT BY TIME!!!'
    bl_list = []
    for i in xrange(len(datatable_merge)):
        entry = datatable_merge[i]
        t1num = entry['t1']
        t2num = entry['t2']
        if tkeys[entry['t1']] > tkeys[entry['t2']]: # reorder telescopes if necessary
            #print entry['t1'], tkeys[entry['t1']], entry['t2'], tkeys[entry['t2']]
            entry['t1'] = t2num
            entry['t2'] = t1num
            entry['u'] = -entry['u']
            entry['v'] = -entry['v']
            entry['rr'] = np.conj(entry['rr'])
            entry['ll'] = np.conj(entry['ll'])
            entry['rl'] = np.conj(entry['rl'])
            entry['lr'] = np.conj(entry['lr'])
            datatable_merge[i] = entry
        bl_list.append(np.array((entry['time'],entry['t1'],entry['t2']),dtype=BLTYPE))
    
    # get unique baseline data
    unique_bl_list, unique_idx_anttime, idx_anttime = np.unique(bl_list, return_index=True, return_inverse=True) 
    _, unique_idx_freq, idx_freq = np.unique(datatable_merge['freq'], return_index=True, return_inverse=True) 
    nap = len(unique_idx_anttime)

    #random group params    
    u = datatable_merge['u'][unique_idx_anttime]
    v = datatable_merge['v'][unique_idx_anttime]
    jds = datatable_merge['time'][unique_idx_anttime]
    #TODO plus one correct?
    t1num = [tkeys[scope] + 1 for scope in datatable_merge['t1'][unique_idx_anttime]]
    t2num = [tkeys[scope] + 1 for scope in datatable_merge['t2'][unique_idx_anttime]]
    bls = 256*np.array(t1num) + np.array(t2num)
    tints = datatable_merge['tint'][unique_idx_anttime]
    rg_params = (u,v,bls,jds, tints)

    #merged uvfits format data table
    outdat = np.zeros((nap, 1, 1, nwindow, nchan, nstokes, 3))
    outdat[:,:,:,:,:,:,2] = -1.0  
    
    vistypes = ['rr','ll','rl','lr']
    for i in xrange(len(datatable_merge)):
        row_freq_idx = idx_freq[i]
        row_dat_idx = idx_anttime[i]
        
        for j in range(len(vistypes)):
            outdat[row_dat_idx,0,0,0,row_freq_idx,j,0] = np.real(datatable_merge[i][vistypes[j]])
            outdat[row_dat_idx,0,0,0,row_freq_idx,j,1] = np.imag(datatable_merge[i][vistypes[j]])
            outdat[row_dat_idx,0,0,0,row_freq_idx,j,2] = datatable_merge[i][vistypes[j]+'weight']

    # return data for saving
    return (obs_info, antenna_info, rg_params, outdat)
            
def save_uvfits(obs_info, antenna_info, rg_params, outdat, fname):   
    """save information already in uvfits format to uvfits file
    """

    # unpack data
    (src, ra, dec, ref_freq, ch_bw, ch1_freq, nchan) = obs_info
    (antnames, antnums, xyz) = antenna_info
    nsta = len(antnames)
    bw = nchan*ch_bw

    (u,v,bls,jds, tints) =  rg_params
    if (len(u) != len(v) != len(bls) != len(jds) != len(tints) != len(outdat)):
        raise Exception("rg parameter shapes and data shape not consistent!")

    ndat = len(u)
    mjd = int(np.min(jds)-2400000.5)
    jd_start = (2400000.5 + mjd)
    fractimes = (jds - jd_start)
    jds_only = np.ones(ndat) * jd_start

    #print "timedur uvfits " , (np.max(jds) - np.min(jds)) * 3600 * 24, (np.max(fractimes) - np.min(fractimes)) * 3600 * 24
    nwindow = 1
    nstokes = 4

    # Open template UVFITS
    #hdulist = fits.open('./template.uvfits')

    # Create new HDU
    hdulist = fits.HDUList()
    hdulist.append(fits.GroupsHDU())

   ##################### DATA TABLE #######################################
    # Data header 
    header = hdulist['PRIMARY'].header

    #mandatory
    header['OBJECT'] = src
    header['TELESCOP'] = 'ALMA' # !AC TODO Can we change this field?  
    header['INSTRUME'] = 'ALMA'
    header['BSCALE'] = 1.0
    header['BZERO'] = 1.0
    header['BUNIT'] = 'JY'
    header['EQUINOX'] = 'J2000'
    header['ALTRPIX'] = 1.e0 #??

    #optional
    header['OBSRA'] = ra * 180./12.
    header['OBSDEC'] = dec
    header['MJD'] = float(mjd)
    #header['VELREF'] = 3 #TODO ??
    #header['DATE-OBS'] = ??
    #header['DATE-MAP'] = ??

    ## DATA AXES ##
    header['NAXIS'] = 7
    header['NAXIS1'] = 0

    # real, imag, weight
    header['CTYPE2'] = 'COMPLEX'
    header['NAXIS2'] = 3
    header['CRVAL2'] = 1.e0
    header['CDELT2'] = 1.e0
    header['CRPIX2'] = 1.e0
    header['CROTA2'] = 0.e0
    # RR, LL, RL, LR
    header['CTYPE3'] = 'STOKES'
    header['NAXIS3'] = nstokes
    header['CRVAL3'] = -1.e0 #corresponds to RR LL RL LR
    header['CRDELT3'] = -1.e0
    header['CRPIX3'] = 1.e0
    header['CROTA3'] = 0.e0
    # frequencies
    header['CTYPE4'] = 'FREQ'
    header['NAXIS4'] = nchan
    header['CRPIX4'] = 1.e0
    header['CRVAL4'] = ch1_freq # is this the right ref freq? in Hz
    header['CDELT4'] = ch_bw
    header['CROTA4'] = 0.e0
    # frequencies
    header['CTYPE5'] = 'IF'
    header['NAXIS5'] = nwindow
    header['CRPIX5'] = 1.e0
    header['CRVAL5'] = 1.e0 
    header['CDELT5'] = 1.e0
    header['CROTA5'] = 0.e0
    # RA
    header['CTYPE6'] = 'RA'
    header['NAXIS6'] = 1.e0
    header['CRPIX6'] = 1.e0
    header['CRVAL6'] = header['OBSRA']
    header['CDELT6'] = 1.e0
    header['CROTA6'] = 0.e0
    # DEC
    header['CTYPE7'] = 'DEC'
    header['NAXIS7'] = 1.e0
    header['CRPIX7'] = 1.e0
    header['CRVAL7'] = header['OBSDEC']
    header['CDELT7'] = 1.e0
    header['CROTA7'] = 0.e0
    
    ##RANDOM PARAMS##
    header['PTYPE1'] = 'UU---SIN'
    header['PSCAL1'] = 1/ref_freq
    header['PZERO1'] = 0.e0
    header['PTYPE2'] = 'VV---SIN'
    header['PSCAL2'] = 1.e0/ref_freq
    header['PZERO2'] = 0.e0
    header['PTYPE3'] = 'WW---SIN'
    header['PSCAL3'] = 1.e0/ref_freq
    header['PZERO3'] = 0.e0
    header['PTYPE4'] = 'BASELINE'
    header['PSCAL4'] = 1.e0
    header['PZERO4'] = 0.e0
    header['PTYPE5'] = 'DATE'
    header['PSCAL5'] = 1.e0
    header['PZERO5'] = 0.e0
    header['PTYPE6'] = 'DATE'
    header['PSCAL6'] = 1.e0
    header['PZERO6'] = 0.0
    header['PTYPE7'] = 'INTTIM'
    header['PSCAL7'] = 1.e0
    header['PZERO7'] = 0.e0
             
    # Save data
    pars = ['UU---SIN', 'VV---SIN', 'WW---SIN', 'BASELINE', 'DATE', 'DATE', 'INTTIM']
    x = fits.GroupData(outdat, parnames=pars, pardata=[u, v, np.zeros(ndat), np.array(bls).reshape(-1), jds_only, fractimes, tints], bitpix=-32)
 
    hdulist['PRIMARY'].data = x
    hdulist['PRIMARY'].header = header

    ####################### AIPS AN TABLE ########################
    #Antenna Header params - TODO do we need to change more of these?? 
    col1 = fits.Column(name='ANNAME', format='8A', array=antnames)
    col2 = fits.Column(name='STABXYZ', format='3D', unit='METERS', array=xyz)
    col3= fits.Column(name='ORBPARM', format='0D', array=np.zeros(0))
    col4 = fits.Column(name='NOSTA', format='1J', array=antnums)
 
    #TODO get the actual information for these parameters
    col5 = fits.Column(name='MNTSTA', format='1J', array=np.zeros(nsta)) #zero = alt-az
    col6 = fits.Column(name='STAXOF', format='1E', unit='METERS', array=np.zeros(nsta)) #zero = no axis  offset
    col7 = fits.Column(name='POLTYA', format='1A', array=np.array(['R' for i in range(nsta)], dtype='|S1')) #RCP 
    col8 = fits.Column(name='POLAA', format='1E', unit='DEGREES', array=np.zeros(nsta)) #feed orientation A
    col9 = fits.Column(name='POLCALA', format='2E', array=np.zeros((nsta,2))) #zero = no pol cal info TODO should have extra dim for nif
    col10 = fits.Column(name='POLTYB', format='1A', array=np.array(['L' for i in range(nsta)], dtype='|S1')) #LCP
    col11 = fits.Column(name='POLAB', format='1E', unit='DEGREES', array=90*np.ones(nsta)) #feed orientation A
    col12 = fits.Column(name='POLCALB', format='2E', array=np.zeros((nsta,2))) #zero = no pol cal info

    tbhdu = fits.BinTableHDU.from_columns(fits.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12]), name='AIPS AN')
    hdulist.append(tbhdu)

    head = hdulist['AIPS AN'].header
    head['EXTVER'] = 1
    head['ARRAYX'] = 0.e0
    head['ARRAYY'] = 0.e0
    head['ARRAYZ'] = 0.e0

    head['RDATE'] = '2000-01-01T00:00:00.0'
    head['GSTIA0'] = 114.38389781355 # GST in degrees for refdate TODO for jan 1 2000
    head['DEGPDY'] = 360.9856 # TODO for jan 1 2000
    head['FREQ']= ref_freq
    head['POLARX'] = 0.e0
    head['POLARY'] = 0.e0

    head['UT1UTC'] = 0.e0   #difference between UT1 and UTC ?
    head['DATUTC'] = 0.e0
    head['TIMESYS'] = 'UTC'

    head['ARRNAM'] = 'ALMA'  # TODO must be recognized by aips/casa
    head['XYZHAND'] = 'RIGHT'
    head['FRAME'] = '????'
    head['NUMORB'] = 0
    head['NO_IF'] = nwindow
    head['NOPCAL'] = 0            #TODO add pol cal information
    head['POLTYPE'] = 'VLBI'
    head['FREQID'] = 1
    
    hdulist['AIPS AN'].header = head # TODO necessary, or is it a pointer?

    ##################### AIPS FQ TABLE #######################################
    # Convert types & columns
    freqid = np.array([1])                                                    #frequency setup
    bandfreq = np.array([0.0 for i in range(nwindow)]).reshape([1,nwindow])   #frequency offset
    chwidth = np.array([ch_bw for i in range(nwindow)]).reshape([1,nwindow])  #ch bw
    totbw = np.array([bw for i in range(nwindow)]).reshape([1,nwindow])       #total bw
    sideband = np.array([1 for i in range(nwindow)]).reshape([1,nwindow])     #sideband (>1 = upper)

    freqid = fits.Column(name="FRQSEL", format="1J", array=freqid)
    bandfreq = fits.Column(name="IF FREQ", format="%dD"%(nwindow), array=bandfreq)
    chwidth = fits.Column(name="CH WIDTH",format="%dE"%(nwindow), array=chwidth)
    totbw = fits.Column(name="TOTAL BANDWIDTH",format="%dE"%(nwindow),array=totbw)
    sideband = fits.Column(name="SIDEBAND",format="%dJ"%(nwindow),array=sideband)
    cols = fits.ColDefs([freqid, bandfreq, chwidth, totbw, sideband])
     
    # create table
    tbhdu = fits.BinTableHDU.from_columns(cols)
 
    # add header information
    tbhdu.header.append(("NO_IF", nwindow, "Number IFs"))
    tbhdu.header.append(("EXTNAME","AIPS FQ"))

    hdulist.append(tbhdu) #TODO no AIPS FQ in template currently


    # Write final HDUList to file
    hdulist.writeto(fname, clobber=True)



#######################################################################
##########################  Main FUNCTION #############################
####################################################################### 
def main(datadir=DATADIR_DEFAULT, recompute_bl_fits=True, clean_bl_fits=False, rot_rate=False, rot_delay=False):
    
    print "****************HOPS2UVFITS*******************"
    print "Creating merged single-source uvfits files for hops fringe files in directory: ", datadir
    print

    scandirs = [os.path.join(datadir,o) for o in os.listdir(datadir) if os.path.isdir(os.path.join(datadir,o))]
    

    scan_fitsFiles = []
    scan_sources = []
    
    i = 1
    N = len(scandirs)
    for scandir in sorted(scandirs):
        
        scandir = scandir + '/'
        
        if recompute_bl_fits:
            # clean up the files in case there were extra ones already there that we no longer want
            if clean_bl_fits:
                print 'REMOVING the old uvfits baseline files due to --clean flag'
                for filename in glob.glob(scandir + '*_hops_bl.uvfits'):
                    os.remove(filename)
            # convert the finge files to baseline uv files  
            print "scan directory %i/%i: %s" % (i,N, scandir)
            convert_bl_fringefiles(datadir=scandir, rot_rate=rot_rate, rot_delay=rot_delay)
        
        #print "Merging baseline uvfits files in directory: ", scandir
        print 'WARNING - U,V coordinate units unknown!' 
        
        bl_fitsFiles = []
        for filename in glob.glob(scandir + '*_hops_bl.uvfits'):
            bl_fitsFiles.append(filename)
        if not len(bl_fitsFiles):
            raise Exception("cannot find any fits files with extension _hops_bl.uvfits in %s" % scandir)


        (obs_info, antenna_info, rg_params, outdat) = merge_hops_uvfits(bl_fitsFiles)
        outname = scandir + "scan_hops_merged.uvfits"
        save_uvfits(obs_info, antenna_info, rg_params, outdat, outname)
        
        scan_fitsFiles.append(outname)
        scan_sources.append(obs_info[0])
        i += 1

        print "Saved scan merged data to ", outname
        print
        
    print scan_sources
    print
    unique_sources = set(scan_sources)
    scan_fitsFiles = np.array(scan_fitsFiles)
    scan_sources = np.array(scan_sources)
    for source in unique_sources:
        print "Merging all scan uvfits files in directory: ", datadir, "for source: ", source
        print 'WARNING - U,V coordinate units unknown!'
        source_scan_fitsFiles = scan_fitsFiles[scan_sources==source]
        (obs_info, antenna_info, rg_params, outdat) = merge_hops_uvfits(source_scan_fitsFiles)
        outname = datadir + source + "_full_hops_merged.uvfits"
        save_uvfits(obs_info, antenna_info, rg_params, outdat, outname)
        print "Saved full merged data to ", outname

    return
 

if __name__=='__main__':
    if len(sys.argv) == 1: 
        datadir = DATADIR_DEFAULT
    else: datadir = sys.argv[-1]
    if datadir[0] == '-': datadir=DATADIR_DEFAULT

    recompute_bl_fits = True
    if "--skip_bl" in sys.argv: recompute_bl_fits = False

    clean_bl_fits = False
    if "--clean" in sys.argv: clean_bl_fits = True
    
    rot_rate = False
    if "--rot_rate" in sys.argv: rot_rate = True
        
    rot_delay = False
    if "--rot_delay" in sys.argv: rot_delay = True

    main(datadir=datadir, recompute_bl_fits=recompute_bl_fits, clean_bl_fits=clean_bl_fits, rot_rate=rot_rate, rot_delay=rot_delay)
