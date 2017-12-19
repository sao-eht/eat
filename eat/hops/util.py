"""
HOPS utilities
"""

#2016-10-31 Lindy Blackburn

from __future__ import division
from __future__ import print_function

# from builtins import next
# from builtins import str
# str = type('') # python 2-3 compatibility issues
# from builtins import zip
# from builtins import range
# from builtins import object

from ..io import util
import numpy as np
from numpy.fft import fft2, fftfreq, fftshift # for fringe fitting
try:
    import mk4 # part of recent HOPS install, need HOPS ENV variables
    dfio = mk4.mk4io_load()
except:
    import warnings
    warnings.warn("cannot import mk4 (did you run hops.bash?), mk4 file access will not work")
    mk4 = None
    dfio = None
import datetime
import ctypes
from argparse import Namespace
import itertools
from collections import OrderedDict
import matplotlib.pyplot as plt
from ..plots import util as putil
from matplotlib.offsetbox import AnchoredText
import glob
import re
import os

# convenient reduces columns to print
showcol_v5 = "datetime timetag scan_id source baseline band polarization amp snr phase_deg delay_rate".split()
# parity columns which should be flipped if baseline is flipped
flipcol_v5 = "phase_deg sbdelay mbdelay delay_rate u v ecphase delay_rate total_phas total_rate total_mbdelay total_sbresid".split()

reversecol = "baseline polarization".split()
flipcol = "resid_phas sbdelay mbdelay delay_rate u v total_phas total_rate total_mbdelay total_sbresid resid_delay mbd_unwrap".split()

# flip in place rows of dataframe (baseline parity) depending on index labels or boolean index flipidx
def flip(df, flipidx):
    # handle string reversal
    for col in reversecol:
        if col in df.columns:
            df.loc[flipidx,col] = df.loc[flipidx,col].str[::-1]
    # handle sign flip (phases, delays, rates)
    for col in flipcol:
        if col in df.columns:
            df.loc[flipidx,col] = -df.loc[flipidx,col]
    # handle swap columns (ref with rem)
    for col in df.columns:
        if col[:3] == 'ref':
            remcol = col.replace('ref', 'rem', 1)
            if remcol in df.columns:
                df.loc[flipidx,[col,remcol]] = df.loc[flipidx,[remcol,col]].values

sites = """
A ALMA
X APEX
L LMT
S SMAP
R SMAR
Z SMT
P Pico
"""
sdict = dict((line.strip().split() for line in sites.strip().split('\n')))

# from fourfit code lex.c
lex = ''.join(['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p',
     'q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F',
     'G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V',
     'W','X','Y','Z','0','1','2','3','4','5','6','7','8','9','$','%'])

# systematics parameters
sys_fac = 1.0 # systematic factor on thermal delay error
sys_par = 2e-6 # 2 ps on fringe delay
sys_cross = 20e-6 # 20 ps on cross hand delay

# restart of backend system
restarts_2017_low = {'X':map(util.tt2dt, ['101-010200'])}

def getpolarization(f):
    b = mk4.mk4fringe(f)
    ch0 = b.t203[0].channels[b.t205.contents.ffit_chan[0].channels[0]]
    return ch0.refpol + ch0.rempol

# return mk4fringe based on object, filename, or glob path
# if glob return the file with latest getmtime time
# maybe this should get latest HOPS rootcode instead..
# remember file path to guess working directory for future calls
# filelist=True will return a list of all files found
# quiet: if True do not echo filename
def getfringefile(b=None, filelist=False, pol=None, quiet=False):
    if b is None and hasattr(getfringefile, 'last'): # try to run with the last
        return getfringefile('/'.join(getfringefile.last), filelist=filelist, pol=pol, quiet=quiet)
    if hasattr(b, '__getitem__'):
        if type(b) is str:
            files = glob.glob(b)
            if len(files) == 0: # try harder to find file
                tok = b.split('/')
                last = getattr(getfringefile, 'last', [])
                if len(tok) < len(last):
                    files = glob.glob('/'.join(last[:-len(tok)] + tok))
        else:
            files = list(itertools.chain(*(getfringefile(bi, filelist=True, quiet=quiet) for bi in b)))
        if(len(files) == 0 and not quiet):
            raise(Exception("cannot find file: %s or %s" % (b, '/'.join(last[:-len(tok)] + tok))))
        files = [f for f in files if '..' not in f] # filter out correlator files
        if pol is not None: # filter by polarization
            files = [f for f in files if getpolarization(f) in set([pol] if type(pol) is str else pol)]
        if(len(files) == 0 and not quiet):
            raise(Exception("cannot find file with polarization " + pol))
        if filelist:
            return sorted(files) if len(files) > 0 else []
        files.sort(key=os.path.getmtime)
        getfringefile.last = files[-1].split('/')
        if not quiet:
            print(files[-1])
        b = mk4.mk4fringe(files[-1]) # use last updated file
    return b

# convenience function to set "datadir" (last file) for getfringefile
# can optionally pass expt_no and scan_id to set subdirectories
def set_datadir(datadir, expt_no='expt_no', scan_id='scan_id'):
    getfringefile.last = datadir.rstrip('/').split('/') + [str(expt_no), str(scan_id), 'baseline.freq_code.extent_no.root_id']

# unwrap short to positive int in multiples from 1e6 to 1024e6
def short2int(short):
    if short in short2int.lookup:
        return short2int.lookup[short]
    else:
        return short2int.lookup_minus_1[short]

if mk4 is not None:
	stype = dict(mk4.ch_struct._fields_)['sample_rate'] # was changed from short to ushort around Feb 2017
	short2int.lookup = {stype(i*1000000).value:i*1000000 for i in range(1024)}
	# look out for float precision error in HOPS vex parser.. appears to happen for 116.0 Ms/s
	short2int.lookup_minus_1 = {stype(i*1000000-1).value:i*1000000 for i in range(1024)}
	short2int.lookup[21632] = 117187500 # special values for ALMA full band 58.593750 MHz ("117.2" Ms/s)
	short2int.lookup[21631] = 117187500 # make sure -1 rounding error case takes priority as well
	short2int.lookup[9132] = 117187500 # full precision in OVEX
	short2int.lookup[9131] = 117187500 # -1 rounding case

def mk4time(time):
    return datetime.datetime.strptime("%d-%03d %02d:%02d:%02d.%06d" %
        (time.year, time.day, time.hour, time.minute, int(time.second), int(0.5+1e6*(time.second-int(time.second)))),
        "%Y-%j %H:%M:%S.%f")

# populate the type_212 visib data into array
# (nap, nchan)
def pop212(b=None, pol=None, weights=False):
    b = getfringefile(b, pol=pol)
    (nchan, nap) = (b.n212, b.t212[0].contents.nap)
    data212 = np.zeros((nchan, nap, 3), dtype=np.float32)
    for i in range(nchan):
        q = (mk4.newphasor*nap).from_address(ctypes.addressof(b.t212[i].contents.data))
        data212[i] = np.frombuffer(q, dtype=np.float32, count=-1).reshape((nap, 3))
    v = data212[:,:,0] * np.exp(1j * data212[:,:,1])
    if weights:
        return v.T, data[:,:,2].T
    else:
        return v.T

# populate the type_230 visib data into array automatically detect sideband
# (nchan, nap, nspec)
def pop230(b=None, pol=None):
    b = getfringefile(b, pol=pol)
    (nchan, nap, nspec) = (b.n212, b.t212[0].contents.nap, b.t230[0].contents.nspec_pts)
    data230 = np.zeros((nchan, nap, nspec//2), dtype=np.complex128)
    for i in range(nchan): # loop over HOPS channels
        idx = b.t205.contents.ffit_chan[i].channels[0] # need to get index into mk4 chdefs
        istart = nspec//2 if (b.t203[0].channels[idx].refsb == 'U') else 0 # USB vs LSB fixed offset
        for j in range(nap):
            # get a complete spectrum block for 1 AP at a time
            q = (mk4.complex_struct*nspec).from_address(
                ctypes.addressof(b.t230[j+i*nap].contents.xpower))
            # type230 frequeny order appears to be [---LSB--> LO ---USB-->]
            data230[i,j,:] = np.frombuffer(q, dtype=np.complex128, count=-1)[istart:istart+nspec//2]
    return data230

# populate type_120 visib data into array -- use FRINGE file, do NOT give it COREL file
# because the FRINGE file will determine parameters incl polarization, but data will come from COREL file
# if you have not run fourfit then *this will not work*
# output data will match fourfit CHANNELS and will contain all DiFX processed AP's (no fourfit time cuts)
# we don't bother flipping LSB because convention is unknown, and recent data should be USB (zoom-band)
# fill value will fill visibs with value for missing data (Null pointer)
def pop120(b=None, pol=None, fill=0):
    if type(b) is str and b[-8:-6] == "..":
        raise Exception("please pass a FRINGE file not a COREL file to this function, as the COREL file will be read automatically")
    b = getfringefile(b, pol=pol) # fringe file
    ctok = getfringefile.last[-1].split('.')
    c = mk4.mk4corel('/'.join(getfringefile.last[:-1] + [ctok[0] + '..' + ctok[-1]])) # corel file
    # use fringe file to get ap length, note that nap in fringe file is not necessarily same as corel
    ap = (mk4time(b.t205.contents.stop) - mk4time(b.t205.contents.start)).total_seconds() / b.t212[0].contents.nap
    T = (mk4time(c.t100.contents.stop) - mk4time(c.t100.contents.start)).total_seconds()
    # some slob just in case roundoff error from non-integer ap, ap_space from HOPS is too big
    (nchan, nap, nspec) = (b.n212, int(1e-6 + T / ap), c.t100.contents.nlags)
    data120 = np.zeros((nchan, nap, nspec), dtype=np.complex64)
    # require spectral type (DiFX)
    firstap = next(a.contents.ap for a in c.index[0].t120[0:c.index[0].ap_space] if a)
    if c.index[0].t120[firstap].contents.type != '\x05':
        raise Exception("only supports SPECTRAL type from DiFX->Mark4")
    # 120: (ap, channel, spectrum), this is mk4 channels (31, 41, ..) not HOPS channels (A, B, ..)
    for i in range(nchan): # loop over HOPS channels
        # by construction the ordering of type_101 and type_203 is the same (fill_203.c)
        # so we can avoid using the mk4 channel pair index to bookkeep
        idx = b.t205.contents.ffit_chan[i].channels[0] # need to get index into mk4 chdefs
        for j in range(nap):
            # get a complete spectrum block for 1 AP at a time
            # HOPS corel struct allocates empty pointers, then fills them according to correlator AP
            # the first correlator AP may be > 0, this non-zero AP corresponds to HOPS AP "0"
            specptr = c.index[idx].t120[j]
            if(j < c.index[idx].ap_space and specptr):
                q = (mk4.spectral*nspec).from_address(
                    ctypes.addressof(c.index[idx].t120[j].contents.ld))
                # type230 frequeny order appears to be [---LSB--> LO ---USB-->]
                data120[i,j,:] = np.frombuffer(q, dtype=np.complex64, count=-1)
            else:
                data120[i,j,:] = fill
    # clear memory (until supported in mk4.py library)
    dfio.clear_mk4corel(ctypes.byref(c))
    return data120

    return Namespace(name=name, ref_freq=ref_freq, nchan=nchan, nap=nap, nspec=nspec, nlags=nlags,
        code=clabel, pol=cinfo[0].refpol + cinfo[0].rempol, sbd=sbd, mbd=mbd, delay=delay, rate=rate, amplitude=amplitude, snr=snr, T=T,
        ap=ap, dtvec=dtvec, trot=trot, fedge=fedge, bw=bw, foffset=foffset, dfvec=dfvec, frot=frot,
        baseline=b.t202.contents.baseline, source=b.t201.contents.source, start=start, stop=stop, utc_central=utc_central,
        scan_name=b.t200.contents.scan_name, scantime=mk4time(b.t200.contents.scantime))

# some HOPS channel parameter info
# same function as HOPS param_struct (unfortunately)
# frot and trot rotate opposite the detected fringe location
# i.e. they subtract the delay, rate under multiplication
def params(b=None, pol=None):
    """extract many basic parameters from fringe file for easy access

    The parameters attempt to follow metadata stored in fringe files.
    This function should be moved to a class with better structure and lazy evaluation.

    Args:
        b: filename, filename pattern, or mk4fringe structure
        pol: polarization to look for if filename pattern is given (see getfringe)

    Returns:
        name: fringe file path from datadir
        ref_freq: reference frequency in MHz
        nchan: number of channels (processed)
        nap: number of accumulation periods (processed)
        nspec: number of spectral points per channel in type_230 (usually double the original number)
        nlags: number of lags from type_202 (also probably double the original FX data)
        code: ffit_chan_id's for the channels in order (from type_205) a, b, c, ...
        expt_no: HOPS 4-digit experiment number
        pol: baseline polarization product label e.g. LL
        sbd: single band residual delay average [us]
        mbd: multi band residual delay fitted value [us]
        delay: unwrapped mbd [us]
        rate: residual delay rate [us/s]
        amplitude: correlation coefficient estimate [10^-4]
        snr: signal-to-noise of scan-average visibility amplitude
        T: processed time [s]
        ap: accumulation period spacing [s]
        days: days since Jan 1 00:00:00 UT of each processed AP center time
        dtvec: a time vector with zero at utc_central (time reference for scan)
        dfvec: a frequency vector by visib type with zero at ref_freq
        trot: time rotators to rotate out fringe solution
        fedge: frequency edge of each channel
        bw: bandwidth of all channels as derived as best as possible from sample frequency info
        foffset: offset of middle of channel from channel edge
        baseline: baseline code
        source: source name
        start: start datetime of processed segment
        stop: stop datetime of processed segment (marks end of last segment) 
        startidx: start idx into type120 of processed APs
        stopidx: stop idx into type120 of processed APs (not inclusive)
        apfilter: boolean indexer for type120 of processed time
        utc_centeral: from fringe file, probaby marks reference time for delay rate compensation
        scan_name: name of scan (various conventions)
        scantime: scantime, generally start time of scan
        timetag: scantime in timetag format
    """
            
    if type(b) is str:
        name = b
        b = getfringefile(b, pol=pol)
    else:
        name = b.id.contents.name
    ref_freq = b.t205.contents.ref_freq
    # dimensions -- nlags in fringe files, may be zero padded by 2x
    (nchan, nap, nlags) = (b.n212, b.t212[0].contents.nap, b.t202.contents.nlags)
    nspec = None if not bool(b.t230[0]) else b.t230[0].contents.nspec_pts
    # channel indexing
    clabel = [q.ffit_chan_id for q in b.t205.contents.ffit_chan[:nchan]]
    cidx = [q.channels[0] for q in b.t205.contents.ffit_chan[:nchan]]
    cinfo = [b.t203[0].channels[i] for i in cidx] # channel info
    # fourfit delay and rate solution
    sbd = b.t208.contents.resid_sbd
    mbd = b.t208.contents.resid_mbd
    amb = b.t208.contents.ambiguity
    offset = (sbd - mbd + 1.5*amb) % amb
    delay = (sbd - offset + 0.5*amb) # unwrap to be close to SBD, us
    rate = b.t208.contents.resid_rate # us/s
    snr = b.t208.contents.snr
    amplitude = b.t208.contents.amplitude
    # time vector and rotator
    (start, stop) = (mk4time(b.t205.contents.start), mk4time(b.t205.contents.stop)) # t212 bounds
    utc_central = mk4time(b.t205.contents.utc_central) # depends only on scan boundary, not cuts
    T = (stop-start).total_seconds()
    # ref_time = mk4time(b.t205.contents.start) + T/2. # place reference time in middle
    ap = T / nap
    days0 = (start - datetime.datetime(start.year, 1, 1)).total_seconds() / 86400. # days since jan1
    days = days0 + (ap * np.arange(nap) + ap/2.)/86400. # days since jan1 of all AP center times
    scanlength = b.t200.contents.stop_offset - b.t200.contents.start_offset
    apfilter = np.zeros(int(1e-6 + scanlength/ap), dtype=bool)
    startidx = int(1e-6 + ((start - mk4time(b.t200.contents.scantime)).total_seconds()
                        - b.t200.contents.start_offset) / ap)
    stopidx = int(1e-6 + ((stop - mk4time(b.t200.contents.scantime)).total_seconds()
                       - b.t200.contents.start_offset) / ap)
    apfilter[startidx:stopidx] = True
    dtvec = ap*np.arange(nap) - (utc_central-start).total_seconds() + 0.5*ap
    trot = np.exp(-1j * rate * dtvec * 2*np.pi*ref_freq) # reverse rotation due to rate to first order
    # frequency matrix (channel, spectrum) and rotator
    fedge = np.array([1e-6 * ch.ref_freq for ch in cinfo])
    flip = np.array([-1 if ch.refsb == 'L' else 1 for ch in cinfo])
    bw = np.array([0.5e-6 * short2int(ch.sample_rate) for ch in cinfo])
    (foffset, dfvec, frot) = (dict(), dict(), dict())
    nlags120 = nlags//2 # guess
    foffset[230] = np.array([(f*np.arange(0.5, nlags)*bwn/nlags)[::f] for (f, bwn) in zip(flip, bw)])
    foffset[120] = np.array([(f*np.arange(0.5, nlags120)*bwn/nlags120)[::f] for (f, bwn) in zip(flip, bw)])
    dfvec[230] = (fedge[:,None] + foffset[230]) - ref_freq
    dfvec[120] = (fedge[:,None] + foffset[120]) - ref_freq
    dfvec[212] = fedge + (flip*bw)/2. - ref_freq
    frot[230] = np.exp(-1j * delay * dfvec[230] * 2*np.pi)
    frot[120] = np.exp(-1j * delay * dfvec[120] * 2*np.pi) # assuming nlags120 = nlags230/2
    frot[212] = np.exp(-1j * delay * dfvec[212] * 2*np.pi) # note type_212 is already rotated in data
    return Namespace(name=name, ref_freq=ref_freq, nchan=nchan, nap=nap, nspec=nspec, nlags=nlags, days=days,
        code=clabel, pol=cinfo[0].refpol + cinfo[0].rempol, sbd=sbd, mbd=mbd, delay=delay, rate=rate, amplitude=amplitude, snr=snr, T=T,
        ap=ap, dtvec=dtvec, trot=trot, fedge=fedge, bw=bw, foffset=foffset, dfvec=dfvec, frot=frot,
        baseline=b.t202.contents.baseline, source=b.t201.contents.source, start=start, stop=stop, utc_central=utc_central,
        scan_name=b.t200.contents.scan_name, scantime=mk4time(b.t200.contents.scantime),
        timetag=util.dt2tt(mk4time(b.t200.contents.scantime)), expt_no=b.t200.contents.expt_no, startidx=startidx, stopidx=stopidx,
        apfilter=apfilter)

# some unstructured channel info for quick printing
def chaninfo(b=None):
    b = getfringefile(b)
    nchan = b.n212
    # putting them in "fourfit" order also puts them in frequency order
    idx = [(q.ffit_chan_id, q.channels[0]) for q in b.t205.contents.ffit_chan[:nchan]] # MAX #64 for ffit_chan
    chinfo = [(hops_id, q.index, q.ref_chan_id, q.rem_chan_id, round(q.ref_freq/1e5)/10., round(q.rem_freq/1e5)/10.,
              round(10.*(q.ref_freq/1e6 - b.t205.contents.ref_freq))/10.,
              q.refsb+q.remsb, short2int(q.sample_rate)/1e6, q.refpol+q.rempol)
              for (hops_id, q) in [(hops_id, b.t203[0].channels[i]) for (hops_id, i) in idx]]
    return chinfo

# helper functions
def nextpow2(x): # next power of 2 for efficient fft
    return np.power(2, int(np.ceil(np.log2(x))))
def expmean(x, s=8, n=4): # robust mean of exponential distribution
    fac = 1.0
    m = np.mean(x)
    for i in range(n):
        fac = 1.0 - (s*np.exp(-s)/(1.-np.exp(-s)))
        m = np.mean(x[x<s*m]) / fac
    return m

# kind: 212 or 230: use type_212 or type_230 data
# res: zero-padding factor for FFT
# showx, showy: how many fringe FWHM to plot
# center: delay,rate center of plot (default: on max), 'hops':use hops value
# dt, df: decimation factors in time, channels
# ni: number of incoherent averages (1=scan average)
# ret: return the full FFT power matrix & other info if true and do not plot anything
# segment: (start_ap, stop_ap) over which to search, slice-like syntax: e.g. (10,-10)
# channels: (start_ch, stop_ch) over which to search, slice-like syntax: e.g. (0, None)
# unrotate_212: unrotate the fourfit soln from the 212 data before fringe search
# delay_off, rate_off: subtract this from the data before doing search
# manual offsets will show up in axis labels, automatic offsets (from centering) will not
# replacedata: new visibility array to substitute with actual data before fringe fitting
def findfringe(fringefile=None, kind=None, res=4, showx=6, showy=6, center=(None, None),
               dt=2, df=None, ni=1, ret=False, showhops=False,
               delay_off=0., rate_off=0., flip=False, segment=(None, None), channels=(None,None),
               pol=None, unrotate_212=True, replacedata=None):
    b = getfringefile(fringefile, pol=pol)
    p = params(b)
    (nchan, nap) = (b.n212, b.t212[0].contents.nap)
    if kind is None:
        kind = 230 if bool(b.t230[0]) else 212 # use type_230 if available
    if kind==212:
        nspec = 1
        df = df or 1
        v = (replacedata if replacedata is not None else pop212(b))[:,:,None]
    elif kind==230:
        nspec = b.t230[0].contents.nspec_pts // 2 # one sideband, assume all channels are same
        df = df or 4 # speed-up if using full spectral resolution
        v = np.swapaxes(replacedata if replacedata is not None else pop230(b), 1, 0)  # put AP as axis 0
        assert(v.shape == (nap, nchan, nspec))   # make sure loaded data has right dimensions
        if flip:
            v = v[:,:,::-1] # test flip frequency order of spectral points
    elif kind==120: # original correlator output
        v = np.swapaxes(replacedata if replacedata is not None else pop120(b), 1, 0)  # put AP as axis 0
        df = df or 2 # arbitrary, but compensate for type_230 inflation factor of x2 (SSB)
        nspec = v.shape[-1]
        assert(v.shape == (nap, nchan, nspec))
        if flip: # fake support for LSB?
            v = v[:,:,::-1] # test flip frequency order of spectral points

    # apply fringe rotations
    if(center=='hops'):
        center = (p.delay*1e3, p.rate*1e6)
    if(kind==212 and unrotate_212):
        delay_off -= p.delay*1e3 # ns
        rate_off -= p.rate*1e6   # ps/s
    # note this will affect the data, and must also be reflected in the delay, rate vectors
    if center[0] is not None:
        delay_off += center[0]
    if center[1] is not None:
        rate_off += center[1]
    print("rotation subtracted from data: %.3f [ns], %.3f [ps/s]" % (delay_off, rate_off))
    frot = np.exp(-1j * 1e-3*delay_off * p.dfvec[kind].reshape((nchan, -1)) * 2*np.pi)
    trot = np.exp(-1j * 1e-6*rate_off * p.dtvec * 2*np.pi*p.ref_freq)
    v = v * trot[:,None,None] * frot[None,:,:]

    v = v[slice(*segment)] # apply time segment cut
    v = v[:,slice(*channels),:] # apply channel cut
    (nap, nchan, nspec) = v.shape  # dimensions of data
    clip = np.fmod(nap, dt*ni) # fit ni non-overlapping time segments after decimation
    if clip > 0: # remove small amount of end data for equal segments
        nap = nap-clip
        v = v[:nap]

    # block averaging factors to speedup, make sure no phase wrap!
    v = v.reshape((ni, nap//dt//ni, dt, nchan*nspec//df, df))
    v = v.sum(axis=(2, 4)) # stack on time, and frequency decimation factors

    # the normalized complex visibility and FFT search delay/rate
    (zpap, zpch) = (nextpow2(res*v.shape[1]), nextpow2(res*v.shape[2])) # zero padding for time and freq
    fringevis = fftshift(fft2(v, s=(zpap, zpch)))
    fqap = fftshift(fftfreq(zpap)) # "frequency" range of the rate space [cycles/sample_spacing]
    fqch = fftshift(fftfreq(zpch)) # "frequency" range of the delay space [cycles/sample_spacing]

    # single-channel spacing [MHz] and decimated spectral point spacing [MHz]
    spacings = set(np.diff(sorted(p.fedge)))
    if len(spacings) > 1:
        raise Exception("channel spacing is discontinuous")
    sb_spacing = spacings.pop()
    spec_spacing = df * sb_spacing / nspec
    # accumulation period [s]
    ap = dt * (mk4time(b.t205.contents.stop) - mk4time(b.t205.contents.start)).total_seconds() / (nap + clip)
    delay = (center[0] if center[0] else 0.) + 1e9 * fqch / (spec_spacing * 1e6) # ns
    rate = (center[1] if center[1] else 0.) + 1e12 * fqap / ap / (p.ref_freq * 1e6) # in ps/s
    dd = delay[1] - delay[0]
    dr = rate[1] - rate[0]

    (left, right, bottom, top) = (delay[0]-dd/2., delay[-1]+dd/2., rate[0]-dr/2., rate[-1]+dr/2.)
    # set the plot aspect relative to nyquist (propotional to fringe FWHM)
    BW = sb_spacing * nchan
    T = ap * v.shape[1]
    fwhm_delay = 1e3 / BW # ns
    fwhm_rate = 1e6 / T / p.ref_freq # ps/s
    aspect = abs(fwhm_delay / fwhm_rate)
    fringepow = np.abs(fringevis)**2 # fringe power before incoherent averaging
    fringepow = fringepow / (0.5 * expmean(fringepow.ravel())) # normalize to snr=1 for noise
    fringepow = np.sum(fringepow, axis=0) # the incoherent average of fringe power

    ns = Namespace(fringepow=fringepow, fringevis=fringevis, BW=BW, T=T, fwhm_delay=fwhm_delay, fwhm_rate=fwhm_rate,
            delay=delay, rate=rate, dd=dd, dr=dr, dt=dt, df=df, ni=ni,
            extent=(left, right, bottom, top), aspect=aspect, params=p)
    if ret:
        return ns
    else:
        plotfringe(ns, showx=showx, showy=showy, center=center, showhops=showhops, kind=kind)

def plotfringe(ns, showx=6., showy=6., center=(None, None), showhops=False, kind=230):

    (fringepow, fwhm_delay, fwhm_rate, delay, rate, extent, aspect, p) = \
        (ns.fringepow, ns.fwhm_delay, ns.fwhm_rate, ns.delay, ns.rate, ns.extent, ns.aspect, ns.params)
    if kind == 212: # use wrapped values
        (hops_delay, hops_rate) = (p.mbd, p.rate)
    else: # use unwrapped values
        (hops_delay, hops_rate) = (p.delay, p.rate)
    if center == 'hops':
        center = (1e3*hops_delay, 1e6*hops_rate)
    (i,j) = np.unravel_index(np.argmax(fringepow), fringepow.shape)
    plot_center = (delay[j] if center[0] is None else center[0], rate[i] if center[1] is None else center[1])

    mask_delay = np.abs(delay - plot_center[0]) > showx*fwhm_delay
    mask_rate = np.abs(rate - plot_center[1]) > showy*fwhm_rate
    fringepow[mask_rate,:] = 0 # mask power outside region of interest
    fringepow[:,mask_delay] = 0
    print(np.max(fringepow))

    plt.imshow(fringepow, cmap='jet', origin='lower', extent=extent,
        aspect=aspect, interpolation='Nearest', vmin=0)
    plt.xlabel('delay [ns]')
    plt.ylabel('rate [ps/s]')
    plt.xlim(plot_center[0] + np.array((-1,1))*showx*fwhm_delay)
    plt.ylim(plot_center[1] + np.array((-1,1))*showy*fwhm_rate)

    # show locatino of fourfit fringe solution
    if showhops:
        plt.plot(1e3*hops_delay, 1e6*hops_rate, 'kx', ms=24, mew=10)
        plt.plot(1e3*hops_delay, 1e6*hops_rate, 'wx', ms=20, mew=6)

    ratio = showy / showx
    plt.setp(plt.gcf(), figwidth=2.+3./np.sqrt(ratio), figheight=2.+3.*np.sqrt(ratio))
    plt.tight_layout()

    (i,j) = np.unravel_index(np.argmax(fringepow), fringepow.shape) # get new max location
    putil.tag('%s [%d]' % (p.scan_name, p.scantime.year), loc='upper left', framealpha=0.85)
    putil.tag('%s(%s) [%s]' % (p.baseline, p.pol, p.source), loc='upper right', framealpha=0.85)
    putil.tag('%.3f ns' % delay[j], loc='lower left', framealpha=0.85)
    putil.tag('%.3f ps/s' % rate[i], loc='lower right', framealpha=0.85)

# coherent or incoherent stacking of two bands fringe plot
# b1, b2: fringe files
# d1, d2: delay offsets to apply
# r1, r2: rate offsets to apply
# p1, p2: phase offsets to apply
# coherent: True (default) for coherent stacking, False for incoherent stacking
# **kwargs: extra arguments to findfringe (dt, df, ni, kind)
def stackfringe(b1, b2, d1=0., d2=0., r1=0., r2=0., p1=0., p2=0., coherent=True, **kwargs):
    # get the FFT of both baselines
    ret1 = findfringe(b1, delay_off=d1, rate_off=r1, ret=True, **kwargs)
    ret2 = findfringe(b2, delay_off=d2, rate_off=r2, ret=True, **kwargs)
    # rotate ret2 using the ddelay and dphase
    return (ret1, ret2)

# average over many files, please make sure frequency setup is the same
# delay, rate: [us, and us/s], if None use fourfit soln
# ap is messy to derive from fringe files (fourfit pulls it from ovex)
# df: decimation factor in frequency for better SNR
# df: decimation factor in time if timeseires==True
# centerphase: subtract out mean phase for fewer wraps
def spectrum(bs, ncol=4, delay=None, rate=None, df=1, dt=1, figsize=None, snrthr=0.,
             timeseries=False, centerphase=False, snrweight=True, kind=230, pol=None, grid=True):
    if type(bs) is str:
        bs = getfringefile(bs, filelist=True, pol=pol)
    if not hasattr(bs, '__len__'):
        bs = [bs,]
    if len(bs) > 1:
        centerphase = True
    vs = None
    for b in bs:
        b = getfringefile(b, pol=pol)
        if b.t208.contents.snr < snrthr:
            print("snr %.2f, skipping" % b.t208.contents.snr)
            continue
        if kind==230 and not bool(b.t230[0]):
            print("skipping no t230")
            continue
        if kind==230:
            v = pop230(b)   # visib array (nchan, nap, nspec/2)
        elif kind==120:
            v = pop120(b)   # visib array (nchan, nap, nspec/2)
        p = params(b) # channel and fringe parameters
        nrow = bool(timeseries) + np.int(np.ceil(p.nchan / ncol))
        delay = p.delay if delay is None else delay
        rate = p.rate if rate is None else rate
        trot = np.exp(-1j * rate * p.dtvec * 2*np.pi*p.ref_freq)
        frot = np.exp(-1j * delay * p.dfvec[kind] * 2*np.pi)
        vrot = v * trot[None,:,None] * frot[:,None,:]
        if centerphase: # rotate out the average phase over all channels
            crot = vrot.sum()
            crot = crot / np.abs(crot)
            vrot = vrot * crot.conj()
        if timeseries:
            vs = (0 if vs is None else vs) + vrot
        else:
            # stack in time (will work for different T) and add back axis
            vs = (0 if vs is None else vs) + vrot.sum(axis=1)[:,None]
    if vs is None: # no files read (snr too low)
        return
    for n in range(p.nchan):
        spec = vs[n].sum(axis=0) # sum over time
        spec = spec.reshape((-1, df)).sum(axis=1) # re-bin over frequencies
        ax1 = locals().get('ax1')
        ax1 = plt.subplot(nrow, ncol, 1+n, sharey=ax1, sharex=ax1)
        amp = np.abs(spec)
        phase = np.angle(spec)
        plt.plot(amp, 'b.-')
        plt.ylim(0, plt.ylim()[1])
        ax2 = plt.twinx()
        plt.plot(phase, 'r.-')
        plt.ylim(-np.pi, np.pi)
        ax2.set_yticklabels([])
        ax2.set_xticklabels([])
        putil.rmgaps(1.0, 2.0)
        if grid:
            plt.grid()
        ax2.add_artist(AnchoredText(p.code[n], loc=1, frameon=False, borderpad=0))
    plt.subplots_adjust(wspace=0, hspace=0)
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax1.set_xlim(-0.5, -0.5+len(spec))
    if timeseries:
        nt = len(p.dtvec)
        dt = min(dt, nt)
        nt = nt - np.fmod(nt, dt) # fit time segments after decimation
        v = vs[:,:nt,:] # clip to multiple of dt
        t = p.dtvec[:nt].reshape((-1, dt)).mean(axis=1)
        v = v.sum(axis=(0,2)).reshape((-1, dt)).sum(axis=1) # sum over channel and spectral points
        amp = np.abs(v)
        phase = np.angle(v)
        plt.subplot(nrow, 1, nrow)
        plt.plot(t, amp, 'b.-')
        plt.ylim(0, plt.ylim()[1])
        plt.gca().set_yticklabels([])
        plt.twinx()
        plt.plot(t, phase, 'r.-')
        plt.ylim(-np.pi, np.pi)
        plt.gca().set_yticklabels([])
        putil.rmgaps(1e6, 2.0)
        if grid:
            plt.grid()
        plt.xlim(-p.T/2., p.T/2.)
    if figsize is None:
        plt.setp(plt.gcf(), figwidth=8, figheight=8.*nrow/ncol)
    else:
        plt.setp(plt.gcf(), figwidth=figsize[0], figheight=figsize[1])

# rotate vs based on delay and rate and plot a 2D vector plot of complex visib
def vecplot(vs, dtvec, dfvec, delay, rate, ref_freq, dt=1, df=1):
    trot = np.exp(-1j * rate * dtvec * 2*np.pi*ref_freq)
    frot = np.exp(-1j * delay * dfvec * 2*np.pi)
    vrot = vs*trot[:,None]*frot[None,:]
    (nt, nf) = vrot.shape
    nt = nt - np.fmod(nt, dt) # fit time segments after decimation
    vrot = vrot[:nt,:]
    vrot = vrot.reshape((nt//dt, dt, nf//df, df))
    vrot = vrot.sum(axis=(1, 3)) # stack on time, and frequency decimation factors
    plt.plot([0,0], [vrot.re, vrot.im], 'b.-', alpha=0.25)
    vtot = np.sum(vrot) / len(vrot.ravel())
    plt.plot([0,0], [vtot.re, vtot.im], 'r.-', lw=2, ms=4, alpha=1.0)

def timeseries(bs, dt=1, pol=None):
    if not hasattr(bs, '__iter__'):
        bs = [bs,]
    nrow = len(bs)
    for (i, b) in enumerate(bs):
        b = getfringefile(b, pol=pol)
        p = params(b)
        plt.subplot(nrow, 1, 1+i)
        v = pop212(b).mean(axis=1) # stack over channels
        nt = len(v)
        dt = min(dt, nt)
        nt = nt - np.fmod(nt, dt) # fit time segments after decimation
        v = v[:nt].reshape((nt//dt, -1)).mean(axis=1) # clip to multiple of dt and stack
        t = p.dtvec[:nt].reshape((-1, dt)).mean(axis=1) + p.T/2.
        amp = np.abs(v)
        phase = np.angle(v)
        plt.plot(t, amp, 'b.-')
        plt.ylim(0, plt.ylim()[1])
        # plt.gca().set_yticklabels([])
        plt.ylabel('amp [1e-4]', color='blue')
        plt.twinx()
        plt.plot(t, phase, 'r.-')
        plt.ylim(-np.pi, np.pi)
        # plt.gca().set_yticklabels([])
        plt.ylabel('phase [rad]', color='red')
        putil.rmgaps(1e6, 2.0)
        plt.xlim(0, p.T)
        plt.xlabel('time [s]')
        plt.gca().add_artist(AnchoredText(p.baseline + ' (' + p.pol + ')', loc=1, frameon=False, borderpad=0))
        plt.gca().add_artist(AnchoredText(p.timetag + ' (' + p.source + ')', loc=2, frameon=False, borderpad=0))
    plt.setp(plt.gcf(), figwidth=8, figheight=2+nrow)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)

# calculate delay at each AP using type120 data
def delayscan(fringefile, res=4, dt=1, df=None, delayrange=(-1e4, 1e4), pol=None, fix_outliers=True, kind=120):
    b = getfringefile(fringefile, pol=pol)
    p = params(b)
    (nchan, nap) = (b.n212, b.t212[0].contents.nap)
    if kind==120:
        v = np.swapaxes(pop120(b), 1, 0)  # put AP as axis 0
    else:
        v = pop212(b)[:,:,None] # add extra axis for subchannel
    df = df or 1 # arbitrary, but compensate for type_230 inflation factor of x2 (SSB)
    nspec = v.shape[-1]
    assert(v.shape == (nap, nchan, nspec))
    clip = np.fmod(nap, dt) # fit ni non-overlapping time segments after decimation
    if clip > 0: # remove small amount of end data for equal segments
        nap = nap-clip
        v = v[:nap]

    # block averaging factors to speedup, make sure no phase wrap!
    v = v.reshape((nap//dt, dt, nchan*nspec//df, df))
    v = v.sum(axis=(1, 3)) # stack on time, and frequency decimation factors

    # the normalized complex visibility and FFT search delay/rate
    zpch = nextpow2(res*v.shape[1]) # zero padding for freq
    fringevis = np.fft.fft(v, n=zpch) # by default operate on axis=-1 (frequency axis)
    fqch = fftfreq(zpch) # "frequency" range of the delay space [cycles/sample_spacing]

    # single-channel spacing [Hz] and decimated spectral point spacing [MHz]
    sb_spacing = np.diff(sorted(b.t203.contents.channels[i].ref_freq for i in range(nchan)))[nchan//2]
    spec_spacing = df * 1e-6 * sb_spacing / nspec
    delay = 1e9 * fqch / (spec_spacing * 1e6) # ns
    dres = delay[1] - delay[0]
    print(dres)

    inside = ((delay >= delayrange[0]) & (delay <= delayrange[1]))
    imax = np.argmax(np.abs(fringevis[:,inside]), axis=-1) # the maximum frequency index
    delays = delay[inside][imax] # the solved delays

    # a little simple logic to clean up noise outliers
    if fix_outliers:
        good = np.ones(len(delays), dtype=np.bool)
        tol = np.isclose(delays[:-1], delays[1:], 0, 2*dres)
        good[1:-1] = (tol[:-1]) | (tol[1:])
        delays = np.where(good, delays, np.nan)

    return delays.ravel()

# This function is added by Kazu Akiyama
def compare_alist_v6(alist1,baseline1,polarization1,
                     alist2=None,baseline2=None,polarization2=None):
    '''Compare two alist data in pandas.DataFrame.

    This function pick up rows at common datetimes in two input alist data and
    at specified sets of the baseline(s), polarization(s), and concatenate two alist data
    into one pandas.DataFrame. The keys for the first and second alist data will be "xxx1"
    (e.g. "snr1", "amp1") and "xxx2" (e.g. "snr2", "amp2"), respectively.

    Args:
        alist1 (pandas.DataFrame):
            the first alist data read by eat.io.hope.read_alist_v6
        baseline1 (string):
            a baseline of the first alist data to be compared
        polarization1 (string):
             a polarization of the first alist data to be compared
        alist2 (optional, pandas.DataFrame):
            the second alist data in pandas.DataFrame read by eat.io.hope.read_alist_v6
            (default: same to alist1)
        baseline2 (optional, string):
            a baseline of the second alist data to be compared
            (default: same to baseline1)
        polarization2 (optional, string):
            a polarization of the second alist data to be compared
            (default: same to polarization1)

    Returns:
        A pandas.DataFrame object includes two alist data when the datetimes of two data sets
        at the specified baselines and polarizations are overlapped.
    '''
    import pandas as pd
    if alist2 is None:
        alist2 = alist1
    if baseline2 is None:
        baseline2 = baseline1
    if polarization2 is None:
        polarization2 = polarization1
    #
    # Get timestamps
    datetimes = sorted(list(set(alist1["datetime"].tolist()+alist2["datetime"].tolist())))
    #
    # Get alist keys
    alist1_keys = alist1.columns.tolist()
    alist1_keys.remove("datetime")
    alist2_keys = alist2.columns.tolist()
    alist2_keys.remove("datetime")
    #
    outdata = pd.DataFrame()
    for datetime in datetimes:
        alist1_idx = (alist1["datetime"]==datetime) * (alist1["baseline"]==baseline1) * (alist1["polarization"]==polarization1)
        alist1_tmp = alist1.loc[alist1_idx, :].reset_index(drop=True)
        alist2_idx = (alist2["datetime"]==datetime) * (alist2["baseline"]==baseline2) * (alist2["polarization"]==polarization2)
        alist2_tmp = alist2.loc[alist2_idx, :].reset_index(drop=True)
        if len(alist1_tmp["datetime"]) * len(alist2_tmp["datetime"]) == 0:
            continue
        #
        outdata_tmp = pd.DataFrame()
        outdata_tmp["datetime"] = [datetime]
        for key in alist1_keys:
            outdata_tmp[key+"1"] = [alist1_tmp.loc[0, key]]
        for key in alist2_keys:
            outdata_tmp[key+"2"] = [alist2_tmp.loc[0, key]]
        outdata = pd.concat([outdata,outdata_tmp], ignore_index=True)
    return outdata

def adhoc(b, pol=None, window_length=None, polyorder=None, snr=None, ref=0, prefix='', timeoffset=0., snrdof=10.,
          roundrobin=True, bowlfix=True, secondorder=True):
    """
    create ad-hoc phases from fringe file (type 212)
    assume a-priori phase bandpass and fringe rotation (delay) has been applied
    use round-robin training/evaluation to avoid self-tuning bias
    some SNR-based selection of averaging timescale is done to tune Sav-Gol filter
    compensate for delay-rate rotator bias for frequencies away from reference frequency (not done)
    check for -1 bad flag data in type212 and interpolate over (not done)

    Args:
        b: numpy array of shape (nap, nchan), of fringe filename, or mk4fringe object from which visibs will be read
           if fringe filename or object is sent, auxiliary information will be provided for control file
        pol: will filter for pol if multiple files or pattern given
        window_length: odd integer length of scipy.signal.savgol_filter applied to data for smoothing (default=7)
        polyorder: order of piecewise smoothing polynomial (default=3)
        snr: manually set SNR to auto determine window_length and polyorder, else take from fringe file
        ref: 0 (first site), 1 (second site), or station letter (e.g. A), for control file string (phases to apply to REM)
        prefix: add prefix to adhoc_filenames (e.g. source directory) as described in control file string
        timeoffset: add timeoffset [units of AP] to each timestamp in the adhoc string
        snrdof: target signal-to-noise per degree-of-freedom in adhoc solution (higher is smoother)
        roundrobin: True to use frequency-slicing round-robing training to avoid self-tuning
        bowlfix: fix bowl effect in 2017 data (must send fringe file for parameters)
        secondorder: fix the second-order effect from delay change over time

    Returns:
        v: visibility vector from which adhoc phase is estimated (not normalized)
        vcorr: corrected visibility (adhoc phase removed)
        phase: estimated baseline phase in degrees
        code: params code (params and control file statements returned if sent mk4 object)
        days: params days vector (note this does not account for fourfit adhoc indexing bug)
        scan_name: params scan_name
        scantime: params scantime
        scantimetag: time-tag of start time
        filename: filename of adhoc phase file
        cfcode: fourfit control file lines
        string: adhoc file contents as string
    """
    from scipy.signal import savgol_filter
    if type(b) is np.ndarray:
        v = b
    else:
        b = getfringefile(b, pol=pol, quiet=True)
        p = params(b)
        v = pop212(b)

    if bowlfix and type(b) is mk4.mk4_fringe:
        rfdict = {'A':-0.2156, 'X':0.163} # ps/s
        ratefix = rfdict.get(p.baseline[1], 0) - rfdict.get(p.baseline[0], 0)
        ratefix_phase = 2*np.pi * p.dfvec[212][None,:] * p.dtvec[:,None] * ratefix*1e-6
        v = v * np.exp(-1j * ratefix_phase) # take bowl effect out of visibs before adhoc phasing
    else:
        ratefix_phase = np.zeros_like(v, dtype=np.float)

    (nap, nchan) = v.shape
    vfull = v.sum(axis=1) # full frequency average
    vchop = np.zeros_like(v)
    phase = np.zeros_like(v, dtype=np.float)

    if type(b) is mk4.mk4_fringe:
        timeoffset = timeoffset * p.ap # use AP if we can
        if snr is None:
            snr = p.snr
        if ref == p.baseline[0]:
            parity = 1
        if ref == p.baseline[1]:
            parity = -1
    else:
        if snr is None:
            snr = 100.*np.sqrt(nap/300.) # some default
    if ref == 0:
        parity = 1
    if ref == 1:
        parity = -1
    # note that snr=10 per measurement is 36deg phase error
    # this should really be balanced against how rapidly the phase may vary between estimates
    nfit = max(1, int((snr / snrdof)**2)) # number of parameters we might be able to fit
    # qualitative behavior of fit depends primarily on window_length/polyorder which sets
    # timescale for free parameters. actual poly degree doesn't matter as much.
    # savgol constraints: polyorder < window_length, window_length is positive odd integer
    if polyorder is None:
        polyorder = min(nfit, 2)
    if window_length is None:
        window_length = 1+2*max(1, int(0.5 + float(nap * polyorder) / float(nfit) / 2.))
    if window_length > nap:
        window_length = 1+2*((nap-1)//2)
    if polyorder >= window_length:
        polyorder = max(0, window_length-1)

    if roundrobin: # apply round-robin training to avoid self-tuning
        for i in range(nchan): # i is the channel to exclude in fit
            # remove evaluation channel
            vtemp = vfull - v[:,i]
            try:
                re = savgol_filter(vtemp.real, window_length=window_length, polyorder=polyorder)
                im = savgol_filter(vtemp.imag, window_length=window_length, polyorder=polyorder)
            except:
                warnings.warn("failed to fit adhoc phases" + " to %s" % b.name if type(b) is mk4.mk4_fringe else '')
                re = np.ones_like(vtemp.real)
                im = np.zeros_like(vtemp.imag)
            vchop[:,i] = re + 1j*im
            phase[:,i] = np.unwrap(np.arctan2(im, re))
    else: # no round-robin, may self-tune but avoid bandpass systematics
        vtemp = vfull
        try:
            re = savgol_filter(vtemp.real, window_length=window_length, polyorder=polyorder)
            im = savgol_filter(vtemp.imag, window_length=window_length, polyorder=polyorder)
        except:
            warnings.warn("failed to fit adhoc phases" + " to %s" % b.name if type(b) is mk4.mk4_fringe else '')
            re = np.ones_like(vtemp.real)
            im = np.zeros_like(vtemp.imag)
        vchop[:,:] = re[:,None] + 1j*im[:,None]
        phase[:,:] = np.unwrap(np.arctan2(im[:,None], re[:,None]))

    # add estimated phase from small true change in delay over time
    if secondorder:
        if type(b) is mk4.mk4_fringe:
            phase += phase * p.dfvec[212] / p.ref_freq
        else: # guess fractional bandwidth
            df = (np.arange(nchan) * 58.59375)
            df -= np.mean(df)
            phase += phase * df / 228000.

    # note that ratefix is already taken out of v
    vcorr = v * np.exp(-1j * phase)

    # add estimated phase from bowl effect (unphysical delay drift)
    phase += ratefix_phase

    if type(b) is mk4.mk4_fringe:
        timetag = p.scantime.strftime("%j-%H%M%S")
        (ref, rem) = [p.baseline[0], p.baseline[1]][::parity]
        adhoc_filename = prefix + "adhoc_%s_%s.dat" % (rem, timetag)
        cf = """
if station %s and scan %s
    adhoc_phase file
    adhoc_file %s
    adhoc_file_chans %s
""" % (rem, timetag, adhoc_filename, ''.join(p.code))
        string = '\n'.join(("%.10f " % (timeoffset/86400. + day) + np.array_str(-ph, precision=3, max_line_width=1e6)[1:-1]
            for (day, ph) in zip(p.days, parity*phase*180/np.pi))) # note adhoc phase file needs sign flip, i.e. phase to be added to data
        return Namespace(code=p.code, days=p.days, phase=phase, v=vchop, vcorr=vcorr, scan_name=p.scan_name, scantime=p.scantime,
                         scantimetag=timetag, filename=adhoc_filename, cfcode=cf, string=string, ratefix_phase=ratefix_phase)
    else:
        return Namespace(v=vchop, phase=phase, vcorr=vcorr)

# helper class for embedded PDF in ipython notebook
class PDF(object):
    def __init__(self, pdfdata):
        self.pdfdata = pdfdata
    def _repr_pdf_(self):
        return self.pdfdata

# grab the ps from type221, convert to pdf if possible
# can chain multiple plots together with wildcard
def fplot(b=None, pol=None, filelist=True, noauto=False):
    import subprocess
    bs = getfringefile(b, pol=pol, filelist=filelist, quiet=True)
    if not hasattr(bs, '__len__'):
        bs = [bs,]
    pslist = []
    for b in bs:
        b = getfringefile(b, filelist=False, quiet=True)
        if(noauto and b.t202.contents.baseline[0] == b.t202.contents.baseline[1]):
            continue
        pslist.append(b.t221.contents.pplot)
    proc = subprocess.Popen("ps2pdf - -".split(), stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    # proc = subprocess.Popen("cat".split(), stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    out = PDF(proc.communicate(input=''.join(pslist))[0])
    return out

# helper class for HOPS control file parsing
class ControlFile(object):

    # extract parameters out of simple control files
    # pass filename or control file contents as string
    # returns list of [condition, statements]
    # condition: string with condition
    # statements: list of [action, value]
    def open(self, cf):
        # action keyword match -- needs to match only beginning of keyword but be careful not to match anything else
        action_kw = "adhoc_ dc_block delay_offs dr_ freqs est_pc gen_cf_record mb_ mbd_ notches optimize_closure pc_ ref_ sb_ skip start stop weak_channel".split()
        pat_act = '\s*(.+?)\s*(\w*' + '|\w*'.join((kw[::-1] for kw in action_kw)) + ')'
        pat_blk = '(.*?)\s*(' + '.*|'.join(action_kw) + '.*)'
        if os.path.exists(cf):
            cf = open(cf).read()
        cf = re.sub('\*.*', '', cf) # strip comments, assume DOTALL is not set
        cf = re.sub('\s+', ' ', cf).strip() # simplify whitespace
        blocks = re.split('\s+if\s+', cf) # isolate if statement blocks

        # separate out actions using reverse findall search
        def splitactions(actions):
            return [[b[::-1], a[::-1]] for (a, b) in re.findall(pat_act, actions[::-1])][::-1]

        # separate conditions from actions in blocks
        return [[a, splitactions(b)] for (a, b) in (re.match(pat_blk, blk).groups() for blk in blocks)]

    # evaluate control file conditional given input parameters
    # assume comments are dropped and all whitespace is reduced to one space (e.g. by controlfile())
    # station: 1 character HOPS code
    # baseline: 2 characters
    # source: source name
    # scan: time-tag with scan start time
    # fourfit syntax: if, else (NYI), and, or, not, () (NYI), <>, to, ?
    # dropmissing: if True, drop condition with missing (undefined) selectors
    def evaluate(self, cond, station=None, baseline=None, source=None, scan=None, dropmissing=False):
        kw = dict(station=station, baseline=baseline, source=source, scan=scan, dropmissing=dropmissing)
        if cond == '':
            return True
        # split operators in reverse precedence
        if " or " in cond:
            (a, b) = cond.split(' or ', 1)
            return(self.evaluate(a, **kw) or self.evaluate(b, **kw))
        if " and " in cond:
            (a, b) = cond.split(' and ', 1)
            return(self.evaluate(a, **kw) and self.evaluate(b, **kw))
        if "not " == cond[:4]:
            return(not self.evaluate(cond[4:], **kw))
        # evaluate conditions
        tok = cond.split(' ')
        if tok[0] == "station":
            return(not dropmissing if station is None else
                   (tok[1] == station) or (tok[1] == '?'))
        if tok[0] == "baseline":
            return(not dropmissing if baseline is None else
                   bool(re.match(tok[1].replace('?', '.'), baseline)))
        if tok[0] == "source":
            return(not dropmissing if baseline is None else
                   bool(re.match(tok[1].replace('?', '.*'), source)))
        if tok[0] == "scan":
            if scan is None:
                return not dropmissing
            if len(tok) == 2: # scan xxx
                return bool(re.match(tok[1].replace('?', '.*'), scan))
            if len(tok) == 3: # scan < xxx
                return cmp(scan, tok[2]) == {'<':-1, '>':1}[tok[1]]
            if len(tok) == 4 and tok[2] == 'to': # scan xxx to yyy
                return((tok[1] <= scan) and (scan <= tok[3]))

    # filter control file dictionary using parameters given to function
    # see evaluate() for details
    def filter(self, station=None, baseline=None, source=None, scan=None, dropmissing=False):
        return ControlFile([[condition, actions] for (condition, actions) in self.cfblocks if
            self.evaluate(condition, station=station, baseline=baseline, source=source, scan=scan, dropmissing=dropmissing)])

    # run through the complete list of control blocks in order and set actions by name
    # returns dictionary with {action: value}
    def actions(self):
        return dict(av for block in self.cfblocks for actionlist in block[1:] for av in actionlist)

    # initialize a control file object from file or string
    def __init__(self, cf):
        if type(cf) == str:
            self.cfblocks = self.open(cf)
        else:
            self.cfblocks = cf

    # write out control codes to string
    def __str__(self):
        lines = []
        for (condition, actions) in self.cfblocks:
            if condition == '': # defaults
                for a in actions:
                    lines.append(' '.join(a))
            else:
                lines.append('if ' + condition)
                for a in actions:
                    lines.append('    ' + ' '.join(a))
        return '\n'.join(lines)

    def __repr(self):
        return self.str()

def wavg(x, sigma=1., col=None, w=None, robust=10.):
    """wavg: weighted average with error propagation

    Args:
        x: data values
        sigma: std error on values, scalar or vector
        col: return Dict labels, for automatic naming of pandas columns under conversion
             col=(mean, [err], [chisq_r], [n_outliers])
        w: optional weights for averaging (otherwise 1/sigma**2)
        robust: if nonzero, remove [robust]-sigma outliers from median value before averaging

    Returns:
        OrderedDict of col=(mean, [err], [chisq_r], [n_outliers])
    """
    sigma = np.broadcast_to(sigma, x.shape)
    ssq = sigma**2
    noutliers = 0
    if robust:
        sigsort = np.sort(sigma)
        merr = 1.253 * sigsort / np.sqrt(np.arange(1, 1+len(x)))
        imin = np.argmin(merr)
        (merrmin, sigthr) = (merr[imin], sigsort[imin]) # at lowest median error
        median = np.median(x[sigma <= sigthr])
        igood = np.abs((x-median)/np.sqrt(merrmin**2 + ssq)) < robust
        (x, sigma, ssq) = (x[igood], sigma[igood], ssq[igood])
        noutliers = np.sum(~igood)
    if w is None:
        w = 1./ssq
    wsq = w**2
    xsum = np.sum(w*x)       # weighted sum
    ssum = np.sum(wsq * ssq) # total variance
    wsum = np.sum(w)         # common divisor
    xavg = xsum / wsum
    eavg = np.sqrt(ssum) / wsum
    chi2 = np.sum((x-xavg)**2/ssq)/(max(1, len(x)-1))
    if col is None:
        return xavg, eavg
    else:
        return_cols = (xavg, eavg, chi2, noutliers)
        from collections import OrderedDict
        return OrderedDict(zip(col, return_cols[:len(col)]))

# segmented RR-LL delay differences
# restarts[site] = [pd.Timestamp list of clock resets]
# assume additional clock reset at 21:00 UT for all sites
def rrll_segmented(a, restarts={}, start='2017-04-04 21:00:00', stop='2017-04-12 21:00:00'):
    """rrll_segmented: general RR-LL delay statistics given alist data

    Args:
        a: dataframe from alist file with rates and delays, delay errors are added if missing
        restarts: special times in which to segment certain stations
        start: start time for establishing day boundaries (default '2017-04-04 21:00:00')
        stop: stop time for establishing day boundaries (default '2017-04-12 21:00:00')
    """
    import pandas as pd
    b = a[a.polarization.isin({'RR', 'LL'})].copy()
    if 'mbd_unwrap' not in b.columns:
        util.unwrap_mbd(b)
    util.add_delayerr(b, bw_factor=sys_fac, mbd_systematic=sys_par, crosspol_systematic=sys_cross)
    g = b.groupby('baseline')
    drange = list(pd.date_range(start, stop))
    for (bl, rows) in g:
        # segments for baseline, adding in any known restarts for either site
        tsbounds = sorted(set(
            itertools.chain(drange, restarts.get(bl[0], []), restarts.get(bl[1], []))))
        # leaving this as CategoryIndex (no "get_values") results in slow pivot_table
        # https://stackoverflow.com/questions/39229005/pivot-table-no-numeric-types-to-aggregate
        # probably pass aggfunc='first' to handle non-numeric types
        b.loc[rows.index, 'segment'] = pd.cut(
            rows.datetime, tsbounds, right=False, labels=None).get_values()
    # convert segment to start, stop values since only 1D objects supported well in pandas
    # for indexing, lose meta-info about right or left closed segment -- too bad
    b['start'] = b.segment.apply(lambda x: x.left)
    b['stop'] = b.segment.apply(lambda x: x.right)
    p = b.pivot_table(aggfunc='first',
        index=['expt_no', 'start', 'stop', 'timetag', 'scan_id', 'source', 'baseline'],
        columns=['polarization'],
        values=['sbdelay', 'mbdelay', 'mbd_unwrap', 'mbd_err', 'snr'])
    p = p.dropna().reset_index(['expt_no', 'source', 'scan_id', 'timetag'])
    p['LLRR'] = p.mbd_unwrap.RR - p.mbd_unwrap.LL
    p['LLRR_err'] = np.sqrt(p.mbd_err.RR**2 + p.mbd_err.LL**2)
    p['LLRR_snr'] = 1./np.sqrt(1./p.snr.RR**2 + 1./p.snr.LL**2)
    rrll_stats = p.groupby(['start', 'stop', 'baseline']).apply(lambda df:
        pd.Series(wavg(df.LLRR, df.LLRR_err,
                       col=['LLRR_mean', 'LLRR_mean_err', 'LLRR_x2', 'LLRR_nout'])))
    rrll_stats['LLRR_nout'] = rrll_stats.LLRR_nout.astype(int)
    # subtract mean RR-LL from each scan
    p['mean_offset'] = rrll_stats.LLRR_mean
    p['mean_err'] = rrll_stats.LLRR_mean_err
    p['offset'] = p.LLRR - p.mean_offset
    p['offset_err'] = np.sqrt(p.LLRR_err**2 + p.mean_err**2)
    p['offset_std'] = p.offset / p.offset_err
    p['outlier'] = p.offset_std.abs() > 10
    p['mbdelay', 'mean'] = ((p.snr.LL**2*p.mbdelay.LL + p.snr.RR**2*(p.mbdelay.RR - p.mean_offset)) /
                            (p.snr.LL**2 + p.snr.RR**2))
    return((p.sort_index(), rrll_stats))

# function to convert delay offsets table row to control code element
def doff2cf(row, nchan=32):
    """doff2cf: convert delay offsets table row to control file codes

    Args:
        row: delay offsets code
        nchan: number of channels to define

    Returns:
        control file codes (str)
    """
    import datetime
    fmt = "%Y-%m-%d %H:%M:%S"
    onesec = datetime.timedelta(seconds=1)
    # some padding on start and stop times
    start_tt = util.dt2tt(datetime.datetime.strptime(str(row.start), fmt) - onesec)
    stop_tt =  util.dt2tt(datetime.datetime.strptime(str(row.stop), fmt) + onesec)
    codes = lex[:nchan]
    delay = -row.LR_delay * 1e3 # convert from us to ns, undo measured R-L
    delay_offs = ' '.join(["%.4f" % delay] * nchan)
    cf = """
if station %s and scan %s to %s
    delay_offs_r %s %s
    pc_delay_r %.4f
""" % (row.site, start_tt, stop_tt, codes, delay_offs, delay)
    return cf

# function to create control codes from ALMA SBD-MBD offset measurement where ALMA is ref station
def sbdmbd2cf(row):
    """sbdmbd2cf: convert MBD-SBD table row to control file codes for ALMA

    Args:
        row: delay offsets code

    Returns:
        control file codes (str)
    """
    import datetime
    fmt = "%Y-%m-%d %H:%M:%S"
    onesec = datetime.timedelta(seconds=1)
    # some padding on start and stop times
    start_tt = util.dt2tt(datetime.datetime.strptime(str(row.start), fmt) - onesec)
    stop_tt =  util.dt2tt(datetime.datetime.strptime(str(row.stop), fmt) + onesec)
    pol = row.polarization[0].lower()
    delay = row.sbdmbd * 1e3 # convert from us to ns, measured on REF station (A)
    cf = """if station %s and scan %s to %s pc_delay_%s %.4f
""" % ('A', start_tt, stop_tt, pol, delay)
    return cf

# ff: fringe filename
# doadhoc: if true, apply on-the-fly adhoc corrections without freq slicing
# dt: averaging time in AP
# df: averaging num channels
# pol: pol filter for ff wildcard
# almaref: if ALMA not in ff, use ALMA to reference adhoc phases withou freq slicing
def vecphase(ff, doadhoc=True, dt=30, df=4, pol=None, almaref=True, replacedata=None):
    b = getfringefile(ff, quiet=True, pol=pol)
    p = params(b)
    v = pop212(b) if replacedata is None else replacedata
    if doadhoc:
        if almaref and 'A' not in p.baseline:
            p1 = params('A' + p.baseline[0] + '*', pol=pol)
            p2 = params('A' + p.baseline[1] + '*', pol=pol)
            ah1 = adhoc('A' + p.baseline[0] + '*', pol=pol)
            ah2 = adhoc('A' + p.baseline[1] + '*', pol=pol)
            ahrot = np.exp(-1j*(ah2.phase.mean(axis=1)-ah1.phase.mean(axis=1))*np.pi/180)
            ahrot *= p.trot.conj() * p1.trot.conj() * p2.trot
        else:
            ah = adhoc(ff)
            ahrot = np.exp(-1j*ah.phase.mean(axis=1)*np.pi/180)
        v = v * ahrot[:,None]
    v = v * np.exp(-1j*np.angle(np.mean(v)))
    (nap, nchan, nspec) = (p.nap, p.nchan, 1)
    clip = 0
    clip = np.fmod(nap, dt) # fit ni non-overlapping time segments after decimation
    if clip > 0: # remove small amount of end data for equal segments
        nap = nap-clip
        v = v[:nap]
    clipf = 0
    clipf = np.fmod(nchan, df)
    if clipf > 0:
        nchan = nchan-clipf
        v = v[:,:nchan]
    v = v.reshape((nap//dt, dt, nchan*nspec//df, df))
    v = v.sum(axis=(1, 3)) # stack on time, and frequency decimation factors
    x = np.arange(v.shape[0])
    y = np.arange(v.shape[1])
    xm = np.arange(v.shape[0])+0.5
    ym = np.arange(v.shape[0])+0.5
    (xx, yy) = np.meshgrid(xm, ym)
    vn = v / np.abs(v)
    qx = vn.real
    qy = vn.imag
    plt.quiver(xx, yy, qx.T, qy.T, pivot='mid')
    plt.xlim(0, v.shape[0])
    plt.ylim(0, v.shape[1])
    plt.grid()
    ax = plt.gca()
    plt.xticks(x)
    plt.yticks(y)
    plt.grid(ls='-', color='black', lw=2)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    for axi in (plt.gca().xaxis, plt.gca().yaxis):
        for tic in axi.get_major_ticks():
            tic.tick1On = tic.tick2On = False
            tic.label1On = tic.label2On = False
    plt.xlabel('time')
    plt.ylabel('frequency')
    plt.title('phase %s %s %s +%ds' % (p.baseline, p.source, p.timetag, p.T))
    plt.setp(plt.gcf(), figwidth=8*p.T/420, figheight=5)

# example delay offsets for ER1
# ('R',3597):5.,('R',3598):20.,('R',3599):35.,('R',3600):5.,('R',3601):20.
# to be added to clock_early [ns]
loffs={
            ('A',3601):-10.,
            ('J',3597):5.,('J',3601):-5.,
            ('L',3598):5.,('L',3599):10.,
            ('P',3597):5.,('P',3598):5.,('P',3599):-10.,('P',3600):5.,
            ('X',3598):-5.,('X',3599):-10.,
            ('Z',3598):5.,('Z',3599):10.,('Z',3601):5.,
        }
hoffs={
            ('A',3601):0.,
            ('J',3597):15.,('J',3600):-40.,
            ('L',3597):-60.,('L',3598):-10.,('L',3600):10.,('L',3601):10.,
            ('R',3600):-48.,
            ('P',3597):10,('P',3598):25,('P',3599):5,('P',3600):-10,('P',3601):-65,
            ('X',3598):5,('X',3598):0,('X',3599):-10,('X',3600):-10,('X',3601):-10,
            ('Z',3597):-30,('Z',3598):-5,('Z',3600):15,('Z',3601):15,
        }

# make one plot from data frame, group by baseline
# delays converted to [ns]
# offs: dict of offsets by site, if REM will be subtracted from baseline value
# rem: use only baselines including station and set as REM station
# col: column in data frame to plot
def trendplot(df, rem='', offs={}, col='sbdelay'):
    band = 'lo' if df.iloc[0].ref_freq < 228000 else 'hi'
    from matplotlib.legend import Legend
    mk = OrderedDict((('LL','.'), ('RR','x'), ('RL','|'), ('LR','_')))
    b = df[df.baseline.str.contains(rem)].copy()
    flip(b, b.baseline.str[0] == rem)
    color = dict(zip(sorted(set(b.baseline)),
           itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])))
    lines = []
    labels = []
    for (name, rows) in b.groupby(['baseline', 'polarization']):
        (bl, pol) = name
        label = bl if pol == 'LL' else '_nolabel'
        offset = np.array([offs.get((bl[1], expt), 0.) - offs.get((bl[0], expt), 0.)
                           for expt in rows.expt_no])
        val = (1e3 if 'mbd' in col or 'sbd' in col or 'delay' in col else 1.) * rows[col]
        h = plt.plot(rows.scan_no, val - offset, marker=mk[pol], ls='none',
                color=color[bl], label=label)
    lines = [plt.Line2D([0], [0], color='k', marker=mk[pol], ls='none') for pol in mk.keys()]
    leg = Legend(plt.gca(), lines, mk.keys(), loc='lower right', ncol=1)
    plt.gca().add_artist(leg)
    plt.xlabel('scan')
    plt.ylabel('%s' % col)
    # plt.xlim(0, plt.xlim()[1]*1.03)
    plt.legend(loc='upper right')
    putil.tag('%s %s' % (rem, band), loc='upper left')
    plt.grid(alpha=0.25)

