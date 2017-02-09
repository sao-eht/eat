"""
HOPS utilities
"""

#2016-10-31 Lindy Blackburn

from ..io import util
import numpy as np
from numpy.fft import fft2, fftfreq, fftshift # for fringe fitting
import mk4 # part of recent HOPS install, need HOPS ENV variables
import datetime
import ctypes
from argparse import Namespace
import matplotlib.pyplot as plt
from ..plots import util as putil
from matplotlib.offsetbox import AnchoredText
import glob
import os

# return mk4fringe based on object, filename, or glob path
# if glob return the file with latest getmtime time
# maybe this should get latest HOPS rootcode instead..
# remember file path to guess working directory for future calls
# filelist=True will return a list of all files found
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
        print files[-1]
        b = mk4.mk4fringe(files[-1]) # use last updated file
    return b
    
# unwrap short to positive int in multiples from 1e6 to 1024e6
def short2int(short):
    return short2int.lookup[short]

short2int.lookup = {ctypes.c_short(i*1000000).value:i*1000000 for i in range(1024)}

def mk4time(time):
    return datetime.datetime.strptime("%d-%03d %02d:%02d:%02d.%06d" %
        (time.year, time.day, time.hour, time.minute, int(time.second), int(0.5+1e6*(time.second-int(time.second)))),
        "%Y-%j %H:%M:%S.%f")

# populate the type_212 visib data into array
# (nap, nchan)
def pop212(b):
    b = getfringefile(b)
    (nchan, nap) = (b.n212, b.t212[0].contents.nap)
    data212 = np.zeros((nchan, nap, 3), dtype=np.float32)
    for i in range(nchan):
        q = (mk4.newphasor*nap).from_address(ctypes.addressof(b.t212[i].contents.data))
        data212[i] = np.frombuffer(q, dtype=np.float32, count=-1).reshape((nap, 3))
    v = data212[:,:,0] * np.exp(1j * data212[:,:,1])
    return v.T

# populate the type_230 visib data into array automatically detect sideband
# (nchan, nap, nspec)
def pop230(b):
    b = getfringefile(b)
    (nchan, nap, nspec) = (b.n212, b.t212[0].contents.nap, b.t230[0].contents.nspec_pts)
    data230 = np.zeros((nchan, nap, nspec/2), dtype=np.complex128)
    for i in range(nchan): # loop over HOPS channels
        idx = b.t205.contents.ffit_chan[i].channels[0] # need to get index into mk4 chdefs
        istart = nspec/2 if (b.t203[0].channels[idx].refsb == 'U') else 0 # USB vs LSB fixed offset
        for j in range(nap):
            # get a complete spectrum block for 1 AP at a time
            q = (mk4.complex_struct*nspec).from_address(
                ctypes.addressof(b.t230[j+i*nap].contents.xpower))
            # type230 frequeny order appears to be [---LSB--> LO ---USB-->]
            data230[i,j,:] = np.frombuffer(q, dtype=np.complex128, count=-1)[istart:istart+nspec/2]
    return data230

# populate type_120 visib data into array -- use FRINGE file, do NOT give it COREL file
# because the FRINGE file will determine parameters incl polarization, but data will come from COREL file
# if you have not run fourfit then *this will not work*
# we don't bother flipping LSB because convention is unknown, and recent data should be USB (zoom-band)
def pop120(b):
    b = getfringefile(b) # fringe file
    ctok = getfringefile.last[-1].split('.')
    c = mk4.mk4corel('/'.join(getfringefile.last[:-1] + [ctok[0] + '..' + ctok[-1]])) # corel file
    # this is not a great way to get nap (what if incomplete?) but nindex appears incorrect..
    (nchan, nap, nspec) = (b.n212, b.t212[0].contents.nap, c.t100.contents.nlags)
    # require spectral type (DiFX)
    if c.index[0].t120[0].contents.type != '\x05':
        raise(Exception("only supports SPECTRAL type from DiFX->Mark4"))
    data120 = np.zeros((nchan, nap, nspec), dtype=np.complex64)
    # 120: (ap, channel, spectrum), this is mk4 channels (31, 41, ..) not HOPS channels (A, B, ..)
    for i in range(nchan): # loop over HOPS channels
        # by construction the ordering of type_101 and type_203 is the same (fill_203.c)
        # so we can avoid using the mk4 channel pair index to bookkeep
        idx = b.t205.contents.ffit_chan[i].channels[0] # need to get index into mk4 chdefs
        for j in range(nap):
            # get a complete spectrum block for 1 AP at a time
            q = (mk4.spectral*nspec).from_address(
                ctypes.addressof(c.index[idx].t120[j].contents.ld))
            # type230 frequeny order appears to be [---LSB--> LO ---USB-->]
            data120[i,j,:] = np.frombuffer(q, dtype=np.complex64, count=-1)
    return data120
     
# some HOPS channel parameter info
# same function as HOPS param_struct (unfortunately)
# frot and trot rotate opposite the detected fringe location
# i.e. they subtract the delay, rate under multiplication
def params(b):
    if type(b) is str:
        name = b
        b = getfringefile(b)
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
    offset = np.fmod(sbd - mbd + 1.5*amb, amb)
    delay = (sbd - offset + 0.5*amb) # unwrap to be close to SBD, us
    rate = b.t208.contents.resid_rate # us/s
    snr = b.t208.contents.snr
    # time vector and rotator
    T = (mk4time(b.t205.contents.stop) - mk4time(b.t205.contents.start)).total_seconds()
    # ref_time = mk4time(b.t205.contents.start) + T/2. # place reference time in middle
    ap = T / nap
    dtvec = ap * np.arange(nap) - (T-ap)/2.
    trot = np.exp(-1j * rate * dtvec * 2*np.pi*ref_freq) # reverse rotation due to rate
    # frequency matrix (channel, spectrum) and rotator
    fedge = np.array([1e-6 * ch.ref_freq for ch in cinfo])
    flip = np.array([-1 if ch.refsb == 'L' else 1 for ch in cinfo])
    bw = np.array([0.5e-6 * short2int(ch.sample_rate) for ch in cinfo])
    (foffset, dfvec, frot) = (dict(), dict(), dict())
    nlags120 = nlags/2 # guess
    foffset[230] = np.array([(f*np.arange(0.5, nlags)*bwn/nlags)[::f] for (f, bwn) in zip(flip, bw)])
    foffset[120] = np.array([(f*np.arange(0.5, nlags120)*bwn/nlags120)[::f] for (f, bwn) in zip(flip, bw)])
    dfvec[230] = (fedge[:,None] + foffset[230]) - ref_freq
    dfvec[120] = (fedge[:,None] + foffset[120]) - ref_freq
    dfvec[212] = fedge + (flip*bw)/2. - ref_freq
    frot[230] = np.exp(-1j * delay * dfvec[230] * 2*np.pi)
    frot[120] = np.exp(-1j * delay * dfvec[120] * 2*np.pi) # assuming nlags120 = nlags230/2
    frot[212] = np.exp(-1j * delay * dfvec[212] * 2*np.pi) # note type_212 is already rotated in data
    return Namespace(name=name, ref_freq=ref_freq, nchan=nchan, nap=nap, nspec=nspec, nlags=nlags,
        code=clabel, pol=cinfo[0].refpol + cinfo[0].rempol, sbd=sbd, mbd=mbd, delay=delay, rate=rate, snr=snr, T=T,
        ap=ap, dtvec=dtvec, trot=trot, fedge=fedge, bw=bw, foffset=foffset, dfvec=dfvec, frot=frot,
        baseline=b.t202.contents.baseline, source=b.t201.contents.source,
        scan_name=b.t200.contents.scan_name, scantime=mk4time(b.t200.contents.scantime))

# some unstructured channel info for quick printing
def chaninfo(b):
    b = getfringefile(b)
    nchan = b.n212
    # putting them in "fourfit" order also puts them in frequency order
    idx = [(q.ffit_chan_id, q.channels[0]) for q in b.t205.contents.ffit_chan[:nchan]] # MAX #64 for ffit_chan
    chinfo = [(hops_id, q.index, q.ref_chan_id, q.rem_chan_id, round(q.ref_freq/1e6), round(q.rem_freq/1e6),
              round(q.ref_freq/1e6 - b.t205.contents.ref_freq),
              q.refsb, q.remsb, short2int(q.sample_rate)/1e6)
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
# delay_off, rate_off: subtract this from the data before doing search
# manual offsets will show up in axis labels, automatic offsets (from centering) will not
def findfringe(fringefile, kind=None, res=4, showx=6, showy=6, center=(None, None),
               dt=2, df=None, ni=1, ret=False, showhops=False,
               delay_off=0., rate_off=0., flip=False, segment=(None, None)):
    b = getfringefile(fringefile)
    p = params(b)
    (nchan, nap) = (b.n212, b.t212[0].contents.nap)
    if kind is None:
        kind = 230 if bool(b.t230[0]) else 212 # use type_230 if available
    if kind==212:
        nspec = 1
        df = df or 1
        v = pop212(b)[:,:,None]
    elif kind==230:
        nspec = b.t230[0].contents.nspec_pts / 2 # one sideband
        df = df or 4 # speed-up if using full spectral resolution
        v = np.swapaxes(pop230(b), 1, 0)  # put AP as axis 0
        assert(v.shape == (nap, nchan, nspec))   # make sure loaded data has right dimensions
        if flip:
            v = v[:,:,::-1] # test flip frequency order of spectral points
    elif kind==120: # original correlator output
        v = np.swapaxes(pop120(b), 1, 0)  # put AP as axis 0
        df = df or 2 # arbitrary, but compensate for type_230 inflation factor of x2 (SSB)
        nspec = v.shape[-1]
        assert(v.shape == (nap, nchan, nspec))
        if flip: # fake support for LSB?
            v = v[:,:,::-1] # test flip frequency order of spectral points

    # apply fringe rotations
    if(center=='hops'):
        center = (p.delay*1e3, p.rate*1e6)
    if(kind==212):
        delay_off -= p.delay*1e3 # ns
        rate_off -= p.rate*1e6   # ps/s
    # note this will affect the data, and must also be reflected in the delay, rate vectors
    if center[0] is not None:
        delay_off += center[0]
    if center[1] is not None:
        rate_off += center[1]
    print "rotation subtracted from data: %.3f [ns], %.3f [ps/s]" % (delay_off, rate_off)
    frot = np.exp(-1j * 1e-3*delay_off * p.dfvec[kind].reshape((nchan, -1)) * 2*np.pi)
    trot = np.exp(-1j * 1e-6*rate_off * p.dtvec * 2*np.pi*p.ref_freq)
    v = v * trot[:,None,None] * frot[None,:,:]

    v = v[slice(*segment)] # apply time segment cut
    nap = len(v)      # number of AP's inside time segment
    clip = np.fmod(nap, dt*ni) # fit ni non-overlapping time segments after decimation
    if clip > 0: # remove small amount of end data for equal segments
        nap = nap-clip
        v = v[:nap]

    # block averaging factors to speedup, make sure no phase wrap!
    v = v.reshape((ni, nap/dt/ni, dt, nchan*nspec/df, df))
    v = v.sum(axis=(2, 4)) # stack on time, and frequency decimation factors

    # the normalized complex visibility and FFT search delay/rate
    (zpap, zpch) = (nextpow2(res*v.shape[1]), nextpow2(res*v.shape[2])) # zero padding for time and freq
    fringevis = fftshift(fft2(v, s=(zpap, zpch)))
    fqap = fftshift(fftfreq(zpap)) # "frequency" range of the rate space [cycles/sample_spacing]
    fqch = fftshift(fftfreq(zpch)) # "frequency" range of the delay space [cycles/sample_spacing]

    # single-channel spacing [Hz] and decimated spectral point spacing [MHz]
    sb_spacing = np.diff(sorted(b.t203.contents.channels[i].ref_freq for i in range(nchan)))[int(nchan/2)]
    spec_spacing = df * 1e-6 * sb_spacing / nspec
    # accumulation period [s]
    ap = dt * (mk4time(b.t205.contents.stop) - mk4time(b.t205.contents.start)).total_seconds() / (nap + clip)
    delay = (center[0] if center[0] else 0.) + 1e9 * fqch / (spec_spacing * 1e6) # ns
    rate = (center[1] if center[1] else 0.) + 1e12 * fqap / ap / (p.ref_freq * 1e6) # in ps/s
    dd = delay[1] - delay[0]
    dr = rate[1] - rate[0]

    (left, right, bottom, top) = (delay[0]-dd/2., delay[-1]+dd/2., rate[0]-dr/2., rate[-1]+dr/2.)
    # set the plot aspect relative to nyquist (propotional to fringe FWHM)
    BW = 1e-6 * sb_spacing * nchan
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
    print np.max(fringepow)

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

    ratio = float(showy)/showx
    plt.setp(plt.gcf(), figwidth=2.+3./np.sqrt(ratio), figheight=2.+3.*np.sqrt(ratio))
    plt.tight_layout()

    (i,j) = np.unravel_index(np.argmax(fringepow), fringepow.shape) # get new max location
    putil.tag('%s [%d]' % (p.scan_name, p.scantime.year), loc='upper left', framealpha=0.85)
    putil.tag('%s [%s]' % (p.baseline, p.source), loc='upper right', framealpha=0.85)
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
             timeseries=False, centerphase=False, snrweight=True, kind=230):
    if type(bs) is str:
        bs = getfringefile(bs, filelist=True)
    if len(bs) > 1:
        centerphase = True
    vs = None
    for b in bs:
        b = getfringefile(b)
        if b.t208.contents.snr < snrthr:
            print "snr %.2f, skipping" % b.t208.contents.snr
            continue
        if not bool(b.t230[0]):
            print "skipping no t230"
            continue
        v = pop230(b)   # visib array (nchan, nap, nspec/2)
        p = params(b) # channel and fringe parameters
        nrow = bool(timeseries) + np.int((float(p.nchan) / ncol) + 0.5)
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
        plt.xlim(-p.T/2., p.T/2.)
    if figsize is None:
        plt.setp(plt.gcf(), figwidth=8, figheight=8.*float(nrow)/ncol)
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
    vrot = vrot.reshape((nt/dt, dt, nf/df, df))
    vrot = vrot.sum(axis=(1, 3)) # stack on time, and frequency decimation factors
    plt.plot([0,0], [vrot.re, vrot.im], 'b.-', alpha=0.25)
    vtot = np.sum(vrot) / len(vrot.ravel())
    plt.plot([0,0], [vtot.re, vtot.im], 'r.-', lw=2, ms=4, alpha=1.0)

def timeseries(bs, dt=1):
    if not hasattr(bs, '__iter__'):
        bs = [bs,]
    nrow = len(bs)
    for (i, b) in enumerate(bs):
        p = params(b)
        plt.subplot(nrow, 1, 1+i)
        v = pop212(b).sum(axis=1) # stack over channels
        nt = len(v)
        dt = min(dt, nt)
        nt = nt - np.fmod(nt, dt) # fit time segments after decimation
        v = v[:nt].reshape((nt/dt, -1)).sum(axis=1) # clip to multiple of dt and stack
        t = p.dtvec[:nt].reshape((-1, dt)).mean(axis=1) + p.T/2.
        amp = np.abs(v)
        phase = np.angle(v)
        plt.plot(t, amp, 'b.-')
        plt.ylim(0, plt.ylim()[1])
        plt.gca().set_yticklabels([])
        plt.twinx()
        plt.plot(t, phase, 'r.-')
        plt.ylim(-np.pi, np.pi)
        plt.gca().set_yticklabels([])
        putil.rmgaps(1e6, 2.0)
        plt.xlim(0, p.T)
        plt.gca().add_artist(AnchoredText(p.baseline, loc=1, frameon=False, borderpad=0))
    plt.setp(plt.gcf(), figwidth=8, figheight=2+nrow)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)

# calculate delay at each AP using type120 data
def delayscan(fringefile, res=4, dt=1, df=None, delayrange=(-1e4, 1e4)):
    b = getfringefile(fringefile)
    p = params(b)
    (nchan, nap) = (b.n212, b.t212[0].contents.nap)
    v = np.swapaxes(pop120(b), 1, 0)  # put AP as axis 0
    df = df or 1 # arbitrary, but compensate for type_230 inflation factor of x2 (SSB)
    nspec = v.shape[-1]
    assert(v.shape == (nap, nchan, nspec))
    clip = np.fmod(nap, dt) # fit ni non-overlapping time segments after decimation
    if clip > 0: # remove small amount of end data for equal segments
        nap = nap-clip
        v = v[:nap]

    # block averaging factors to speedup, make sure no phase wrap!
    v = v.reshape((nap/dt, dt, nchan*nspec/df, df))
    v = v.sum(axis=(1, 3)) # stack on time, and frequency decimation factors

    # the normalized complex visibility and FFT search delay/rate
    zpch = nextpow2(res*v.shape[1]) # zero padding for freq
    fringevis = np.fft.fft(v, n=zpch) # by default operate on axis=-1 (frequency axis)
    fqch = fftfreq(zpch) # "frequency" range of the delay space [cycles/sample_spacing]

    # single-channel spacing [Hz] and decimated spectral point spacing [MHz]
    sb_spacing = np.diff(sorted(b.t203.contents.channels[i].ref_freq for i in range(nchan)))[int(nchan/2)]
    spec_spacing = df * 1e-6 * sb_spacing / nspec
    delay = 1e9 * fqch / (spec_spacing * 1e6) # ns

    inside = ((delay >= delayrange[0]) & (delay <= delayrange[1]))
    imax = np.argmax(np.abs(fringevis[:,inside]), axis=-1) # the maximum frequency index
    delays = delay[inside][imax] # the solved delays

    return delays.ravel()

