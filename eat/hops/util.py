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


# unwrap short to positive int in multiples from 1e6 to 1024e6
def short2int(short):
    return short2int.lookup[short]

short2int.lookup = {ctypes.c_short(i*1000000).value:i*1000000 for i in range(1024)}

def mk4time(time):
    return datetime.datetime.strptime("%d-%03d %02d:%02d:%02d.%06d" %
        (time.year, time.day, time.hour, time.minute, int(time.second), int(0.5+1e6*(time.second-int(time.second)))),
        "%Y-%j %H:%M:%S.%f")

# populate the type_212 visib data into array
def pop212(b):
    if type(b) is str:
        b = mk4.mk4fringe(b)
    (nchan, nap) = (b.n212, b.t212[0].contents.nap)
    data212 = np.zeros((nchan, nap, 3), dtype=np.float32)
    for i in range(nchan):
        q = (mk4.newphasor*nap).from_address(ctypes.addressof(b.t212[i].contents.data))
        data212[i] = np.frombuffer(q, dtype=np.float32, count=-1).reshape((nap, 3))
    v = data212[:,:,0] * np.exp(1j * data212[:,:,1])
    return v.T

# populate the type_230 visib data into array automatically detect sideband
def pop230(b):
    if type(b) is str:
        b = mk4.mk4fringe(b)
    (nchan, nap, nspec) = (b.n212, b.t212[0].contents.nap, b.t230[0].contents.nspec_pts)
    data230 = np.zeros((nchan, nap, nspec/2), dtype=np.complex128)
    for i in range(nchan): # loop over HOPS channels
        idx = b.t205.contents.ffit_chan[i].channels[0] # need to get index into mk4 chdefs
        flip = 1 if (b.t203[0].channels[idx].refsb == 'L') else -1 # LSB vs USB organization in t230
        for j in range(nap):
            # grab on spectrum for 1 AP
            q = (mk4.complex_struct*nspec).from_address(
                ctypes.addressof(b.t230[j+i*nap].contents.xpower))
            data230[i,j,:] = np.frombuffer(q, dtype=np.complex128, count=-1)[::flip][:nspec/2]
    return data230

# some HOPS channel parameter info
# same function as HOPS param_struct (unfortunately)
def params(b):
    if type(b) is str:
        name = b
        b = mk4.mk4fringe(b)
    else:
        name = b.id.contents.name
    ref_freq = b.t205.contents.ref_freq
    # dimensions
    (nchan, nap) = (b.n212, b.t212[0].contents.nap)
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
    fs = np.array([short2int(ch.sample_rate) for ch in cinfo])
    bw = 1e-6 * fs/2. # to MHz
    if nspec:
        foffset = np.array([(2*bwn/nspec) * np.arange(0.5, nspec/2) +
            (-bwn if ch.refsb == 'L' else bwn) for (ch, bwn) in zip (cinfo, bw)])
        dfvec = (fedge[:,None] + foffset) - ref_freq
        frot = np.exp(-1j * delay * dfvec * 2*np.pi)
    else:
        (offset, dfvec, frot) = (None, None, None)
    return Namespace(name=name, ref_freq=ref_freq, nchan=nchan, nap=nap, nspec=nspec,
        code=clabel, sbd=sbd, mbd=mbd, delay=delay, rate=rate, snr=snr, T=T,
        ap=ap, dtvec=dtvec, trot=trot, fedge=fedge, fs=fs, bw=bw, dfvec=dfvec, frot=frot,
        baseline=b.t202.contents.baseline, source=b.t201.contents.source)

# some unstructured channel info for quick printing
def chaninfo(b):
    if type(b) is str:
        b = mk4.mk4fringe(b)
    nchan = b.n212
    # putting them in "fourfit" order also puts them in frequency order
    idx = [(q.ffit_chan_id, q.channels[0]) for q in b.t205.contents.ffit_chan[:nchan]] # MAX #64 for ffit_chan
    chinfo = [(hops_id, q.index, q.ref_chan_id, q.rem_chan_id, round(q.ref_freq/1e6), round(q.rem_freq/1e6),
              round(q.ref_freq/1e6 - b.t205.contents.ref_freq),
              q.refsb, q.remsb, short2int(q.sample_rate)/1e6)
              for (hops_id, q) in [(hops_id, b.t203[0].channels[i]) for (hops_id, i) in idx]]
    return chinfo

# kind: use type_212 or type_230 data
# res: zero-padding factor for FFT
# show: how many fringe FWHM to plot
# center: delay,rate center of plot (default: on max)
# dt, df: decimation factors in time, channels
# ni: number of incoherent averages (1=scan average)
# note: factors must evenly divide data
# ret: return the full FFT power matrix if true and do not plot anything
# clip: remove some AP's from end to fit into factors
def findfringe(fringefile, kind=230, res=4, showx=6, showy=6, center=None,
               dt=2, df=None, ni=1, ret=False, flip=False, showhops=False):

    def nextpow2(x): # next power of 2 for efficient fft
        return np.power(2, int(np.ceil(np.log2(x))))
    def expmean(x, s=8, n=4): # robust mean of exponential distribution
        fac = 1.0
        m = np.mean(x)
        for i in range(n):
            fac = 1.0 - (s*np.exp(-s)/(1.-np.exp(-s)))
            m = np.mean(x[x<s*m]) / fac
        return m

    b = mk4.mk4fringe(fringefile)
    p = params(b)
    (nchan, nap) = (b.n212, b.t212[0].contents.nap)
    clip = np.fmod(nap, dt*ni) # fit ni non-overlapping time segments after decimation
    if kind==212:
        nspec = 1
        df = df or 1
        v = pop212(b)
    elif kind==230:
        nspec = b.t230[0].contents.nspec_pts / 2 # one sideband
        df = df or 4 # speed-up if using full spectral resolution
        v = np.swapaxes(pop230(b), 1, 0)  # put AP as axis 0
        assert(v.shape == (nap, nchan, nspec))   # make sure loaded data has right dimensions
        if flip:
            v = v[:,:,::-1] # reverse SBD channels for testing
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

    # necessary parameters for conversion into delay/rate
    # reference frequency for HOPS [MHz]
    ref_freq = b.t205.contents.ref_freq
    # single-channel spacing [Hz] and decimated spectral point spacing [MHz]
    sb_spacing = np.diff(sorted(b.t203.contents.channels[i].ref_freq for i in range(nchan)))[int(nchan/2)]
    spec_spacing = df * 1e-6 * sb_spacing / nspec
    # accumulation period [s]
    ap = dt * (mk4time(b.t205.contents.stop) - mk4time(b.t205.contents.start)).total_seconds() / (nap + clip)
    delay = 1e9 * fqch / (spec_spacing * 1e6) # ns
    rate = 1e12 * fqap / ap / (ref_freq * 1e6) # in ps/s
    if kind==212:
        delay += p.delay*1e3
        rate += p.rate*1e6
    dd = delay[1] - delay[0]
    dr = rate[1] - rate[0]

    (left, right, bottom, top) = (delay[0]-dd/2., delay[-1]+dd/2., rate[0]-dr/2., rate[-1]+dr/2.)
    # set the plot aspect relative to nyquist (propotional to fringe FWHM)
    BW = 1e-6 * sb_spacing * nchan
    T = ap * v.shape[1]
    fwhm_delay = 1e3 / BW # ns
    fwhm_rate = 1e6 / T / ref_freq # ps/s
    aspect = abs(fwhm_delay / fwhm_rate)
    fringepow = np.abs(fringevis)**2 # fringe power before incoherent averaging
    fringepow = fringepow / (0.5 * expmean(fringepow.ravel())) # normalize to snr=1 for noise
    fringepow = np.sum(fringepow, axis=0) # the incoherent average of fringe power
    
    if ret:
        return Namespace(fringepow=fringepow, BW=BW, T=T, fwhm_delay=fwhm_delay, fwhm_rate=fwhm_rate,
            delay=delay, rate=rate, dd=dd, dr=dr, ref_freq=ref_freq, dt=dt, df=df, ni=ni,
            extent=(left, right, bottom, top), aspect=aspect)

    (i,j) = np.unravel_index(np.argmax(fringepow), fringepow.shape)
    if center is None:
        center = (None, None)
    center = (delay[j] if center[0] is None else center[0], rate[i] if center[1] is None else center[1])

    mask_delay = np.abs(delay - center[0]) > showx*fwhm_delay
    mask_rate = np.abs(rate - center[1]) > showy*fwhm_rate
    fringepow[mask_rate,:] = 0 # mask power outside region of interest
    fringepow[:,mask_delay] = 0
    print np.max(fringepow)

    plt.imshow(fringepow, cmap='jet', origin='lower', extent=(left, right, bottom, top),
        aspect=aspect, interpolation='Nearest')
    plt.xlabel('delay [ns]')
    plt.ylabel('rate [ps/s]')
    plt.xlim(center[0] + np.array((-1,1))*showx*fwhm_delay)
    plt.ylim(center[1] + np.array((-1,1))*showy*fwhm_rate)

    # show locatino of fourfit fringe solution
    if showhops:
        plt.plot(b.t208.contents.resid_mbd*1e3, b.t208.contents.resid_rate*1e6, 'kx', ms=24, mew=10)
        plt.plot(b.t208.contents.resid_mbd*1e3, b.t208.contents.resid_rate*1e6, 'wx', ms=20, mew=6)

    ratio = float(showy)/showx
    plt.setp(plt.gcf(), figwidth=2.+3./np.sqrt(ratio), figheight=2.+3.*np.sqrt(ratio))
    plt.tight_layout()

    (i,j) = np.unravel_index(np.argmax(fringepow), fringepow.shape) # get new max location
    putil.tag('%s [%d]' % (b.t200.contents.scan_name, b.t200.contents.scantime.year), loc='upper left', framealpha=0.75)
    putil.tag('%s [%s]' % (b.t202.contents.baseline, b.t201.contents.source), loc='upper right', framealpha=0.75)
    putil.tag('%.3f ns' % delay[j], loc='lower left', framealpha=0.75)
    putil.tag('%.3f ps/s' % rate[i], loc='lower right', framealpha=0.75)

# average over many files, please make sure frequency setup is the same
# delay, rate: [us, and us/s], if None use fourfit soln
# ap is messy to derive from fringe files (fourfit pulls it from ovex)
# df: decimation factor in frequency for better SNR
# df: decimation factor in time if timeseires==True
def spectrum(bs, ncol=4, delay=None, rate=None, df=1, dt=1, figsize=None, snrthr=0.,
             timeseries=False, centerphase=None, snrweight=True):
    if not hasattr(bs, '__iter__'):
        if centerphase is None:
            centerphase = False
        bs = [bs,]
    else:
        if centerphase is None:
            centerphase = True
    vs = None
    for b in bs:
        if type(b) is str:
            print b
            b = mk4.mk4fringe(b)
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
        frot = np.exp(-1j * delay * p.dfvec * 2*np.pi)
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
