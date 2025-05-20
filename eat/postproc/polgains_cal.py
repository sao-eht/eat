# polgains_cal.py
# functions for calibrating RCP-LCP phase offsets
#
#    Copyright (C) 2019 Maciek Wielgus (maciek.wielgus(at)gmail.com)
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.


from __future__ import division
from __future__ import print_function

from builtins import str
from builtins import range
from builtins import object

import numpy as np
import scipy.optimize as opt
import time
import copy
from multiprocessing import cpu_count, Pool

import ehtim.obsdata
import ehtim.parloop as parloop
import ehtim.observing.obs_helpers as obsh
import ehtim.const_def as ehc

MAXIT = 5000

###################################################################################################
# Polarimetric-Phase-Gains-Calibration
###################################################################################################


def polgains_cal(obs, reference='AA', sites=[], method='phase', minimizer_method='BFGS', pad_amp=0.,
                 gain_tol=.2, solution_interval=0.0, scan_solutions=False,
                 caltable=False, processes=-1, show_solution=False, msgtype='bar'):

    # TODO: function to globalize the polarimetric solution in time
    # given provided absolute calibration of the reference station
    # so that we have a meaningful EVPA
    """Polarimeteric-phase-gains-calibrate a dataset.
        Numerically solves for polarimetric gains to align RCP and LCP feeds.
        Uses all baselines to find the solution. Effectively assumes phase of Stokes V to be zero.
        Because fits are local, it's not providing absolute phase calibration.

       Args:
        obs (Obsdata): The observation to be calibrated
        reference (str): station used as reference to break the degeneracy
                         (LCP on baselines to the reference station remains unchanged)
        sites (list): list of sites to include in the polarimetric calibration.
                      Empty list calibrates all sites
        method (str): chooses what to calibrate, 'phase' or 'both'
                      'phase' is default, most useful (instrumental offsets),
                      'both' will align RCP/LCP amplitudes as well
        minimizer_method (str): Method for scipy.optimize.minimize (e.g., 'CG', 'BFGS')
        pad_amp (float): adds fractional uncertainty to amplitude sigmas in quadrature
        gain_tol (float): prior on gains not to be too crazy
        solution_interval (float): solution interval in seconds; one gain is derived per interval.
                                   If 0.0, a solution is determined for each unique time.
        scan_solutions (bool): If True, determine one gain per site per scan
                               Supersedes solution_interval.
        caltable (bool): if True, returns a Caltable instead of an Obsdata
        processes (int): number of cores to use in multiprocessing
        show_solution (bool): if True, display the solution as it is calculated
        msgtype (str): type of progress message to be printed, default is 'bar'

       Returns:
           (Obsdata): the calibrated observation, if caltable==False
           (Caltable): the derived calibration table, if caltable==True
    """
    # circular representation is needed
    if obs.polrep != 'circ':
        obs = obs.switch_polrep('circ')

    if len(sites) == 0:
        print("No stations specified in polgain cal!")
        print('Defaulting to calibrating all stations with reference station as: ' + reference)
        sites = np.array([x for x in obs.tarr['site'] if x != reference], dtype='<U32')

    # get scans
    scans = obs.tlist(t_gather=solution_interval, scan_gather=scan_solutions)

    # if there is only one scan, we need to set the dtype manually
    if scans.ndim == 2 and scans.shape[0] == 1:
        # get the dtype of the first and only scan
        scans_dtype = obs.data.dtype

        # get the only element and create a recarray of the correct dtype
        tuples = scans[0]
        rec = np.array(list(tuples), dtype=scans_dtype)

        # pack this into a length-1 object array
        scans = np.empty((1,), dtype=object)
        scans[0] = rec

    scans_cal = copy.copy(scans)

    # Make the pool for parallel processing
    if processes > 0:
        counter = parloop.Counter(initval=0, maxval=len(scans))
        if processes > len(scans):
            processes = len(scans)
        print("Using Multiprocessing with %d Processes" % processes)
        pool = Pool(processes=processes, initializer=init, initargs=(counter,))
    elif processes == 0:
        counter = parloop.Counter(initval=0, maxval=len(scans))
        processes = int(cpu_count())
        if processes > len(scans):
            processes = len(scans)
        print("Using Multiprocessing with %d Processes" % processes)
        pool = Pool(processes=processes, initializer=init, initargs=(counter,))
    else:
        print("Not Using Multiprocessing")

    # loop over scans and calibrate
    tstart = time.time()
    if processes > 0:  # with multiprocessing
        scans_cal = pool.map(get_polgains_scan_cal,
                             [[i, len(scans), scans[i], reference,
                               sites, method, pad_amp, caltable, show_solution, msgtype]
                              for i in range(len(scans))
                              ])

    else:  # without multiprocessing
        for i in range(len(scans)):
            obsh.prog_msg(i, len(scans), msgtype=msgtype, nscan_last=i - 1)
            scans_cal[i] = polgains_cal_scan(scans[i], reference, sites,
                                             method=method, minimizer_method=minimizer_method,
                                             show_solution=show_solution, caltable=caltable,
                                             pad_amp=pad_amp)

    tstop = time.time()
    print("\npolgain_cal time: %f s" % (tstop - tstart))

    if caltable:  # create and return  a caltable
        allsites = obs.tarr['site']
        caldict = {k: v.reshape(1) for k, v in scans_cal[0].items()}
        for i in range(1, len(scans_cal)):
            row = scans_cal[i]
            if len(row) == 0:
                continue

            for site in allsites:
                try:
                    dat = row[site]
                except KeyError:
                    continue

                try:
                    caldict[site] = np.append(caldict[site], row[site])
                except KeyError:
                    caldict[site] = [dat]

        caltable = ehtim.caltable.Caltable(obs.ra, obs.dec, obs.rf, obs.bw, caldict, obs.tarr,
                                           source=obs.source, mjd=obs.mjd, timetype=obs.timetype)
        out = caltable

    else:  # return the calibrated observation
        arglist, argdict = obs.obsdata_args()
        arglist[4] = np.concatenate(scans_cal)
        out = ehtim.obsdata.Obsdata(*arglist, **argdict)

    # close multiprocessing jobs
    if processes != -1:
        pool.close()

    return out


def polgains_cal_mixed(obs, reference=['AA'], reference_gains=[], sites=[], method='phase', minimizer_method='BFGS', pad_amp=0.,
                 gain_tol=.2, solution_interval=0.0, scan_solutions=False,
                 caltable=False, processes=-1, show_solution=False, msgtype='bar'):

    # TODO: function to globalize the polarimetric solution in time
    # given provided absolute calibration of the reference station
    # so that we have a meaningful EVPA
    """Polarimeteric-phase-gains-calibrate a dataset.
        Numerically solves for polarimetric gains to align RCP and LCP feeds.
        Uses all baselines to find the solution. Effectively assumes phase of Stokes V to be zero.
        Because fits are local, it's not providing absolute phase calibration.

       Args:
        obs (Obsdata): The observation to be calibrated
        reference (str): station used as reference to break the degeneracy
                         (LCP on baselines to the reference station remains unchanged)
        sites (list): list of sites to include in the polarimetric calibration.
                      Empty list calibrates all sites
        method (str): chooses what to calibrate, 'phase' or 'both'
                      'phase' is default, most useful (instrumental offsets),
                      'both' will align RCP/LCP amplitudes as well
        minimizer_method (str): Method for scipy.optimize.minimize (e.g., 'CG', 'BFGS')
        pad_amp (float): adds fractional uncertainty to amplitude sigmas in quadrature
        gain_tol (float): prior on gains not to be too crazy
        solution_interval (float): solution interval in seconds; one gain is derived per interval.
                                   If 0.0, a solution is determined for each unique time.
        scan_solutions (bool): If True, determine one gain per site per scan
                               Supersedes solution_interval.
        caltable (bool): if True, returns a Caltable instead of an Obsdata
        processes (int): number of cores to use in multiprocessing
        show_solution (bool): if True, display the solution as it is calculated
        msgtype (str): type of progress message to be printed, default is 'bar'

       Returns:
           (Obsdata): the calibrated observation, if caltable==False
           (Caltable): the derived calibration table, if caltable==True
    """
    ALMA_LRgain=0.0-1.0*1j
    # circular representation is needed
    if obs.polrep != 'circ':
        obs = obs.switch_polrep('circ')

    if len(sites) == 0:
        print("No stations specified in polgain cal!")
        print('Defaulting to calibrating all stations with reference station as: ' + str(reference))
        sites = np.array([x for x in obs.tarr['site'] if x not in reference], dtype='<U32')

    # get scans
    scans = obs.tlist(t_gather=solution_interval, scan_gather=scan_solutions)
    scans_cal = copy.copy(scans)

    # Make the pool for parallel processing
    if processes > 0:
        counter = parloop.Counter(initval=0, maxval=len(scans))
        if processes > len(scans):
            processes = len(scans)
        print("Using Multiprocessing with %d Processes" % processes)
        pool = Pool(processes=processes, initializer=init, initargs=(counter,))
    elif processes == 0:
        counter = parloop.Counter(initval=0, maxval=len(scans))
        processes = int(cpu_count())
        if processes > len(scans):
            processes = len(scans)
        print("Using Multiprocessing with %d Processes" % processes)
        pool = Pool(processes=processes, initializer=init, initargs=(counter,))
    else:
        print("Not Using Multiprocessing")

    # loop over scans and calibrate
    tstart = time.time()
    if processes > 0:  # with multiprocessing
        scans_cal = pool.map(get_polgains_scan_cal_mixed,
                             [[i, len(scans), scans[i], reference,reference_gains,
                               sites, method, pad_amp, caltable, show_solution, msgtype]
                              for i in range(len(scans))
                              ])

    else:  # without multiprocessing
        for i in range(len(scans)):
            obsh.prog_msg(i, len(scans), msgtype=msgtype, nscan_last=i - 1)
            scans_cal[i] = polgains_cal_scan_mixed(scans[i], reference,reference_gains, sites,
                                             method=method, minimizer_method=minimizer_method,
                                             show_solution=show_solution, caltable=caltable,
                                             pad_amp=pad_amp)

    tstop = time.time()
    print("\npolgain_cal time: %f s" % (tstop - tstart))

    if caltable:  # create and return  a caltable
        allsites = obs.tarr['site']
        caldict = {k: v.reshape(1) for k, v in scans_cal[0].items()}
        for i in range(1, len(scans_cal)):
            row = scans_cal[i]
            if len(row) == 0:
                continue

            for site in allsites:
                try:
                    dat = row[site]
                except KeyError:
                    continue

                try:
                    caldict[site] = np.append(caldict[site], row[site])
                except KeyError:
                    caldict[site] = [dat]

        caltable = ehtim.caltable.Caltable(obs.ra, obs.dec, obs.rf, obs.bw, caldict, obs.tarr,
                                           source=obs.source, mjd=obs.mjd, timetype=obs.timetype)
        out = caltable

    else:  # return the calibrated observation
        arglist, argdict = obs.obsdata_args()
        arglist[4] = np.concatenate(scans_cal)
        out = ehtim.obsdata.Obsdata(*arglist, **argdict)

    # close multiprocessing jobs
    if processes != -1:
        pool.close()

    return out


def polgains_cal_scan(scan, reference='AA', sites=[], method='phase', minimizer_method='BFGS',
                      show_solution=False, pad_amp=0., gain_tol=.2, caltable=False, msgtype='bar'):
    """Polarimeteric-phase-gains-calibrate a dataset.
        Numerically solves for polarimetric gains to align RCP and LCP feeds.
        Uses all baselines to find the solution. Effectively assumes phase of Stokes V to be zero.
        Because fits are local, it's not providing absolute phase calibration.

       Args:
        obs (Obsdata): The observation to be calibrated
        reference (str): station used as reference to break the degeneracy
                         (LCP on baselines to the reference station remains unchanged)
        sites (list): list of sites to include in the polarimetric calibration.
                      Empty list calibrates all sites
        method (str): chooses what to calibrate, 'phase' or 'both'
                      'phase' is default, most useful (instrumental offsets),
                      'both' will align RCP/LCP amplitudes as well
        minimizer_method (str): Method for scipy.optimize.minimize (e.g., 'CG', 'BFGS')
        pad_amp (float): adds fractional uncertainty to amplitude sigmas in quadrature
        gain_tol (float): prior on gains not to be too crazy
        caltable (bool): if True, returns a Caltable instead of an Obsdata
        show_solution (bool): if True, display the solution as it is calculated
        msgtype (str): type of progress message to be printed, default is 'bar'

       Returns:
           (Obsdata): the calibrated observation, if caltable==False
           (Caltable): the derived calibration table, if caltable==True
    """
    indices_no_vis_nans = (~np.isnan(scan[ehc.vis_poldict['RR']])) & (
        ~np.isnan(scan[ehc.vis_poldict['LL']]))
    scan_no_vis_nans = scan[indices_no_vis_nans]
    allsites_no_vis_nans = list(set(np.hstack((scan_no_vis_nans['t1'], scan_no_vis_nans['t2']))))

    #print('Calibrating '+method)
    # all the sites in the scan
    allsites = list(set(np.hstack((scan['t1'], scan['t2']))))
    if len(sites) == 0:
        print("No stations specified in polgain cal")
        print("defaulting to calibrating all non-reference stations!")
        sites = allsites

    # only include sites that are present, not singlepol and not reference
    sites = [s for s in sites if (s in allsites_no_vis_nans) & (s != reference)]

    # create a dictionary to keep track of gains; reference site and singlepol sites get key -1
    tkey = {b: a for a, b in enumerate(sites)}
    # make two lists of gain keys that relates scan bl gains to solved site ones
    # -1 means that this station does not have a gain that is being solved for
    g1_keys = []
    g2_keys = []
    for row in scan:
        try:
            g1_keys.append(tkey[row['t1']])
        except KeyError:
            g1_keys.append(-1)
        try:
            g2_keys.append(tkey[row['t2']])
        except KeyError:
            g2_keys.append(-1)

    # get scan visibilities of the specified polarization
    visRR = scan_no_vis_nans[ehc.vis_poldict['RR']]
    visLL = scan_no_vis_nans[ehc.vis_poldict['LL']]
    sigmaRR = scan_no_vis_nans[ehc.sig_poldict['RR']]
    sigmaLL = scan_no_vis_nans[ehc.sig_poldict['LL']]
    sigma0 = np.sqrt(sigmaRR**2 + sigmaLL**2)
    sigma = np.sqrt(sigma0**2 + (pad_amp*0.5*(np.abs(visRR)+np.abs(visLL)))**2)
    # sigma_inv = 1.0/np.sqrt(sigma**2 + (pad_amp*0.5*(np.abs(visRR)+np.abs(visLL)))**2)
    # initial guesses for parameters
    
    n_gains = len(sites)
    gpar_guess = np.ones(n_gains, dtype=np.complex128).view(dtype=np.float64)

    # error function
    def errfunc(g):
        g = g.view(dtype=np.complex128)
        g = np.append(g, 1. + 0.j)
        if method == "phase":
            g = g / np.abs(g)
        if method == "amp":
            g = np.abs(g)
        g1 = g[g1_keys]
        g2 = g[g2_keys]
        
        if method=='amp':
            #chisq = np.sum(np.abs(( np.abs(visRR) - np.abs(g1[indices_no_vis_nans]) *
            #                   np.abs(g2[indices_no_vis_nans]) * np.abs(visLL)) / sigma)**2)
            
            chisq = np.sum(np.abs(np.log(np.abs(visRR)/( np.abs(g1[indices_no_vis_nans])*
                                                 np.abs(g2[indices_no_vis_nans])*np.abs(visLL) )))/sigma**2)
        else:
            chisq = np.sum(np.abs((visRR - g1[indices_no_vis_nans] *
                               g2[indices_no_vis_nans].conj() * visLL) / sigma)**2)
        
        #chsiq_g=0
        #if (method == 'both')|(method == 'amp'): 
        chisq_g = np.sum((np.log(np.abs(g))**2 / gain_tol**2))
        if method =='phase':
            chisq_g=0.0
            
        return chisq + chisq_g
    
    if np.max(g1_keys) > -1 or np.max(g2_keys) > -1:
        # run the minimizer to get a solution (but only run if there's at least one gain to fit)
        optdict = {'maxiter': MAXIT}  # minimizer params
        res = opt.minimize(errfunc, gpar_guess, method=minimizer_method, options=optdict)

        # get solution
        g_fit = res.x.view(np.complex128)
        if method == "phase":
            g_fit = g_fit / np.abs(g_fit)
        if method == "amp":
            g_fit = np.abs(g_fit) +0.j

        if show_solution:
            print("Padding errors with fractional error of ",pad_amp)
            print(sites)
            print(g_fit)
    else:
        g_fit = []
    g_fit = np.append(g_fit, 1.)
    # Derive a calibration table or apply the solution to the scan
    if caltable:
        allsites = list(set(scan['t1']).union(set(scan['t2'])))

        caldict = {}
        for site in allsites:
            if site in sites:
                site_key = tkey[site]
            else:
                site_key = -1

            # Convention is that we calibrate RCP phase to align with the LCP phase
            rscale = g_fit[site_key]**(-1)
            lscale = 1. + 0.j

            caldict[site] = np.array((scan['time'][0], rscale, lscale), dtype=ehc.DTCAL)
        out = caldict
    elif method=='amp':
        g1_fit = g_fit[g1_keys]
        g2_fit = g_fit[g2_keys]
        g1_inv = g1_fit**(-1)
        g2_inv = g2_fit**(-1)
        # scale visibilities
        scan['rrvis'] *= np.sqrt(g1_inv * g2_inv.conj())
        scan['llvis'] *= 1./np.sqrt(g1_inv * g2_inv.conj())
        scan['rlvis'] *= np.sqrt(g1_inv/g2_inv.conj())
        scan['lrvis'] *= np.sqrt(g2_inv.conj()/g1_inv)
        # don't scale sigmas
        out = scan
        
    else:
        g1_fit = g_fit[g1_keys]
        g2_fit = g_fit[g2_keys]
        g1_inv = g1_fit**(-1)
        g2_inv = g2_fit**(-1)
        # scale visibilities
        scan['rrvis'] *= g1_inv * g2_inv.conj()
        scan['llvis'] *= 1. + 0.j
        scan['rlvis'] *= g1_inv
        scan['lrvis'] *= g2_inv.conj()
        # don't scale sigmas
        out = scan
    return out


def polgains_cal_scan_mixed(scan, reference=['AA'],reference_gains=[], sites=[], method='phase', minimizer_method='BFGS',
                      show_solution=False, pad_amp=0., gain_tol=.2, caltable=False, msgtype='bar'):
    """Polarimeteric-phase-gains-calibrate a dataset.
        Version performing a mixed global-local calibration.
        Numerically solves for polarimetric gains to align RCP and LCP feeds.
        Uses all baselines to find the solution. Effectively assumes phase of Stokes V to be zero.
        Because fits are local, it's not providing absolute phase calibration.

       Args:
        obs (Obsdata): The observation to be calibrated
        reference (str): station used as reference to break the degeneracy
                         (LCP on baselines to the reference station remains unchanged)
        sites (list): list of sites to include in the polarimetric calibration.
                      Empty list calibrates all sites
        method (str): chooses what to calibrate, 'phase' or 'both'
                      'phase' is default, most useful (instrumental offsets),
                      'both' will align RCP/LCP amplitudes as well
        minimizer_method (str): Method for scipy.optimize.minimize (e.g., 'CG', 'BFGS')
        pad_amp (float): adds fractional uncertainty to amplitude sigmas in quadrature
        gain_tol (float): prior on gains not to be too crazy
        caltable (bool): if True, returns a Caltable instead of an Obsdata
        show_solution (bool): if True, display the solution as it is calculated
        msgtype (str): type of progress message to be printed, default is 'bar'

       Returns:
           (Obsdata): the calibrated observation, if caltable==False
           (Caltable): the derived calibration table, if caltable==True
    """
    
    
    #============================================
    #===Precalibration of reference stations here
    if (method=='phase')&(len(reference_gains)>0):
        
        tkey_ref = {b: a for a, b in enumerate(reference)}
        # make two lists of gain keys that relates scan bl gains to solved site ones
        # -1 means that this station does not have a gain that is being solved for
        g1_keys_ref = []
        g2_keys_ref = []
        for row in scan:
            try:
                g1_keys_ref.append(tkey_ref[row['t1']])
            except KeyError:
                g1_keys_ref.append(-1)
            try:
                g2_keys_ref.append(tkey_ref[row['t2']])
            except KeyError:
                g2_keys_ref.append(-1)
                       
        g_precal = np.array(reference_gains+[1.+0.*1j])
        g1_pre = g_precal[g1_keys_ref]
        g2_pre = g_precal[g2_keys_ref]
        g1_inv = g1_pre**(-1)
        g2_inv = g2_pre**(-1)
        # scale visibilities
        scan['rrvis'] *= g1_inv * g2_inv.conj()
        scan['llvis'] *= 1. + 0.j
        scan['rlvis'] *= g1_inv
        scan['lrvis'] *= g2_inv.conj()
     #============================================
     #============================================
    
    
    indices_no_vis_nans = (~np.isnan(scan[ehc.vis_poldict['RR']])) & (
        ~np.isnan(scan[ehc.vis_poldict['LL']]))
    scan_no_vis_nans = scan[indices_no_vis_nans]
    allsites_no_vis_nans = list(set(np.hstack((scan_no_vis_nans['t1'], scan_no_vis_nans['t2']))))

    #print('Calibrating '+method)
    # all the sites in the scan
    allsites = list(set(np.hstack((scan['t1'], scan['t2']))))
    if len(sites) == 0:
        print("No stations specified in polgain cal")
        print("defaulting to calibrating all non-reference stations!")
        sites = allsites

    # only include sites that are present, not singlepol and not reference
    sites = [s for s in sites if (s in allsites_no_vis_nans) & (s not in reference)]

    # create a dictionary to keep track of gains; reference site and singlepol sites get key -1
    tkey = {b: a for a, b in enumerate(sites)}
    # make two lists of gain keys that relates scan bl gains to solved site ones
    # -1 means that this station does not have a gain that is being solved for
    g1_keys = []
    g2_keys = []
    for row in scan:
        try:
            g1_keys.append(tkey[row['t1']])
        except KeyError:
            g1_keys.append(-1)
        try:
            g2_keys.append(tkey[row['t2']])
        except KeyError:
            g2_keys.append(-1)
    
    #print(' tkey: ',tkey)
    #print(" g1_keys: ", g1_keys)
    #print(" g2_keys: ", g2_keys)

    # get scan visibilities of the specified polarization
    visRR = scan_no_vis_nans[ehc.vis_poldict['RR']]
    visLL = scan_no_vis_nans[ehc.vis_poldict['LL']]
    sigmaRR = scan_no_vis_nans[ehc.sig_poldict['RR']]
    sigmaLL = scan_no_vis_nans[ehc.sig_poldict['LL']]
    sigma0 = np.sqrt(sigmaRR**2 + sigmaLL**2)
    sigma = np.sqrt(sigma0**2 + (pad_amp*0.5*(np.abs(visRR)+np.abs(visLL)))**2)
    # sigma_inv = 1.0/np.sqrt(sigma**2 + (pad_amp*0.5*(np.abs(visRR)+np.abs(visLL)))**2)
    # initial guesses for parameters
    
    n_gains = len(sites)
    gpar_guess = np.ones(n_gains, dtype=np.complex128).view(dtype=np.float64)

    # error function
    def errfunc(g):
        g = g.view(dtype=np.complex128)
        g = np.append(g, 1. + 0.j)
        if method == "phase":
            g = g / np.abs(g)
        if method == "amp":
            g = np.abs(g)
        g1 = g[g1_keys]
        g2 = g[g2_keys]
        
        if method=='amp':
            #chisq = np.sum(np.abs(( np.abs(visRR) - np.abs(g1[indices_no_vis_nans]) *
            #                   np.abs(g2[indices_no_vis_nans]) * np.abs(visLL)) / sigma)**2)
            
            chisq = np.sum(np.abs(np.log(np.abs(visRR)/( np.abs(g1[indices_no_vis_nans])*
                                                 np.abs(g2[indices_no_vis_nans])*np.abs(visLL) )))/sigma**2)
        else:
            chisq = np.sum(np.abs((visRR - g1[indices_no_vis_nans] *
                               g2[indices_no_vis_nans].conj() * visLL) / sigma)**2)
        
        #chsiq_g=0
        #if (method == 'both')|(method == 'amp'): 
        chisq_g = np.sum((np.log(np.abs(g))**2 / gain_tol**2))
        if method =='phase':
            chisq_g=0.0
            
        return chisq + chisq_g


    if np.max(g1_keys) > -1 or np.max(g2_keys) > -1:
        # run the minimizer to get a solution (but only run if there's at least one gain to fit)
        optdict = {'maxiter': MAXIT}  # minimizer params
        res = opt.minimize(errfunc, gpar_guess, method=minimizer_method, options=optdict)

        # get solution
        g_fit = res.x.view(np.complex128)
        if method == "phase":
            g_fit = g_fit / np.abs(g_fit)
        if method == "amp":
            g_fit = np.abs(g_fit) +0.j

        if show_solution:
            print("Padding errors with fractional error of ",pad_amp)
            print(sites)
            print(g_fit)
    else:
        g_fit = []
    g_fit = np.append(g_fit, 1.)
    #print(" n_gains: ", n_gains)
    #print(" g_fit: ", g_fit)
    #print(" g1_keys: ", type(g1_keys))

    #print(" wtf: ", g_fit[g1_keys])
    #g_fit is an array of resulting gains, g1_keys is a list of which site is t1 
    
    # Derive a calibration table or apply the solution to the scan
    if caltable:
        allsites = list(set(scan['t1']).union(set(scan['t2'])))

        caldict = {}
        for site in allsites:
            if site in sites:
                site_key = tkey[site]
            else:
                site_key = -1

            # Convention is that we calibrate RCP phase to align with the LCP phase
            rscale = g_fit[site_key]**(-1)
            lscale = 1. + 0.j

            caldict[site] = np.array((scan['time'][0], rscale, lscale), dtype=ehc.DTCAL)
        out = caldict
    elif method=='amp':
        g1_fit = g_fit[g1_keys]
        g2_fit = g_fit[g2_keys]
        g1_inv = g1_fit**(-1)
        g2_inv = g2_fit**(-1)
        # scale visibilities
        scan['rrvis'] *= np.sqrt(g1_inv * g2_inv.conj())
        scan['llvis'] *= 1./np.sqrt(g1_inv * g2_inv.conj())
        scan['rlvis'] *= np.sqrt(g1_inv/g2_inv.conj())
        scan['lrvis'] *= np.sqrt(g2_inv.conj()/g1_inv)
        # don't scale sigmas
        out = scan
        
    else:
        g1_fit = g_fit[g1_keys]
        g2_fit = g_fit[g2_keys]
        g1_inv = g1_fit**(-1)
        g2_inv = g2_fit**(-1)
        # scale visibilities
        scan['rrvis'] *= g1_inv * g2_inv.conj()
        scan['llvis'] *= 1. + 0.j
        scan['rlvis'] *= g1_inv
        scan['lrvis'] *= g2_inv.conj()
        # don't scale sigmas
        out = scan
    return out



def init(x):
    global counter
    counter = x


def get_polgains_scan_cal(args):
    return get_polgains_scan_cal2(*args)


def get_polgains_scan_cal2(i, n, scan, reference, sites, method, pad_amp,gain_tol, caltable,
                           show_solution, msgtype):

    if n > 1:
        global counter
        counter.increment()
        obsh.prog_msg(counter.value(), counter.maxval, msgtype, counter.value() - 1)

    return polgains_cal_scan(scan, reference, sites,
                             method=method, caltable=caltable, show_solution=show_solution,
                             pad_amp=pad_amp,gain_tol=gain_tol, msgtype=msgtype)

def get_polgains_scan_cal_mixed(args):
    return get_polgains_scan_cal2_mixed(*args)


def get_polgains_scan_cal2_mixed(i, n, scan, reference, reference_gains, sites, method, pad_amp,gain_tol, caltable,
                           show_solution, msgtype):

    if n > 1:
        global counter
        counter.increment()
        obsh.prog_msg(counter.value(), counter.maxval, msgtype, counter.value() - 1)

    return polgains_cal_scan(scan, reference, reference_gains, sites,
                             method=method, caltable=caltable, show_solution=show_solution,
                             pad_amp=pad_amp,gain_tol=gain_tol, msgtype=msgtype)


