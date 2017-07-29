from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()

import math
import numpy as np
from scipy.optimize import least_squares

def factor(bb, initial_guess=None, weight=1.0):
    """
    Factor out site-based delay and rate from baseline-based slopes

    The linear drift (slopes) of phase with frequency and time are
    usually interpreted as delays and rates, and are removed from the
    VLBI data in fringe fitting.  Let n be the number of feeds.  There
    are n (n-1) such slopes in total.  Using all of them, such as
    currently done in HOPS, breaks the closure relationships in
    general.

    Global fringe solution takes into account the assumption that
    delays and rates are site-based.  The main advantage is that it
    preserves the closure relationships.  Even better, "one person's
    noise is another person's signal", the remaining slopes after
    global fringe fit actually contain information about the source
    images.  Being able to factor out these site-based "noise" from
    the baseline-based "data", therefore, is essential for developing
    new high-order image reconstruction methods.

    We implement here a very general approach to factor out site-based
    information from baseline-based information.  Let obs[] be the
    observational data and sol[] be the solution array, the simplest
    error function is

        chi[ref, rem] = obs[ref, rem] - (sol[ref] - sol[rem])

    so that the minimization is performed over

        chi^2 = sum_baselines chi[ref, rem]^2 / sigma[ref, rem]^2

    However, it is clear that sol[] is not uniquely determined because
    chi[ref, rem] is invariant to a global constant offset to sol[].
    The simplest fix is to add the regularizer

        w mean(sol)^2

    Args:
        bb:               A numpy structured array or pandas dataframe of
                          baseline-based input data
        initial_guess:    Initial conditions of the minimizer
        weight:           Weight of the regularizer

    Returns:
        A dictionary of site-based data being factored out

    """
    feeds = set(bb['ref']) | set(bb['rem'])
    fmap  = {f: i for i, f in enumerate(feeds)}

    ref = np.array([fmap[f] for f in bb['ref']])
    rem = np.array([fmap[f] for f in bb['rem']])
    obs = np.array(                  bb['val'] )
    try:
        err = np.array(bb['err'])
    except:
        err = 1.0
    def regchi(sol):
        # closure (as in functional languages) on ref, rem, obs, and err
        return np.append((obs - (sol[ref] - sol[rem])) / err,
                         math.sqrt(weight) * np.mean(sol))

    if initial_guess is None:
        initial_guess = np.zeros(len(feeds))
    elif len(initial_guess) != len(feeds):
        raise IndexError("Lengths of initial_guess ({}) and feeds ({}) "
                         "do not match".format(len(initial_guess),
                                               len(feeds)))

    sol = least_squares(regchi, initial_guess)
    if sol.success:
        return {f: sol.x[i] for i, f in enumerate(feeds)}
    else:
        return None
