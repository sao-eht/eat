from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()

import math
import numpy as np
from scipy.optimize import least_squares

def factor(bb, initial_guess=None,
           regularizer='Tikhonov', weight=1.0):
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

        err[ref, rem] = obs[ref, rem] - (sol[ref] - sol[rem])

    so that the minimization is performed over

        chi^2 = sum_baselines err[ref, rem]^2 / sigma[ref, rem]^2

    However, it is clear that sol[] is not uniquely determined because
    err[ref, rem] is invariant to a global constant offset to sol[].
    The simplest fix is to add the regularizer

        w sum_feeds sol^2

    This is equivalent to using the Tikhonov regularizer with Tikhonov
    matrix w I.

    Args:
        bb:    A numpy structured array or pandas dataframe of
               baseline-based input data

    Returns:
        A dictionary of zero-mean site-based data being factored out

    """
    feeds = set(bb['ref']) | set(bb['rem'])
    map   = {f: i for i, f in enumerate(feeds)}

    ref = np.array([map[f] for f in bb['ref']])
    rem = np.array([map[f] for f in bb['rem']])
    obs = np.array(                 bb['val'] )
    def err(sol): # closure (as in functional languages) on ref, rem, and obs
        if regularizer is None:
            return obs - (sol[ref] - sol[rem])
        else:
            reg = sol if regularizer == 'Tikhonov' else np.mean(sol)
            return np.append(obs - (sol[ref] - sol[rem]),
                             math.sqrt(weight) * reg)

    if initial_guess is None:
        initial_guess = np.zeros(len(feeds))
    elif len(initial_guess) != len(feeds):
        raise IndexError("Lengths of initial_guess ({}) and feeds ({}) "
                         "do not match".format(len(initial_guess),
                                               len(feeds)))

    sol = least_squares(err, initial_guess)
    if sol.success:
        if regularizer == 'Tikhonov':
            n = 1.0 + weight/len(feeds)
        else:
            n = 1.0
        v  = sol.x * n
        v -= np.mean(v) # FIXME: is the regularizer still necessary?
        return {f: v[i] for i, f in enumerate(feeds)}
    else:
        return None
