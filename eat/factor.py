import numpy as np
from scipy.optimize import least_squares

def factor(bb, weight=1.0):
    """
    Factor out site-based delay and rate from baseline-based slopes

    The linear drift (slopes) of phase with frequency and time are
    usually interpreted as delays and rates, and are removed from the
    VLBI data in fringe fitting.  Let n be the number of sites.  There
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

        w^2 sum_sites sol^2

    This is equivalent to using the Tikhonov regularizer with Tikhonov
    matrix w I.

    Args:
        bb:    Baseline-based input data

    Returns:
        Site-based data being factored out

    """
    sites = sorted(set(bb['ref']) | set(bb['rem']))
    sol   = np.zeros(len(sites))
    map   = {s: i for i, s in enumerate(sites)}

    ref = np.array([map[s] for s in bb['ref']])
    rem = np.array([map[s] for s in bb['rem']])
    obs = np.array(                 bb['val'] )
    def err(sol): # closure (as in functional languages) on ref, rem, and obs
        return np.append(obs - (sol[ref] - sol[rem]), weight * sol)

    sol = least_squares(err, sol)

    if sol.success:
        return sol.x
    else:
        return None
