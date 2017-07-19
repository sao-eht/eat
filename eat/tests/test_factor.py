import pandas as pd
import numpy  as np
from ..factor import *

def test_factor():
    """
    Test if factor() successfully factors out site-based delay and rate
    """

    # Randomly generate a dictionary of zero-mean site-based rates/delays
    sites = "abcde"
    r  = 2 * np.random.rand(len(sites)) - 1
    r -= np.mean(r)
    sb = {s:r[i] for i, s in enumerate(sites)}

    # Generate baseline-based rates/delays using `sb`
    bb = np.array([(ref, rem, sb[ref] - sb[rem]) for ref in sites
                                                 for rem in sites if rem > ref],
                  dtype=[('ref', 'U3'), ('rem', 'U2'), ('val', 'f16')])

    # Use `eat.factor()` to factor out site-based delays/rates from
    # baseline-based delays/ratse
    sol = factor(bb, regularizer=None, weight=10.0)
    assert sol is not None

    # The solution is in general different from the original
    # rates/delays by a constant
    d = np.array([sol[s] - sb[s] for s in sites])
    assert np.max(d) - np.min(d) < 1e-9
