import numpy as np

def factor(bb):
    """
    Factor out site-based delay and rate from baseline-based delay and rate

    Args:
        bb:    Baseline-based input data

    Returns:
        Site-based data being factored out
    """
    sites = np.unique(np.append(bb['ref'], bb['rem']))
    types = [('site', sites.dtype), ('value', 'f8')]
    sb    = np.array([(s, 0) for s in sites], dtype=types)

    return sb
