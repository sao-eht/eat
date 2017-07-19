import pandas as pd
from ..factor import *

def test_factor():
    """
    Test if factor() successfully factors out site-based delay and rate
    """
    bb = [('a','b',0), ('b','c',0), ('c','d',0), ('d','e',0), ('e','a',0),
          ('a','c',0), ('b','d',0), ('c','e',0), ('d','a',0), ('e','b',0)]

    x  = np.array(bb, dtype=[('ref', 'U3'),('rem', 'U2'), ('val', 'f16')])
    sb = factor(x)
    assert sb is not None

    y  = pd.DataFrame(x)
    sb = factor(y)
    assert sb is not None
