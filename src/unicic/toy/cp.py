#!/usr/bin/env python3
'''
The "toy" problem for testing, implemented for cupy
'''

import cupy
xp = cupy

Random = cupy.random.default_rng
def uniform(rng, low=0.0, high=1.0, size=None):
    r = rng.random(size)
    return r * (high - low) + low

from . import np as _np

# def predict_scalar(q, ms, xp=xp):
#     return _np.predict_scalar(q, ms, xp)

def predict(q, ms, xp=xp):
    return _np.predict(q, ms, xp)

def statvar_cnp(q, ms, xp=xp):
    return _np.statvar_cnp(q, ms, xp)

def fluctuate(Npred, rng, xp=xp):
    return rng.poisson(Npred)

def inv(a, xp=xp):
    '''
    Return the multiplicative inverse of a.

    Raises ValueError if singular
    '''
    # fixme: this error catching is not correct for cupy
    try:
        return xp.linalg.inv(a)
    except LinAlgError as lae:
        raise ValueError("singular matrix") from lae


chi2 = _np.chi2
