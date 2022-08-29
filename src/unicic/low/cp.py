#!/usr/bin/env python3
'''
The low level unicic api for cupy
'''

import cupy
xp = cupy

Random = cupy.random.default_rng
def uniform(rng, low=0.0, high=1.0, size=None):
    r = rng.random(size)
    return r * (high - low) + low

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

