#!/usr/bin/env python3
'''
The unicic low level API for Numpy.

'''

import numpy
xp = numpy

Random = numpy.random.default_rng

def uniform(rng, low=0.0, high=1.0, size=None):
    return rng.uniform(low, high, size)


def fluctuate(Npred, rng, xp=xp):
    '''
    Return fluctuated measure of expectation
    '''
    return rng.poisson(Npred)

def inv(a, xp=xp):
    '''
    Return the multiplicative inverse of a.

    Raises ValueError if singular
    '''
    try:
        return xp.linalg.inv(a)
    except LinAlgError as lae:
        raise ValueError("singular matrix") from lae


