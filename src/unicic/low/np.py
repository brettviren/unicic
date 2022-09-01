#!/usr/bin/env python3
'''
The unicic low level API for Numpy.

'''

import multiprocessing
from functools import reduce

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
        return numpy.linalg.inv(a)
    except numpy.linalg.LinAlgError as lae:
        raise ValueError("singular matrix") from lae
    except ValueError as err:
        print('You probably need to fix your covariance matrix.')
        raise

def gridspace(start, stop, num, endpoint=True, xp=xp):
    '''Return points on a grid.

    start and stop must have same shape and may be scalar or 1D array.

    num must be scalar or shaped same as start and stop.

    If start is None, zeros shaped as stop are assumed.

    If scalar, then shape (num,) returned.

    If array, then shape (product, start.size) returned where product
    is the product of elements of num.

    '''
    stop = xp.array(stop)
    if start is None:
        start = xp.zeros_like(stop)
    else:
        start = xp.array(start)
    num = xp.array(num)
    ndims = len(start.shape)
    if ndims == 0:              # full scalar
        return xp.linspace(start, stop, int(num), endpoint=endpoint)

    if len(num.shape) == 0 and ndims > 0:
        num = xp.array([num]*start.size)

    axes = [xp.linspace(a, o, int(n), endpoint=endpoint) for a,o,n in zip(start,stop,num)]
    mg = xp.meshgrid(*axes)
    return xp.vstack([g.reshape(-1) for g in mg]).T


def map_reduce(mfunc, rfunc, iterable, *, initial=None, nproc=1):
    '''Map mfunc on each in iterable and reduce that with rfunc.

    If initial is given it is provided first to rfunc.

    If nproc > 1 then multiprocessing is used for the map.

    Note, if mfunc is a or call vectorized functions nproc>1 may lead
    to processor overload.

    '''
    if nproc > 1:
        with multiprocessing.Pool(processes=nproc) as pool:
            res = pool.map(mfunc, iterable)
    else:
        res = map(mfunc, iterable)
        
    if initial is None:
        return reduce(rfunc, res)
    return reduce(rfunc, res, initial)
