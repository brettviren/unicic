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

def linspace(start, stop, num, endpoint=True, xp=xp):
    '''Return evenly spaced samples over interval.

    start and stop may, together, be scalar or array.

    num may be scalar or array.

    If num is array and has the same shape as start and stop, the
    linspace production is mapped over each element after all three
    arrays are flattened.  A tuple of the flattened map result is
    returned with each element a 1D array.

    If num is array but has differing shape as start and stop then the
    linspace is mapped only over the flattened num array.  A tuple of
    this map is returned with each element having one more dimension
    as start/stop.

    '''
    stop = xp.array(stop)
    if start is None:
        start = xp.zeros_like(stop)
    else:
        start = xp.array(start)

    num = xp.array(num)

    if len(num.shape) == 0: 
        # Scalar num, return is shaped (num,)
        return xp.linspace(start, stop, int(num), endpoint=endpoint)
    
    if num.shape == start.shape and num.shape == stop.shape:
        fls = [xp.linspace(a,o,int(n)) for a,o,n in zip(start.reshape(-1), stop.reshape(-1), num.reshape(-1))]
        return tuple(fls)
    
    fls = [xp.linspace(start, stop, int(n)) for n in num.reshape(-1)]
    return tuple(fls)


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
