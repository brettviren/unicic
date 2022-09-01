#!/usr/bin/env python3
'''
The Jax implementation of the low level unicic API
'''

from functools import reduce

import jax
# https://github.com/google/jax#current-gotchas
jax.config.update('jax_enable_x64', True)

import jax.numpy as xp

from . import np as _np

class Random:
    def __init__(self, seed):
        self._key = jax.random.PRNGKey(seed)

    @property
    def key(self):
        self._key, sub = jax.random.split(self._key)
        return sub

    def __getattr__(self, name):
        meth = getattr(jax.random, name)
        def call_meth(*a, **k):
            return meth(self.key, *a, **k)
        return call_meth

def uniform(rng, low=0.0, high=1.0, size=None):
    shape=()
    if size:
        shape=(size,)
    return jax.random.uniform(rng.key, shape=shape, minval=low, maxval=high)

def fluctuate(Npred, rng, xp=xp):
    return rng.poisson(Npred)

def inv(a, xp=xp):
    '''
    Return the multiplicative inverse of a.

    Raises ValueError if singular
    '''
    # fixme: how to error catch to convert to ValueError?
    return xp.linalg.inv(a)

def gridspace(start, stop, num, endpoint=True, xp=xp):
    return _np.gridspace(start, stop, num, endpoint=endpoint, xp=xp)

def map_reduce(mfunc, rfunc, iterable, *, initial=None, nproc=1):
    '''Map mfunc on each in iterable and reduce that with rfunc.

    If initial is given it is provided first to rfunc.

    '''

    if nproc > 1:
        nproc = 1
    if nproc == 1:
        return _np.map_reduce(mfunc, rfunc, iterable, initial=initial, nproc=1)
    else:
        pmfunc = jax.pmap(mfunc)
        res = pmfunc(iterable)
        
    if initial is None:
        return reduce(rfunc, res)
    return reduce(rfunc, res, initial)
    
