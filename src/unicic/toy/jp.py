#!/usr/bin/env python3
'''
The "toy" problem for testing, implemented for jax
'''

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


def predict(q, ms, xp=xp):
    return _np.predict(q, ms, xp)

def statvar_cnp(Npred, Ndata, xp=xp):
    return _np.statvar_cnp(Npred, Ndata, xp=xp)

def fluctuate(Npred, rng, xp=xp):
    return rng.poisson(Npred)

def inv(a, xp=xp):
    '''
    Return the multiplicative inverse of a.

    Raises ValueError if singular
    '''
    # fixme: how to error catch to convert to ValueError?
    return xp.linalg.inv(a)

chi2 = _np.chi2
