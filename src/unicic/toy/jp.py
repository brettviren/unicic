#!/usr/bin/env python3
'''
The "toy" problem for testing, implemented for jax
'''

import jax
import jax.numpy as xp

from . import np as _np

class Random:
    def __init__(self, seed):
        self._key = jax.random.PRNGKey(seed)

    def __getattr__(self, name):
        meth = getattr(jax.random, name)
        def call_meth(*a, **k):
            self._key, sub = jax.random.split(self._key)
            return meth(sub, *a, **k)
        return call_meth

def predict_scalar(q, ms, xp=xp):
    return _np.predict_scalar(q, ms, xp)

predict_batched = jax.vmap(predict_scalar, in_axes=(0,None))

def statvar_cnp_scalar(Npred, N, xp=xp):
    return _np.statvar_cnp_scalar(Npred, N, xp=xp)

statvar_cnp_batched = jax.vmap(statvar_cnp_scalar, in_axes=(0,None))

def fluctuate(Npred, rng, xp=xp):
    return rng.poisson(Npred)
