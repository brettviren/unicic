#!/usr/bin/env python3
'''
The "toy" problem for testing, implemented for cupy
'''

import cupy
xp = cupy

Random = cupy.random.default_rng

from . import np

def predict_scalar(q, ms, xp=xp):
    return np.predict_scalar(q, ms, xp)

def predict_batched(q, ms, xp=xp):
    return np.predict_batched(q, ms, xp)

def statvar_cnp_scalar(q, ms, xp=xp):
    return np.statvar_cnp_scalar(q, ms, xp)

def statvar_cnp_batched(q, ms, xp=xp):
    return np.statvar_cnp_batched(q, ms, xp)


def fluctuate(Npred, rng, xp=xp):
    return rng.poisson(Npred)


