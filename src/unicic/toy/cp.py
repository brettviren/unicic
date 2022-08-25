#!/usr/bin/env python3
'''
The "toy" problem for testing, implemented for cupy
'''

import cupy
xp = cupy

from . import np

def predict_scalar(q, ms, xp=xp):
    return np.predict_scalar(q, ms, xp)

def predict_batched(q, ms, xp=xp):
    return np.predict_batched(q, ms, xp)

